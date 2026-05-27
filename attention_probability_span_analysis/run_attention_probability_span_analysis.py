#!/usr/bin/env python3
"""
Generate Probability-span attention tables from tokenwise-K processed H5 embeddings.

Inputs:
  - embeddings H5 produced by process_generations_more_embs_from_h5.py with:
      --attention_score_tokenwise_k_mode
      --collect_qkvo_embeddings

Outputs:
  - attention_table_probability_pos_<0..N-1>.png
  - attention_tables.npz
  - summary.json
  - config.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GUESS_PREFIX = "\n\nGuess:"
PROBABILITY_MARKER = "\nProbability:"


@dataclass(frozen=True)
class ExampleTensors:
    prompt_k: List[np.ndarray]
    guess_k: List[np.ndarray]
    sem_answer_k: List[np.ndarray]
    probability_k: List[np.ndarray]
    probability_q: List[np.ndarray]


class StrictProbabilitySpanError(ValueError):
    """Raised when input decoded_tokens violate strict probability span expectations."""


class MissingDecodedTokensError(ValueError):
    """Raised when processed H5 response does not include decoded_tokens."""


def _token_index_for_char_offset(decoded_tokens: Sequence[str], char_offset: int) -> int:
    cumulative = 0
    for i, tok in enumerate(decoded_tokens):
        cumulative += len(tok)
        if cumulative > char_offset:
            return i
    return max(0, len(decoded_tokens) - 1)


def parse_guess_and_probability_indices(decoded_tokens: Sequence[str]) -> Tuple[int, int, int] | None:
    full_str = "".join(decoded_tokens)
    if not full_str.startswith(GUESS_PREFIX):
        return None

    last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1
    rfind_start = full_str.rfind(PROBABILITY_MARKER)
    if rfind_start < 0:
        return None

    first_prob_token_index = _token_index_for_char_offset(decoded_tokens, rfind_start)
    prob_whitespace_token_index = (
        _token_index_for_char_offset(decoded_tokens, rfind_start + len(PROBABILITY_MARKER) - 1) + 1
    )
    if prob_whitespace_token_index >= len(decoded_tokens):
        return None
    if decoded_tokens[prob_whitespace_token_index].strip() != "":
        return None
    end_prob_token_index = prob_whitespace_token_index + 1

    if (
        last_guess_token_index <= 0
        or last_guess_token_index >= len(decoded_tokens)
        or end_prob_token_index >= len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None
    return last_guess_token_index, first_prob_token_index, end_prob_token_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one attention table PNG per probability token position using "
            "softmax(QK^T/sqrt(d_k)) over prompt/guess/semantic/probability key groups."
        )
    )
    parser.add_argument("--embeddings_h5", type=str, required=True, help="Processed embeddings H5 path.")
    parser.add_argument(
        "--expected_probability_tokens",
        type=int,
        default=7,
        help="Expected number of Probability span tokens (strict, default: 7).",
    )
    parser.add_argument(
        "--expected_guess_tokens",
        type=int,
        default=5,
        help="Expected number of Guess span tokens (default: 5).",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of attention heads used to split q/k hidden dimensions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to attention_probability_span_analysis/results/<run_id>/",
    )
    parser.add_argument(
        "--include_self_probability_attention",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Include attention from probability token position p to its own key (prob_p) "
            "in the table for position p."
        ),
    )
    parser.add_argument(
        "--include_layer_averaged_tables",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Generate additional per-probability tables with rows aggregated by layer "
            "(mean across heads), while keeping standard layer.head tables."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(cli_output_dir: str | None) -> Path:
    if cli_output_dir:
        out_dir = Path(cli_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    base_dir = SCRIPT_DIR / "results"
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    run_idx = max((int(p.name) for p in existing), default=0) + 1
    out_dir = base_dir / str(run_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_scalar(value.item())
    return value


def _read_string_list(node: h5py.Dataset | h5py.Group) -> List[str]:
    if isinstance(node, h5py.Dataset):
        raw = node[()]
        if isinstance(raw, np.ndarray):
            values = raw.tolist()
        else:
            values = [raw]
        return [str(_decode_scalar(v)) for v in values]

    node_type = node.attrs.get("__type__")
    if isinstance(node_type, bytes):
        node_type = node_type.decode("utf-8")
    if node_type not in {"list", "tuple"}:
        raise ValueError(f"Expected decoded_tokens to be list/tuple group, got type={node_type!r}")

    length = int(node.attrs.get("__len__", 0))
    out: List[str] = []
    for i in range(length):
        out.append(str(_decode_scalar(node[str(i)][()])))
    return out


def _require_group_path(root: h5py.Group, path_parts: Sequence[str]) -> h5py.Group:
    current: h5py.Group | h5py.Dataset = root
    joined = "/".join(path_parts)
    for part in path_parts:
        if not isinstance(current, h5py.Group) or part not in current:
            raise KeyError(f"Missing required group path: {joined}")
        current = current[part]
    if not isinstance(current, h5py.Group):
        raise ValueError(f"Expected group at path {joined}, found dataset")
    return current


def _read_list_of_arrays(list_group: h5py.Group, *, expected_len: int | None = None) -> List[np.ndarray]:
    node_type = list_group.attrs.get("__type__")
    if isinstance(node_type, bytes):
        node_type = node_type.decode("utf-8")
    if node_type not in {"list", "tuple"}:
        raise ValueError(f"Expected list/tuple group, got {node_type!r}")
    length = int(list_group.attrs.get("__len__", 0))
    if expected_len is not None and length != expected_len:
        raise ValueError(f"Expected list length {expected_len}, got {length}")
    out: List[np.ndarray] = []
    for i in range(length):
        child = list_group[str(i)]
        if not isinstance(child, h5py.Dataset):
            raise ValueError(f"Expected dataset at index {i}, found group")
        out.append(np.asarray(child[()], dtype=np.float32))
    return out


def _read_optional_list(root: h5py.Group, path_parts: Sequence[str]) -> List[np.ndarray] | None:
    group = _require_group_path(root, path_parts)
    node_type = group.attrs.get("__type__")
    if isinstance(node_type, bytes):
        node_type = node_type.decode("utf-8")
    if node_type == "none":
        return None
    return _read_list_of_arrays(group)


def _extract_layer_hidden(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"Expected rank-4 tensor [layers,batch,seq,hidden], got shape={arr.shape}")
    if arr.shape[1] != 1 or arr.shape[2] < 1:
        raise ValueError(f"Expected batch=1 and seq>=1 tensor, got shape={arr.shape}")
    return arr[:, 0, -1, :]


def _split_heads(layer_hidden: np.ndarray, num_heads: int) -> np.ndarray:
    if layer_hidden.ndim != 2:
        raise ValueError(f"Expected [layers, hidden_dim], got shape={layer_hidden.shape}")
    hidden_dim = layer_hidden.shape[1]
    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"Hidden dim {hidden_dim} is not divisible by num_heads={num_heads}. "
            "Set --num_heads to match model configuration."
        )
    head_dim = hidden_dim // num_heads
    return layer_hidden.reshape(layer_hidden.shape[0], num_heads, head_dim)


def _to_token_layer_heads(arrays: Sequence[np.ndarray], num_heads: int) -> np.ndarray:
    if not arrays:
        raise ValueError("Expected non-empty token tensor list")
    split = [_split_heads(_extract_layer_hidden(arr), num_heads) for arr in arrays]
    return np.stack(split, axis=0)  # [tokens, layers, heads, head_dim]


def _to_token_layer_kv_heads(arrays: Sequence[np.ndarray], q_head_dim: int) -> np.ndarray:
    if not arrays:
        raise ValueError("Expected non-empty token tensor list")
    if q_head_dim <= 0:
        raise ValueError(f"q_head_dim must be positive, got {q_head_dim}.")

    split: List[np.ndarray] = []
    for arr in arrays:
        layer_hidden = _extract_layer_hidden(arr)
        if layer_hidden.ndim != 2:
            raise ValueError(f"Expected [layers, hidden_dim], got shape={layer_hidden.shape}")
        hidden_dim = layer_hidden.shape[1]
        if hidden_dim % q_head_dim != 0:
            raise ValueError(
                f"K hidden dim {hidden_dim} is not divisible by Q head dim {q_head_dim}. "
                "Cannot infer KV heads for GQA."
            )
        num_kv_heads = hidden_dim // q_head_dim
        split.append(layer_hidden.reshape(layer_hidden.shape[0], num_kv_heads, q_head_dim))
    return np.stack(split, axis=0)  # [tokens, layers, kv_heads, head_dim]


def _kv_head_index_for_q_head(q_head_idx: int, num_q_heads: int, num_kv_heads: int) -> int:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads={num_q_heads} is not divisible by num_kv_heads={num_kv_heads}; "
            "cannot map query heads to KV heads."
        )
    group_size = num_q_heads // num_kv_heads
    return q_head_idx // group_size


def _validate_probability_span(decoded_tokens: Sequence[str], expected_probability_tokens: int) -> Tuple[int, int]:
    parsed = parse_guess_and_probability_indices(list(decoded_tokens))
    if parsed is None:
        raise StrictProbabilitySpanError(
            "Could not parse Guess/Probability token spans from decoded tokens."
        )
    _, first_prob_idx, end_prob_idx = parsed
    prob_len = end_prob_idx - first_prob_idx + 1
    if prob_len != expected_probability_tokens:
        raise StrictProbabilitySpanError(
            f"Expected probability span length {expected_probability_tokens}, got {prob_len} "
            f"(first_prob={first_prob_idx}, end_prob={end_prob_idx})."
        )
    return first_prob_idx, end_prob_idx


def _load_example_tensors(
    response_group: h5py.Group,
    *,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
) -> ExampleTensors:
    prompt_k = _read_list_of_arrays(
        _require_group_path(response_group, ("embeddings_prompt_k_tokens", "k"))
    )
    sem_answer_k = _read_list_of_arrays(
        _require_group_path(response_group, ("embeddings_sem_answer_k_tokens", "k"))
    )
    if not sem_answer_k:
        raise ValueError("Semantic-answer K token list is empty.")

    guess_k = _read_optional_list(response_group, ("embeddings_guess", "k"))
    probability_k = _read_optional_list(response_group, ("embeddings_probability", "k"))
    probability_q = _read_optional_list(response_group, ("embeddings_probability", "q"))

    if guess_k is None or probability_k is None or probability_q is None:
        raise ValueError(
            "Required q/k tensors missing. Re-run processing with --collect_qkvo_embeddings."
        )

    if len(guess_k) != expected_guess_tokens:
        raise ValueError(
            f"Expected {expected_guess_tokens} guess token tensors, got {len(guess_k)}."
        )
    if len(probability_k) != expected_probability_tokens:
        raise ValueError(
            f"Expected {expected_probability_tokens} probability K tensors, got {len(probability_k)}."
        )
    if len(probability_q) != expected_probability_tokens:
        raise ValueError(
            f"Expected {expected_probability_tokens} probability Q tensors, got {len(probability_q)}."
        )

    return ExampleTensors(
        prompt_k=prompt_k,
        guess_k=guess_k,
        sem_answer_k=sem_answer_k,
        probability_k=probability_k,
        probability_q=probability_q,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    denom = np.sum(e)
    if denom <= 0.0:
        raise ValueError("Softmax denominator is non-positive.")
    return e / denom


def _compute_row_values_for_position(
    *,
    prompt_k_lh: np.ndarray,
    guess_k_lh: np.ndarray,
    sem_k_lh: np.ndarray,
    prob_k_lh: np.ndarray,
    prob_q_lh: np.ndarray,
    probability_position: int,
    include_self_probability_attention: bool,
) -> np.ndarray:
    _, num_layers, num_q_heads, q_head_dim = prob_q_lh.shape
    _, _, num_kv_heads, k_head_dim = prob_k_lh.shape
    if q_head_dim != k_head_dim:
        raise ValueError(
            f"Q/K head dimension mismatch: q_head_dim={q_head_dim}, k_head_dim={k_head_dim}."
        )
    prompt_len = prompt_k_lh.shape[0]
    sem_len = sem_k_lh.shape[0]
    if prompt_len <= 0:
        raise ValueError("Prompt K token count must be positive.")
    if sem_len <= 0:
        raise ValueError("Semantic-answer K token count must be positive.")

    prob_col_count = probability_position + (1 if include_self_probability_attention else 0)
    column_count = 1 + guess_k_lh.shape[0] + 1 + prob_col_count
    out = np.zeros((num_layers * num_q_heads, column_count), dtype=np.float64)
    scale = math.sqrt(float(q_head_dim))

    for layer_idx in range(num_layers):
        for q_head_idx in range(num_q_heads):
            kv_head_idx = _kv_head_index_for_q_head(q_head_idx, num_q_heads, num_kv_heads)
            q = prob_q_lh[probability_position, layer_idx, q_head_idx, :]  # [head_dim]
            prompt_keys = prompt_k_lh[:, layer_idx, kv_head_idx, :]  # [prompt_len, head_dim]
            guess_keys = guess_k_lh[:, layer_idx, kv_head_idx, :]  # [n_guess, head_dim]
            sem_keys = sem_k_lh[:, layer_idx, kv_head_idx, :]  # [sem_len, head_dim]
            prob_key_end = probability_position + (1 if include_self_probability_attention else 0)
            prior_prob_keys = (
                prob_k_lh[:prob_key_end, layer_idx, kv_head_idx, :]
                if prob_key_end > 0
                else np.zeros((0, q_head_dim), dtype=np.float32)
            )

            keys = np.concatenate((prompt_keys, guess_keys, sem_keys, prior_prob_keys), axis=0)
            logits = np.matmul(keys, q) / scale
            attn = _softmax(logits)

            prompt_slice_end = prompt_len
            guess_slice_end = prompt_slice_end + guess_keys.shape[0]
            sem_slice_end = guess_slice_end + sem_len

            prompt_mass = float(np.sum(attn[:prompt_slice_end])) / float(prompt_len)
            guess_mass = attn[prompt_slice_end:guess_slice_end]
            sem_mass = float(np.sum(attn[guess_slice_end:sem_slice_end])) / float(sem_len)
            prior_prob_mass = (
                attn[sem_slice_end:] if prob_key_end > 0 else np.zeros((0,), dtype=np.float64)
            )

            row = np.concatenate(
                (
                    np.asarray([prompt_mass], dtype=np.float64),
                    guess_mass.astype(np.float64),
                    np.asarray([sem_mass], dtype=np.float64),
                    prior_prob_mass.astype(np.float64),
                ),
                axis=0,
            )
            row_idx = layer_idx * num_q_heads + q_head_idx
            out[row_idx, :] = row
    return out


def _column_labels(
    expected_guess_tokens: int,
    probability_position: int,
    include_self_probability_attention: bool,
) -> List[str]:
    labels = ["mean_prompt"]
    labels.extend(f"guess_{i}" for i in range(expected_guess_tokens))
    labels.append("mean_sem_ans")
    prob_end = probability_position + (1 if include_self_probability_attention else 0)
    labels.extend(f"prob_{j}" for j in range(prob_end))
    return labels


def _row_labels(num_layers: int, num_heads: int) -> List[str]:
    return [f"{layer}.{head}" for layer in range(num_layers) for head in range(num_heads)]


def _layer_labels(num_layers: int) -> List[str]:
    return [f"layer_{layer}" for layer in range(num_layers)]


def _average_heads_by_layer(matrix: np.ndarray, num_heads: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected rank-2 table matrix, got shape={matrix.shape}.")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}.")
    total_rows, n_cols = matrix.shape
    if total_rows % num_heads != 0:
        raise ValueError(
            f"Cannot average heads by layer: total_rows={total_rows} is not divisible by num_heads={num_heads}."
        )
    num_layers = total_rows // num_heads
    reshaped = matrix.reshape(num_layers, num_heads, n_cols)
    return np.mean(reshaped, axis=1)


def _render_table_png(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    n_rows, n_cols = matrix.shape
    fig_width = max(12.0, 3.2 + 1.2 * n_cols)
    fig_height = max(10.0, 2.5 + 0.17 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)

    formatted = [[f"{v:.6f}" for v in row] for row in matrix]
    table = ax.table(
        cellText=formatted,
        rowLabels=list(row_labels),
        colLabels=list(col_labels),
        cellLoc="center",
        loc="center",
    )

    # Per-row normalization: within each layer.head row, map the minimum
    # attention to alpha=0 and the maximum to alpha=1 for a blue shade.
    blue_rgb = (0.0, 0.0, 1.0)
    eps = 1e-12
    for r in range(n_rows):
        row_vals = matrix[r]
        row_min = float(np.min(row_vals))
        row_max = float(np.max(row_vals))
        denom = row_max - row_min
        for c in range(n_cols):
            if denom <= eps:
                alpha = 0.0
            else:
                alpha = float((row_vals[c] - row_min) / denom)
            table[(r + 1, c)].set_facecolor((*blue_rgb, alpha))
            # Improve readability for highly opaque cells.
            table[(r + 1, c)].get_text().set_color("white" if alpha >= 0.6 else "black")

    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.08)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_config(
    *,
    output_dir: Path,
    embeddings_h5: Path,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
    include_self_probability_attention: bool,
    include_layer_averaged_tables: bool,
    num_heads: int,
    total_examples_seen: int,
    examples_used: int,
    examples_skipped: int,
) -> None:
    lines = [
        "Attention Probability Span Analysis Config",
        "=========================================",
        "",
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        "",
        "[Inputs]",
        f"embeddings_h5={embeddings_h5}",
        "",
        "[Run]",
        f"expected_probability_tokens={expected_probability_tokens}",
        f"expected_guess_tokens={expected_guess_tokens}",
        f"include_self_probability_attention={include_self_probability_attention}",
        f"include_layer_averaged_tables={include_layer_averaged_tables}",
        f"num_heads={num_heads}",
        f"total_examples_seen={total_examples_seen}",
        f"examples_used={examples_used}",
        f"examples_skipped={examples_skipped}",
    ]
    (output_dir / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_example_ids(emb_h5: h5py.File) -> Iterable[str]:
    emb_examples = _require_group_path(emb_h5, ("examples",))
    def _sort_key(example_id: str) -> Tuple[int, int | str]:
        try:
            return (0, int(example_id))
        except ValueError:
            return (1, example_id)

    return sorted(emb_examples.keys(), key=_sort_key)


def main() -> None:
    args = parse_args()
    embeddings_h5 = Path(args.embeddings_h5)
    output_dir = resolve_output_dir(args.output_dir)

    if not embeddings_h5.exists():
        raise FileNotFoundError(f"Embeddings H5 not found: {embeddings_h5}")
    if args.expected_probability_tokens <= 0:
        raise ValueError("--expected_probability_tokens must be positive.")
    if args.expected_guess_tokens <= 0:
        raise ValueError("--expected_guess_tokens must be positive.")
    if args.num_heads <= 0:
        raise ValueError("--num_heads must be positive.")

    sum_tables: Dict[int, np.ndarray] = {}
    count_tables: Dict[int, int] = {p: 0 for p in range(args.expected_probability_tokens)}
    skipped_examples = 0
    used_examples = 0
    total_seen = 0
    row_labels: List[str] | None = None
    num_q_heads_used: int | None = None

    with h5py.File(embeddings_h5, "r") as emb_h5:
        emb_examples = _require_group_path(emb_h5, ("examples",))

        for example_id in _iter_example_ids(emb_h5):
            print(f"Processing example {example_id}")
            total_seen += 1
            try:
                emb_example = _require_group_path(emb_examples, (example_id,))
                response_group = _require_group_path(emb_example, ("responses", "0"))
                if "decoded_tokens" not in response_group:
                    raise MissingDecodedTokensError(
                        "Missing decoded_tokens in processed response. "
                        "Re-run processing with --attention_score_tokenwise_k_mode."
                    )
                decoded_tokens = _read_string_list(response_group["decoded_tokens"])
                first_prob_idx, _ = _validate_probability_span(
                    decoded_tokens, args.expected_probability_tokens
                )
                parsed = parse_guess_and_probability_indices(decoded_tokens)
                if parsed is None:
                    raise StrictProbabilitySpanError(
                        "Could not parse Guess/Probability token spans from decoded tokens."
                    )
                last_guess_idx, _, _ = parsed

                tensors = _load_example_tensors(
                    response_group,
                    expected_probability_tokens=args.expected_probability_tokens,
                    expected_guess_tokens=args.expected_guess_tokens,
                )

                expected_sem_len = first_prob_idx - last_guess_idx
                if expected_sem_len <= 0:
                    raise ValueError("Semantic-answer token span is empty.")
                if len(tensors.sem_answer_k) != expected_sem_len:
                    raise ValueError(
                        f"Semantic-answer K length mismatch: got {len(tensors.sem_answer_k)}, "
                        f"expected {expected_sem_len}."
                    )

                prob_q_lh = _to_token_layer_heads(tensors.probability_q, args.num_heads)
                q_head_dim = prob_q_lh.shape[-1]
                prompt_k_lh = _to_token_layer_kv_heads(tensors.prompt_k, q_head_dim)
                guess_k_lh = _to_token_layer_kv_heads(tensors.guess_k, q_head_dim)
                sem_k_lh = _to_token_layer_kv_heads(tensors.sem_answer_k, q_head_dim)
                prob_k_lh = _to_token_layer_kv_heads(tensors.probability_k, q_head_dim)

                if row_labels is None:
                    row_labels = _row_labels(prob_q_lh.shape[1], prob_q_lh.shape[2])
                if num_q_heads_used is None:
                    num_q_heads_used = prob_q_lh.shape[2]

                for p in range(args.expected_probability_tokens):
                    per_example = _compute_row_values_for_position(
                        prompt_k_lh=prompt_k_lh,
                        guess_k_lh=guess_k_lh,
                        sem_k_lh=sem_k_lh,
                        prob_k_lh=prob_k_lh,
                        prob_q_lh=prob_q_lh,
                        probability_position=p,
                        include_self_probability_attention=args.include_self_probability_attention,
                    )
                    if p not in sum_tables:
                        sum_tables[p] = np.zeros_like(per_example, dtype=np.float64)
                    sum_tables[p] += per_example
                    count_tables[p] += 1

                used_examples += 1
            except Exception as exc:
                # Keep malformed examples from crashing a full run unless strict
                # probability span validation failed.
                if isinstance(exc, (StrictProbabilitySpanError, MissingDecodedTokensError)):
                    raise
                skipped_examples += 1
                print(f"Skipping example {example_id} due to error: {exc}")
                continue

    if used_examples == 0:
        raise RuntimeError("No valid examples were processed; cannot produce tables.")
    if row_labels is None:
        raise RuntimeError("Could not infer row labels from processed tensors.")
    if num_q_heads_used is None:
        raise RuntimeError("Could not infer number of query heads from processed tensors.")

    mean_tables: Dict[int, np.ndarray] = {}
    summary_payload = {
        "expected_probability_tokens": args.expected_probability_tokens,
        "expected_guess_tokens": args.expected_guess_tokens,
        "include_self_probability_attention": args.include_self_probability_attention,
        "include_layer_averaged_tables": args.include_layer_averaged_tables,
        "num_heads": args.num_heads,
        "total_examples_seen": total_seen,
        "examples_used": used_examples,
        "examples_skipped": skipped_examples,
        "tables": {},
    }

    for p in range(args.expected_probability_tokens):
        count = count_tables[p]
        if count <= 0:
            raise RuntimeError(f"No examples contributed to probability position {p}.")
        mean_table = sum_tables[p] / float(count)
        mean_tables[p] = mean_table
        columns = _column_labels(
            args.expected_guess_tokens,
            p,
            args.include_self_probability_attention,
        )
        png_path = output_dir / f"attention_table_probability_pos_{p}.png"
        _render_table_png(
            matrix=mean_table,
            row_labels=row_labels,
            col_labels=columns,
            output_path=png_path,
            title=f"Probability Position {p}: Normalized Attention Table",
        )
        summary_payload["tables"][f"probability_pos_{p}"] = {
            "columns": columns,
            "shape": list(mean_table.shape),
            "png": str(png_path),
            "values": mean_table.tolist(),
        }
        if args.include_layer_averaged_tables:
            layer_avg_table = _average_heads_by_layer(mean_table, num_q_heads_used)
            layer_avg_png_path = output_dir / f"attention_table_probability_pos_{p}__layer_avg.png"
            _render_table_png(
                matrix=layer_avg_table,
                row_labels=_layer_labels(layer_avg_table.shape[0]),
                col_labels=columns,
                output_path=layer_avg_png_path,
                title=f"Probability Position {p}: Layer-Averaged Attention Table",
            )
            summary_payload["tables"][f"probability_pos_{p}"]["layer_avg_shape"] = list(
                layer_avg_table.shape
            )
            summary_payload["tables"][f"probability_pos_{p}"]["layer_avg_png"] = str(
                layer_avg_png_path
            )
            summary_payload["tables"][f"probability_pos_{p}"]["layer_avg_values"] = (
                layer_avg_table.tolist()
            )

    npz_payload = {f"probability_pos_{p}": table.astype(np.float32) for p, table in mean_tables.items()}
    np.savez(output_dir / "attention_tables.npz", **npz_payload)
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_config(
        output_dir=output_dir,
        embeddings_h5=embeddings_h5,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_guess_tokens=args.expected_guess_tokens,
        include_self_probability_attention=args.include_self_probability_attention,
        include_layer_averaged_tables=args.include_layer_averaged_tables,
        num_heads=args.num_heads,
        total_examples_seen=total_seen,
        examples_used=used_examples,
        examples_skipped=skipped_examples,
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
