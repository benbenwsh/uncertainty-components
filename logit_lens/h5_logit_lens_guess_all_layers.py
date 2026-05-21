"""
Logit-lens analysis for guess token positions 0..5 across all layers.

For each hidden state (token_pos, layer_idx), this script applies:
    normed_hidden = RMSNorm(hidden; model.norm.weight, eps)
    logits = normed_hidden @ W_U^T + b
    probs = softmax(logits)

Then it:
- records top-n tokens for a small number of individual examples
- computes cosine similarity between probs(token_pos, layer_idx) and
  probs(token=5, last_layer) per example
- saves per-token and combined cosine plots (single combined dataset)

Input format:
- HDF5 files produced by process_generations_verbalised_embeddings_h5.py
- Root group "examples", where each example stores a dict-like structure
  containing "responses" and per-response "embeddings_guess".
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

REF_TOKEN_INDEX = 5
NUM_GUESS_TOKENS = 6
EPS = 1e-12


def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_h5_node(node):
    """Recursively read object written by native HDF5 writer."""
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "S":
                return np.array([x.decode("utf-8") for x in value], dtype=object).tolist()
            if value.dtype.kind == "O":
                return [_decode_h5_string(x) for x in value.tolist()]
            return value
        return _decode_h5_string(value)

    node_type = node.attrs.get("__type__", "")
    if isinstance(node_type, bytes):
        node_type = node_type.decode("utf-8")
    if node_type == "none":
        return None
    if node_type in ("list", "tuple"):
        length = int(node.attrs.get("__len__", len(node.keys())))
        items = [_read_h5_node(node[str(i)]) for i in range(length)]
        return tuple(items) if node_type == "tuple" else items

    out = {}
    for key in node.keys():
        out[key] = _read_h5_node(node[key])
    return out


def _read_token_stack_ds(example_group, token_idx: int):
    """
    Fast-path loader: returns HDF5 dataset for embeddings_guess[token_idx] if present.
    Structure expected from native writer:
      examples/<id>/responses/0/embeddings_guess/<token_idx>
    """
    responses = example_group.get("responses")
    if responses is None or not isinstance(responses, h5py.Group):
        return None
    resp0 = responses.get("0")
    if resp0 is None or not isinstance(resp0, h5py.Group):
        return None
    emb_guess = resp0.get("embeddings_guess")
    if emb_guess is None or not isinstance(emb_guess, h5py.Group):
        return None
    ds = emb_guess.get(str(token_idx))
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    return ds


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _stack_guess_token(emb) -> np.ndarray:
    arr = _tensor_to_numpy(emb)
    return arr[:, 0, -1, :]


def _example_stacks(emb_guess: list) -> list[np.ndarray] | None:
    if len(emb_guess) != NUM_GUESS_TOKENS:
        return None
    stacks = [_stack_guess_token(emb_guess[t]) for t in range(NUM_GUESS_TOKENS)]
    n_layers, hidden_dim = stacks[0].shape
    for s in stacks:
        if s.shape != (n_layers, hidden_dim):
            return None
    return stacks


def _iter_examples_from_h5(path: str, max_examples: int | None = None):
    """
    Stream valid examples from HDF5 one-by-one as
    (example_id, ex_array[NUM_GUESS_TOKENS, n_layers, hidden_dim]).
    """
    emitted = 0
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        desc = f"Streaming examples ({Path(path).name})"
        for example_id in tqdm(examples_group.keys(), desc=desc, leave=False):
            example_id = str(example_id)
            example_group = examples_group[example_id]

            token_ds = [_read_token_stack_ds(example_group, t) for t in range(NUM_GUESS_TOKENS)]
            if all(ds is not None for ds in token_ds):
                first = token_ds[0]
                if first.ndim == 4:
                    n_layers = int(first.shape[0])
                    hidden_dim = int(first.shape[-1])
                    ok = True
                    for ds in token_ds:
                        if ds.ndim != 4:
                            ok = False
                            break
                        if int(ds.shape[0]) != n_layers or int(ds.shape[-1]) != hidden_dim:
                            ok = False
                            break
                    if ok:
                        ex = np.zeros((NUM_GUESS_TOKENS, n_layers, hidden_dim), dtype=np.float32)
                        for t, ds in enumerate(token_ds):
                            ex[t] = np.asarray(ds[:, 0, -1, :], dtype=np.float32)
                        yield example_id, ex
                        emitted += 1
                        if max_examples is not None and emitted >= max_examples:
                            return
                        continue

            # Slow-path fallback for unexpected legacy structure.
            example_data = _read_h5_node(example_group)
            if not isinstance(example_data, dict):
                continue
            responses = example_data.get("responses", [])
            if not responses:
                continue
            emb_guess = responses[0].get("embeddings_guess")
            if emb_guess is None:
                continue
            stacks = _example_stacks(emb_guess)
            if stacks is None:
                continue
            ex = np.stack(stacks, axis=0).astype(np.float32, copy=False)
            yield example_id, ex
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return


def _iter_examples_from_paths(paths: list[str], max_examples_per_path: int | None = None):
    for path in paths:
        for example_id, ex in _iter_examples_from_h5(path, max_examples=max_examples_per_path):
            # ex: [NUM_GUESS_TOKENS, n_layers, hidden_dim]
            yield path, example_id, ex


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_unembedding(model_name_or_path: str, device: torch.device) -> tuple:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as exc:
        print(
            f"WARNING: fast tokenizer load failed ({exc}). "
            "Falling back to slow tokenizer (use_fast=False)."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    if not hasattr(model, "lm_head"):
        raise ValueError("Model has no lm_head; cannot perform logit lens.")
    core_model = getattr(model, "model", model)
    final_norm = getattr(core_model, "norm", None)
    if final_norm is None or not hasattr(final_norm, "weight"):
        raise ValueError("Model has no final norm weight; expected model.norm.weight for logit lens.")
    unembed_w = model.lm_head.weight.detach().to(device=device, dtype=torch.float32)
    unembed_b = getattr(model.lm_head, "bias", None)
    if unembed_b is not None:
        unembed_b = unembed_b.detach().to(device=device, dtype=torch.float32)
    norm_weight = final_norm.weight.detach().to(device=device, dtype=torch.float32)
    norm_eps = getattr(final_norm, "eps", None)
    if norm_eps is None:
        norm_eps = getattr(final_norm, "variance_epsilon", 1e-6)
    norm_eps = float(norm_eps)
    print(f"norm_eps: {norm_eps}")
    return tokenizer, unembed_w, unembed_b, norm_weight, norm_eps


def _apply_rmsnorm(hidden: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
    hidden_sq_mean = hidden.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(hidden_sq_mean + norm_eps)
    return (hidden * inv_rms) * norm_weight


def _probs_from_hidden(
    hidden: torch.Tensor,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    hidden = _apply_rmsnorm(hidden, norm_weight=norm_weight, norm_eps=norm_eps)
    logits = hidden @ w_u.T # matmul transpose
    if b_u is not None:
        logits = logits + b_u
    # logits = logits - torch.max(logits)
    probs = torch.softmax(logits, dim=-1)
    return probs


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    return float(np.dot(a, b) / max(an * bn, EPS))


def _minmax_normalize_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    c_min = float(np.min(curve))
    c_max = float(np.max(curve))
    denom = c_max - c_min
    if denom <= EPS:
        return np.zeros_like(curve, dtype=np.float32)
    return ((curve - c_min) / denom).astype(np.float32, copy=False)


def _analyze_combined_streaming(
    input_paths: list[str],
    n_layers: int,
    hidden_dim: int,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    device: torch.device,
    max_examples_per_path: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        cosine_values: (n_examples, n_tokens, n_layers)
        kl_values: (n_examples, n_tokens, n_layers)
        ce_values: (n_examples, n_tokens, n_layers)
        mean_minmax_top1_by_token: (n_tokens, n_layers)
        mean_minmax_ref_top1_by_token: (n_tokens - 1, n_layers), token positions 0..4
        combined_example_ids: list[str], one per example in cosine_values
    """
    cosine_rows: list[np.ndarray] = []
    kl_rows: list[np.ndarray] = []
    ce_rows: list[np.ndarray] = []
    combined_example_ids: list[str] = []
    n_examples = 0
    top1_minmax_sum = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float64)
    ref_top1_minmax_sum = np.zeros((NUM_GUESS_TOKENS - 1, n_layers), dtype=np.float64)

    with torch.no_grad():
        for source_path, example_id, ex in _iter_examples_from_paths(
            input_paths, max_examples_per_path=max_examples_per_path
        ):
            if ex.shape != (NUM_GUESS_TOKENS, n_layers, hidden_dim):
                print(f"ERROR: Example shape mismatch: {ex.shape} != (NUM_GUESS_TOKENS, n_layers, hidden_dim)")
                continue
            combined_example_id = f"{Path(source_path).name}:{example_id}"
            ref_hidden_np = ex[REF_TOKEN_INDEX, n_layers - 1, :]
            ref_hidden = torch.from_numpy(ref_hidden_np).to(device=device, dtype=torch.float32)
            ref_probs_t = _probs_from_hidden(ref_hidden, w_u, b_u, norm_weight, norm_eps)
            ref_probs = ref_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
            ref_log_probs = np.log(ref_probs.clip(min=EPS))
            ref_entropy = -float(np.sum(ref_probs * ref_log_probs))
            ref_norm = float(np.linalg.norm(ref_probs))
            ref_top1_id = int(np.argmax(ref_probs))

            ex_cos = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
            ex_kl = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
            ex_ce = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
            for token_pos in range(NUM_GUESS_TOKENS):
                hidden_layers = torch.from_numpy(ex[token_pos]).to(device=device, dtype=torch.float32)  # [L, H]
                probs_t = _probs_from_hidden(hidden_layers, w_u, b_u, norm_weight, norm_eps)  # [L, V]
                probs_by_layer = probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
                log_probs_by_layer = np.log(probs_by_layer.clip(min=EPS))

                numer = np.sum(probs_by_layer * ref_probs[None, :], axis=-1)
                denom = np.maximum(np.linalg.norm(probs_by_layer, axis=-1) * ref_norm, EPS)
                ex_cos[token_pos] = (numer / denom).astype(np.float32, copy=False)
                ce_layers = -np.sum(ref_probs[None, :] * log_probs_by_layer, axis=-1)
                ex_ce[token_pos] = ce_layers.astype(np.float32, copy=False)
                ex_kl[token_pos] = (ce_layers - ref_entropy).astype(np.float32, copy=False)

                top1_id = int(np.argmax(probs_by_layer[-1]))
                top1_curve = probs_by_layer[:, top1_id].astype(np.float32, copy=False)
                top1_minmax_sum[token_pos] += _minmax_normalize_curve(top1_curve)

                if token_pos < REF_TOKEN_INDEX:
                    ref_curve = probs_by_layer[:, ref_top1_id].astype(np.float32, copy=False)
                    ref_top1_minmax_sum[token_pos] += _minmax_normalize_curve(ref_curve)

            cosine_rows.append(ex_cos)
            kl_rows.append(ex_kl)
            ce_rows.append(ex_ce)
            combined_example_ids.append(combined_example_id)
            n_examples += 1

    if n_examples == 0:
        joined_paths = ", ".join(input_paths)
        raise ValueError(f"No valid examples found in input path(s): {joined_paths}")

    cosine_values = np.stack(cosine_rows, axis=0)
    kl_values = np.stack(kl_rows, axis=0)
    ce_values = np.stack(ce_rows, axis=0)
    mean_minmax_top1_by_token = (top1_minmax_sum / n_examples).astype(np.float32, copy=False)
    mean_minmax_ref_top1_by_token = (ref_top1_minmax_sum / n_examples).astype(np.float32, copy=False)
    return (
        cosine_values,
        kl_values,
        ce_values,
        mean_minmax_top1_by_token,
        mean_minmax_ref_top1_by_token,
        combined_example_ids,
    )


def _infer_h5_shape(paths: list[str], max_examples_per_path: int | None = None) -> tuple[int, int, int]:
    """
    Returns (n_valid_examples, n_layers, hidden_dim) by scanning all input paths lazily.
    """
    n_valid = 0
    n_layers = None
    hidden_dim = None
    for _path, _example_id, ex in _iter_examples_from_paths(paths, max_examples_per_path=max_examples_per_path):
        if n_layers is None:
            n_layers = int(ex.shape[1])
            hidden_dim = int(ex.shape[2])
        if ex.shape[1] != n_layers or ex.shape[2] != hidden_dim:
            continue
        n_valid += 1
    if n_valid == 0 or n_layers is None or hidden_dim is None:
        joined_paths = ", ".join(paths)
        raise ValueError(f"No valid examples found in input path(s): {joined_paths}")
    return n_valid, n_layers, hidden_dim


def _topn_for_distribution(probs: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_n, int(probs.shape[0]))
    idx = np.argpartition(-probs, k - 1)[:k]
    sorted_local = idx[np.argsort(-probs[idx])]
    top_ids = np.zeros((top_n,), dtype=np.int32)
    top_vals = np.zeros((top_n,), dtype=np.float32)
    top_ids[:k] = sorted_local
    top_vals[:k] = probs[sorted_local]
    return top_ids, top_vals


def _collect_report_examples_from_path(
    source_path: str,
    n_layers: int,
    hidden_dim: int,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    device: torch.device,
    top_n: int,
    top_token_examples: int,
) -> list[dict]:
    report_examples: list[dict] = []
    with torch.no_grad():
        for example_id, ex in _iter_examples_from_h5(source_path, max_examples=top_token_examples):
            if ex.shape != (NUM_GUESS_TOKENS, n_layers, hidden_dim):
                continue
            combined_example_id = f"{Path(source_path).name}:{example_id}"
            ref_hidden_np = ex[REF_TOKEN_INDEX, n_layers - 1, :]
            ref_hidden = torch.from_numpy(ref_hidden_np).to(device=device, dtype=torch.float32)
            ref_probs_t = _probs_from_hidden(ref_hidden, w_u, b_u, norm_weight, norm_eps)
            ref_probs = ref_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
            ref_top1_id = int(np.argmax(ref_probs))

            top_ids_record = np.zeros((NUM_GUESS_TOKENS, n_layers, top_n), dtype=np.int32)
            top_vals_record = np.zeros((NUM_GUESS_TOKENS, n_layers, top_n), dtype=np.float32)
            final_top1_ids = np.zeros((NUM_GUESS_TOKENS,), dtype=np.int32)
            final_top1_curves = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
            ref_token_curves = np.zeros((NUM_GUESS_TOKENS - 1, n_layers), dtype=np.float32)

            for token_pos in range(NUM_GUESS_TOKENS):
                hidden_layers = torch.from_numpy(ex[token_pos]).to(device=device, dtype=torch.float32)  # [L, H]
                probs_t = _probs_from_hidden(hidden_layers, w_u, b_u, norm_weight, norm_eps)  # [L, V]
                probs_by_layer = probs_t.detach().cpu().numpy().astype(np.float32, copy=False)

                for layer_idx in range(n_layers):
                    probs = probs_by_layer[layer_idx]
                    top_ids, top_vals = _topn_for_distribution(probs, top_n=top_n)
                    top_ids_record[token_pos, layer_idx] = top_ids
                    top_vals_record[token_pos, layer_idx] = top_vals

                top1_id = int(np.argmax(probs_by_layer[-1]))
                final_top1_ids[token_pos] = top1_id
                final_top1_curves[token_pos] = probs_by_layer[:, top1_id].astype(np.float32, copy=False)
                if token_pos < REF_TOKEN_INDEX:
                    ref_token_curves[token_pos] = probs_by_layer[:, ref_top1_id].astype(np.float32, copy=False)

            report_examples.append(
                {
                    "combined_example_id": combined_example_id,
                    "source_path": source_path,
                    "example_id": example_id,
                    "top_ids": top_ids_record,
                    "top_vals": top_vals_record,
                    "final_top1_ids": final_top1_ids,
                    "final_top1_curves": final_top1_curves,
                    "ref_top1_id": int(ref_top1_id),
                    "ref_token_curves": ref_token_curves,
                }
            )
    return report_examples


def _format_token(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        token = tokenizer.decode([int(token_id)])
    decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return f"id={int(token_id)} token={repr(token)} decoded={repr(decoded)}"


def _format_token_decoded(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _save_topk_table_png(
    out_path: Path,
    tokenizer,
    top_ids: np.ndarray,   # [L, K]
    top_vals: np.ndarray,  # [L, K]
) -> None:
    n_layers, top_k = top_ids.shape
    col_labels = ["Layer"]
    for k in range(top_k):
        col_labels.append(f"Top-{k + 1} token")
        col_labels.append(f"Top-{k + 1} prob")

    rows = []
    for layer_idx in range(n_layers):
        row = [f"layer_{layer_idx + 1}"]
        for k in range(top_k):
            tok_id = int(top_ids[layer_idx, k])
            prob = float(top_vals[layer_idx, k])
            row.append(_format_token_decoded(tokenizer, tok_id))
            row.append(f"{prob:.6f}")
        rows.append(row)

    fig_w = max(11.0, 2.1 * len(col_labels))
    fig_h = max(6.0, 0.4 * n_layers + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    for c in range(len(col_labels)):
        header_cell = table[(0, c)]
        header_cell.set_facecolor((0.85, 0.9, 1.0, 1.0))
        header_cell.set_text_props(weight="bold")

    for r in range(n_layers):
        alpha = float(np.clip(top_vals[r, 0], 0.0, 1.0))
        for c in range(1, len(col_labels)):
            cell = table[(r + 1, c)]
            cell.set_facecolor((0.1, 0.3, 1.0, alpha))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_top_tokens_report(
    out_path: Path,
    tokenizer,
    top_n: int,
    top_token_examples: int,
    report_examples: list[dict],
    n_layers: int,
) -> None:
    lines: list[str] = []
    lines.append("LOGIT LENS TOP TOKENS REPORT")
    lines.append("=" * 80)
    lines.append(f"requested_examples: {top_token_examples}")
    lines.append(f"written_examples: {len(report_examples)}")
    lines.append(f"top_n: {top_n}")
    lines.append(f"token_positions: 0..{NUM_GUESS_TOKENS - 1}")
    lines.append(f"layers: 1..{n_layers}")
    lines.append("")

    for ex_idx, ex_record in enumerate(report_examples, start=1):
        lines.append(f"EXAMPLE {ex_idx}")
        lines.append("-" * 80)
        lines.append(f"combined_example_id: {ex_record['combined_example_id']}")
        lines.append(f"example_id: {ex_record['example_id']}")
        lines.append(f"source_path: {ex_record['source_path']}")
        for token_pos in range(NUM_GUESS_TOKENS):
            lines.append(f"[tok_{token_pos}_guess]")
            for layer_idx in range(n_layers):
                top_ids = ex_record["top_ids"][token_pos, layer_idx]
                top_vals = ex_record["top_vals"][token_pos, layer_idx]
                entries = [
                    f"{rank + 1}. {_format_token(tokenizer, int(tok_id))} prob={float(prob):.8f}"
                    for rank, (tok_id, prob) in enumerate(zip(top_ids, top_vals))
                ]
                lines.append(f"  layer_{layer_idx + 1}:")
                lines.extend([f"    {entry}" for entry in entries])
            lines.append("")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _write_topk_tables_and_example_curves(
    examples_dir: Path,
    tokenizer,
    report_examples: list[dict],
    n_layers: int,
) -> None:
    for ex_idx, ex_record in enumerate(report_examples, start=1):
        ref_top1_id = int(ex_record["ref_top1_id"])
        ref_label = _format_token(tokenizer, ref_top1_id)
        for token_pos in range(NUM_GUESS_TOKENS):
            table_out = examples_dir / f"topk_table_example_{ex_idx}_tok_{token_pos}_guess.png"
            _save_topk_table_png(
                out_path=table_out,
                tokenizer=tokenizer,
                top_ids=ex_record["top_ids"][token_pos],
                top_vals=ex_record["top_vals"][token_pos],
            )

            final_top1_id = int(ex_record["final_top1_ids"][token_pos])
            token_label = _format_token(tokenizer, final_top1_id)
            top1_curve = np.asarray(ex_record["final_top1_curves"][token_pos], dtype=np.float32)
            _plot_metric_by_layer(
                layer_numbers=list(range(1, n_layers + 1)),
                values=top1_curve,
                std_values=None,
                token_pos=token_pos,
                title=f"Example {ex_idx}: final top-1 token probability by layer",
                ylabel=f"P(token={token_label})",
                out_name=f"top1_final_token_prob_by_layer_example_{ex_idx}_tok_{token_pos}_guess.png",
                out_dir=examples_dir,
                show_error_bars=False,
            )

            if token_pos < REF_TOKEN_INDEX:
                ref_curve = np.asarray(ex_record["ref_token_curves"][token_pos], dtype=np.float32)
                _plot_metric_by_layer(
                    layer_numbers=list(range(1, n_layers + 1)),
                    values=ref_curve,
                    std_values=None,
                    token_pos=token_pos,
                    title=f"Example {ex_idx}: ref tok5 final top-1 probability by layer",
                    ylabel=f"P(ref_token={ref_label})",
                    out_name=f"ref_tok5_final_top1_prob_by_layer_example_{ex_idx}_tok_{token_pos}_guess.png",
                    out_dir=examples_dir,
                    show_error_bars=False,
                )

        _plot_combined_ref_overlay_tokens(
            layer_numbers=list(range(1, n_layers + 1)),
            means=np.asarray(ex_record["ref_token_curves"], dtype=np.float32),
            title=f"Example {ex_idx}: ref tok5 final top-1 probability by layer (combined)",
            ylabel=f"P(ref_token={ref_label})",
            out_name=f"ref_tok5_final_top1_prob_by_layer_example_{ex_idx}_combined_tok0_to_tok4.png",
            out_dir=examples_dir,
        )


def _aggregate_metric(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return mean, std


def _plot_metric_by_layer(
    layer_numbers: list[int],
    values: np.ndarray,
    std_values: np.ndarray | None,
    token_pos: int,
    title: str,
    ylabel: str,
    out_name: str,
    out_dir: Path,
    show_error_bars: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, values, "o-", markersize=4)
    if show_error_bars and std_values is not None:
        ax.fill_between(layer_numbers, values - std_values, values + std_values, alpha=0.2)
    ax.set_xlabel("Layer number")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (token position {token_pos})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _style_for_token_order(order_idx: int, total_tokens: int) -> dict:
    if total_tokens <= 1:
        return {"linewidth": 2.2, "alpha": 1.0}
    frac = order_idx / float(total_tokens - 1)
    return {"linewidth": 1.5 + 1.7 * frac, "alpha": 0.7 + 0.3 * frac}


def _marker_for_token_order(order_idx: int) -> str:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    return markers[order_idx % len(markers)]


def _plot_combined_tokens(
    layer_numbers: list[int],
    means: np.ndarray,
    title: str,
    ylabel: str,
    out_name: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    base_color = "tab:blue"
    for token_pos in range(NUM_GUESS_TOKENS):
        style = _style_for_token_order(token_pos, NUM_GUESS_TOKENS)
        marker = _marker_for_token_order(token_pos)
        ax.plot(
            layer_numbers,
            means[token_pos],
            marker=marker,
            markersize=4,
            label=f"Token {token_pos}",
            color=base_color,
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
    ax.set_xlabel("Layer number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_combined_ref_overlay_tokens(
    layer_numbers: list[int],
    means: np.ndarray,  # [5, L], token positions 0..4
    title: str,
    ylabel: str,
    out_name: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    base_color = "tab:blue"
    for token_pos in range(REF_TOKEN_INDEX):
        style = _style_for_token_order(token_pos, REF_TOKEN_INDEX)
        marker = _marker_for_token_order(token_pos)
        ax.plot(
            layer_numbers,
            means[token_pos],
            marker=marker,
            markersize=4,
            label=f"Token {token_pos}",
            color=base_color,
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
    ax.set_xlabel("Layer number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "logit_lens_guess_all_layers"
    base.mkdir(parents=True, exist_ok=True)
    k = 1
    while (base / str(k)).exists():
        k += 1
    run_base = base / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _write_config_txt(
    run_dir: Path,
    args: argparse.Namespace,
    input_paths: list[str],
    report_source_path: str,
    b_u_is_none: bool,
    n_layers: int,
    hidden_dim: int,
    vocab_size: int,
    n_examples: int,
    elapsed_seconds: float,
) -> None:
    lines = [
        "Logit lens guess analysis (all layers, token positions 0..5)",
        "=" * 72,
        f"Model: {args.model_name_or_path}",
        f"Device: {args.device if args.device else 'auto'}",
        f"Unembedding bias is None: {b_u_is_none}",
        f"Reference token index: {REF_TOKEN_INDEX}",
        "Reference layer: last layer",
        f"Top n tokens: {args.top_n}",
        f"Top-token report examples: {args.top_token_examples}",
        "Normalization for probability curves: per-example min-max over layers to [0,1]",
        f"Max examples per input path: {args.max_examples_per_split}",
        f"Train path: {args.train_path}",
        f"Test path: {args.test_path}",
        f"Input paths used: {', '.join(input_paths)}",
        f"Top-token report source path: {report_source_path}",
        f"Output dir: {run_dir}",
        f"Total layers: {n_layers}",
        f"Hidden dim: {hidden_dim}",
        f"Vocab size: {vocab_size}",
        f"Valid combined examples: {n_examples}",
        f"Show error bars: {args.error_bars}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
        "",
        f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(run_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run logit-lens analysis for guess token positions 0..5.")
    parser.add_argument("--train_path", type=str, required=True)
    # Two paths are accepted to mirror how data is generated and to allow larger
    # combined runs, but the underlying computation is identical for every example.
    parser.add_argument("--test_path", type=str, default=None, help="Optional second HDF5 path to combine.")
    parser.add_argument("--val_path", type=str, default=None, help="Deprecated alias for --test_path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Run dir: output_dir/logit_lens_guess_all_layers/<id>/",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
    )
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument(
        "--top_token_examples",
        type=int,
        default=3,
        help="Number of individual examples to dump in logit_lens_top_tokens.txt.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--error_bars", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--max_examples_per_split",
        type=int,
        default=None,
        help="Optional cap of examples per input path for quicker debugging/smoke tests.",
    )
    args = parser.parse_args()

    if args.test_path is None and args.val_path is not None:
        args.test_path = args.val_path

    if args.top_n <= 0:
        raise ValueError("--top_n must be >= 1")
    if args.top_token_examples <= 0:
        raise ValueError("--top_token_examples must be >= 1")

    run_base = _get_run_base_dir(Path(args.output_dir))
    figures_dir = run_base / "figures"
    similarity_metrics_dir = figures_dir / "similarity_metrics"
    examples_dir = figures_dir / "examples"
    mean_top1_prob_dir = figures_dir / "mean_top_1_prob"
    similarity_metrics_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    mean_top1_prob_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_base}")

    input_paths = [args.train_path]
    use_test_path = bool(args.test_path) and Path(args.test_path).exists()
    if args.test_path and not use_test_path:
        print(f"WARNING: test_path does not exist, falling back to train_path for report examples: {args.test_path}")
    if use_test_path:
        input_paths.append(args.test_path)

    report_source_path = args.test_path if use_test_path else args.train_path

    print("Scanning HDF5 lazily...")
    n_examples, n_layers, hidden_dim = _infer_h5_shape(
        input_paths, max_examples_per_path=args.max_examples_per_split
    )
    print(f"Input paths: {len(input_paths)} | Valid combined examples: {n_examples}")

    device = _resolve_device(args.device)
    print(f"Loading model/tokenizer on device: {device}")
    # w_u is weight of unembedding layer, b_u is bias of unembedding layer
    tokenizer, w_u, b_u, norm_weight, norm_eps = _load_unembedding(args.model_name_or_path, device)
    vocab_size, model_hidden = w_u.shape
    if hidden_dim != model_hidden:
        raise ValueError(
            f"Hidden dim mismatch between embeddings and model lm_head: "
            f"{hidden_dim} vs {model_hidden}"
        )
    if hidden_dim != int(norm_weight.shape[0]):
        raise ValueError(
            f"Hidden dim mismatch between embeddings and model norm weight: "
            f"{hidden_dim} vs {int(norm_weight.shape[0])}"
        )

    print("Analyzing combined dataset...")
    (
        cosine_values,
        kl_values,
        ce_values,
        mean_minmax_top1_by_token,
        mean_minmax_ref_top1_by_token,
        combined_example_ids,
    ) = _analyze_combined_streaming(
        input_paths=input_paths,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        w_u=w_u,
        b_u=b_u,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        device=device,
        max_examples_per_path=args.max_examples_per_split,
    )
    print(
        f"Collecting top-token report examples from: {report_source_path} "
        f"(first {args.top_token_examples} valid examples)"
    )
    report_examples = _collect_report_examples_from_path(
        source_path=report_source_path,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        w_u=w_u,
        b_u=b_u,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        device=device,
        top_n=args.top_n,
        top_token_examples=args.top_token_examples,
    )

    cosine_mean, cosine_std = _aggregate_metric(cosine_values)
    kl_mean, kl_std = _aggregate_metric(kl_values)
    ce_mean, ce_std = _aggregate_metric(ce_values)
    layer_numbers = np.arange(1, n_layers + 1, dtype=np.int32)

    np.savez(
        run_base / "logit_lens_stats.npz",
        cosine_values=cosine_values,
        kl_values=kl_values,
        ce_values=ce_values,
        cosine_mean=cosine_mean,
        cosine_std=cosine_std,
        kl_mean=kl_mean,
        kl_std=kl_std,
        ce_mean=ce_mean,
        ce_std=ce_std,
        mean_minmax_top1_by_token=mean_minmax_top1_by_token,
        mean_minmax_ref_top1_by_token=mean_minmax_ref_top1_by_token,
        layer_numbers=layer_numbers,
        combined_example_ids=np.asarray(combined_example_ids, dtype=object),
    )
    print(f"Saved {run_base / 'logit_lens_stats.npz'}")

    _write_top_tokens_report(
        out_path=run_base / "logit_lens_top_tokens.txt",
        tokenizer=tokenizer,
        top_n=args.top_n,
        top_token_examples=args.top_token_examples,
        report_examples=report_examples,
        n_layers=n_layers,
    )
    print(f"Saved {run_base / 'logit_lens_top_tokens.txt'}")

    _write_topk_tables_and_example_curves(
        examples_dir=examples_dir,
        tokenizer=tokenizer,
        report_examples=report_examples,
        n_layers=n_layers,
    )

    _write_config_txt(
        run_dir=run_base,
        args=args,
        input_paths=input_paths,
        report_source_path=report_source_path,
        b_u_is_none=(b_u is None),
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        n_examples=n_examples,
        elapsed_seconds=time.perf_counter() - start_time,
    )
    print(f"Saved {run_base / 'config.txt'}")

    if args.plot:
        layer_list = layer_numbers.tolist()
        for token_pos in range(NUM_GUESS_TOKENS):
            _plot_metric_by_layer(
                layer_numbers=layer_list,
                values=cosine_mean[token_pos],
                std_values=cosine_std[token_pos],
                token_pos=token_pos,
                title="Logit-lens cosine by layer",
                ylabel="Cosine similarity",
                out_name=f"cosine_by_layer_tok_{token_pos}_guess.png",
                out_dir=similarity_metrics_dir,
                show_error_bars=args.error_bars,
            )
            _plot_metric_by_layer(
                layer_numbers=layer_list,
                values=kl_mean[token_pos],
                std_values=kl_std[token_pos],
                token_pos=token_pos,
                title="Validation KL divergence by layer",
                ylabel="KL divergence",
                out_name=f"kl_by_layer_tok_{token_pos}_guess.png",
                out_dir=similarity_metrics_dir,
                show_error_bars=args.error_bars,
            )
            _plot_metric_by_layer(
                layer_numbers=layer_list,
                values=mean_minmax_top1_by_token[token_pos],
                std_values=None,
                token_pos=token_pos,
                title="Mean min-max normalized final top-1 token probability by layer",
                ylabel="Min-max normalized probability",
                out_name=f"mean_normalized_top1_final_token_prob_by_layer_tok_{token_pos}_guess.png",
                out_dir=mean_top1_prob_dir,
                show_error_bars=False,
            )
            if token_pos < REF_TOKEN_INDEX:
                _plot_metric_by_layer(
                    layer_numbers=layer_list,
                    values=mean_minmax_ref_top1_by_token[token_pos],
                    std_values=None,
                    token_pos=token_pos,
                    title="Mean min-max normalized ref tok5 final top-1 probability by layer",
                    ylabel="Min-max normalized probability",
                    out_name=f"mean_normalized_ref_tok5_final_top1_prob_by_layer_tok_{token_pos}_guess.png",
                    out_dir=mean_top1_prob_dir,
                    show_error_bars=False,
                )

        _plot_combined_tokens(
            layer_numbers=layer_list,
            means=cosine_mean,
            title="Logit-lens cosine across token positions (combined)",
            ylabel="Cosine similarity",
            out_name="cosine_combined_all_tokens.png",
            out_dir=similarity_metrics_dir,
        )
        _plot_combined_tokens(
            layer_numbers=layer_list,
            means=kl_mean,
            title="Validation KL divergence across token positions (combined)",
            ylabel="KL divergence",
            out_name="kl_combined_all_tokens.png",
            out_dir=similarity_metrics_dir,
        )
        _plot_combined_tokens(
            layer_numbers=layer_list,
            means=mean_minmax_top1_by_token,
            title="Mean min-max normalized final top-1 probability across token positions (combined)",
            ylabel="Min-max normalized probability",
            out_name="mean_normalized_top1_final_token_prob_combined_all_tokens.png",
            out_dir=mean_top1_prob_dir,
        )
        _plot_combined_ref_overlay_tokens(
            layer_numbers=layer_list,
            means=mean_minmax_ref_top1_by_token,
            title="Mean min-max normalized ref tok5 final top-1 probability across token positions 0..4",
            ylabel="Min-max normalized probability",
            out_name="mean_normalized_ref_tok5_final_top1_prob_combined_tok0_to_tok4.png",
            out_dir=mean_top1_prob_dir,
        )

    print(f"Done. Outputs under {run_base}")


if __name__ == "__main__":
    main()
