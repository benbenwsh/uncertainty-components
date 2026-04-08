"""
Logit-lens analysis for guess token positions 0..5 across all layers.

For each hidden state (token_pos, layer_idx), this script applies:
    logits = hidden @ W_U^T + b
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
    unembed_w = model.lm_head.weight.detach().to(device=device, dtype=torch.float32)
    unembed_b = getattr(model.lm_head, "bias", None)
    if unembed_b is not None:
        unembed_b = unembed_b.detach().to(device=device, dtype=torch.float32)
    return tokenizer, unembed_w, unembed_b


def _probs_from_hidden(hidden: torch.Tensor, w_u: torch.Tensor, b_u: torch.Tensor | None) -> torch.Tensor:
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


def _analyze_combined_streaming(
    input_paths: list[str],
    n_layers: int,
    hidden_dim: int,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    device: torch.device,
    top_n: int,
    top_token_examples: int,
    max_examples_per_path: int | None = None,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Returns:
        cosine_values: (n_examples, n_tokens, n_layers)
        combined_example_ids: list[str], one per example in cosine_values
        report_examples: first top_token_examples examples with per-layer top tokens
    """
    cosine_rows: list[np.ndarray] = []
    combined_example_ids: list[str] = []
    report_examples: list[dict] = []
    n_examples = 0

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
            ref_probs_t = _probs_from_hidden(ref_hidden, w_u, b_u)
            ref_probs = ref_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)

            ex_cos = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
            top_ids_record = np.zeros((NUM_GUESS_TOKENS, n_layers, top_n), dtype=np.int32)
            top_vals_record = np.zeros((NUM_GUESS_TOKENS, n_layers, top_n), dtype=np.float32)
            for token_pos in range(NUM_GUESS_TOKENS):
                for layer_idx in range(n_layers):
                    hidden_np = ex[token_pos, layer_idx, :]
                    hidden = torch.from_numpy(hidden_np).to(device=device, dtype=torch.float32)
                    probs_t = _probs_from_hidden(hidden, w_u, b_u)
                    probs = probs_t.detach().cpu().numpy().astype(np.float32, copy=False)

                    ex_cos[token_pos, layer_idx] = _cosine_similarity(probs, ref_probs)
                    top_ids, top_vals = _topn_for_distribution(probs, top_n=top_n)
                    top_ids_record[token_pos, layer_idx] = top_ids
                    top_vals_record[token_pos, layer_idx] = top_vals
            cosine_rows.append(ex_cos)
            combined_example_ids.append(combined_example_id)
            if len(report_examples) < top_token_examples:
                report_examples.append(
                    {
                        "combined_example_id": combined_example_id,
                        "source_path": source_path,
                        "example_id": example_id,
                        "top_ids": top_ids_record.copy(),
                        "top_vals": top_vals_record.copy(),
                    }
                )
            n_examples += 1

    if n_examples == 0:
        joined_paths = ", ".join(input_paths)
        raise ValueError(f"No valid examples found in input path(s): {joined_paths}")

    cosine_values = np.stack(cosine_rows, axis=0)
    return cosine_values, combined_example_ids, report_examples


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


def _format_token(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        token = tokenizer.decode([int(token_id)])
    decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return f"id={int(token_id)} token={repr(token)} decoded={repr(decoded)}"


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


def _aggregate_cosine(cosine_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(cosine_values, axis=0)
    std = np.std(cosine_values, axis=0)
    return mean, std


def _plot_token_similarity_by_layer(
    layer_numbers: list[int],
    cosine_mean: np.ndarray,
    cosine_std: np.ndarray,
    token_pos: int,
    out_dir: Path,
    show_error_bars: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, cosine_mean, "o-", label="Combined", markersize=4)
    if show_error_bars:
        ax.fill_between(layer_numbers, cosine_mean - cosine_std, cosine_mean + cosine_std, alpha=0.2)
    ax.set_xlabel("Layer number")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Logit-lens cosine by layer (token position {token_pos})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"cosine_by_layer_tok_{token_pos}_guess.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_combined_tokens(layer_numbers: list[int], means: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_GUESS_TOKENS))
    for token_pos in range(NUM_GUESS_TOKENS):
        ax.plot(
            layer_numbers,
            means[token_pos],
            marker="o",
            markersize=4,
            label=f"Token {token_pos}",
            color=colors[token_pos],
        )
    ax.set_xlabel("Layer number")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Logit-lens cosine across token positions (combined)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "cosine_combined_all_tokens.png"
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
        f"Max examples per input path: {args.max_examples_per_split}",
        f"Train path: {args.train_path}",
        f"Val path: {args.val_path}",
        f"Input paths used: {', '.join(input_paths)}",
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
    parser.add_argument("--val_path", type=str, default=None, help="Optional second HDF5 path to combine.")
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
    parser.add_argument("--top_n", type=int, default=5)
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

    if args.top_n <= 0:
        raise ValueError("--top_n must be >= 1")
    if args.top_token_examples <= 0:
        raise ValueError("--top_token_examples must be >= 1")

    run_base = _get_run_base_dir(Path(args.output_dir))
    print(f"Run directory: {run_base}")

    input_paths = [args.train_path]
    if args.val_path:
        input_paths.append(args.val_path)

    print("Scanning HDF5 lazily...")
    n_examples, n_layers, hidden_dim = _infer_h5_shape(
        input_paths, max_examples_per_path=args.max_examples_per_split
    )
    print(f"Input paths: {len(input_paths)} | Valid combined examples: {n_examples}")

    device = _resolve_device(args.device)
    print(f"Loading model/tokenizer on device: {device}")
    # w_u is weight of unembedding layer, b_u is bias of unembedding layer
    tokenizer, w_u, b_u = _load_unembedding(args.model_name_or_path, device)
    vocab_size, model_hidden = w_u.shape
    if hidden_dim != model_hidden:
        raise ValueError(
            f"Hidden dim mismatch between embeddings and model lm_head: "
            f"{hidden_dim} vs {model_hidden}"
        )

    print("Analyzing combined dataset...")
    cosine_values, combined_example_ids, report_examples = _analyze_combined_streaming(
        input_paths=input_paths,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        w_u=w_u,
        b_u=b_u,
        device=device,
        top_n=args.top_n,
        top_token_examples=args.top_token_examples,
        max_examples_per_path=args.max_examples_per_split,
    )

    cosine_mean, cosine_std = _aggregate_cosine(cosine_values)
    layer_numbers = np.arange(1, n_layers + 1, dtype=np.int32)

    np.savez(
        run_base / "logit_lens_stats.npz",
        cosine_values=cosine_values,
        cosine_mean=cosine_mean,
        cosine_std=cosine_std,
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

    _write_config_txt(
        run_dir=run_base,
        args=args,
        input_paths=input_paths,
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
            _plot_token_similarity_by_layer(
                layer_numbers=layer_list,
                cosine_mean=cosine_mean[token_pos],
                cosine_std=cosine_std[token_pos],
                token_pos=token_pos,
                out_dir=run_base,
                show_error_bars=args.error_bars,
            )
        _plot_combined_tokens(layer_list, cosine_mean, run_base)

    print(f"Done. Outputs under {run_base}")


if __name__ == "__main__":
    main()
