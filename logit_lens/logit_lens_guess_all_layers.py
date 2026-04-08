"""
Logit-lens analysis for guess token positions 0..4 across all layers.

For each hidden state (token_pos, layer_idx), this script applies:
    logits = hidden @ W_U^T + b
    probs = softmax(logits)

Then it:
- records top-n tokens from mean probabilities for each split/token/layer
- computes cosine similarity between probs(token_pos, layer_idx) and
  probs(token=4, last_layer) per example
- saves per-token and combined cosine plots (train/validation)
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REF_TOKEN_INDEX = 5
NUM_GUESS_TOKENS = 6
EPS = 1e-12


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _load_pickle_batches_guess_only(path: str) -> dict:
    data = {}
    with open(path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
            except EOFError:
                break
            if not isinstance(batch, dict):
                print(f"WARNING: Skipping non-dict batch: {type(batch)}")
                continue
            for example_id, example_data in batch.items():
                responses = example_data.get("responses", [])
                if not responses:
                    continue
                emb_guess = responses[0].get("embeddings_guess")
                if emb_guess is None:
                    continue
                data[example_id] = {"responses": [{"embeddings_guess": emb_guess}]}
    return data


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


def _collect_examples(data: dict, max_examples: int | None = None) -> tuple[list[np.ndarray], int, int]:
    examples: list[np.ndarray] = []
    n_layers = None
    hidden_dim = None
    for _eid, example_data in data.items():
        responses = example_data.get("responses", [])
        if len(responses) != 1:
            continue
        emb_guess = responses[0].get("embeddings_guess")
        if emb_guess is None:
            continue
        stacks = _example_stacks(emb_guess)
        if stacks is None:
            continue
        ex = np.stack(stacks, axis=0).astype(np.float32, copy=False)  # (5, n_layers, hidden_dim)
        if n_layers is None:
            n_layers = ex.shape[1]
            hidden_dim = ex.shape[2]
        if ex.shape[1] != n_layers or ex.shape[2] != hidden_dim:
            continue
        examples.append(ex)
        if max_examples is not None and len(examples) >= max_examples:
            break
    if not examples or n_layers is None or hidden_dim is None:
        raise ValueError("No valid examples found in data.")
    return examples, n_layers, hidden_dim


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
    unembed_w = model.lm_head.weight.detach().to(device=device, dtype=torch.float32)  # (vocab, hidden)
    unembed_b = getattr(model.lm_head, "bias", None)
    if unembed_b is not None:
        unembed_b = unembed_b.detach().to(device=device, dtype=torch.float32)
    return tokenizer, unembed_w, unembed_b


def _probs_from_hidden(hidden: torch.Tensor, w_u: torch.Tensor, b_u: torch.Tensor | None) -> torch.Tensor:
    logits = hidden @ w_u.T
    if b_u is not None:
        logits = logits + b_u
    logits = logits - torch.max(logits)
    probs = torch.softmax(logits, dim=-1)
    return probs


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    return float(np.dot(a, b) / max(an * bn, EPS))


def _analyze_split(
    examples: list[np.ndarray],
    n_layers: int,
    vocab_size: int,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        cosine_values: (n_examples, 5, n_layers)
        mean_probs: (5, n_layers, vocab_size)
        std_probs: (5, n_layers, vocab_size)
    """
    cosine_values = np.zeros((len(examples), NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
    sum_probs = np.zeros((NUM_GUESS_TOKENS, n_layers, vocab_size), dtype=np.float64)
    sumsq_probs = np.zeros((NUM_GUESS_TOKENS, n_layers, vocab_size), dtype=np.float64)

    with torch.no_grad():
        for ex_idx, ex in enumerate(examples):
            ref_hidden_np = ex[REF_TOKEN_INDEX, n_layers - 1, :]
            ref_hidden = torch.from_numpy(ref_hidden_np).to(device=device, dtype=torch.float32)
            ref_probs_t = _probs_from_hidden(ref_hidden, w_u, b_u)
            ref_probs = ref_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)

            for token_pos in range(NUM_GUESS_TOKENS):
                for layer_idx in range(n_layers):
                    hidden_np = ex[token_pos, layer_idx, :]
                    hidden = torch.from_numpy(hidden_np).to(device=device, dtype=torch.float32)
                    probs_t = _probs_from_hidden(hidden, w_u, b_u)
                    probs = probs_t.detach().cpu().numpy().astype(np.float32, copy=False)

                    cosine_values[ex_idx, token_pos, layer_idx] = _cosine_similarity(probs, ref_probs)
                    sum_probs[token_pos, layer_idx] += probs
                    sumsq_probs[token_pos, layer_idx] += probs * probs

    n = float(len(examples))
    mean_probs = (sum_probs / n).astype(np.float32)
    var_probs = np.maximum((sumsq_probs / n) - (mean_probs.astype(np.float64) ** 2), 0.0)
    std_probs = np.sqrt(var_probs).astype(np.float32)
    return cosine_values, mean_probs, std_probs


def _topn_for_distribution(probs: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(-probs, top_n - 1)[:top_n]
    sorted_local = idx[np.argsort(-probs[idx])]
    return sorted_local, probs[sorted_local]


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
    train_mean_probs: np.ndarray,
    val_mean_probs: np.ndarray,
) -> None:
    n_layers = train_mean_probs.shape[1]
    lines: list[str] = []
    lines.append("LOGIT LENS TOP TOKENS REPORT")
    lines.append("=" * 80)
    lines.append(f"top_n: {top_n}")
    lines.append(f"token_positions: 0..{NUM_GUESS_TOKENS - 1}")
    lines.append(f"layers: 1..{n_layers}")
    lines.append("")

    ref_probs = train_mean_probs[REF_TOKEN_INDEX, n_layers - 1]
    ref_ids, ref_vals = _topn_for_distribution(ref_probs, top_n=1)
    lines.append("REFERENCE (EMPHASIZED)")
    lines.append("-" * 80)
    lines.append("Reference = train mean probs at token_pos=4, last_layer")
    lines.append(f"*** MOST LIKELY TOKEN: {_format_token(tokenizer, int(ref_ids[0]))} prob={float(ref_vals[0]):.8f} ***")
    lines.append("")

    for split_name, mean_probs in [("train", train_mean_probs), ("validation", val_mean_probs)]:
        lines.append(f"SPLIT: {split_name.upper()}")
        lines.append("-" * 80)
        for token_pos in range(NUM_GUESS_TOKENS):
            lines.append(f"[tok_{token_pos}_guess]")
            for layer_idx in range(n_layers):
                probs = mean_probs[token_pos, layer_idx]
                top_ids, top_vals = _topn_for_distribution(probs, top_n=top_n)
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
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    token_pos: int,
    out_dir: Path,
    show_error_bars: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, train_mean, "o-", label="Train", markersize=4)
    ax.plot(layer_numbers, val_mean, "s-", label="Validation", markersize=4)
    if show_error_bars:
        ax.fill_between(layer_numbers, train_mean - train_std, train_mean + train_std, alpha=0.2)
        ax.fill_between(layer_numbers, val_mean - val_std, val_mean + val_std, alpha=0.2)
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


def _plot_combined_tokens(layer_numbers: list[int], means: np.ndarray, split_name: str, out_dir: Path) -> None:
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
    ax.set_title(f"Logit-lens cosine across token positions ({split_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"cosine_combined_all_tokens_{split_name.lower()}.png"
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
    n_layers: int,
    hidden_dim: int,
    vocab_size: int,
    n_train_examples: int,
    n_val_examples: int,
    elapsed_seconds: float,
) -> None:
    lines = [
        "Logit lens guess analysis (all layers, token positions 0..4)",
        "=" * 72,
        f"Model: {args.model_name_or_path}",
        f"Device: {args.device if args.device else 'auto'}",
        f"Reference token index: {REF_TOKEN_INDEX}",
        "Reference layer: last layer",
        f"Top n tokens: {args.top_n}",
        f"Max examples per split: {args.max_examples_per_split}",
        f"Train path: {args.train_path}",
        f"Val path: {args.val_path}",
        f"Output dir: {run_dir}",
        f"Total layers: {n_layers}",
        f"Hidden dim: {hidden_dim}",
        f"Vocab size: {vocab_size}",
        f"Valid train examples: {n_train_examples}",
        f"Valid val examples: {n_val_examples}",
        f"Show error bars: {args.error_bars}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
        "",
        f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(run_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run logit-lens analysis for guess token positions 0..4.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
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
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--error_bars", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--max_examples_per_split",
        type=int,
        default=None,
        help="Optional cap of examples per split for quicker debugging/smoke tests.",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top_n must be >= 1")

    run_base = _get_run_base_dir(Path(args.output_dir))
    print(f"Run directory: {run_base}")

    print("Loading pickles...")
    train_data = _load_pickle_batches_guess_only(args.train_path)
    val_data = _load_pickle_batches_guess_only(args.val_path)
    print(f"Raw train examples: {len(train_data)}, raw val examples: {len(val_data)}")

    train_examples, n_layers_tr, hidden_dim_tr = _collect_examples(
        train_data, max_examples=args.max_examples_per_split
    )
    val_examples, n_layers_va, hidden_dim_va = _collect_examples(
        val_data, max_examples=args.max_examples_per_split
    )
    if n_layers_tr != n_layers_va or hidden_dim_tr != hidden_dim_va:
        raise ValueError("Mismatch between train and val layer/hidden dimensions.")
    n_layers = n_layers_tr
    hidden_dim = hidden_dim_tr

    device = _resolve_device(args.device)
    print(f"Loading model/tokenizer on device: {device}")
    tokenizer, w_u, b_u = _load_unembedding(args.model_name_or_path, device)
    vocab_size, model_hidden = w_u.shape
    if hidden_dim != model_hidden:
        raise ValueError(
            f"Hidden dim mismatch between embeddings and model lm_head: "
            f"{hidden_dim} vs {model_hidden}"
        )

    print("Analyzing train split...")
    train_cos, train_mean_probs, train_std_probs = _analyze_split(
        examples=train_examples,
        n_layers=n_layers,
        vocab_size=vocab_size,
        w_u=w_u,
        b_u=b_u,
        device=device,
    )
    print("Analyzing validation split...")
    val_cos, val_mean_probs, val_std_probs = _analyze_split(
        examples=val_examples,
        n_layers=n_layers,
        vocab_size=vocab_size,
        w_u=w_u,
        b_u=b_u,
        device=device,
    )

    train_cos_mean, train_cos_std = _aggregate_cosine(train_cos)
    val_cos_mean, val_cos_std = _aggregate_cosine(val_cos)
    layer_numbers = np.arange(1, n_layers + 1, dtype=np.int32)

    np.savez(
        run_base / "logit_lens_stats.npz",
        train_cos=train_cos,
        val_cos=val_cos,
        train_cos_mean=train_cos_mean,
        train_cos_std=train_cos_std,
        val_cos_mean=val_cos_mean,
        val_cos_std=val_cos_std,
        train_mean_probs=train_mean_probs,
        train_std_probs=train_std_probs,
        val_mean_probs=val_mean_probs,
        val_std_probs=val_std_probs,
        layer_numbers=layer_numbers,
    )
    print(f"Saved {run_base / 'logit_lens_stats.npz'}")

    _write_top_tokens_report(
        out_path=run_base / "logit_lens_top_tokens.txt",
        tokenizer=tokenizer,
        top_n=args.top_n,
        train_mean_probs=train_mean_probs,
        val_mean_probs=val_mean_probs,
    )
    print(f"Saved {run_base / 'logit_lens_top_tokens.txt'}")

    _write_config_txt(
        run_dir=run_base,
        args=args,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        n_train_examples=len(train_examples),
        n_val_examples=len(val_examples),
        elapsed_seconds=time.perf_counter() - start_time,
    )
    print(f"Saved {run_base / 'config.txt'}")

    if args.plot:
        layer_list = layer_numbers.tolist()
        for token_pos in range(NUM_GUESS_TOKENS):
            _plot_token_similarity_by_layer(
                layer_numbers=layer_list,
                train_mean=train_cos_mean[token_pos],
                train_std=train_cos_std[token_pos],
                val_mean=val_cos_mean[token_pos],
                val_std=val_cos_std[token_pos],
                token_pos=token_pos,
                out_dir=run_base,
                show_error_bars=args.error_bars,
            )
        _plot_combined_tokens(layer_list, train_cos_mean, "train", run_base)
        _plot_combined_tokens(layer_list, val_cos_mean, "validation", run_base)

    print(f"Done. Outputs under {run_base}")


if __name__ == "__main__":
    main()
