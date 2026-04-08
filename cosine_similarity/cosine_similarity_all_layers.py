"""
Compute cosine similarity across all layers for guess token positions 0..4.

For each example, this script compares:
- query embedding: embeddings_guess[token_pos][layer_idx]
- reference embedding: embeddings_guess[4][last_layer]

It aggregates mean/std cosine similarity over examples, then saves:
- one per-token-position graph (single curve vs layer)
- one combined graph overlaying token positions 0..4
- npz artifact with numeric arrays
- config.txt run metadata
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REF_TOKEN_INDEX = 5
NUM_GUESS_TOKENS = 6  # token positions 0..5 inclusive
EPS = 1e-12


def _tensor_to_numpy(obj):
    """Convert tensor or array-like to numpy."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _load_pickle_batches_guess_only(path: str) -> dict:
    """Load multi-dump pickle; keep examples with embeddings_guess only."""
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
    """Single token slice -> (n_layers, hidden_dim)."""
    arr = _tensor_to_numpy(emb)
    return arr[:, 0, -1, :]


def _example_stacks(emb_guess: list) -> list[np.ndarray] | None:
    """Return list of (n_layers, hidden_dim) for tokens 0..4, or None if invalid."""
    if len(emb_guess) != NUM_GUESS_TOKENS:
        return None

    stacks = []
    for token_pos in range(NUM_GUESS_TOKENS):
        stacks.append(_stack_guess_token(emb_guess[token_pos]))

    n_layers = stacks[0].shape[0]
    hidden_dim = stacks[0].shape[1]
    for s in stacks:
        if s.shape != (n_layers, hidden_dim):
            return None
    return stacks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    denom = max(a_norm * b_norm, EPS)
    return float(np.dot(a, b) / denom)


def _collect_example_cosines(data: dict) -> tuple[np.ndarray, int, int]:
    """
    Collect cosine similarities per example.

    Returns:
        cos_array: shape (n_examples, NUM_GUESS_TOKENS, n_layers)
        n_layers: number of layers
        n_examples: number of valid examples
    """
    per_example = []
    n_layers = None

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

        if n_layers is None:
            n_layers = stacks[0].shape[0]

        ref = stacks[REF_TOKEN_INDEX][n_layers - 1, :].astype(np.float32, copy=False)

        token_layer_cos = np.zeros((NUM_GUESS_TOKENS, n_layers), dtype=np.float32)
        for token_pos in range(NUM_GUESS_TOKENS):
            for layer_idx in range(n_layers):
                h = stacks[token_pos][layer_idx, :].astype(np.float32, copy=False)
                token_layer_cos[token_pos, layer_idx] = _cosine_similarity(h, ref)

        per_example.append(token_layer_cos)

    if not per_example or n_layers is None:
        raise ValueError("No valid examples found for cosine similarity computation.")

    cos_array = np.stack(per_example, axis=0)
    return cos_array, n_layers, cos_array.shape[0]


def _aggregate_stats(cos_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate mean/std over examples. Input shape: (n_examples, n_tokens, n_layers)."""
    mean = np.mean(cos_array, axis=0)
    std = np.std(cos_array, axis=0)
    return mean, std


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "cosine_similarity_all_layers"
    base.mkdir(parents=True, exist_ok=True)
    k = 1
    while (base / str(k)).exists():
        k += 1
    run_base = base / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _plot_token_similarity_by_layer(
    layer_numbers: list[int],
    mean_vals: np.ndarray,
    std_vals: np.ndarray,
    token_pos: int,
    out_dir: Path,
    show_error_bars: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, mean_vals, "o-", markersize=4)

    if show_error_bars:
        ax.fill_between(layer_numbers, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

    ax.set_xlabel("Layer number")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Cosine similarity by layer (token position {token_pos})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"cosine_by_layer_tok_{token_pos}_guess.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_combined_tokens(
    layer_numbers: list[int],
    means: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Combined plot for one split.

    means shape: (NUM_GUESS_TOKENS, n_layers)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    base_color = np.array(mcolors.to_rgb("#1f77b4"), dtype=np.float32)
    light_color = np.ones(3, dtype=np.float32)
    for token_pos in range(NUM_GUESS_TOKENS):
        # Ordinal encoding: earlier tokens are lighter/thinner, later tokens darker/thicker.
        t = token_pos / max(NUM_GUESS_TOKENS - 1, 1)
        blend = 0.72 * (1.0 - t)
        line_color = base_color * (1.0 - blend) + light_color * blend
        linewidth = 1.5 + 1.1 * t
        ax.plot(
            layer_numbers,
            means[token_pos],
            marker="o",
            markersize=4,
            label=f"Token {token_pos}",
            color=line_color,
            linewidth=linewidth,
        )

    ax.set_xlabel("Layer number")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity by layer across token positions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "cosine_combined_all_tokens.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _write_config_txt(
    run_dir: Path,
    args: argparse.Namespace,
    n_layers: int,
    n_combined_examples: int,
) -> None:
    lines = [
        "Cosine similarity analysis (all layers, guess token positions 0..4)",
        "=" * 72,
        f"Reference token index: {REF_TOKEN_INDEX} (last guess token)",
        "Reference layer: last layer",
        f"Token positions analyzed: 0..{NUM_GUESS_TOKENS - 1}",
        f"Train path: {args.train_path}",
        f"Val path: {args.val_path}",
        f"Output dir: {run_dir}",
        f"Total layers: {n_layers}",
        f"Valid combined examples: {n_combined_examples}",
        f"Show error bars: {args.error_bars}",
        "",
        f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(run_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity between embeddings_guess[token_pos][layer] and "
            "embeddings_guess[4][last_layer], then plot by layer."
        )
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Run dir: output_dir/cosine_similarity_all_layers/<id>/",
    )
    parser.add_argument("--plot", action="store_true", help="Save per-token and combined plots")
    parser.add_argument(
        "--error_bars",
        action="store_true",
        help="Show +/-1 std shaded regions on per-token plots",
    )
    args = parser.parse_args()

    run_base = _get_run_base_dir(Path(args.output_dir))
    print(f"Run directory: {run_base}")

    print("Loading pickles...")
    train_data = _load_pickle_batches_guess_only(args.train_path)
    val_data = _load_pickle_batches_guess_only(args.val_path)
    print(f"Raw train examples: {len(train_data)}, raw val examples: {len(val_data)}")
    combined_data = {}
    combined_data.update(train_data)
    for example_id, example_data in val_data.items():
        key = example_id if example_id not in combined_data else f"val::{example_id}"
        combined_data[key] = example_data

    combined_cos, n_layers, n_combined_examples = _collect_example_cosines(combined_data)
    layer_numbers = list(range(1, n_layers + 1))
    combined_mean, combined_std = _aggregate_stats(combined_cos)

    np.savez(
        run_base / "cosine_similarity_stats.npz",
        combined_cos=combined_cos,
        combined_mean=combined_mean,
        combined_std=combined_std,
        layer_numbers=np.asarray(layer_numbers, dtype=np.int32),
    )
    print(f"Saved {run_base / 'cosine_similarity_stats.npz'}")

    _write_config_txt(run_base, args, n_layers, n_combined_examples)
    print(f"Saved {run_base / 'config.txt'}")

    if args.plot:
        for token_pos in range(NUM_GUESS_TOKENS):
            _plot_token_similarity_by_layer(
                layer_numbers=layer_numbers,
                mean_vals=combined_mean[token_pos],
                std_vals=combined_std[token_pos],
                token_pos=token_pos,
                out_dir=run_base,
                show_error_bars=args.error_bars,
            )
        _plot_combined_tokens(layer_numbers, combined_mean, run_base)

    print(f"Done. Outputs under {run_base}")


if __name__ == "__main__":
    main()
