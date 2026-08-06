#!/usr/bin/env python3
"""Probability-span mass-mean direction analysis (H5-only).

Computes layer x token mass-mean directions from a verbalised-embeddings H5,
writes per-(layer, token) pickles, and emits:
  - magnitude heatmap over all TL layers
  - consecutive-layer cosine-similarity heatmap (layers 1..n_layers-1)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layerwise_mean_ablation.run_mean_ablation import load_examples_h5
from mass_mean_probe.run_mass_mean_probe import (
    _as_layer_hidden,
    _expected_probability_span_token_budget,
    _extract_res_field,
    _prefix_tokens_for_linguistic_confidence,
    compute_low_high_span_means_and_directions,
    configure_prefix_tokens_for_model,
)


MODULE_NAME = "direction_analysis"
RESULTS_STEM = "mass_mean_directions"


def _resolve_run_root(cli_output_dir: Optional[str]) -> str:
    if cli_output_dir:
        os.makedirs(cli_output_dir, exist_ok=True)
        return cli_output_dir

    base_dir = os.path.join(MODULE_NAME, "results", RESULTS_STEM)
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_root = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_root, exist_ok=True)
    return run_root


def _attach_output_log(run_root: str) -> str:
    output_log_path = os.path.join(run_root, "output.log")
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    return output_log_path


def _render_token_label(token: str) -> str:
    escaped = token.encode("unicode_escape").decode("ascii")
    return escaped if escaped else "<empty>"


def _infer_n_layers(examples_h5: Dict[str, dict], *, new_h5_format: bool) -> int:
    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or not responses:
            continue
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            continue
        emb_prob = _extract_res_field(
            resp0, ex_id, "embeddings_probability", new_h5_format=new_h5_format
        )
        if emb_prob is None:
            continue
        if isinstance(emb_prob, list):
            if not emb_prob:
                continue
            layer_hidden = _as_layer_hidden(emb_prob[0])
        else:
            layer_hidden = _as_layer_hidden(emb_prob)
        n_stream = int(layer_hidden.shape[0])
        if n_stream < 2:
            raise ValueError(
                f"Example {ex_id} residual stream length {n_stream} is too short "
                "(need embedding + at least one resid_post)."
            )
        return n_stream - 1
    raise ValueError("Could not infer n_layers: no usable embeddings_probability found in H5.")


def _token_labels_from_prefix(
    *,
    span_token_count: int,
    linguistic_confidence_prompt: bool,
) -> List[str]:
    prefix_alts = _prefix_tokens_for_linguistic_confidence(linguistic_confidence_prompt)
    labels: List[str] = []
    for i in range(span_token_count):
        if i < len(prefix_alts) and prefix_alts[i]:
            labels.append(str(prefix_alts[i][0]))
        else:
            labels.append(f"value_{i - len(prefix_alts)}")
    return labels


def _minmax_alpha(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    alpha = (values - vmin) / (vmax - vmin)
    return np.nan_to_num(alpha.astype(np.float32), nan=0.0)


def write_rgba_heatmap(
    *,
    path: str,
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    matrix_values: np.ndarray,
    title: str,
    xlabel: str,
    value_fmt: str = "{:.3f}",
) -> Tuple[float, float]:
    """Blue RGBA heatmap with min-max alpha. Returns (vmin, vmax) of finite values."""
    n_rows, n_cols = matrix_values.shape
    finite = matrix_values[np.isfinite(matrix_values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 0.0
    alpha_grid = _minmax_alpha(matrix_values)

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.1
    rgba[:, :, 1] = 0.3
    rgba[:, :, 2] = 1.0
    rgba[:, :, 3] = alpha_grid

    fig, ax = plt.subplots(figsize=(max(9.0, 1.4 * n_cols), max(6.0, 0.35 * n_rows)))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix_values[r, c]
            text = "NA" if not np.isfinite(val) else value_fmt.format(float(val))
            ax.text(c, r, text, ha="center", va="center", fontsize=8, color="black")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(layer) for layer in run_layers])
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [f"{i}:{_render_token_label(tok)}" for i, tok in enumerate(token_labels)],
        rotation=30,
        ha="left",
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Layer")
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.35, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


def _cosine_similarity_rows(direction: np.ndarray) -> np.ndarray:
    """Cosine between layer L and L-1 for L=1..n-1. Shape (n_layers-1, T)."""
    n_layers, n_tokens, _ = direction.shape
    if n_layers < 2:
        raise ValueError("Need at least 2 layers to compute consecutive-layer cosine similarity.")
    out = np.full((n_layers - 1, n_tokens), np.nan, dtype=np.float32)
    for layer_idx in range(1, n_layers):
        for tok_idx in range(n_tokens):
            a = direction[layer_idx, tok_idx]
            b = direction[layer_idx - 1, tok_idx]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 0.0:
                out[layer_idx - 1, tok_idx] = np.nan
            else:
                out[layer_idx - 1, tok_idx] = float(np.dot(a, b) / denom)
    return out


def write_direction_pickles(
    run_root: str,
    direction_probability: np.ndarray,
) -> None:
    n_layers, n_tokens, _ = direction_probability.shape
    for layer_idx in range(n_layers):
        layer_dir = os.path.join(run_root, str(layer_idx))
        os.makedirs(layer_dir, exist_ok=True)
        for tok_idx in range(n_tokens):
            payload = {
                "layer": layer_idx,
                "token": tok_idx,
                "direction": np.asarray(
                    direction_probability[layer_idx, tok_idx], dtype=np.float32
                ),
            }
            path = os.path.join(layer_dir, f"{tok_idx}.pkl")
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    n_layers: int,
    span_token_count: int,
    direction_probability_shape: Tuple[int, ...],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    magnitude_range: Tuple[float, float],
    cosine_range: Tuple[float, float],
    finished_at: str,
) -> None:
    lines = [
        "Direction Analysis Config",
        "=========================",
        "",
        "[Run]",
        f"finished_at={finished_at}",
        f"model_name={args.model_name}",
        f"input_h5={args.input_h5}",
        f"new_h5_format={args.new_h5_format}",
        f"n_layers={n_layers}",
        f"direction_probability_shape={direction_probability_shape}",
        "",
        "[Confidence groups]",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_count={low_conf_count}",
        f"high_conf_count={high_conf_count}",
        f"h5_example_count={h5_example_count}",
        "",
        "[Span]",
        f"linguistic_confidence_prompt={args.linguistic_confidence_prompt}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_confidence_tokens={args.expected_confidence_tokens}",
        f"extend_probability_span={args.extend_probability_span}",
        f"span_token_count={span_token_count}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        "",
        "[Heatmaps]",
        f"magnitude_min={magnitude_range[0]}",
        f"magnitude_max={magnitude_range[1]}",
        f"cosine_min={cosine_range[0]}",
        f"cosine_max={cosine_range[1]}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute probability-span mass-mean directions from an H5 file, "
            "write per-(layer, token) pickles, and emit magnitude / cosine heatmaps."
        )
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, read 'res' from the new {res,attn,mlp} H5 response format.",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument(
        "--expected_confidence_tokens",
        type=int,
        default=5,
        help=(
            "When --linguistic_confidence_prompt, expected Confidence: span token count "
            "(instead of --expected_probability_tokens)."
        ),
    )
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--extend_probability_span",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true (and not using linguistic confidence), treat probability span length as "
            "expected_probability_tokens + 2."
        ),
    )
    parser.add_argument(
        "--linguistic_confidence_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use Confidence: span budget; if false, use Probability: span budget.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            f"Optional run directory. If unset, auto-creates under "
            f"{MODULE_NAME}/results/{RESULTS_STEM}/."
        ),
    )
    args = parser.parse_args()
    configure_prefix_tokens_for_model(args.model_name)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_root = _resolve_run_root(args.output_dir)
    _attach_output_log(run_root)
    logging.info("Saving outputs to %s", run_root)

    span_token_count = _expected_probability_span_token_budget(
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_confidence_tokens=args.expected_confidence_tokens,
        extend_probability_span=args.extend_probability_span,
    )
    direction_prob_token_budget = (
        args.expected_probability_tokens
        if args.linguistic_confidence_prompt
        else span_token_count
    )

    logging.info("Loading H5: %s", args.input_h5)
    examples_h5 = load_examples_h5(Path(args.input_h5))
    n_layers = _infer_n_layers(examples_h5, new_h5_format=args.new_h5_format)
    run_layers = list(range(n_layers))
    logging.info("Inferred n_layers=%d from residual stream", n_layers)

    _mean_low, _mean_high, direction, low_ids, high_ids = compute_low_high_span_means_and_directions(
        examples_h5,
        run_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        expected_probability_tokens=direction_prob_token_budget,
        expected_guess_tokens=args.expected_guess_tokens,
        new_h5_format=args.new_h5_format,
    )
    direction_probability = np.asarray(direction["probability"], dtype=np.float32)
    if direction_probability.ndim != 3:
        raise ValueError(
            f"Expected direction['probability'] shape (layers, tokens, d_model), "
            f"got {direction_probability.shape}."
        )
    if direction_probability.shape[0] != n_layers:
        raise ValueError(
            f"Direction layer count {direction_probability.shape[0]} != inferred n_layers {n_layers}."
        )
    if direction_probability.shape[1] != direction_prob_token_budget:
        raise ValueError(
            f"Direction token count {direction_probability.shape[1]} != "
            f"budget {direction_prob_token_budget}."
        )

    n_tokens = int(direction_probability.shape[1])
    token_labels = _token_labels_from_prefix(
        span_token_count=n_tokens,
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
    )
    xlabel = (
        "Confidence token position"
        if args.linguistic_confidence_prompt
        else "Probability token position"
    )

    logging.info("Writing direction pickles under %s/{{layer}}/{{token}}.pkl", run_root)
    write_direction_pickles(run_root, direction_probability)

    magnitudes = np.linalg.norm(direction_probability, axis=-1).astype(np.float32)
    mag_path = os.path.join(run_root, "magnitude_heatmap.png")
    mag_vmin, mag_vmax = write_rgba_heatmap(
        path=mag_path,
        run_layers=run_layers,
        token_labels=token_labels,
        matrix_values=magnitudes,
        title="Layer x token mass-mean direction magnitude (blue alpha = min-max)",
        xlabel=xlabel,
        value_fmt="{:.3f}",
    )
    logging.info("Wrote %s (range [%.6f, %.6f])", mag_path, mag_vmin, mag_vmax)

    cosine_matrix = _cosine_similarity_rows(direction_probability)
    cosine_layers = list(range(1, n_layers))
    cos_path = os.path.join(run_root, "cosine_similarity_heatmap.png")
    cos_vmin, cos_vmax = write_rgba_heatmap(
        path=cos_path,
        run_layers=cosine_layers,
        token_labels=token_labels,
        matrix_values=cosine_matrix,
        title="Cosine similarity vs previous layer (blue alpha = min-max; row 0 excluded)",
        xlabel=xlabel,
        value_fmt="{:.3f}",
    )
    logging.info("Wrote %s (range [%.6f, %.6f])", cos_path, cos_vmin, cos_vmax)

    finished_at = datetime.now().isoformat(timespec="seconds")
    write_config_txt(
        os.path.join(run_root, "config.txt"),
        args=args,
        n_layers=n_layers,
        span_token_count=n_tokens,
        direction_probability_shape=tuple(direction_probability.shape),
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        magnitude_range=(mag_vmin, mag_vmax),
        cosine_range=(cos_vmin, cos_vmax),
        finished_at=finished_at,
    )
    summary = {
        "finished_at": finished_at,
        "input_h5": args.input_h5,
        "model_name": args.model_name,
        "new_h5_format": args.new_h5_format,
        "n_layers": n_layers,
        "span_token_count": n_tokens,
        "direction_probability_shape": list(direction_probability.shape),
        "low_conf_count": len(low_ids),
        "high_conf_count": len(high_ids),
        "h5_example_count": len(examples_h5),
        "magnitude_min": mag_vmin,
        "magnitude_max": mag_vmax,
        "cosine_min": cos_vmin,
        "cosine_max": cos_vmax,
        "token_labels": token_labels,
        "magnitude_heatmap": mag_path,
        "cosine_similarity_heatmap": cos_path,
    }
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logging.info("Done. Outputs in %s", run_root)


if __name__ == "__main__":
    main()
