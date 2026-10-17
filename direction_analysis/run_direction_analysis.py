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
from typing import List, Optional, Sequence, Tuple

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mass_mean_probe.run_mass_mean_probe import (
    _expected_probability_span_token_budget,
    _prefix_tokens_for_linguistic_confidence,
    configure_prefix_tokens_for_model,
)
from direction_analysis.h5_probability_directions import (
    compute_probability_mass_mean_direction_streaming,
)
from plotting.paper_heatmaps import (
    PAPER_DPI,
    apply_input_output_token_axes,
    paper_png_path,
    sparse_layer_yticks,
)


MODULE_NAME = "direction_analysis"
RESULTS_STEM = "mass_mean_directions"
_LABEL_FS = 16
_TICK_FS = 13
_CELL_FS = 10
_ANNOTATED_DPI = 180
WHITE_TO_BLUE = LinearSegmentedColormap.from_list(
    "direction_white_blue",
    [(1.0, 1.0, 1.0), (0.1, 0.3, 1.0)],
)


def _resolve_run_root(cli_output_dir: Optional[str]) -> str:
    if cli_output_dir:
        os.makedirs(cli_output_dir, exist_ok=True)
        return cli_output_dir

    # Anchor to this file's directory so cwd does not create nested paths
    # (e.g. direction_analysis/direction_analysis/results/...).
    base_dir = str(SCRIPT_DIR / "results" / RESULTS_STEM)
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


def _row_minmax_alpha(values: np.ndarray) -> np.ndarray:
    """Per-row min-max: max in the row is 1, min is 0."""
    out = np.zeros_like(values, dtype=np.float32)
    for r in range(values.shape[0]):
        out[r] = _minmax_alpha(values[r])
    return out


def _alpha_grid(values: np.ndarray, alpha_mode: str) -> np.ndarray:
    if alpha_mode == "row":
        return _row_minmax_alpha(values)
    if alpha_mode == "global":
        return _minmax_alpha(values)
    raise ValueError(f"Unknown alpha_mode: {alpha_mode}")


def _finite_range(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.min(finite)), float(np.max(finite))


def _style_direction_heatmap_axes(
    ax,
    *,
    n_rows: int,
    n_cols: int,
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    tick_rotation: float = 30,
    sparse_layer_ticks: bool = False,
    model_name: Optional[str] = None,
    linguistic_confidence_prompt: bool = False,
    path_hint: str = "",
) -> None:
    if not sparse_layer_ticks:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([str(layer) for layer in run_layers], fontsize=_TICK_FS)
        ax.set_ylabel("Layer", fontsize=_LABEL_FS)
    apply_input_output_token_axes(
        ax,
        n_cols=n_cols,
        token_labels=token_labels,
        rotation=tick_rotation,
        fontsize=_TICK_FS,
        label_fontsize=_LABEL_FS,
        model_name=model_name,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        path_hint=path_hint,
    )
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.35, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    if sparse_layer_ticks:
        sparse_layer_yticks(
            ax,
            run_layers,
            fontsize=_TICK_FS,
            label_fontsize=_LABEL_FS,
        )


def write_rgba_heatmap(
    *,
    path: str,
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    matrix_values: np.ndarray,
    alpha_mode: str = "global",
    value_fmt: str = "{:.3f}",
    model_name: Optional[str] = None,
    linguistic_confidence_prompt: bool = False,
    path_hint: str = "",
) -> Tuple[float, float]:
    """Blue RGBA heatmap. Returns (vmin, vmax) of finite raw values."""
    n_rows, n_cols = matrix_values.shape
    vmin, vmax = _finite_range(matrix_values)
    alpha_grid = _alpha_grid(matrix_values, alpha_mode)

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
            ax.text(c, r, text, ha="center", va="center", fontsize=_CELL_FS, color="black")
    _style_direction_heatmap_axes(
        ax,
        n_rows=n_rows,
        n_cols=n_cols,
        run_layers=run_layers,
        token_labels=token_labels,
        tick_rotation=30,
        model_name=model_name,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        path_hint=path_hint,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_ANNOTATED_DPI, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


def write_rgba_heatmap_paper(
    *,
    path: str,
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    matrix_values: np.ndarray,
    alpha_mode: str = "global",
    cbar_label: str,
    model_name: Optional[str] = None,
    linguistic_confidence_prompt: bool = False,
    path_hint: str = "",
) -> Path:
    """Paper heatmap: no cell text, white-to-blue scale, colorbar."""
    out = Path(path)
    n_rows, n_cols = matrix_values.shape
    if alpha_mode == "row":
        display = _row_minmax_alpha(matrix_values)
        vmin, vmax = 0.0, 1.0
    else:
        display = np.ma.masked_invalid(np.asarray(matrix_values, dtype=np.float64))
        vmin, vmax = _finite_range(matrix_values)
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig_w = max(5.8, 0.62 * n_cols + 3.0)
    fig_h = max(4.6, 0.16 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        display,
        aspect="auto",
        interpolation="nearest",
        cmap=WHITE_TO_BLUE,
        vmin=vmin,
        vmax=vmax,
    )
    _style_direction_heatmap_axes(
        ax,
        n_rows=n_rows,
        n_cols=n_cols,
        run_layers=run_layers,
        token_labels=token_labels,
        tick_rotation=45,
        sparse_layer_ticks=True,
        model_name=model_name,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        path_hint=path_hint,
    )
    fig.tight_layout()
    cax = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.06, 0.0, 0.045, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)
    fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return out


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
        f"l_vs_un={getattr(args, 'l_vs_un', False)}",
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
            "expected_probability_tokens + 2. If true together with "
            "--linguistic_confidence_prompt, accept embeddings_probability lengths "
            ">= expected_confidence_tokens and use only the first N of those tokens."
        ),
    )
    parser.add_argument(
        "--linguistic_confidence_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use Confidence: span budget; if false, use Probability: span budget.",
    )
    parser.add_argument(
        "--l_vs_un",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true together with --linguistic_confidence_prompt, keep only examples "
            "whose Confidence: span token at index 6 decodes as L or Un (same as DLA). "
            "Ignored when --linguistic_confidence_prompt is off."
        ),
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
    direction_prob_token_budget = span_token_count
    allow_at_least_expected = (
        bool(args.linguistic_confidence_prompt) and bool(args.extend_probability_span)
    )
    apply_l_vs_un = bool(args.linguistic_confidence_prompt) and bool(args.l_vs_un)
    if args.l_vs_un and not args.linguistic_confidence_prompt:
        logging.warning("Ignoring --l_vs_un because --linguistic_confidence_prompt is off.")
    logging.info(
        "Span budget=%d allow_at_least_expected=%s l_vs_un=%s",
        direction_prob_token_budget,
        allow_at_least_expected,
        apply_l_vs_un,
    )

    logging.info("Computing probability mass-mean directions (streaming H5)...")
    direction_probability, low_ids, high_ids, n_layers, h5_example_count = (
        compute_probability_mass_mean_direction_streaming(
            args.input_h5,
            expected_probability_tokens=direction_prob_token_budget,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
            new_h5_format=args.new_h5_format,
            allow_at_least_expected=allow_at_least_expected,
            l_vs_un=apply_l_vs_un,
            model_name=args.model_name if apply_l_vs_un else None,
        )
    )
    run_layers = list(range(n_layers))
    logging.info(
        "Direction shape=%s | n_layers=%d | low=%d high=%d examples=%d",
        tuple(direction_probability.shape),
        n_layers,
        len(low_ids),
        len(high_ids),
        h5_example_count,
    )
    if direction_probability.ndim != 3:
        raise ValueError(
            f"Expected direction shape (layers, tokens, d_model), "
            f"got {direction_probability.shape}."
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
    logging.info("Writing direction pickles under %s/{{layer}}/{{token}}.pkl", run_root)
    write_direction_pickles(run_root, direction_probability)

    magnitudes = np.linalg.norm(direction_probability, axis=-1).astype(np.float32)
    mag_path = os.path.join(run_root, "magnitude_heatmap.png")
    mag_vmin, mag_vmax = write_rgba_heatmap(
        path=mag_path,
        run_layers=run_layers,
        token_labels=token_labels,
        matrix_values=magnitudes,
        alpha_mode="row",
        value_fmt="{:.3f}",
        model_name=args.model_name,
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
    )
    mag_paper = write_rgba_heatmap_paper(
        path=str(paper_png_path(mag_path)),
        run_layers=run_layers,
        token_labels=token_labels,
        matrix_values=magnitudes,
        alpha_mode="row",
        cbar_label="Relative magnitude within layer",
        model_name=args.model_name,
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
    )
    logging.info("Wrote %s and %s (range [%.6f, %.6f])", mag_path, mag_paper, mag_vmin, mag_vmax)

    cosine_matrix = _cosine_similarity_rows(direction_probability)
    cosine_layers = list(range(1, n_layers))
    cos_path = os.path.join(run_root, "cosine_similarity_heatmap.png")
    cos_vmin, cos_vmax = write_rgba_heatmap(
        path=cos_path,
        run_layers=cosine_layers,
        token_labels=token_labels,
        matrix_values=cosine_matrix,
        alpha_mode="global",
        value_fmt="{:.3f}",
        model_name=args.model_name,
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
    )
    cos_paper = write_rgba_heatmap_paper(
        path=str(paper_png_path(cos_path)),
        run_layers=cosine_layers,
        token_labels=token_labels,
        matrix_values=cosine_matrix,
        alpha_mode="global",
        cbar_label="Cosine similarity vs previous layer",
        model_name=args.model_name,
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
    )
    logging.info("Wrote %s and %s (range [%.6f, %.6f])", cos_path, cos_paper, cos_vmin, cos_vmax)

    finished_at = datetime.now().isoformat(timespec="seconds")
    write_config_txt(
        os.path.join(run_root, "config.txt"),
        args=args,
        n_layers=n_layers,
        span_token_count=n_tokens,
        direction_probability_shape=tuple(direction_probability.shape),
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=h5_example_count,
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
        "h5_example_count": h5_example_count,
        "linguistic_confidence_prompt": args.linguistic_confidence_prompt,
        "l_vs_un": args.l_vs_un,
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
