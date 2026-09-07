"""Attention-span heatmap writers (matplotlib only).

Annotated and paper plots share a sequential blue ``imshow`` of the raw
attention matrix. Column labels are drawn on both the top and bottom; there
are no plot titles. These helpers write whatever path the caller passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np

from plotting.paper_heatmaps import (
    ANNOTATED_DPI,
    PAPER_DPI,
    _CELL_FS,
    _LABEL_FS,
    _TICK_FS,
    _TOKENWISE_LAYER_TICK_FS,
    _add_cell_grid,
    paper_png_path,
)

_CBAR_LABEL = "Attention"
_CMAP = "Blues"


def _finite_vmax(arr: np.ndarray) -> float:
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.max(finite))
    if vmax <= 0:
        return 1.0
    return vmax


def _y_tick_labels(row_labels: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for lab in row_labels:
        text = str(lab)
        if text.startswith("layer_"):
            rest = text[len("layer_") :]
            labels.append(rest if rest.isdigit() else text)
        else:
            labels.append(text)
    return labels


def _figure_size(n_rows: int, n_cols: int, *, annotated: bool) -> tuple[float, float]:
    fig_w = max(8.0, 0.85 * n_cols + 3.8)
    if annotated:
        fig_h = max(6.0, 0.22 * n_rows + 2.4)
    else:
        fig_h = max(5.0, 0.16 * n_rows + 2.6)
    return fig_w, fig_h


def apply_column_labels_top_and_bottom(
    ax,
    col_labels: Sequence[str],
    *,
    rotation: float = 45,
    fontsize: int = _TICK_FS,
) -> object:
    """Same column labels on top and bottom; no x-axis titles."""
    labels = [str(x) for x in col_labels]
    n_cols = len(labels)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)
    ax.tick_params(axis="x", which="major", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xlabel("")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(np.arange(n_cols))
    ax_top.set_xticklabels(labels, rotation=rotation, ha="left", fontsize=fontsize)
    ax_top.set_xlabel("")
    return ax_top


def _draw_attention_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    annotate: bool,
):
    n_rows, n_cols = matrix.shape
    data = np.ma.masked_invalid(np.asarray(matrix, dtype=np.float64))
    vmax = _finite_vmax(matrix)
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=_CMAP,
        vmin=0.0,
        vmax=vmax,
    )
    if annotate:
        for r in range(n_rows):
            for c in range(n_cols):
                val = float(matrix[r, c]) if np.isfinite(matrix[r, c]) else float("nan")
                if not np.isfinite(val):
                    text = "NA"
                    color = "black"
                else:
                    text = f"{val:.3f}"
                    color = "white" if (vmax > 0 and val / vmax >= 0.55) else "black"
                ax.text(c, r, text, ha="center", va="center", fontsize=_CELL_FS, color=color)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(_y_tick_labels(row_labels), fontsize=_TOKENWISE_LAYER_TICK_FS)
    ax.set_ylabel("Layer", fontsize=_LABEL_FS)
    _add_cell_grid(ax, n_rows, n_cols, alpha=0.4)
    apply_column_labels_top_and_bottom(ax, col_labels)
    return im


def write_attention_span_plot_annotated(
    path: Union[str, Path],
    *,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> Path:
    """Annotated attention heatmap: cell numbers, no colorbar, no title."""
    out = Path(path)
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=_figure_size(n_rows, n_cols, annotated=True))
    _draw_attention_heatmap(
        ax,
        matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        annotate=True,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=ANNOTATED_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out


def write_attention_span_plot_paper(
    path: Union[str, Path],
    *,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> Path:
    """Paper attention heatmap: no cell text; colorbar inset after layout."""
    out = Path(path)
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=_figure_size(n_rows, n_cols, annotated=False))
    im = _draw_attention_heatmap(
        ax,
        matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        annotate=False,
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
    cbar.set_label(_CBAR_LABEL, fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)

    fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return out


def write_attention_span_plot_pair(
    annotated_path: Union[str, Path],
    *,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> tuple[Path, Path]:
    """Write annotated PNG then the ``*_paper.png`` sibling."""
    annotated = write_attention_span_plot_annotated(
        annotated_path,
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
    )
    paper = write_attention_span_plot_paper(
        paper_png_path(annotated_path),
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
    )
    return annotated, paper
