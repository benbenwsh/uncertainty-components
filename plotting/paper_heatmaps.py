"""Tokenwise / DLA heatmap writers.

Tokenwise annotated and paper plots share signed red/blue coloring
(blue = increase vs baseline, red = decrease) and put input tokens on top.
These helpers write whatever path the caller passes; they never rename files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np

PAPER_DPI = 300
ANNOTATED_DPI = 180
DLA_ANNOTATED_DPI = 160
_GRID_COLOR = "0.75"
_GRID_LW = 0.3
_TICK_FS = 14
_LABEL_FS = 18
_CELL_FS = 14
_TOKENWISE_LAYER_TICK_FS = 10
_SIGNED_CBAR_LABEL = "Δ confidence vs baseline"

BRIGHT_RDBU = LinearSegmentedColormap.from_list(
    "bright_rdbu",
    ["#FF3B30", "#FFFFFF", "#007AFF"],
)


def paper_png_path(path: Union[str, Path]) -> Path:
    """``foo.png`` → ``foo_paper.png`` (never the original path)."""
    p = Path(path)
    if p.stem.endswith("_paper"):
        return p
    return p.with_name(f"{p.stem}_paper{p.suffix}")


def v2_png_path(path: Union[str, Path]) -> Path:
    """``foo.png`` → ``foo_v2.png``; ``foo_paper.png`` → ``foo_paper_v2.png``."""
    p = Path(path)
    if p.stem.endswith("_v2"):
        return p
    return p.with_name(f"{p.stem}_v2{p.suffix}")


def v3_png_path(path: Union[str, Path]) -> Path:
    """``foo.png`` → ``foo_v3.png``; ``foo_paper.png`` → ``foo_paper_v3.png``."""
    p = Path(path)
    if p.stem.endswith("_v3"):
        return p
    return p.with_name(f"{p.stem}_v3{p.suffix}")


def v4_png_path(path: Union[str, Path]) -> Path:
    """``foo.png`` → ``foo_v4.png``; ``foo_paper.png`` → ``foo_paper_v4.png``."""
    p = Path(path)
    if p.stem.endswith("_v4"):
        return p
    return p.with_name(f"{p.stem}_v4{p.suffix}")


def v5_png_path(path: Union[str, Path]) -> Path:
    """``foo.png`` → ``foo_v5.png``; ``foo_paper.png`` → ``foo_paper_v5.png``."""
    p = Path(path)
    if p.stem.endswith("_v5"):
        return p
    return p.with_name(f"{p.stem}_v5{p.suffix}")


def v6_png_path(path: Union[str, Path]) -> Path:
    """``foo.png`` → ``foo_v6.png``; ``foo_paper.png`` → ``foo_paper_v6.png``."""
    p = Path(path)
    if p.stem.endswith("_v6"):
        return p
    return p.with_name(f"{p.stem}_v6{p.suffix}")


def render_token_label(token: str) -> str:
    escaped = token.encode("unicode_escape").decode("ascii")
    return escaped if escaped else "<empty>"


def grid_input_output_tick_labels(
    token_labels: Sequence[str], n_cols: int
) -> tuple[list[str], list[str]]:
    output_labels = [f"{i}:{render_token_label(tok)}" for i, tok in enumerate(token_labels)]
    if len(output_labels) < n_cols:
        output_labels.extend([""] * (n_cols - len(output_labels)))
    output_labels = output_labels[:n_cols]
    input_labels = ["<last_answer_token>"] + output_labels[:-1]
    if len(input_labels) < n_cols:
        input_labels.extend([""] * (n_cols - len(input_labels)))
    return input_labels[:n_cols], output_labels


def apply_input_output_token_axes(
    ax,
    *,
    n_cols: int,
    token_labels: Sequence[str],
    rotation: float = 30,
    fontsize: int = _TICK_FS,
    label_fontsize: int = _LABEL_FS,
) -> object:
    """Output tokens on the bottom; input tokens on top."""
    input_labels, output_labels = grid_input_output_tick_labels(token_labels, n_cols)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(output_labels, rotation=rotation, ha="right", fontsize=fontsize)
    ax.tick_params(axis="x", which="major", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel("Output tokens", fontsize=label_fontsize)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(np.arange(n_cols))
    ax_top.set_xticklabels(input_labels, rotation=rotation, ha="left", fontsize=fontsize)
    ax_top.set_xlabel("Input tokens", fontsize=label_fontsize)
    return ax_top


def _finite_absmax(arr: np.ndarray) -> float:
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.max(np.abs(finite)))


def _add_cell_grid(ax, n_rows: int, n_cols: int, *, alpha: float = 0.8) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=_GRID_COLOR, linestyle="-", linewidth=_GRID_LW, alpha=alpha)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)


def write_layer_token_grid_plot_annotated(
    *,
    path: Union[str, Path],
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    matrix_values: np.ndarray,
    matrix_deviation_signed: np.ndarray,
) -> Path:
    """Annotated layer×token heatmap: cell scores, signed red/blue, input tokens on top."""
    out = Path(path)
    n_rows, n_cols = matrix_values.shape
    data = np.ma.masked_invalid(np.asarray(matrix_deviation_signed, dtype=np.float64))
    vmax = _finite_absmax(matrix_deviation_signed)
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(max(9.0, 1.4 * n_cols), max(6.0, 0.35 * n_rows)))
    ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=BRIGHT_RDBU,
        vmin=-vmax,
        vmax=vmax,
    )
    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix_values[r, c]
            text = "NA" if np.isnan(val) else f"{val:.3f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=_CELL_FS, color="black")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(layer) for layer in run_layers], fontsize=_TOKENWISE_LAYER_TICK_FS)
    ax.set_ylabel("Layer", fontsize=_LABEL_FS)
    apply_input_output_token_axes(ax, n_cols=n_cols, token_labels=token_labels, rotation=30)
    _add_cell_grid(ax, n_rows, n_cols, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=ANNOTATED_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def write_layer_token_grid_plot_paper(
    *,
    path: Union[str, Path],
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    matrix_values: np.ndarray,
    matrix_deviation_desired: np.ndarray,
    mean_from_low_confidence: bool = False,
    ablate_with_same_confidence: bool = False,
    linguistic_confidence_prompt: bool = False,
) -> Path:
    """Compact layer×token heatmap: no cell text, colorbar, input tokens on top."""
    del matrix_values, mean_from_low_confidence, ablate_with_same_confidence
    del linguistic_confidence_prompt
    out = Path(path)
    n_rows, n_cols = matrix_deviation_desired.shape
    data = np.ma.masked_invalid(np.asarray(matrix_deviation_desired, dtype=np.float64))
    vmax = _finite_absmax(matrix_deviation_desired)
    if vmax <= 0:
        vmax = 1.0

    fig_w = max(6.2, 0.62 * n_cols + 3.4)
    fig_h = max(5.0, 0.16 * n_rows + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=BRIGHT_RDBU,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(layer) for layer in run_layers], fontsize=_TOKENWISE_LAYER_TICK_FS)
    ax.set_ylabel("Layer", fontsize=_LABEL_FS)
    apply_input_output_token_axes(ax, n_cols=n_cols, token_labels=token_labels, rotation=45)
    _add_cell_grid(ax, n_rows, n_cols)
    fig.tight_layout()

    # twiny() is incompatible with colorbar-as-subplot + tight_layout, which
    # was pulling the scale into the grid. Place it after layout, outside ax.
    # Keep the anchor bbox narrow so bbox_inches="tight" does not reserve a
    # second full-axes-width of empty space to the right.
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
    cbar.set_label(_SIGNED_CBAR_LABEL, fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)

    fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return out


def write_diverging_heatmap_annotated(
    path: Union[str, Path],
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    rounding_dp: int = 2,
) -> Path:
    """Annotated layer×component heatmap: cell scores, bright red/blue, labels on top."""
    del title
    out = Path(path)
    n_rows, n_cols = matrix.shape
    data = np.ma.masked_invalid(np.asarray(matrix, dtype=np.float64))
    vmax = _finite_absmax(matrix)
    if vmax <= 0:
        vmax = 1.0

    fig_w = max(8.0, 0.55 * n_cols + 2.0)
    fig_h = max(6.0, 0.28 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=BRIGHT_RDBU,
        vmin=-vmax,
        vmax=vmax,
    )
    fmt = f"{{:.{rounding_dp}f}}"
    for r in range(n_rows):
        for c in range(n_cols):
            ax.text(
                c,
                r,
                fmt.format(float(matrix[r, c])),
                ha="center",
                va="center",
                fontsize=_CELL_FS,
                color="black",
            )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(row_labels), fontsize=_TICK_FS)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(list(col_labels), rotation=45, ha="left", fontsize=_TICK_FS)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Layer", fontsize=_LABEL_FS)
    ax.set_xlabel("Component", fontsize=_LABEL_FS)
    fig.tight_layout()
    fig.savefig(out, dpi=DLA_ANNOTATED_DPI)
    plt.close(fig)
    return out


def write_diverging_heatmap_paper(
    path: Union[str, Path],
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
) -> Path:
    """Compact layer×component heatmap with a symmetric diverging colorbar."""
    del title
    out = Path(path)
    n_rows, n_cols = matrix.shape
    data = np.ma.masked_invalid(np.asarray(matrix, dtype=np.float64))
    vmax = _finite_absmax(matrix)
    if vmax <= 0:
        vmax = 1.0

    fig_w = max(4.9, 0.28 * n_cols + 2.6)
    fig_h = max(4.7, 0.16 * n_rows + 2.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=BRIGHT_RDBU,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(row_labels), fontsize=_TICK_FS)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(list(col_labels), rotation=45, ha="right", fontsize=_TICK_FS)
    ax.tick_params(axis="x", which="major", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.xaxis.set_label_position("bottom")
    ax.set_ylabel("Layer", fontsize=_LABEL_FS)
    ax.set_xlabel("Component", fontsize=_LABEL_FS)
    _add_cell_grid(ax, n_rows, n_cols)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Attribution", fontsize=_LABEL_FS)
    cbar.ax.tick_params(labelsize=_TICK_FS)

    fig.tight_layout()
    fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out
