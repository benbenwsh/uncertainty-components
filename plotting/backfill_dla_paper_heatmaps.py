#!/usr/bin/env python3
"""Rebuild heatmap_paper.png from existing DLA / direction-gradient-attr runs.

Walks each module's results/ recursively (including manually nested folders).
Never overwrites the original annotated heatmap.png. Skips existing paper PNGs
unless --force is set. Reads sibling ranked.txt (attributions.h5 is unused).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.paper_heatmaps import paper_png_path, write_diverging_heatmap_paper

DLA_MODULES = (
    "direct_logit_attribution",
    "direction_gradient_attr",
)

HEATMAP_NAME = "heatmap.png"
_LABEL_LINE_RE = re.compile(
    r"^(L\d+_(?:attn|mlp_block|mlp|H\d+))\t([^\t\n]+)"
)
_FINE_HEAD_RE = re.compile(r"^L(\d+)_H(\d+)$")
_FINE_MLP_RE = re.compile(r"^L(\d+)_mlp_block$")
_COARSE_ATTN_RE = re.compile(r"^L(\d+)_attn$")
_COARSE_MLP_RE = re.compile(r"^L(\d+)_mlp$")


def parse_ranked_component_values(path: Path) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LABEL_LINE_RE.match(line.strip())
        if not match:
            continue
        labels.append(match.group(1))
        values.append(float(match.group(2)))
    if not labels:
        raise ValueError(f"No component rows in {path}")
    return labels, np.asarray(values, dtype=np.float32)


def infer_layout(labels: list[str]) -> tuple[str, int, int]:
    max_layer = -1
    max_head = -1
    n_fine = 0
    n_coarse = 0
    for lab in labels:
        m = _FINE_HEAD_RE.fullmatch(lab)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
            max_head = max(max_head, int(m.group(2)))
            n_fine += 1
            continue
        m = _FINE_MLP_RE.fullmatch(lab)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
            n_fine += 1
            continue
        m = _COARSE_ATTN_RE.fullmatch(lab)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
            n_coarse += 1
            continue
        m = _COARSE_MLP_RE.fullmatch(lab)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
            n_coarse += 1
    if max_layer < 0:
        raise ValueError(f"Could not infer layers from labels: {labels[:8]!r}")
    n_layers = max_layer + 1
    if n_fine:
        return "fine", n_layers, max(max_head + 1, 1)
    return "coarse", n_layers, 0


def labels_to_heatmap_matrix(
    labels: list[str],
    values: np.ndarray,
    *,
    granularity: str,
    n_layers: int,
    n_heads: int,
) -> tuple[np.ndarray, list[str]]:
    by_label = {lab: float(values[i]) for i, lab in enumerate(labels)}
    if granularity == "coarse":
        col_labels = ["attn", "mlp"]
        mat = np.zeros((n_layers, 2), dtype=np.float32)
        for layer in range(n_layers):
            mat[layer, 0] = by_label.get(f"L{layer}_attn", 0.0)
            mat[layer, 1] = by_label.get(f"L{layer}_mlp", 0.0)
        return mat, col_labels
    if granularity == "fine":
        col_labels = [f"H{h}" for h in range(n_heads)] + ["mlp_block"]
        mat = np.zeros((n_layers, n_heads + 1), dtype=np.float32)
        for layer in range(n_layers):
            for head in range(n_heads):
                mat[layer, head] = by_label.get(f"L{layer}_H{head}", 0.0)
            mat[layer, n_heads] = by_label.get(f"L{layer}_mlp_block", 0.0)
        return mat, col_labels
    raise ValueError(f"Unknown granularity {granularity!r}")


def title_from_heatmap_path(heatmap_path: Path) -> str:
    leaf = heatmap_path.parent.name
    experiment = heatmap_path.parent.parent.name
    return f"{experiment} / {leaf}"


def backfill_one(heatmap_path: Path, *, force: bool) -> str:
    paper_path = paper_png_path(heatmap_path)
    if paper_path.exists() and not force:
        return "skip"
    ranked_path = heatmap_path.parent / "ranked.txt"
    if not ranked_path.is_file():
        logging.warning("No ranked.txt next to %s; skipping", heatmap_path)
        return "fail"
    labels, values = parse_ranked_component_values(ranked_path)
    granularity, n_layers, n_heads = infer_layout(labels)
    matrix, col_labels = labels_to_heatmap_matrix(
        labels,
        values,
        granularity=granularity,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    write_diverging_heatmap_paper(
        paper_path,
        matrix,
        row_labels=[str(layer) for layer in range(n_layers)],
        col_labels=col_labels,
        title=title_from_heatmap_path(heatmap_path),
    )
    return "wrote"


def iter_heatmap_paths(results_root: Path) -> list[Path]:
    return sorted(
        p
        for p in results_root.rglob(HEATMAP_NAME)
        if p.is_file() and p.name == HEATMAP_NAME
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: inferred from this file).",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(DLA_MODULES),
        help="Module directories under repo_root whose results/ should be walked.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing *_paper.png files (still never touches the original heatmap).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = {"wrote": 0, "skip": 0, "fail": 0}
    for module in args.modules:
        results_root = args.repo_root / module / "results"
        if not results_root.is_dir():
            logging.warning("No results dir for %s (%s)", module, results_root)
            continue
        paths = iter_heatmap_paths(results_root)
        logging.info("%s: %d annotated heatmap(s)", module, len(paths))
        for heatmap_path in paths:
            try:
                status = backfill_one(heatmap_path, force=args.force)
            except Exception:
                logging.exception("Failed to backfill %s", heatmap_path)
                status = "fail"
            counts[status] = counts.get(status, 0) + 1
            if status == "wrote":
                logging.info("Wrote %s", paper_png_path(heatmap_path))
    logging.info(
        "DLA/DGA paper heatmap backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
