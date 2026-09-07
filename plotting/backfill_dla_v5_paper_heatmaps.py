#!/usr/bin/env python3
"""Write DLA/DGA paper v5 heatmaps (larger fonts, no title).

For each ``heatmap.png`` under direct_logit_attribution and
direction_gradient_attr results/, create ``heatmap_paper_v5.png`` only.

Never overwrites earlier annotated, paper, v2, v3, or v4 PNGs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.backfill_dla_paper_heatmaps import (
    DLA_MODULES,
    infer_layout,
    iter_heatmap_paths,
    labels_to_heatmap_matrix,
    parse_ranked_component_values,
    title_from_heatmap_path,
)
from plotting.paper_heatmaps import (
    paper_png_path,
    v2_png_path,
    v3_png_path,
    v4_png_path,
    v5_png_path,
    write_diverging_heatmap_paper,
)


def _paper_v5_path(heatmap_path: Path) -> Path:
    paper_v5 = v5_png_path(paper_png_path(heatmap_path))
    protected = {
        heatmap_path,
        paper_png_path(heatmap_path),
        v2_png_path(heatmap_path),
        v2_png_path(paper_png_path(heatmap_path)),
        v3_png_path(heatmap_path),
        v3_png_path(paper_png_path(heatmap_path)),
        v4_png_path(heatmap_path),
        v4_png_path(paper_png_path(heatmap_path)),
    }
    if paper_v5 in protected:
        raise RuntimeError(f"Refusing to overwrite original heatmaps for {heatmap_path}")
    return paper_v5


def backfill_one(heatmap_path: Path, *, force: bool) -> str:
    paper_v5 = _paper_v5_path(heatmap_path)
    if paper_v5.exists() and not force:
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
        paper_v5,
        matrix,
        row_labels=[str(layer) for layer in range(n_layers)],
        col_labels=col_labels,
        title=title_from_heatmap_path(heatmap_path),
    )
    logging.info("Wrote %s", paper_v5)
    return "wrote"


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
        help="Regenerate existing *_paper_v5.png files (still never touches earlier heatmaps).",
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
    logging.info(
        "DLA/DGA paper v5 heatmap backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
