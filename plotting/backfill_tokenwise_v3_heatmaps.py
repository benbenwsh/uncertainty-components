#!/usr/bin/env python3
"""Write bright red/blue v3 tokenwise heatmaps without touching existing PNGs.

For each ``layer_token_deviation_grid.png`` under the four tokenwise results/
trees, create:

- ``layer_token_deviation_grid_v3.png`` (annotated, cell scores)
- ``layer_token_deviation_grid_paper_v3.png`` (research-ready)

Never overwrites ``layer_token_deviation_grid.png``, ``*_paper.png``, or ``*_v2.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.backfill_tokenwise_paper_heatmaps import (
    TOKENWISE_MODULES,
    _matrices_from_summary,
    iter_grid_paths,
)
from plotting.paper_heatmaps import (
    paper_png_path,
    v3_png_path,
    write_layer_token_grid_plot_annotated,
    write_layer_token_grid_plot_paper,
)


def _v3_paths(grid_path: Path) -> tuple[Path, Path]:
    annotated_v3 = v3_png_path(grid_path)
    paper_v3 = v3_png_path(paper_png_path(grid_path))
    protected = {grid_path, paper_png_path(grid_path)}
    if annotated_v3 in protected or paper_v3 in protected:
        raise RuntimeError(f"Refusing to overwrite original heatmaps for {grid_path}")
    return annotated_v3, paper_v3


def backfill_one(grid_path: Path, *, force: bool) -> str:
    annotated_v3, paper_v3 = _v3_paths(grid_path)
    need_annotated = force or not annotated_v3.exists()
    need_paper = force or not paper_v3.exists()
    if not need_annotated and not need_paper:
        return "skip"

    summary_path = grid_path.parent / "summary.json"
    if not summary_path.is_file():
        logging.warning("No summary.json next to %s; skipping", grid_path)
        return "fail"
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    run_layers, token_labels, values, signed = _matrices_from_summary(summary)

    if need_annotated:
        write_layer_token_grid_plot_annotated(
            path=annotated_v3,
            run_layers=run_layers,
            token_labels=token_labels,
            matrix_values=values,
            matrix_deviation_signed=signed,
        )
        logging.info("Wrote %s", annotated_v3)
    if need_paper:
        write_layer_token_grid_plot_paper(
            path=paper_v3,
            run_layers=run_layers,
            token_labels=token_labels,
            matrix_values=values,
            matrix_deviation_desired=signed,
        )
        logging.info("Wrote %s", paper_v3)
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
        default=list(TOKENWISE_MODULES),
        help="Module directories under repo_root whose results/ should be walked.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing *_v3.png files (still never touches earlier heatmaps).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = {"wrote": 0, "skip": 0, "fail": 0}
    for module in args.modules:
        results_root = args.repo_root / module / "results"
        if not results_root.is_dir():
            logging.warning("No results dir for %s (%s)", module, results_root)
            continue
        paths = iter_grid_paths(results_root)
        logging.info("%s: %d annotated grid(s)", module, len(paths))
        for grid_path in paths:
            try:
                status = backfill_one(grid_path, force=args.force)
            except Exception:
                logging.exception("Failed to backfill %s", grid_path)
                status = "fail"
            counts[status] = counts.get(status, 0) + 1
    logging.info(
        "Tokenwise v3 heatmap backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
