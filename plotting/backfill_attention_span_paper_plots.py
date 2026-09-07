#!/usr/bin/env python3
"""Write attention-span paper heatmaps from existing summary.json files.

Walks ``attention_probability_span_analysis/results/`` recursively (including
nested confidence-group folders). For each ``tables.probability_pos_*`` entry
with values, writes ``attention_table_probability_pos_{p}_paper.png`` when the
original annotated PNG exists. Also writes the ``__layer_avg_paper.png``
sibling when ``layer_avg_values`` is present.

Never overwrites original annotated PNGs, JSON, NPZ, or ``config.txt``.
Skips existing paper PNGs unless ``--force`` is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.attention_span_plots import write_attention_span_plot_paper
from plotting.paper_heatmaps import paper_png_path

MODULE_NAME = "attention_probability_span_analysis"
_POS_KEY_RE = re.compile(r"^probability_pos_(\d+)$")
_PROTECTED_NAMES = {"summary.json", "attention_tables.npz", "config.txt"}


@dataclass(frozen=True)
class PaperJob:
    original_png: Path
    paper_png: Path
    matrix: np.ndarray
    row_labels: list[str]
    col_labels: list[str]


def _as_matrix(raw: object, *, label: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{label} is not a rank-2 matrix, got shape={arr.shape}")
    return arr


def _row_labels_for_matrix(
    matrix: np.ndarray,
    *,
    layer_avg: bool,
    num_heads: int | None,
) -> list[str]:
    n_rows = int(matrix.shape[0])
    if layer_avg:
        return [str(i) for i in range(n_rows)]
    if num_heads is not None and num_heads > 0 and n_rows % num_heads == 0:
        n_layers = n_rows // num_heads
        return [f"{layer}.{head}" for layer in range(n_layers) for head in range(num_heads)]
    return [str(i) for i in range(n_rows)]


def _jobs_from_summary(summary_path: Path) -> list[PaperJob]:
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path} is not a JSON object")
    tables = summary.get("tables")
    if not isinstance(tables, dict):
        return []

    num_heads_raw = summary.get("num_heads")
    try:
        num_heads = int(num_heads_raw) if num_heads_raw is not None else None
    except (TypeError, ValueError):
        num_heads = None

    run_dir = summary_path.parent
    jobs: list[PaperJob] = []
    for key, payload in tables.items():
        match = _POS_KEY_RE.fullmatch(str(key))
        if match is None or not isinstance(payload, dict):
            continue
        pos = match.group(1)
        columns = payload.get("columns")
        values = payload.get("values")
        if not isinstance(columns, list) or values is None:
            continue
        col_labels = [str(c) for c in columns]

        original = run_dir / f"attention_table_probability_pos_{pos}.png"
        if original.is_file():
            matrix = _as_matrix(values, label=f"{summary_path} {key}.values")
            jobs.append(
                PaperJob(
                    original_png=original,
                    paper_png=paper_png_path(original),
                    matrix=matrix,
                    row_labels=_row_labels_for_matrix(
                        matrix, layer_avg=False, num_heads=num_heads
                    ),
                    col_labels=col_labels,
                )
            )

        layer_avg_values = payload.get("layer_avg_values")
        if layer_avg_values is None:
            continue
        layer_avg_original = run_dir / f"attention_table_probability_pos_{pos}__layer_avg.png"
        if not layer_avg_original.is_file():
            continue
        layer_avg_matrix = _as_matrix(
            layer_avg_values, label=f"{summary_path} {key}.layer_avg_values"
        )
        jobs.append(
            PaperJob(
                original_png=layer_avg_original,
                paper_png=paper_png_path(layer_avg_original),
                matrix=layer_avg_matrix,
                row_labels=_row_labels_for_matrix(
                    layer_avg_matrix, layer_avg=True, num_heads=num_heads
                ),
                col_labels=col_labels,
            )
        )
    return jobs


def iter_paper_jobs(results_root: Path) -> list[PaperJob]:
    jobs: list[PaperJob] = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        if not summary_path.is_file():
            continue
        jobs.extend(_jobs_from_summary(summary_path))
    jobs.sort(key=lambda job: (0 if "__layer_avg" in job.original_png.name else 1, str(job.original_png)))
    return jobs


def backfill_one(job: PaperJob, *, force: bool) -> str:
    if job.paper_png.name in _PROTECTED_NAMES or job.paper_png == job.original_png:
        raise RuntimeError(f"Refusing to overwrite protected file {job.paper_png}")
    if job.paper_png.exists() and not force:
        return "skip"
    write_attention_span_plot_paper(
        job.paper_png,
        matrix=job.matrix,
        row_labels=job.row_labels,
        col_labels=job.col_labels,
    )
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
        "--results_root",
        type=Path,
        default=None,
        help="Override path to attention_probability_span_analysis/results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing *_paper.png files (still never touches originals).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many paper jobs (layer-avg jobs are first).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results_root = args.results_root or (args.repo_root / MODULE_NAME / "results")
    if not results_root.is_dir():
        logging.warning("No results dir at %s", results_root)
        raise SystemExit(1)

    jobs = iter_paper_jobs(results_root)
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    logging.info("%s: %d paper job(s)", MODULE_NAME, len(jobs))

    counts = {"wrote": 0, "skip": 0, "fail": 0}
    for job in jobs:
        try:
            status = backfill_one(job, force=args.force)
        except Exception:
            logging.exception("Failed to backfill %s", job.original_png)
            status = "fail"
        counts[status] = counts.get(status, 0) + 1
        if status == "wrote":
            logging.info("Wrote %s", job.paper_png)
    logging.info(
        "Attention span paper plot backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
