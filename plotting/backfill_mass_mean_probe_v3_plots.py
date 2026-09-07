#!/usr/bin/env python3
"""Write mass_mean_probe v3 confidence-vs-alpha plots without overwriting originals.

For each original ``verbalised_confidence_vs_alpha__*.png`` under
mass_mean_probe/results/, create ``{stem}_v3.png`` from sibling ``summary.json``
and ``config.txt``. Never overwrites original or ``*_v2.png`` files.
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

from plotting.backfill_mass_mean_probe_v2_plots import (
    PLOT_PREFIX,
    _mode_from_plot_name,
    _parse_config_txt,
    _parse_targets,
    _parse_threshold,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
)
from plotting.mass_mean_probe_plots import plot_mode_confidence_from_summary
from plotting.paper_heatmaps import v2_png_path, v3_png_path

_VERSIONED_SUFFIXES = ("_v2", "_v3")


def iter_original_plot_paths(results_root: Path) -> list[Path]:
    return sorted(
        p
        for p in results_root.rglob(f"{PLOT_PREFIX}*.png")
        if p.is_file()
        and p.name.startswith(PLOT_PREFIX)
        and not any(p.stem.endswith(suffix) for suffix in _VERSIONED_SUFFIXES)
    )


def backfill_one(plot_path: Path, *, force: bool) -> str:
    out_path = v3_png_path(plot_path)
    protected = {plot_path, v2_png_path(plot_path)}
    if out_path in protected:
        raise RuntimeError(f"Refusing to overwrite original plot {plot_path}")
    if out_path.exists() and not force:
        return "skip"

    run_dir = plot_path.parent
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.txt"
    if not summary_path.is_file():
        logging.warning("No summary.json next to %s; skipping", plot_path)
        return "fail"
    if not config_path.is_file():
        logging.warning("No config.txt next to %s; skipping", plot_path)
        return "fail"

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        logging.warning("summary.json is not a dict in %s; skipping", run_dir)
        return "fail"

    config = _parse_config_txt(config_path)
    low_thr = _parse_threshold(
        config.get("low_conf_threshold"),
        default=DEFAULT_LOW_THRESHOLD,
        name="low_conf_threshold",
        run_dir=run_dir,
    )
    high_thr = _parse_threshold(
        config.get("high_conf_threshold"),
        default=DEFAULT_HIGH_THRESHOLD,
        name="high_conf_threshold",
        run_dir=run_dir,
    )
    targets = _parse_targets(config.get("ablation_targets"))
    mode_name = _mode_from_plot_name(plot_path)
    mode_payload = summary.get(mode_name)
    if not isinstance(mode_payload, dict):
        logging.warning("No mode %r in %s; skipping", mode_name, summary_path)
        return "fail"

    plot_mode_confidence_from_summary(
        mode_name=mode_name,
        mode_payload=mode_payload,
        ablation_targets=targets,
        output_path=str(out_path),
        low_conf_threshold=low_thr,
        high_conf_threshold=high_thr,
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
        "--force",
        action="store_true",
        help="Regenerate existing *_v3.png files (still never touches earlier plots).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results_root = args.repo_root / "mass_mean_probe" / "results"
    if not results_root.is_dir():
        logging.warning("No results dir at %s", results_root)
        raise SystemExit(1)

    paths = iter_original_plot_paths(results_root)
    logging.info("mass_mean_probe: %d original plot(s)", len(paths))
    counts = {"wrote": 0, "skip": 0, "fail": 0}
    for plot_path in paths:
        try:
            status = backfill_one(plot_path, force=args.force)
        except Exception:
            logging.exception("Failed to backfill %s", plot_path)
            status = "fail"
        counts[status] = counts.get(status, 0) + 1
    logging.info(
        "mass_mean_probe v3 plot backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
