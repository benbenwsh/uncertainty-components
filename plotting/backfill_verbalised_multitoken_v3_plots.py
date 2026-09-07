#!/usr/bin/env python3
"""Write v3 run-folder plots with a dotted layer-0 baseline.

Walks ``verbalised_confidence_probes/results/mult_toks_all_layers`` one or two
levels deep (``<run_id>`` or ``<model>/<run_id>``). Regenerates only existing
run-folder ``*_all_tokens_*.png`` graphs as sibling ``*_v3.png`` files.

Guess, probability, and combined charts get a dotted horizontal baseline at
tok_0 layer-0 loss (combined always uses tok_0_guess). Prompt / sem_answer
charts are unchanged besides the shared v2 styling.

Never overwrites original or ``*_v2.png`` files. Loads metrics from
``tok_*/layer_*/config.txt``, falling back to probe pickles.
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

from plotting.backfill_verbalised_multitoken_v2_plots import (
    _run_has_more_graphs,
    iter_mult_toks_run_dirs,
)

try:
    from verbalised_confidence_probes.h5_train_multitoken_verbalised_confidence_probe import (
        _load_all_token_metrics_from_config_txt,
        _plot_metrics_all_tokens,
    )
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "verbalised_confidence_probes"))
    from h5_train_multitoken_verbalised_confidence_probe import (
        _load_all_token_metrics_from_config_txt,
        _plot_metrics_all_tokens,
    )


def backfill_one(run_base: Path, *, force: bool) -> str:
    all_token_metrics = _load_all_token_metrics_from_config_txt(run_base)
    if not all_token_metrics:
        logging.warning("No token metrics found in %s; skipping", run_base)
        return "fail"

    _plot_metrics_all_tokens(
        run_base,
        all_token_metrics,
        more_graphs=_run_has_more_graphs(run_base),
        name_suffix="_v3",
        skip_existing=not force,
        only_if_original_exists=True,
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
        help="Override path to mult_toks_all_layers (default: <repo>/verbalised_confidence_probes/results/mult_toks_all_layers).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing *_v3.png files (still never touches original or v2 PNGs).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results_root = args.results_root or (
        args.repo_root / "verbalised_confidence_probes" / "results" / "mult_toks_all_layers"
    )
    if not results_root.is_dir():
        logging.error("No mult_toks_all_layers directory at %s", results_root)
        raise SystemExit(1)

    run_dirs = iter_mult_toks_run_dirs(results_root)
    logging.info("Found %d run folder(s) under %s", len(run_dirs), results_root)
    counts = {"wrote": 0, "skip": 0, "fail": 0}
    for run_base in run_dirs:
        logging.info("Backfilling %s", run_base)
        try:
            status = backfill_one(run_base, force=args.force)
        except Exception:
            logging.exception("Failed to backfill %s", run_base)
            status = "fail"
        counts[status] = counts.get(status, 0) + 1
    logging.info(
        "Verbalised multitoken v3 plot backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
