#!/usr/bin/env python3
"""Rebuild layer_token_deviation_grid_paper.png from existing tokenwise runs.

Walks each module's results/ recursively (including manually nested folders).
Never overwrites the original annotated grid PNG. Skips existing paper PNGs
unless --force is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.paper_heatmaps import paper_png_path, write_layer_token_grid_plot_paper

TOKENWISE_MODULES = (
    "subblock_tokenwise_mass_mean_shift",
    "subblock_tokenwise_mean_ablation",
    "tokenwise_probability_mean_ablation",
    "tokenwise_probability_mass_mean_probe",
)

GRID_NAME = "layer_token_deviation_grid.png"
_BOOL_RE = re.compile(r"^(True|False)$", re.I)


def _parse_config_txt(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key = key.strip()
        if key:
            values[key] = raw.strip()
    return values


def _as_bool(raw: object) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    text = str(raw).strip()
    if _BOOL_RE.fullmatch(text):
        return text.lower() == "true"
    return None


def _infer_mean_from_low_confidence(
    summary: dict, config: dict[str, str], run_dir: Path
) -> bool:
    parsed = _as_bool(summary.get("mean_from_low_confidence"))
    if parsed is not None:
        return parsed
    parsed = _as_bool(config.get("mean_from_low_confidence"))
    if parsed is not None:
        return parsed
    steering = str(summary.get("steering_target") or config.get("steering_target") or "").lower()
    if steering == "high":
        return True
    if steering == "low":
        return False
    path_s = str(run_dir).lower()
    if "same_low" in path_s or "low_to_high" in path_s:
        return True
    if "same_high" in path_s or "high_to_low" in path_s:
        return False
    logging.warning("Could not infer mean_from_low_confidence for %s; defaulting to False", run_dir)
    return False


def _infer_ablate_with_same_confidence(
    summary: dict, config: dict[str, str], run_dir: Path
) -> bool:
    parsed = _as_bool(summary.get("ablate_with_same_confidence"))
    if parsed is not None:
        return parsed
    parsed = _as_bool(config.get("ablate_with_same_confidence"))
    if parsed is not None:
        return parsed
    path_s = str(run_dir).lower()
    return "same_low" in path_s or "same_high" in path_s


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _directed_from_signed(signed: np.ndarray, *, mean_from_low_confidence: bool) -> np.ndarray:
    out = np.full_like(signed, np.nan, dtype=np.float64)
    finite = np.isfinite(signed)
    if mean_from_low_confidence:
        out[finite] = np.maximum(-signed[finite], 0.0)
    else:
        out[finite] = np.maximum(signed[finite], 0.0)
    return out


def _matrices_from_summary(summary: dict) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    run_layers = [int(x) for x in summary["run_layers"]]
    token_positions = summary.get("token_positions") or {}
    span = summary.get("span_token_count")
    if span is None:
        if not token_positions:
            raise ValueError("summary.json has no token_positions / span_token_count")
        span = max(int(k) for k in token_positions) + 1
    span = int(span)
    token_labels: list[str] = []
    for pos in range(span):
        entry = token_positions.get(str(pos)) or token_positions.get(pos) or {}
        token_labels.append(str(entry.get("token_label", f"pos_{pos}")))

    conf_map = summary.get("layer_token_confidence") or {}
    dev_map = summary.get("layer_token_deviation_from_baseline") or {}
    baseline = (summary.get("baseline") or {}).get("mean_confidence")
    n_rows, n_cols = len(run_layers), span
    values = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    signed = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for r, layer in enumerate(run_layers):
        layer_key = str(layer)
        conf_row = conf_map.get(layer_key) or {}
        dev_row = dev_map.get(layer_key) or {}
        for c in range(n_cols):
            col_key = str(c)
            values[r, c] = _to_float(conf_row.get(col_key))
            if col_key in dev_row:
                signed[r, c] = _to_float(dev_row.get(col_key))
            elif np.isfinite(values[r, c]) and baseline is not None:
                signed[r, c] = float(values[r, c]) - float(baseline)
    return run_layers, token_labels, values, signed


def backfill_one(grid_path: Path, *, force: bool) -> str:
    paper_path = paper_png_path(grid_path)
    if paper_path.exists() and not force:
        return "skip"
    run_dir = grid_path.parent
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        logging.warning("No summary.json next to %s; skipping", grid_path)
        return "fail"
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    config = _parse_config_txt(run_dir / "config.txt")
    run_layers, token_labels, values, signed = _matrices_from_summary(summary)
    mean_from_low = _infer_mean_from_low_confidence(summary, config, run_dir)
    same_conf = _infer_ablate_with_same_confidence(summary, config, run_dir)
    if same_conf:
        desired = signed
    else:
        desired = _directed_from_signed(signed, mean_from_low_confidence=mean_from_low)
    write_layer_token_grid_plot_paper(
        path=paper_path,
        run_layers=run_layers,
        token_labels=token_labels,
        matrix_values=values,
        matrix_deviation_desired=desired,
        mean_from_low_confidence=mean_from_low,
        ablate_with_same_confidence=same_conf,
        linguistic_confidence_prompt=bool(summary.get("linguistic_confidence_prompt", False)),
    )
    return "wrote"


def iter_grid_paths(results_root: Path) -> list[Path]:
    return sorted(p for p in results_root.rglob(GRID_NAME) if p.is_file())


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
        help="Regenerate existing *_paper.png files (still never touches the original grid).",
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
            if status == "wrote":
                logging.info("Wrote %s", paper_png_path(grid_path))
    logging.info(
        "Tokenwise paper heatmap backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
