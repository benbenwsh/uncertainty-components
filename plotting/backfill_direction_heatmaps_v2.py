#!/usr/bin/env python3
"""Write v2 annotated and paper direction heatmaps from existing runs.

Walks ``direction_analysis/results/mass_mean_directions`` for run folders that
have layer pickles (and preferably ``summary.json``). Reconstructs magnitude
and consecutive-layer cosine matrices, then writes:

- ``magnitude_heatmap_v2.png`` / ``cosine_similarity_heatmap_v2.png``
- ``magnitude_heatmap_paper.png`` / ``cosine_similarity_heatmap_paper.png``

Never overwrites original annotated heatmaps. Skips existing v2/paper files
unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.paper_heatmaps import paper_png_path, v2_png_path
from direction_analysis.run_direction_analysis import (
    _cosine_similarity_rows,
    write_rgba_heatmap,
    write_rgba_heatmap_paper,
)

MAG_NAME = "magnitude_heatmap.png"
COS_NAME = "cosine_similarity_heatmap.png"


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / MAG_NAME).is_file() or (path / COS_NAME).is_file():
        return True
    if (path / "summary.json").is_file():
        return True
    return any(d.is_dir() and d.name.isdigit() for d in path.iterdir())


def iter_direction_run_dirs(results_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not results_root.is_dir():
        return run_dirs
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_run_dir(child):
            run_dirs.append(child)
            continue
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and _is_run_dir(grandchild):
                run_dirs.append(grandchild)
    return run_dirs


def _load_direction_stack(run_base: Path) -> np.ndarray | None:
    layer_dirs = sorted(
        [d for d in run_base.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not layer_dirs:
        return None
    layers: list[np.ndarray] = []
    n_tokens = None
    hidden_dim = None
    for layer_dir in layer_dirs:
        token_pkls = sorted(
            [p for p in layer_dir.glob("*.pkl") if p.stem.isdigit()],
            key=lambda p: int(p.stem),
        )
        if not token_pkls:
            logging.warning("No token pickles in %s", layer_dir)
            return None
        vecs = []
        for pkl_path in token_pkls:
            try:
                with pkl_path.open("rb") as f:
                    payload = pickle.load(f)
            except Exception as exc:
                logging.warning("Failed to read %s: %s", pkl_path, exc)
                return None
            direction = payload.get("direction") if isinstance(payload, dict) else None
            if direction is None:
                logging.warning("No direction in %s", pkl_path)
                return None
            vec = np.asarray(direction, dtype=np.float32).reshape(-1)
            vecs.append(vec)
        stacked = np.stack(vecs, axis=0)
        if n_tokens is None:
            n_tokens = stacked.shape[0]
            hidden_dim = stacked.shape[1]
        elif stacked.shape != (n_tokens, hidden_dim):
            logging.warning("Shape mismatch in %s: %s", layer_dir, stacked.shape)
            return None
        layers.append(stacked)
    return np.stack(layers, axis=0)


def _token_labels_for_run(run_base: Path, n_tokens: int) -> list[str]:
    summary_path = run_base / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        labels = summary.get("token_labels")
        if isinstance(labels, list) and labels:
            labels = [str(x) for x in labels]
            if len(labels) < n_tokens:
                labels.extend([f"value_{i}" for i in range(len(labels), n_tokens)])
            return labels[:n_tokens]
    return [f"tok_{i}" for i in range(n_tokens)]


def backfill_one(run_base: Path, *, force: bool) -> str:
    mag_orig = run_base / MAG_NAME
    cos_orig = run_base / COS_NAME
    mag_v2 = v2_png_path(mag_orig)
    cos_v2 = v2_png_path(cos_orig)
    mag_paper = paper_png_path(mag_orig)
    cos_paper = paper_png_path(cos_orig)
    targets = (mag_v2, cos_v2, mag_paper, cos_paper)
    if mag_v2 == mag_orig or cos_v2 == cos_orig or mag_paper == mag_orig or cos_paper == cos_orig:
        raise RuntimeError(f"Refusing to overwrite originals in {run_base}")
    if not force and all(p.exists() for p in targets):
        return "skip"

    stack = _load_direction_stack(run_base)
    if stack is None:
        logging.warning("Could not reconstruct directions in %s; skipping", run_base)
        return "fail"

    n_layers, n_tokens, _ = stack.shape
    token_labels = _token_labels_for_run(run_base, n_tokens)
    magnitudes = np.linalg.norm(stack, axis=-1).astype(np.float32)
    cosine_matrix = _cosine_similarity_rows(stack)
    cosine_layers = list(range(1, n_layers))
    run_layers = list(range(n_layers))

    if force or not mag_v2.exists():
        write_rgba_heatmap(
            path=str(mag_v2),
            run_layers=run_layers,
            token_labels=token_labels,
            matrix_values=magnitudes,
            alpha_mode="row",
        )
        logging.info("Wrote %s", mag_v2)
    if force or not mag_paper.exists():
        write_rgba_heatmap_paper(
            path=str(mag_paper),
            run_layers=run_layers,
            token_labels=token_labels,
            matrix_values=magnitudes,
            alpha_mode="row",
            cbar_label="Relative magnitude within layer",
        )
        logging.info("Wrote %s", mag_paper)
    if force or not cos_v2.exists():
        write_rgba_heatmap(
            path=str(cos_v2),
            run_layers=cosine_layers,
            token_labels=token_labels,
            matrix_values=cosine_matrix,
            alpha_mode="global",
        )
        logging.info("Wrote %s", cos_v2)
    if force or not cos_paper.exists():
        write_rgba_heatmap_paper(
            path=str(cos_paper),
            run_layers=cosine_layers,
            token_labels=token_labels,
            matrix_values=cosine_matrix,
            alpha_mode="global",
            cbar_label="Cosine similarity vs previous layer",
        )
        logging.info("Wrote %s", cos_paper)
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
        help="Override path to mass_mean_directions (default: <repo>/direction_analysis/results/mass_mean_directions).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing *_v2.png / *_paper.png (never touches originals).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results_root = args.results_root or (
        args.repo_root / "direction_analysis" / "results" / "mass_mean_directions"
    )
    if not results_root.is_dir():
        logging.error("No mass_mean_directions directory at %s", results_root)
        raise SystemExit(1)

    run_dirs = iter_direction_run_dirs(results_root)
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
        "Direction heatmap v2 backfill done: wrote=%d skip=%d fail=%d",
        counts["wrote"],
        counts["skip"],
        counts["fail"],
    )
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
