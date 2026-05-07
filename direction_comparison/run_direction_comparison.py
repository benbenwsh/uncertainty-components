#!/usr/bin/env python3
"""
Compare direction vectors across mass-mean runs and verbalised probe outputs.

Inputs:
  - One or more mass-mean run directories containing layer_<idx>_directions.pkl
  - One verbalised-confidence probe directory containing:
      tok_<n>_guess/layer_<k>/verbalised_confidence_probe.pkl
      tok_<n>_probability|prob/layer_<k>/verbalised_confidence_probe.pkl

Outputs:
  - results.txt: per-layer/per-token cosine similarities and norms
  - config.txt: experiment setup metadata
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import itertools
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_layers_spec(spec: str) -> List[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Layer spec cannot be empty.")
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        layers = list(range(int(a.strip()), int(b.strip()) + 1))
    else:
        layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not layers:
        raise ValueError("No layers parsed from layer spec.")
    for layer_idx in layers:
        if layer_idx < 0:
            raise ValueError(f"Layer index must be non-negative, got {layer_idx}.")
    return layers


def resolve_output_dir(cli_output_dir: Optional[str]) -> Path:
    if cli_output_dir:
        out_dir = Path(cli_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    repo_dir = Path(__file__).resolve().parent
    base_dir = repo_dir / "results"
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    run_idx = max((int(p.name) for p in existing), default=0) + 1
    out_dir = base_dir / str(run_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


@dataclass
class SourceVectors:
    source_label: str
    kind: str  # mass_mean | verbalised_probe
    by_layer_guess: Dict[int, List[np.ndarray]]
    by_layer_probability: Dict[int, List[np.ndarray]]


def _parse_mass_mean_layer_filename(name: str) -> Optional[int]:
    match = re.fullmatch(r"layer_(\d+)_directions\.pkl", name)
    if match is None:
        return None
    return int(match.group(1))


def load_mass_mean_source(path: Path, requested_layers: Sequence[int], source_label: str) -> SourceVectors:
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Mass-mean path is not a directory: {path}")

    layer_files: Dict[int, Path] = {}
    for child in path.iterdir():
        if child.is_file():
            layer_idx = _parse_mass_mean_layer_filename(child.name)
            if layer_idx is not None:
                layer_files[layer_idx] = child

    requested_set = set(requested_layers)
    found_set = set(layer_files.keys())
    if found_set != requested_set:
        missing = sorted(requested_set - found_set)
        extra = sorted(found_set - requested_set)
        raise ValueError(
            f"Mass-mean directory {path} has layer pickle mismatch. "
            f"requested={sorted(requested_set)} found={sorted(found_set)} missing={missing} extra={extra}"
        )

    by_layer_guess: Dict[int, List[np.ndarray]] = {}
    by_layer_probability: Dict[int, List[np.ndarray]] = {}

    for layer_idx in requested_layers:
        file_path = layer_files[layer_idx]
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {file_path}.")
        if int(payload.get("layer_idx")) != layer_idx:
            raise ValueError(
                f"Payload layer_idx mismatch in {file_path}: "
                f"payload={payload.get('layer_idx')} expected={layer_idx}"
            )
        guess_vecs = payload.get("guess_prefix_directions")
        prob_vecs = payload.get("probability_prefix_directions")
        if not isinstance(guess_vecs, list) or not isinstance(prob_vecs, list):
            raise ValueError(f"Missing guess/probability direction lists in {file_path}.")
        by_layer_guess[layer_idx] = [np.asarray(v, dtype=np.float32).reshape(-1) for v in guess_vecs]
        by_layer_probability[layer_idx] = [np.asarray(v, dtype=np.float32).reshape(-1) for v in prob_vecs]

    return SourceVectors(
        source_label=source_label,
        kind="mass_mean",
        by_layer_guess=by_layer_guess,
        by_layer_probability=by_layer_probability,
    )


def _parse_token_dir_name(dirname: str) -> Optional[Tuple[int, str]]:
    match = re.fullmatch(r"tok_(\d+)_(guess|probability|prob)", dirname)
    if match is None:
        return None
    tok_idx = int(match.group(1))
    raw_kind = match.group(2)
    kind = "probability" if raw_kind in {"probability", "prob"} else "guess"
    return tok_idx, kind


def _normalize_token_indices(entries: List[Tuple[int, Path]], kind: str) -> List[Tuple[int, Path]]:
    if not entries:
        return []
    entries = sorted(entries, key=lambda x: x[0])
    token_indices = [idx for idx, _ in entries]
    n = len(entries)
    zero_expected = list(range(0, n))
    one_expected = list(range(1, n + 1))
    if token_indices == zero_expected:
        return entries
    if token_indices == one_expected:
        return [(idx - 1, p) for idx, p in entries]
    raise ValueError(
        f"Token directory indices for {kind} must be contiguous and either zero-based {zero_expected} "
        f"or one-based {one_expected}; got {token_indices}."
    )


def _load_probe_direction(probe_file: Path) -> np.ndarray:
    with open(probe_file, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Probe payload at {probe_file} must contain 'model'.")
    model = payload["model"]
    if not hasattr(model, "coef_"):
        raise ValueError(f"Probe model at {probe_file} has no coef_ attribute.")
    theta = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    norm_sq = float(np.dot(theta, theta))
    if norm_sq <= 0.0:
        raise ValueError(f"Probe weight at {probe_file} has zero norm.")
    return (theta / norm_sq).astype(np.float32)


def load_verbalised_probe_source(path: Path, requested_layers: Sequence[int], source_label: str) -> SourceVectors:
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Verbalised probe path is not a directory: {path}")

    guess_entries: List[Tuple[int, Path]] = []
    prob_entries: List[Tuple[int, Path]] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_token_dir_name(child.name)
        if parsed is None:
            continue
        tok_idx, kind = parsed
        if kind == "guess":
            guess_entries.append((tok_idx, child))
        else:
            prob_entries.append((tok_idx, child))

    guess_dirs = _normalize_token_indices(guess_entries, "guess")
    prob_dirs = _normalize_token_indices(prob_entries, "probability")

    by_layer_guess: Dict[int, List[np.ndarray]] = {}
    by_layer_probability: Dict[int, List[np.ndarray]] = {}

    for layer_idx in requested_layers:
        layer_folder = f"layer_{layer_idx + 1}"
        guess_vectors: List[np.ndarray] = []
        for _, token_dir in guess_dirs:
            probe_file = token_dir / layer_folder / "verbalised_confidence_probe.pkl"
            if not probe_file.exists():
                guess_vectors = []
                break
            guess_vectors.append(_load_probe_direction(probe_file))
        if guess_vectors:
            by_layer_guess[layer_idx] = guess_vectors

        prob_vectors: List[np.ndarray] = []
        for _, token_dir in prob_dirs:
            probe_file = token_dir / layer_folder / "verbalised_confidence_probe.pkl"
            if not probe_file.exists():
                prob_vectors = []
                break
            prob_vectors.append(_load_probe_direction(probe_file))
        if prob_vectors:
            by_layer_probability[layer_idx] = prob_vectors

    return SourceVectors(
        source_label=source_label,
        kind="verbalised_probe",
        by_layer_guess=by_layer_guess,
        by_layer_probability=by_layer_probability,
    )


def _pairwise_source_names(sources: Sequence[str]) -> List[Tuple[str, str]]:
    return [(a, b) for a, b in itertools.combinations(sources, 2)]


def _compute_token_stats(
    per_source_vectors: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], Optional[float]], List[str]]:
    norms: Dict[str, float] = {}
    cosines: Dict[Tuple[str, str], Optional[float]] = {}
    notes: List[str] = []
    for name, vec in per_source_vectors.items():
        norms[name] = _norm(vec)
    for a, b in _pairwise_source_names(sorted(per_source_vectors.keys())):
        va = per_source_vectors[a]
        vb = per_source_vectors[b]
        if va.shape != vb.shape:
            notes.append(f"shape_mismatch: {a} {va.shape} vs {b} {vb.shape}")
            cosines[(a, b)] = None
            continue
        cosines[(a, b)] = _cosine(va, vb)
    return norms, cosines, notes


def _gather_token_vectors(
    sources: Sequence[SourceVectors],
    *,
    layer_idx: int,
    span_kind: str,  # guess|probability
    token_pos: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for source in sources:
        by_layer = source.by_layer_guess if span_kind == "guess" else source.by_layer_probability
        vecs = by_layer.get(layer_idx)
        if vecs is None:
            continue
        if token_pos < 0 or token_pos >= len(vecs):
            continue
        out[source.source_label] = vecs[token_pos]
    return out


def _max_token_count_for_layer(sources: Sequence[SourceVectors], layer_idx: int, span_kind: str) -> int:
    max_n = 0
    for source in sources:
        by_layer = source.by_layer_guess if span_kind == "guess" else source.by_layer_probability
        vecs = by_layer.get(layer_idx)
        if vecs is not None:
            max_n = max(max_n, len(vecs))
    return max_n


def write_results(
    path: Path,
    *,
    requested_layers: Sequence[int],
    sources: Sequence[SourceVectors],
) -> None:
    lines: List[str] = []
    lines.append("Direction Comparison Results")
    lines.append("===========================")
    lines.append("")

    for layer_idx in requested_layers:
        lines.append(f"[Layer {layer_idx}]")
        for span_kind in ("guess", "probability"):
            pretty = "Guess" if span_kind == "guess" else "Probability"
            lines.append(f"{pretty} span:")
            max_tokens = _max_token_count_for_layer(sources, layer_idx, span_kind)
            if max_tokens == 0:
                lines.append("  no vectors available for this layer/span")
                continue
            for tok in range(max_tokens):
                token_vectors = _gather_token_vectors(
                    sources,
                    layer_idx=layer_idx,
                    span_kind=span_kind,
                    token_pos=tok,
                )
                lines.append(f"  token_{tok}:")
                if len(token_vectors) < 1:
                    lines.append("    no source vectors available")
                    continue
                norms, cosines, notes = _compute_token_stats(token_vectors)
                lines.append("    norms:")
                for source_name in sorted(norms.keys()):
                    lines.append(f"      {source_name}: {norms[source_name]:.8f}")
                lines.append("    cosine_similarities:")
                if cosines:
                    for (a, b) in sorted(cosines.keys()):
                        val = cosines[(a, b)]
                        if val is None:
                            lines.append(f"      {a}__vs__{b}: None")
                        else:
                            lines.append(f"      {a}__vs__{b}: {val:.8f}")
                else:
                    lines.append("      not enough sources for pairwise cosine")
                if notes:
                    lines.append("    notes:")
                    for note in notes:
                        lines.append(f"      {note}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_config(
    path: Path,
    *,
    args: argparse.Namespace,
    requested_layers: Sequence[int],
    sources: Sequence[SourceVectors],
    finished_at: str,
) -> None:
    lines: List[str] = []
    lines.extend(
        [
            "Direction Comparison Configuration",
            "=================================",
            "",
            "[Inputs]",
            f"mass_mean_dirs={args.mass_mean_dirs}",
            f"verbalised_probe_dir={args.verbalised_probe_dir}",
            "",
            "[Layer Spec]",
            f"layers_spec={args.layers}",
            f"layers_resolved={','.join(str(x) for x in requested_layers)}",
            "mass_mean_layer_indexing=zero_indexed",
            "verbalised_probe_layer_folder_indexing=one_indexed_layer_<idx_plus_1>",
            "",
            "[Direction Rules]",
            "mass_mean_direction_source=layer_<idx>_directions.pkl",
            "verbalised_probe_direction_source=verbalised_confidence_probe.pkl:model.coef_",
            "verbalised_probe_direction_normalization=theta_div_norm_theta_squared",
            "",
            "[Sources]",
        ]
    )
    for source in sources:
        guess_layers = sorted(source.by_layer_guess.keys())
        prob_layers = sorted(source.by_layer_probability.keys())
        lines.append(f"{source.source_label}.kind={source.kind}")
        lines.append(f"{source.source_label}.guess_layers={guess_layers}")
        lines.append(f"{source.source_label}.probability_layers={prob_layers}")
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direction vectors across mass-mean and verbalised-probe sources.")
    parser.add_argument(
        "--mass_mean_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing layer_<idx>_directions.pkl files.",
    )
    parser.add_argument(
        "--verbalised_probe_dir",
        type=str,
        required=True,
        help="Directory containing tok_<n>_guess / tok_<n>_probability(or prob) with layer_<k>/verbalised_confidence_probe.pkl.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="12-15",
        help="Inclusive range '12-15' or comma list '12,13,14,15' (zero-indexed).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to direction_comparison/results/<incrementing_run_id>/",
    )
    args = parser.parse_args()

    requested_layers = parse_layers_spec(args.layers)
    out_dir = resolve_output_dir(args.output_dir)

    sources: List[SourceVectors] = []
    for i, mass_dir in enumerate(args.mass_mean_dirs, start=1):
        label = f"mass_mean_{i}"
        sources.append(load_mass_mean_source(Path(mass_dir), requested_layers, source_label=label))
    sources.append(
        load_verbalised_probe_source(
            Path(args.verbalised_probe_dir),
            requested_layers=requested_layers,
            source_label="verbalised_probe",
        )
    )

    results_path = out_dir / "results.txt"
    config_path = out_dir / "config.txt"
    write_results(results_path, requested_layers=requested_layers, sources=sources)
    write_config(
        config_path,
        args=args,
        requested_layers=requested_layers,
        sources=sources,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
