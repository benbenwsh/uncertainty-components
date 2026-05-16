#!/usr/bin/env python3
"""
Compare direction vectors across mass-mean runs and verbalised probe outputs.

Inputs:
  - One or more mass-mean run directories containing layer_<idx>_directions.pkl
  - One verbalised-confidence probe directory containing:
      tok_<n>_guess/layer_<k>/verbalised_confidence_probe.pkl
      tok_<n>_probability|prob/layer_<k>/verbalised_confidence_probe.pkl

Outputs:
  - results.json: per-layer/per-token cosine similarities and norms
  - norms_plot.png: line plot of norms
  - cosine_plot.png: line plot of cosine similarities
  - config.txt: experiment setup metadata
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
import itertools
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def build_results_payload(
    *,
    requested_layers: Sequence[int],
    sources: Sequence[SourceVectors],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    for layer_idx in requested_layers:
        layer_key = str(layer_idx)
        layer_obj: Dict[str, Any] = {"guess_span": {}, "probability_span": {}}
        for span_kind in ("guess", "probability"):
            span_key = "guess_span" if span_kind == "guess" else "probability_span"
            span_obj: Dict[str, Any] = {}
            max_tokens = _max_token_count_for_layer(sources, layer_idx, span_kind)
            for tok in range(max_tokens):
                token_vectors = _gather_token_vectors(
                    sources,
                    layer_idx=layer_idx,
                    span_kind=span_kind,
                    token_pos=tok,
                )
                token_key = f"token_{tok}"
                if token_vectors:
                    norms, cosines, notes = _compute_token_stats(token_vectors)
                else:
                    norms, cosines, notes = {}, {}, []
                cosine_obj = {f"{a}__vs__{b}": val for (a, b), val in sorted(cosines.items())}
                token_obj: Dict[str, Any] = {
                    "norms": {name: float(norms[name]) for name in sorted(norms.keys())},
                    "cosine_similarities": cosine_obj,
                }
                if notes:
                    token_obj["notes"] = notes
                span_obj[token_key] = token_obj
            layer_obj[span_key] = span_obj
        payload[layer_key] = layer_obj

    return payload


def write_results_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _token_index(token_key: str) -> int:
    match = re.fullmatch(r"token_(\d+)", token_key)
    if match is None:
        raise ValueError(f"Unexpected token key format: {token_key}")
    return int(match.group(1))


def _layer_style(layer_idx: int, sorted_layers: Sequence[int]) -> float:
    if len(sorted_layers) <= 1:
        return 1.5
    pos = sorted_layers.index(layer_idx)
    frac = pos / float(len(sorted_layers) - 1)
    # Smaller layer index -> thinner.
    linewidth = 0.5 + 4.5 * frac
    return linewidth


def _source_kind(series_name: str) -> str:
    return "mass_mean" if series_name.startswith("mass_mean_") else "verbalised_probe"


def _cosine_pair_kind(pair_name: str) -> str:
    parts = pair_name.split("__vs__")
    if len(parts) != 2:
        return "mixed_pair"
    a, b = parts
    a_mass = a.startswith("mass_mean_")
    b_mass = b.startswith("mass_mean_")
    if a_mass and b_mass:
        return "mass_pair"
    if (not a_mass) and (not b_mass):
        return "probe_pair"
    return "mixed_pair"


def _collect_series_names(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    norm_series: set[str] = set()
    cosine_series: set[str] = set()
    for layer_obj in data.values():
        for span_key in ("guess_span", "probability_span"):
            span_obj = layer_obj.get(span_key, {})
            for token_obj in span_obj.values():
                norm_series.update(token_obj.get("norms", {}).keys())
                cosine_series.update(token_obj.get("cosine_similarities", {}).keys())
    return sorted(norm_series), sorted(cosine_series)


def _collect_cosine_pair_names(data: Dict[str, Any]) -> List[str]:
    _, cosine_series = _collect_series_names(data)
    return sorted(cosine_series)


def _build_distinct_color_map(keys: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    if not keys:
        return {}
    cmap = plt.get_cmap("tab20")
    n = len(keys)
    out: Dict[str, Tuple[float, float, float]] = {}
    for idx, key in enumerate(keys):
        if n <= 20:
            rgba = cmap(idx / max(1, n - 1))
        else:
            rgba = plt.get_cmap("hsv")(idx / n)
        out[key] = mcolors.to_rgb(rgba)
    return out


def _split_pair_name(pair_name: str) -> Tuple[Optional[str], Optional[str]]:
    parts = pair_name.split("__vs__")
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _resolve_direction_color_map(data: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    norm_series, cosine_series = _collect_series_names(data)
    direction_names: set[str] = set(norm_series)
    for pair_name in cosine_series:
        a, b = _split_pair_name(pair_name)
        if a is not None:
            direction_names.add(a)
        if b is not None:
            direction_names.add(b)
    direction_color_map = _build_distinct_color_map(sorted(direction_names))
    fixed_colors = {
        "mass_mean_1": mcolors.to_rgb("red"),
        "mass_mean_2": mcolors.to_rgb("orange"),
        "verbalised_probe": mcolors.to_rgb("green"),
    }
    for name, color in fixed_colors.items():
        if name in direction_color_map:
            direction_color_map[name] = color
    return direction_color_map


def _cosine_pair_color(
    pair_name: str,
    direction_color_map: Dict[str, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    a, b = _split_pair_name(pair_name)
    if a is None or b is None:
        return mcolors.to_rgb("black")
    a_color = direction_color_map.get(a)
    b_color = direction_color_map.get(b)
    if a_color is None and b_color is None:
        return mcolors.to_rgb("black")
    if a_color is None:
        return b_color  # type: ignore[return-value]
    if b_color is None:
        return a_color
    # Use the first direction's color for deterministic pair coloring.
    return a_color


def _cosine_marker(pair_name: str) -> str:
    pair_kind = _cosine_pair_kind(pair_name)
    if pair_kind == "mass_pair":
        return "o"
    return "s"


def _apply_compact_legend(
    ax: Any,
    *,
    series_color_map: Dict[str, Tuple[float, float, float]],
    layers: Sequence[int],
    color_title: str = "Direction colors",
) -> None:
    span_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Guess span"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.8, label="Probability span (dotted)"),
    ]
    color_handles = [
        Line2D([0], [0], color=series_color_map[name], linestyle="-", linewidth=2.0, label=name)
        for name in sorted(series_color_map.keys())
    ]
    layer_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=_layer_style(layer_idx, layers),
            label=f"Layer {layer_idx}",
        )
        for layer_idx in sorted(layers)
    ]

    legend_span = ax.legend(handles=span_handles, loc="upper left", title="Span style", fontsize=8, title_fontsize=9)
    ax.add_artist(legend_span)
    legend_color = ax.legend(
        handles=color_handles,
        loc="upper center",
        title=color_title,
        fontsize=8,
        title_fontsize=9,
        ncol=2,
    )
    ax.add_artist(legend_color)
    ax.legend(handles=layer_handles, loc="upper right", title="Layer thickness", fontsize=8, title_fontsize=9, ncol=1)


def plot_norms_from_json(json_path: Path, output_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not data:
        return
    layers = sorted(int(k) for k in data.keys())
    fig, ax = plt.subplots(figsize=(12, 7))
    direction_color_map = _resolve_direction_color_map(data)
    markers = {"mass_mean": "o", "verbalised_probe": "s"}
    linestyles = {"guess_span": "-", "probability_span": ":"}

    for layer_idx in layers:
        layer_key = str(layer_idx)
        layer_obj = data.get(layer_key, {})
        guess_tokens = layer_obj.get("guess_span", {})
        prob_tokens = layer_obj.get("probability_span", {})
        guess_max = max((_token_index(tk) for tk in guess_tokens.keys()), default=-1)
        prob_max = max((_token_index(tk) for tk in prob_tokens.keys()), default=-1)
        max_token = max(guess_max, prob_max)
        if max_token < 0:
            continue
        x_positions = list(range(max_token + 1))
        linewidth = _layer_style(layer_idx, layers)

        for span_key, linestyle in linestyles.items():
            span_obj = layer_obj.get(span_key, {})
            source_names = sorted(
                {
                    source_name
                    for token_obj in span_obj.values()
                    for source_name in token_obj.get("norms", {}).keys()
                }
            )
            for source_name in source_names:
                y_values: List[float] = []
                for tok in x_positions:
                    token_obj = span_obj.get(f"token_{tok}", {})
                    val = token_obj.get("norms", {}).get(source_name)
                    y_values.append(np.nan if val is None else float(val))
                source_kind = _source_kind(source_name)
                ax.plot(
                    x_positions,
                    y_values,
                    label="_nolegend_",
                    color=direction_color_map[source_name],
                    marker=markers[source_kind],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    markersize=4.5,
                )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Norm")
    ax.set_title("Direction Norms by Layer and Span")
    ax.grid(True, alpha=0.25)
    _apply_compact_legend(ax, series_color_map=direction_color_map, layers=layers)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cosines_from_json(json_path: Path, output_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not data:
        return
    layers = sorted(int(k) for k in data.keys())
    fig, ax = plt.subplots(figsize=(12, 7))
    direction_color_map = _resolve_direction_color_map(data)
    cosine_pair_names = _collect_cosine_pair_names(data)
    cosine_legend_color_map = {
        pair_name: _cosine_pair_color(pair_name, direction_color_map) for pair_name in cosine_pair_names
    }
    # Pin the first three comparison series to blue/orange/purple.
    preferred_pairs = [
        "mass_mean_1__vs__mass_mean_2",
        "mass_mean_1__vs__verbalised_probe",
        "mass_mean_2__vs__verbalised_probe",
    ]
    fixed_pair_colors = [mcolors.to_rgb("blue"), mcolors.to_rgb("orange"), mcolors.to_rgb("purple")]
    assigned: set[str] = set()
    for pair_name, color in zip(preferred_pairs, fixed_pair_colors):
        if pair_name in cosine_legend_color_map:
            cosine_legend_color_map[pair_name] = color
            assigned.add(pair_name)
    for pair_name in cosine_pair_names:
        if pair_name in assigned:
            continue
        if len(assigned) < 3:
            cosine_legend_color_map[pair_name] = fixed_pair_colors[len(assigned)]
            assigned.add(pair_name)
    linestyles = {"guess_span": "-", "probability_span": ":"}

    for layer_idx in layers:
        layer_key = str(layer_idx)
        layer_obj = data.get(layer_key, {})
        guess_tokens = layer_obj.get("guess_span", {})
        prob_tokens = layer_obj.get("probability_span", {})
        guess_max = max((_token_index(tk) for tk in guess_tokens.keys()), default=-1)
        prob_max = max((_token_index(tk) for tk in prob_tokens.keys()), default=-1)
        max_token = max(guess_max, prob_max)
        if max_token < 0:
            continue
        x_positions = list(range(max_token + 1))
        linewidth = _layer_style(layer_idx, layers)

        for span_key, linestyle in linestyles.items():
            span_obj = layer_obj.get(span_key, {})
            pair_names = sorted(
                {
                    pair_name
                    for token_obj in span_obj.values()
                    for pair_name in token_obj.get("cosine_similarities", {}).keys()
                }
            )
            for pair_name in pair_names:
                y_values: List[float] = []
                for tok in x_positions:
                    token_obj = span_obj.get(f"token_{tok}", {})
                    val = token_obj.get("cosine_similarities", {}).get(pair_name)
                    y_values.append(np.nan if val is None else float(val))
                ax.plot(
                    x_positions,
                    y_values,
                    label="_nolegend_",
                    color=cosine_legend_color_map[pair_name],
                    marker=_cosine_marker(pair_name),
                    linewidth=linewidth,
                    linestyle=linestyle,
                    markersize=4.5,
                )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Direction Cosine Similarities by Layer and Span")
    ax.grid(True, alpha=0.25)
    _apply_compact_legend(
        ax,
        series_color_map=cosine_legend_color_map,
        layers=layers,
        color_title="Direction comparison colors",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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

    results_path = out_dir / "results.json"
    norms_plot_path = out_dir / "norms_plot.png"
    cosine_plot_path = out_dir / "cosine_plot.png"
    config_path = out_dir / "config.txt"
    results_payload = build_results_payload(requested_layers=requested_layers, sources=sources)
    write_results_json(results_path, results_payload)
    plot_norms_from_json(results_path, norms_plot_path)
    plot_cosines_from_json(results_path, cosine_plot_path)
    write_config(
        config_path,
        args=args,
        requested_layers=requested_layers,
        sources=sources,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {norms_plot_path}")
    print(f"Wrote {cosine_plot_path}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
