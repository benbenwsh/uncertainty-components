#!/usr/bin/env python3
"""Rank trivia-Gemma subblock change scores for tokens 0, 1, and 6.

Reads existing individual-layer summaries (does not modify run outputs) and
writes a ranked list plus a unique top-10 component string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_RESULTS = SCRIPT_DIR / "results" / "individual_layers" / "gemma"
OUTPUT_DIR = GEMMA_RESULTS / "change_score_analysis"

TOKEN_POSITIONS = (0, 1, 6)

RUN_PAIRS = (
    {
        "name": "both",
        "low_to_high": "3_trivia_gemma_both_low_to_high",
        "high_to_low": "2_trivia_gemma_both_high_to_low",
        "average": True,
        "label": lambda layer: f"a{layer}m{layer}",
    },
    {
        "name": "attn",
        "low_to_high": "4_gemma_attn_low_to_high",
        "high_to_low": "5_gemma_attn_high_to_low",
        "average": False,
        "label": lambda layer: f"a{layer}",
    },
    {
        "name": "mlp",
        "low_to_high": "7_trivia_gemma_mlp_low_to_high",
        "high_to_low": "6_gemma_mlp_high_to_low",
        "average": False,
        "label": lambda layer: f"m{layer}",
    },
)


def _load_summary(run_dir_name: str) -> dict:
    path = GEMMA_RESULTS / run_dir_name / "summary.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _steered_mean(summary: dict, layer: int, token_pos: int) -> float:
    return float(summary["layer_token_confidence"][str(layer)][str(token_pos)])


def _baseline(summary: dict) -> float:
    return float(summary["baseline"]["mean_confidence"])


def _normalized_up_score(summary: dict, layer: int, token_pos: int) -> float:
    baseline = _baseline(summary)
    steered = _steered_mean(summary, layer, token_pos)
    return (steered - baseline) / (1.0 - baseline)


def _normalized_down_score(summary: dict, layer: int, token_pos: int) -> float:
    baseline = _baseline(summary)
    steered = _steered_mean(summary, layer, token_pos)
    return (baseline - steered) / baseline


def _layers(summary: dict) -> List[int]:
    return [int(layer) for layer in summary["run_layers"]]


def compute_ranked_rows() -> List[Tuple[float, int, str]]:
    rows: List[Tuple[float, int, str]] = []
    for spec in RUN_PAIRS:
        low_summary = _load_summary(spec["low_to_high"])
        high_summary = _load_summary(spec["high_to_low"])
        layers = _layers(low_summary)
        if layers != _layers(high_summary):
            raise ValueError(
                f"Layer mismatch for {spec['name']}: "
                f"{layers} vs {_layers(high_summary)}"
            )
        for layer in layers:
            component = spec["label"](layer)
            for token_pos in TOKEN_POSITIONS:
                s_up = _normalized_up_score(low_summary, layer, token_pos)
                s_down = _normalized_down_score(high_summary, layer, token_pos)
                combined = s_up + s_down
                if spec["average"]:
                    combined /= 2.0
                rows.append((combined, token_pos, component))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    return rows


def unique_top_components(rows: Sequence[Tuple[float, int, str]], k: int = 10) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for _, _, component in rows:
        if component in seen:
            continue
        seen.add(component)
        ordered.append(component)
        if len(ordered) >= k:
            break
    return ordered


def write_outputs(rows: Sequence[Tuple[float, int, str]], top_components: Sequence[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked_path = OUTPUT_DIR / "ranked_change_scores.txt"
    top10_path = OUTPUT_DIR / "top10_components.txt"

    with ranked_path.open("w", encoding="utf-8") as f:
        for score, token_pos, component in rows:
            f.write(f"{score:.6f}  token={token_pos}  {component}\n")

    with top10_path.open("w", encoding="utf-8") as f:
        f.write(",".join(top_components) + "\n")

    print(f"Wrote {ranked_path} ({len(rows)} rows)")
    print(f"Wrote {top10_path}: {','.join(top_components)}")


def main() -> None:
    rows = compute_ranked_rows()
    top_components = unique_top_components(rows)
    write_outputs(rows, top_components)


if __name__ == "__main__":
    main()
