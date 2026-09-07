"""Confidence-vs-alpha line plots for mass_mean_probe."""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt

_TICK_FS = 14
_LABEL_FS = 18
_LEGEND_FS = 14
_TARGET_COLORS = {"high": "tab:blue", "low": "tab:orange"}
_TARGET_LEGEND = {"high": "high_to_low", "low": "low_to_high"}


def plot_mode_confidence_from_summary(
    *,
    mode_name: str,
    mode_payload: Dict[str, Dict[str, Dict[str, object]]],
    ablation_targets: Sequence[str],
    output_path: str,
    low_conf_threshold: float,
    high_conf_threshold: float,
) -> None:
    del mode_name
    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    for target in ("high", "low"):
        if target not in ablation_targets:
            continue
        alpha_map = mode_payload.get(target, {})
        if not isinstance(alpha_map, dict):
            continue
        points: list[Tuple[float, float, int]] = []
        for metrics in alpha_map.values():
            if not isinstance(metrics, dict):
                continue
            mean_conf = metrics.get("mean_confidence")
            alpha_val = metrics.get("alpha_value")
            sample_count = metrics.get("sample_count", 0)
            if mean_conf is None or alpha_val is None:
                continue
            points.append((float(alpha_val), float(mean_conf), int(sample_count)))
        points.sort(key=lambda tup: tup[0])
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                color=_TARGET_COLORS[target],
                label=_TARGET_LEGEND[target],
            )
            for x_val, y_val, sample_count in points:
                ax.annotate(
                    str(sample_count),
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=_TICK_FS,
                    color=_TARGET_COLORS[target],
                )

    ax.axhline(
        y=float(high_conf_threshold),
        color=_TARGET_COLORS["high"],
        linestyle=":",
        linewidth=1.2,
        label="threshold_high",
    )
    ax.axhline(
        y=float(low_conf_threshold),
        color=_TARGET_COLORS["low"],
        linestyle=":",
        linewidth=1.2,
        label="threshold_low",
    )

    ax.set_xlabel("Alpha", fontsize=_LABEL_FS)
    ax.set_ylabel("Verbalised confidence", fontsize=_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=_LEGEND_FS)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)
