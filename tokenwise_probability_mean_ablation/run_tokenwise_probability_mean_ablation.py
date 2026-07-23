#!/usr/bin/env python3
"""Tokenwise probability-span mean ablation runner.

This script mirrors the layerwise mean-ablation setup but removes ablation modes and
always performs:
1) baseline no-replacement generation, and
2) seven independent single-token probability-span replacements (positions 0..6).

Outputs are intentionally compact:
- ablation_results_mini.json
- config.txt
- summary.json
- plot PNG(s)
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layerwise_mean_ablation.run_mean_ablation import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT,
    _build_resid_post_mean_replace_hooks,
    _completion_token_index_to_abs_pos,
    _greedy_extend_with_fwd_hooks,
    collect_confidence_group_ids,
    compute_verbalised_embedding_group_means,
    configure_prefix_tokens_for_model,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    greedy_generate,
    load_examples_h5,
    load_hooked_transformer,
    load_trivia_qa,
    parse_ablate_layers,
    parse_guess_and_probability_indices,
    parse_mode_confidence_from_response,
    split_answerable_indices,
)


TRAIN_RATIO = 0.9


def _token_mode_key(token_position: int) -> str:
    return f"probability_token_{token_position}_mean_replace"


def _resolve_run_root(cli_output_dir: Optional[str], *, individual_layers: bool) -> str:
    if cli_output_dir:
        os.makedirs(cli_output_dir, exist_ok=True)
        return cli_output_dir

    if individual_layers:
        base_dir = os.path.join("tokenwise_probability_mean_ablation", "results", "individual_layers")
    else:
        base_dir = os.path.join("tokenwise_probability_mean_ablation", "results")
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_root = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_root, exist_ok=True)
    return run_root


def _mini_output_json_path(run_root: str) -> str:
    return os.path.join(run_root, "ablation_results_mini.json")


def _config_txt_path(run_root: str) -> str:
    return os.path.join(run_root, "config.txt")


def _summary_json_path(run_root: str) -> str:
    return os.path.join(run_root, "summary.json")


def _line_plot_path(run_root: str) -> str:
    return os.path.join(run_root, "verbalised_confidence_by_probability_token.png")


def _grid_plot_path(run_root: str) -> str:
    return os.path.join(run_root, "layer_token_deviation_grid.png")


def _render_token_label(token: str) -> str:
    escaped = token.encode("unicode_escape").decode("ascii")
    return escaped if escaped else "<empty>"


def _extract_probability_tokens(
    decoded_tokens: Sequence[str],
    *,
    expected_probability_tokens: int,
) -> Optional[List[str]]:
    parsed = parse_guess_and_probability_indices(list(decoded_tokens))
    if parsed is None:
        return None
    _, first_prob, end_prob = parsed
    span = list(decoded_tokens[first_prob : end_prob + 1])
    if len(span) != expected_probability_tokens:
        return None
    return span


def _absolute_prob_single_position(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    token_position: int,
    expected_probability_tokens: int,
) -> List[int]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    if (end_prob - first_prob + 1) != expected_probability_tokens:
        return []
    target_rel_pos = first_prob + token_position
    seq_len = prompt_len + len(decoded_tokens)
    abs_pos = _completion_token_index_to_abs_pos(prompt_len, target_rel_pos)
    if 0 <= abs_pos < seq_len:
        return [abs_pos]
    return []


def build_single_token_probability_hooks(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    *,
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
    token_position: int,
    expected_probability_tokens: int,
    log_context: str = "",
) -> List[Tuple[str, Callable]]:
    _last_logged_key: Dict[str, object | None] = {"value": None}

    def _abs_positions() -> List[int]:
        decoded_tokens = decoded_tokens_provider()
        positions = _absolute_prob_single_position(
            prompt_len,
            decoded_tokens,
            token_position=token_position,
            expected_probability_tokens=expected_probability_tokens,
        )
        if positions:
            seq_len = seq_len_provider()
            log_key = (seq_len, tuple(positions))
            if _last_logged_key["value"] != log_key:
                _last_logged_key["value"] = log_key
                prefix = f"{log_context} " if log_context else ""
                rendered_tokens = [_render_token_label(token) for token in decoded_tokens]
                logging.info(
                    "%sAblation forward pass (parse ok): ablation_positions=%s prompt_len=%d seq_len=%d decoded_tokens=%s",
                    prefix,
                    positions,
                    prompt_len,
                    seq_len,
                    rendered_tokens,
                )
        return positions

    return _build_resid_post_mean_replace_hooks(
        layer_to_mean_vectors,
        seq_len_provider=seq_len_provider,
        abs_positions_provider=_abs_positions,
        strict_num_prob_positions=True,
    )


def greedy_generate_probability_single_token_mean_replaced(
    model,
    *,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    token_position: int,
    expected_probability_tokens: int,
    log_context: str = "",
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _seq_len() -> int:
        return int(tokens.shape[1])

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_single_token_probability_hooks(
        layer_to_mean_vectors,
        prompt_len=prompt_len,
        seq_len_provider=_seq_len,
        decoded_tokens_provider=_decoded_tokens,
        token_position=token_position,
        expected_probability_tokens=expected_probability_tokens,
        log_context=log_context,
    )
    return _greedy_extend_with_fwd_hooks(
        model,
        local_prompt,
        max_new_tokens,
        tokens,
        decoded_tokens,
        hooks,
    )


def _compute_selected_ids(
    eval_ds,
    ablation_target_ids: Sequence[str],
    split_name: str,
    num_samples: int,
) -> Tuple[Dict[str, int], List[str]]:
    split_target = (
        round(num_samples * TRAIN_RATIO)
        if split_name == "train"
        else round(num_samples * (1 - TRAIN_RATIO))
    )
    id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
    split_target_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)
    selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
    return id_to_index, selected_ids


def build_none_cache(
    *,
    train_ds,
    val_ds,
    model,
    ablation_target_ids: Sequence[str],
    num_samples: int,
    fewshot_prefix: str,
    max_new_tokens: int,
    parse_mode_verbalised_confidence: bool,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}
    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        id_to_index, selected_ids = _compute_selected_ids(
            eval_ds, ablation_target_ids, split_name, num_samples
        )
        for ex_id in selected_ids:
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                continue
            example = eval_ds[int(ds_idx)]
            question = example["question"]
            local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + question
            response, decoded_tokens = greedy_generate(
                model=model,
                local_prompt=local_prompt,
                max_new_tokens=max_new_tokens,
                fwd_hooks=None,
            )
            mode_confidence = (
                parse_mode_confidence_from_response(response)
                if parse_mode_verbalised_confidence
                else None
            )
            none_cache[split_name][ex_id] = {
                "response": response,
                "decoded_tokens": decoded_tokens,
                "verbalised_confidence": mode_confidence,
            }
    return none_cache


def _baseline_confidence_stats(
    none_cache: Dict[str, Dict[str, Dict[str, object]]],
) -> Tuple[Optional[float], int]:
    values: List[float] = []
    for split_name in ("train", "validation"):
        for record in none_cache[split_name].values():
            conf = record.get("verbalised_confidence")
            if conf is not None:
                values.append(float(conf))
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def run_tokenwise_evaluation(
    *,
    train_ds,
    val_ds,
    model,
    fewshot_prefix: str,
    layer_to_verbalised_embedding_means_eval: Dict[int, Dict[str, torch.Tensor]],
    none_cache: Dict[str, Dict[str, Dict[str, object]]],
    ablation_target_ids: Sequence[str],
    num_samples: int,
    max_new_tokens: int,
    parse_mode_verbalised_confidence: bool,
    expected_probability_tokens: int,
    mean_from_low_confidence: bool,
) -> Tuple[
    Dict[str, dict],
    Dict[int, Optional[float]],
    Dict[int, int],
    Dict[int, int],
    Dict[int, Counter[str]],
]:
    mini_results = {"train": {}, "validation": {}}
    token_position_values: Dict[int, List[float]] = {
        i: [] for i in range(expected_probability_tokens)
    }
    token_position_counts: Dict[int, int] = {i: 0 for i in range(expected_probability_tokens)}
    responses_identical_true: Dict[int, int] = {i: 0 for i in range(expected_probability_tokens)}
    token_label_counters: Dict[int, Counter[str]] = {
        i: Counter() for i in range(expected_probability_tokens)
    }

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        id_to_index, selected_ids = _compute_selected_ids(
            eval_ds, ablation_target_ids, split_name, num_samples
        )
        logging.info(
            "Evaluating %d examples on %s split for tokenwise ablation.",
            len(selected_ids),
            split_name,
        )
        for i, ex_id in enumerate(selected_ids):
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                raise ValueError(f"Example id {ex_id} missing from {split_name} split.")
            example = eval_ds[int(ds_idx)]
            question = example["question"]
            local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + question
            mini_entry = {"question": question}

            cached = none_cache[split_name].get(ex_id)
            if cached is None:
                raise ValueError(
                    f"Missing baseline cache for split={split_name}, ex_id={ex_id}."
                )
            baseline_response = str(cached["response"])
            baseline_decoded_tokens = list(cached["decoded_tokens"])
            baseline_confidence = (
                cached.get("verbalised_confidence")
                if parse_mode_verbalised_confidence
                else None
            )
            mini_entry["no_replacement"] = {"response": baseline_response}
            if parse_mode_verbalised_confidence:
                mini_entry["no_replacement"]["verbalised_confidence"] = baseline_confidence

            prob_tokens = _extract_probability_tokens(
                baseline_decoded_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if prob_tokens is not None:
                for pos, token in enumerate(prob_tokens):
                    token_label_counters[pos][token] += 1

            for token_position in range(expected_probability_tokens):
                key = _token_mode_key(token_position)
                layer_to_mean_vectors_eval = {
                    layer_idx: layer_to_verbalised_embedding_means_eval[layer_idx]["probability"][
                        token_position : token_position + 1, :
                    ]
                    for layer_idx in layer_to_verbalised_embedding_means_eval
                }
                response, _decoded_tokens = (
                    greedy_generate_probability_single_token_mean_replaced(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=max_new_tokens,
                        layer_to_mean_vectors=layer_to_mean_vectors_eval,
                        token_position=token_position,
                        expected_probability_tokens=expected_probability_tokens,
                        log_context=f"{split_name} {ex_id} {_token_mode_key(token_position)}",
                    )
                )
                confidence = (
                    parse_mode_confidence_from_response(response)
                    if parse_mode_verbalised_confidence
                    else None
                )

                item = {"response": response}
                if parse_mode_verbalised_confidence:
                    item["verbalised_confidence"] = confidence
                responses_identical = response == baseline_response
                item["responses_identical"] = responses_identical
                if responses_identical:
                    responses_identical_true[token_position] += 1

                if parse_mode_verbalised_confidence:
                    if confidence is None or baseline_confidence is None:
                        meets_baseline_direction = None
                    elif mean_from_low_confidence:
                        meets_baseline_direction = confidence < baseline_confidence
                    else:
                        meets_baseline_direction = confidence > baseline_confidence
                    item["meets_none_confidence_direction"] = meets_baseline_direction
                    if confidence is not None:
                        token_position_values[token_position].append(float(confidence))

                token_position_counts[token_position] += 1
                mini_entry[key] = item

            mini_results[split_name][ex_id] = mini_entry
            logging.info(
                "[%s %d/%d] %s complete",
                split_name,
                i + 1,
                len(selected_ids),
                ex_id,
            )

    token_position_means: Dict[int, Optional[float]] = {}
    for token_position in range(expected_probability_tokens):
        values = token_position_values[token_position]
        token_position_means[token_position] = (
            float(np.mean(values)) if values else None
        )

    return (
        mini_results,
        token_position_means,
        token_position_counts,
        responses_identical_true,
        token_label_counters,
    )


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    run_layers: Sequence[int],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    baseline_mean: Optional[float],
    baseline_count: int,
    token_position_means: Dict[int, Optional[float]],
    token_position_counts: Dict[int, int],
    responses_identical_true: Dict[int, int],
    finished_at: str,
) -> None:
    lines = [
        "Tokenwise Probability Mean Ablation Config",
        "=========================================",
        "",
        "[Run]",
        f"finished_at={finished_at}",
        f"model_name={args.model_name}",
        f"input_h5={args.input_h5}",
        f"new_h5_format={args.new_h5_format}",
        f"device={device}",
        f"dtype={args.dtype}",
        f"model_n_layers={model_n_layers}",
        f"run_layers={','.join(str(x) for x in run_layers)}",
        f"individual_layers={args.individual_layers}",
        "",
        "[Sampling]",
        f"random_seed={args.random_seed}",
        f"num_samples={args.num_samples}",
        f"train_ratio={TRAIN_RATIO}",
        f"num_few_shot={args.num_few_shot}",
        f"prompt_indices={','.join(str(i) for i in prompt_indices)}",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        "",
        "[Prompt]",
        f"brief_prompt={args.brief_prompt}",
        f"brief_always={args.brief_always}",
        f"enable_brief={args.enable_brief}",
        f"use_context={args.use_context}",
        "",
        "[Confidence groups]",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={'low_confidence' if args.mean_from_low_confidence else 'high_confidence'}",
        f"ablation_target_group={'high_confidence' if args.mean_from_low_confidence else 'low_confidence'}",
        f"low_conf_count={low_conf_count}",
        f"high_conf_count={high_conf_count}",
        f"h5_example_count={h5_example_count}",
        "",
        "[Tokenwise settings]",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"parse_mode_verbalised_confidence={args.parse_mode_verbalised_confidence}",
        "",
        "[Baseline]",
        f"no_replacement_mean_verbalised_confidence={baseline_mean}",
        f"no_replacement_sample_count={baseline_count}",
        "",
        "[Per-position means]",
    ]

    for pos in range(args.expected_probability_tokens):
        lines.append(
            "position_"
            f"{pos}: mean={token_position_means.get(pos)} "
            f"count={token_position_counts.get(pos, 0)} "
            f"responses_identical_true={responses_identical_true.get(pos, 0)}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _token_labels_from_counters(
    token_label_counters: Dict[int, Counter[str]],
    *,
    expected_probability_tokens: int,
) -> List[str]:
    labels: List[str] = []
    for pos in range(expected_probability_tokens):
        counter = token_label_counters[pos]
        if not counter:
            labels.append(f"pos_{pos}")
        else:
            token = counter.most_common(1)[0][0]
            labels.append(token)
    return labels


def write_line_plot(
    *,
    path: str,
    token_position_means: Dict[int, Optional[float]],
    baseline_mean: Optional[float],
    expected_probability_tokens: int,
    token_labels: Sequence[str],
) -> None:
    xs: List[int] = []
    ys: List[float] = []
    for pos in range(expected_probability_tokens):
        y_val = token_position_means.get(pos)
        if y_val is not None:
            xs.append(pos)
            ys.append(float(y_val))

    fig, ax = plt.subplots(figsize=(10, 5))
    if ys:
        ax.plot(xs, ys, marker="o", label="single_token_mean_replace")
    if baseline_mean is not None:
        ax.axhline(
            y=float(baseline_mean),
            linestyle=":",
            linewidth=1.6,
            label="no_replacement (baseline)",
        )
    ax.set_xlim(-0.3, max(expected_probability_tokens - 1, 0) + 0.3)
    ax.set_xlabel("Probability token position")
    ax.set_ylabel("Verbalised confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Tokenwise probability-span ablation")
    ax.set_xticks(list(range(expected_probability_tokens)))
    ax.set_xticklabels(
        [f"{i}:{_render_token_label(tok)}" for i, tok in enumerate(token_labels)],
        rotation=20,
        ha="right",
    )
    ax.grid(True, alpha=0.3)
    if ys or baseline_mean is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_layer_token_grid_plot(
    *,
    path: str,
    run_layers: Sequence[int],
    token_labels: Sequence[str],
    matrix_values: np.ndarray,  # [n_layers, n_tokens], may include nan
    matrix_deviation_abs: np.ndarray,  # [n_layers, n_tokens], nan-safe
) -> None:
    n_rows, n_cols = matrix_values.shape
    max_dev = float(np.nanmax(matrix_deviation_abs)) if np.isfinite(matrix_deviation_abs).any() else 0.0
    if max_dev > 0:
        alpha = np.clip(matrix_deviation_abs / max_dev, 0.0, 1.0)
    else:
        alpha = np.zeros_like(matrix_deviation_abs)
    alpha = np.nan_to_num(alpha, nan=0.0)

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.1
    rgba[:, :, 1] = 0.3
    rgba[:, :, 2] = 1.0
    rgba[:, :, 3] = alpha

    fig, ax = plt.subplots(figsize=(max(9.0, 1.4 * n_cols), max(6.0, 0.35 * n_rows)))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix_values[r, c]
            text = "NA" if np.isnan(val) else f"{val:.3f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=8, color="black")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(layer) for layer in run_layers])
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [f"{i}:{_render_token_label(tok)}" for i, tok in enumerate(token_labels)],
        rotation=30,
        ha="left",
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Probability token position")
    ax.set_ylabel("Layer")
    ax.set_title("Layer x token confidence (blue alpha = |deviation from baseline|)")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.35, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenwise probability mean replacement inference (TransformerLens)."
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, read 'res' from the new {res,attn,mlp} H5 response format.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_max_new_tokens", type=int, default=50)
    parser.add_argument("--brief_prompt", type=str, default="default", choices=["default", "chat"])
    parser.add_argument("--brief_always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_brief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ablate_layers",
        type=str,
        default="12-15",
        help="Inclusive range '12-15' or comma list '12,13,14,15' (0-indexed).",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--mean_from_low_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), compute means from low-confidence examples and ablate high-confidence examples. "
            "If false, reverse source and target groups."
        ),
    )
    parser.add_argument("--expected_probability_tokens", type=int, default=5)
    parser.add_argument("--expected_guess_tokens", type=int, default=2)
    parser.add_argument(
        "--parse_mode_verbalised_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse verbalised confidence from each generated response.",
    )
    parser.add_argument(
        "--individual_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, ignore --ablate_layers and run one tokenwise sweep per layer, "
            "then emit a layer x token grid."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional run directory. If unset, auto-creates under tokenwise_probability_mean_ablation/results.",
    )
    args = parser.parse_args()
    configure_prefix_tokens_for_model(args.model_name)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "Run parameters: model_name=%s input_h5=%s dtype=%s num_samples=%s "
        "ablate_layers=%s individual_layers=%s low_conf_threshold=%s high_conf_threshold=%s "
        "mean_from_low_confidence=%s expected_probability_tokens=%s expected_guess_tokens=%s "
        "new_h5_format=%s random_seed=%s output_dir=%s",
        args.model_name,
        args.input_h5,
        args.dtype,
        args.num_samples,
        args.ablate_layers,
        args.individual_layers,
        args.low_conf_threshold,
        args.high_conf_threshold,
        args.mean_from_low_confidence,
        args.expected_probability_tokens,
        args.expected_guess_tokens,
        args.new_h5_format,
        args.random_seed,
        args.output_dir,
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_trivia_qa(args.random_seed)
    random.seed(args.random_seed)
    answerable_train = split_answerable_indices(train_ds)
    if len(answerable_train) < args.num_few_shot:
        raise ValueError("Not enough answerable training examples for few-shot.")
    prompt_indices = random.sample(answerable_train, args.num_few_shot)

    brief = BRIEF_PROMPTS[args.brief_prompt]
    brief_always_effective = args.brief_always if args.enable_brief else True
    fewshot_prefix = construct_fewshot_prompt_from_indices(
        train_ds,
        prompt_indices,
        brief,
        brief_always=brief_always_effective,
        use_context=args.use_context,
    )

    logging.info("Loading HookedTransformer: %s", args.model_name)
    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    logging.info("Model loaded")
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers)
    run_layers = list(range(model.cfg.n_layers)) if args.individual_layers else ablate_layers

    logging.info("Loading examples: %s", args.input_h5)
    examples_h5 = load_examples_h5(Path(args.input_h5))
    logging.info("Examples loaded")
    verbalised_embedding_means, low_ids, high_ids = compute_verbalised_embedding_group_means(
        examples_h5,
        run_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_guess_tokens=args.expected_guess_tokens,
        mean_from_low_confidence=args.mean_from_low_confidence,
        new_h5_format=args.new_h5_format,
    )
    logging.info("Verbalised embedding means computed")
    if not args.parse_mode_verbalised_confidence:
        # Keep confidence-group checks and IDs consistent with the base implementation.
        low_ids, high_ids = collect_confidence_group_ids(
            examples_h5,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
        )

    mean_source_ids = low_ids if args.mean_from_low_confidence else high_ids
    ablation_target_ids = high_ids if args.mean_from_low_confidence else low_ids
    if not ablation_target_ids:
        raise ValueError("No ablation target IDs found for the configured thresholds.")
    if not mean_source_ids:
        raise ValueError("No mean source IDs found for the configured thresholds.")

    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]] = {}
    for i, layer_idx in enumerate(run_layers):
        layer_to_verbalised_embedding_means[layer_idx] = {
            "probability": torch.tensor(
                verbalised_embedding_means["probability"][i], device=device, dtype=torch_dtype
            )
        }

    run_root = _resolve_run_root(args.output_dir, individual_layers=args.individual_layers)
    logging.info("Run root: %s", run_root)

    none_cache = build_none_cache(
        train_ds=train_ds,
        val_ds=val_ds,
        model=model,
        ablation_target_ids=ablation_target_ids,
        num_samples=args.num_samples,
        fewshot_prefix=fewshot_prefix,
        max_new_tokens=args.model_max_new_tokens,
        parse_mode_verbalised_confidence=args.parse_mode_verbalised_confidence,
    )
    baseline_mean, baseline_count = _baseline_confidence_stats(none_cache)

    if not args.individual_layers:
        (
            mini_results,
            token_position_means,
            token_position_counts,
            responses_identical_true,
            token_label_counters,
        ) = run_tokenwise_evaluation(
            train_ds=train_ds,
            val_ds=val_ds,
            model=model,
            fewshot_prefix=fewshot_prefix,
            layer_to_verbalised_embedding_means_eval=layer_to_verbalised_embedding_means,
            none_cache=none_cache,
            ablation_target_ids=ablation_target_ids,
            num_samples=args.num_samples,
            max_new_tokens=args.model_max_new_tokens,
            parse_mode_verbalised_confidence=args.parse_mode_verbalised_confidence,
            expected_probability_tokens=args.expected_probability_tokens,
            mean_from_low_confidence=args.mean_from_low_confidence,
        )
        token_labels = _token_labels_from_counters(
            token_label_counters, expected_probability_tokens=args.expected_probability_tokens
        )

        mini_path = _mini_output_json_path(run_root)
        with open(mini_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", mini_path)

        config_path = _config_txt_path(run_root)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        write_config_txt(
            config_path,
            args=args,
            device=device,
            model_n_layers=model.cfg.n_layers,
            run_layers=run_layers,
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            baseline_mean=baseline_mean,
            baseline_count=baseline_count,
            token_position_means=token_position_means,
            token_position_counts=token_position_counts,
            responses_identical_true=responses_identical_true,
            finished_at=finished_at,
        )
        logging.info("Wrote %s", config_path)

        summary = {
            "run_root": run_root,
            "baseline": {
                "mode": "no_replacement",
                "mean_confidence": baseline_mean,
                "sample_count": baseline_count,
            },
            "expected_probability_tokens": args.expected_probability_tokens,
            "token_positions": {
                str(pos): {
                    "mode_key": _token_mode_key(pos),
                    "token_label": token_labels[pos],
                    "mean_confidence": token_position_means[pos],
                    "sample_count": token_position_counts[pos],
                    "responses_identical_true": responses_identical_true[pos],
                    "deviation_from_baseline": (
                        None
                        if token_position_means[pos] is None or baseline_mean is None
                        else float(token_position_means[pos] - baseline_mean)
                    ),
                }
                for pos in range(args.expected_probability_tokens)
            },
        }

        summary_path = _summary_json_path(run_root)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", summary_path)

        line_plot_path = _line_plot_path(run_root)
        write_line_plot(
            path=line_plot_path,
            token_position_means=token_position_means,
            baseline_mean=baseline_mean,
            expected_probability_tokens=args.expected_probability_tokens,
            token_labels=token_labels,
        )
        logging.info("Wrote %s", line_plot_path)
        return

    per_layer_token_means: Dict[int, Dict[int, Optional[float]]] = {}
    per_layer_token_counts: Dict[int, Dict[int, int]] = {}
    per_layer_identical_counts: Dict[int, Dict[int, int]] = {}
    global_token_label_counter: Dict[int, Counter[str]] = {
        i: Counter() for i in range(args.expected_probability_tokens)
    }

    for layer_idx in run_layers:
        logging.info("Running individual layer tokenwise sweep for layer %d", layer_idx)
        layer_map = {layer_idx: layer_to_verbalised_embedding_means[layer_idx]}
        (
            mini_results,
            token_position_means,
            token_position_counts,
            responses_identical_true,
            token_label_counters,
        ) = run_tokenwise_evaluation(
            train_ds=train_ds,
            val_ds=val_ds,
            model=model,
            fewshot_prefix=fewshot_prefix,
            layer_to_verbalised_embedding_means_eval=layer_map,
            none_cache=none_cache,
            ablation_target_ids=ablation_target_ids,
            num_samples=args.num_samples,
            max_new_tokens=args.model_max_new_tokens,
            parse_mode_verbalised_confidence=args.parse_mode_verbalised_confidence,
            expected_probability_tokens=args.expected_probability_tokens,
            mean_from_low_confidence=args.mean_from_low_confidence,
        )
        per_layer_token_means[layer_idx] = token_position_means
        per_layer_token_counts[layer_idx] = token_position_counts
        per_layer_identical_counts[layer_idx] = responses_identical_true
        for pos in range(args.expected_probability_tokens):
            global_token_label_counter[pos].update(token_label_counters[pos])

        layer_dir = os.path.join(run_root, str(layer_idx))
        os.makedirs(layer_dir, exist_ok=True)
        layer_mini_path = _mini_output_json_path(layer_dir)
        with open(layer_mini_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", layer_mini_path)

        layer_config_path = _config_txt_path(layer_dir)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        write_config_txt(
            layer_config_path,
            args=args,
            device=device,
            model_n_layers=model.cfg.n_layers,
            run_layers=[layer_idx],
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            baseline_mean=baseline_mean,
            baseline_count=baseline_count,
            token_position_means=token_position_means,
            token_position_counts=token_position_counts,
            responses_identical_true=responses_identical_true,
            finished_at=finished_at,
        )
        logging.info("Wrote %s", layer_config_path)

    token_labels = _token_labels_from_counters(
        global_token_label_counter, expected_probability_tokens=args.expected_probability_tokens
    )

    matrix_values = np.full((len(run_layers), args.expected_probability_tokens), np.nan, dtype=np.float32)
    matrix_dev_abs = np.full((len(run_layers), args.expected_probability_tokens), np.nan, dtype=np.float32)
    for r, layer_idx in enumerate(run_layers):
        for c in range(args.expected_probability_tokens):
            val = per_layer_token_means[layer_idx].get(c)
            if val is None:
                continue
            matrix_values[r, c] = float(val)
            if baseline_mean is not None:
                matrix_dev_abs[r, c] = abs(float(val) - float(baseline_mean))
            else:
                matrix_dev_abs[r, c] = np.nan

    line_token_means: Dict[int, Optional[float]] = {}
    line_token_counts: Dict[int, int] = {}
    line_identical_counts: Dict[int, int] = {}
    for pos in range(args.expected_probability_tokens):
        vals = [per_layer_token_means[layer][pos] for layer in run_layers if per_layer_token_means[layer][pos] is not None]
        line_token_means[pos] = float(np.mean(vals)) if vals else None
        line_token_counts[pos] = sum(per_layer_token_counts[layer][pos] for layer in run_layers)
        line_identical_counts[pos] = sum(per_layer_identical_counts[layer][pos] for layer in run_layers)

    summary = {
        "run_root": run_root,
        "individual_layers": True,
        "run_layers": list(run_layers),
        "baseline": {
            "mode": "no_replacement",
            "mean_confidence": baseline_mean,
            "sample_count": baseline_count,
        },
        "expected_probability_tokens": args.expected_probability_tokens,
        "token_positions": {
            str(pos): {
                "mode_key": _token_mode_key(pos),
                "token_label": token_labels[pos],
                "mean_confidence_across_layers": line_token_means[pos],
                "sample_count_across_layers": line_token_counts[pos],
                "responses_identical_true_across_layers": line_identical_counts[pos],
                "deviation_from_baseline_across_layers": (
                    None
                    if line_token_means[pos] is None or baseline_mean is None
                    else float(line_token_means[pos] - baseline_mean)
                ),
            }
            for pos in range(args.expected_probability_tokens)
        },
        "layer_token_confidence": {
            str(layer_idx): {
                str(pos): per_layer_token_means[layer_idx].get(pos)
                for pos in range(args.expected_probability_tokens)
            }
            for layer_idx in run_layers
        },
        "layer_token_deviation_from_baseline": {
            str(layer_idx): {
                str(pos): (
                    None
                    if per_layer_token_means[layer_idx].get(pos) is None or baseline_mean is None
                    else float(per_layer_token_means[layer_idx][pos] - baseline_mean)
                )
                for pos in range(args.expected_probability_tokens)
            }
            for layer_idx in run_layers
        },
    }
    summary_path = _summary_json_path(run_root)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Wrote %s", summary_path)

    root_config_path = _config_txt_path(run_root)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    write_config_txt(
        root_config_path,
        args=args,
        device=device,
        model_n_layers=model.cfg.n_layers,
        run_layers=run_layers,
        prompt_indices=prompt_indices,
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        baseline_mean=baseline_mean,
        baseline_count=baseline_count,
        token_position_means=line_token_means,
        token_position_counts=line_token_counts,
        responses_identical_true=line_identical_counts,
        finished_at=finished_at,
    )
    logging.info("Wrote %s", root_config_path)

    grid_path = _grid_plot_path(run_root)
    write_layer_token_grid_plot(
        path=grid_path,
        run_layers=run_layers,
        token_labels=token_labels,
        matrix_values=matrix_values,
        matrix_deviation_abs=matrix_dev_abs,
    )
    logging.info("Wrote %s", grid_path)

    line_plot_path = _line_plot_path(run_root)
    write_line_plot(
        path=line_plot_path,
        token_position_means=line_token_means,
        baseline_mean=baseline_mean,
        expected_probability_tokens=args.expected_probability_tokens,
        token_labels=token_labels,
    )
    logging.info("Wrote %s", line_plot_path)


if __name__ == "__main__":
    main()
