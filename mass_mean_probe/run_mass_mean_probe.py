#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Greedy decoding on TriviaQA with mass mean-direction probing via TransformerLens.

This script computes:
  - low-confidence mean hidden states over probability-marker positions
  - high-confidence mean hidden states over the same positions
  - direction = high_mean - low_mean

Then, for each selected ablation target group and alpha:
  - low targets:  resid_post += alpha * direction
  - high targets: resid_post -= alpha * direction

Ablation modes apply additive direction perturbation along span-specific directions
(``*_mean_replace`` names are kept for compatibility with mean-ablation scripts).
Includes full and subset ``Probability:`` span modes, including row-index subsets
(see ``--ablation_mode`` help).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import logging
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.weight_processing import ProcessWeights

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PROCESS_GEN_DIR = _REPO_ROOT / "process_generations"
if str(_PROCESS_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESS_GEN_DIR))

from ans_gen.generate_answers_h5 import (  # noqa: E402
    CONFIDENCE_PROMPT_LINGUISTIC,
    LINGUISTIC_TO_PROBABILITY,
    parse_linguistic_confidence_from_response,
)
from process_generations_more_embs_from_h5 import (  # noqa: E402
    GEMMA_CONFIDENCE_PREFIX_TOKENS,
    MISTRAL_CONFIDENCE_PREFIX_TOKENS,
    QWEN_CONFIDENCE_PREFIX_TOKENS,
)
from layerwise_mean_ablation.run_mean_ablation import (  # noqa: E402
    LAYER_INDEXING_NOTE,
    PROBABILITY_ROW_INDEX_MODES,
    SUPPORTED_MODEL_NAMES,
    hook_name_for_display_layer,
    probability_row_indices_for_mode,
    truncate_sorted_ids,
    validate_last_a_panl_and_pc_mode,
)

CONFIDENCE_PROMPT_NUMERIC = (
    "Provide your best guess and the probability that it is correct (0.0 to 1.0) "
    "for the following question. Give ONLY the guess and probability, no other words "
    "or explanation. For example:\n\n"
    "Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n "
    "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any "
    "extra commentary whatsoever; just the probability!>\n\n"
    "The question is: "
)

BRIEF_PROMPTS = {
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "<end_of_turn>"]

# Gemma-3 token alternatives (fixed length; each inner list = allowed tokens at that position)
GEMMA_GUESS_PREFIX_TOKENS = [
    ["\n", "\n\n"],
    ["Guess"],
    [":"],
]
GEMMA_PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    ["Probability", " Probability"],
    [":"],
    [" "],
]

# Qwen2.5 token alternatives (from ans_gen/generated_answers/3_32B_200 decoded tokens)
QWEN_GUESS_PREFIX_TOKENS = [
    [" Guess", "Guess"],
    [":"],
]
QWEN_PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    [" Probability"],
    [":"],
    [" "],
]

# Mistral-7B-Instruct-v0.1 (from ans_gen/generated_answers/1_svamp_mistral)
MISTRAL_GUESS_PREFIX_TOKENS = [
    ["\n"],
    ["\n"],
    ["Gu"],
    ["ess"],
    [":"],
]
MISTRAL_PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    ["Pro"],
    ["b"],
    ["ability"],
    [":"],
    [""],  # space before number decodes as empty string
]

# Active tables; set via configure_prefix_tokens_for_model(model_name).
GUESS_PREFIX_TOKENS: list[list[str]] = GEMMA_GUESS_PREFIX_TOKENS
PROBABILITY_PREFIX_TOKENS: list[list[str]] = GEMMA_PROBABILITY_PREFIX_TOKENS
CONFIDENCE_PREFIX_TOKENS: list[list[str]] = GEMMA_CONFIDENCE_PREFIX_TOKENS


def configure_prefix_tokens_for_model(model_name: str) -> None:
    """Set GUESS/PROBABILITY/CONFIDENCE_PREFIX_TOKENS from exact model_name (case-sensitive)."""
    global GUESS_PREFIX_TOKENS, PROBABILITY_PREFIX_TOKENS, CONFIDENCE_PREFIX_TOKENS
    if model_name == "google/gemma-3-12b-it":
        GUESS_PREFIX_TOKENS = GEMMA_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = GEMMA_PROBABILITY_PREFIX_TOKENS
        CONFIDENCE_PREFIX_TOKENS = GEMMA_CONFIDENCE_PREFIX_TOKENS
    elif model_name == "Qwen/Qwen2.5-32B-Instruct":
        GUESS_PREFIX_TOKENS = QWEN_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = QWEN_PROBABILITY_PREFIX_TOKENS
        CONFIDENCE_PREFIX_TOKENS = QWEN_CONFIDENCE_PREFIX_TOKENS
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        GUESS_PREFIX_TOKENS = MISTRAL_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = MISTRAL_PROBABILITY_PREFIX_TOKENS
        CONFIDENCE_PREFIX_TOKENS = MISTRAL_CONFIDENCE_PREFIX_TOKENS
    else:
        raise ValueError(
            f"Unsupported model_name for Guess/Probability token parsing: {model_name!r}. "
            f"Supported: {list(SUPPORTED_MODEL_NAMES)}."
        )

DEFAULT_SEMANTIC_SIMILARITY_MODEL = "all-MiniLM-L6-v2"

SEMANTIC_SIMILARITY_MODES = frozenset({
    "guess_tokens_mean_replace",
    "all_pre_guess_tokens_mean_replace",
    "guess_then_guess_probability_mean_replace",
    "semantic_answer_mean_replace",
    "current_generated_token_mean_replace",
    "current_generated_window5_mean_replace",
    "prompt_tokens_mean_replace",
    "sem_ans_tokens_during_gen",
    "all_tokens_mean_replace",
    "generated_tokens_mean_replace",
})

GENERATED_TOKENS_SOURCE_CHOICES = ("probability_prefix_last_token",)
WHOLE_SEQUENCE_MODES = frozenset({
    "all_tokens_mean_replace",
    "generated_tokens_mean_replace",
})


def _prefix_tokens_for_linguistic_confidence(
    linguistic_confidence_prompt: bool,
) -> list[list[str]]:
    return CONFIDENCE_PREFIX_TOKENS if linguistic_confidence_prompt else PROBABILITY_PREFIX_TOKENS


def _match_token_prefix(
    decoded_tokens: List[str],
    prefix_tokens: list[list[str]],
    *,
    start: int = 0,
) -> int | None:
    """Return start index of first match of prefix_tokens (2D alts) at/after `start`, else None."""
    prefix_len = len(prefix_tokens)
    if prefix_len == 0:
        return None
    for i in range(start, len(decoded_tokens) - prefix_len + 1):
        if all(decoded_tokens[i + j] in prefix_tokens[j] for j in range(prefix_len)):
            return i
    return None


def parse_guess_and_marker_indices(
    decoded_tokens: List[str],
    *,
    linguistic_confidence_prompt: bool,
) -> tuple[int, int, int] | None:
    """Return (last_guess_token_index, first_span_token_index, end_span_token_index) completion indices.

    Uses ``PROBABILITY_PREFIX_TOKENS`` or ``CONFIDENCE_PREFIX_TOKENS`` according to
    ``linguistic_confidence_prompt``. The span always starts at the first occurrence of the marker.
    For numeric ``Probability:``, the prefix includes a whitespace token and the span ends at the
    first value token after that space. For linguistic ``Confidence:``, there is no required space;
    the span ends at the first token after the marker.
    """
    guess_start = _match_token_prefix(decoded_tokens, GUESS_PREFIX_TOKENS, start=0)
    if guess_start is None:
        return None

    last_guess_token_index = guess_start + len(GUESS_PREFIX_TOKENS)
    prefix_tokens = _prefix_tokens_for_linguistic_confidence(linguistic_confidence_prompt)
    marker_start = _match_token_prefix(
        decoded_tokens, prefix_tokens, start=last_guess_token_index
    )
    if marker_start is None:
        return None

    first_span_token_index = marker_start
    end_span_token_index = marker_start + len(prefix_tokens)

    if (
        last_guess_token_index <= 0
        or last_guess_token_index >= len(decoded_tokens)
        or end_span_token_index > len(decoded_tokens)
        or last_guess_token_index >= first_span_token_index
        or first_span_token_index >= end_span_token_index
    ):
        return None
    return (last_guess_token_index, first_span_token_index, end_span_token_index)


def parse_guess_and_probability_indices(decoded_tokens: List[str]) -> tuple[int, int, int] | None:
    return parse_guess_and_marker_indices(
        decoded_tokens,
        linguistic_confidence_prompt=False,
    )


def parse_guess_start_index(decoded_tokens: List[str]) -> Optional[int]:
    guess_start = _match_token_prefix(decoded_tokens, GUESS_PREFIX_TOKENS, start=0)
    if guess_start is None:
        return None
    last_guess_token_index = guess_start + len(GUESS_PREFIX_TOKENS)
    if last_guess_token_index <= 0 or last_guess_token_index > len(decoded_tokens):
        return None
    return last_guess_token_index


def load_eval_dataset(dataset_name: str, seed: int):
    """Load train/validation splits via semantic_uncertainty.data_utils.load_ds."""
    sem_unc_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "semantic_uncertainty")
    if sem_unc_root not in sys.path:
        sys.path.insert(0, sem_unc_root)
    from uncertainty.data.data_utils import load_ds

    train_ds, val_ds = load_ds(dataset_name, seed=seed)
    if train_ds is None or val_ds is None:
        raise ValueError(f"Unsupported or failed dataset load: {dataset_name}")
    return train_ds, val_ds


def split_answerable_indices(dataset: Dataset) -> List[int]:
    return [i for i, ex in enumerate(dataset) if len(ex["answers"]["text"]) > 0]


def construct_fewshot_prompt_from_indices(
    dataset: Dataset,
    example_indices: Sequence[int],
    brief: str,
    brief_always: bool,
    use_context: bool,
) -> str:
    prompt = brief if not brief_always else ""
    for example_index in example_indices:
        example = dataset[int(example_index)]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        piece = ""
        if brief_always:
            piece += brief
        if use_context and context is not None:
            piece += f"Context: {context}\n"
        piece += f"Question: {question}\n"
        piece += f"Answer: {answer}\n\n" if answer else "Answer:"
        prompt += piece
    return prompt


def encode_example_id(example_id) -> str:
    return quote(str(example_id), safe="")


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        return cli_output_path
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(repo_dir, "results")
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_dir = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "ablation_results.json")


def mini_output_json_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "ablation_results_mini.json")


def config_txt_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "config.txt")


def summary_json_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "summary.json")


def mode_to_output_key(mode: str) -> str:
    if mode == "none":
        return "no_replacement"
    if mode in {
        "probability_tokens_mean_replace",
        "probability_last_token_mean_replace",
        "probability_span_except_last_token_mean_replace",
        "probability_value_mean_replace",
        "semantic_answer_mean_replace",
        "all_pre_probability_tokens_mean_replace",
        "guess_tokens_mean_replace",
        "all_pre_guess_tokens_mean_replace",
        "guess_then_guess_probability_mean_replace",
        "current_generated_token_mean_replace",
        "current_generated_window5_mean_replace",
        "prompt_tokens_mean_replace",
        "sem_ans_tokens_during_gen",
        "all_tokens_mean_replace",
        "generated_tokens_mean_replace",
    }:
        return mode
    if mode in PROBABILITY_ROW_INDEX_MODES:
        return mode
    raise ValueError(f"Unknown mode: {mode}")


def _format_alpha(alpha: float) -> str:
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace(".", "p")


def parse_probability_from_response(response_str: str) -> float | None:
    if not response_str or not isinstance(response_str, str):
        return None
    matches = list(re.finditer(r"probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?", response_str, re.IGNORECASE))
    if not matches:
        matches = list(re.finditer(r"probability\s*:\s*(\d+(?:[.,]\d+)?)", response_str, re.IGNORECASE))
    if not matches:
        return None
    raw = matches[0].group(1).strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0 or value > 1:
        return None
    return value


def parse_mode_confidence_from_response(response: str, *, linguistic_prompt: bool) -> Optional[float]:
    if linguistic_prompt:
        parsed = parse_linguistic_confidence_from_response(response)
    else:
        parsed = parse_probability_from_response(response)
    return float(parsed) if parsed is not None else None


def parse_semantic_answer_from_response(
    response_str: str,
    *,
    linguistic_confidence_prompt: bool = False,
) -> Optional[str]:
    """Extract guess text between ``Guess:`` and the first marker span."""
    if not response_str or not isinstance(response_str, str):
        return None
    if linguistic_confidence_prompt:
        pattern = r"guess\s*:\s*(.*?)\s*confidence\s*:"
    else:
        pattern = r"guess\s*:\s*(.*?)\s*probability\s*:"
    match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer if answer else None


def compute_verbalised_confidence_effect(
    baseline_conf: float,
    mode_conf: float,
    *,
    mean_from_low_confidence: bool,
) -> Optional[float]:
    if mean_from_low_confidence:
        diff = max(0.0, float(baseline_conf) - float(mode_conf))
        denom = float(baseline_conf)
    else:
        diff = max(0.0, float(mode_conf) - float(baseline_conf))
        denom = 1.0 - float(baseline_conf)
    if denom <= 0.0:
        return None
    return diff / denom


def compute_uncertainty_score(semantic_similarity: float, verbalised_confidence_effect: float) -> float:
    return float(semantic_similarity) * float(verbalised_confidence_effect)


def load_sentence_transformer_for_metrics(model_name: str = DEFAULT_SEMANTIC_SIMILARITY_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def batch_compute_semantic_similarities(
    sentence_model,
    pairs: Sequence[Tuple[str, str]],
) -> List[float]:
    if not pairs:
        return []
    from sentence_transformers import util

    texts_a = [text_a for text_a, _text_b in pairs]
    texts_b = [_text_b for _text_a, _text_b in pairs]
    emb_a = sentence_model.encode(texts_a, convert_to_tensor=True)
    emb_b = sentence_model.encode(texts_b, convert_to_tensor=True)
    sims = util.cos_sim(emb_a, emb_b).diag()
    return [max(0.0, float(sim.item())) for sim in sims]


def _mean_and_count(values: Sequence[float]) -> Tuple[Optional[float], int]:
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def _entry_key_for_mode_target_alpha(mode: str, target: str, alpha: float) -> str:
    return f"{mode_to_output_key(mode)}__target_{target}__alpha_{_format_alpha(alpha)}"


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    ablate_layers: Sequence[int],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    finished_at: str,
    summary_payload: Optional[Dict[str, object]] = None,
) -> None:
    lines = [
        "Mass Mean Probe Configuration",
        "=============================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        "",
        "[Data]",
        f"input_h5={args.input_h5}",
        f"dataset={args.dataset}",
        f"new_h5_format={args.new_h5_format}",
        f"h5_example_count={h5_example_count}",
        f"random_seed={args.random_seed}",
        f"num_samples={args.num_samples}",
        f"num_few_shot={args.num_few_shot}",
        f"fewshot_prompt_indices={','.join(str(i) for i in prompt_indices)}",
        "",
        "[Prompt/Generation]",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"brief_prompt={args.brief_prompt}",
        f"enable_brief={args.enable_brief}",
        f"brief_always={args.brief_always}",
        f"use_context={args.use_context}",
        "",
        "[Ablation]",
        f"ablation_mode={args.ablation_mode}",
        f"generated_tokens_source={args.generated_tokens_source}",
        f"ablation_targets={args.ablation_targets}",
        f"alpha={args.alpha}",
        "non_none_mode_behavior=additive_direction_perturbation",
        "direction_definition=high_mean_minus_low_mean",
        "confidence_direction_expectation_for_low_targets=perturbed_confidence_gt_none",
        "confidence_direction_expectation_for_high_targets=perturbed_confidence_lt_none",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"layer_indexing={LAYER_INDEXING_NOTE}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_confidence_tokens={args.expected_confidence_tokens}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        f"normalize_span_directions={args.normalize_span_directions}",
        f"parse_mode_verbalised_confidence={args.parse_mode_verbalised_confidence}",
        f"linguistic_confidence_prompt={args.linguistic_confidence_prompt}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
    ]
    if summary_payload is not None:
        lines.extend(["", "[Derived metrics]", "Per-mode aggregates from summary.json (all targets/alphas).", ""])
        for mode_name in [m for m in args.ablation_mode if m != "none"]:
            mode_block = summary_payload.get(mode_name)
            if not isinstance(mode_block, dict):
                continue
            for target in args.ablation_targets:
                target_block = mode_block.get(target)
                if not isinstance(target_block, dict):
                    continue
                for alpha_key, metrics in sorted(target_block.items()):
                    if not isinstance(metrics, dict):
                        continue
                    parts = [f"{mode_name} target={target} alpha={alpha_key}"]
                    mean_conf = metrics.get("mean_confidence")
                    if mean_conf is not None:
                        parts.append(f"mean_confidence={float(mean_conf):.6f}")
                    sem_mean = metrics.get("mean_semantic_similarity")
                    if sem_mean is not None:
                        parts.append(f"mean_semantic_similarity={float(sem_mean):.6f}")
                    vce_mean = metrics.get("mean_verbalised_confidence_effect")
                    if vce_mean is not None:
                        parts.append(f"mean_verbalised_confidence_effect={float(vce_mean):.6f}")
                    unc_mean = metrics.get("mean_uncertainty_score")
                    if unc_mean is not None:
                        parts.append(f"mean_uncertainty_score={float(unc_mean):.6f}")
                    lines.append(" ".join(parts))
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_summary_json(
    *,
    non_none_modes: Sequence[str],
    ablation_targets: Sequence[str],
    alphas: Sequence[float],
    mode_confidence_values: Dict[str, Dict[str, Dict[float, List[float]]]],
    mode_responses_identical_true: Dict[str, Dict[str, Dict[float, int]]],
    baseline_values_by_target: Dict[str, List[float]],
    mode_semantic_similarity_values: Optional[Dict[str, Dict[str, Dict[float, List[float]]]]] = None,
    mode_verbalised_confidence_effect_values: Optional[Dict[str, Dict[str, Dict[float, List[float]]]]] = None,
    mode_uncertainty_score_values: Optional[Dict[str, Dict[str, Dict[float, List[float]]]]] = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for mode in non_none_modes:
        mode_payload: Dict[str, Dict[str, Dict[str, Optional[float] | int]]] = {}
        for target in ("low", "high"):
            if target not in ablation_targets:
                continue
            per_alpha_payload: Dict[str, Dict[str, Optional[float] | int]] = {}
            for alpha in sorted(alphas):
                alpha_key = _format_alpha(alpha)
                values = mode_confidence_values.get(mode, {}).get(target, {}).get(alpha, [])
                mean_conf = float(np.mean(values)) if values else None
                alpha_metrics: Dict[str, Optional[float] | int] = {
                    "alpha_value": float(alpha),
                    "mean_confidence": mean_conf,
                    "sample_count": len(values),
                    "responses_identical_count": int(
                        mode_responses_identical_true.get(mode, {}).get(target, {}).get(alpha, 0)
                    ),
                }
                if mode_semantic_similarity_values is not None:
                    sem_values = mode_semantic_similarity_values.get(mode, {}).get(target, {}).get(alpha, [])
                    sem_mean, sem_count = _mean_and_count(sem_values)
                    alpha_metrics["mean_semantic_similarity"] = sem_mean
                    alpha_metrics["semantic_similarity_sample_count"] = sem_count
                if mode_verbalised_confidence_effect_values is not None:
                    vce_values = mode_verbalised_confidence_effect_values.get(mode, {}).get(target, {}).get(alpha, [])
                    vce_mean, vce_count = _mean_and_count(vce_values)
                    alpha_metrics["mean_verbalised_confidence_effect"] = vce_mean
                    alpha_metrics["verbalised_confidence_effect_sample_count"] = vce_count
                if mode_uncertainty_score_values is not None:
                    unc_values = mode_uncertainty_score_values.get(mode, {}).get(target, {}).get(alpha, [])
                    unc_mean, unc_count = _mean_and_count(unc_values)
                    alpha_metrics["mean_uncertainty_score"] = unc_mean
                    alpha_metrics["uncertainty_score_sample_count"] = unc_count
                per_alpha_payload[alpha_key] = alpha_metrics
            mode_payload[target] = per_alpha_payload
        summary[mode] = mode_payload

    baseline_payload: Dict[str, Dict[str, Optional[float] | int]] = {}
    for target in ("low", "high"):
        values = baseline_values_by_target.get(target, [])
        baseline_payload[target] = {
            "mean_confidence": float(np.mean(values)) if values else None,
            "sample_count": len(values),
            "responses_identical_count": 0,
        }
    summary["no_replacement_baseline"] = baseline_payload
    return summary


def write_summary_json(path: str, summary_payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)


def _plot_mode_confidence_from_summary(
    *,
    mode_name: str,
    mode_payload: Dict[str, Dict[str, Dict[str, object]]],
    baseline_payload: Dict[str, Dict[str, object]],
    ablation_targets: Sequence[str],
    output_path: str,
) -> None:
    target_colors = {"high": "tab:blue", "low": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for target in ("high", "low"):
        if target not in ablation_targets:
            continue
        alpha_map = mode_payload.get(target, {})
        if not isinstance(alpha_map, dict):
            continue
        points: List[Tuple[float, float, int]] = []
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
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=target_colors[target], label=f"target={target}")
            for x_val, y_val, sample_count in points:
                ax.annotate(
                    str(sample_count),
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=target_colors[target],
                )

    for baseline_target in ("high", "low"):
        baseline_metrics = baseline_payload.get(baseline_target, {})
        baseline_mean = baseline_metrics.get("mean_confidence") if isinstance(baseline_metrics, dict) else None
        if baseline_mean is None:
            continue
        ax.axhline(
            y=float(baseline_mean),
            color=target_colors[baseline_target],
            linestyle=":",
            linewidth=1.0,
            label=f"baseline_{baseline_target}",
        )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Verbalised confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Verbalised confidence vs alpha ({mode_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def write_summary_plots_from_json(
    *,
    summary_payload: Dict[str, object],
    ablation_targets: Sequence[str],
    output_dir: str,
) -> None:
    baseline_payload = summary_payload.get("no_replacement_baseline", {})
    if not isinstance(baseline_payload, dict):
        raise ValueError("summary payload baseline must be a dict")
    for mode_name, mode_payload in summary_payload.items():
        if mode_name == "no_replacement_baseline":
            continue
        if not isinstance(mode_payload, dict):
            continue
        plot_path = os.path.join(output_dir, f"verbalised_confidence_vs_alpha__{mode_name}.png")
        _plot_mode_confidence_from_summary(
            mode_name=mode_name,
            mode_payload=mode_payload,
            baseline_payload=baseline_payload,
            ablation_targets=ablation_targets,
            output_path=plot_path,
        )


def write_summary_plots_from_file(*, summary_json_file: str, ablation_targets: Sequence[str], output_dir: str) -> None:
    with open(summary_json_file, "r", encoding="utf-8") as f:
        summary_payload = json.load(f)
    if not isinstance(summary_payload, dict):
        raise ValueError(f"Expected dict in summary JSON, got {type(summary_payload).__name__}")
    write_summary_plots_from_json(
        summary_payload=summary_payload,
        ablation_targets=ablation_targets,
        output_dir=output_dir,
    )


@contextmanager
def _without_tl_noop_fp32_upcast():
    """Skip TL's full-state-dict fp32 upcast when no weight processing is requested."""
    original = ProcessWeights.__dict__["process_weights"]
    original_fn = original.__func__ if isinstance(original, staticmethod) else original

    def _process_weights(
        state_dict,
        cfg,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
        adapter=None,
    ):
        if not (
            fold_ln
            or center_writing_weights
            or center_unembed
            or fold_value_biases
            or refactor_factored_attn_matrices
        ):
            return state_dict
        return original_fn(
            state_dict,
            cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            adapter=adapter,
        )

    ProcessWeights.process_weights = staticmethod(_process_weights)
    try:
        yield
    finally:
        ProcessWeights.process_weights = original


def load_hooked_transformer(
    model_name: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> HookedTransformer:
    def _from_pretrained(**extra_kwargs):
        with _without_tl_noop_fp32_upcast():
            return HookedTransformer.from_pretrained_no_processing(
                model_name,
                device=device,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )

    try:
        model = _from_pretrained()
    except Exception as exc:
        if "PyPreTokenizerTypeWrapper" not in str(exc):
            raise
        logging.warning("Fast tokenizer load failed (%s). Retrying with use_fast=False.", exc)
        hf_token = os.environ.get("HF_TOKEN", None)
        slow_tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_bos_token=True, trust_remote_code=True, use_fast=False, token=hf_token
        )
        model = _from_pretrained(tokenizer=slow_tokenizer)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_h5_node(node):
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "S":
                return np.array([x.decode("utf-8") for x in value], dtype=object).tolist()
            if value.dtype.kind == "O":
                return [_decode_h5_string(x) for x in value.tolist()]
            return value
        return _decode_h5_string(value)

    node_type = node.attrs.get("__type__", "")
    if isinstance(node_type, bytes):
        node_type = node_type.decode("utf-8")
    if node_type == "none":
        return None
    if node_type in ("list", "tuple"):
        length = int(node.attrs.get("__len__", len(node.keys())))
        items = [_read_h5_node(node[str(i)]) for i in range(length)]
        return tuple(items) if node_type == "tuple" else items
    return {k: _read_h5_node(node[k]) for k in node.keys()}


_NEW_H5_REQUIRED_COMPONENTS = ("res", "attn", "mlp")
_NEW_H5_REQUIRED_EMBEDDING_FIELDS = (
    "embeddings_mean_prompt",
    "embeddings_guess",
    "embeddings_mean_sem_answer",
    "embeddings_probability",
)


def _extract_res_field(resp0: dict, ex_id: str, field_name: str, *, new_h5_format: bool):
    """Return the embedding payload for ``field_name`` on ``resp0``.

    When ``new_h5_format`` is set, every required embedding field is expected to be a
    dict containing all of ``res``/``attn``/``mlp`` (non-null). Mass-mean direction probing
    only consumes residual-stream activations; ``attn``/``mlp`` are validated-but-unused.
    """
    value = resp0.get(field_name)
    if not new_h5_format:
        return value
    if not isinstance(value, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be a dict with "
            f"keys {_NEW_H5_REQUIRED_COMPONENTS} when --new_h5_format is set."
        )
    for component in _NEW_H5_REQUIRED_COMPONENTS:
        if component not in value:
            raise ValueError(
                f"Example {ex_id} responses/0/{field_name} missing component '{component}' "
                f"(required when --new_h5_format is set)."
            )
        if value[component] is None:
            raise ValueError(
                f"Example {ex_id} responses/0/{field_name}/{component} is null "
                f"(must be populated when --new_h5_format is set)."
            )
    return value["res"]


def load_examples_h5(path: Path) -> Dict[str, dict]:
    examples: Dict[str, dict] = {}
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        ex_group = h5_file["examples"]
        for example_id in ex_group.keys():
            obj = _read_h5_node(ex_group[example_id])
            if isinstance(obj, dict):
                examples[str(example_id)] = obj
    return examples


def parse_ablate_layers(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        layers = list(range(int(a.strip()), int(b.strip()) + 1))
    else:
        layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}) for this model.")
    return layers


def _as_layer_hidden(arr_like: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 4:
        return arr[:, 0, -1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected embedding tensor shape: {arr.shape}; expected 4D or 2D.")


def _is_expected_or_plus_two(actual_len: int, expected_len: int) -> bool:
    return actual_len in (expected_len, expected_len + 2)


def _expected_probability_span_token_budget(
    *,
    linguistic_confidence_prompt: bool,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    extend_probability_span: bool = False,
) -> int:
    """Token budget for span length checks and truncation on decoded completions."""
    if linguistic_confidence_prompt:
        return expected_confidence_tokens
    return expected_probability_tokens + (2 if extend_probability_span else 0)


def _completion_token_index_to_abs_pos(prompt_len: int, completion_index: int) -> int:
    """Map completion-relative token index (0 = first generated token) to full-sequence position.

    First generated token aligns with ``prompt_len - 1`` (same convention as
    ``layerwise_mean_ablation.run_mean_ablation``).
    """
    return prompt_len + completion_index - 1


def _absolute_prob_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    extend_probability_span: bool = False,
) -> List[int]:
    parsed = parse_guess_and_marker_indices(
        decoded_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    apply_extend = extend_probability_span and not linguistic_confidence_prompt
    if apply_extend and end_prob + 2 > len(decoded_tokens):
        return []
    span_end = end_prob + (2 if apply_extend else 0)
    rel_positions = list(range(first_prob, span_end + 1))
    span_budget = _expected_probability_span_token_budget(
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        extend_probability_span=extend_probability_span,
    )
    logging.info(f"rel_positions: {rel_positions}")
    logging.info(f"span_budget: {span_budget}, length of rel_positions: {len(rel_positions)}")
    logging.info(f"prompt_len: {prompt_len}, decoded_tokens length: {len(decoded_tokens)}")
    if not _is_expected_or_plus_two(len(rel_positions), span_budget):
        return []
    rel_positions = rel_positions[:span_budget]
    final_tokens = [_completion_token_index_to_abs_pos(prompt_len, pos) for pos in rel_positions]
    logging.info(f"final_tokens: {final_tokens}")
    return final_tokens


def _absolute_prob_positions_at_row_indices(
    prompt_len: int,
    decoded_tokens: List[str],
    row_indices: Sequence[int],
    *,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    extend_probability_span: bool = False,
) -> List[int]:
    """Absolute positions for selected rows of the H5 probability-prefix span (0-indexed)."""
    full_positions = _absolute_prob_positions(
        prompt_len,
        decoded_tokens,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        extend_probability_span=extend_probability_span,
    )
    if not full_positions:
        return []
    out: List[int] = []
    for idx in row_indices:
        if idx < 0 or idx >= len(full_positions):
            return []
        out.append(full_positions[idx])
    return out


def _absolute_prob_last_token_only(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    extend_probability_span: bool = False,
) -> List[int]:
    """Absolute index for the last token in the Probability:/Confidence: marker span."""
    parsed = parse_guess_and_marker_indices(
        decoded_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    apply_extend = extend_probability_span and not linguistic_confidence_prompt
    if apply_extend and end_prob + 2 > len(decoded_tokens):
        return []
    span_end = end_prob + (2 if apply_extend else 0)
    full_rel_positions = list(range(first_prob, span_end + 1))
    span_budget = _expected_probability_span_token_budget(
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        extend_probability_span=extend_probability_span,
    )
    if not _is_expected_or_plus_two(len(full_rel_positions), span_budget):
        return []
    return [_completion_token_index_to_abs_pos(prompt_len, span_end)]


def _absolute_prob_marker_except_last_token(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    extend_probability_span: bool = False,
) -> List[int]:
    """Absolute indices for the Probability:/Confidence: marker span excluding the last span token."""
    parsed = parse_guess_and_marker_indices(
        decoded_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    apply_extend = extend_probability_span and not linguistic_confidence_prompt
    if apply_extend and end_prob + 2 > len(decoded_tokens):
        return []
    span_end = end_prob + (2 if apply_extend else 0)
    full_rel_positions = list(range(first_prob, span_end + 1))
    span_budget = _expected_probability_span_token_budget(
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        extend_probability_span=extend_probability_span,
    )
    if not _is_expected_or_plus_two(len(full_rel_positions), span_budget):
        return []
    return [
        _completion_token_index_to_abs_pos(prompt_len, pos) for pos in range(first_prob, span_end)
    ]


def _absolute_sem_answer_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    linguistic_confidence_prompt: bool = False,
) -> List[int]:
    """Absolute positions for semantic-answer completion tokens (between Guess: and marker span)."""
    parsed = parse_guess_and_marker_indices(
        decoded_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    if parsed is None:
        return []
    last_guess_token_index, first_span_token_index, _ = parsed
    seq_len = prompt_len + len(decoded_tokens)
    out: List[int] = []
    for k in range(last_guess_token_index, first_span_token_index):
        p = _completion_token_index_to_abs_pos(prompt_len, k)
        if 0 <= p < seq_len:
            out.append(p)
    return out


def _absolute_sem_answer_positions_during_gen(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    linguistic_confidence_prompt: bool = False,
) -> List[int]:
    """Semantic-answer positions during generation (guess parsed) or after full parse."""
    full_positions = _absolute_sem_answer_positions(
        prompt_len,
        decoded_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    if full_positions:
        return full_positions
    guess_start = parse_guess_start_index(decoded_tokens)
    if guess_start is None:
        return []
    seq_len = prompt_len + len(decoded_tokens)
    out: List[int] = []
    for k in range(guess_start, len(decoded_tokens)):
        p = _completion_token_index_to_abs_pos(prompt_len, k)
        if 0 <= p < seq_len:
            out.append(p)
    return out


def _absolute_guess_span_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
) -> List[int]:
    last_guess_token_index = parse_guess_start_index(decoded_tokens)
    if last_guess_token_index is None:
        return []
    guess_positions_rel = list(range(0, last_guess_token_index))
    if len(guess_positions_rel) != expected_guess_tokens:
        return []
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]
    return [_completion_token_index_to_abs_pos(prompt_len, k) for k in guess_positions_rel]


def _absolute_pre_probability_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    extend_probability_span: bool = False,
) -> Optional[Dict[str, List[int]]]:
    """Absolute positions for pre-probability mean ablation.

    ``prompt`` uses indices ``0 .. prompt_len-2`` (excludes the last prompt position, which
    aligns with the first generated token under the completion-index mapping).
    """
    parsed = parse_guess_and_marker_indices(
        decoded_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    if parsed is None:
        return None
    last_guess_token_index, first_prob_token_index, end_prob_token_index = parsed

    guess_positions_rel = list(range(0, last_guess_token_index))
    if len(guess_positions_rel) != expected_guess_tokens:
        return None
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]

    apply_extend = extend_probability_span and not linguistic_confidence_prompt
    if apply_extend and end_prob_token_index + 2 > len(decoded_tokens):
        return None
    span_end = end_prob_token_index + (2 if apply_extend else 0)
    probability_positions_rel = list(range(first_prob_token_index, span_end + 1))
    span_budget = _expected_probability_span_token_budget(
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        extend_probability_span=extend_probability_span,
    )
    if not _is_expected_or_plus_two(len(probability_positions_rel), span_budget):
        return None
    probability_positions_rel = probability_positions_rel[:span_budget]

    return {
        "prompt": list(range(0, prompt_len - 1)),
        "guess": [_completion_token_index_to_abs_pos(prompt_len, k) for k in guess_positions_rel],
        "sem_answer": [
            _completion_token_index_to_abs_pos(prompt_len, k)
            for k in range(last_guess_token_index, first_prob_token_index)
        ],
        "probability": [
            _completion_token_index_to_abs_pos(prompt_len, k) for k in probability_positions_rel
        ],
    }


def _absolute_probability_value_start_position(
    prompt_len: int,
    decoded_tokens: List[str],
) -> Optional[int]:
    """Absolute position of first probability-value token (numeric Probability: parser only)."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return None
    _, _, end_prob_token_index = parsed
    seq_len = prompt_len + len(decoded_tokens)
    abs_pos = _completion_token_index_to_abs_pos(prompt_len, end_prob_token_index)
    if 0 <= abs_pos < seq_len:
        return abs_pos
    return None


def _absolute_all_pre_guess_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
) -> List[int]:
    guess_positions = _absolute_guess_span_positions(
        prompt_len,
        decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
    )
    if not guess_positions:
        return []
    return list(range(0, prompt_len - 1)) + guess_positions


def _generation_contains_stop(decoded_completion: str) -> bool:
    return any(s in decoded_completion for s in STOP_SEQUENCES)


def _eos_token_ids(tokenizer) -> set[int]:
    """Normalize tokenizer EOS ids (scalar or list) into a set of ints."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return set()
    if isinstance(eos_id, (list, tuple, set)):
        return {int(x) for x in eos_id if x is not None}
    return {int(eos_id)}


def _should_stop_generation(decoded_completion: str, next_id: int, tokenizer) -> bool:
    if _generation_contains_stop(decoded_completion):
        return True
    return next_id in _eos_token_ids(tokenizer)


def _strip_stop_suffixes(text: str) -> str:
    stop_at = len(text)
    for stop in STOP_SEQUENCES:
        if text.endswith(stop):
            stop_at = len(text) - len(stop)
            break
    return text[:stop_at].strip()


def _postprocess_response_from_full_decode(
    model: HookedTransformer,
    full_sequence_tokens: torch.Tensor,
    input_text: str,
) -> str:
    full_answer = model.tokenizer.decode(full_sequence_tokens[0], skip_special_tokens=True)
    if full_answer.startswith(input_text):
        answer = full_answer[len(input_text) :]
    else:
        logging.warning("Decoded text does not start with prompt; using full decoded text.")
        answer = full_answer
    return _strip_stop_suffixes(answer)


def greedy_generate(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    decoded_tokens: List[str] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks or [])
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def _positions_for_whole_sequence_mode(mode: str, *, prompt_len: int, seq_len: int) -> List[int]:
    if mode == "all_tokens_mean_replace":
        return list(range(seq_len))
    if mode == "generated_tokens_mean_replace":
        return list(range(prompt_len - 1, seq_len))
    raise ValueError(f"Unknown whole-sequence ablation mode: {mode!r}")


def _generated_token_direction_for_source(
    layer_delta: Dict[str, torch.Tensor],
    source: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if source == "probability_prefix_last_token":
        return layer_delta["probability"][-1].to(dtype)
    raise ValueError(f"Unknown generated_tokens_source: {source!r}")


def _apply_steering_at_positions_with_generated_source(
    activation: torch.Tensor,
    layer_delta: Dict[str, torch.Tensor],
    positions: Sequence[int],
    *,
    prompt_len: int,
    generated_tokens_source: str,
) -> torch.Tensor:
    prompt_vec = layer_delta["prompt_mean"].to(activation.dtype)
    generated_vec = _generated_token_direction_for_source(
        layer_delta,
        generated_tokens_source,
        dtype=activation.dtype,
    )
    for abs_pos in positions:
        if not (0 <= abs_pos < activation.shape[1]):
            continue
        delta_vec = prompt_vec if abs_pos < prompt_len - 1 else generated_vec
        activation[:, abs_pos, :] = activation[:, abs_pos, :] + delta_vec
    return activation


def _direction_mode_activation_applier_builder(
    mode: str,
    *,
    model_name: str,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    generated_tokens_source: str = "probability_prefix_last_token",
) -> Callable[[int, Callable[[], List[str]]], Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]]:
    def _builder(
        prompt_len: int,
        decoded_tokens_provider: Callable[[], List[str]],
    ) -> Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        if mode == "probability_tokens_mean_replace":
            def _apply_probability_tokens_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_vecs = layer_delta["probability"].to(activation.dtype)
                logging.info(f"mode: {mode}")
                prob_positions = _absolute_prob_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if not prob_positions:
                    return activation
                n_pos = len(prob_positions)
                if linguistic_confidence_prompt:
                    if n_pos > prob_vecs.shape[0]:
                        raise ValueError(
                            f"Linguistic confidence span has {n_pos} token positions but direction has "
                            f"{prob_vecs.shape[0]} probability components."
                        )
                    prob_use = prob_vecs[:n_pos]
                else:
                    if n_pos != prob_vecs.shape[0]:
                        raise ValueError(f"Expected {prob_vecs.shape[0]} probability tokens, got {n_pos}.")
                    prob_use = prob_vecs
                for pos_i, abs_pos in enumerate(prob_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_use[pos_i]
                return activation

            return _apply_probability_tokens_mean_replace

        if mode in PROBABILITY_ROW_INDEX_MODES:
            row_indices = probability_row_indices_for_mode(mode, model_name)

            def _apply_probability_row_indices_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_vecs = layer_delta["probability"].to(activation.dtype)
                prob_positions = _absolute_prob_positions_at_row_indices(
                    prompt_len,
                    decoded_tokens_provider(),
                    row_indices,
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if not prob_positions:
                    return activation
                for row_idx, abs_pos in zip(row_indices, prob_positions):
                    if row_idx < 0 or row_idx >= prob_vecs.shape[0]:
                        return activation
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_vecs[row_idx]
                return activation

            return _apply_probability_row_indices_mean_replace

        if mode == "probability_last_token_mean_replace":
            def _apply_probability_last_token_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_vecs = layer_delta["probability"].to(activation.dtype)
                prob_positions = _absolute_prob_last_token_only(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if not prob_positions:
                    return activation
                abs_pos = prob_positions[0]
                if 0 <= abs_pos < activation.shape[1]:
                    activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_vecs[-1]
                return activation

            return _apply_probability_last_token_mean_replace

        if mode == "probability_span_except_last_token_mean_replace":
            def _apply_probability_span_except_last_token_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_vecs = layer_delta["probability"].to(activation.dtype)
                prob_positions = _absolute_prob_marker_except_last_token(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if not prob_positions:
                    return activation
                n = min(len(prob_positions), max(0, prob_vecs.shape[0] - 1))
                if n == 0:
                    return activation
                for pos_i, abs_pos in enumerate(prob_positions[:n]):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_vecs[pos_i]
                return activation

            return _apply_probability_span_except_last_token_mean_replace

        if mode == "probability_value_mean_replace":
            def _apply_probability_value_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_vecs = layer_delta["probability"].to(activation.dtype)
                prob_value_mean = layer_delta["probability_value_mean"].to(activation.dtype)
                if prob_vecs.shape[0] < 1:
                    raise ValueError("Expected at least one probability direction row for probability_value mode.")

                start_abs = _absolute_probability_value_start_position(prompt_len, decoded_tokens_provider())
                if start_abs is None:
                    return activation
                seq_len = activation.shape[1]
                if not (0 <= start_abs < seq_len):
                    return activation

                # First probability-value token uses the last probability-span row.
                activation[:, start_abs, :] = activation[:, start_abs, :] + prob_vecs[-1]
                # Subsequent tail positions use a shared probability-value direction.
                if start_abs + 1 < seq_len:
                    activation[:, start_abs + 1 : seq_len, :] = (
                        activation[:, start_abs + 1 : seq_len, :] + prob_value_mean
                    )
                return activation

            return _apply_probability_value_mean_replace

        if mode == "guess_tokens_mean_replace":
            def _apply_guess_tokens_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                guess_vecs = layer_delta["guess"].to(activation.dtype)
                guess_positions = _absolute_guess_span_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_guess_tokens=expected_guess_tokens,
                )
                if not guess_positions:
                    return activation
                if len(guess_positions) != guess_vecs.shape[0]:
                    raise ValueError(f"Expected {guess_vecs.shape[0]} guess tokens, got {len(guess_positions)}.")
                for pos_i, abs_pos in enumerate(guess_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + guess_vecs[pos_i]
                return activation

            return _apply_guess_tokens_mean_replace

        if mode == "semantic_answer_mean_replace":
            def _apply_semantic_answer_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                sem_answer_vec = layer_delta["sem_answer_mean"].to(activation.dtype)
                sem_positions = _absolute_sem_answer_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if not sem_positions:
                    return activation
                for abs_pos in sem_positions:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + sem_answer_vec
                return activation

            return _apply_semantic_answer_mean_replace

        if mode == "all_pre_probability_tokens_mean_replace":
            def _apply_all_pre_probability_tokens_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prompt_vec = layer_delta["prompt_mean"].to(activation.dtype)
                guess_vecs = layer_delta["guess"].to(activation.dtype)
                sem_answer_vec = layer_delta["sem_answer_mean"].to(activation.dtype)
                prob_vecs = layer_delta["probability"].to(activation.dtype)

                positions = _absolute_pre_probability_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if positions is None:
                    return activation

                for abs_pos in positions["prompt"]:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prompt_vec
                if len(positions["guess"]) != guess_vecs.shape[0]:
                    raise ValueError(
                        f"Expected {guess_vecs.shape[0]} guess tokens, got {len(positions['guess'])}."
                    )
                for pos_i, abs_pos in enumerate(positions["guess"]):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + guess_vecs[pos_i]
                for abs_pos in positions["sem_answer"]:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + sem_answer_vec
                n_prob = len(positions["probability"])
                if linguistic_confidence_prompt:
                    if n_prob > prob_vecs.shape[0]:
                        raise ValueError(
                            f"Linguistic confidence span has {n_prob} token positions but direction has "
                            f"{prob_vecs.shape[0]} probability components."
                        )
                    prob_use = prob_vecs[:n_prob]
                else:
                    if n_prob != prob_vecs.shape[0]:
                        raise ValueError(
                            f"Expected {prob_vecs.shape[0]} probability tokens, got {n_prob}."
                        )
                    prob_use = prob_vecs
                for pos_i, abs_pos in enumerate(positions["probability"]):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_use[pos_i]
                return activation

            return _apply_all_pre_probability_tokens_mean_replace

        if mode == "all_pre_guess_tokens_mean_replace":
            def _apply_all_pre_guess_tokens_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prompt_vec = layer_delta["prompt_mean"].to(activation.dtype)
                guess_vecs = layer_delta["guess"].to(activation.dtype)

                all_pre_guess_positions = _absolute_all_pre_guess_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_guess_tokens=expected_guess_tokens,
                )
                if not all_pre_guess_positions:
                    return activation

                for abs_pos in all_pre_guess_positions[: prompt_len - 1]:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prompt_vec
                guess_positions = all_pre_guess_positions[prompt_len - 1 :]
                if len(guess_positions) != guess_vecs.shape[0]:
                    raise ValueError(f"Expected {guess_vecs.shape[0]} guess tokens, got {len(guess_positions)}.")
                for pos_i, abs_pos in enumerate(guess_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + guess_vecs[pos_i]
                return activation

            return _apply_all_pre_guess_tokens_mean_replace

        if mode == "guess_then_guess_probability_mean_replace":
            def _apply_guess_then_guess_probability_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                guess_vecs = layer_delta["guess"].to(activation.dtype)
                prob_vecs = layer_delta["probability"].to(activation.dtype)

                guess_positions = _absolute_guess_span_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_guess_tokens=expected_guess_tokens,
                )
                if guess_positions:
                    if len(guess_positions) != guess_vecs.shape[0]:
                        raise ValueError(
                            f"Expected {guess_vecs.shape[0]} guess tokens, got {len(guess_positions)}."
                        )
                    for pos_i, abs_pos in enumerate(guess_positions):
                        if 0 <= abs_pos < activation.shape[1]:
                            activation[:, abs_pos, :] = activation[:, abs_pos, :] + guess_vecs[pos_i]

                prob_positions = _absolute_prob_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if prob_positions:
                    n_pos = len(prob_positions)
                    if linguistic_confidence_prompt:
                        if n_pos > prob_vecs.shape[0]:
                            raise ValueError(
                                f"Linguistic confidence span has {n_pos} token positions but direction has "
                                f"{prob_vecs.shape[0]} probability components."
                            )
                        prob_use = prob_vecs[:n_pos]
                    else:
                        if n_pos != prob_vecs.shape[0]:
                            raise ValueError(
                                f"Expected {prob_vecs.shape[0]} probability tokens, got {n_pos}."
                            )
                        prob_use = prob_vecs
                    for pos_i, abs_pos in enumerate(prob_positions):
                        if 0 <= abs_pos < activation.shape[1]:
                            activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_use[pos_i]
                return activation

            return _apply_guess_then_guess_probability_mean_replace

        if mode == "current_generated_token_mean_replace":
            def _apply_current_generated_token_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_last_vec = layer_delta["probability"][-1].to(activation.dtype)
                abs_pos = int(activation.shape[1]) - 1
                if 0 <= abs_pos < activation.shape[1]:
                    activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_last_vec
                return activation

            return _apply_current_generated_token_mean_replace

        if mode == "current_generated_window5_mean_replace":
            def _apply_current_generated_window5_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                prob_last_vec = layer_delta["probability"][-1].to(activation.dtype)
                seq_len = int(activation.shape[1])
                for abs_pos in range(max(0, seq_len - 5), seq_len):
                    activation[:, abs_pos, :] = activation[:, abs_pos, :] + prob_last_vec
                return activation

            return _apply_current_generated_window5_mean_replace

        if mode == "prompt_tokens_mean_replace":
            def _apply_prompt_tokens_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                if parse_guess_start_index(decoded_tokens_provider()) is None:
                    return activation
                prompt_vec = layer_delta["prompt_mean"].to(activation.dtype)
                for abs_pos in range(0, prompt_len - 1):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + prompt_vec
                return activation

            return _apply_prompt_tokens_mean_replace

        if mode == "sem_ans_tokens_during_gen":
            def _apply_sem_ans_tokens_during_gen(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                sem_answer_vec = layer_delta["sem_answer_mean"].to(activation.dtype)
                sem_positions = _absolute_sem_answer_positions_during_gen(
                    prompt_len,
                    decoded_tokens_provider(),
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                if not sem_positions:
                    return activation
                for abs_pos in sem_positions:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + sem_answer_vec
                return activation

            return _apply_sem_ans_tokens_during_gen

        if mode in ("all_tokens_mean_replace", "generated_tokens_mean_replace"):

            def _apply_whole_sequence_mean_replace(
                activation: torch.Tensor,
                layer_delta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                seq_len = int(activation.shape[1])
                positions = _positions_for_whole_sequence_mode(
                    mode, prompt_len=prompt_len, seq_len=seq_len
                )
                return _apply_steering_at_positions_with_generated_source(
                    activation,
                    layer_delta,
                    positions,
                    prompt_len=prompt_len,
                    generated_tokens_source=generated_tokens_source,
                )

            return _apply_whole_sequence_mean_replace

        raise ValueError(f"Unknown ablation mode for perturb hooks: {mode!r}")

    return _builder


def build_direction_perturb_hooks(
    layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]],
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    generated_tokens_source: str = "probability_prefix_last_token",
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    activation_applier_builder = _direction_mode_activation_applier_builder(
        mode,
        model_name=model_name,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        generated_tokens_source=generated_tokens_source,
    )
    activation_applier = activation_applier_builder(prompt_len, decoded_tokens_provider)
    for layer in layer_to_span_delta:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                layer_delta = layer_to_span_delta[layer_idx]
                return activation_applier(activation, layer_delta)

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_direction_perturbed(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]],
    mode: str,
    model_name: str,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
    generated_tokens_source: str = "probability_prefix_last_token",
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_direction_perturb_hooks(
        layer_to_span_delta=layer_to_span_delta,
        mode=mode,
        model_name=model_name,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
        generated_tokens_source=generated_tokens_source,
    )
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def compute_low_high_span_means_and_directions(
    examples_h5: Dict[str, dict],
    ablate_layers: Sequence[int],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
    new_h5_format: bool = False,
    h5_res_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], set[str], set[str]]:
    low_vectors: Dict[str, List[np.ndarray]] = {
        "prompt_mean": [],
        "guess": [],
        "sem_answer_mean": [],
        "probability": [],
        "probability_value_mean": [],
    }
    high_vectors: Dict[str, List[np.ndarray]] = {
        "prompt_mean": [],
        "guess": [],
        "sem_answer_mean": [],
        "probability": [],
        "probability_value_mean": [],
    }
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    resid_post_layers = (
        np.asarray(h5_res_indices)
        if h5_res_indices is not None
        else np.asarray(ablate_layers) + 1
    )

    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError(f"Example {ex_id} must have exactly one response, got {0 if responses is None else len(responses)}.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 is not a dict.")

        conf = float(resp0.get("verbalised_confidence"))
        is_low = conf <= low_conf_threshold
        is_high = conf >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        if not (is_low or is_high):
            continue

        emb_prompt = _extract_res_field(
            resp0, ex_id, "embeddings_mean_prompt", new_h5_format=new_h5_format
        )
        emb_guess = _extract_res_field(
            resp0, ex_id, "embeddings_guess", new_h5_format=new_h5_format
        )
        emb_sem_answer = _extract_res_field(
            resp0, ex_id, "embeddings_mean_sem_answer", new_h5_format=new_h5_format
        )
        emb_prob = _extract_res_field(
            resp0, ex_id, "embeddings_probability", new_h5_format=new_h5_format
        )
        emb_prob_val = _extract_res_field(
            resp0, ex_id, "embeddings_mean_prob_val", new_h5_format=new_h5_format
        )
        if emb_prompt is None or emb_guess is None or emb_sem_answer is None or emb_prob is None or emb_prob_val is None:
            raise ValueError(
                f"Example {ex_id} is missing one of required fields: "
                "embeddings_mean_prompt, embeddings_guess, embeddings_mean_sem_answer, "
                "embeddings_probability, embeddings_mean_prob_val."
            )
        if not isinstance(emb_guess, list):
            raise ValueError(f"Example {ex_id} responses/0/embeddings_guess must be a list.")
        if len(emb_guess) != expected_guess_tokens:
            raise ValueError(
                f"Example {ex_id} embeddings_guess len={len(emb_guess)}; "
                f"expected {expected_guess_tokens}."
            )
        emb_guess = emb_guess[:expected_guess_tokens]
        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} responses/0/embeddings_probability must be a list.")
        if not _is_expected_or_plus_two(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability len={len(emb_prob)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        prompt_selected = _as_layer_hidden(emb_prompt)[resid_post_layers, :]
        sem_answer_selected = _as_layer_hidden(emb_sem_answer)[resid_post_layers, :]
        guess_selected = np.stack(
            [_as_layer_hidden(tok_arr)[resid_post_layers, :] for tok_arr in emb_guess], axis=1
        )
        prob_selected = np.stack(
            [_as_layer_hidden(tok_arr)[resid_post_layers, :] for tok_arr in emb_prob], axis=1
        )
        prob_val_selected = _as_layer_hidden(emb_prob_val)[resid_post_layers, :]
        if is_low:
            low_vectors["prompt_mean"].append(prompt_selected)
            low_vectors["guess"].append(guess_selected)
            low_vectors["sem_answer_mean"].append(sem_answer_selected)
            low_vectors["probability"].append(prob_selected)
            low_vectors["probability_value_mean"].append(prob_val_selected)
        if is_high:
            high_vectors["prompt_mean"].append(prompt_selected)
            high_vectors["guess"].append(guess_selected)
            high_vectors["sem_answer_mean"].append(sem_answer_selected)
            high_vectors["probability"].append(prob_selected)
            high_vectors["probability_value_mean"].append(prob_val_selected)

    if not low_vectors["probability"]:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_vectors["probability"]:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")

    mean_low: Dict[str, np.ndarray] = {}
    mean_high: Dict[str, np.ndarray] = {}
    direction: Dict[str, np.ndarray] = {}
    for key in ("prompt_mean", "guess", "sem_answer_mean", "probability", "probability_value_mean"):
        mean_low[key] = np.mean(np.stack(low_vectors[key], axis=0), axis=0).astype(np.float32)
        mean_high[key] = np.mean(np.stack(high_vectors[key], axis=0), axis=0).astype(np.float32)
        direction[key] = (mean_high[key] - mean_low[key]).astype(np.float32)
    return mean_low, mean_high, direction, low_ids, high_ids


def normalize_direction_spans_to_unit_norm_budget(
    direction_by_span: Dict[str, np.ndarray],
    *,
    spans: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for span in spans:
        if span not in direction_by_span:
            raise ValueError(f"Cannot normalize missing span directions: {span!r}")
        span_direction = direction_by_span[span]
        if span_direction.ndim != 3:
            raise ValueError(
                f"Expected direction[{span!r}] to have shape (layers, token_positions, d_model), "
                f"got ndim={span_direction.ndim} shape={span_direction.shape}."
            )

        num_layers, num_token_positions, _ = span_direction.shape
        target_sum = float(num_layers * num_token_positions)
        sum_before = float(np.sum(np.linalg.norm(span_direction, axis=-1)))

        if sum_before <= 0.0:
            logging.warning(
                "Skipping direction normalization for span=%s because sum of norms is zero.",
                span,
            )
            scale = 1.0
        else:
            scale = target_sum / sum_before
            direction_by_span[span] = (span_direction * scale).astype(np.float32, copy=False)

        sum_after = float(np.sum(np.linalg.norm(direction_by_span[span], axis=-1)))
        stats[span] = {
            "target_sum": target_sum,
            "sum_before": sum_before,
            "sum_after": sum_after,
            "scale": float(scale),
        }
    return stats


def write_layer_direction_pickles(
    *,
    output_json_path: str,
    ablate_layers: Sequence[int],
    direction_by_span: Dict[str, np.ndarray],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> None:
    out_dir = os.path.dirname(output_json_path)
    for layer_i, layer_idx in enumerate(ablate_layers):
        payload = {
            "layer_idx": int(layer_idx),
            "expected_guess_tokens": int(expected_guess_tokens),
            "expected_probability_tokens": int(expected_probability_tokens),
            "prompt_mean_direction": direction_by_span["prompt_mean"][layer_i],
            "guess_prefix_directions": [direction_by_span["guess"][layer_i, tok_i] for tok_i in range(expected_guess_tokens)],
            "semantic_answer_mean_direction": direction_by_span["sem_answer_mean"][layer_i],
            "probability_prefix_directions": [
                direction_by_span["probability"][layer_i, tok_i] for tok_i in range(expected_probability_tokens)
            ],
            "probability_value_mean_direction": direction_by_span["probability_value_mean"][layer_i],
        }
        pickle_path = os.path.join(out_dir, f"layer_{layer_idx}_directions.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(payload, f)
        logging.info("Wrote %s", pickle_path)


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mass mean direction probe inference (TransformerLens).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-12b-it",
        choices=list(SUPPORTED_MODEL_NAMES),
    )
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, expect the new process_generations_more_embs_from_h5.py output where each "
            "embedding field is a dict containing 'res', 'attn', and 'mlp' subfields. Validates "
            "all three are present per example and reads from 'res' (attn/mlp are not used for "
            "mass-mean residual-stream direction probing)."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["trivia_qa", "squad", "bioasq", "nq", "svamp", "gsm8k"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=400,
        help="Max examples per confidence group (sorted IDs, then truncated). No train/val ratio split.",
    )
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
        help=(
            "Inclusive range '12-15' or comma list '12,13,14,15' (display indices: "
            "0=embedding resid-pre, k>=1=resid-post of TL block k-1)."
        ),
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=[
            "none",
            "probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "probability_value_mean_replace",
            "semantic_answer_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
            "prompt_tokens_mean_replace",
            "sem_ans_tokens_during_gen",
            "current_generated_token_mean_replace",
            "current_generated_window5_mean_replace",
            "last_a_mean_replace",
            "last_a_and_panl_mean_replace",
            "last_a_panl_and_pc_mean_replace",
            "panl_mean_replace",
            "pc_mean_replace",
            "all_tokens_mean_replace",
            "generated_tokens_mean_replace",
        ],
        choices=[
            "none",
            "probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "probability_value_mean_replace",
            "semantic_answer_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
            "prompt_tokens_mean_replace",
            "sem_ans_tokens_during_gen",
            "current_generated_token_mean_replace",
            "current_generated_window5_mean_replace",
            "last_a_mean_replace",
            "last_a_and_panl_mean_replace",
            "last_a_panl_and_pc_mean_replace",
            "panl_mean_replace",
            "pc_mean_replace",
            "all_tokens_mean_replace",
            "generated_tokens_mean_replace",
        ],
        help=(
            "Ablation mode(s) to run. probability_last_token_mean_replace: perturb only the "
            "last token in the Probability: marker span (last H5 probability row). "
            "probability_span_except_last_token_mean_replace: perturb all Probability: span "
            "tokens except that last span token (remaining H5 probability rows). "
            "current_generated_token_mean_replace: perturb only the current last sequence token "
            "at each decode step using the probability-last direction. "
            "current_generated_window5_mean_replace: perturb the current last 5 sequence tokens "
            "at each decode step using the probability-last direction. "
            "last_a_mean_replace: same gating as probability_tokens_mean_replace but "
            "only H5 probability row 0 (last answer token). last_a_and_panl_mean_replace: rows 0 and 1 "
            "(last answer token and post-answer newline). "
            "last_a_panl_and_pc_mean_replace: same gating as probability_tokens_mean_replace but only "
            "H5 probability rows 0, 1, and a model-specific pre-confidence index "
            "(Mistral 6, Gemma 3; unsupported for Qwen). "
            "panl_mean_replace: only H5 probability row 1 (post-answer newline). "
            "pc_mean_replace: only the model-specific pre-confidence index "
            "(Mistral 6, Gemma 3; unsupported for Qwen). "
            "semantic_answer_mean_replace: no-op until Guess/Probability(or Confidence) parse "
            "succeeds, then perturb only semantic-answer tokens using the shared "
            "sem_answer_mean direction. "
            "prompt_tokens_mean_replace: no-op until Guess: prefix parses, then perturb all "
            "prompt token positions with the shared prompt_mean direction. "
            "sem_ans_tokens_during_gen: no-op until Guess: prefix parses; perturb semantic-answer "
            "tokens generated so far (and keep ablating those positions after the span is complete). "
            "all_tokens_mean_replace: perturb every sequence position at each decode step; "
            "prompt_mean for prompt positions, selected --generated_tokens_source for all other "
            "positions (no span parsing). "
            "generated_tokens_mean_replace: same steering logic, but only generated positions "
            "(prompt_len-1 onward). "
            "Both subset modes apply only with the numeric Probability: prompt."
        ),
    )
    parser.add_argument(
        "--generated_tokens_source",
        type=str,
        nargs="+",
        default=["probability_prefix_last_token"],
        choices=list(GENERATED_TOKENS_SOURCE_CHOICES),
        help=(
            "Generated-token direction source for all_tokens_mean_replace and "
            "generated_tokens_mean_replace. Prompt positions always use prompt_mean; every "
            "non-prompt position uses the selected source. Currently only "
            "probability_prefix_last_token (probability[-1]) is supported."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        required=True,
        help="One or more real-valued alpha scale factors (perturbation strength along the direction).",
    )
    parser.add_argument(
        "--ablation_targets",
        type=str,
        nargs="+",
        required=True,
        choices=["low", "high"],
        help="One or both targets to perturb: low, high.",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument(
        "--expected_confidence_tokens",
        type=int,
        default=5,
        help=(
            "When --linguistic_confidence_prompt, expected number of completion tokens in the Confidence: "
            "span for position parsing checks and truncation (instead of --expected_probability_tokens)."
        ),
    )
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--normalize_span_directions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, normalize guess/probability directions so each span's summed per-(layer,token) "
            "vector norms equals num_layers * num_token_positions."
        ),
    )
    parser.add_argument(
        "--parse_mode_verbalised_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse verbalised confidence from generated responses and report aggregate means.",
    )
    parser.add_argument(
        "--linguistic_confidence_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, use the natural-language Confidence: prompt and map phrases to numeric confidence; "
            "if false (default), use the numeric Probability: prompt and float parsing."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "mass_mean_probe/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    args = parser.parse_args()
    configure_prefix_tokens_for_model(args.model_name)
    validate_last_a_panl_and_pc_mode(args.model_name, args.ablation_mode)
    if args.linguistic_confidence_prompt and not CONFIDENCE_PREFIX_TOKENS:
        raise ValueError(
            "--linguistic_confidence_prompt requires a non-empty Confidence: prefix table "
            f"for --model_name={args.model_name!r}. Fill GEMMA/QWEN_CONFIDENCE_PREFIX_TOKENS "
            "in process_generations_more_embs_from_h5.py or use Mistral."
        )

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline comparisons.")
    if any(m in WHOLE_SEQUENCE_MODES for m in args.ablation_mode):
        if len(args.generated_tokens_source) != 1:
            raise ValueError(
                "--generated_tokens_source must specify exactly one value when running "
                f"whole-sequence modes; got {args.generated_tokens_source}"
            )
    generated_tokens_source = args.generated_tokens_source[0]
    if args.normalize_span_directions:
        normalization_supported_modes = {
            "none",
            "probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "guess_tokens_mean_replace",
            "semantic_answer_mean_replace",
            "guess_then_guess_probability_mean_replace",
            "current_generated_token_mean_replace",
            "current_generated_window5_mean_replace",
            "prompt_tokens_mean_replace",
            "sem_ans_tokens_during_gen",
        } | set(PROBABILITY_ROW_INDEX_MODES)
        unsupported_modes = [mode for mode in args.ablation_mode if mode not in normalization_supported_modes]
        if unsupported_modes:
            raise ValueError(
                "normalize_span_directions only supports ablation modes "
                f"{sorted(normalization_supported_modes)}. Unsupported: {unsupported_modes}."
            )
    args.ablation_targets = _dedupe_preserve_order(args.ablation_targets)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    alphas_outside_unit = [a for a in args.alpha if a < 0.0 or a > 1.0]
    if alphas_outside_unit:
        logging.warning(
            "Alpha values outside [0, 1] will be used as-is: %s",
            alphas_outside_unit,
        )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_eval_dataset(args.dataset, args.random_seed)
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

    confidence_prompt = (
        CONFIDENCE_PROMPT_LINGUISTIC if args.linguistic_confidence_prompt else CONFIDENCE_PROMPT_NUMERIC
    )

    logging.info("Loading HookedTransformer: %s", args.model_name)
    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers + 1)

    examples_h5 = load_examples_h5(Path(args.input_h5))
    mean_low, mean_high, direction, low_ids, high_ids = compute_low_high_span_means_and_directions(
        examples_h5,
        ablate_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_guess_tokens=args.expected_guess_tokens,
        new_h5_format=args.new_h5_format,
        h5_res_indices=ablate_layers,
    )
    if args.normalize_span_directions:
        # Mutates direction dict in place
        normalization_stats = normalize_direction_spans_to_unit_norm_budget(
            direction,
            spans=("guess", "probability"),
        )
        for span_name, span_stats in normalization_stats.items():
            logging.info(
                "Normalized span=%s direction norms: before=%.6f after=%.6f target=%.6f scale=%.6f",
                span_name,
                span_stats["sum_before"],
                span_stats["sum_after"],
                span_stats["target_sum"],
                span_stats["scale"],
            )
    del mean_low, mean_high

    direction_flat_head = np.ravel(direction["probability"])[:5].tolist()
    logging.info(
        "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), layers=%s, "
        "direction.shape=%s direction.flat[:5]=%s alphas=%s",
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        ablate_layers,
        direction["probability"].shape,
        direction_flat_head,
        list(args.alpha),
    )
    eval_low_ids = set(truncate_sorted_ids(low_ids, args.num_samples))
    eval_high_ids = set(truncate_sorted_ids(high_ids, args.num_samples))
    eval_ids = eval_low_ids | eval_high_ids
    logging.info(
        "Eval sample cap num_samples=%d: eval_low=%d eval_high=%d (from full low=%d high=%d)",
        args.num_samples,
        len(eval_low_ids),
        len(eval_high_ids),
        len(low_ids),
        len(high_ids),
    )

    out_path = resolve_output_json_path(args.output_json)

    sentence_transformer = None
    compute_derived_metrics = "none" in args.ablation_mode and len(args.ablation_mode) > 1
    if compute_derived_metrics and any(m in SEMANTIC_SIMILARITY_MODES for m in args.ablation_mode):
        logging.info(
            "Loading sentence-transformers model %s for semantic_similarity.",
            DEFAULT_SEMANTIC_SIMILARITY_MODEL,
        )
        sentence_transformer = load_sentence_transformer_for_metrics()

    results = {"train": {}, "validation": {}}
    mini_results = {"train": {}, "validation": {}}
    mode_confidence_values: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        mode: {target: {float(alpha): [] for alpha in args.alpha} for target in args.ablation_targets}
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    mode_responses_identical_true: Dict[str, Dict[str, Dict[float, int]]] = {
        mode: {target: {float(alpha): 0 for alpha in args.alpha} for target in args.ablation_targets}
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    mode_semantic_similarity_values: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        mode: {target: {float(alpha): [] for alpha in args.alpha} for target in args.ablation_targets}
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    mode_verbalised_confidence_effect_values: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        mode: {target: {float(alpha): [] for alpha in args.alpha} for target in args.ablation_targets}
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    mode_uncertainty_score_values: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        mode: {target: {float(alpha): [] for alpha in args.alpha} for target in args.ablation_targets}
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    pending_semantic_similarity: List[
        Tuple[str, str, str, str, float, str, str, str]
    ] = []
    baseline_values_by_target: Dict[str, List[float]] = {"low": [], "high": []}
    non_none_modes = [m for m in args.ablation_mode if m != "none"]

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
        selected_ids = sorted(ex_id for ex_id in eval_ids if ex_id in id_to_index)
        logging.info("Generating for %d examples (%s split).", len(selected_ids), split_name)

        for i, ex_id in enumerate(selected_ids):
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                continue
            example = eval_ds[int(ds_idx)]
            local_prompt = fewshot_prefix + confidence_prompt + example["question"]
            entry = {"question": example["question"]}
            mini_entry = {"question": example["question"]}

            # TODO: why is the baseline compulsory?

            baseline_response, baseline_decoded_tokens = greedy_generate(
                model=model,
                local_prompt=local_prompt,
                max_new_tokens=args.model_max_new_tokens,
                fwd_hooks=None,
            )
            baseline_confidence = (
                parse_mode_confidence_from_response(
                    baseline_response, linguistic_prompt=args.linguistic_confidence_prompt
                )
                if args.parse_mode_verbalised_confidence
                else None
            )
            entry["no_replacement"] = {
                "response": baseline_response,
                "decoded_tokens": baseline_decoded_tokens,
            }
            mini_entry["no_replacement"] = {"response": baseline_response}
            if args.parse_mode_verbalised_confidence:
                entry["no_replacement"]["verbalised_confidence"] = baseline_confidence
                mini_entry["no_replacement"]["verbalised_confidence"] = baseline_confidence
            baseline_semantic_answer = parse_semantic_answer_from_response(
                baseline_response,
                linguistic_confidence_prompt=args.linguistic_confidence_prompt,
            )
            if baseline_semantic_answer is not None:
                entry["no_replacement"]["semantic_answer"] = baseline_semantic_answer
                mini_entry["no_replacement"]["semantic_answer"] = baseline_semantic_answer

            ex_is_low = ex_id in low_ids
            ex_is_high = ex_id in high_ids
            if args.parse_mode_verbalised_confidence and baseline_confidence is not None:
                if ex_is_low:
                    baseline_values_by_target["low"].append(float(baseline_confidence))
                if ex_is_high:
                    baseline_values_by_target["high"].append(float(baseline_confidence))

            for target in args.ablation_targets:
                # Only evaluate target/group combinations this example belongs to.
                if target == "low" and not ex_is_low:
                    continue
                if target == "high" and not ex_is_high:
                    continue
                sign = 1.0 if target == "low" else -1.0
                for mode in non_none_modes:
                    for alpha in args.alpha:
                        layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]] = {}
                        for layer_i, layer_idx in enumerate(ablate_layers):
                            layer_to_span_delta[layer_idx] = {
                                "prompt_mean": torch.tensor(
                                    sign * alpha * direction["prompt_mean"][layer_i], device=device, dtype=torch_dtype
                                ),
                                "guess": torch.tensor(
                                    sign * alpha * direction["guess"][layer_i], device=device, dtype=torch_dtype
                                ),
                                "sem_answer_mean": torch.tensor(
                                    sign * alpha * direction["sem_answer_mean"][layer_i], device=device, dtype=torch_dtype
                                ),
                                "probability": torch.tensor(
                                    sign * alpha * direction["probability"][layer_i], device=device, dtype=torch_dtype
                                ),
                                "probability_value_mean": torch.tensor(
                                    sign * alpha * direction["probability_value_mean"][layer_i],
                                    device=device,
                                    dtype=torch_dtype,
                                ),
                            }
                        response, decoded_tokens = greedy_generate_direction_perturbed(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_span_delta=layer_to_span_delta,
                            mode=mode,
                            model_name=args.model_name,
                            expected_guess_tokens=args.expected_guess_tokens,
                            expected_probability_tokens=args.expected_probability_tokens,
                            expected_confidence_tokens=args.expected_confidence_tokens,
                            linguistic_confidence_prompt=args.linguistic_confidence_prompt,
                            generated_tokens_source=generated_tokens_source,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(
                                response, linguistic_prompt=args.linguistic_confidence_prompt
                            )
                            if args.parse_mode_verbalised_confidence
                            else None
                        )

                        key = _entry_key_for_mode_target_alpha(mode, target, alpha)
                        entry[key] = {"response": response, "decoded_tokens": decoded_tokens}
                        mini_entry[key] = {"response": response}
                        mode_semantic_answer = parse_semantic_answer_from_response(
                            response,
                            linguistic_confidence_prompt=args.linguistic_confidence_prompt,
                        )
                        if mode_semantic_answer is not None:
                            entry[key]["semantic_answer"] = mode_semantic_answer
                            mini_entry[key]["semantic_answer"] = mode_semantic_answer
                        responses_identical = response == baseline_response
                        entry[key]["responses_identical"] = responses_identical
                        mini_entry[key]["responses_identical"] = responses_identical
                        if args.parse_mode_verbalised_confidence:
                            entry[key]["verbalised_confidence"] = mode_confidence
                            mini_entry[key]["verbalised_confidence"] = mode_confidence
                            if mode_confidence is not None:
                                mode_confidence_values[mode][target][float(alpha)].append(float(mode_confidence))
                            if responses_identical:
                                mode_responses_identical_true[mode][target][float(alpha)] += 1
                            if mode_confidence is None or baseline_confidence is None:
                                meets_none_confidence_direction = None
                            elif target == "low":
                                meets_none_confidence_direction = mode_confidence > baseline_confidence
                            else:
                                meets_none_confidence_direction = mode_confidence < baseline_confidence
                            entry[key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                            mini_entry[key]["meets_none_confidence_direction"] = meets_none_confidence_direction

            if compute_derived_metrics:
                if args.parse_mode_verbalised_confidence and baseline_confidence is not None:
                    for target in args.ablation_targets:
                        if target == "low" and not ex_is_low:
                            continue
                        if target == "high" and not ex_is_high:
                            continue
                        mean_from_low_confidence = target == "high"
                        for mode in non_none_modes:
                            for alpha in args.alpha:
                                key = _entry_key_for_mode_target_alpha(mode, target, alpha)
                                if key not in entry:
                                    continue
                                mode_confidence = entry[key].get("verbalised_confidence")
                                if mode_confidence is None:
                                    continue
                                vce = compute_verbalised_confidence_effect(
                                    float(baseline_confidence),
                                    float(mode_confidence),
                                    mean_from_low_confidence=mean_from_low_confidence,
                                )
                                if vce is not None:
                                    entry[key]["verbalised_confidence_effect"] = vce
                                    mini_entry[key]["verbalised_confidence_effect"] = vce
                                    mode_verbalised_confidence_effect_values[mode][target][float(alpha)].append(
                                        float(vce)
                                    )

                if sentence_transformer is not None and baseline_semantic_answer is not None:
                    for mode in non_none_modes:
                        if mode not in SEMANTIC_SIMILARITY_MODES:
                            continue
                        for target in args.ablation_targets:
                            if target == "low" and not ex_is_low:
                                continue
                            if target == "high" and not ex_is_high:
                                continue
                            for alpha in args.alpha:
                                key = _entry_key_for_mode_target_alpha(mode, target, alpha)
                                if key not in entry:
                                    continue
                                mode_semantic_answer = entry[key].get("semantic_answer")
                                if mode_semantic_answer is None:
                                    continue
                                pending_semantic_similarity.append(
                                    (
                                        split_name,
                                        ex_id,
                                        mode,
                                        target,
                                        float(alpha),
                                        key,
                                        str(baseline_semantic_answer),
                                        str(mode_semantic_answer),
                                    )
                                )

            results[split_name][ex_id] = entry
            mini_results[split_name][ex_id] = mini_entry
            logging.info("[%s %d/%d] %s first line: %r", split_name, i + 1, len(selected_ids), ex_id, baseline_response[:120])

    if pending_semantic_similarity and sentence_transformer is not None:
        pairs = [(baseline_text, mode_text) for *_rest, baseline_text, mode_text in pending_semantic_similarity]
        similarities = batch_compute_semantic_similarities(sentence_transformer, pairs)
        for task, similarity in zip(pending_semantic_similarity, similarities):
            split_name, ex_id, mode, target, alpha, key, _baseline_text, _mode_text = task
            entry = results[split_name][ex_id][key]
            mini_entry = mini_results[split_name][ex_id][key]
            entry["semantic_similarity"] = similarity
            mini_entry["semantic_similarity"] = similarity
            mode_semantic_similarity_values[mode][target][alpha].append(similarity)
            vce = entry.get("verbalised_confidence_effect")
            if vce is not None:
                uncertainty = compute_uncertainty_score(similarity, float(vce))
                entry["uncertainty_score"] = uncertainty
                mini_entry["uncertainty_score"] = uncertainty
                mode_uncertainty_score_values[mode][target][alpha].append(uncertainty)

    derived_metric_kwargs: Dict[str, object] = {}
    if compute_derived_metrics:
        derived_metric_kwargs = {
            "mode_semantic_similarity_values": mode_semantic_similarity_values,
            "mode_verbalised_confidence_effect_values": mode_verbalised_confidence_effect_values,
            "mode_uncertainty_score_values": mode_uncertainty_score_values,
        }

    summary_payload = _build_summary_json(
        non_none_modes=non_none_modes,
        ablation_targets=args.ablation_targets,
        alphas=args.alpha,
        mode_confidence_values=mode_confidence_values,
        mode_responses_identical_true=mode_responses_identical_true,
        baseline_values_by_target=baseline_values_by_target,
        **derived_metric_kwargs,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    mini_out_path = mini_output_json_path(out_path)
    with open(mini_out_path, "w", encoding="utf-8") as f:
        json.dump(mini_results, f, ensure_ascii=False, indent=2)
    summary_out_path = summary_json_path(out_path)
    write_summary_json(summary_out_path, summary_payload)
    write_summary_plots_from_file(
        summary_json_file=summary_out_path,
        ablation_targets=args.ablation_targets,
        output_dir=os.path.dirname(out_path),
    )
    write_layer_direction_pickles(
        output_json_path=out_path,
        ablate_layers=ablate_layers,
        direction_by_span=direction,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
    )
    config_out_path = config_txt_path(out_path)
    write_config_txt(
        config_out_path,
        args=args,
        device=device,
        model_n_layers=model.cfg.n_layers,
        ablate_layers=ablate_layers,
        prompt_indices=prompt_indices,
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        summary_payload=summary_payload if compute_derived_metrics else None,
    )
    logging.info("Wrote %s", out_path)
    logging.info("Wrote %s", mini_out_path)
    logging.info("Wrote %s", summary_out_path)
    logging.info("Wrote %s", config_out_path)


if __name__ == "__main__":
    main()
