#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Greedy decoding on TriviaQA with layerwise mean activation replacement via TransformerLens.

Mean activations are computed from low verbalised-confidence examples
(responses/0/verbalised_confidence <= 0.1) in a processed verbalised-embeddings H5
file. During generation for high verbalised-confidence examples
(responses/0/verbalised_confidence >= 0.9), activations at the "Probability:" span
positions are replaced with those low-confidence means.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import logging
import os
import random
import re
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

# ---------------------------------------------------------------------------
# Prompts (match generate_answers_with_confidence.py / utils defaults)
# ---------------------------------------------------------------------------

CONFIDENCE_PROMPT = (
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

# ---------------------------------------------------------------------------
# Probability span parsing (token-prefix match)
# ---------------------------------------------------------------------------

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

SUPPORTED_MODEL_NAMES = (
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-32B-Instruct",
)

# Active tables; set via configure_prefix_tokens_for_model(model_name).
GUESS_PREFIX_TOKENS: list[list[str]] = GEMMA_GUESS_PREFIX_TOKENS
PROBABILITY_PREFIX_TOKENS: list[list[str]] = GEMMA_PROBABILITY_PREFIX_TOKENS


def configure_prefix_tokens_for_model(model_name: str) -> None:
    """Set GUESS/PROBABILITY_PREFIX_TOKENS from exact model_name (case-sensitive)."""
    global GUESS_PREFIX_TOKENS, PROBABILITY_PREFIX_TOKENS
    if model_name == "google/gemma-3-12b-it":
        GUESS_PREFIX_TOKENS = GEMMA_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = GEMMA_PROBABILITY_PREFIX_TOKENS
    elif model_name == "Qwen/Qwen2.5-32B-Instruct":
        GUESS_PREFIX_TOKENS = QWEN_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = QWEN_PROBABILITY_PREFIX_TOKENS
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        GUESS_PREFIX_TOKENS = MISTRAL_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = MISTRAL_PROBABILITY_PREFIX_TOKENS
    else:
        raise ValueError(
            f"Unsupported model_name for Guess/Probability token parsing: {model_name!r}. "
            f"Supported: {list(SUPPORTED_MODEL_NAMES)}."
        )

GUESS_SPAN_ABLATION_MODES = frozenset({
    "guess_tokens_mean_replace",
    "all_pre_guess_tokens_mean_replace",
    "guess_then_guess_and_probability_tokens_mean_replace",
})

DEFAULT_SEMANTIC_SIMILARITY_MODEL = "all-MiniLM-L6-v2"

LAST_A_MODE = "last_a_mean_replace"
LAST_A_AND_PANL_MODE = "last_a_and_panl_mean_replace"
LAST_A_PANL_AND_PC_MODE = "last_a_panl_and_pc_mean_replace"
PANL_MODE = "panl_mean_replace"
PC_MODE = "pc_mean_replace"

PROBABILITY_ROW_INDEX_MODES = frozenset({
    LAST_A_MODE,
    LAST_A_AND_PANL_MODE,
    LAST_A_PANL_AND_PC_MODE,
    PANL_MODE,
    PC_MODE,
})

_PC_INDEX_MODES = frozenset({LAST_A_PANL_AND_PC_MODE, PC_MODE})


def _pc_row_index(model_name: str) -> int:
    if model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        return 6
    if model_name == "google/gemma-3-12b-it":
        return 3
    if model_name == "Qwen/Qwen2.5-32B-Instruct":
        raise ValueError(f"PC-index ablation modes are not defined for {model_name}")
    raise ValueError(
        f"PC-index ablation modes are not defined for {model_name!r}. "
        f"Supported: {list(SUPPORTED_MODEL_NAMES)}."
    )


def probability_row_indices_for_mode(mode: str, model_name: str) -> Tuple[int, ...]:
    if mode == LAST_A_MODE:
        return (0,)
    if mode == LAST_A_AND_PANL_MODE:
        return (0, 1)
    if mode == PANL_MODE:
        return (1,)
    if mode == LAST_A_PANL_AND_PC_MODE:
        return (0, 1, _pc_row_index(model_name))
    if mode == PC_MODE:
        return (_pc_row_index(model_name),)
    raise ValueError(f"Unknown probability-row-index ablation mode: {mode!r}")


def validate_last_a_panl_and_pc_mode(model_name: str, ablation_modes: Sequence[str]) -> None:
    requested_pc_modes = [mode for mode in ablation_modes if mode in _PC_INDEX_MODES]
    if requested_pc_modes and model_name == "Qwen/Qwen2.5-32B-Instruct":
        raise ValueError(
            f"{', '.join(requested_pc_modes)} is not defined for {model_name}"
        )


def _match_token_prefix(
    decoded_tokens: list,
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


def parse_guess_and_probability_indices(
    decoded_tokens: list,
) -> tuple[int, int, int] | None:
    """
    Compute token indices for the two embedding subsets (Guess and Probability).

    last_guess_token_index: token after "Guess:" (first token of answer)
    first_prob_token_index: "\n" token before "Probability:" (first occurrence of marker)
    end_prob_token_index: token after ``Probability:`` + whitespace (first token of prob value)

    Returns (last_guess_token_index, first_prob_token_index, end_prob_token_index)
    or None on failure.
    """
    guess_start = _match_token_prefix(decoded_tokens, GUESS_PREFIX_TOKENS, start=0)
    if guess_start is None:
        return None

    last_guess_token_index = guess_start + len(GUESS_PREFIX_TOKENS)

    prob_start = _match_token_prefix(
        decoded_tokens, PROBABILITY_PREFIX_TOKENS, start=last_guess_token_index
    )
    if prob_start is None:
        return None

    first_prob_token_index = prob_start
    end_prob_token_index = prob_start + len(PROBABILITY_PREFIX_TOKENS)

    if (
        last_guess_token_index <= 0
        or last_guess_token_index >= len(decoded_tokens)
        or end_prob_token_index > len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None

    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def parse_guess_start_index(decoded_tokens: List[str]) -> Optional[int]:
    """Return first token index of semantic answer (token right after ``Guess:``)."""
    guess_start = _match_token_prefix(decoded_tokens, GUESS_PREFIX_TOKENS, start=0)
    if guess_start is None:
        return None
    last_guess_token_index = guess_start + len(GUESS_PREFIX_TOKENS)
    if last_guess_token_index <= 0 or last_guess_token_index > len(decoded_tokens):
        return None
    return last_guess_token_index


# ---------------------------------------------------------------------------
# Dataset + few-shot
# ---------------------------------------------------------------------------


def load_eval_dataset(dataset_name: str, seed: int) -> Tuple[Dataset, Dataset]:
    """Load train/validation splits via semantic_uncertainty.data_utils.load_ds."""
    import sys

    sem_unc_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "semantic_uncertainty")
    if sem_unc_root not in sys.path:
        sys.path.insert(0, sem_unc_root)
    from uncertainty.data.data_utils import load_ds

    train_ds, val_ds = load_ds(dataset_name, seed=seed)
    if train_ds is None or val_ds is None:
        raise ValueError(f"Unsupported or failed dataset load: {dataset_name}")
    return train_ds, val_ds


def load_trivia_qa(seed: int) -> Tuple[Dataset, Dataset]:
    return load_eval_dataset("trivia_qa", seed)


def split_answerable_indices(dataset: Dataset) -> List[int]:
    return [i for i, ex in enumerate(dataset) if len(ex["answers"]["text"]) > 0]


def construct_fewshot_prompt_from_indices(
    dataset: Dataset,
    example_indices: Sequence[int],
    brief: str,
    brief_always: bool,
    use_context: bool,
) -> str:
    if not brief_always:
        prompt = brief
    else:
        prompt = ""

    for example_index in example_indices:
        example = dataset[int(example_index)]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        piece = ""
        if brief_always:
            piece += brief
        if use_context and (context is not None):
            piece += f"Context: {context}\n"
        piece += f"Question: {question}\n"
        if answer:
            piece += f"Answer: {answer}\n\n"
        else:
            piece += "Answer:"
        prompt = prompt + piece

    return prompt


def encode_example_id(example_id) -> str:
    return quote(str(example_id), safe="")


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        return cli_output_path
    base_dir = os.path.join("layerwise_mean_ablation", "results")
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_dir = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "ablation_results.json")


def mini_output_json_path(full_output_path: str) -> str:
    out_dir = os.path.dirname(full_output_path)
    return os.path.join(out_dir, "ablation_results_mini.json")


def config_txt_path(full_output_path: str) -> str:
    out_dir = os.path.dirname(full_output_path)
    return os.path.join(out_dir, "config.txt")


def resolve_confidence_groups(
    *,
    mean_from_low_confidence: bool,
    ablate_with_same_confidence: bool = False,
) -> Tuple[str, str]:
    """Return (mean_source_group, ablation_target_group) labels."""
    source_group = "low_confidence" if mean_from_low_confidence else "high_confidence"
    if ablate_with_same_confidence:
        return source_group, source_group
    target_group = "high_confidence" if mean_from_low_confidence else "low_confidence"
    return source_group, target_group


# (combo_dir_name, mean_from_low_confidence, ablate_with_same_confidence)
CONFIDENCE_GROUP_PAIR_COMBOS: Tuple[Tuple[str, bool, bool], ...] = (
    ("mean_low_target_high", True, False),
    ("mean_high_target_low", False, False),
    ("mean_low_target_low", True, True),
    ("mean_high_target_high", False, True),
)


def resolve_combo_target_ids(
    low_ids: set[str],
    high_ids: set[str],
    *,
    mean_from_low_confidence: bool,
    ablate_with_same_confidence: bool,
) -> set[str]:
    source_ids = low_ids if mean_from_low_confidence else high_ids
    if ablate_with_same_confidence:
        return source_ids
    return high_ids if mean_from_low_confidence else low_ids


def layer_tensor_means_from_numpy(
    verbalised_embedding_means: Dict[str, np.ndarray],
    run_layers: Sequence[int],
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[str, torch.Tensor]]:
    layer_to_means: Dict[int, Dict[str, torch.Tensor]] = {}
    for i, layer_idx in enumerate(run_layers):
        layer_to_means[int(layer_idx)] = {
            "prompt_mean": torch.tensor(verbalised_embedding_means["prompt_mean"][i], device=device, dtype=torch_dtype),
            "guess": torch.tensor(verbalised_embedding_means["guess"][i], device=device, dtype=torch_dtype),
            "sem_answer_mean": torch.tensor(
                verbalised_embedding_means["sem_answer_mean"][i], device=device, dtype=torch_dtype
            ),
            "probability": torch.tensor(verbalised_embedding_means["probability"][i], device=device, dtype=torch_dtype),
            "probability_value_mean": torch.tensor(
                verbalised_embedding_means["probability_value_mean"][i], device=device, dtype=torch_dtype
            ),
        }
    return layer_to_means


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
    mean_from_low_confidence: bool,
    parse_mode_verbalised_confidence: bool,
    mode_confidence_means: Dict[str, Optional[float]],
    mode_confidence_counts: Dict[str, int],
    mode_responses_identical_true: Dict[str, int],
    finished_at: str,
    mode_semantic_similarity_means: Optional[Dict[str, Optional[float]]] = None,
    mode_semantic_similarity_counts: Optional[Dict[str, int]] = None,
    mode_verbalised_confidence_effect_means: Optional[Dict[str, Optional[float]]] = None,
    mode_verbalised_confidence_effect_counts: Optional[Dict[str, int]] = None,
    mode_uncertainty_score_means: Optional[Dict[str, Optional[float]]] = None,
    mode_uncertainty_score_counts: Optional[Dict[str, int]] = None,
    ablate_with_same_confidence: bool = False,
    all_confidence_group_pairs: bool = False,
) -> None:
    source_group, target_group = resolve_confidence_groups(
        mean_from_low_confidence=mean_from_low_confidence,
        ablate_with_same_confidence=ablate_with_same_confidence,
    )
    lines = [
        "Layerwise Mean Ablation Configuration",
        "====================================",
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
        f"all_confidence_group_pairs={all_confidence_group_pairs}",
        f"mean_from_low_confidence={mean_from_low_confidence}",
        f"ablate_with_same_confidence={ablate_with_same_confidence}",
        f"mean_source_group={source_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"layer_indexing={LAYER_INDEXING_NOTE}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"parse_mode_verbalised_confidence={parse_mode_verbalised_confidence}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Mode Confidence Metrics]",
        "Values below are mean verbalised confidence.",
        "",
    ]
    for mode_name in args.ablation_mode:
        mode_mean = mode_confidence_means.get(mode_name)
        count_val = int(mode_confidence_counts.get(mode_name, 0))
        out_key = mode_to_output_key(mode_name)
        if mode_name == "none":
            if mode_mean is None:
                lines.append(f"{out_key}=None ({count_val})")
            else:
                lines.append(f"{out_key}={mode_mean:.6f} ({count_val})")
        else:
            identical_n = int(mode_responses_identical_true.get(mode_name, 0))
            if mode_mean is None:
                line = f"{out_key}=None ({count_val}) [responses_identical: {identical_n}]"
            else:
                line = f"{out_key}={mode_mean:.6f} ({count_val}) [responses_identical: {identical_n}]"
            if mode_semantic_similarity_means is not None:
                sem_mean = mode_semantic_similarity_means.get(mode_name)
                sem_count = int(mode_semantic_similarity_counts.get(mode_name, 0) if mode_semantic_similarity_counts else 0)
                if sem_mean is not None and sem_count > 0:
                    line += f" semantic_similarity={sem_mean:.6f} ({sem_count})"
            if mode_verbalised_confidence_effect_means is not None:
                vce_mean = mode_verbalised_confidence_effect_means.get(mode_name)
                vce_count = int(
                    mode_verbalised_confidence_effect_counts.get(mode_name, 0)
                    if mode_verbalised_confidence_effect_counts
                    else 0
                )
                if vce_mean is not None and vce_count > 0:
                    line += f" verbalised_confidence_effect={vce_mean:.6f} ({vce_count})"
            if mode_uncertainty_score_means is not None:
                unc_mean = mode_uncertainty_score_means.get(mode_name)
                unc_count = int(mode_uncertainty_score_counts.get(mode_name, 0) if mode_uncertainty_score_counts else 0)
                if unc_mean is not None and unc_count > 0:
                    line += f" uncertainty_score={unc_mean:.6f} ({unc_count})"
            lines.append(line)
    lines.extend(
        [
            "",
            "[Run]",
            f"finished_at={finished_at}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_individual_layer_summary_and_plot(
    individual_root: str,
    *,
    args: argparse.Namespace,
    model_n_layers: int,
    run_layers: Sequence[int],
    mean_from_low_confidence: bool,
    ablate_with_same_confidence: bool,
    all_confidence_group_pairs: bool,
    per_layer_mode_means: Dict[int, Dict[str, Optional[float]]],
) -> None:
    mean_source_group, ablation_target_group = resolve_confidence_groups(
        mean_from_low_confidence=mean_from_low_confidence,
        ablate_with_same_confidence=ablate_with_same_confidence,
    )
    modes_non_none = [mode for mode in args.ablation_mode if mode != "none"]
    baseline_none_mean = None
    if "none" in args.ablation_mode and per_layer_mode_means:
        first_layer = run_layers[0]
        baseline_none_mean = per_layer_mode_means[first_layer].get("none")

    summary_path = os.path.join(individual_root, "summary.txt")
    summary_lines = [
        "Individual Layer Mean Ablation Summary",
        "=====================================",
        "",
        "[Setup]",
        f"model_name={args.model_name}",
        f"input_h5={args.input_h5}",
        f"dataset={args.dataset}",
        f"ablation_mode={args.ablation_mode}",
        f"num_layers={model_n_layers}",
        f"run_layers={','.join(str(layer) for layer in run_layers)}",
        f"num_samples={args.num_samples}",
        f"num_few_shot={args.num_few_shot}",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"all_confidence_group_pairs={all_confidence_group_pairs}",
        f"mean_from_low_confidence={mean_from_low_confidence}",
        f"ablate_with_same_confidence={ablate_with_same_confidence}",
        f"mean_source_group={mean_source_group}",
        f"ablation_target_group={ablation_target_group}",
        f"parse_mode_verbalised_confidence={args.parse_mode_verbalised_confidence}",
        "",
    ]
    if "none" in args.ablation_mode:
        if baseline_none_mean is None:
            summary_lines.append("none_mean_verbalised_confidence=None")
        else:
            summary_lines.append(f"none_mean_verbalised_confidence={baseline_none_mean:.6f}")
        summary_lines.append("")

    summary_lines.append("[Per-layer verbalised confidence]")
    if modes_non_none:
        header = "layer\t" + "\t".join(modes_non_none)
        summary_lines.append(header)
        for layer_idx in run_layers:
            row_vals = []
            for mode in modes_non_none:
                val = per_layer_mode_means[layer_idx].get(mode)
                row_vals.append("None" if val is None else f"{val:.6f}")
            summary_lines.append(f"{layer_idx}\t" + "\t".join(row_vals))
    else:
        summary_lines.append("No non-none modes selected.")
    summary_lines.append("")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logging.info("Wrote %s", summary_path)

    plot_path = os.path.join(individual_root, "verbalised_confidence_by_layer.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in modes_non_none:
        ys: List[float] = []
        xs: List[int] = []
        for layer_idx in run_layers:
            y_val = per_layer_mode_means[layer_idx].get(mode)
            if y_val is not None:
                xs.append(layer_idx)
                ys.append(float(y_val))
        if ys:
            ax.plot(xs, ys, marker="o", label=mode)
    if baseline_none_mean is not None:
        ax.axhline(y=float(baseline_none_mean), linestyle="--", label="none (baseline)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Verbalised confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Individual layer verbalised confidence")
    ax.grid(True, alpha=0.3)
    if modes_non_none or baseline_none_mean is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", plot_path)


def mode_to_output_key(mode: str) -> str:
    if mode == "none":
        return "no_replacement"
    if mode == "probability_tokens_mean_replace":
        return "probability_tokens_mean_replace"
    if mode == "probability_last_token_mean_replace":
        return "probability_last_token_mean_replace"
    if mode == "probability_span_except_last_token_mean_replace":
        return "probability_span_except_last_token_mean_replace"
    if mode in PROBABILITY_ROW_INDEX_MODES:
        return mode
    if mode == "all_pre_probability_tokens_mean_replace":
        return "all_pre_probability_tokens_mean_replace"
    if mode == "guess_tokens_mean_replace":
        return "guess_tokens_mean_replace"
    if mode == "all_pre_guess_tokens_mean_replace":
        return "all_pre_guess_tokens_mean_replace"
    if mode == "guess_then_guess_and_probability_tokens_mean_replace":
        return "guess_then_guess_and_probability_tokens_mean_replace"
    if mode == "probability_value_mean_replace":
        return "probability_value_mean_replace"
    if mode == "semantic_answer_mean_replace":
        return "semantic_answer_mean_replace"
    if mode == "semantic_answer_including_first_prob_mean_replace":
        return "semantic_answer_including_first_prob_mean_replace"
    if mode == "prompt_tokens_mean_replace":
        return "prompt_tokens_mean_replace"
    if mode == "sem_ans_tokens_during_gen":
        return "sem_ans_tokens_during_gen"
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def parse_mode_confidence_from_response(response: str) -> Optional[float]:
    parsed = parse_probability_from_response(response)
    if parsed is None:
        return None
    return float(parsed)


def parse_probability_from_response(response_str: str) -> float | None:
    """
    Extract probability in [0,1] from a response string.
    Uses the first occurrence of "probability:".
    """
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


def parse_semantic_answer_from_response(response_str: str) -> Optional[str]:
    """Extract guess text between ``Guess:`` and the first ``Probability:`` marker."""
    if not response_str or not isinstance(response_str, str):
        return None
    match = re.search(
        r"guess\s*:\s*(.*?)\s*probability\s*:",
        response_str,
        re.IGNORECASE | re.DOTALL,
    )
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


@contextmanager
def _without_tl_noop_fp32_upcast():
    """Skip TL's full-state-dict fp32 upcast when no weight processing is requested.

    ``ProcessWeights.process_weights`` always upcasts every tensor to float32 even
    when fold/center flags are all False. That temporary copy OOMs large models
    (e.g. Qwen-32B) on both host RAM and 80GB GPUs.
    """
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
            # Disable fold/center processing so the fp32 upcast is unnecessary.
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
            model_name,
            add_bos_token=True,
            trust_remote_code=True,
            use_fast=False,
            token=hf_token,
        )
        model = _from_pretrained(tokenizer=slow_tokenizer)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------
# H5 reading + mean activation prep
# ---------------------------------------------------------------------------


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

    out = {}
    for k in node.keys():
        out[k] = _read_h5_node(node[k])
    return out


_NEW_H5_REQUIRED_COMPONENTS = ("res", "attn", "mlp")
_NEW_H5_REQUIRED_EMBEDDING_FIELDS = (
    "embeddings_mean_prompt",
    "embeddings_guess",
    "embeddings_mean_sem_answer",
    "embeddings_probability",
    "embeddings_mean_prob_val",
)


def _extract_res_field(resp0: dict, ex_id: str, field_name: str, *, new_h5_format: bool):
    """Return the embedding payload for ``field_name`` on ``resp0``.

    When ``new_h5_format`` is set, every required embedding field is expected to be a
    dict containing all of ``res``/``attn``/``mlp`` (non-null). Layerwise residual-stream
    ablation only consumes ``res``; ``attn``/``mlp`` are validated-but-unused.
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
            if not isinstance(obj, dict):
                continue
            examples[str(example_id)] = obj
    return examples


LAYER_INDEXING_NOTE = (
    "display: 0=embedding resid_pre (blocks.0.hook_resid_pre); "
    "k>=1=resid_post of TL block k-1"
)


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


def hook_name_for_display_layer(display_layer: int) -> str:
    """Map display layer index to the residual-stream intervention hook.

    Display 0 is the input embedding (resid-pre of block 0). Display k>=1 is
    resid-post of TransformerLens block k-1.
    """
    if display_layer == 0:
        return "blocks.0.hook_resid_pre"
    return f"blocks.{display_layer - 1}.hook_resid_post"


def _as_layer_hidden(arr_like: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 4:
        return arr[:, 0, -1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected embedding tensor shape: {arr.shape}; expected 4D or 2D.")


def _is_expected_or_plus_two(actual_len: int, expected_len: int) -> bool:
    return actual_len in (expected_len, expected_len + 2)


def compute_confidence_group_means(
    examples_h5: Dict[str, dict],
    ablate_layers: Sequence[int],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    mean_from_low_confidence: bool,
    new_h5_format: bool = False,
    h5_res_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, set[str], set[str]]:
    """
    Returns:
      means: [num_selected_layers, num_probability_tokens, hidden_dim]
      low_conf_ids: confidence <= 0.1 example ids
      high_conf_ids: confidence >= 0.9 example ids
    """
    source_vectors: List[np.ndarray] = []
    low_ids: set[str] = set()
    high_ids: set[str] = set()

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

        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
            continue

        emb_prob = _extract_res_field(
            resp0, ex_id, "embeddings_probability", new_h5_format=new_h5_format
        )
        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} responses/0/embeddings_probability must be a list.")
        if not _is_expected_or_plus_two(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability len={len(emb_prob)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        token_vectors: List[np.ndarray] = []
        resid_post_layers = (
            np.asarray(h5_res_indices)
            if h5_res_indices is not None
            else np.asarray(ablate_layers) + 1
        )
        for tok_arr in emb_prob:
            layer_hidden = _as_layer_hidden(tok_arr)  # [n_layers + 1, hidden_dim]
            # Default: HF res[0] is the embedding; resid_post of TL block i is res[i + 1].
            # Pass h5_res_indices to index those rows directly (display k == res[k]).
            selected = layer_hidden[resid_post_layers, :]
            token_vectors.append(selected)
        stacked = np.stack(token_vectors, axis=1)  # [num_selected_layers, num_prob_tokens - 1, hidden_dim]
        source_vectors.append(stacked)

    if not source_vectors:
        source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(f"No {source_name} examples found at threshold {operator} {threshold}.")

    if mean_from_low_confidence and not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if (not mean_from_low_confidence) and not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")

    means = np.mean(np.stack(source_vectors, axis=0), axis=0).astype(np.float32) # [num_selected_layers, num_prob_tokens, hidden_dim]
    return means, low_ids, high_ids


def collect_confidence_group_ids(
    examples_h5: Dict[str, dict],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
) -> Tuple[set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError(f"Example {ex_id} must have exactly one response, got {0 if responses is None else len(responses)}.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 is not a dict.")
        conf = float(resp0.get("verbalised_confidence"))
        if conf <= low_conf_threshold:
            low_ids.add(ex_id)
        if conf >= high_conf_threshold:
            high_ids.add(ex_id)
    return low_ids, high_ids


def truncate_sorted_ids(ids: Sequence[str], num_samples: int) -> List[str]:
    return sorted(ids)[: min(num_samples, len(ids))]


def compute_verbalised_embedding_group_means(
    examples_h5: Dict[str, dict],
    ablate_layers: Sequence[int],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
    mean_from_low_confidence: bool,
    new_h5_format: bool = False,
    h5_res_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    """Build per-layer mean replacement vectors for verbalised-embedding regions.

    This scans examples, selects a confidence group (low or high), and averages
    embeddings across those examples for:
      - prompt mean token (`embeddings_mean_prompt`)
      - per-position Guess span tokens (`embeddings_guess`)
      - semantic-answer mean token (`embeddings_mean_sem_answer`)
      - Probability marker-span rows (`embeddings_probability`)
      - Mean probability-value embedding (`embeddings_mean_prob_val`)

    ``embeddings_probability`` lists may have length ``expected_probability_tokens``
    or ``expected_probability_tokens + 2``; only the first
    ``expected_probability_tokens`` rows are used.

    By default H5 ``res`` rows are ``ablate_layers + 1`` (skip embedding; resid-post
    of TransformerLens block i is ``res[i + 1]``). Pass ``h5_res_indices`` to index
    those rows directly (display layer k == ``res[k]``).

    Returns:
      - dict of mean tensors keyed by region (`prompt_mean`, `guess`,
        `sem_answer_mean`, `probability`, `probability_value_mean`) with layer
        dimension restricted to `ablate_layers` (or `h5_res_indices` if set)
      - low-confidence example IDs
      - high-confidence example IDs
    """
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    prompt_vectors: List[np.ndarray] = []
    sem_answer_vectors: List[np.ndarray] = []
    guess_vectors: List[np.ndarray] = []
    probability_vectors: List[np.ndarray] = []
    probability_value_mean_vectors: List[np.ndarray] = []

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

        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
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
        emb_mean_prob_val = _extract_res_field(
            resp0, ex_id, "embeddings_mean_prob_val", new_h5_format=new_h5_format
        )
        if (
            emb_prompt is None
            or emb_guess is None
            or emb_sem_answer is None
            or emb_prob is None
            or emb_mean_prob_val is None
        ):
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

        resid_post_layers = (
            np.asarray(h5_res_indices)
            if h5_res_indices is not None
            else np.asarray(ablate_layers) + 1
        )
        prompt_layer_hidden = _as_layer_hidden(emb_prompt)[resid_post_layers, :]
        sem_answer_layer_hidden = _as_layer_hidden(emb_sem_answer)[resid_post_layers, :]
        mean_prob_val_layer_hidden = _as_layer_hidden(emb_mean_prob_val)[resid_post_layers, :]

        guess_selected: List[np.ndarray] = []
        for tok_arr in emb_guess:
            layer_hidden = _as_layer_hidden(tok_arr)[resid_post_layers, :]
            guess_selected.append(layer_hidden)
        guess_stacked = np.stack(guess_selected, axis=1)

        prob_selected: List[np.ndarray] = []
        for tok_arr in emb_prob:
            layer_hidden = _as_layer_hidden(tok_arr)[resid_post_layers, :]
            prob_selected.append(layer_hidden)
        prob_stacked = np.stack(prob_selected, axis=1)

        prompt_vectors.append(prompt_layer_hidden)
        sem_answer_vectors.append(sem_answer_layer_hidden)
        guess_vectors.append(guess_stacked)
        probability_vectors.append(prob_stacked)
        probability_value_mean_vectors.append(mean_prob_val_layer_hidden)

    if not prompt_vectors:
        source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(f"No {source_name} examples found at threshold {operator} {threshold}.")

    if mean_from_low_confidence and not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if (not mean_from_low_confidence) and not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")

    out_means = {
        "prompt_mean": np.mean(np.stack(prompt_vectors, axis=0), axis=0).astype(np.float32),
        "guess": np.mean(np.stack(guess_vectors, axis=0), axis=0).astype(np.float32),
        "sem_answer_mean": np.mean(np.stack(sem_answer_vectors, axis=0), axis=0).astype(np.float32),
        "probability": np.mean(np.stack(probability_vectors, axis=0), axis=0).astype(np.float32),
        "probability_value_mean": np.mean(np.stack(probability_value_mean_vectors, axis=0), axis=0).astype(np.float32),
    }
    return out_means, low_ids, high_ids


# ---------------------------------------------------------------------------
# Hook builder + generation
# ---------------------------------------------------------------------------


def _completion_token_index_to_abs_pos(prompt_len: int, completion_index: int) -> int:
    """Map completion-relative token index (0 = first generated token) to full-sequence position.

    First generated token is produced from the last prompt position, so index ``k`` maps to
    ``prompt_len + k - 1`` (for ``k >= 1`` this is the usual prompt offset; for ``k == 0`` this is
    ``prompt_len - 1``).
    """
    return prompt_len + completion_index - 1


def _absolute_prob_positions(prompt_len: int, decoded_tokens: List[str]) -> List[int]:
    """Absolute indices for completion tokens ``first_prob`` … ``end_prob`` (H5 span; excludes value token)."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    # TODO: remove this after checking
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed

    seq_len = prompt_len + len(decoded_tokens)
    out: List[int] = []
    for k in range(first_prob, end_prob+1):  # Do not include the first token of the prob value
        p = _completion_token_index_to_abs_pos(prompt_len, k)
        if 0 <= p < seq_len:
            out.append(p)
    return out


def _absolute_prob_positions_at_row_indices(
    prompt_len: int,
    decoded_tokens: List[str],
    row_indices: Sequence[int],
) -> List[int]:
    """Absolute positions for selected rows of the H5 probability-prefix span (0-indexed)."""
    full_positions = _absolute_prob_positions(prompt_len, decoded_tokens)
    if not full_positions:
        return []
    out: List[int] = []
    for idx in row_indices:
        if idx < 0 or idx >= len(full_positions):
            return []
        out.append(full_positions[idx])
    return out


def _absolute_prob_last_token_only(prompt_len: int, decoded_tokens: List[str]) -> List[int]:
    """Absolute index for the last token in the ``Probability:`` marker span only (H5 last probability row)."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, _, end_prob = parsed
    seq_len = prompt_len + len(decoded_tokens)
    p = _completion_token_index_to_abs_pos(prompt_len, end_prob)
    if 0 <= p < seq_len:
        return [p]
    return []


def _absolute_prob_marker_except_last_token(prompt_len: int, decoded_tokens: List[str]) -> List[int]:
    """Absolute indices for the ``Probability:`` marker span excluding the last token."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    seq_len = prompt_len + len(decoded_tokens)
    out: List[int] = []
    for k in range(first_prob, end_prob):
        p = _completion_token_index_to_abs_pos(prompt_len, k)
        if 0 <= p < seq_len:
            out.append(p)
    return out


def _absolute_sem_answer_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    include_first_prob: bool = False,
) -> List[int]:
    """Absolute indices for semantic-answer completion tokens (between ``Guess:`` answer start and ``Probability:`` marker).

    Completion indices are ``range(last_guess_token_index, first_prob_token_index)`` by default.
    If ``include_first_prob`` is true, the range is inclusive of ``first_prob_token_index``
    (the residual whose input is the last semantic-answer token).
    """
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    last_guess_token_index, first_prob_token_index, _ = parsed
    seq_len = prompt_len + len(decoded_tokens)
    end = first_prob_token_index + (1 if include_first_prob else 0)
    out: List[int] = []
    for k in range(last_guess_token_index, end):
        p = _completion_token_index_to_abs_pos(prompt_len, k)
        if 0 <= p < seq_len:
            out.append(p)
    return out


def _absolute_sem_answer_positions_during_gen(prompt_len: int, decoded_tokens: List[str]) -> List[int]:
    """Semantic-answer positions during generation (guess parsed) or after full parse."""
    full_positions = _absolute_sem_answer_positions(prompt_len, decoded_tokens)
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


def _absolute_pre_probability_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Optional[Dict[str, List[int]]]:
    """Map parse result to absolute positions. ``prompt`` is ``0 .. prompt_len-2`` (exclude last prompt index)."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return None
    last_guess_token_index, first_prob_token_index, end_prob_token_index = parsed

    guess_positions_rel = list(range(0, last_guess_token_index))
    if len(guess_positions_rel) != expected_guess_tokens:
        return None
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]

    probability_positions_rel = list(range(first_prob_token_index, end_prob_token_index+1))
    if not _is_expected_or_plus_two(len(probability_positions_rel), expected_probability_tokens):
        return None
    probability_positions_rel = probability_positions_rel[:expected_probability_tokens]

    prompt_positions_abs = list(range(0, prompt_len - 1))
    guess_positions_abs = [_completion_token_index_to_abs_pos(prompt_len, k) for k in guess_positions_rel]
    sem_answer_positions_abs = [
        _completion_token_index_to_abs_pos(prompt_len, k)
        for k in range(last_guess_token_index, first_prob_token_index)
    ]
    probability_positions_abs = [
        _completion_token_index_to_abs_pos(prompt_len, k) for k in probability_positions_rel
    ]
    ans = {
        "prompt": prompt_positions_abs,
        "guess": guess_positions_abs,
        "sem_answer": sem_answer_positions_abs,
        "probability": probability_positions_abs,
    }
    return ans


def _absolute_guess_span_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
) -> List[int]:
    """Absolute indices for completion tokens covering the ``Guess:`` prefix span (H5 ``embeddings_guess`` layout)."""
    last_guess_token_index = parse_guess_start_index(decoded_tokens)
    if last_guess_token_index is None:
        return []
    guess_positions_rel = list(range(0, last_guess_token_index))
    if len(guess_positions_rel) != expected_guess_tokens:
        return []
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]
    return [_completion_token_index_to_abs_pos(prompt_len, k) for k in guess_positions_rel]


def _build_resid_post_mean_replace_hooks(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    *,
    seq_len_provider: Callable[[], int],
    abs_positions_provider: Callable[[], List[int]],
    strict_num_prob_positions: bool,
    hook_name_for_layer: Optional[Callable[[int], str]] = None,
) -> List[Tuple[str, Callable]]:
    """
    Shared residual-stream mean-replace factory: ``abs_positions_provider`` returns
    the same layout as ``_absolute_prob_positions`` (length ``num_prob_tokens`` when
    full). Dict keys are display layers; default hook mapping is
    ``hook_name_for_display_layer``. Pass ``hook_name_for_layer`` to override.
    """
    num_prob_tokens = next(iter(layer_to_mean_vectors.values())).shape[0]
    hooks: List[Tuple[str, Callable]] = []
    name_fn = hook_name_for_layer or hook_name_for_display_layer

    for layer in layer_to_mean_vectors:
        hook_name = name_fn(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                _ = seq_len_provider()  # keep parity with zero-ablation callback shape
                abs_positions = abs_positions_provider()
                if not abs_positions:
                    return activation
                if len(abs_positions) != num_prob_tokens:
                    if strict_num_prob_positions:
                        raise ValueError(
                            f"Number of absolute probability positions {len(abs_positions)} "
                            f"does not match number of probability tokens {num_prob_tokens}."
                        )
                    return activation

                replacement = layer_to_mean_vectors[layer_idx]  # [num_prob_tokens, hidden_dim]
                for pos_i, abs_pos in enumerate(abs_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = replacement[pos_i].to(activation.dtype)
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def build_mean_replace_hooks(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
) -> List[Tuple[str, Callable]]:
    """
    Replace hook_resid_post activations at dynamic probability-token positions.
    """

    def _abs_positions() -> List[int]:
        return _absolute_prob_positions(prompt_len, decoded_tokens_provider())

    return _build_resid_post_mean_replace_hooks(
        layer_to_mean_vectors,
        seq_len_provider=seq_len_provider,
        abs_positions_provider=_abs_positions,
        strict_num_prob_positions=True,
    )


def build_mean_replace_hooks_probability_last_token(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
) -> List[Tuple[str, Callable]]:
    def _abs_positions() -> List[int]:
        return _absolute_prob_last_token_only(prompt_len, decoded_tokens_provider())

    return _build_resid_post_mean_replace_hooks(
        layer_to_mean_vectors,
        seq_len_provider=seq_len_provider,
        abs_positions_provider=_abs_positions,
        strict_num_prob_positions=True,
    )


def build_mean_replace_hooks_probability_span_except_last_token(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
) -> List[Tuple[str, Callable]]:
    def _abs_positions() -> List[int]:
        return _absolute_prob_marker_except_last_token(prompt_len, decoded_tokens_provider())

    return _build_resid_post_mean_replace_hooks(
        layer_to_mean_vectors,
        seq_len_provider=seq_len_provider,
        abs_positions_provider=_abs_positions,
        strict_num_prob_positions=True,
    )


def build_mean_replace_hooks_probability_row_indices(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
    *,
    row_indices: Sequence[int],
) -> List[Tuple[str, Callable]]:
    indices_tuple = tuple(row_indices)

    def _abs_positions() -> List[int]:
        return _absolute_prob_positions_at_row_indices(
            prompt_len, decoded_tokens_provider(), indices_tuple
        )

    return _build_resid_post_mean_replace_hooks(
        layer_to_mean_vectors,
        seq_len_provider=seq_len_provider,
        abs_positions_provider=_abs_positions,
        strict_num_prob_positions=True,
    )


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
    return_generated_ids: bool = False,
) -> Tuple[str, List[str]] | Tuple[str, List[str], List[int]]:
    tokens = model.to_tokens(local_prompt)
    generated: List[int] = []
    decoded_tokens: List[str] = []

    with torch.inference_mode():
        for _step in range(max_new_tokens):
            out = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=fwd_hooks or [],
            )
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            generated.append(next_id)

            piece = model.tokenizer.decode([next_id], skip_special_tokens=False)
            decoded_tokens.append(piece)

            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)

            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break

    response = _postprocess_response_from_full_decode(model, tokens, local_prompt)
    if return_generated_ids:
        return response, decoded_tokens, generated
    return response, decoded_tokens


def _greedy_extend_with_fwd_hooks(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    tokens: torch.Tensor,
    decoded_tokens: List[str],
    fwd_hooks: List[Tuple[str, Callable]],
) -> Tuple[str, List[str]]:
    """Greedy-decode from ``tokens`` (prompt), appending to ``decoded_tokens``, with fixed ``fwd_hooks``."""
    with torch.inference_mode():
        for _step in range(max_new_tokens):
            out = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=fwd_hooks,
            )
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())

            piece = model.tokenizer.decode([next_id], skip_special_tokens=False)
            decoded_tokens.append(piece)

            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)

            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break

    response = _postprocess_response_from_full_decode(model, tokens, local_prompt)
    return response, decoded_tokens


def greedy_generate_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_mean_vectors: Dict[int, torch.Tensor],
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _seq_len() -> int:
        return prompt_len + len(decoded_tokens)

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_mean_replace_hooks(
        layer_to_mean_vectors=layer_to_mean_vectors,
        prompt_len=prompt_len,
        seq_len_provider=_seq_len,
        decoded_tokens_provider=_decoded_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def greedy_generate_probability_span_subset_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    *,
    subset: str,
) -> Tuple[str, List[str]]:
    """Greedy decode with mean replacement on a subset of the ``Probability:`` marker span (parse-gated)."""
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _seq_len() -> int:
        return prompt_len + len(decoded_tokens)

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    if subset == "last_token":
        hooks = build_mean_replace_hooks_probability_last_token(
            layer_to_mean_vectors=layer_to_mean_vectors,
            prompt_len=prompt_len,
            seq_len_provider=_seq_len,
            decoded_tokens_provider=_decoded_tokens,
        )
    elif subset == "except_last_token":
        hooks = build_mean_replace_hooks_probability_span_except_last_token(
            layer_to_mean_vectors=layer_to_mean_vectors,
            prompt_len=prompt_len,
            seq_len_provider=_seq_len,
            decoded_tokens_provider=_decoded_tokens,
        )
    else:
        raise ValueError(f"Unknown probability span subset: {subset!r}")

    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def greedy_generate_probability_row_indices_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    *,
    row_indices: Sequence[int],
) -> Tuple[str, List[str]]:
    """Greedy decode with mean replacement on selected H5 probability-prefix row indices."""
    indices_list = list(row_indices)
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _seq_len() -> int:
        return prompt_len + len(decoded_tokens)

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    layer_to_subset_vectors = {
        layer_idx: layer_to_mean_vectors[layer_idx][indices_list, :]
        for layer_idx in layer_to_mean_vectors
    }
    hooks = build_mean_replace_hooks_probability_row_indices(
        layer_to_subset_vectors,
        prompt_len=prompt_len,
        seq_len_provider=_seq_len,
        decoded_tokens_provider=_decoded_tokens,
        row_indices=indices_list,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_pre_probability_mean_replace_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    required_keys = {"prompt_mean", "guess", "sem_answer_mean", "probability"}

    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                positions = _absolute_pre_probability_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
                if positions is None:
                    return activation

                region_means = layer_to_verbalised_embedding_means[layer_idx]
                missing_keys = required_keys - set(region_means.keys())
                if missing_keys:
                    raise ValueError(
                        f"Layer {layer_idx} verbalised embedding means missing keys. "
                        f"Got {sorted(region_means.keys())}, missing {sorted(missing_keys)}."
                    )

                guess_mean = region_means["guess"]
                prob_mean = region_means["probability"]
                if guess_mean.ndim != 2 or prob_mean.ndim != 2:
                    raise ValueError(
                        f"Layer {layer_idx} guess/probability mean must be rank-2 [n_pos, hidden]. "
                        f"Got guess {tuple(guess_mean.shape)}, probability {tuple(prob_mean.shape)}."
                    )
                if len(positions["guess"]) != int(guess_mean.shape[0]):
                    raise ValueError(
                        f"Guess position count {len(positions['guess'])} does not match replacement count {int(guess_mean.shape[0])}."
                    )
                if len(positions["probability"]) != int(prob_mean.shape[0]):
                    raise ValueError(
                        f"Probability position count {len(positions['probability'])} does not match replacement count {int(prob_mean.shape[0])}."
                    )

                prompt_mean = region_means["prompt_mean"].to(activation.dtype)
                sem_answer_mean = region_means["sem_answer_mean"].to(activation.dtype)
                guess_mean = guess_mean.to(activation.dtype)
                prob_mean = prob_mean.to(activation.dtype)

                for abs_pos in positions["prompt"]:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = prompt_mean
                for pos_i, abs_pos in enumerate(positions["guess"]):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = guess_mean[pos_i]
                for abs_pos in positions["sem_answer"]:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = sem_answer_mean
                for pos_i, abs_pos in enumerate(positions["probability"]):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = prob_mean[pos_i]
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_pre_probability_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[str, List[str]]:
    """Greedy generate with mean-ablation on all token positions before the first probability value token (tokens before end of "Probability:")"""
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_pre_probability_mean_replace_hooks(
        layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_guess_tokens_mean_replace_hooks(
    layer_to_guess_mean_vectors: Dict[int, torch.Tensor],
    *,
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
) -> List[Tuple[str, Callable]]:
    """Replace resid_post only at the dynamic ``Guess:`` prefix span (per-position means from H5 ``guess``)."""

    def _abs_positions() -> List[int]:
        return _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens_provider(),
            expected_guess_tokens=expected_guess_tokens,
        )

    return _build_resid_post_mean_replace_hooks(
        layer_to_guess_mean_vectors,
        seq_len_provider=seq_len_provider,
        abs_positions_provider=_abs_positions,
        strict_num_prob_positions=True,
    )


def greedy_generate_guess_tokens_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_guess_mean_vectors: Dict[int, torch.Tensor],
    *,
    expected_guess_tokens: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _seq_len() -> int:
        return prompt_len + len(decoded_tokens)

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_guess_tokens_mean_replace_hooks(
        layer_to_guess_mean_vectors,
        prompt_len=prompt_len,
        seq_len_provider=_seq_len,
        decoded_tokens_provider=_decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_all_pre_guess_mean_replace_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
) -> List[Tuple[str, Callable]]:
    """Replace resid_post at prompt with ``prompt_mean`` and at each ``Guess:`` span token with the matching ``guess`` row."""

    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                guess_positions = _absolute_guess_span_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_guess_tokens=expected_guess_tokens,
                )
                if not guess_positions:
                    return activation

                region = layer_to_verbalised_embedding_means[layer_idx]
                prompt_mean = region["prompt_mean"].to(activation.dtype)
                guess_mean = region["guess"].to(activation.dtype)
                if guess_mean.ndim != 2:
                    raise ValueError(
                        f"Layer {layer_idx} guess mean must be rank-2 [n_pos, hidden]. Got {tuple(guess_mean.shape)}."
                    )
                if len(guess_positions) != int(guess_mean.shape[0]):
                    raise ValueError(
                        f"Layer {layer_idx}: Guess position count {len(guess_positions)} does not match "
                        f"replacement count {int(guess_mean.shape[0])}."
                    )

                for abs_pos in range(max(0, prompt_len - 1)):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = prompt_mean
                for pos_i, abs_pos in enumerate(guess_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = guess_mean[pos_i]
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_all_pre_guess_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    expected_guess_tokens: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_all_pre_guess_mean_replace_hooks(
        layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_guess_then_guess_and_probability_mean_replace_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    """
    Dynamic two-phase hooks:
    - If ``Guess:`` span is parseable, ablate guess span tokens.
    - If both ``Guess:`` and ``Probability:`` spans are parseable, also ablate probability span tokens.
    """
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                decoded_tokens = decoded_tokens_provider()
                guess_positions = _absolute_guess_span_positions(
                    prompt_len,
                    decoded_tokens,
                    expected_guess_tokens=expected_guess_tokens,
                )
                if not guess_positions:
                    return activation

                region = layer_to_verbalised_embedding_means[layer_idx]
                guess_mean = region["guess"].to(activation.dtype)
                if guess_mean.ndim != 2:
                    raise ValueError(
                        f"Layer {layer_idx} guess mean must be rank-2 [n_pos, hidden]. Got {tuple(guess_mean.shape)}."
                    )
                if len(guess_positions) != int(guess_mean.shape[0]):
                    raise ValueError(
                        f"Layer {layer_idx}: Guess position count {len(guess_positions)} does not match "
                        f"replacement count {int(guess_mean.shape[0])}."
                    )

                for pos_i, abs_pos in enumerate(guess_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = guess_mean[pos_i]

                probability_positions = _absolute_prob_positions(prompt_len, decoded_tokens)
                if not probability_positions:
                    return activation

                prob_mean = region["probability"].to(activation.dtype)
                if prob_mean.ndim != 2:
                    raise ValueError(
                        f"Layer {layer_idx} probability mean must be rank-2 [n_pos, hidden]. "
                        f"Got {tuple(prob_mean.shape)}."
                    )
                if len(probability_positions) != int(prob_mean.shape[0]):
                    # The precomputed means are fixed-length; only ablate once this parse shape matches.
                    if not _is_expected_or_plus_two(len(probability_positions), expected_probability_tokens):
                        raise ValueError(
                            f"Layer {layer_idx}: Probability position count {len(probability_positions)} is not "
                            f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
                        )
                    return activation

                for pos_i, abs_pos in enumerate(probability_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = prob_mean[pos_i]
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_guess_then_guess_and_probability_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_guess_then_guess_and_probability_mean_replace_hooks(
        layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def _absolute_probability_value_start_position(prompt_len: int, decoded_tokens: List[str]) -> Optional[int]:
    """Absolute position for first probability-value token (parser ``end_prob_token_index``)."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return None
    _, _, end_prob_token_index = parsed
    seq_len = prompt_len + len(decoded_tokens)
    abs_pos = _completion_token_index_to_abs_pos(prompt_len, end_prob_token_index)
    if 0 <= abs_pos < seq_len:
        return abs_pos
    return None


def build_probability_value_mean_replace_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    seq_len_provider: Callable[[], int],
    decoded_tokens_provider: Callable[[], List[str]],
) -> List[Tuple[str, Callable]]:
    """
    Dynamic hooks for probability value ablation:
    - no-op until ``parse_guess_and_probability_indices`` succeeds;
    - from first value token through current last position, use:
      * first position -> ``probability[-1]``
      * remaining positions -> ``probability_value_mean``.
    """
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                _ = seq_len_provider()  # keep parity with other dynamic hook builders
                start_abs = _absolute_probability_value_start_position(prompt_len, decoded_tokens_provider())
                if start_abs is None:
                    return activation

                seq_len = int(activation.shape[1])
                if start_abs >= seq_len:
                    return activation

                region = layer_to_verbalised_embedding_means[layer_idx]
                prob_mean = region["probability"]
                shared_value_mean = region["probability_value_mean"]
                if prob_mean.ndim != 2:
                    raise ValueError(
                        f"Layer {layer_idx} probability mean must be rank-2 [n_pos, hidden]. "
                        f"Got {tuple(prob_mean.shape)}."
                    )
                if int(prob_mean.shape[0]) < 1:
                    raise ValueError(f"Layer {layer_idx} probability mean has no positions to index.")
                if shared_value_mean.ndim != 1:
                    raise ValueError(
                        f"Layer {layer_idx} probability_value_mean must be rank-1 [hidden]. "
                        f"Got {tuple(shared_value_mean.shape)}."
                    )

                first_value_vec = prob_mean[-1].to(activation.dtype)
                shared_value_vec = shared_value_mean.to(activation.dtype)

                activation[:, start_abs, :] = first_value_vec
                if start_abs + 1 < seq_len:
                    activation[:, start_abs + 1 : seq_len, :] = shared_value_vec
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_probability_value_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _seq_len() -> int:
        return prompt_len + len(decoded_tokens)

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_probability_value_mean_replace_hooks(
        layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        seq_len_provider=_seq_len,
        decoded_tokens_provider=_decoded_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_semantic_answer_mean_replace_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    include_first_prob: bool = False,
) -> List[Tuple[str, Callable]]:
    """
    Dynamic hooks for semantic-answer ablation:
    - no-op until ``parse_guess_and_probability_indices`` succeeds;
    - replace every semantic-answer token with the shared ``sem_answer_mean`` vector.
    If ``include_first_prob``, also replace the residual at ``first_prob_token_index``.
    """
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                sem_positions = _absolute_sem_answer_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    include_first_prob=include_first_prob,
                )
                if not sem_positions:
                    return activation

                region = layer_to_verbalised_embedding_means[layer_idx]
                sem_answer_mean = region["sem_answer_mean"]
                if sem_answer_mean.ndim != 1:
                    raise ValueError(
                        f"Layer {layer_idx} sem_answer_mean must be rank-1 [hidden]. "
                        f"Got {tuple(sem_answer_mean.shape)}."
                    )

                shared_vec = sem_answer_mean.to(activation.dtype)
                for abs_pos in sem_positions:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = shared_vec
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_semantic_answer_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    include_first_prob: bool = False,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_semantic_answer_mean_replace_hooks(
        layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
        include_first_prob=include_first_prob,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_prompt_tokens_mean_replace_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
) -> List[Tuple[str, Callable]]:
    """
    Dynamic hooks for prompt ablation:
    - no-op until ``parse_guess_start_index`` succeeds;
    - replace every prompt token position with the shared ``prompt_mean`` vector.
    """
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if parse_guess_start_index(decoded_tokens_provider()) is None:
                    return activation

                region = layer_to_verbalised_embedding_means[layer_idx]
                prompt_mean = region["prompt_mean"]
                if prompt_mean.ndim != 1:
                    raise ValueError(
                        f"Layer {layer_idx} prompt_mean must be rank-1 [hidden]. "
                        f"Got {tuple(prompt_mean.shape)}."
                    )

                shared_vec = prompt_mean.to(activation.dtype)
                for abs_pos in range(max(0, prompt_len - 1)):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = shared_vec
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_prompt_tokens_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_prompt_tokens_mean_replace_hooks(
        layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_sem_ans_tokens_during_gen_hooks(
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
) -> List[Tuple[str, Callable]]:
    """
    Dynamic hooks for semantic-answer ablation during generation:
    - no-op until ``parse_guess_start_index`` succeeds;
    - replace semantic-answer tokens generated so far (and keep ablating those positions
      after the span is complete).
    """
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_verbalised_embedding_means:
        hook_name = hook_name_for_display_layer(layer)

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                sem_positions = _absolute_sem_answer_positions_during_gen(
                    prompt_len, decoded_tokens_provider()
                )
                if not sem_positions:
                    return activation

                region = layer_to_verbalised_embedding_means[layer_idx]
                sem_answer_mean = region["sem_answer_mean"]
                if sem_answer_mean.ndim != 1:
                    raise ValueError(
                        f"Layer {layer_idx} sem_answer_mean must be rank-1 [hidden]. "
                        f"Got {tuple(sem_answer_mean.shape)}."
                    )

                shared_vec = sem_answer_mean.to(activation.dtype)
                for abs_pos in sem_positions:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = shared_vec
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_sem_ans_tokens_during_gen(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_verbalised_embedding_means: Dict[int, Dict[str, torch.Tensor]],
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_sem_ans_tokens_during_gen_hooks(
        layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Layerwise mean activation replacement inference (TransformerLens).")
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
            "layerwise residual-stream ablation)."
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
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_and_probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "last_a_mean_replace",
            "last_a_and_panl_mean_replace",
            "last_a_panl_and_pc_mean_replace",
            "panl_mean_replace",
            "pc_mean_replace",
            "probability_value_mean_replace",
            "semantic_answer_mean_replace",
            "semantic_answer_including_first_prob_mean_replace",
            "prompt_tokens_mean_replace",
            "sem_ans_tokens_during_gen",
        ],
        choices=[
            "none",
            "probability_tokens_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_and_probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "last_a_mean_replace",
            "last_a_and_panl_mean_replace",
            "last_a_panl_and_pc_mean_replace",
            "panl_mean_replace",
            "pc_mean_replace",
            "probability_value_mean_replace",
            "semantic_answer_mean_replace",
            "semantic_answer_including_first_prob_mean_replace",
            "prompt_tokens_mean_replace",
            "sem_ans_tokens_during_gen",
        ],
        help=(
            "One or more modes to run. none: no hooks. probability_tokens_mean_replace: "
            "replace resid at H5 probability span. probability_last_token_mean_replace: "
            "same gating as probability_tokens_mean_replace but only the last marker-span token. "
            "probability_span_except_last_token_mean_replace: same gating but all marker-span "
            "tokens except that last token. last_a_mean_replace: same gating but "
            "only H5 probability row 0 (last answer token). last_a_and_panl_mean_replace: rows 0 and 1 "
            "(last answer token and post-answer newline). "
            "last_a_panl_and_pc_mean_replace: rows 0, 1, and a model-specific pre-confidence index "
            "(Mistral 6, Gemma 3; unsupported for Qwen). "
            "panl_mean_replace: only H5 probability row 1 (post-answer newline). "
            "pc_mean_replace: only the model-specific pre-confidence index "
            "(Mistral 6, Gemma 3; unsupported for Qwen). "
            "all_pre_probability_tokens_mean_replace: "
            "replace resid at prompt + Guess + semantic-answer + Probability marker positions. "
            "guess_tokens_mean_replace: replace resid only at the Guess: prefix span (per-position means). "
            "all_pre_guess_tokens_mean_replace: replace resid at every prompt position with prompt_mean, and at "
            "each Guess: prefix token with the corresponding row of guess (H5 per-position means). "
            "guess_then_guess_and_probability_tokens_mean_replace: greedy decode with dynamic span ablation; "
            "once Guess: is parseable, ablate the Guess: span, and once both Guess: and Probability: are parseable, "
            "ablate both spans. probability_value_mean_replace: no hooks until Guess/Probability parse succeeds; "
            "then ablate from first probability-value token through current last token each step, using "
            "probability[-1] for the first position and embeddings_mean_prob_val mean for later positions. "
            "semantic_answer_mean_replace: no hooks until Guess/Probability parse succeeds; then replace "
            "every semantic-answer token with the shared embeddings_mean_sem_answer group mean. "
            "semantic_answer_including_first_prob_mean_replace: same as semantic_answer_mean_replace but "
            "the ablated span also includes first_prob_token_index (residual whose input is the last "
            "semantic-answer token). "
            "prompt_tokens_mean_replace: no-op until Guess: prefix parses; then replace all prompt token "
            "positions with the shared embeddings_mean_prompt group mean. "
            "sem_ans_tokens_during_gen: no-op until Guess: prefix parses; replace semantic-answer tokens "
            "generated so far (and keep ablating those positions after the span is complete). "
        ),
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--all_confidence_group_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), run all four mean-source × target-group combinations "
            "(mean_low_target_high, mean_high_target_low, mean_low_target_low, mean_high_target_high) "
            "and ignore --mean_from_low_confidence / --ablate_with_same_confidence. "
            "If false, run a single combination from those two flags."
        ),
    )
    parser.add_argument(
        "--mean_from_low_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), compute means from low-confidence examples and ablate high-confidence examples. "
            "If false, reverse source and target groups. "
            "When --ablate_with_same_confidence is set, targets match the mean-source group instead. "
            "Ignored when --all_confidence_group_pairs is true."
        ),
    )
    parser.add_argument(
        "--ablate_with_same_confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, ablate examples from the same confidence group used for the mean "
            "(within-group control). If false (default), ablate the opposite group. "
            "Ignored when --all_confidence_group_pairs is true."
        ),
    )
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--parse_mode_verbalised_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), parse verbalised confidence from each generated mode response, "
            "store comparison metadata in output JSON, and write per-mode means to config.txt."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "layerwise_mean_ablation/results/<incrementing_run_id>/ablation_results.json. "
            "When --all_confidence_group_pairs is true, combo subdirectories are created under "
            "that run directory."
        ),
    )
    parser.add_argument(
        "--individual_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, ignore --ablate_layers and run a separate ablation for each "
            "display layer (n_layers+1 rows: 0=embedding resid-pre, k>=1=resid-post "
            "of TL block k-1). Outputs are written under "
            "results/individual_layers/<run_id>/<layer_idx>/, or under "
            "results/individual_layers/<run_id>/<combo>/<layer_idx>/ when "
            "--all_confidence_group_pairs is true."
        ),
    )
    args = parser.parse_args()
    configure_prefix_tokens_for_model(args.model_name)
    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    validate_last_a_panl_and_pc_mode(args.model_name, args.ablation_mode)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

    logging.info("Loading HookedTransformer: %s", args.model_name)
    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers + 1)
    run_layers = list(range(model.cfg.n_layers + 1)) if args.individual_layers else ablate_layers

    examples_h5 = load_examples_h5(Path(args.input_h5))
    modes = set(args.ablation_mode)
    need_verbalised_embedding_means = (
        "probability_tokens_mean_replace" in modes
        or "probability_last_token_mean_replace" in modes
        or "probability_span_except_last_token_mean_replace" in modes
        or any(mode in PROBABILITY_ROW_INDEX_MODES for mode in modes)
        or "all_pre_probability_tokens_mean_replace" in modes
        or "guess_tokens_mean_replace" in modes
        or "all_pre_guess_tokens_mean_replace" in modes
        or "guess_then_guess_and_probability_tokens_mean_replace" in modes
        or "probability_value_mean_replace" in modes
        or "semantic_answer_mean_replace" in modes
        or "semantic_answer_including_first_prob_mean_replace" in modes
        or "prompt_tokens_mean_replace" in modes
        or "sem_ans_tokens_during_gen" in modes
    )

    low_ids: set[str]
    high_ids: set[str]
    means_by_source: Dict[bool, Optional[Dict[int, Dict[str, torch.Tensor]]]] = {True: None, False: None}

    if args.all_confidence_group_pairs:
        logging.info(
            "all_confidence_group_pairs=True; ignoring --mean_from_low_confidence=%s and "
            "--ablate_with_same_confidence=%s.",
            args.mean_from_low_confidence,
            args.ablate_with_same_confidence,
        )
        source_flags: Tuple[bool, ...] = (True, False)
        combos: List[Tuple[Optional[str], bool, bool]] = [
            (name, mean_from_low, same) for name, mean_from_low, same in CONFIDENCE_GROUP_PAIR_COMBOS
        ]
    else:
        source_flags = (bool(args.mean_from_low_confidence),)
        combos = [(None, bool(args.mean_from_low_confidence), bool(args.ablate_with_same_confidence))]

    if need_verbalised_embedding_means:
        for mean_from_low in source_flags:
            np_means, low_ids, high_ids = compute_verbalised_embedding_group_means(
                examples_h5,
                run_layers,
                low_conf_threshold=args.low_conf_threshold,
                high_conf_threshold=args.high_conf_threshold,
                expected_probability_tokens=args.expected_probability_tokens,
                expected_guess_tokens=args.expected_guess_tokens,
                mean_from_low_confidence=mean_from_low,
                new_h5_format=args.new_h5_format,
                h5_res_indices=run_layers,
            )
            means_by_source[mean_from_low] = layer_tensor_means_from_numpy(
                np_means, run_layers, device=device, torch_dtype=torch_dtype
            )
    else:
        low_ids, high_ids = collect_confidence_group_ids(
            examples_h5,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
        )

    logging.info("Low-confidence example IDs: %s", low_ids)
    logging.info("High-confidence example IDs: %s", high_ids)
    eval_low_ids = set(truncate_sorted_ids(low_ids, args.num_samples))
    eval_high_ids = set(truncate_sorted_ids(high_ids, args.num_samples))
    logging.info(
        "Eval sample cap num_samples=%d: eval_low=%d eval_high=%d (from full low=%d high=%d)",
        args.num_samples,
        len(eval_low_ids),
        len(eval_high_ids),
        len(low_ids),
        len(high_ids),
    )
    logging.info(
        "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), layers=%s, "
        "all_confidence_group_pairs=%s, mean_from_low_confidence=%s, ablate_with_same_confidence=%s, "
        "individual_layers=%s",
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        run_layers,
        args.all_confidence_group_pairs,
        args.mean_from_low_confidence,
        args.ablate_with_same_confidence,
        args.individual_layers,
    )
    for combo_name, mean_from_low, same in combos:
        mean_source_ids = low_ids if mean_from_low else high_ids
        combo_target_ids = resolve_combo_target_ids(
            eval_low_ids,
            eval_high_ids,
            mean_from_low_confidence=mean_from_low,
            ablate_with_same_confidence=same,
        )
        mean_source_group, ablation_target_group = resolve_confidence_groups(
            mean_from_low_confidence=mean_from_low,
            ablate_with_same_confidence=same,
        )
        logging.info(
            "Confidence combo %s: mean_source=%s ablation_target=%s "
            "(mean_source_count=%d ablation_target_count=%d)",
            combo_name or "single",
            mean_source_group,
            ablation_target_group,
            len(mean_source_ids),
            len(combo_target_ids),
        )

    def run_one_evaluation(
        # This contains {"prompt_mean", "guess", "sem_answer_mean", "probability", "probability_value_mean"}
        layer_to_verbalised_embedding_means_eval: Optional[Dict[int, Dict[str, torch.Tensor]]],
        ablation_target_ids: set[str],
        mean_from_low_confidence: bool,
        cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
        sentence_transformer=None,
    ) -> Tuple[
        Dict[str, dict],
        Dict[str, dict],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, Dict[str, Dict[str, object]]],
        Dict[str, int],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, Optional[float]],
        Dict[str, int],
    ]:
        if layer_to_verbalised_embedding_means_eval is not None:
            for layer_idx in layer_to_verbalised_embedding_means_eval:
                logging.info(
                    "layer_to_verbalised_embedding_means_eval[%s]['guess'].shape: %s",
                    layer_idx,
                    layer_to_verbalised_embedding_means_eval[layer_idx]["guess"].shape,
                )
                logging.info(
                    "layer_to_verbalised_embedding_means_eval[%s]['probability'].shape: %s",
                    layer_idx,
                    layer_to_verbalised_embedding_means_eval[layer_idx]["probability"].shape,
                )
                logging.info(
                    "layer_to_verbalised_embedding_means_eval[%s]['sem_answer_mean'].shape: %s",
                    layer_idx,
                    layer_to_verbalised_embedding_means_eval[layer_idx]["sem_answer_mean"].shape,
                )
                logging.info(
                    "layer_to_verbalised_embedding_means_eval[%s]['prompt_mean'].shape: %s",
                    layer_idx,
                    layer_to_verbalised_embedding_means_eval[layer_idx]["prompt_mean"].shape,
                )
                logging.info(
                    "layer_to_verbalised_embedding_means_eval[%s]['probability_value_mean'].shape: %s",
                    layer_idx,
                    layer_to_verbalised_embedding_means_eval[layer_idx]["probability_value_mean"].shape,
                )
                break
        results = {"train": {}, "validation": {}}
        mini_results = {"train": {}, "validation": {}}
        mode_confidence_values: Dict[str, List[float]] = {mode_name: [] for mode_name in args.ablation_mode}
        mode_responses_identical_true: Dict[str, int] = {
            mode_name: 0 for mode_name in args.ablation_mode if mode_name != "none"
        }
        mode_semantic_similarity_values: Dict[str, List[float]] = {
            mode_name: [] for mode_name in args.ablation_mode if mode_name != "none"
        }
        mode_verbalised_confidence_effect_values: Dict[str, List[float]] = {
            mode_name: [] for mode_name in args.ablation_mode if mode_name != "none"
        }
        mode_uncertainty_score_values: Dict[str, List[float]] = {
            mode_name: [] for mode_name in args.ablation_mode if mode_name != "none"
        }
        pending_semantic_similarity: List[
            Tuple[str, str, str, str, str]
        ] = []

        modes = args.ablation_mode
        has_none_and_other_modes = ("none" in modes) and (len(modes) > 1)
        used_none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}

        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
            selected_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)

            if not selected_ids:
                logging.warning(
                    "No ablation target IDs available for %s split (mean_from_low_confidence=%s).",
                    split_name,
                    mean_from_low_confidence,
                )
                continue

            logging.info(
                "Generating for %d examples (%s split, target confidence subset).",
                len(selected_ids),
                split_name,
            )

            for i, ex_id in enumerate(selected_ids):
                ds_idx = id_to_index.get(ex_id)
                if ds_idx is None:
                    raise ValueError(f"Example id {ex_id} selected from H5 but not found in {split_name} split.")
                example = eval_ds[int(ds_idx)]
                question = example["question"]
                local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + question

                entry = {"question": question}
                mini_entry = {"question": question}

                for mode in modes:
                    key = mode_to_output_key(mode)
                    if mode == "none" and cached_none is not None and ex_id in cached_none[split_name]:
                        cached = cached_none[split_name][ex_id]
                        response = str(cached["response"])
                        decoded_tokens = list(cached["decoded_tokens"])
                        mode_confidence = cached.get("verbalised_confidence")
                    elif mode == "none":
                        # Generate baseline none mode response
                        response, decoded_tokens = greedy_generate(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            fwd_hooks=None,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                        used_none_cache[split_name][ex_id] = {
                            "response": response,
                            "decoded_tokens": decoded_tokens,
                            "verbalised_confidence": mode_confidence,
                        }
                    elif mode == "probability_tokens_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError("Mode probability_tokens_mean_replace requested but probability means are unavailable.")
                        layer_to_mean_vectors_eval = {
                            layer_idx: layer_to_verbalised_embedding_means_eval[layer_idx]["probability"]
                            for layer_idx in layer_to_verbalised_embedding_means_eval
                        }
                        response, decoded_tokens = greedy_generate_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_mean_vectors=layer_to_mean_vectors_eval,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "probability_last_token_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode probability_last_token_mean_replace requested but probability means are unavailable."
                            )
                        layer_to_mean_vectors_eval = {
                            layer_idx: layer_to_verbalised_embedding_means_eval[layer_idx]["probability"][-1:, :]
                            for layer_idx in layer_to_verbalised_embedding_means_eval
                        }
                        response, decoded_tokens = greedy_generate_probability_span_subset_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_mean_vectors=layer_to_mean_vectors_eval,
                            subset="last_token",
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "probability_span_except_last_token_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode probability_span_except_last_token_mean_replace requested but "
                                "probability means are unavailable."
                            )
                        layer_to_mean_vectors_eval = {
                            layer_idx: layer_to_verbalised_embedding_means_eval[layer_idx]["probability"][:-1, :]
                            for layer_idx in layer_to_verbalised_embedding_means_eval
                        }
                        response, decoded_tokens = greedy_generate_probability_span_subset_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_mean_vectors=layer_to_mean_vectors_eval,
                            subset="except_last_token",
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode in PROBABILITY_ROW_INDEX_MODES:
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                f"Mode {mode} requested but probability means are unavailable."
                            )
                        row_indices = probability_row_indices_for_mode(mode, args.model_name)
                        layer_to_mean_vectors_eval = {
                            layer_idx: layer_to_verbalised_embedding_means_eval[layer_idx]["probability"]
                            for layer_idx in layer_to_verbalised_embedding_means_eval
                        }
                        response, decoded_tokens = greedy_generate_probability_row_indices_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_mean_vectors=layer_to_mean_vectors_eval,
                            row_indices=row_indices,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "all_pre_probability_tokens_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode all_pre_probability_tokens_mean_replace requested but verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_pre_probability_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                            expected_guess_tokens=args.expected_guess_tokens,
                            expected_probability_tokens=args.expected_probability_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "guess_tokens_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode guess_tokens_mean_replace requested but verbalised embedding means are unavailable."
                            )
                        layer_to_guess_mean_vectors = {
                            layer_idx: layer_to_verbalised_embedding_means_eval[layer_idx]["guess"]
                            for layer_idx in layer_to_verbalised_embedding_means_eval
                        }

                        response, decoded_tokens = greedy_generate_guess_tokens_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_guess_mean_vectors=layer_to_guess_mean_vectors,
                            expected_guess_tokens=args.expected_guess_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "all_pre_guess_tokens_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode all_pre_guess_tokens_mean_replace requested but verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_all_pre_guess_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                            expected_guess_tokens=args.expected_guess_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "guess_then_guess_and_probability_tokens_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode guess_then_guess_and_probability_tokens_mean_replace requested but "
                                "verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_guess_then_guess_and_probability_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                            expected_guess_tokens=args.expected_guess_tokens,
                            expected_probability_tokens=args.expected_probability_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "probability_value_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode probability_value_mean_replace requested but verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_probability_value_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode in (
                        "semantic_answer_mean_replace",
                        "semantic_answer_including_first_prob_mean_replace",
                    ):
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                f"Mode {mode} requested but verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_semantic_answer_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                            include_first_prob=(
                                mode == "semantic_answer_including_first_prob_mean_replace"
                            ),
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "prompt_tokens_mean_replace":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode prompt_tokens_mean_replace requested but verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_prompt_tokens_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "sem_ans_tokens_during_gen":
                        if layer_to_verbalised_embedding_means_eval is None:
                            raise ValueError(
                                "Mode sem_ans_tokens_during_gen requested but verbalised embedding means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_sem_ans_tokens_during_gen(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_verbalised_embedding_means=layer_to_verbalised_embedding_means_eval,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    else:
                        raise ValueError(f"Unknown ablation mode: {mode!r}")

                    entry[key] = {"response": response, "decoded_tokens": decoded_tokens}
                    mini_entry[key] = {"response": response}
                    if args.parse_mode_verbalised_confidence:
                        entry[key]["verbalised_confidence"] = mode_confidence
                        mini_entry[key]["verbalised_confidence"] = mode_confidence
                        if mode_confidence is not None:
                            mode_confidence_values[mode].append(float(mode_confidence))
                    logging.info(
                        "[%s %d/%d] %s %s first line: %r",
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                        key,
                        response[:120],
                    )

                if has_none_and_other_modes:
                    baseline_key = mode_to_output_key("none")
                    baseline_response = entry[baseline_key]["response"]
                    baseline_confidence = (
                        entry[baseline_key].get("verbalised_confidence")
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    for mode in modes:
                        if mode == "none":
                            continue
                        mode_key = mode_to_output_key(mode)
                        responses_identical = entry[mode_key]["response"] == baseline_response
                        entry[mode_key]["responses_identical"] = responses_identical
                        mini_entry[mode_key]["responses_identical"] = responses_identical
                        if responses_identical:
                            mode_responses_identical_true[mode] += 1

                        if args.parse_mode_verbalised_confidence:
                            mode_confidence = entry[mode_key].get("verbalised_confidence")
                            if mode_confidence is None or baseline_confidence is None:
                                meets_none_confidence_direction = None
                            elif mean_from_low_confidence:
                                meets_none_confidence_direction = mode_confidence < baseline_confidence
                            else:
                                meets_none_confidence_direction = mode_confidence > baseline_confidence
                            entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                            mini_entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction

                for mode in modes:
                    mode_key = mode_to_output_key(mode)
                    semantic_answer = parse_semantic_answer_from_response(str(entry[mode_key]["response"]))
                    if semantic_answer is not None:
                        entry[mode_key]["semantic_answer"] = semantic_answer
                        mini_entry[mode_key]["semantic_answer"] = semantic_answer

                if has_none_and_other_modes:
                    baseline_key = mode_to_output_key("none")
                    baseline_semantic_answer = entry[baseline_key].get("semantic_answer")
                    baseline_confidence = (
                        entry[baseline_key].get("verbalised_confidence")
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    for mode in modes:
                        if mode == "none":
                            continue
                        mode_key = mode_to_output_key(mode)
                        mode_semantic_answer = entry[mode_key].get("semantic_answer")

                        if args.parse_mode_verbalised_confidence:
                            mode_confidence = entry[mode_key].get("verbalised_confidence")
                            if mode_confidence is not None and baseline_confidence is not None:
                                vce = compute_verbalised_confidence_effect(
                                    float(baseline_confidence),
                                    float(mode_confidence),
                                    mean_from_low_confidence=mean_from_low_confidence,
                                )
                                if vce is not None:
                                    entry[mode_key]["verbalised_confidence_effect"] = vce
                                    mini_entry[mode_key]["verbalised_confidence_effect"] = vce
                                    mode_verbalised_confidence_effect_values[mode].append(vce)

                        if (
                            mode in GUESS_SPAN_ABLATION_MODES
                            and sentence_transformer is not None
                            and baseline_semantic_answer is not None
                            and mode_semantic_answer is not None
                        ):
                            pending_semantic_similarity.append(
                                (
                                    split_name,
                                    ex_id,
                                    mode_key,
                                    str(baseline_semantic_answer),
                                    str(mode_semantic_answer),
                                )
                            )

                results[split_name][ex_id] = entry
                mini_results[split_name][ex_id] = mini_entry

        if pending_semantic_similarity and sentence_transformer is not None:
            pairs = [(baseline_text, mode_text) for _split, _ex, _key, baseline_text, mode_text in pending_semantic_similarity]
            similarities = batch_compute_semantic_similarities(sentence_transformer, pairs)
            mode_key_to_mode = {mode_to_output_key(mode): mode for mode in modes if mode != "none"}
            for task, similarity in zip(pending_semantic_similarity, similarities):
                split_name, ex_id, mode_key, _baseline_text, _mode_text = task
                mode_name = mode_key_to_mode.get(mode_key)
                if mode_name is None:
                    continue
                entry = results[split_name][ex_id][mode_key]
                mini_entry = mini_results[split_name][ex_id][mode_key]
                entry["semantic_similarity"] = similarity
                mini_entry["semantic_similarity"] = similarity
                mode_semantic_similarity_values[mode_name].append(similarity)
                vce = entry.get("verbalised_confidence_effect")
                if vce is not None:
                    uncertainty = compute_uncertainty_score(similarity, float(vce))
                    entry["uncertainty_score"] = uncertainty
                    mini_entry["uncertainty_score"] = uncertainty
                    mode_uncertainty_score_values[mode_name].append(uncertainty)

        mode_confidence_means: Dict[str, Optional[float]] = {}
        mode_confidence_counts: Dict[str, int] = {}
        for mode_name in modes:
            values = mode_confidence_values[mode_name]
            mode_confidence_means[mode_name] = float(np.mean(values)) if values else None
            mode_confidence_counts[mode_name] = len(values)

        mode_semantic_similarity_means: Dict[str, Optional[float]] = {}
        mode_semantic_similarity_counts: Dict[str, int] = {}
        mode_verbalised_confidence_effect_means: Dict[str, Optional[float]] = {}
        mode_verbalised_confidence_effect_counts: Dict[str, int] = {}
        mode_uncertainty_score_means: Dict[str, Optional[float]] = {}
        mode_uncertainty_score_counts: Dict[str, int] = {}
        for mode_name in modes:
            if mode_name == "none":
                continue
            sem_mean, sem_count = _mean_and_count(mode_semantic_similarity_values[mode_name])
            vce_mean, vce_count = _mean_and_count(mode_verbalised_confidence_effect_values[mode_name])
            unc_mean, unc_count = _mean_and_count(mode_uncertainty_score_values[mode_name])
            mode_semantic_similarity_means[mode_name] = sem_mean
            mode_semantic_similarity_counts[mode_name] = sem_count
            mode_verbalised_confidence_effect_means[mode_name] = vce_mean
            mode_verbalised_confidence_effect_counts[mode_name] = vce_count
            mode_uncertainty_score_means[mode_name] = unc_mean
            mode_uncertainty_score_counts[mode_name] = unc_count

        return (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            used_none_cache,
            mode_responses_identical_true,
            mode_semantic_similarity_means,
            mode_semantic_similarity_counts,
            mode_verbalised_confidence_effect_means,
            mode_verbalised_confidence_effect_counts,
            mode_uncertainty_score_means,
            mode_uncertainty_score_counts,
        )

    # Compute the output for baseline none mode (no ablation) so individual-layer runs can reuse the same baseline.
    def build_none_cache(ablation_target_ids: set[str]) -> Dict[str, Dict[str, Dict[str, object]]]:
        none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}
        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
            selected_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)
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
                    max_new_tokens=args.model_max_new_tokens,
                    fwd_hooks=None,
                )
                mode_confidence = (
                    parse_mode_confidence_from_response(response)
                    if args.parse_mode_verbalised_confidence
                    else None
                )
                none_cache[split_name][ex_id] = {
                    "response": response,
                    "decoded_tokens": decoded_tokens,
                    "verbalised_confidence": mode_confidence,
                }
        return none_cache

    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)

    sentence_transformer = None
    if "none" in args.ablation_mode and len(args.ablation_mode) > 1:
        if any(mode in GUESS_SPAN_ABLATION_MODES for mode in args.ablation_mode):
            logging.info(
                "Loading sentence-transformers model %s for semantic_similarity.",
                DEFAULT_SEMANTIC_SIMILARITY_MODEL,
            )
            sentence_transformer = load_sentence_transformer_for_metrics()

    none_cache_by_target: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    precompute_none = "none" in args.ablation_mode and (
        args.individual_layers or args.all_confidence_group_pairs
    )
    if precompute_none:
        for _combo_name, mean_from_low, same in combos:
            _source_group, target_group = resolve_confidence_groups(
                mean_from_low_confidence=mean_from_low,
                ablate_with_same_confidence=same,
            )
            if target_group in none_cache_by_target:
                continue
            target_ids = resolve_combo_target_ids(
                eval_low_ids,
                eval_high_ids,
                mean_from_low_confidence=mean_from_low,
                ablate_with_same_confidence=same,
            )
            logging.info(
                "Computing baseline none-mode for target group %s (%d ids).",
                target_group,
                len(target_ids),
            )
            none_cache_by_target[target_group] = build_none_cache(target_ids)

    def persist_run_outputs(
        json_path: str,
        *,
        results: dict,
        mini_results: dict,
        ablate_layers_for_config: Sequence[int],
        mean_from_low_confidence: bool,
        ablate_with_same_confidence: bool,
        mode_confidence_means: Dict[str, Optional[float]],
        mode_confidence_counts: Dict[str, int],
        mode_responses_identical_true: Dict[str, int],
        derived_metric_kwargs: Dict[str, object],
    ) -> None:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", json_path)

        mini_out_path = mini_output_json_path(json_path)
        with open(mini_out_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", mini_out_path)

        config_out_path = config_txt_path(json_path)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        write_config_txt(
            config_out_path,
            args=args,
            device=device,
            model_n_layers=model.cfg.n_layers,
            ablate_layers=ablate_layers_for_config,
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            mean_from_low_confidence=mean_from_low_confidence,
            parse_mode_verbalised_confidence=args.parse_mode_verbalised_confidence,
            mode_confidence_means=mode_confidence_means,
            mode_confidence_counts=mode_confidence_counts,
            mode_responses_identical_true=mode_responses_identical_true,
            finished_at=finished_at,
            ablate_with_same_confidence=ablate_with_same_confidence,
            all_confidence_group_pairs=args.all_confidence_group_pairs,
            **derived_metric_kwargs,
        )
        logging.info("Wrote %s", config_out_path)

    def derived_metrics_from_eval(
        mode_semantic_similarity_means,
        mode_semantic_similarity_counts,
        mode_verbalised_confidence_effect_means,
        mode_verbalised_confidence_effect_counts,
        mode_uncertainty_score_means,
        mode_uncertainty_score_counts,
    ) -> Dict[str, object]:
        if not ("none" in args.ablation_mode and len(args.ablation_mode) > 1):
            return {}
        return {
            "mode_semantic_similarity_means": mode_semantic_similarity_means,
            "mode_semantic_similarity_counts": mode_semantic_similarity_counts,
            "mode_verbalised_confidence_effect_means": mode_verbalised_confidence_effect_means,
            "mode_verbalised_confidence_effect_counts": mode_verbalised_confidence_effect_counts,
            "mode_uncertainty_score_means": mode_uncertainty_score_means,
            "mode_uncertainty_score_counts": mode_uncertainty_score_counts,
        }

    if not args.individual_layers:
        for combo_name, mean_from_low, same in combos:
            _source_group, target_group = resolve_confidence_groups(
                mean_from_low_confidence=mean_from_low,
                ablate_with_same_confidence=same,
            )
            combo_target_ids = resolve_combo_target_ids(
                eval_low_ids,
                eval_high_ids,
                mean_from_low_confidence=mean_from_low,
                ablate_with_same_confidence=same,
            )
            layer_means = means_by_source[mean_from_low]
            combo_json_path = (
                out_path
                if combo_name is None
                else os.path.join(run_root, combo_name, "ablation_results.json")
            )
            logging.info(
                "Running mean-ablation combo %s (mean_source=%s, target=%s).",
                combo_name or "single",
                "low_confidence" if mean_from_low else "high_confidence",
                target_group,
            )
            (
                results,
                mini_results,
                mode_confidence_means,
                mode_confidence_counts,
                _,
                mode_responses_identical_true,
                mode_semantic_similarity_means,
                mode_semantic_similarity_counts,
                mode_verbalised_confidence_effect_means,
                mode_verbalised_confidence_effect_counts,
                mode_uncertainty_score_means,
                mode_uncertainty_score_counts,
            ) = run_one_evaluation(
                layer_means,
                combo_target_ids,
                mean_from_low,
                cached_none=none_cache_by_target.get(target_group),
                sentence_transformer=sentence_transformer,
            )
            persist_run_outputs(
                combo_json_path,
                results=results,
                mini_results=mini_results,
                ablate_layers_for_config=ablate_layers,
                mean_from_low_confidence=mean_from_low,
                ablate_with_same_confidence=same,
                mode_confidence_means=mode_confidence_means,
                mode_confidence_counts=mode_confidence_counts,
                mode_responses_identical_true=mode_responses_identical_true,
                derived_metric_kwargs=derived_metrics_from_eval(
                    mode_semantic_similarity_means,
                    mode_semantic_similarity_counts,
                    mode_verbalised_confidence_effect_means,
                    mode_verbalised_confidence_effect_counts,
                    mode_uncertainty_score_means,
                    mode_uncertainty_score_counts,
                ),
            )
        return

    run_root_norm = run_root.rstrip(os.sep)
    run_id = os.path.basename(run_root_norm)
    results_root = os.path.dirname(run_root_norm)
    individual_root = os.path.join(results_root, "individual_layers", run_id)
    os.makedirs(individual_root, exist_ok=True)

    for combo_name, mean_from_low, same in combos:
        _source_group, target_group = resolve_confidence_groups(
            mean_from_low_confidence=mean_from_low,
            ablate_with_same_confidence=same,
        )
        combo_target_ids = resolve_combo_target_ids(
            eval_low_ids,
            eval_high_ids,
            mean_from_low_confidence=mean_from_low,
            ablate_with_same_confidence=same,
        )
        layer_means = means_by_source[mean_from_low]
        combo_root = individual_root if combo_name is None else os.path.join(individual_root, combo_name)
        os.makedirs(combo_root, exist_ok=True)
        cached_none = none_cache_by_target.get(target_group)
        logging.info(
            "Running individual-layer combo %s (mean_source=%s, target=%s).",
            combo_name or "single",
            "low_confidence" if mean_from_low else "high_confidence",
            target_group,
        )

        per_layer_mode_means: Dict[int, Dict[str, Optional[float]]] = {}
        for layer_idx in run_layers:
            logging.info("Running individual-layer ablation for layer %d (%s)", layer_idx, combo_name or "single")
            layer_dir = os.path.join(combo_root, str(layer_idx))
            os.makedirs(layer_dir, exist_ok=True)
            layer_pre_map = None if layer_means is None else {layer_idx: layer_means[layer_idx]}
            (
                results,
                mini_results,
                mode_confidence_means,
                mode_confidence_counts,
                _,
                mode_responses_identical_true,
                mode_semantic_similarity_means,
                mode_semantic_similarity_counts,
                mode_verbalised_confidence_effect_means,
                mode_verbalised_confidence_effect_counts,
                mode_uncertainty_score_means,
                mode_uncertainty_score_counts,
            ) = run_one_evaluation(
                layer_pre_map,
                combo_target_ids,
                mean_from_low,
                cached_none=cached_none,
                sentence_transformer=sentence_transformer,
            )
            per_layer_mode_means[layer_idx] = mode_confidence_means
            persist_run_outputs(
                os.path.join(layer_dir, "ablation_results.json"),
                results=results,
                mini_results=mini_results,
                ablate_layers_for_config=[layer_idx],
                mean_from_low_confidence=mean_from_low,
                ablate_with_same_confidence=same,
                mode_confidence_means=mode_confidence_means,
                mode_confidence_counts=mode_confidence_counts,
                mode_responses_identical_true=mode_responses_identical_true,
                derived_metric_kwargs=derived_metrics_from_eval(
                    mode_semantic_similarity_means,
                    mode_semantic_similarity_counts,
                    mode_verbalised_confidence_effect_means,
                    mode_verbalised_confidence_effect_counts,
                    mode_uncertainty_score_means,
                    mode_uncertainty_score_counts,
                ),
            )

        write_individual_layer_summary_and_plot(
            combo_root,
            args=args,
            model_n_layers=model.cfg.n_layers,
            run_layers=run_layers,
            mean_from_low_confidence=mean_from_low,
            ablate_with_same_confidence=same,
            all_confidence_group_pairs=args.all_confidence_group_pairs,
            per_layer_mode_means=per_layer_mode_means,
        )


if __name__ == "__main__":
    main()
