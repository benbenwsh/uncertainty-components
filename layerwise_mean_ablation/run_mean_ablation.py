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
from datetime import datetime
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
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

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

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n"]

# ---------------------------------------------------------------------------
# Probability span parsing (aligned with process_generations_verbalised_embeddings_h5.py)
# ---------------------------------------------------------------------------

GUESS_PREFIX = "\n\nGuess:"
PROBABILITY_MARKER = "\nProbability:"


def _token_index_for_char_offset(decoded_tokens: List[str], char_offset: int) -> int:
    cumulative = 0
    for i, tok in enumerate(decoded_tokens):
        cumulative += len(tok)
        if cumulative > char_offset:
            return i
    return max(0, len(decoded_tokens) - 1)

# def parse_guess_and_probability_indices(
#     decoded_tokens: List[str],
#     full_str: str,
# ) -> Optional[Tuple[int, int, int]]:
#     """Parse ``full_str`` (``join(decoded_tokens)``) into token indices in ``decoded_tokens``.

#     Returns (first_token_after_GUESS_PREFIX, first_prob_token, last_prob_token), or ``None`` if format or bounds checks fail.
#     """
#     if not full_str.startswith(GUESS_PREFIX):
#         return None

#     last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1

#     rfind_start = full_str.rfind(PROBABILITY_MARKER)
#     if rfind_start < 0:
#         return None

#     first_prob_token_index = _token_index_for_char_offset(decoded_tokens, rfind_start)
#     end_prob_token_index = _token_index_for_char_offset(
#         decoded_tokens, rfind_start + len(PROBABILITY_MARKER) - 1
#     ) + 1

#     if (
#         last_guess_token_index <= 0
#         or last_guess_token_index >= len(decoded_tokens)
#         or end_prob_token_index >= len(decoded_tokens)
#         or last_guess_token_index >= first_prob_token_index
#         or first_prob_token_index >= end_prob_token_index
#     ):
#         return None

#     return (last_guess_token_index, first_prob_token_index, end_prob_token_index)

def parse_guess_and_probability_indices(
    decoded_tokens: list,
) -> tuple[int, int, int] | None:
    """
    Compute token indices for the two embedding subsets (Guess and Probability).

    last_guess_token_index: token after "Guess:" (first token of answer)
    first_prob_token_index: "\n" token before "Probability:"
    end_prob_token_index: token after "Probability:\s" (first token of prob value)

    Returns (last_guess_token_index, first_prob_token_index, end_prob_token_index)
    or None on failure.
    """
    full_str = "".join(decoded_tokens)
    if not full_str.startswith(GUESS_PREFIX):
        return None

    last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1

    rfind_start = full_str.rfind(PROBABILITY_MARKER)
    if rfind_start < 0:
        return None

    first_prob_token_index = _token_index_for_char_offset(decoded_tokens, rfind_start) # No -1 because rfind_start is an index not length
    prob_whitespace_token_index = _token_index_for_char_offset(
        decoded_tokens, rfind_start + len(PROBABILITY_MARKER) - 1
    ) + 1
    if prob_whitespace_token_index >= len(decoded_tokens):
        return None
    if decoded_tokens[prob_whitespace_token_index].strip() != "":
        return None
    end_prob_token_index = prob_whitespace_token_index + 1

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
    full_str = "".join(decoded_tokens)
    if not full_str.startswith(GUESS_PREFIX):
        return None
    last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1
    if last_guess_token_index <= 0 or last_guess_token_index > len(decoded_tokens):
        return None
    return last_guess_token_index


# ---------------------------------------------------------------------------
# Dataset + few-shot
# ---------------------------------------------------------------------------


def load_trivia_qa(seed: int) -> Tuple[Dataset, Dataset]:
    raw = load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")["unmodified"]
    split = raw.train_test_split(test_size=0.2, seed=seed)
    return split["train"], split["test"]


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
    finished_at: str,
) -> None:
    source_group = "low_confidence" if mean_from_low_confidence else "high_confidence"
    target_group = "high_confidence" if mean_from_low_confidence else "low_confidence"
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
        f"mean_from_low_confidence={mean_from_low_confidence}",
        f"mean_source_group={source_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"parse_mode_verbalised_confidence={parse_mode_verbalised_confidence}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Mode Confidence Metrics]",
    ]
    for mode_name in args.ablation_mode:
        mode_mean = mode_confidence_means.get(mode_name)
        if mode_mean is None:
            lines.append(f"{mode_name}_mean_verbalised_confidence=None")
        else:
            lines.append(f"{mode_name}_mean_verbalised_confidence={mode_mean:.6f}")
        lines.append(
            f"{mode_name}_verbalised_confidence_sample_count={int(mode_confidence_counts.get(mode_name, 0))}"
        )
    lines.extend(
        [
            "",
            "[Run]",
            f"finished_at={finished_at}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def mode_to_output_key(mode: str) -> str:
    if mode == "none":
        return "no_replacement"
    if mode == "probability_tokens_mean_replace":
        return "probability_tokens_mean_replace"
    if mode == "all_pre_probability_tokens_mean_replace":
        return "all_pre_probability_tokens_mean_replace"
    if mode == "guess_tokens_mean_replace":
        return "guess_tokens_mean_replace"
    if mode == "all_pre_guess_tokens_mean_replace":
        return "all_pre_guess_tokens_mean_replace"
    if mode == "guess_then_guess_and_probability_tokens_mean_replace":
        return "guess_then_guess_and_probability_tokens_mean_replace"
    if mode == "post_guess_all_but_last_sem_answer_mean_replace":
        return "post_guess_all_but_last_sem_answer_mean_replace"
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def parse_mode_confidence_from_response(response: str) -> Optional[float]:
    parsed = parse_probability_from_response(response)
    if parsed is None:
        return None
    return float(parsed)


def parse_probability_from_response(response_str: str) -> float | None:
    """
    Extract probability in [0,1] from a response string.
    Uses the last occurrence of "probability:" and supports percentages.
    """
    if not response_str or not isinstance(response_str, str):
        return None
    matches = list(re.finditer(r"probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?", response_str, re.IGNORECASE))
    if not matches:
        matches = list(re.finditer(r"probability\s*:\s*(\d+(?:[.,]\d+)?)", response_str, re.IGNORECASE))
    if not matches:
        return None
    raw = matches[-1].group(1).strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value > 1:
        value = value / 100.0
    if value < 0 or value > 1:
        return None
    return value


def load_hooked_transformer(
    model_name: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> HookedTransformer:
    try:
        return HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            device=device,
            dtype=torch_dtype,
        )
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
        return HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            device=device,
            dtype=torch_dtype,
            tokenizer=slow_tokenizer,
        )


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


def _is_expected_or_plus_one(actual_len: int, expected_len: int) -> bool:
    return actual_len in (expected_len, expected_len + 1)


def compute_confidence_group_means(
    examples_h5: Dict[str, dict],
    ablate_layers: Sequence[int],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    mean_from_low_confidence: bool,
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

        emb_prob = resp0.get("embeddings_probability")
        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} responses/0/embeddings_probability must be a list.")
        if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability len={len(emb_prob)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        token_vectors: List[np.ndarray] = []
        for tok_arr in emb_prob[:-1]:
            layer_hidden = _as_layer_hidden(tok_arr)  # [n_layers, hidden_dim]
            selected = layer_hidden[np.asarray(ablate_layers), :]  # [num_selected_layers, hidden_dim]
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


def compute_pre_probability_group_means(
    examples_h5: Dict[str, dict],
    ablate_layers: Sequence[int],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
    mean_from_low_confidence: bool,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    """Build per-layer mean replacement vectors for all pre-probability regions.

    This scans examples, selects a confidence group (low or high), and averages
    embeddings across those examples for:
      - prompt mean token (`embeddings_mean_prompt`)
      - per-position Guess span tokens (`embeddings_guess`)
      - semantic-answer mean token (`embeddings_mean_sem_answer`)
      - Probability marker span tokens excluding the first value token
        (`embeddings_probability[:-1]`)

    Returns:
      - dict of mean tensors keyed by region (`prompt_mean`, `guess`,
        `sem_answer_mean`, `probability`) with layer dimension restricted to
        `ablate_layers`
      - low-confidence example IDs
      - high-confidence example IDs
    """
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    prompt_vectors: List[np.ndarray] = []
    sem_answer_vectors: List[np.ndarray] = []
    guess_vectors: List[np.ndarray] = []
    probability_vectors: List[np.ndarray] = []

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

        emb_prompt = resp0.get("embeddings_mean_prompt")
        emb_guess = resp0.get("embeddings_guess")
        emb_sem_answer = resp0.get("embeddings_mean_sem_answer")
        emb_prob = resp0.get("embeddings_probability")
        if emb_prompt is None or emb_guess is None or emb_sem_answer is None or emb_prob is None:
            raise ValueError(
                f"Example {ex_id} is missing one of required fields: "
                "embeddings_mean_prompt, embeddings_guess, embeddings_mean_sem_answer, embeddings_probability."
            )
        if not isinstance(emb_guess, list):
            raise ValueError(f"Example {ex_id} responses/0/embeddings_guess must be a list.")
        if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_guess len={len(emb_guess)}; "
                f"expected {expected_guess_tokens} or {expected_guess_tokens + 1}."
            )
        emb_guess = emb_guess[:expected_guess_tokens]
        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} responses/0/embeddings_probability must be a list.")
        if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability len={len(emb_prob)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        prompt_layer_hidden = _as_layer_hidden(emb_prompt)[np.asarray(ablate_layers), :]
        sem_answer_layer_hidden = _as_layer_hidden(emb_sem_answer)[np.asarray(ablate_layers), :]

        guess_selected: List[np.ndarray] = []
        for tok_arr in emb_guess:
            layer_hidden = _as_layer_hidden(tok_arr)[np.asarray(ablate_layers), :]
            guess_selected.append(layer_hidden)
        guess_stacked = np.stack(guess_selected, axis=1)

        prob_selected: List[np.ndarray] = []
        for tok_arr in emb_prob:
            layer_hidden = _as_layer_hidden(tok_arr)[np.asarray(ablate_layers), :]
            prob_selected.append(layer_hidden)
        prob_stacked = np.stack(prob_selected, axis=1)

        prompt_vectors.append(prompt_layer_hidden)
        sem_answer_vectors.append(sem_answer_layer_hidden)
        guess_vectors.append(guess_stacked)
        probability_vectors.append(prob_stacked)

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
    }
    return out_means, low_ids, high_ids


# ---------------------------------------------------------------------------
# Hook builder + generation
# ---------------------------------------------------------------------------


def _absolute_prob_positions(prompt_len: int, decoded_tokens: List[str]) -> List[int]:
    """Absolute indices for completion tokens ``first_prob`` … ``end_prob`` inclusive (H5 span)."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    # TODO: remove this after checking
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed

    seq_len = prompt_len + len(decoded_tokens)
    out: List[int] = []
    for k in range(first_prob, end_prob): # Do not include the first token of the prob value
        p = prompt_len + k
        if p < seq_len:
            out.append(p)
    return out


def _absolute_pre_probability_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Optional[Dict[str, List[int]]]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return None
    last_guess_token_index, first_prob_token_index, end_prob_token_index = parsed

    guess_positions_rel = list(range(0, last_guess_token_index))
    if not _is_expected_or_plus_one(len(guess_positions_rel), expected_guess_tokens):
        return None
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]

    probability_positions_rel = list(range(first_prob_token_index, end_prob_token_index))
    if not _is_expected_or_plus_one(len(probability_positions_rel), expected_probability_tokens):
        return None
    probability_positions_rel = probability_positions_rel[:expected_probability_tokens]

    prompt_positions_abs = list(range(0, prompt_len))
    guess_positions_abs = [prompt_len + k for k in guess_positions_rel]
    sem_answer_positions_abs = [prompt_len + k for k in range(last_guess_token_index, first_prob_token_index)]
    probability_positions_abs = [prompt_len + k for k in probability_positions_rel]
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
    if not _is_expected_or_plus_one(len(guess_positions_rel), expected_guess_tokens):
        return []
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]
    return [prompt_len + k for k in guess_positions_rel]


def _build_resid_post_mean_replace_hooks(
    layer_to_mean_vectors: Dict[int, torch.Tensor],
    *,
    seq_len_provider: Callable[[], int],
    abs_positions_provider: Callable[[], List[int]],
    strict_num_prob_positions: bool,
) -> List[Tuple[str, Callable]]:
    """
    Shared ``hook_resid_post`` factory: ``abs_positions_provider`` returns the same
    layout as ``_absolute_prob_positions`` (length ``num_prob_tokens`` when full).
    """
    num_prob_tokens = next(iter(layer_to_mean_vectors.values())).shape[0]
    hooks: List[Tuple[str, Callable]] = []

    for layer in layer_to_mean_vectors:
        hook_name = f"blocks.{layer}.hook_resid_post"

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


def _generation_contains_stop(decoded_completion: str) -> bool:
    return any(s in decoded_completion for s in STOP_SEQUENCES)


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
    generated: List[int] = []
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id

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

            completion_str = "".join(decoded_tokens)
            if _generation_contains_stop(completion_str):
                break
            if eos_id is not None and next_id == eos_id:
                break

    response = _postprocess_response_from_full_decode(model, tokens, local_prompt)
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
    eos_id = model.tokenizer.eos_token_id
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

            completion_str = "".join(decoded_tokens)
            if _generation_contains_stop(completion_str):
                break
            if eos_id is not None and next_id == eos_id:
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
        return int(tokens.shape[1])

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_mean_replace_hooks(
        layer_to_mean_vectors=layer_to_mean_vectors,
        prompt_len=prompt_len,
        seq_len_provider=_seq_len,
        decoded_tokens_provider=_decoded_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_pre_probability_mean_replace_hooks(
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    required_keys = {"prompt_mean", "guess", "sem_answer_mean", "probability"}

    for layer in layer_to_pre_probability_means:
        hook_name = f"blocks.{layer}.hook_resid_post"

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

                region_means = layer_to_pre_probability_means[layer_idx]
                if set(region_means.keys()) != required_keys:
                    raise ValueError(
                        f"Layer {layer_idx} pre-probability means missing keys. "
                        f"Got {sorted(region_means.keys())}, expected {sorted(required_keys)}."
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
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
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
        layer_to_pre_probability_means=layer_to_pre_probability_means,
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
        return int(tokens.shape[1])

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
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
) -> List[Tuple[str, Callable]]:
    """Replace resid_post at prompt with ``prompt_mean`` and at each ``Guess:`` span token with the matching ``guess`` row."""

    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_pre_probability_means:
        hook_name = f"blocks.{layer}.hook_resid_post"

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

                region = layer_to_pre_probability_means[layer_idx]
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

                for abs_pos in range(prompt_len):
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
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
    *,
    expected_guess_tokens: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens() -> List[str]:
        return decoded_tokens

    hooks = build_all_pre_guess_mean_replace_hooks(
        layer_to_pre_probability_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_guess_then_guess_and_probability_mean_replace_hooks(
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
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
    for layer in layer_to_pre_probability_means:
        hook_name = f"blocks.{layer}.hook_resid_post"

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

                region = layer_to_pre_probability_means[layer_idx]
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
                    if not _is_expected_or_plus_one(len(probability_positions), expected_probability_tokens):
                        raise ValueError(
                            f"Layer {layer_idx}: Probability position count {len(probability_positions)} is not "
                            f"expected {expected_probability_tokens} or {expected_probability_tokens + 1}."
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
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
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
        layer_to_pre_probability_means=layer_to_pre_probability_means,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    return _greedy_extend_with_fwd_hooks(model, local_prompt, max_new_tokens, tokens, decoded_tokens, hooks)


def build_all_but_last_position_mean_replace_hooks(
    layer_to_mean: Dict[int, torch.Tensor],
) -> List[Tuple[str, Callable]]:
    """Replace resid_post at indices ``0 .. seq_len-2`` with a per-layer mean vector; leave last position unchanged."""

    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_mean:
        hook_name = f"blocks.{layer}.hook_resid_post"

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                seq_len = int(activation.shape[1])
                if seq_len < 2:
                    return activation
                vec = layer_to_mean[layer_idx].to(activation.dtype)
                activation[:, : seq_len - 1, :] = vec
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def _use_post_guess_all_but_last_phase2(decoded_tokens: List[str]) -> bool:
    last_guess_token_index = parse_guess_start_index(decoded_tokens)
    if last_guess_token_index is None:
        return False
    return len(decoded_tokens) >= last_guess_token_index


def greedy_generate_post_guess_all_but_last_mean_replaced(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    layer_to_pre_probability_means: Dict[int, Dict[str, torch.Tensor]],
) -> Tuple[str, List[str]]:
    """
    Phase 1: greedy decode with no hooks until ``parse_guess_start_index`` succeeds and
    ``len(decoded_tokens) >= last_guess_token_index`` (Guess: prefix span complete).
    Phase 2: same greedy loop but with hooks that mean-replace resid_post at all positions except the last.
    Replacement per layer: ``sem_answer_mean`` from pre-probability group means.
    """
    layer_to_sem_answer: Dict[int, torch.Tensor] = {
        layer_idx: layer_to_pre_probability_means[layer_idx]["sem_answer_mean"]
        for layer_idx in layer_to_pre_probability_means
    }
    phase2_hooks = build_all_but_last_position_mean_replace_hooks(layer_to_sem_answer)

    tokens = model.to_tokens(local_prompt)
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id

    with torch.inference_mode():
        for _step in range(max_new_tokens):
            fwd_hooks = phase2_hooks if _use_post_guess_all_but_last_phase2(decoded_tokens) else []
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

            completion_str = "".join(decoded_tokens)
            if _generation_contains_stop(completion_str):
                break
            if eos_id is not None and next_id == eos_id:
                break

    response = _postprocess_response_from_full_decode(model, tokens, local_prompt)
    return response, decoded_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    TRAIN_RATIO = 0.9
    parser = argparse.ArgumentParser(description="Layerwise mean activation replacement inference (TransformerLens).")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
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
        help="Inclusive range '12-15' or comma list '12,13,14,15' (Zero-indexing!).",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=["none", "probability_tokens_mean_replace", "all_pre_probability_tokens_mean_replace", "guess_tokens_mean_replace", "all_pre_guess_tokens_mean_replace", "guess_then_guess_and_probability_tokens_mean_replace", "post_guess_all_but_last_sem_answer_mean_replace"],
        choices=[
            "none",
            "probability_tokens_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_and_probability_tokens_mean_replace",
            "post_guess_all_but_last_sem_answer_mean_replace",
        ],
        help=(
            "One or more modes to run. none: no hooks. probability_tokens_mean_replace: "
            "replace resid at H5 probability span. all_pre_probability_tokens_mean_replace: "
            "replace resid at prompt + Guess + semantic-answer + Probability marker positions. "
            "guess_tokens_mean_replace: replace resid only at the Guess: prefix span (per-position means). "
            "all_pre_guess_tokens_mean_replace: replace resid at every prompt position with prompt_mean, and at "
            "each Guess: prefix token with the corresponding row of guess (H5 per-position means). "
            "guess_then_guess_and_probability_tokens_mean_replace: greedy decode with dynamic span ablation; "
            "once Guess: is parseable, ablate the Guess: span, and once both Guess: and Probability: are parseable, "
            "ablate both spans. "
            "post_guess_all_but_last_sem_answer_mean_replace: greedy decode with no hooks until the Guess: "
            "prefix span is complete, then replace resid at all positions except the final timestep with "
            "sem_answer_mean (per forward); if the prefix never becomes parseable, stays unhooked."
        ),
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--mean_from_low_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), compute means from low-confidence examples and ablate high-confidence examples. "
            "If false, compute means from high-confidence examples and ablate low-confidence examples."
        ),
    )
    parser.add_argument("--expected_probability_tokens", type=int, default=6)
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
            "layerwise_mean_ablation/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    parser.add_argument(
        "--individual_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, ignore --ablate_layers and run a separate ablation for each layer. "
            "Outputs are written under results/individual_layers/<run_id>/<layer_idx>/."
        ),
    )
    args = parser.parse_args()
    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers)
    run_layers = list(range(model.cfg.n_layers)) if args.individual_layers else ablate_layers

    examples_h5 = load_examples_h5(Path(args.input_h5))
    modes = set(args.ablation_mode)
    need_pre_probability_means = (
        "probability_tokens_mean_replace" in modes
        or
        "all_pre_probability_tokens_mean_replace" in modes
        or "guess_tokens_mean_replace" in modes
        or "all_pre_guess_tokens_mean_replace" in modes
        or "guess_then_guess_and_probability_tokens_mean_replace" in modes
        or "post_guess_all_but_last_sem_answer_mean_replace" in modes
    )

    low_ids: set[str]
    high_ids: set[str]
    pre_probability_means: Optional[Dict[str, np.ndarray]] = None

    if need_pre_probability_means:
        pre_probability_means, low_ids, high_ids = compute_pre_probability_group_means(
            examples_h5,
            run_layers,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
            expected_probability_tokens=args.expected_probability_tokens,
            expected_guess_tokens=args.expected_guess_tokens,
            mean_from_low_confidence=args.mean_from_low_confidence,
        )
    else:
        low_ids, high_ids = collect_confidence_group_ids(
            examples_h5,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
        )

    logging.info("Low-confidence example IDs: %s", low_ids)
    logging.info("High-confidence example IDs: %s", high_ids)
    logging.info(
        "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), layers=%s, mean_from_low_confidence=%s, individual_layers=%s",
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        run_layers,
        args.mean_from_low_confidence,
        args.individual_layers,
    )

    mean_source_ids = low_ids if args.mean_from_low_confidence else high_ids
    ablation_target_ids = high_ids if args.mean_from_low_confidence else low_ids
    logging.info(
        "Mean source group=%s (%d ids), ablation target group=%s (%d ids).",
        "low_confidence" if args.mean_from_low_confidence else "high_confidence",
        len(mean_source_ids),
        "high_confidence" if args.mean_from_low_confidence else "low_confidence",
        len(ablation_target_ids),
    )

    layer_to_pre_probability_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    if need_pre_probability_means:
        if pre_probability_means is None:
            raise ValueError("Internal error: requested means but pre_probability_means was not computed.")
        layer_to_pre_probability_means = {}
        for i, layer_idx in enumerate(run_layers):
            layer_to_pre_probability_means[layer_idx] = {
                "prompt_mean": torch.tensor(pre_probability_means["prompt_mean"][i], device=device, dtype=torch_dtype),
                "guess": torch.tensor(pre_probability_means["guess"][i], device=device, dtype=torch_dtype),
                "sem_answer_mean": torch.tensor(pre_probability_means["sem_answer_mean"][i], device=device, dtype=torch_dtype),
                "probability": torch.tensor(pre_probability_means["probability"][i], device=device, dtype=torch_dtype),
            }

    def run_one_evaluation(
        # This contains {"prompt_mean", "guess", "sem_answer_mean", "probability"}
        layer_to_pre_probability_means_eval: Optional[Dict[int, Dict[str, torch.Tensor]]],
        cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    ) -> Tuple[
        Dict[str, dict],
        Dict[str, dict],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, Dict[str, Dict[str, object]]],
    ]:
        if layer_to_pre_probability_means_eval is not None:
            for layer_idx in layer_to_pre_probability_means_eval:
                logging.info(
                    "layer_to_pre_probability_means_eval[%s]['guess'].shape: %s",
                    layer_idx,
                    layer_to_pre_probability_means_eval[layer_idx]["guess"].shape,
                )
                logging.info(
                    "layer_to_pre_probability_means_eval[%s]['probability'].shape: %s",
                    layer_idx,
                    layer_to_pre_probability_means_eval[layer_idx]["probability"].shape,
                )
                logging.info(
                    "layer_to_pre_probability_means_eval[%s]['sem_answer_mean'].shape: %s",
                    layer_idx,
                    layer_to_pre_probability_means_eval[layer_idx]["sem_answer_mean"].shape,
                )
                logging.info(
                    "layer_to_pre_probability_means_eval[%s]['prompt_mean'].shape: %s",
                    layer_idx,
                    layer_to_pre_probability_means_eval[layer_idx]["prompt_mean"].shape,
                )
                break
        results = {"train": {}, "validation": {}}
        mini_results = {"train": {}, "validation": {}}
        mode_confidence_values: Dict[str, List[float]] = {mode_name: [] for mode_name in args.ablation_mode}

        modes = args.ablation_mode
        has_none_and_other_modes = ("none" in modes) and (len(modes) > 1)
        used_none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}

        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = (
                round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(args.num_samples * (1 - TRAIN_RATIO))
            )
            id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
            split_target_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)

            if split_target > 0 and not split_target_ids:
                logging.warning(
                    "No ablation target IDs available for %s split (mean_from_low_confidence=%s).",
                    split_name,
                    args.mean_from_low_confidence,
                )
                continue

            selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
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
                        if layer_to_pre_probability_means_eval is None:
                            raise ValueError("Mode probability_tokens_mean_replace requested but probability means are unavailable.")
                        layer_to_mean_vectors_eval = {
                            layer_idx: layer_to_pre_probability_means_eval[layer_idx]["probability"]
                            for layer_idx in layer_to_pre_probability_means_eval
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
                    elif mode == "all_pre_probability_tokens_mean_replace":
                        if layer_to_pre_probability_means_eval is None:
                            raise ValueError(
                                "Mode all_pre_probability_tokens_mean_replace requested but pre-probability means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_pre_probability_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_pre_probability_means=layer_to_pre_probability_means_eval,
                            expected_guess_tokens=args.expected_guess_tokens,
                            expected_probability_tokens=args.expected_probability_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "guess_tokens_mean_replace":
                        if layer_to_pre_probability_means_eval is None:
                            raise ValueError(
                                "Mode guess_tokens_mean_replace requested but pre-probability means are unavailable."
                            )
                        layer_to_guess_mean_vectors = {
                            layer_idx: layer_to_pre_probability_means_eval[layer_idx]["guess"]
                            for layer_idx in layer_to_pre_probability_means_eval
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
                        if layer_to_pre_probability_means_eval is None:
                            raise ValueError(
                                "Mode all_pre_guess_tokens_mean_replace requested but pre-probability means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_all_pre_guess_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_pre_probability_means=layer_to_pre_probability_means_eval,
                            expected_guess_tokens=args.expected_guess_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "guess_then_guess_and_probability_tokens_mean_replace":
                        if layer_to_pre_probability_means_eval is None:
                            raise ValueError(
                                "Mode guess_then_guess_and_probability_tokens_mean_replace requested but "
                                "pre-probability means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_guess_then_guess_and_probability_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_pre_probability_means=layer_to_pre_probability_means_eval,
                            expected_guess_tokens=args.expected_guess_tokens,
                            expected_probability_tokens=args.expected_probability_tokens,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                    elif mode == "post_guess_all_but_last_sem_answer_mean_replace":
                        if layer_to_pre_probability_means_eval is None:
                            raise ValueError(
                                "Mode post_guess_all_but_last_sem_answer_mean_replace requested but "
                                "pre-probability means are unavailable."
                            )
                        response, decoded_tokens = greedy_generate_post_guess_all_but_last_mean_replaced(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_pre_probability_means=layer_to_pre_probability_means_eval,
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

                        if args.parse_mode_verbalised_confidence:
                            mode_confidence = entry[mode_key].get("verbalised_confidence")
                            if mode_confidence is None or baseline_confidence is None:
                                meets_none_confidence_direction = None
                            elif args.mean_from_low_confidence:
                                meets_none_confidence_direction = mode_confidence < baseline_confidence
                            else:
                                meets_none_confidence_direction = mode_confidence > baseline_confidence
                            entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                            mini_entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction

                results[split_name][ex_id] = entry
                mini_results[split_name][ex_id] = mini_entry

        mode_confidence_means: Dict[str, Optional[float]] = {}
        mode_confidence_counts: Dict[str, int] = {}
        for mode_name in modes:
            values = mode_confidence_values[mode_name]
            mode_confidence_means[mode_name] = float(np.mean(values)) if values else None
            mode_confidence_counts[mode_name] = len(values)
        return results, mini_results, mode_confidence_means, mode_confidence_counts, used_none_cache

    # Compute the output for baseline none mode (no ablation) so individual-layer runs can reuse the same baseline.
    def build_none_cache() -> Dict[str, Dict[str, Dict[str, object]]]:
        none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}
        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = (
                round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(args.num_samples * (1 - TRAIN_RATIO))
            )
            id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
            split_target_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)
            selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
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

    if not args.individual_layers:
        results, mini_results, mode_confidence_means, mode_confidence_counts, _ = run_one_evaluation(
            layer_to_pre_probability_means,
            cached_none=None,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", out_path)

        mini_out_path = mini_output_json_path(out_path)
        with open(mini_out_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", mini_out_path)

        config_out_path = config_txt_path(out_path)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
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
            mean_from_low_confidence=args.mean_from_low_confidence,
            parse_mode_verbalised_confidence=args.parse_mode_verbalised_confidence,
            mode_confidence_means=mode_confidence_means,
            mode_confidence_counts=mode_confidence_counts,
            finished_at=finished_at,
        )
        logging.info("Wrote %s", config_out_path)
        return

    run_root_norm = run_root.rstrip(os.sep)
    run_id = os.path.basename(run_root_norm)
    results_root = os.path.dirname(run_root_norm)
    individual_root = os.path.join(results_root, "individual_layers", run_id)
    os.makedirs(individual_root, exist_ok=True)

    cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None
    if "none" in args.ablation_mode:
        logging.info("Computing baseline none-mode once for individual layer sweep.")
        cached_none = build_none_cache()

    per_layer_mode_means: Dict[int, Dict[str, Optional[float]]] = {}
    for layer_idx in run_layers:
        logging.info("Running individual-layer ablation for layer %d", layer_idx)
        layer_dir = os.path.join(individual_root, str(layer_idx))
        os.makedirs(layer_dir, exist_ok=True)
        layer_pre_map = (
            None if layer_to_pre_probability_means is None else {layer_idx: layer_to_pre_probability_means[layer_idx]}
        )
        results, mini_results, mode_confidence_means, mode_confidence_counts, _ = run_one_evaluation(
            layer_pre_map,
            cached_none=cached_none,
        )
        per_layer_mode_means[layer_idx] = mode_confidence_means

        layer_out_path = os.path.join(layer_dir, "ablation_results.json")
        with open(layer_out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", layer_out_path)

        layer_mini_path = os.path.join(layer_dir, "ablation_results_mini.json")
        with open(layer_mini_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", layer_mini_path)

        layer_config_path = os.path.join(layer_dir, "config.txt")
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        write_config_txt(
            layer_config_path,
            args=args,
            device=device,
            model_n_layers=model.cfg.n_layers,
            ablate_layers=[layer_idx],
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            mean_from_low_confidence=args.mean_from_low_confidence,
            parse_mode_verbalised_confidence=args.parse_mode_verbalised_confidence,
            mode_confidence_means=mode_confidence_means,
            mode_confidence_counts=mode_confidence_counts,
            finished_at=finished_at,
        )
        logging.info("Wrote %s", layer_config_path)

    summary_path = os.path.join(individual_root, "summary.txt")
    modes_non_none = [mode for mode in args.ablation_mode if mode != "none"]
    baseline_none_mean = None
    if "none" in args.ablation_mode and per_layer_mode_means:
        first_layer = run_layers[0]
        baseline_none_mean = per_layer_mode_means[first_layer].get("none")

    summary_lines = [
        "Individual Layer Mean Ablation Summary",
        "=====================================",
        "",
        "[Setup]",
        f"model_name={args.model_name}",
        f"input_h5={args.input_h5}",
        f"ablation_mode={args.ablation_mode}",
        f"num_layers={model.cfg.n_layers}",
        f"run_layers={','.join(str(layer) for layer in run_layers)}",
        f"num_samples={args.num_samples}",
        f"num_few_shot={args.num_few_shot}",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={'low_confidence' if args.mean_from_low_confidence else 'high_confidence'}",
        f"ablation_target_group={'high_confidence' if args.mean_from_low_confidence else 'low_confidence'}",
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


if __name__ == "__main__":
    main()
