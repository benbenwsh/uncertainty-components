import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_zero_ablation.run_blockwise_zero_ablation import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT,
    _absolute_all_pre_guess_positions,
    _absolute_guess_span_positions,
    _absolute_guess_then_guess_probability_positions,
    _absolute_pre_probability_positions,
    _absolute_prob_except_last_token_positions,
    _absolute_prob_last_token_only_positions,
    _absolute_prob_positions,
    collect_confidence_group_ids,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    greedy_generate,
    load_examples_h5,
    load_hooked_transformer,
    load_trivia_qa,
    mode_to_output_key,
    parse_ablate_layers,
    parse_mode_confidence_from_response,
    split_answerable_indices,
)
from layerwise_mean_ablation.run_mean_ablation import (
    _absolute_probability_value_start_position,
    _as_layer_hidden,
    _is_expected_or_plus_one,
)


TRAIN_RATIO = 0.9
REQUIRED_EMBEDDING_FIELDS = (
    "embeddings_mean_prompt",
    "embeddings_guess",
    "embeddings_mean_sem_answer",
    "embeddings_probability",
    "embeddings_mean_prob_val",
)
ABLATION_MODES_DEFAULT = [
    "none",
    "probability_tokens_mean_replace",
    "probability_last_token_mean_replace",
    "probability_span_except_last_token_mean_replace",
    "all_pre_probability_tokens_mean_replace",
    "guess_tokens_mean_replace",
    "all_pre_guess_tokens_mean_replace",
    "guess_then_guess_probability_mean_replace",
    "probability_value_mean_replace",
    "current_generated_token_mean_replace",
]
ABLATION_UNIT_MODES = ["head", "grouped_head", "whole_concat"]


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("headwise_mean_ablation") / "results"
        base.mkdir(parents=True, exist_ok=True)
        run_id = 1
        while (base / str(run_id)).exists():
            run_id += 1
        run_dir = base / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(run_dir / "ablation_results_mini.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def config_txt_path(mini_output_path: str) -> str:
    return os.path.join(os.path.dirname(mini_output_path), "config.txt")


def parse_head_indices(spec: Optional[str], n_heads: int) -> List[int]:
    if spec is None or spec.strip().lower() == "all":
        return list(range(n_heads))
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        heads = list(range(int(a.strip()), int(b.strip()) + 1))
    else:
        heads = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for head_idx in heads:
        if head_idx < 0 or head_idx >= n_heads:
            raise ValueError(f"Head index {head_idx} out of range [0, {n_heads}).")
    return heads


def _is_all_heads_spec(spec: Optional[str]) -> bool:
    return spec is None or spec.strip().lower() == "all"


def resolve_n_key_value_heads(model_cfg, n_heads: int) -> int:
    for attr in (
        "n_key_value_heads",
        "num_key_value_heads",
        "n_kv_heads",
        "num_kv_heads",
        "n_heads_kv",
    ):
        value = getattr(model_cfg, attr, None)
        if value is not None:
            n_kv_heads = int(value)
            if n_kv_heads <= 0:
                raise ValueError(f"Invalid {attr}={n_kv_heads}. Must be > 0.")
            if n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"Invalid GQA head configuration: n_heads={n_heads} is not divisible by {attr}={n_kv_heads}."
                )
            return n_kv_heads
    return int(n_heads)


def build_grouped_head_units(*, n_heads: int, n_kv_heads: int) -> Dict[str, List[int]]:
    if n_kv_heads <= 0:
        raise ValueError(f"n_kv_heads must be > 0, got {n_kv_heads}.")
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"Cannot build grouped heads: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}."
        )
    n_rep = n_heads // n_kv_heads
    grouped_units: Dict[str, List[int]] = {}
    for kv_idx in range(n_kv_heads):
        start = kv_idx * n_rep
        grouped_units[f"group_{kv_idx}"] = list(range(start, start + n_rep))
    return grouped_units


def _validate_concat_field(resp0: dict, ex_id: str, field_name: str):
    field = resp0.get(field_name)
    if not isinstance(field, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be an object with key 'concat'."
        )
    if "concat" not in field:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} missing key 'concat'. "
            "Please regenerate processed H5 with --collect_concat_embeddings."
        )
    value = field.get("concat")
    if value is None:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name}/concat is null. "
            "Please regenerate processed H5 with --collect_concat_embeddings."
        )
    return value


def _validate_component_field(resp0: dict, ex_id: str, field_name: str, component: str):
    field = resp0.get(field_name)
    if not isinstance(field, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be an object with key '{component}'."
        )
    if component not in field:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} missing key '{component}'. "
            "Please regenerate processed H5 with --collect_qkvo_embeddings."
        )
    value = field.get(component)
    if value is None:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name}/{component} is null. "
            "Please regenerate processed H5 with --collect_qkvo_embeddings."
        )
    return value


def _reshape_layer_hidden_to_heads(arr_like: np.ndarray, *, n_heads: int, d_head: int) -> np.ndarray:
    arr = _as_layer_hidden(arr_like)
    if arr.ndim != 2:
        raise ValueError(f"Expected [layers, d_model] after _as_layer_hidden, got shape {arr.shape}.")
    if arr.shape[-1] != n_heads * d_head:
        raise ValueError(
            f"Hidden dim {arr.shape[-1]} does not match n_heads*d_head={n_heads*d_head}."
        )
    return arr.reshape(arr.shape[0], n_heads, d_head)


def _reshape_layer_hidden_to_kv_heads(arr_like: np.ndarray, *, n_kv_heads: int, d_head: int) -> np.ndarray:
    arr = _as_layer_hidden(arr_like)
    if arr.ndim != 2:
        raise ValueError(f"Expected [layers, d_kv] after _as_layer_hidden, got shape {arr.shape}.")
    if arr.shape[-1] != n_kv_heads * d_head:
        raise ValueError(
            f"Hidden dim {arr.shape[-1]} does not match n_kv_heads*d_head={n_kv_heads*d_head}."
        )
    return arr.reshape(arr.shape[0], n_kv_heads, d_head)


def compute_concat_headwise_means(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    n_heads: int,
    d_head: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()

    prompt_vectors: List[np.ndarray] = []
    sem_answer_vectors: List[np.ndarray] = []
    guess_vectors: List[np.ndarray] = []
    probability_vectors: List[np.ndarray] = []
    probability_value_mean_vectors: List[np.ndarray] = []
    layer_idx = np.asarray(ablate_layers)

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

        for field_name in REQUIRED_EMBEDDING_FIELDS:
            _validate_concat_field(resp0, ex_id, field_name)

        prob = resp0.get("verbalised_confidence")
        if prob is None:
            raise ValueError(f"Example {ex_id} responses/0/verbalised_confidence is missing.")
        prob = float(prob)
        is_low = prob <= low_conf_threshold
        is_high = prob >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
            continue

        emb_prompt = _validate_concat_field(resp0, ex_id, "embeddings_mean_prompt")
        emb_guess = _validate_concat_field(resp0, ex_id, "embeddings_guess")
        emb_sem_answer = _validate_concat_field(resp0, ex_id, "embeddings_mean_sem_answer")
        emb_prob = _validate_concat_field(resp0, ex_id, "embeddings_probability")
        emb_mean_prob_val = _validate_concat_field(resp0, ex_id, "embeddings_mean_prob_val")

        if not isinstance(emb_guess, list):
            raise ValueError(f"Example {ex_id} embeddings_guess/concat must be a list.")
        if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_guess/concat len={len(emb_guess)}; expected "
                f"{expected_guess_tokens} or {expected_guess_tokens + 1}."
            )
        emb_guess = emb_guess[:expected_guess_tokens]

        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} embeddings_probability/concat must be a list.")
        if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/concat len={len(emb_prob)}; expected "
                f"{expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        prompt_heads = _reshape_layer_hidden_to_heads(
            emb_prompt, n_heads=n_heads, d_head=d_head
        )[layer_idx, :, :]
        sem_answer_heads = _reshape_layer_hidden_to_heads(
            emb_sem_answer, n_heads=n_heads, d_head=d_head
        )[layer_idx, :, :]
        mean_prob_val_heads = _reshape_layer_hidden_to_heads(
            emb_mean_prob_val, n_heads=n_heads, d_head=d_head
        )[layer_idx, :, :]

        guess_selected: List[np.ndarray] = []
        for tok_arr in emb_guess:
            guess_selected.append(
                _reshape_layer_hidden_to_heads(tok_arr, n_heads=n_heads, d_head=d_head)[layer_idx, :, :]
            )
        guess_stacked = np.stack(guess_selected, axis=1)  # [L, T_guess, H, D]

        prob_selected: List[np.ndarray] = []
        for tok_arr in emb_prob:
            prob_selected.append(
                _reshape_layer_hidden_to_heads(tok_arr, n_heads=n_heads, d_head=d_head)[layer_idx, :, :]
            )
        prob_stacked = np.stack(prob_selected, axis=1)  # [L, T_prob, H, D]

        prompt_vectors.append(prompt_heads)
        sem_answer_vectors.append(sem_answer_heads)
        guess_vectors.append(guess_stacked)
        probability_vectors.append(prob_stacked)
        probability_value_mean_vectors.append(mean_prob_val_heads)

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if not prompt_vectors:
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(
            f"No {source_name} examples usable for concat means at threshold {operator} {threshold}."
        )

    means = {
        "prompt_mean": np.mean(np.stack(prompt_vectors, axis=0), axis=0).astype(np.float32),
        "guess": np.mean(np.stack(guess_vectors, axis=0), axis=0).astype(np.float32),
        "sem_answer_mean": np.mean(np.stack(sem_answer_vectors, axis=0), axis=0).astype(np.float32),
        "probability": np.mean(np.stack(probability_vectors, axis=0), axis=0).astype(np.float32),
        "probability_value_mean": np.mean(
            np.stack(probability_value_mean_vectors, axis=0), axis=0
        ).astype(np.float32),
    }
    return means, low_ids, high_ids


def compute_kv_headwise_means(
    examples_h5: Dict[str, dict],
    *,
    component: str,
    ablate_layers: Sequence[int],
    n_kv_heads: int,
    d_head: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    if component not in ("k", "v"):
        raise ValueError(f"Unsupported KV component {component!r}; expected 'k' or 'v'.")
    low_ids: set[str] = set()
    high_ids: set[str] = set()

    prompt_vectors: List[np.ndarray] = []
    sem_answer_vectors: List[np.ndarray] = []
    guess_vectors: List[np.ndarray] = []
    probability_vectors: List[np.ndarray] = []
    probability_value_mean_vectors: List[np.ndarray] = []
    layer_idx = np.asarray(ablate_layers)

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

        for field_name in REQUIRED_EMBEDDING_FIELDS:
            _validate_component_field(resp0, ex_id, field_name, component)

        prob = resp0.get("verbalised_confidence")
        if prob is None:
            raise ValueError(f"Example {ex_id} responses/0/verbalised_confidence is missing.")
        prob = float(prob)
        is_low = prob <= low_conf_threshold
        is_high = prob >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
            continue

        emb_prompt = _validate_component_field(resp0, ex_id, "embeddings_mean_prompt", component)
        emb_guess = _validate_component_field(resp0, ex_id, "embeddings_guess", component)
        emb_sem_answer = _validate_component_field(resp0, ex_id, "embeddings_mean_sem_answer", component)
        emb_prob = _validate_component_field(resp0, ex_id, "embeddings_probability", component)
        emb_mean_prob_val = _validate_component_field(resp0, ex_id, "embeddings_mean_prob_val", component)

        if not isinstance(emb_guess, list):
            raise ValueError(f"Example {ex_id} embeddings_guess/{component} must be a list.")
        if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_guess/{component} len={len(emb_guess)}; expected "
                f"{expected_guess_tokens} or {expected_guess_tokens + 1}."
            )
        emb_guess = emb_guess[:expected_guess_tokens]

        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} embeddings_probability/{component} must be a list.")
        if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/{component} len={len(emb_prob)}; expected "
                f"{expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        prompt_heads = _reshape_layer_hidden_to_kv_heads(
            emb_prompt, n_kv_heads=n_kv_heads, d_head=d_head
        )[layer_idx, :, :]
        sem_answer_heads = _reshape_layer_hidden_to_kv_heads(
            emb_sem_answer, n_kv_heads=n_kv_heads, d_head=d_head
        )[layer_idx, :, :]
        mean_prob_val_heads = _reshape_layer_hidden_to_kv_heads(
            emb_mean_prob_val, n_kv_heads=n_kv_heads, d_head=d_head
        )[layer_idx, :, :]

        guess_selected: List[np.ndarray] = []
        for tok_arr in emb_guess:
            guess_selected.append(
                _reshape_layer_hidden_to_kv_heads(tok_arr, n_kv_heads=n_kv_heads, d_head=d_head)[
                    layer_idx, :, :
                ]
            )
        guess_stacked = np.stack(guess_selected, axis=1)  # [L, T_guess, H_kv, D]

        prob_selected: List[np.ndarray] = []
        for tok_arr in emb_prob:
            prob_selected.append(
                _reshape_layer_hidden_to_kv_heads(tok_arr, n_kv_heads=n_kv_heads, d_head=d_head)[
                    layer_idx, :, :
                ]
            )
        prob_stacked = np.stack(prob_selected, axis=1)  # [L, T_prob, H_kv, D]

        prompt_vectors.append(prompt_heads)
        sem_answer_vectors.append(sem_answer_heads)
        guess_vectors.append(guess_stacked)
        probability_vectors.append(prob_stacked)
        probability_value_mean_vectors.append(mean_prob_val_heads)

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if not prompt_vectors:
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(
            f"No {source_name} examples usable for KV ({component}) means at threshold {operator} {threshold}."
        )

    means = {
        "prompt_mean": np.mean(np.stack(prompt_vectors, axis=0), axis=0).astype(np.float32),
        "guess": np.mean(np.stack(guess_vectors, axis=0), axis=0).astype(np.float32),
        "sem_answer_mean": np.mean(np.stack(sem_answer_vectors, axis=0), axis=0).astype(np.float32),
        "probability": np.mean(np.stack(probability_vectors, axis=0), axis=0).astype(np.float32),
        "probability_value_mean": np.mean(
            np.stack(probability_value_mean_vectors, axis=0), axis=0
        ).astype(np.float32),
    }
    return means, low_ids, high_ids


def compute_concat_whole_means(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()

    prompt_vectors: List[np.ndarray] = []
    sem_answer_vectors: List[np.ndarray] = []
    guess_vectors: List[np.ndarray] = []
    probability_vectors: List[np.ndarray] = []
    probability_value_mean_vectors: List[np.ndarray] = []
    layer_idx = np.asarray(ablate_layers)

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

        for field_name in REQUIRED_EMBEDDING_FIELDS:
            _validate_concat_field(resp0, ex_id, field_name)

        prob = resp0.get("verbalised_confidence")
        if prob is None:
            raise ValueError(f"Example {ex_id} responses/0/verbalised_confidence is missing.")
        prob = float(prob)
        is_low = prob <= low_conf_threshold
        is_high = prob >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
            continue

        emb_prompt = _validate_concat_field(resp0, ex_id, "embeddings_mean_prompt")
        emb_guess = _validate_concat_field(resp0, ex_id, "embeddings_guess")
        emb_sem_answer = _validate_concat_field(resp0, ex_id, "embeddings_mean_sem_answer")
        emb_prob = _validate_concat_field(resp0, ex_id, "embeddings_probability")
        emb_mean_prob_val = _validate_concat_field(resp0, ex_id, "embeddings_mean_prob_val")

        if not isinstance(emb_guess, list):
            raise ValueError(f"Example {ex_id} embeddings_guess/concat must be a list.")
        if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_guess/concat len={len(emb_guess)}; expected "
                f"{expected_guess_tokens} or {expected_guess_tokens + 1}."
            )
        emb_guess = emb_guess[:expected_guess_tokens]

        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} embeddings_probability/concat must be a list.")
        if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/concat len={len(emb_prob)}; expected "
                f"{expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        prompt_hidden = _as_layer_hidden(emb_prompt)[layer_idx, :]
        sem_answer_hidden = _as_layer_hidden(emb_sem_answer)[layer_idx, :]
        mean_prob_val_hidden = _as_layer_hidden(emb_mean_prob_val)[layer_idx, :]

        guess_selected: List[np.ndarray] = []
        for tok_arr in emb_guess:
            guess_selected.append(_as_layer_hidden(tok_arr)[layer_idx, :])
        guess_stacked = np.stack(guess_selected, axis=1)  # [L, T_guess, d_model]

        prob_selected: List[np.ndarray] = []
        for tok_arr in emb_prob:
            prob_selected.append(_as_layer_hidden(tok_arr)[layer_idx, :])
        prob_stacked = np.stack(prob_selected, axis=1)  # [L, T_prob, d_model]

        prompt_vectors.append(prompt_hidden)
        sem_answer_vectors.append(sem_answer_hidden)
        guess_vectors.append(guess_stacked)
        probability_vectors.append(prob_stacked)
        probability_value_mean_vectors.append(mean_prob_val_hidden)

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if not prompt_vectors:
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(
            f"No {source_name} examples usable for concat means at threshold {operator} {threshold}."
        )

    means = {
        "prompt_mean": np.mean(np.stack(prompt_vectors, axis=0), axis=0).astype(np.float32),
        "guess": np.mean(np.stack(guess_vectors, axis=0), axis=0).astype(np.float32),
        "sem_answer_mean": np.mean(np.stack(sem_answer_vectors, axis=0), axis=0).astype(np.float32),
        "probability": np.mean(np.stack(probability_vectors, axis=0), axis=0).astype(np.float32),
        "probability_value_mean": np.mean(
            np.stack(probability_value_mean_vectors, axis=0), axis=0
        ).astype(np.float32),
    }
    return means, low_ids, high_ids


def compute_kv_whole_means(
    examples_h5: Dict[str, dict],
    *,
    component: str,
    ablate_layers: Sequence[int],
    n_kv_heads: int,
    d_head: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    if component not in ("k", "v"):
        raise ValueError(f"Unsupported KV component {component!r}; expected 'k' or 'v'.")
    low_ids: set[str] = set()
    high_ids: set[str] = set()

    prompt_vectors: List[np.ndarray] = []
    sem_answer_vectors: List[np.ndarray] = []
    guess_vectors: List[np.ndarray] = []
    probability_vectors: List[np.ndarray] = []
    probability_value_mean_vectors: List[np.ndarray] = []
    layer_idx = np.asarray(ablate_layers)
    expected_d_kv = n_kv_heads * d_head

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

        for field_name in REQUIRED_EMBEDDING_FIELDS:
            _validate_component_field(resp0, ex_id, field_name, component)

        prob = resp0.get("verbalised_confidence")
        if prob is None:
            raise ValueError(f"Example {ex_id} responses/0/verbalised_confidence is missing.")
        prob = float(prob)
        is_low = prob <= low_conf_threshold
        is_high = prob >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
            continue

        emb_prompt = _validate_component_field(resp0, ex_id, "embeddings_mean_prompt", component)
        emb_guess = _validate_component_field(resp0, ex_id, "embeddings_guess", component)
        emb_sem_answer = _validate_component_field(resp0, ex_id, "embeddings_mean_sem_answer", component)
        emb_prob = _validate_component_field(resp0, ex_id, "embeddings_probability", component)
        emb_mean_prob_val = _validate_component_field(resp0, ex_id, "embeddings_mean_prob_val", component)

        if not isinstance(emb_guess, list):
            raise ValueError(f"Example {ex_id} embeddings_guess/{component} must be a list.")
        if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_guess/{component} len={len(emb_guess)}; expected "
                f"{expected_guess_tokens} or {expected_guess_tokens + 1}."
            )
        emb_guess = emb_guess[:expected_guess_tokens]

        if not isinstance(emb_prob, list):
            raise ValueError(f"Example {ex_id} embeddings_probability/{component} must be a list.")
        if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/{component} len={len(emb_prob)}; expected "
                f"{expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob = emb_prob[:expected_probability_tokens]

        prompt_hidden = _as_layer_hidden(emb_prompt)[layer_idx, :]
        sem_answer_hidden = _as_layer_hidden(emb_sem_answer)[layer_idx, :]
        mean_prob_val_hidden = _as_layer_hidden(emb_mean_prob_val)[layer_idx, :]
        if prompt_hidden.shape[-1] != expected_d_kv:
            raise ValueError(
                f"KV ({component}) hidden dim {prompt_hidden.shape[-1]} does not match n_kv_heads*d_head={expected_d_kv}."
            )

        guess_selected: List[np.ndarray] = []
        for tok_arr in emb_guess:
            tok_hidden = _as_layer_hidden(tok_arr)[layer_idx, :]
            if tok_hidden.shape[-1] != expected_d_kv:
                raise ValueError(
                    f"KV ({component}) hidden dim {tok_hidden.shape[-1]} does not match n_kv_heads*d_head={expected_d_kv}."
                )
            guess_selected.append(tok_hidden)
        guess_stacked = np.stack(guess_selected, axis=1)  # [L, T_guess, d_kv]

        prob_selected: List[np.ndarray] = []
        for tok_arr in emb_prob:
            tok_hidden = _as_layer_hidden(tok_arr)[layer_idx, :]
            if tok_hidden.shape[-1] != expected_d_kv:
                raise ValueError(
                    f"KV ({component}) hidden dim {tok_hidden.shape[-1]} does not match n_kv_heads*d_head={expected_d_kv}."
                )
            prob_selected.append(tok_hidden)
        prob_stacked = np.stack(prob_selected, axis=1)  # [L, T_prob, d_kv]

        prompt_vectors.append(prompt_hidden)
        sem_answer_vectors.append(sem_answer_hidden)
        guess_vectors.append(guess_stacked)
        probability_vectors.append(prob_stacked)
        probability_value_mean_vectors.append(mean_prob_val_hidden)

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if not prompt_vectors:
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(
            f"No {source_name} examples usable for KV ({component}) means at threshold {operator} {threshold}."
        )

    means = {
        "prompt_mean": np.mean(np.stack(prompt_vectors, axis=0), axis=0).astype(np.float32),
        "guess": np.mean(np.stack(guess_vectors, axis=0), axis=0).astype(np.float32),
        "sem_answer_mean": np.mean(np.stack(sem_answer_vectors, axis=0), axis=0).astype(np.float32),
        "probability": np.mean(np.stack(probability_vectors, axis=0), axis=0).astype(np.float32),
        "probability_value_mean": np.mean(
            np.stack(probability_value_mean_vectors, axis=0), axis=0
        ).astype(np.float32),
    }
    return means, low_ids, high_ids


def _build_layer_head_means(
    means: Dict[str, np.ndarray],
    *,
    ablate_layers: Sequence[int],
    head_indices: Sequence[int],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    out: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    for layer_i, layer in enumerate(ablate_layers):
        out[int(layer)] = {}
        for head_idx in head_indices:
            out[int(layer)][int(head_idx)] = {
                "prompt_mean": torch.tensor(
                    means["prompt_mean"][layer_i, head_idx], device=device, dtype=torch_dtype
                ),
                "guess": torch.tensor(means["guess"][layer_i, :, head_idx], device=device, dtype=torch_dtype),
                "sem_answer_mean": torch.tensor(
                    means["sem_answer_mean"][layer_i, head_idx], device=device, dtype=torch_dtype
                ),
                "probability": torch.tensor(
                    means["probability"][layer_i, :, head_idx], device=device, dtype=torch_dtype
                ),
                "probability_value_mean": torch.tensor(
                    means["probability_value_mean"][layer_i, head_idx], device=device, dtype=torch_dtype
                ),
            }
    return out


def _build_layer_kv_head_means(
    means: Dict[str, np.ndarray],
    *,
    ablate_layers: Sequence[int],
    kv_head_indices: Sequence[int],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    out: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    for layer_i, layer in enumerate(ablate_layers):
        out[int(layer)] = {}
        for kv_head_idx in kv_head_indices:
            out[int(layer)][int(kv_head_idx)] = {
                "prompt_mean": torch.tensor(
                    means["prompt_mean"][layer_i, kv_head_idx], device=device, dtype=torch_dtype
                ),
                "guess": torch.tensor(means["guess"][layer_i, :, kv_head_idx], device=device, dtype=torch_dtype),
                "sem_answer_mean": torch.tensor(
                    means["sem_answer_mean"][layer_i, kv_head_idx], device=device, dtype=torch_dtype
                ),
                "probability": torch.tensor(
                    means["probability"][layer_i, :, kv_head_idx], device=device, dtype=torch_dtype
                ),
                "probability_value_mean": torch.tensor(
                    means["probability_value_mean"][layer_i, kv_head_idx], device=device, dtype=torch_dtype
                ),
            }
    return out


def _build_layer_concat_means(
    means: Dict[str, np.ndarray],
    *,
    ablate_layers: Sequence[int],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_i, layer in enumerate(ablate_layers):
        out[int(layer)] = {
            "prompt_mean": torch.tensor(means["prompt_mean"][layer_i], device=device, dtype=torch_dtype),
            "guess": torch.tensor(means["guess"][layer_i], device=device, dtype=torch_dtype),
            "sem_answer_mean": torch.tensor(
                means["sem_answer_mean"][layer_i], device=device, dtype=torch_dtype
            ),
            "probability": torch.tensor(means["probability"][layer_i], device=device, dtype=torch_dtype),
            "probability_value_mean": torch.tensor(
                means["probability_value_mean"][layer_i], device=device, dtype=torch_dtype
            ),
        }
    return out


def _build_layer_kv_concat_means(
    means: Dict[str, np.ndarray],
    *,
    ablate_layers: Sequence[int],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_i, layer in enumerate(ablate_layers):
        out[int(layer)] = {
            "prompt_mean": torch.tensor(means["prompt_mean"][layer_i], device=device, dtype=torch_dtype),
            "guess": torch.tensor(means["guess"][layer_i], device=device, dtype=torch_dtype),
            "sem_answer_mean": torch.tensor(
                means["sem_answer_mean"][layer_i], device=device, dtype=torch_dtype
            ),
            "probability": torch.tensor(means["probability"][layer_i], device=device, dtype=torch_dtype),
            "probability_value_mean": torch.tensor(
                means["probability_value_mean"][layer_i], device=device, dtype=torch_dtype
            ),
        }
    return out


def _positions_and_replacements_for_mode(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_head_means: Dict[str, torch.Tensor],
) -> Tuple[List[int], List[torch.Tensor]]:
    prompt_mean = layer_head_means["prompt_mean"]
    guess = layer_head_means["guess"]
    sem_answer_mean = layer_head_means["sem_answer_mean"]
    probability = layer_head_means["probability"]
    probability_value_mean = layer_head_means["probability_value_mean"]

    if mode == "probability_tokens_mean_replace":
        positions = _absolute_prob_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=int(probability.shape[0])
        )
        vectors = [probability[i] for i in range(min(len(positions), int(probability.shape[0])))]
        return positions[: len(vectors)], vectors

    if mode == "probability_last_token_mean_replace":
        positions = _absolute_prob_last_token_only_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=int(probability.shape[0])
        )
        if not positions:
            return [], []
        return [positions[0]], [probability[-1]]

    if mode == "probability_span_except_last_token_mean_replace":
        positions = _absolute_prob_except_last_token_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=int(probability.shape[0])
        )
        n = min(len(positions), max(0, int(probability.shape[0]) - 1))
        return positions[:n], [probability[i] for i in range(n)]

    if mode == "guess_tokens_mean_replace":
        positions = _absolute_guess_span_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=int(guess.shape[0])
        )
        n = min(len(positions), int(guess.shape[0]))
        return positions[:n], [guess[i] for i in range(n)]

    if mode == "all_pre_guess_tokens_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=int(guess.shape[0])
        )
        prompt_positions = _absolute_all_pre_guess_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=int(guess.shape[0])
        )
        prompt_count = max(0, len(prompt_positions) - len(guess_positions))
        positions: List[int] = []
        vectors: List[torch.Tensor] = []
        for p in prompt_positions[:prompt_count]:
            positions.append(p)
            vectors.append(prompt_mean)
        n_guess = min(len(guess_positions), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(guess_positions[i])
            vectors.append(guess[i])
        return positions, vectors

    if mode == "all_pre_probability_tokens_mean_replace":
        spans = _absolute_pre_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
            expected_probability_tokens=int(probability.shape[0]),
        )
        if spans is None:
            return [], []
        positions = []
        vectors = []
        for p in spans["prompt"]:
            positions.append(p)
            vectors.append(prompt_mean)
        n_guess = min(len(spans["guess"]), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(spans["guess"][i])
            vectors.append(guess[i])
        for p in spans["sem_answer"]:
            positions.append(p)
            vectors.append(sem_answer_mean)
        n_prob = min(len(spans["probability"]), int(probability.shape[0]))
        for i in range(n_prob):
            positions.append(spans["probability"][i])
            vectors.append(probability[i])
        return positions, vectors

    if mode == "guess_then_guess_probability_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=int(guess.shape[0])
        )
        all_positions = _absolute_guess_then_guess_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
            expected_probability_tokens=int(probability.shape[0]),
        )
        prob_positions = all_positions[len(guess_positions) :]
        positions = []
        vectors = []
        n_guess = min(len(guess_positions), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(guess_positions[i])
            vectors.append(guess[i])
        n_prob = min(len(prob_positions), int(probability.shape[0]))
        for i in range(n_prob):
            positions.append(prob_positions[i])
            vectors.append(probability[i])
        return positions, vectors

    if mode == "probability_value_mean_replace":
        start_abs = _absolute_probability_value_start_position(prompt_len, decoded_tokens)
        if start_abs is None:
            return [], []
        seq_len = prompt_len + len(decoded_tokens)
        if start_abs >= seq_len:
            return [], []
        positions = list(range(start_abs, seq_len))
        if not positions:
            return [], []
        vectors = [probability[-1]] + [probability_value_mean] * (len(positions) - 1)
        return positions, vectors

    if mode == "current_generated_token_mean_replace":
        current_abs_pos = prompt_len + len(decoded_tokens) - 1
        if current_abs_pos < 0:
            return [], []
        return [current_abs_pos], [probability[-1]]

    raise ValueError(f"Unsupported mode for mean replacement: {mode!r}")


def build_headwise_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    head_idx: int,
) -> List[Tuple[str, Callable]]:
    head_idx = int(head_idx)
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.attn.hook_z"
        layer_means = layer_to_means[int(layer)]

        def _make_hook(local_layer_means: Dict[str, torch.Tensor]) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_z activation with shape [batch, seq, heads, d_head], got {tuple(activation.shape)}."
                    )
                if not (0 <= head_idx < activation.shape[2]):
                    raise ValueError(
                        f"Head index {head_idx} out of range for hook_z activation with {activation.shape[2]} heads."
                    )
                decoded_tokens = decoded_tokens_provider()
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_head_means=local_layer_means,
                )
                if not positions:
                    return activation
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != int(activation.shape[3]):
                            raise ValueError(
                                f"Replacement vector size {vector.numel()} does not match d_head {activation.shape[3]}."
                            )
                        activation[:, abs_pos, head_idx, :] = vector.to(
                            device=activation.device, dtype=activation.dtype
                        )
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer_means)))
    return hooks


def greedy_generate_headwise_mean_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    mode: str,
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    head_idx: int,
    kv_layer_to_k_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    kv_layer_to_v_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    kv_head_idx: Optional[int] = None,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_headwise_mean_replace_hooks(
        layer_indices=layer_indices,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_means=layer_to_means,
        head_idx=head_idx,
    )
    if kv_layer_to_k_means is not None and kv_layer_to_v_means is not None:
        if kv_head_idx is None:
            raise ValueError("kv_head_idx is required when KV mean replacement is enabled.")
        hooks.extend(
            build_kv_headwise_mean_replace_hooks(
                layer_indices=layer_indices,
                kv_name="k",
                mode=mode,
                prompt_len=prompt_len,
                decoded_tokens_provider=_decoded_tokens_provider,
                layer_to_means=kv_layer_to_k_means,
                kv_head_idx=kv_head_idx,
            )
        )
        hooks.extend(
            build_kv_headwise_mean_replace_hooks(
                layer_indices=layer_indices,
                kv_name="v",
                mode=mode,
                prompt_len=prompt_len,
                decoded_tokens_provider=_decoded_tokens_provider,
                layer_to_means=kv_layer_to_v_means,
                kv_head_idx=kv_head_idx,
            )
        )
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def build_grouped_head_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_head_means: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    grouped_head_indices: Sequence[int],
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    grouped_head_indices = [int(head_idx) for head_idx in grouped_head_indices]
    if not grouped_head_indices:
        raise ValueError("grouped_head_indices cannot be empty.")
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.attn.hook_z"
        layer_head_means = layer_to_head_means[int(layer)]

        def _make_hook(
            local_layer_head_means: Dict[int, Dict[str, torch.Tensor]], *, local_layer: int
        ) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_z activation with shape [batch, seq, heads, d_head], got {tuple(activation.shape)}."
                    )
                decoded_tokens = decoded_tokens_provider()
                positions: Optional[List[int]] = None
                replacements_by_head: Dict[int, List[torch.Tensor]] = {}
                for head_idx in grouped_head_indices:
                    if not (0 <= head_idx < activation.shape[2]):
                        raise ValueError(
                            f"Head index {head_idx} out of range for hook_z activation with {activation.shape[2]} heads."
                        )
                    head_means = local_layer_head_means.get(head_idx)
                    if head_means is None:
                        raise ValueError(
                            f"Missing grouped-head means for layer {local_layer}, head {head_idx}."
                        )
                    head_positions, head_vectors = _positions_and_replacements_for_mode(
                        mode=mode,
                        prompt_len=prompt_len,
                        decoded_tokens=decoded_tokens,
                        layer_head_means=head_means,
                    )
                    if positions is None:
                        positions = head_positions
                    elif positions != head_positions:
                        raise ValueError(
                            f"Inconsistent replacement positions for grouped heads at layer {local_layer}: "
                            f"expected {positions}, got {head_positions} for head {head_idx}."
                        )
                    replacements_by_head[head_idx] = head_vectors

                if not positions:
                    return activation
                for pos_i, abs_pos in enumerate(positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        for head_idx in grouped_head_indices:
                            vector = replacements_by_head[head_idx][pos_i]
                            if int(vector.numel()) != int(activation.shape[3]):
                                raise ValueError(
                                    f"Replacement vector size {vector.numel()} does not match d_head {activation.shape[3]}."
                                )
                            activation[:, abs_pos, head_idx, :] = vector.to(
                                device=activation.device, dtype=activation.dtype
                            )
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer_head_means, local_layer=int(layer))))
    return hooks


def greedy_generate_grouped_head_mean_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    mode: str,
    layer_to_head_means: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    grouped_head_indices: Sequence[int],
    kv_layer_to_k_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    kv_layer_to_v_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    kv_head_idx: Optional[int] = None,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_grouped_head_mean_replace_hooks(
        layer_indices=layer_indices,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_head_means=layer_to_head_means,
        grouped_head_indices=grouped_head_indices,
    )
    if kv_layer_to_k_means is not None and kv_layer_to_v_means is not None:
        if kv_head_idx is None:
            raise ValueError("kv_head_idx is required when KV mean replacement is enabled.")
        hooks.extend(
            build_kv_headwise_mean_replace_hooks(
                layer_indices=layer_indices,
                kv_name="k",
                mode=mode,
                prompt_len=prompt_len,
                decoded_tokens_provider=_decoded_tokens_provider,
                layer_to_means=kv_layer_to_k_means,
                kv_head_idx=kv_head_idx,
            )
        )
        hooks.extend(
            build_kv_headwise_mean_replace_hooks(
                layer_indices=layer_indices,
                kv_name="v",
                mode=mode,
                prompt_len=prompt_len,
                decoded_tokens_provider=_decoded_tokens_provider,
                layer_to_means=kv_layer_to_v_means,
                kv_head_idx=kv_head_idx,
            )
        )
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def build_concat_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    n_heads: int,
    d_head: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    n_heads = int(n_heads)
    d_head = int(d_head)
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.attn.hook_z"
        layer_means = layer_to_means[int(layer)]

        def _make_hook(local_layer_means: Dict[str, torch.Tensor]) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_z activation with shape [batch, seq, heads, d_head], got {tuple(activation.shape)}."
                    )
                if activation.shape[2] != n_heads or activation.shape[3] != d_head:
                    raise ValueError(
                        f"hook_z activation shape mismatch: got heads={activation.shape[2]}, d_head={activation.shape[3]}, "
                        f"expected heads={n_heads}, d_head={d_head}."
                    )
                decoded_tokens = decoded_tokens_provider()
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_head_means=local_layer_means,
                )
                if not positions:
                    return activation
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != n_heads * d_head:
                            raise ValueError(
                                f"Replacement vector size {vector.numel()} does not match n_heads*d_head={n_heads*d_head}."
                            )
                        activation[:, abs_pos, :, :] = vector.to(
                            device=activation.device, dtype=activation.dtype
                        ).reshape(n_heads, d_head)
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer_means)))
    return hooks


def build_kv_headwise_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    kv_name: str,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    kv_head_idx: int,
) -> List[Tuple[str, Callable]]:
    if kv_name not in ("k", "v"):
        raise ValueError(f"Unsupported kv_name {kv_name!r}; expected 'k' or 'v'.")
    kv_head_idx = int(kv_head_idx)
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.attn.hook_{kv_name}"
        layer_means = layer_to_means[int(layer)]

        def _make_hook(local_layer_means: Dict[str, torch.Tensor], *, local_layer: int) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_{kv_name} activation with shape [batch, seq, kv_heads, d_head], got {tuple(activation.shape)}."
                    )
                if not (0 <= kv_head_idx < activation.shape[2]):
                    raise ValueError(
                        f"KV head index {kv_head_idx} out of range for hook_{kv_name} activation with {activation.shape[2]} heads."
                    )
                decoded_tokens = decoded_tokens_provider()
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_head_means=local_layer_means,
                )
                if not positions:
                    return activation
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != int(activation.shape[3]):
                            raise ValueError(
                                f"KV replacement vector size {vector.numel()} does not match d_head {activation.shape[3]} "
                                f"for hook_{kv_name} at layer {local_layer}."
                            )
                        activation[:, abs_pos, kv_head_idx, :] = vector.to(
                            device=activation.device, dtype=activation.dtype
                        )
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer_means, local_layer=int(layer))))
    return hooks


def build_kv_concat_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    kv_name: str,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    n_kv_heads: int,
    d_head: int,
) -> List[Tuple[str, Callable]]:
    if kv_name not in ("k", "v"):
        raise ValueError(f"Unsupported kv_name {kv_name!r}; expected 'k' or 'v'.")
    hooks: List[Tuple[str, Callable]] = []
    n_kv_heads = int(n_kv_heads)
    d_head = int(d_head)
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.attn.hook_{kv_name}"
        layer_means = layer_to_means[int(layer)]

        def _make_hook(local_layer_means: Dict[str, torch.Tensor], *, local_layer: int) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_{kv_name} activation with shape [batch, seq, kv_heads, d_head], got {tuple(activation.shape)}."
                    )
                if activation.shape[2] != n_kv_heads or activation.shape[3] != d_head:
                    raise ValueError(
                        f"hook_{kv_name} activation shape mismatch at layer {local_layer}: "
                        f"got heads={activation.shape[2]}, d_head={activation.shape[3]}, "
                        f"expected heads={n_kv_heads}, d_head={d_head}."
                    )
                decoded_tokens = decoded_tokens_provider()
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_head_means=local_layer_means,
                )
                if not positions:
                    return activation
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != n_kv_heads * d_head:
                            raise ValueError(
                                f"KV replacement vector size {vector.numel()} does not match n_kv_heads*d_head={n_kv_heads*d_head} "
                                f"for hook_{kv_name} at layer {local_layer}."
                            )
                        activation[:, abs_pos, :, :] = vector.to(
                            device=activation.device, dtype=activation.dtype
                        ).reshape(n_kv_heads, d_head)
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer_means, local_layer=int(layer))))
    return hooks


def greedy_generate_concat_mean_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    mode: str,
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    n_heads: int,
    d_head: int,
    kv_layer_to_k_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    kv_layer_to_v_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    kv_head_idx: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_concat_mean_replace_hooks(
        layer_indices=layer_indices,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_means=layer_to_means,
        n_heads=n_heads,
        d_head=d_head,
    )
    if kv_layer_to_k_means is not None and kv_layer_to_v_means is not None:
        if kv_head_idx is not None:
            hooks.extend(
                build_kv_headwise_mean_replace_hooks(
                    layer_indices=layer_indices,
                    kv_name="k",
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    layer_to_means=kv_layer_to_k_means,
                    kv_head_idx=kv_head_idx,
                )
            )
            hooks.extend(
                build_kv_headwise_mean_replace_hooks(
                    layer_indices=layer_indices,
                    kv_name="v",
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    layer_to_means=kv_layer_to_v_means,
                    kv_head_idx=kv_head_idx,
                )
            )
        else:
            if n_kv_heads is None:
                raise ValueError("n_kv_heads is required when whole-KV mean replacement is enabled.")
            hooks.extend(
                build_kv_concat_mean_replace_hooks(
                    layer_indices=layer_indices,
                    kv_name="k",
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    layer_to_means=kv_layer_to_k_means,
                    n_kv_heads=n_kv_heads,
                    d_head=d_head,
                )
            )
            hooks.extend(
                build_kv_concat_mean_replace_hooks(
                    layer_indices=layer_indices,
                    kv_name="v",
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    layer_to_means=kv_layer_to_v_means,
                    n_kv_heads=n_kv_heads,
                    d_head=d_head,
                )
            )
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def write_headwise_mode_plots(
    *,
    mode_head_confidence_means: Dict[str, Dict[int, Optional[float]]],
    ablation_modes: Sequence[str],
    head_indices: Sequence[int],
    output_dir: str,
) -> None:
    head_order = sorted(int(h) for h in head_indices)
    baseline_none = mode_head_confidence_means.get("none", {})
    baseline_values = [v for v in baseline_none.values() if v is not None]
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else None

    for mode_name in ablation_modes:
        if mode_name == "none":
            continue
        ys: List[Optional[float]] = [mode_head_confidence_means.get(mode_name, {}).get(h) for h in head_order]
        valid_pairs = [(h, y) for h, y in zip(head_order, ys) if y is not None]
        if not valid_pairs and baseline_mean is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        if valid_pairs:
            xs = [h for h, _ in valid_pairs]
            yvals = [float(y) for _, y in valid_pairs]
            ax.plot(xs, yvals, marker="o", label=mode_name)
        if baseline_mean is not None:
            ax.axhline(y=baseline_mean, linestyle="--", label="none (baseline)")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Verbalised confidence")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Headwise verbalised confidence ({mode_name})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"verbalised_confidence_by_head__{mode_name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote %s", plot_path)


def write_grouped_mode_plots(
    *,
    mode_group_confidence_means: Dict[str, Dict[int, Optional[float]]],
    ablation_modes: Sequence[str],
    group_indices: Sequence[int],
    output_dir: str,
) -> None:
    group_order = sorted(int(g) for g in group_indices)
    baseline_none = mode_group_confidence_means.get("none", {})
    baseline_values = [v for v in baseline_none.values() if v is not None]
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else None

    for mode_name in ablation_modes:
        if mode_name == "none":
            continue
        ys: List[Optional[float]] = [mode_group_confidence_means.get(mode_name, {}).get(g) for g in group_order]
        valid_pairs = [(g, y) for g, y in zip(group_order, ys) if y is not None]
        if not valid_pairs and baseline_mean is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        if valid_pairs:
            xs = [g for g, _ in valid_pairs]
            yvals = [float(y) for _, y in valid_pairs]
            ax.plot(xs, yvals, marker="o", label=mode_name)
        if baseline_mean is not None:
            ax.axhline(y=baseline_mean, linestyle="--", label="none (baseline)")
        ax.set_xlabel("Group index")
        ax.set_ylabel("Verbalised confidence")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Grouped-head verbalised confidence ({mode_name})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"verbalised_confidence_by_group__{mode_name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote %s", plot_path)


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    model_n_heads: int,
    model_d_head: int,
    ablate_layers: Sequence[int],
    ablate_heads: Sequence[int],
    ablation_unit_mode: str,
    ablation_unit_keys: Sequence[str],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    mode_head_confidence_means: Dict[str, Dict[str, Optional[float]]],
    mode_head_confidence_counts: Dict[str, Dict[str, int]],
    mode_responses_identical_true: Dict[str, Dict[str, int]],
    finished_at: str,
) -> None:
    source_group = "low_confidence" if args.mean_from_low_confidence else "high_confidence"
    target_group = "high_confidence" if args.mean_from_low_confidence else "low_confidence"
    lines = [
        "Headwise Mean Ablation Configuration",
        "===================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"model_n_heads={model_n_heads}",
        f"model_d_head={model_d_head}",
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
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={source_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"ablate_heads_spec={args.ablate_heads}",
        f"ablate_heads_resolved={','.join(str(head) for head in ablate_heads) if ablate_heads else 'ignored'}",
        f"ablation_unit_mode={ablation_unit_mode}",
        f"ablate_kv_mean={args.ablate_kv_mean}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"num_ablated_heads={len(ablate_heads) if ablation_unit_mode != 'whole_concat' else 0}",
        f"num_ablation_units={len(ablation_unit_keys)}",
        f"ablation_units={','.join(ablation_unit_keys)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Mode Ablation Unit Confidence Metrics]",
        "Values below are mean verbalised confidence per ablation unit.",
        "",
    ]
    for mode_name in args.ablation_mode:
        per_unit = mode_head_confidence_means.get(mode_name, {})
        per_unit_counts = mode_head_confidence_counts.get(mode_name, {})
        for unit_key in ablation_unit_keys:
            mode_mean = per_unit.get(unit_key)
            valid_count = int(per_unit_counts.get(unit_key, 0))
            metric_key = f"{mode_name}__{unit_key}"
            if mode_name == "none":
                if mode_mean is None:
                    lines.append(f"{metric_key}=None ({valid_count})")
                else:
                    lines.append(f"{metric_key}={mode_mean:.6f} ({valid_count})")
            else:
                identical_n = int(mode_responses_identical_true.get(mode_name, {}).get(unit_key, 0))
                if mode_mean is None:
                    lines.append(
                        f"{metric_key}=None ({valid_count}) [responses_identical: {identical_n}]"
                    )
                else:
                    lines.append(
                        f"{metric_key}={mode_mean:.6f} ({valid_count}) [responses_identical: {identical_n}]"
                    )
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="Path to processed train/validation verbalised embedding H5 with concat fields.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_max_new_tokens", type=int, default=30)
    parser.add_argument("--brief_prompt", type=str, default="default", choices=["default", "chat"])
    parser.add_argument("--brief_always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_brief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablate_layers", type=str, default="12-15")
    parser.add_argument(
        "--ablate_heads",
        type=str,
        default="all",
        help=(
            "Head indices to ablate: 'all', inclusive range '0-31', or list '0,2,4'. "
            "When --ablation_unit_mode=grouped_head, this must be 'all'."
        ),
    )
    parser.add_argument(
        "--ablation_unit_mode",
        type=str,
        default="head",
        choices=ABLATION_UNIT_MODES,
        help=(
            "Atomic ablation unit: 'head' ablates each selected head individually, "
            "'grouped_head' ablates each GQA grouped-head unit (requires --ablate_heads=all), "
            "and 'whole_concat' ablates the full concatenated embedding as one unit."
        ),
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=ABLATION_MODES_DEFAULT,
        choices=ABLATION_MODES_DEFAULT,
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--mean_from_low_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--ablate_kv_mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also mean-patch attention K/V at the same mode-dependent positions as hook_z mean replacement."
        ),
    )
    parser.add_argument("--parse_mode_verbalised_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline plotting.")
    ablation_unit_mode = str(args.ablation_unit_mode)
    if ablation_unit_mode == "grouped_head" and not _is_all_heads_spec(args.ablate_heads):
        raise ValueError(
            "When --ablation_unit_mode=grouped_head, --ablate_heads must be 'all'."
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)
    os.makedirs(run_root, exist_ok=True)

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
    model_n_layers = int(model.cfg.n_layers)
    model_n_heads = int(model.cfg.n_heads)
    model_d_head = int(model.cfg.d_head)
    model_d_model = int(model.cfg.d_model)
    if model_n_heads * model_d_head != model_d_model:
        raise ValueError(
            f"Model shape mismatch: n_heads*d_head={model_n_heads*model_d_head}, d_model={model_d_model}."
        )
    model_n_kv_heads = resolve_n_key_value_heads(model.cfg, model_n_heads)
    kv_heads_per_query_group = model_n_heads // model_n_kv_heads

    ablate_layers = parse_ablate_layers(args.ablate_layers, model_n_layers)
    if ablation_unit_mode == "whole_concat":
        ablate_heads: List[int] = []
        logging.info(
            "ablation_unit_mode=whole_concat: ignoring --ablate_heads=%s",
            args.ablate_heads,
        )
    else:
        ablate_heads = parse_head_indices(args.ablate_heads, model_n_heads)

    examples_h5 = load_examples_h5(Path(args.input_h5))
    if ablation_unit_mode == "whole_concat":
        means, low_ids, high_ids = compute_concat_whole_means(
            examples_h5,
            ablate_layers=ablate_layers,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
            mean_from_low_confidence=args.mean_from_low_confidence,
            expected_probability_tokens=args.expected_probability_tokens,
            expected_guess_tokens=args.expected_guess_tokens,
        )
    else:
        means, low_ids, high_ids = compute_concat_headwise_means(
            examples_h5,
            ablate_layers=ablate_layers,
            n_heads=model_n_heads,
            d_head=model_d_head,
            low_conf_threshold=args.low_conf_threshold,
            high_conf_threshold=args.high_conf_threshold,
            mean_from_low_confidence=args.mean_from_low_confidence,
            expected_probability_tokens=args.expected_probability_tokens,
            expected_guess_tokens=args.expected_guess_tokens,
        )
    kv_means_k: Optional[Dict[str, np.ndarray]] = None
    kv_means_v: Optional[Dict[str, np.ndarray]] = None
    if args.ablate_kv_mean:
        if ablation_unit_mode == "whole_concat":
            kv_means_k, low_ids_k, high_ids_k = compute_kv_whole_means(
                examples_h5,
                component="k",
                ablate_layers=ablate_layers,
                n_kv_heads=model_n_kv_heads,
                d_head=model_d_head,
                low_conf_threshold=args.low_conf_threshold,
                high_conf_threshold=args.high_conf_threshold,
                mean_from_low_confidence=args.mean_from_low_confidence,
                expected_probability_tokens=args.expected_probability_tokens,
                expected_guess_tokens=args.expected_guess_tokens,
            )
            kv_means_v, low_ids_v, high_ids_v = compute_kv_whole_means(
                examples_h5,
                component="v",
                ablate_layers=ablate_layers,
                n_kv_heads=model_n_kv_heads,
                d_head=model_d_head,
                low_conf_threshold=args.low_conf_threshold,
                high_conf_threshold=args.high_conf_threshold,
                mean_from_low_confidence=args.mean_from_low_confidence,
                expected_probability_tokens=args.expected_probability_tokens,
                expected_guess_tokens=args.expected_guess_tokens,
            )
        else:
            kv_means_k, low_ids_k, high_ids_k = compute_kv_headwise_means(
                examples_h5,
                component="k",
                ablate_layers=ablate_layers,
                n_kv_heads=model_n_kv_heads,
                d_head=model_d_head,
                low_conf_threshold=args.low_conf_threshold,
                high_conf_threshold=args.high_conf_threshold,
                mean_from_low_confidence=args.mean_from_low_confidence,
                expected_probability_tokens=args.expected_probability_tokens,
                expected_guess_tokens=args.expected_guess_tokens,
            )
            kv_means_v, low_ids_v, high_ids_v = compute_kv_headwise_means(
                examples_h5,
                component="v",
                ablate_layers=ablate_layers,
                n_kv_heads=model_n_kv_heads,
                d_head=model_d_head,
                low_conf_threshold=args.low_conf_threshold,
                high_conf_threshold=args.high_conf_threshold,
                mean_from_low_confidence=args.mean_from_low_confidence,
                expected_probability_tokens=args.expected_probability_tokens,
                expected_guess_tokens=args.expected_guess_tokens,
            )
        if low_ids_k != low_ids_v or high_ids_k != high_ids_v:
            raise ValueError("Confidence grouping mismatch between KV(k) and KV(v) mean builders.")
        if low_ids_k != low_ids or high_ids_k != high_ids:
            raise ValueError("Confidence grouping mismatch between concat and KV mean builders.")

    low_ids_check, high_ids_check = collect_confidence_group_ids(
        examples_h5,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
    )
    if low_ids_check != low_ids or high_ids_check != high_ids:
        raise ValueError("Confidence grouping mismatch between mean builder and collector.")

    if args.mean_from_low_confidence:
        ablation_target_ids = high_ids
        target_group = "high_confidence"
    else:
        ablation_target_ids = low_ids
        target_group = "low_confidence"
    if not ablation_target_ids:
        raise ValueError(f"No examples available in ablation target group: {target_group}.")

    grouped_head_units: Dict[str, List[int]] = {}
    grouped_layer_means_by_unit: Dict[str, Dict[int, Dict[int, Dict[str, torch.Tensor]]]] = {}
    layer_kv_k_concat_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    layer_kv_v_concat_means: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    layer_kv_k_head_means: Optional[Dict[int, Dict[int, Dict[str, torch.Tensor]]]] = None
    layer_kv_v_head_means: Optional[Dict[int, Dict[int, Dict[str, torch.Tensor]]]] = None
    kv_unit_index_by_key: Dict[str, int] = {}
    if ablation_unit_mode == "whole_concat":
        layer_concat_means = _build_layer_concat_means(
            means,
            ablate_layers=ablate_layers,
            device=device,
            torch_dtype=torch_dtype,
        )
        ablation_unit_keys = ["whole_concat"]
        if args.ablate_kv_mean:
            if kv_means_k is None or kv_means_v is None:
                raise ValueError("KV mean replacement was enabled but KV means are missing.")
            layer_kv_k_concat_means = _build_layer_kv_concat_means(
                kv_means_k,
                ablate_layers=ablate_layers,
                device=device,
                torch_dtype=torch_dtype,
            )
            layer_kv_v_concat_means = _build_layer_kv_concat_means(
                kv_means_v,
                ablate_layers=ablate_layers,
                device=device,
                torch_dtype=torch_dtype,
            )
    else:
        layer_head_means = _build_layer_head_means(
            means,
            ablate_layers=ablate_layers,
            head_indices=ablate_heads,
            device=device,
            torch_dtype=torch_dtype,
        )
        if args.ablate_kv_mean:
            if kv_means_k is None or kv_means_v is None:
                raise ValueError("KV mean replacement was enabled but KV means are missing.")
            layer_kv_k_head_means = _build_layer_kv_head_means(
                kv_means_k,
                ablate_layers=ablate_layers,
                kv_head_indices=list(range(model_n_kv_heads)),
                device=device,
                torch_dtype=torch_dtype,
            )
            layer_kv_v_head_means = _build_layer_kv_head_means(
                kv_means_v,
                ablate_layers=ablate_layers,
                kv_head_indices=list(range(model_n_kv_heads)),
                device=device,
                torch_dtype=torch_dtype,
            )
        if ablation_unit_mode == "head":
            ablation_unit_keys = [f"head_{head_idx}" for head_idx in ablate_heads]
            kv_unit_index_by_key = {
                f"head_{head_idx}": int(head_idx) // kv_heads_per_query_group for head_idx in ablate_heads
            }
        elif ablation_unit_mode == "grouped_head":
            grouped_head_units = build_grouped_head_units(
                n_heads=model_n_heads, n_kv_heads=model_n_kv_heads
            )
            ablation_unit_keys = list(grouped_head_units.keys())
            kv_unit_index_by_key = {f"group_{kv_idx}": kv_idx for kv_idx in range(model_n_kv_heads)}
            grouped_layer_means_by_unit = {
                unit_key: {
                    layer: {
                        head_idx: layer_head_means[layer][head_idx]
                        for head_idx in grouped_head_indices
                    }
                    for layer in ablate_layers
                }
                for unit_key, grouped_head_indices in grouped_head_units.items()
            }
            logging.info(
                "ablation_unit_mode=grouped_head: n_heads=%d n_kv_heads=%d grouped_units=%d heads_per_group=%d",
                model_n_heads,
                model_n_kv_heads,
                len(grouped_head_units),
                kv_heads_per_query_group,
            )
        else:
            raise ValueError(
                f"Unsupported --ablation_unit_mode {ablation_unit_mode!r}; expected one of {ABLATION_UNIT_MODES}."
            )
    logging.info(
        "Loaded %d H5 examples. low_conf=%d high_conf=%d target_group=%s target_ids=%d layers=%s ablation_units=%d ablate_kv_mean=%s",
        len(examples_h5),
        len(low_ids),
        len(high_ids),
        target_group,
        len(ablation_target_ids),
        ablate_layers,
        len(ablation_unit_keys),
        args.ablate_kv_mean,
    )

    results_mini = {"train": {}, "validation": {}}
    mode_head_confidence_values: Dict[str, Dict[str, List[float]]] = {
        mode: {unit: [] for unit in ablation_unit_keys} for mode in args.ablation_mode
    }
    mode_responses_identical_true: Dict[str, Dict[str, int]] = {
        mode: {unit: 0 for unit in ablation_unit_keys} for mode in args.ablation_mode if mode != "none"
    }

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        split_target = (
            round(args.num_samples * TRAIN_RATIO)
            if split_name == "train"
            else round(args.num_samples * (1 - TRAIN_RATIO))
        )
        id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
        split_target_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)
        if split_target > 0 and not split_target_ids:
            logging.warning("No ablation target IDs available for %s split.", split_name)
            continue
        selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
        logging.info("Generating for %d examples (%s split).", len(selected_ids), split_name)

        for i, ex_id in enumerate(selected_ids):
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                raise ValueError(f"Example id {ex_id} selected from H5 but missing in {split_name} split.")
            example = eval_ds[int(ds_idx)]
            local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + example["question"]
            entry = {"question": example["question"]}

            baseline_response, baseline_decoded_tokens = greedy_generate(
                model=model,
                local_prompt=local_prompt,
                max_new_tokens=args.model_max_new_tokens,
                fwd_hooks=None,
            )
            baseline_mode_confidence = (
                parse_mode_confidence_from_response(baseline_response)
                if args.parse_mode_verbalised_confidence
                else None
            )

            for mode_name in args.ablation_mode:
                key = mode_to_output_key(mode_name)
                if mode_name == "none":
                    entry[key] = {
                        "response": baseline_response,
                        "verbalised_confidence": baseline_mode_confidence,
                    }
                    if baseline_mode_confidence is not None:
                        for unit_key in ablation_unit_keys:
                            mode_head_confidence_values[mode_name][unit_key].append(
                                float(baseline_mode_confidence)
                            )
                    continue

                entry[key] = {}
                if ablation_unit_mode == "whole_concat":
                    response, _ = greedy_generate_concat_mean_ablated(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=args.model_max_new_tokens,
                        layer_indices=ablate_layers,
                        mode=mode_name,
                        layer_to_means=layer_concat_means,
                        n_heads=model_n_heads,
                        d_head=model_d_head,
                        kv_layer_to_k_means=layer_kv_k_concat_means if args.ablate_kv_mean else None,
                        kv_layer_to_v_means=layer_kv_v_concat_means if args.ablate_kv_mean else None,
                        n_kv_heads=model_n_kv_heads if args.ablate_kv_mean else None,
                    )
                    mode_confidence = (
                        parse_mode_confidence_from_response(response)
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    responses_identical = response == baseline_response
                    if responses_identical:
                        mode_responses_identical_true[mode_name]["whole_concat"] += 1
                    if mode_confidence is not None:
                        mode_head_confidence_values[mode_name]["whole_concat"].append(float(mode_confidence))
                    entry[key]["whole_concat"] = {
                        "response": response,
                        "verbalised_confidence": mode_confidence,
                        "responses_identical": responses_identical,
                    }
                    logging.info(
                        "[%s %d/%d] %s %s/whole_concat first line: %r",
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                        key,
                        response[:120],
                    )
                elif ablation_unit_mode == "head":
                    for head_idx in ablate_heads:
                        unit_key = f"head_{head_idx}"
                        kv_head_idx = kv_unit_index_by_key[unit_key] if args.ablate_kv_mean else None
                        response, _ = greedy_generate_headwise_mean_ablated(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_indices=ablate_layers,
                            mode=mode_name,
                            layer_to_means={layer: layer_head_means[layer][head_idx] for layer in ablate_layers},
                            head_idx=head_idx,
                            kv_layer_to_k_means=(
                                {layer: layer_kv_k_head_means[layer][kv_head_idx] for layer in ablate_layers}
                                if args.ablate_kv_mean and layer_kv_k_head_means is not None and kv_head_idx is not None
                                else None
                            ),
                            kv_layer_to_v_means=(
                                {layer: layer_kv_v_head_means[layer][kv_head_idx] for layer in ablate_layers}
                                if args.ablate_kv_mean and layer_kv_v_head_means is not None and kv_head_idx is not None
                                else None
                            ),
                            kv_head_idx=kv_head_idx,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                        responses_identical = response == baseline_response
                        if responses_identical:
                            mode_responses_identical_true[mode_name][unit_key] += 1
                        if mode_confidence is not None:
                            mode_head_confidence_values[mode_name][unit_key].append(float(mode_confidence))
                        entry[key][unit_key] = {
                            "response": response,
                            "verbalised_confidence": mode_confidence,
                            "responses_identical": responses_identical,
                        }

                        logging.info(
                            "[%s %d/%d] %s %s/head_%d first line: %r",
                            split_name,
                            i + 1,
                            len(selected_ids),
                            ex_id,
                            key,
                            head_idx,
                            response[:120],
                        )
                elif ablation_unit_mode == "grouped_head":
                    for unit_key, grouped_head_indices in grouped_head_units.items():
                        kv_head_idx = kv_unit_index_by_key[unit_key] if args.ablate_kv_mean else None
                        response, _ = greedy_generate_grouped_head_mean_ablated(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_indices=ablate_layers,
                            mode=mode_name,
                            layer_to_head_means=grouped_layer_means_by_unit[unit_key],
                            grouped_head_indices=grouped_head_indices,
                            kv_layer_to_k_means=(
                                {layer: layer_kv_k_head_means[layer][kv_head_idx] for layer in ablate_layers}
                                if args.ablate_kv_mean and layer_kv_k_head_means is not None and kv_head_idx is not None
                                else None
                            ),
                            kv_layer_to_v_means=(
                                {layer: layer_kv_v_head_means[layer][kv_head_idx] for layer in ablate_layers}
                                if args.ablate_kv_mean and layer_kv_v_head_means is not None and kv_head_idx is not None
                                else None
                            ),
                            kv_head_idx=kv_head_idx,
                        )
                        mode_confidence = (
                            parse_mode_confidence_from_response(response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                        responses_identical = response == baseline_response
                        if responses_identical:
                            mode_responses_identical_true[mode_name][unit_key] += 1
                        if mode_confidence is not None:
                            mode_head_confidence_values[mode_name][unit_key].append(float(mode_confidence))
                        entry[key][unit_key] = {
                            "response": response,
                            "verbalised_confidence": mode_confidence,
                            "responses_identical": responses_identical,
                        }
                        logging.info(
                            "[%s %d/%d] %s %s/%s heads=%s first line: %r",
                            split_name,
                            i + 1,
                            len(selected_ids),
                            ex_id,
                            key,
                            unit_key,
                            grouped_head_indices,
                            response[:120],
                        )
                else:
                    raise ValueError(
                        f"Unsupported --ablation_unit_mode {ablation_unit_mode!r}; expected one of {ABLATION_UNIT_MODES}."
                    )

            results_mini[split_name][ex_id] = entry

    mode_head_confidence_means: Dict[str, Dict[str, Optional[float]]] = {}
    mode_head_confidence_counts: Dict[str, Dict[str, int]] = {}
    for mode_name in args.ablation_mode:
        mode_head_confidence_means[mode_name] = {}
        mode_head_confidence_counts[mode_name] = {}
        for unit_key in ablation_unit_keys:
            vals = mode_head_confidence_values[mode_name][unit_key]
            mode_head_confidence_means[mode_name][unit_key] = float(np.mean(vals)) if vals else None
            mode_head_confidence_counts[mode_name][unit_key] = len(vals)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_mini, f, ensure_ascii=False, indent=2)
    write_config_txt(
        config_txt_path(out_path),
        args=args,
        device=device,
        model_n_layers=model_n_layers,
        model_n_heads=model_n_heads,
        model_d_head=model_d_head,
        ablate_layers=ablate_layers,
        ablate_heads=ablate_heads,
        ablation_unit_mode=ablation_unit_mode,
        ablation_unit_keys=ablation_unit_keys,
        prompt_indices=prompt_indices,
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        mode_head_confidence_means=mode_head_confidence_means,
        mode_head_confidence_counts=mode_head_confidence_counts,
        mode_responses_identical_true=mode_responses_identical_true,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if ablation_unit_mode == "head":
        write_headwise_mode_plots(
            mode_head_confidence_means={
                mode_name: {
                    int(unit_key.replace("head_", "")): value
                    for unit_key, value in mode_head_confidence_means[mode_name].items()
                }
                for mode_name in args.ablation_mode
            },
            ablation_modes=args.ablation_mode,
            head_indices=ablate_heads,
            output_dir=os.path.dirname(out_path),
        )
    elif ablation_unit_mode == "grouped_head":
        write_grouped_mode_plots(
            mode_group_confidence_means={
                mode_name: {
                    int(unit_key.replace("group_", "")): value
                    for unit_key, value in mode_head_confidence_means[mode_name].items()
                }
                for mode_name in args.ablation_mode
            },
            ablation_modes=args.ablation_mode,
            group_indices=[int(unit_key.replace("group_", "")) for unit_key in ablation_unit_keys],
            output_dir=os.path.dirname(out_path),
        )
    else:
        logging.info(
            "ablation_unit_mode=%s: skipping plot generation (only head/grouped_head modes are plotted).",
            ablation_unit_mode,
        )
    logging.info("Saved mini outputs to %s", out_path)


if __name__ == "__main__":
    main()
