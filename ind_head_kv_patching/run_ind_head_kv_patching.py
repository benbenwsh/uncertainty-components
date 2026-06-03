#!/usr/bin/env python3
"""Selected-head Q/K/V patching ablation runner for structured Guess/Probability decoding."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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
MODE_NONE = "none"
MODE_PROBABILITY_TOKENS = "probability_tokens_mean_replace"
MODE_PROBABILITY_LAST_TOKEN = "probability_last_token_mean_replace"
MODE_PROBABILITY_SPAN_EXCEPT_LAST_TOKEN = "probability_span_except_last_token_mean_replace"
MODE_ALL_PRE_PROBABILITY_TOKENS = "all_pre_probability_tokens_mean_replace"
MODE_GUESS_TOKENS = "guess_tokens_mean_replace"
MODE_ALL_PRE_GUESS_TOKENS = "all_pre_guess_tokens_mean_replace"
MODE_GUESS_THEN_GUESS_PROBABILITY = "guess_then_guess_probability_mean_replace"
MODE_PROBABILITY_VALUE = "probability_value_mean_replace"
MODE_ANS_ADJ_TOKENS = "ans_adj_tokens_mean_replace"
MODE_SOURCE_REQUIREMENTS = {
    MODE_PROBABILITY_TOKENS: {"probability"},
    MODE_PROBABILITY_LAST_TOKEN: {"probability"},
    MODE_PROBABILITY_SPAN_EXCEPT_LAST_TOKEN: {"probability"},
    MODE_ALL_PRE_PROBABILITY_TOKENS: {"prompt_mean", "guess", "sem_answer_mean", "probability"},
    MODE_GUESS_TOKENS: {"guess"},
    MODE_ALL_PRE_GUESS_TOKENS: {"prompt_mean", "guess"},
    MODE_GUESS_THEN_GUESS_PROBABILITY: {"guess", "probability"},
    MODE_PROBABILITY_VALUE: {"probability", "probability_value_mean"},
    MODE_ANS_ADJ_TOKENS: {"probability"},
}


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path(__file__).resolve().parent / "results"
        base.mkdir(parents=True, exist_ok=True)
        run_id = 1
        while (base / str(run_id)).exists():
            run_id += 1
        run_dir = base / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(run_dir / "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def mini_output_json_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "ablation_results_mini.json")


def config_txt_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "config.txt")


def mode_to_output_key(mode: str) -> str:
    if mode == MODE_NONE:
        return "no_replacement"
    if mode == MODE_PROBABILITY_TOKENS:
        return "probability_tokens_mean_replace"
    if mode == MODE_PROBABILITY_LAST_TOKEN:
        return "probability_last_token_mean_replace"
    if mode == MODE_PROBABILITY_SPAN_EXCEPT_LAST_TOKEN:
        return "probability_span_except_last_token_mean_replace"
    if mode == MODE_ALL_PRE_PROBABILITY_TOKENS:
        return "all_pre_probability_tokens_mean_replace"
    if mode == MODE_GUESS_TOKENS:
        return "guess_tokens_mean_replace"
    if mode == MODE_ALL_PRE_GUESS_TOKENS:
        return "all_pre_guess_tokens_mean_replace"
    if mode == MODE_GUESS_THEN_GUESS_PROBABILITY:
        return "guess_then_guess_probability_mean_replace"
    if mode == MODE_PROBABILITY_VALUE:
        return "probability_value_mean_replace"
    if mode == MODE_ANS_ADJ_TOKENS:
        return "ans_adj_tokens_mean_replace"
    raise ValueError(f"Unsupported mode: {mode}")


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


def parse_layer_head_pairs(spec: str, *, n_layers: int, n_kv_heads: int) -> Dict[int, List[int]]:
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("--ablate_heads_by_layer must be a non-empty comma-separated <layer>.<head> list.")
    out: Dict[int, Set[int]] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if item.count(".") != 1:
            raise ValueError(
                f"Invalid head token {item!r}. Expected format <layer_idx>.<head_idx>, e.g. 12.3."
            )
        layer_str, head_str = item.split(".", 1)
        try:
            layer_idx = int(layer_str)
            head_idx = int(head_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid head token {item!r}. layer/head must be integers in <layer_idx>.<head_idx>."
            ) from exc
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}).")
        if head_idx < 0 or head_idx >= n_kv_heads:
            raise ValueError(f"KV head index {head_idx} out of range [0, {n_kv_heads}).")
        out.setdefault(layer_idx, set()).add(head_idx)
    if not out:
        raise ValueError("No valid <layer_idx>.<head_idx> entries found in --ablate_heads_by_layer.")
    return {layer: sorted(heads) for layer, heads in sorted(out.items())}


def format_layer_head_pairs(layer_to_heads: Dict[int, Sequence[int]]) -> str:
    parts: List[str] = []
    for layer_idx in sorted(layer_to_heads):
        for head_idx in sorted(set(int(h) for h in layer_to_heads[layer_idx])):
            parts.append(f"{layer_idx}.{head_idx}")
    return ",".join(parts)


def parse_patch_components(spec: str) -> List[str]:
    raw = (spec or "").strip().lower()
    if not raw:
        raise ValueError("--patch_components must be a non-empty CSV of q,k,v (for example: k,v).")
    allowed = {"q", "k", "v"}
    parsed: List[str] = []
    seen: Set[str] = set()
    for token in raw.split(","):
        component = token.strip()
        if not component:
            continue
        if component not in allowed:
            raise ValueError(f"Invalid component {component!r} in --patch_components. Allowed: q,k,v.")
        if component in seen:
            continue
        seen.add(component)
        parsed.append(component)
    if not parsed:
        raise ValueError("--patch_components must include at least one of q,k,v.")
    return parsed


def expand_kv_heads_to_q_heads(selected_kv_heads: Sequence[int], *, n_q_heads: int, n_kv_heads: int) -> List[int]:
    if n_q_heads <= 0 or n_kv_heads <= 0:
        raise ValueError(f"Invalid head counts for expansion: n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}.")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"Invalid GQA head configuration for expansion: {n_q_heads} not divisible by {n_kv_heads}.")
    if n_q_heads == n_kv_heads:
        return sorted(set(int(h) for h in selected_kv_heads))
    group_size = n_q_heads // n_kv_heads
    expanded: Set[int] = set()
    for kv_head in selected_kv_heads:
        kv_head = int(kv_head)
        if kv_head < 0 or kv_head >= n_kv_heads:
            raise ValueError(f"KV head index {kv_head} out of range [0, {n_kv_heads}) for Q expansion.")
        start = kv_head * group_size
        expanded.update(range(start, start + group_size))
    return sorted(expanded)


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    patch_components: Sequence[str],
    ablate_layers: Sequence[int],
    selected_layer_heads: Dict[int, Sequence[int]],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    mode_confidence_means: Dict[str, Optional[float]],
    mode_confidence_counts: Dict[str, int],
    mode_responses_identical_true: Dict[str, int],
    finished_at: str,
) -> None:
    source_group = "low_confidence" if args.mean_from_low_confidence else "high_confidence"
    target_group = "high_confidence" if args.mean_from_low_confidence else "low_confidence"
    lines = [
        "Individual Head QKV Patching Configuration",
        "========================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        f"model_n_q_heads={n_q_heads}",
        f"model_n_kv_heads={n_kv_heads}",
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
        f"fixed_patch_strategy=span_key_positions",
        f"ablation_mode={args.ablation_mode}",
        f"patch_components={','.join(patch_components)}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_heads_by_layer_spec={args.ablate_heads_by_layer}",
        f"ablate_heads_by_layer_resolved={format_layer_head_pairs(selected_layer_heads)}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={source_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Mode Confidence Metrics]",
    ]
    for mode_name in args.ablation_mode:
        mode_mean = mode_confidence_means.get(mode_name)
        valid_count = int(mode_confidence_counts.get(mode_name, 0))
        if mode_name == MODE_NONE:
            lines.append(f"{mode_name}={'None' if mode_mean is None else f'{mode_mean:.6f}'} ({valid_count})")
        else:
            identical_n = int(mode_responses_identical_true.get(mode_name, 0))
            lines.append(
                f"{mode_name}={'None' if mode_mean is None else f'{mode_mean:.6f}'} "
                f"({valid_count}) [responses_identical: {identical_n}]"
            )
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _h5_field_name_for_source(source_name: str) -> str:
    mapping = {
        "prompt_mean": "embeddings_mean_prompt",
        "guess": "embeddings_guess",
        "sem_answer_mean": "embeddings_mean_sem_answer",
        "probability": "embeddings_probability",
        "probability_value_mean": "embeddings_mean_prob_val",
    }
    if source_name not in mapping:
        raise ValueError(f"Unsupported source_name: {source_name}")
    return mapping[source_name]


def _validate_component_field(
    resp0: dict,
    *,
    ex_id: str,
    field_name: str,
    component: str,
    expect_list: bool,
):
    field_value = resp0.get(field_name)
    if not isinstance(field_value, dict):
        raise ValueError(f"Example {ex_id} responses/0/{field_name} must be an object.")
    if component not in field_value:
        raise ValueError(f"Example {ex_id} responses/0/{field_name} missing key '{component}'.")
    value = field_value.get(component)
    if value is None:
        raise ValueError(f"Example {ex_id} responses/0/{field_name}/{component} is null.")
    if expect_list and not isinstance(value, list):
        raise ValueError(f"Example {ex_id} responses/0/{field_name}/{component} must be a list.")
    return value


def compute_component_group_means(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    patch_components: Sequence[str],
    required_sources: Set[str],
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    component_source_vectors: Dict[str, List[np.ndarray]] = {}
    for component in patch_components:
        for source_name in sorted(required_sources):
            component_source_vectors[f"{component}_{source_name}"] = []
    layer_idx = np.asarray(ablate_layers)

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

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

        for source_name in sorted(required_sources):
            field_name = _h5_field_name_for_source(source_name)
            expect_list = source_name in {"guess", "probability"}
            for component in patch_components:
                value = _validate_component_field(
                    resp0,
                    ex_id=ex_id,
                    field_name=field_name,
                    component=component,
                    expect_list=expect_list,
                )
                key = f"{component}_{source_name}"
                if source_name == "guess":
                    if not _is_expected_or_plus_one(len(value), expected_guess_tokens):
                        raise ValueError(
                            f"Example {ex_id} {field_name}/{component} has length {len(value)}; "
                            f"expected {expected_guess_tokens} or {expected_guess_tokens + 1}."
                        )
                    token_values = value[:expected_guess_tokens]
                    selected: List[np.ndarray] = []
                    for token_arr in token_values:
                        selected.append(_as_layer_hidden(token_arr)[layer_idx, :])
                    component_source_vectors[key].append(np.stack(selected, axis=1))
                    continue
                if source_name == "probability":
                    if not _is_expected_or_plus_one(len(value), expected_probability_tokens):
                        raise ValueError(
                            f"Example {ex_id} {field_name}/{component} has length {len(value)}; "
                            f"expected {expected_probability_tokens} or {expected_probability_tokens + 1}."
                        )
                    token_values = value[:expected_probability_tokens]
                    selected: List[np.ndarray] = []
                    for token_arr in token_values:
                        selected.append(_as_layer_hidden(token_arr)[layer_idx, :])
                    component_source_vectors[key].append(np.stack(selected, axis=1))
                    continue
                component_source_vectors[key].append(_as_layer_hidden(value)[layer_idx, :])

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if any(not vectors for vectors in component_source_vectors.values()):
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        required_fields = ",".join(_h5_field_name_for_source(src) for src in sorted(required_sources))
        raise ValueError(
            f"No {source_name} examples with non-null component embeddings in [{required_fields}] found "
            f"under threshold condition verbalised_confidence {operator} {threshold}."
        )
    component_means: Dict[str, np.ndarray] = {}
    for key, vectors in component_source_vectors.items():
        component_means[key] = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
    return (
        component_means,
        low_ids,
        high_ids,
    )


def _build_layer_component_means(
    component_means: Dict[str, np.ndarray],
    *,
    ablate_layers: Sequence[int],
    component_mean_keys: Sequence[str],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[str, torch.Tensor]]:
    layer_to_means: Dict[int, Dict[str, torch.Tensor]] = {}
    for local_idx, layer_i in enumerate(ablate_layers):
        per_layer: Dict[str, torch.Tensor] = {}
        for key in component_mean_keys:
            per_layer[key] = torch.tensor(component_means[key][local_idx], device=device, dtype=torch_dtype)
        layer_to_means[int(layer_i)] = per_layer
    return layer_to_means


def _reshape_vector_for_activation(
    vector: torch.Tensor,
    activation: torch.Tensor,
    layer: int,
    component_name: str,
) -> torch.Tensor:
    heads = int(activation.shape[-2])
    d_head = int(activation.shape[-1])
    expected = heads * d_head
    vector = vector.to(device=activation.device, dtype=activation.dtype)
    if int(vector.numel()) == expected:
        return vector.reshape(heads, d_head)
    if int(vector.numel()) == d_head:
        return vector.unsqueeze(0).repeat(heads, 1)
    raise ValueError(
        f"Layer {layer} {component_name} replacement size mismatch: vector numel={vector.numel()}, "
        f"activation per-position shape=({heads}, {d_head})."
    )


def _sequence_source_length(
    layer_means: Dict[str, torch.Tensor],
    *,
    patch_components: Sequence[str],
    source_name: str,
) -> int:
    expected: Optional[int] = None
    for component in patch_components:
        key = f"{component}_{source_name}"
        source_tensor = layer_means.get(key)
        if source_tensor is None:
            raise ValueError(f"Missing layer means component key: {key}.")
        if source_tensor.ndim < 2:
            raise ValueError(
                f"Layer means key {key} expected rank >= 2 (sequence source), got shape {tuple(source_tensor.shape)}."
            )
        cur_len = int(source_tensor.shape[0])
        if expected is None:
            expected = cur_len
        elif cur_len != expected:
            raise ValueError(f"Inconsistent source lengths for {source_name}: {expected} vs {cur_len}.")
    if expected is None:
        raise ValueError(f"No patch components provided while computing source length for {source_name}.")
    return expected


def _build_patch_plan_entries(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_means: Dict[str, torch.Tensor],
    patch_components: Sequence[str],  # used for source-shape validation consistency
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[int, str, int]]:
    if mode == MODE_PROBABILITY_TOKENS:
        positions = _absolute_prob_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        n = min(len(positions), n_prob)
        return [(positions[i], "probability", i) for i in range(n)]

    if mode == MODE_PROBABILITY_LAST_TOKEN:
        positions = _absolute_prob_last_token_only_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        if not positions or n_prob <= 0:
            return []
        return [(positions[0], "probability", n_prob - 1)]

    if mode == MODE_PROBABILITY_SPAN_EXCEPT_LAST_TOKEN:
        positions = _absolute_prob_except_last_token_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        n = min(len(positions), max(0, n_prob - 1))
        return [(positions[i], "probability", i) for i in range(n)]

    if mode == MODE_ANS_ADJ_TOKENS:
        positions = _absolute_prob_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        n = min(len(positions), n_prob, 2)
        return [(positions[i], "probability", i) for i in range(n)]

    if mode == MODE_GUESS_TOKENS:
        positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        n_guess = _sequence_source_length(layer_means, patch_components=patch_components, source_name="guess")
        n = min(len(positions), n_guess)
        return [(positions[i], "guess", i) for i in range(n)]

    if mode == MODE_ALL_PRE_GUESS_TOKENS:
        guess_positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        all_positions = _absolute_all_pre_guess_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        if not all_positions:
            return []
        prompt_count = max(0, len(all_positions) - len(guess_positions))
        n_guess = _sequence_source_length(layer_means, patch_components=patch_components, source_name="guess")
        entries: List[Tuple[int, str, int]] = []
        for abs_pos in all_positions[:prompt_count]:
            entries.append((abs_pos, "prompt_mean", -1))
        for i in range(min(len(guess_positions), n_guess)):
            entries.append((guess_positions[i], "guess", i))
        return entries

    if mode == MODE_ALL_PRE_PROBABILITY_TOKENS:
        spans = _absolute_pre_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        if spans is None:
            return []
        n_guess = _sequence_source_length(layer_means, patch_components=patch_components, source_name="guess")
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        entries: List[Tuple[int, str, int]] = []
        for abs_pos in spans["prompt"]:
            entries.append((abs_pos, "prompt_mean", -1))
        for i in range(min(len(spans["guess"]), n_guess)):
            entries.append((spans["guess"][i], "guess", i))
        for abs_pos in spans["sem_answer"]:
            entries.append((abs_pos, "sem_answer_mean", -1))
        for i in range(min(len(spans["probability"]), n_prob)):
            entries.append((spans["probability"][i], "probability", i))
        return entries

    if mode == MODE_GUESS_THEN_GUESS_PROBABILITY:
        guess_positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        all_positions = _absolute_guess_then_guess_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        prob_positions = all_positions[len(guess_positions) :]
        n_guess = _sequence_source_length(layer_means, patch_components=patch_components, source_name="guess")
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        entries: List[Tuple[int, str, int]] = []
        for i in range(min(len(guess_positions), n_guess)):
            entries.append((guess_positions[i], "guess", i))
        for i in range(min(len(prob_positions), n_prob)):
            entries.append((prob_positions[i], "probability", i))
        return entries

    if mode == MODE_PROBABILITY_VALUE:
        start_abs = _absolute_probability_value_start_position(prompt_len, decoded_tokens)
        if start_abs is None:
            return []
        seq_len = prompt_len + len(decoded_tokens)
        if start_abs >= seq_len:
            return []
        n_prob = _sequence_source_length(layer_means, patch_components=patch_components, source_name="probability")
        if n_prob <= 0:
            return []
        positions = list(range(start_abs, seq_len))
        if not positions:
            return []
        entries = [(positions[0], "probability", n_prob - 1)]
        entries.extend((abs_pos, "probability_value_mean", -1) for abs_pos in positions[1:])
        return entries

    raise ValueError(f"Unsupported ablation mode for patch planning: {mode}")


def _is_patch_active(mode: str) -> bool:
    if mode == MODE_NONE:
        return False
    if mode not in MODE_SOURCE_REQUIREMENTS:
        raise ValueError(f"Unsupported ablation mode for activation: {mode}")
    return True


def _vector_from_layer_means(
    layer_means: Dict[str, torch.Tensor],
    *,
    component: str,
    source_name: str,
    token_index: int,
) -> torch.Tensor:
    key = f"{component}_{source_name}"
    source_tensor = layer_means.get(key)
    if source_tensor is None:
        raise ValueError(f"Missing layer means key: {key}.")
    if token_index >= 0:
        if source_tensor.ndim < 2:
            raise ValueError(
                f"Layer means key {key} expected rank >= 2 for token-indexed source, got {tuple(source_tensor.shape)}."
            )
        if token_index >= int(source_tensor.shape[0]):
            raise ValueError(
                f"Token index {token_index} out of range for {key} with length {int(source_tensor.shape[0])}."
            )
        return source_tensor[token_index]
    if source_tensor.ndim == 1:
        return source_tensor
    if source_tensor.ndim == 2 and int(source_tensor.shape[0]) == 1:
        return source_tensor[0]
    raise ValueError(
        f"Layer means key {key} expected rank-1 (or [1,hidden]) for scalar source, got {tuple(source_tensor.shape)}."
    )


def build_qkv_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    mode: str,
    prompt_len: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    patch_components: Sequence[str],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    n_q_heads: int,
    n_kv_heads: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        layer_means = layer_to_means[layer]
        selected_kv_heads = [int(h) for h in selected_heads_by_layer.get(int(layer), [])]
        if not selected_kv_heads:
            raise ValueError(f"No selected KV heads for layer {layer}.")
        selected_q_heads = expand_kv_heads_to_q_heads(selected_kv_heads, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)

        def _make_component_hook(
            local_layer_means: Dict[str, torch.Tensor],
            local_layer: int,
            local_heads: List[int],
            component: str,
        ) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                decoded_tokens = decoded_tokens_provider()
                if not _is_patch_active(mode):
                    return activation
                patch_plan_entries = _build_patch_plan_entries(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_means=local_layer_means,
                    patch_components=patch_components,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
                if not patch_plan_entries:
                    return activation
                n_heads = int(activation.shape[2])
                for head_idx in local_heads:
                    if head_idx < 0 or head_idx >= n_heads:
                        raise ValueError(
                            f"{component.upper()} head index {head_idx} out of range for layer {local_layer} "
                            f"hook_{component} with {n_heads} heads."
                        )
                for abs_pos, source_name, token_idx in patch_plan_entries:
                    if not (0 <= abs_pos < activation.shape[1]):
                        continue
                    source_vector = _vector_from_layer_means(
                        local_layer_means,
                        component=component,
                        source_name=source_name,
                        token_index=token_idx,
                    )
                    replacement = _reshape_vector_for_activation(
                        source_vector, activation, local_layer, f"{component}_{source_name}"
                    )
                    for head_idx in local_heads:
                        activation[:, abs_pos, head_idx, :] = replacement[head_idx]
                return activation

            return hook_fn

        if "q" in patch_components:
            hook_q_name = f"blocks.{layer}.attn.hook_q"
            hooks.append((hook_q_name, _make_component_hook(layer_means, int(layer), selected_q_heads, "q")))
        if "k" in patch_components:
            hook_k_name = f"blocks.{layer}.attn.hook_k"
            hooks.append((hook_k_name, _make_component_hook(layer_means, int(layer), selected_kv_heads, "k")))
        if "v" in patch_components:
            hook_v_name = f"blocks.{layer}.attn.hook_v"
            hooks.append((hook_v_name, _make_component_hook(layer_means, int(layer), selected_kv_heads, "v")))
    return hooks


def greedy_generate_qkv_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    mode: str,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    patch_components: Sequence[str],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    n_q_heads: int,
    n_kv_heads: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_qkv_mean_replace_hooks(
        layer_indices=layer_indices,
        mode=mode,
        prompt_len=prompt_len,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_means=layer_to_means,
        patch_components=patch_components,
        selected_heads_by_layer=selected_heads_by_layer,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
    )
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--input_h5",
        type=str,
        default=None,
        help=(
            "Path to processed HDF5 containing responses/0 embedding fields keyed by {q,k,v}, "
            "including embeddings_probability and (for some modes) embeddings_guess, "
            "embeddings_mean_prompt, embeddings_mean_sem_answer, embeddings_mean_prob_val."
        ),
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
        "--ablate_heads_by_layer",
        type=str,
        default=None,
        help=(
            "Optional comma-separated <layer>.<head> list, e.g. '12.3,12.7,13.2'. "
            "When set, this selection takes precedence and --ablate_layers is ignored."
        ),
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=[
            MODE_NONE,
            MODE_PROBABILITY_TOKENS,
            MODE_PROBABILITY_LAST_TOKEN,
            MODE_PROBABILITY_SPAN_EXCEPT_LAST_TOKEN,
            MODE_ALL_PRE_PROBABILITY_TOKENS,
            MODE_GUESS_TOKENS,
            MODE_ALL_PRE_GUESS_TOKENS,
            MODE_GUESS_THEN_GUESS_PROBABILITY,
            MODE_PROBABILITY_VALUE,
        ],
        choices=[
            MODE_NONE,
            MODE_PROBABILITY_TOKENS,
            MODE_PROBABILITY_LAST_TOKEN,
            MODE_PROBABILITY_SPAN_EXCEPT_LAST_TOKEN,
            MODE_ALL_PRE_PROBABILITY_TOKENS,
            MODE_GUESS_TOKENS,
            MODE_ALL_PRE_GUESS_TOKENS,
            MODE_GUESS_THEN_GUESS_PROBABILITY,
            MODE_PROBABILITY_VALUE,
            MODE_ANS_ADJ_TOKENS,
        ],
    )
    parser.add_argument(
        "--patch_components",
        type=str,
        default="k,v",
        help="CSV list of attention components to patch. Allowed: q,k,v. Example: q,k,v",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--mean_from_low_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--parse_mode_verbalised_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    patch_components = parse_patch_components(args.patch_components)
    if args.expected_guess_tokens <= 0:
        raise ValueError("--expected_guess_tokens must be > 0.")
    if args.expected_probability_tokens <= 0:
        raise ValueError("--expected_probability_tokens must be > 0.")
    required_sources: Set[str] = set()
    for mode_name in args.ablation_mode:
        required_sources.update(MODE_SOURCE_REQUIREMENTS.get(mode_name, set()))
    component_mean_keys = [f"{component}_{source}" for component in patch_components for source in sorted(required_sources)]

    if not args.input_h5:
        raise ValueError("--input_h5 is required.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)

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
    model_n_q_heads = int(model.cfg.n_heads)
    model_n_kv_heads = resolve_n_key_value_heads(model.cfg, model_n_q_heads)
    ablate_layers_from_flag = parse_ablate_layers(args.ablate_layers, model_n_layers)
    if not ablate_layers_from_flag:
        raise ValueError("No layers selected via --ablate_layers.")

    if args.ablate_heads_by_layer:
        selected_heads_all_layers = parse_layer_head_pairs(
            args.ablate_heads_by_layer, n_layers=model_n_layers, n_kv_heads=model_n_kv_heads
        )
        run_layers = sorted(selected_heads_all_layers.keys())
        if not run_layers:
            raise ValueError("No layers selected via --ablate_heads_by_layer.")
        logging.info(
            "Using --ablate_heads_by_layer selection; ignoring --ablate_layers=%s.",
            args.ablate_layers,
        )
    else:
        run_layers = sorted(ablate_layers_from_flag)
        selected_heads_all_layers = {
            int(layer_idx): list(range(model_n_kv_heads)) for layer_idx in run_layers
        }
        logging.info(
            "No --ablate_heads_by_layer provided; using all KV heads across --ablate_layers=%s.",
            args.ablate_layers,
        )

    examples_h5 = load_examples_h5(Path(args.input_h5))
    component_means, low_ids, high_ids = compute_component_group_means(
        examples_h5,
        ablate_layers=run_layers,
        patch_components=patch_components,
        required_sources=required_sources,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        mean_from_low_confidence=args.mean_from_low_confidence,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
    )

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

    layer_component_means = _build_layer_component_means(
        component_means,
        ablate_layers=run_layers,
        component_mean_keys=component_mean_keys,
        device=device,
        torch_dtype=torch_dtype,
    )
    logging.info(
        (
            "Loaded %d H5 examples. low_conf=%d high_conf=%d target_group=%s target_ids=%d "
            "layers=%s patch_components=%s required_sources=%s"
        ),
        len(examples_h5),
        len(low_ids),
        len(high_ids),
        target_group,
        len(ablation_target_ids),
        run_layers,
        ",".join(patch_components),
        ",".join(sorted(required_sources)),
    )

    def run_one_evaluation(
        layer_subset: Sequence[int],
        selected_heads_subset: Dict[int, List[int]],
    ) -> Tuple[
        Dict[str, dict],
        Dict[str, dict],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, int],
    ]:
        layer_subset_means = {int(layer_idx): layer_component_means[int(layer_idx)] for layer_idx in layer_subset}
        results = {"train": {}, "validation": {}}
        mini_results = {"train": {}, "validation": {}}
        mode_confidence_values: Dict[str, List[float]] = {mode_name: [] for mode_name in args.ablation_mode}
        mode_responses_identical_true: Dict[str, int] = {
            mode_name: 0 for mode_name in args.ablation_mode if mode_name != MODE_NONE
        }

        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(
                args.num_samples * (1 - TRAIN_RATIO)
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
                    raise ValueError(f"Example id {ex_id} selected from H5 but not found in {split_name} split.")
                example = eval_ds[int(ds_idx)]
                local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + example["question"]
                entry = {"question": example["question"]}
                mini_entry = {"question": example["question"]}

                baseline_response: Optional[str] = None
                baseline_mode_confidence: Optional[float] = None
                if MODE_NONE in args.ablation_mode:
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
                    baseline_key = mode_to_output_key(MODE_NONE)
                    entry[baseline_key] = {
                        "response": baseline_response,
                        "decoded_tokens": baseline_decoded_tokens,
                    }
                    mini_entry[baseline_key] = {"response": baseline_response}
                    if args.parse_mode_verbalised_confidence:
                        entry[baseline_key]["verbalised_confidence"] = baseline_mode_confidence
                        mini_entry[baseline_key]["verbalised_confidence"] = baseline_mode_confidence
                        if baseline_mode_confidence is not None:
                            mode_confidence_values[MODE_NONE].append(float(baseline_mode_confidence))

                for mode_name in args.ablation_mode:
                    if mode_name == MODE_NONE:
                        continue
                    response, decoded_tokens = greedy_generate_qkv_ablated(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=args.model_max_new_tokens,
                        layer_indices=layer_subset,
                        mode=mode_name,
                        expected_guess_tokens=args.expected_guess_tokens,
                        expected_probability_tokens=args.expected_probability_tokens,
                        layer_to_means=layer_subset_means,
                        patch_components=patch_components,
                        selected_heads_by_layer=selected_heads_subset,
                        n_q_heads=model_n_q_heads,
                        n_kv_heads=model_n_kv_heads,
                    )
                    mode_confidence = (
                        parse_mode_confidence_from_response(response)
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    mode_key = mode_to_output_key(mode_name)
                    entry[mode_key] = {"response": response, "decoded_tokens": decoded_tokens}
                    mini_entry[mode_key] = {"response": response}
                    if args.parse_mode_verbalised_confidence:
                        entry[mode_key]["verbalised_confidence"] = mode_confidence
                        mini_entry[mode_key]["verbalised_confidence"] = mode_confidence
                        if mode_confidence is not None:
                            mode_confidence_values[mode_name].append(float(mode_confidence))

                    if baseline_response is not None:
                        responses_identical = response == baseline_response
                        entry[mode_key]["responses_identical"] = responses_identical
                        mini_entry[mode_key]["responses_identical"] = responses_identical
                        if responses_identical:
                            mode_responses_identical_true[mode_name] += 1
                        if args.parse_mode_verbalised_confidence:
                            if mode_confidence is None or baseline_mode_confidence is None:
                                meets_none_confidence_direction = None
                            elif args.mean_from_low_confidence:
                                meets_none_confidence_direction = mode_confidence > baseline_mode_confidence
                            else:
                                meets_none_confidence_direction = mode_confidence < baseline_mode_confidence
                            entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                            mini_entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction

                    logging.info(
                        "[%s %d/%d] %s %s first line: %r",
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                        mode_key,
                        response[:120],
                    )
                results[split_name][ex_id] = entry
                mini_results[split_name][ex_id] = mini_entry

        mode_confidence_means: Dict[str, Optional[float]] = {}
        mode_confidence_counts: Dict[str, int] = {}
        for mode_name in args.ablation_mode:
            vals = mode_confidence_values[mode_name]
            mode_confidence_means[mode_name] = float(np.mean(vals)) if vals else None
            mode_confidence_counts[mode_name] = len(vals)
        return (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            mode_responses_identical_true,
        )
    selected_subset = {layer: selected_heads_all_layers[layer] for layer in run_layers if layer in selected_heads_all_layers}
    missing_layer_heads = [layer for layer in run_layers if layer not in selected_subset]
    if missing_layer_heads:
        raise ValueError(
            "No selected heads provided for layers in this run: "
            + ",".join(str(layer) for layer in missing_layer_heads)
        )
    (
        results,
        mini_results,
        mode_confidence_means,
        mode_confidence_counts,
        mode_responses_identical_true,
    ) = run_one_evaluation(run_layers, selected_subset)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(mini_output_json_path(out_path), "w", encoding="utf-8") as f:
        json.dump(mini_results, f, ensure_ascii=False, indent=2)
    write_config_txt(
        config_txt_path(out_path),
        args=args,
        device=device,
        model_n_layers=model_n_layers,
        n_q_heads=model_n_q_heads,
        n_kv_heads=model_n_kv_heads,
        patch_components=patch_components,
        ablate_layers=run_layers,
        selected_layer_heads=selected_subset,
        prompt_indices=prompt_indices,
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        mode_confidence_means=mode_confidence_means,
        mode_confidence_counts=mode_confidence_counts,
        mode_responses_identical_true=mode_responses_identical_true,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    logging.info("Saved full outputs to %s", out_path)


if __name__ == "__main__":
    main()

