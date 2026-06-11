#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Greedy decoding on TriviaQA with subblock-aware mass mean-direction probing.

This script extends mass mean probing from residual stream to transformer
subblocks (`attn` / `mlp`). For each selected layer, it computes:
  - low-confidence mean hidden states
  - high-confidence mean hidden states
  - direction = high_mean - low_mean

Directions are computed per subblock, per span, and per token position where
applicable (guess/probability spans). During generation, additive steering is
applied to either attn-out or mlp-out hooks using the same dynamic span modes
as the original mass mean probe.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_zero_ablation.run_blockwise_zero_ablation import SUBBLOCK_TO_HOOK
from mass_mean_probe.run_mass_mean_probe import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT_LINGUISTIC,
    CONFIDENCE_PROMPT_NUMERIC,
    _absolute_all_pre_guess_positions,
    _absolute_guess_span_positions,
    _absolute_pre_probability_positions,
    _absolute_prob_positions,
    _absolute_prob_positions_at_row_indices,
    _as_layer_hidden,
    _dedupe_preserve_order,
    _format_alpha,
    _generation_contains_stop,
    _is_expected_or_plus_one,
    _postprocess_response_from_full_decode,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    load_examples_h5,
    load_hooked_transformer,
    load_trivia_qa,
    mode_to_output_key,
    parse_ablate_layers,
    parse_mode_confidence_from_response,
    PROBABILITY_ROW_INDEX_MODES,
    split_answerable_indices,
)


TRAIN_RATIO = 0.9
TARGET_COMPONENTS = ("attn", "mlp")
REQUIRED_COMPONENTS = ("res", "attn", "mlp")
REQUIRED_EMBEDDING_FIELDS = (
    "embeddings_mean_prompt",
    "embeddings_guess",
    "embeddings_mean_sem_answer",
    "embeddings_probability",
)


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("subblock_mass_mean_probe") / "results"
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


def summary_json_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "mode_confidence_summary.json")


def _validate_component_field(resp0: dict, ex_id: str, field_name: str, component: str):
    field = resp0.get(field_name)
    if not isinstance(field, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be an object with keys {REQUIRED_COMPONENTS}."
        )
    if component not in field:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} missing component '{component}'. "
            f"Expected keys include {REQUIRED_COMPONENTS}."
        )
    value = field.get(component)
    if value is None:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name}/{component} is null. "
            "Input H5 must contain populated res/attn/mlp subfields."
        )
    return value


def compute_low_high_span_means_and_directions_by_component(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
    components: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    set[str],
    set[str],
]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    layer_indices = np.asarray(ablate_layers)

    low_vectors: Dict[str, Dict[str, List[np.ndarray]]] = {
        component: {"prompt_mean": [], "guess": [], "sem_answer_mean": [], "probability": []}
        for component in components
    }
    high_vectors: Dict[str, Dict[str, List[np.ndarray]]] = {
        component: {"prompt_mean": [], "guess": [], "sem_answer_mean": [], "probability": []}
        for component in components
    }

    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError(
                f"Example {ex_id} must have exactly one response, got {0 if responses is None else len(responses)}."
            )
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 is not a dict.")

        for field_name in REQUIRED_EMBEDDING_FIELDS:
            for component in REQUIRED_COMPONENTS:
                _validate_component_field(resp0, ex_id, field_name, component)

        conf = float(resp0.get("verbalised_confidence"))
        is_low = conf <= low_conf_threshold
        is_high = conf >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        if not (is_low or is_high):
            continue

        for component in components:
            emb_prompt = _validate_component_field(resp0, ex_id, "embeddings_mean_prompt", component)
            emb_guess = _validate_component_field(resp0, ex_id, "embeddings_guess", component)
            emb_sem_answer = _validate_component_field(resp0, ex_id, "embeddings_mean_sem_answer", component)
            emb_prob = _validate_component_field(resp0, ex_id, "embeddings_probability", component)

            if not isinstance(emb_guess, list):
                raise ValueError(
                    f"Example {ex_id} responses/0/embeddings_guess/{component} must be a list."
                )
            if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
                raise ValueError(
                    f"Example {ex_id} embeddings_guess/{component} len={len(emb_guess)}; expected "
                    f"{expected_guess_tokens} or {expected_guess_tokens + 1}."
                )
            emb_guess = emb_guess[:expected_guess_tokens]

            if not isinstance(emb_prob, list):
                raise ValueError(
                    f"Example {ex_id} responses/0/embeddings_probability/{component} must be a list."
                )
            if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
                raise ValueError(
                    f"Example {ex_id} embeddings_probability/{component} len={len(emb_prob)}; expected "
                    f"{expected_probability_tokens} or {expected_probability_tokens + 1}."
                )
            emb_prob = emb_prob[:expected_probability_tokens]

            prompt_selected = _as_layer_hidden(emb_prompt)[layer_indices, :]
            sem_answer_selected = _as_layer_hidden(emb_sem_answer)[layer_indices, :]
            guess_selected = np.stack([_as_layer_hidden(tok_arr)[layer_indices, :] for tok_arr in emb_guess], axis=1)
            prob_selected = np.stack([_as_layer_hidden(tok_arr)[layer_indices, :] for tok_arr in emb_prob], axis=1)

            if is_low:
                low_vectors[component]["prompt_mean"].append(prompt_selected)
                low_vectors[component]["guess"].append(guess_selected)
                low_vectors[component]["sem_answer_mean"].append(sem_answer_selected)
                low_vectors[component]["probability"].append(prob_selected)
            if is_high:
                high_vectors[component]["prompt_mean"].append(prompt_selected)
                high_vectors[component]["guess"].append(guess_selected)
                high_vectors[component]["sem_answer_mean"].append(sem_answer_selected)
                high_vectors[component]["probability"].append(prob_selected)

    for component in components:
        if not low_vectors[component]["probability"]:
            raise ValueError(
                f"No low-confidence examples found at threshold <= {low_conf_threshold} for component={component}."
            )
        if not high_vectors[component]["probability"]:
            raise ValueError(
                f"No high-confidence examples found at threshold >= {high_conf_threshold} for component={component}."
            )

    mean_low_by_component: Dict[str, Dict[str, np.ndarray]] = {component: {} for component in components}
    mean_high_by_component: Dict[str, Dict[str, np.ndarray]] = {component: {} for component in components}
    direction_by_component: Dict[str, Dict[str, np.ndarray]] = {component: {} for component in components}

    for component in components:
        for span in ("prompt_mean", "guess", "sem_answer_mean", "probability"):
            mean_low = np.mean(np.stack(low_vectors[component][span], axis=0), axis=0).astype(np.float32)
            mean_high = np.mean(np.stack(high_vectors[component][span], axis=0), axis=0).astype(np.float32)
            direction = (mean_high - mean_low).astype(np.float32)
            mean_low_by_component[component][span] = mean_low
            mean_high_by_component[component][span] = mean_high
            direction_by_component[component][span] = direction

    return mean_low_by_component, mean_high_by_component, direction_by_component, low_ids, high_ids


def normalize_component_direction_spans_to_unit_norm_budget(
    direction_by_component: Dict[str, Dict[str, np.ndarray]],
    *,
    components: Sequence[str],
    spans: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for component in components:
        stats[component] = {}
        for span in spans:
            if span not in direction_by_component[component]:
                raise ValueError(f"Cannot normalize missing span direction: component={component} span={span!r}")
            span_direction = direction_by_component[component][span]
            if span_direction.ndim != 3:
                raise ValueError(
                    f"Expected direction[{component!r}][{span!r}] to have shape "
                    "(layers, token_positions, d_model), "
                    f"got ndim={span_direction.ndim} shape={span_direction.shape}."
                )

            num_layers, num_token_positions, _ = span_direction.shape
            target_sum = float(num_layers * num_token_positions)
            sum_before = float(np.sum(np.linalg.norm(span_direction, axis=-1)))

            if sum_before <= 0.0:
                logging.warning(
                    "Skipping direction normalization for component=%s span=%s because sum of norms is zero.",
                    component,
                    span,
                )
                scale = 1.0
            else:
                scale = target_sum / sum_before
                direction_by_component[component][span] = (span_direction * scale).astype(np.float32, copy=False)
            sum_after = float(np.sum(np.linalg.norm(direction_by_component[component][span], axis=-1)))
            stats[component][span] = {
                "target_sum": target_sum,
                "sum_before": sum_before,
                "sum_after": sum_after,
                "scale": float(scale),
            }
    return stats


def _direction_mode_activation_applier_builder(
    mode: str,
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
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
            row_indices = PROBABILITY_ROW_INDEX_MODES[mode]

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

        raise ValueError(f"Unknown ablation mode for perturb hooks: {mode!r}")

    return _builder


def build_subblock_direction_perturb_hooks(
    layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]],
    *,
    subblock: str,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    activation_applier_builder = _direction_mode_activation_applier_builder(
        mode,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    activation_applier = activation_applier_builder(prompt_len, decoded_tokens_provider)
    hook_suffix = SUBBLOCK_TO_HOOK[subblock]

    for layer in layer_to_span_delta:
        hook_name = f"blocks.{layer}.{hook_suffix}"

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                layer_delta = layer_to_span_delta[layer_idx]
                return activation_applier(activation, layer_delta)

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_direction_perturbed_subblock(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]],
    subblock: str,
    mode: str,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_subblock_direction_perturb_hooks(
        layer_to_span_delta=layer_to_span_delta,
        subblock=subblock,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        linguistic_confidence_prompt=linguistic_confidence_prompt,
    )
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            completion_str = "".join(decoded_tokens)
            if _generation_contains_stop(completion_str):
                break
            if eos_id is not None and next_id == eos_id:
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def write_layer_direction_pickles(
    *,
    output_json_path: str,
    ablate_layers: Sequence[int],
    direction_by_component: Dict[str, Dict[str, np.ndarray]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    components: Sequence[str],
) -> None:
    out_dir = os.path.dirname(output_json_path)
    for layer_i, layer_idx in enumerate(ablate_layers):
        component_payload = {}
        for component in components:
            component_payload[component] = {
                "prompt_mean_direction": direction_by_component[component]["prompt_mean"][layer_i],
                "guess_prefix_directions": [
                    direction_by_component[component]["guess"][layer_i, tok_i]
                    for tok_i in range(expected_guess_tokens)
                ],
                "semantic_answer_mean_direction": direction_by_component[component]["sem_answer_mean"][layer_i],
                "probability_prefix_directions": [
                    direction_by_component[component]["probability"][layer_i, tok_i]
                    for tok_i in range(expected_probability_tokens)
                ],
            }
        payload = {
            "layer_idx": int(layer_idx),
            "expected_guess_tokens": int(expected_guess_tokens),
            "expected_probability_tokens": int(expected_probability_tokens),
            "components": component_payload,
        }
        pickle_path = os.path.join(out_dir, f"layer_{layer_idx}_directions.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(payload, f)
        logging.info("Wrote %s", pickle_path)


def _build_summary_json(
    *,
    non_none_modes: Sequence[str],
    components: Sequence[str],
    ablation_targets: Sequence[str],
    alphas: Sequence[float],
    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
    mode_responses_identical_true: Dict[str, Dict[str, Dict[str, Dict[float, int]]]],
    baseline_values_by_target: Dict[str, List[float]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for mode in non_none_modes:
        mode_payload: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float] | int]]]] = {}
        for component in components:
            component_payload: Dict[str, Dict[str, Dict[str, Optional[float] | int]]] = {}
            for target in ("low", "high"):
                if target not in ablation_targets:
                    continue
                alpha_payload: Dict[str, Dict[str, Optional[float] | int]] = {}
                for alpha in sorted(alphas):
                    values = mode_confidence_values.get(mode, {}).get(component, {}).get(target, {}).get(alpha, [])
                    alpha_key = _format_alpha(alpha)
                    alpha_payload[alpha_key] = {
                        "alpha_value": float(alpha),
                        "mean_confidence": float(np.mean(values)) if values else None,
                        "sample_count": len(values),
                        "responses_identical_count": int(
                            mode_responses_identical_true.get(mode, {})
                            .get(component, {})
                            .get(target, {})
                            .get(alpha, 0)
                        ),
                    }
                component_payload[target] = alpha_payload
            mode_payload[component] = component_payload
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


def _plot_component_mode_confidence(
    *,
    mode_name: str,
    component: str,
    component_payload: Dict[str, Dict[str, Dict[str, object]]],
    baseline_payload: Dict[str, Dict[str, object]],
    ablation_targets: Sequence[str],
    output_path: str,
) -> None:
    target_colors = {"high": "tab:blue", "low": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for target in ("high", "low"):
        if target not in ablation_targets:
            continue
        alpha_map = component_payload.get(target, {})
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
        points.sort(key=lambda x: x[0])
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=target_colors[target], label=f"target={target}")
            for x_val, y_val, count in points:
                ax.annotate(f"n={count}", (x_val, y_val), textcoords="offset points", xytext=(0, 7), ha="center")

        baseline_mean = baseline_payload.get(target, {}).get("mean_confidence")
        if baseline_mean is not None:
            ax.axhline(
                float(baseline_mean),
                linestyle="--",
                linewidth=1.1,
                color=target_colors[target],
                alpha=0.35,
            )

    ax.set_xlabel("alpha")
    ax.set_ylabel("mean verbalised confidence")
    ax.set_title(f"{mode_name} ({component}): confidence vs alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary_plots_from_json(
    *,
    summary_payload: Dict[str, object],
    components: Sequence[str],
    ablation_targets: Sequence[str],
    output_dir: str,
) -> None:
    baseline_payload = summary_payload.get("no_replacement_baseline", {})
    for mode_name, mode_payload in summary_payload.items():
        if mode_name == "no_replacement_baseline":
            continue
        if not isinstance(mode_payload, dict):
            continue
        for component in components:
            component_payload = mode_payload.get(component, {})
            if not isinstance(component_payload, dict):
                continue
            png_name = f"{mode_to_output_key(mode_name)}__{component}_mean_confidence_vs_alpha.png"
            output_path = os.path.join(output_dir, png_name)
            _plot_component_mode_confidence(
                mode_name=mode_name,
                component=component,
                component_payload=component_payload,
                baseline_payload=baseline_payload,
                ablation_targets=ablation_targets,
                output_path=output_path,
            )
            logging.info("Wrote %s", output_path)


def _flatten_summary_means_for_config(
    *,
    non_none_modes: Sequence[str],
    components: Sequence[str],
    ablation_targets: Sequence[str],
    alphas: Sequence[float],
    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for mode in non_none_modes:
        for component in components:
            for target in ablation_targets:
                for alpha in alphas:
                    vals = mode_confidence_values[mode][component][target][float(alpha)]
                    k = (
                        f"{mode}__{component}__target_{target}__alpha_{_format_alpha(alpha)}"
                        "__mean_verbalised_confidence"
                    )
                    out[k] = float(np.mean(vals)) if vals else None
    return out


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
    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
    finished_at: str,
) -> None:
    non_none_modes = [m for m in args.ablation_mode if m != "none"]
    flattened_means = _flatten_summary_means_for_config(
        non_none_modes=non_none_modes,
        components=args.ablate_subblocks,
        ablation_targets=args.ablation_targets,
        alphas=args.alpha,
        mode_confidence_values=mode_confidence_values,
    )
    lines = [
        "Subblock Mass Mean Probe Configuration",
        "======================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        "",
        "[Data]",
        f"input_h5={args.input_h5}",
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
        f"ablation_targets={args.ablation_targets}",
        f"ablate_subblocks={args.ablate_subblocks}",
        f"alpha={args.alpha}",
        "non_none_mode_behavior=additive_direction_perturbation",
        "direction_definition=high_mean_minus_low_mean",
        "confidence_direction_expectation_for_low_targets=perturbed_confidence_gt_none",
        "confidence_direction_expectation_for_high_targets=perturbed_confidence_lt_none",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
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
        "",
        "[Mode Confidence Metrics]",
    ]
    for key in sorted(flattened_means):
        mean_val = flattened_means[key]
        if mean_val is None:
            lines.append(f"{key}=None")
        else:
            lines.append(f"{key}={mean_val:.6f}")

    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def greedy_generate(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks or [])
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            completion_str = "".join(decoded_tokens)
            if _generation_contains_stop(completion_str):
                break
            if eos_id is not None and next_id == eos_id:
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subblock-aware mass mean direction probe inference (TransformerLens)."
    )
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must be true for subblock probing; requires res/attn/mlp subfields in H5.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
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
        help="Inclusive range '12-15' or comma list '12,13,14,15' (zero-indexed).",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=[
            "none",
            "probability_tokens_mean_replace",
            "probability_first_token_mean_replace",
            "probability_first_two_tokens_mean_replace",
            "probability_first_two_and_index6_tokens_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
        ],
        choices=[
            "none",
            "probability_tokens_mean_replace",
            "probability_first_token_mean_replace",
            "probability_first_two_tokens_mean_replace",
            "probability_first_two_and_index6_tokens_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
        ],
    )
    parser.add_argument(
        "--ablate_subblocks",
        type=str,
        nargs="+",
        required=True,
        choices=["attn", "mlp"],
        help="Subblocks to steer independently.",
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
            "subblock_mass_mean_probe/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    args = parser.parse_args()

    if not args.new_h5_format:
        raise ValueError(
            "Subblock-aware mass-mean requires --new_h5_format because it needs attn/mlp embedding subfields."
        )
    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline comparisons.")
    if len(set(args.ablate_subblocks)) != len(args.ablate_subblocks):
        raise ValueError(f"Duplicate subblocks are not allowed: {args.ablate_subblocks}")
    if args.normalize_span_directions:
        normalization_supported_modes = {
            "none",
            "probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
        } | set(PROBABILITY_ROW_INDEX_MODES)
        unsupported_modes = [mode for mode in args.ablation_mode if mode not in normalization_supported_modes]
        if unsupported_modes:
            raise ValueError(
                "normalize_span_directions only supports ablation modes "
                f"{sorted(normalization_supported_modes)}. Unsupported: {unsupported_modes}."
            )

    args.ablation_targets = _dedupe_preserve_order(args.ablation_targets)
    args.ablate_subblocks = _dedupe_preserve_order(args.ablate_subblocks)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    alphas_outside_unit = [a for a in args.alpha if a < 0.0 or a > 1.0]
    if alphas_outside_unit:
        logging.warning("Alpha values outside [0, 1] will be used as-is: %s", alphas_outside_unit)

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

    confidence_prompt = (
        CONFIDENCE_PROMPT_LINGUISTIC if args.linguistic_confidence_prompt else CONFIDENCE_PROMPT_NUMERIC
    )

    logging.info("Loading HookedTransformer: %s", args.model_name)
    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers)

    examples_h5 = load_examples_h5(Path(args.input_h5))
    _, _, direction_by_component, low_ids, high_ids = compute_low_high_span_means_and_directions_by_component(
        examples_h5,
        ablate_layers=ablate_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_guess_tokens=args.expected_guess_tokens,
        components=args.ablate_subblocks,
    )
    if args.normalize_span_directions:
        normalization_stats = normalize_component_direction_spans_to_unit_norm_budget(
            direction_by_component,
            components=args.ablate_subblocks,
            spans=("guess", "probability"),
        )
        for component, comp_stats in normalization_stats.items():
            for span_name, span_stats in comp_stats.items():
                logging.info(
                    "Normalized component=%s span=%s direction norms: before=%.6f after=%.6f target=%.6f scale=%.6f",
                    component,
                    span_name,
                    span_stats["sum_before"],
                    span_stats["sum_after"],
                    span_stats["target_sum"],
                    span_stats["scale"],
                )

    logging.info(
        "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), layers=%s, components=%s, alphas=%s",
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        ablate_layers,
        args.ablate_subblocks,
        list(args.alpha),
    )

    out_path = resolve_output_json_path(args.output_json)
    results = {"train": {}, "validation": {}}
    mini_results = {"train": {}, "validation": {}}

    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]] = {
        mode: {
            component: {target: {float(alpha): [] for alpha in args.alpha} for target in args.ablation_targets}
            for component in args.ablate_subblocks
        }
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    mode_responses_identical_true: Dict[str, Dict[str, Dict[str, Dict[float, int]]]] = {
        mode: {
            component: {target: {float(alpha): 0 for alpha in args.alpha} for target in args.ablation_targets}
            for component in args.ablate_subblocks
        }
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    baseline_values_by_target: Dict[str, List[float]] = {"low": [], "high": []}
    non_none_modes = [m for m in args.ablation_mode if m != "none"]

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        split_target = (
            round(args.num_samples * TRAIN_RATIO)
            if split_name == "train"
            else round(args.num_samples * (1 - TRAIN_RATIO))
        )
        id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
        target_union_ids: set[str] = set(low_ids) | set(high_ids)
        split_target_ids = sorted(ex_id for ex_id in target_union_ids if ex_id in id_to_index)
        selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
        logging.info("Generating for %d examples (%s split).", len(selected_ids), split_name)

        for i, ex_id in enumerate(selected_ids):
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                continue
            example = eval_ds[int(ds_idx)]
            local_prompt = fewshot_prefix + confidence_prompt + example["question"]
            entry = {"question": example["question"]}
            mini_entry = {"question": example["question"]}

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

            ex_is_low = ex_id in low_ids
            ex_is_high = ex_id in high_ids
            if args.parse_mode_verbalised_confidence and baseline_confidence is not None:
                if ex_is_low:
                    baseline_values_by_target["low"].append(float(baseline_confidence))
                if ex_is_high:
                    baseline_values_by_target["high"].append(float(baseline_confidence))

            for target in args.ablation_targets:
                if target == "low" and not ex_is_low:
                    continue
                if target == "high" and not ex_is_high:
                    continue
                sign = 1.0 if target == "low" else -1.0
                for mode in non_none_modes:
                    for alpha in args.alpha:
                        key = f"{mode_to_output_key(mode)}__target_{target}__alpha_{_format_alpha(float(alpha))}"
                        if key not in entry:
                            entry[key] = {}
                            mini_entry[key] = {}
                        for subblock in args.ablate_subblocks:
                            layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]] = {}
                            for layer_i, layer_idx in enumerate(ablate_layers):
                                component_direction = direction_by_component[subblock]
                                layer_to_span_delta[layer_idx] = {
                                    "prompt_mean": torch.tensor(
                                        sign * alpha * component_direction["prompt_mean"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                    "guess": torch.tensor(
                                        sign * alpha * component_direction["guess"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                    "sem_answer_mean": torch.tensor(
                                        sign * alpha * component_direction["sem_answer_mean"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                    "probability": torch.tensor(
                                        sign * alpha * component_direction["probability"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                }

                            response, decoded_tokens = greedy_generate_direction_perturbed_subblock(
                                model=model,
                                local_prompt=local_prompt,
                                max_new_tokens=args.model_max_new_tokens,
                                layer_to_span_delta=layer_to_span_delta,
                                subblock=subblock,
                                mode=mode,
                                expected_guess_tokens=args.expected_guess_tokens,
                                expected_probability_tokens=args.expected_probability_tokens,
                                expected_confidence_tokens=args.expected_confidence_tokens,
                                linguistic_confidence_prompt=args.linguistic_confidence_prompt,
                            )
                            mode_confidence = (
                                parse_mode_confidence_from_response(
                                    response, linguistic_prompt=args.linguistic_confidence_prompt
                                )
                                if args.parse_mode_verbalised_confidence
                                else None
                            )

                            entry[key][subblock] = {"response": response, "decoded_tokens": decoded_tokens}
                            mini_entry[key][subblock] = {"response": response}
                            responses_identical = response == baseline_response
                            entry[key][subblock]["responses_identical"] = responses_identical
                            mini_entry[key][subblock]["responses_identical"] = responses_identical

                            if args.parse_mode_verbalised_confidence:
                                entry[key][subblock]["verbalised_confidence"] = mode_confidence
                                mini_entry[key][subblock]["verbalised_confidence"] = mode_confidence
                                if mode_confidence is not None:
                                    mode_confidence_values[mode][subblock][target][float(alpha)].append(
                                        float(mode_confidence)
                                    )
                                if responses_identical:
                                    mode_responses_identical_true[mode][subblock][target][float(alpha)] += 1
                                if mode_confidence is None or baseline_confidence is None:
                                    meets_none_confidence_direction = None
                                elif target == "low":
                                    meets_none_confidence_direction = mode_confidence > baseline_confidence
                                else:
                                    meets_none_confidence_direction = mode_confidence < baseline_confidence
                                entry[key][subblock]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )
                                mini_entry[key][subblock]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )

            results[split_name][ex_id] = entry
            mini_results[split_name][ex_id] = mini_entry
            logging.info(
                "[%s %d/%d] %s first line: %r",
                split_name,
                i + 1,
                len(selected_ids),
                ex_id,
                baseline_response[:120],
            )

    summary_payload = _build_summary_json(
        non_none_modes=non_none_modes,
        components=args.ablate_subblocks,
        ablation_targets=args.ablation_targets,
        alphas=args.alpha,
        mode_confidence_values=mode_confidence_values,
        mode_responses_identical_true=mode_responses_identical_true,
        baseline_values_by_target=baseline_values_by_target,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    mini_out_path = mini_output_json_path(out_path)
    with open(mini_out_path, "w", encoding="utf-8") as f:
        json.dump(mini_results, f, ensure_ascii=False, indent=2)
    summary_out_path = summary_json_path(out_path)
    write_summary_json(summary_out_path, summary_payload)
    write_summary_plots_from_json(
        summary_payload=summary_payload,
        components=args.ablate_subblocks,
        ablation_targets=args.ablation_targets,
        output_dir=os.path.dirname(out_path),
    )
    write_layer_direction_pickles(
        output_json_path=out_path,
        ablate_layers=ablate_layers,
        direction_by_component=direction_by_component,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
        components=args.ablate_subblocks,
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
        mode_confidence_values=mode_confidence_values,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    logging.info("Wrote %s", out_path)
    logging.info("Wrote %s", mini_out_path)
    logging.info("Wrote %s", summary_out_path)
    logging.info("Wrote %s", config_out_path)


if __name__ == "__main__":
    main()
