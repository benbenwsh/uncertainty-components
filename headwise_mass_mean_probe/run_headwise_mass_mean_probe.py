#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Greedy decoding on TriviaQA with headwise mass mean-direction probing.

This script mirrors subblock mass-mean probing, but applies additive steering at
attention head activations inside ``hook_z`` (pre-W_O). It supports:
  - layer-head steering (selected via --ablate_heads_by_layer)
  - whole-concat steering (selected via --whole_concat_mode)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mass_mean_probe.run_mass_mean_probe import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT_LINGUISTIC,
    CONFIDENCE_PROMPT_NUMERIC,
    _absolute_all_pre_guess_positions,
    _absolute_guess_span_positions,
    _absolute_pre_probability_positions,
    _absolute_prob_positions,
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
    split_answerable_indices,
)


TRAIN_RATIO = 0.9
REQUIRED_EMBEDDING_FIELDS = (
    "embeddings_mean_prompt",
    "embeddings_guess",
    "embeddings_mean_sem_answer",
    "embeddings_probability",
)
ABLATION_MODES_DEFAULT = [
    "none",
    "probability_tokens_mean_replace",
    "all_pre_probability_tokens_mean_replace",
    "guess_tokens_mean_replace",
    "all_pre_guess_tokens_mean_replace",
    "guess_then_guess_probability_mean_replace",
]


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("headwise_mass_mean_probe") / "results"
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


def parse_layer_head_pairs(spec: str, *, n_layers: int, n_heads: int) -> Dict[int, List[int]]:
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
        if head_idx < 0 or head_idx >= n_heads:
            raise ValueError(f"Head index {head_idx} out of range [0, {n_heads}).")
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


def _reshape_layer_hidden_to_heads(arr_like: np.ndarray, *, n_heads: int, d_head: int) -> np.ndarray:
    arr = _as_layer_hidden(arr_like)
    if arr.ndim != 2:
        raise ValueError(f"Expected [layers, d_model] after _as_layer_hidden, got shape {arr.shape}.")
    if arr.shape[-1] != n_heads * d_head:
        raise ValueError(
            f"Hidden dim {arr.shape[-1]} does not match n_heads*d_head={n_heads*d_head}."
        )
    return arr.reshape(arr.shape[0], n_heads, d_head)


def _init_span_buckets() -> Dict[str, List[np.ndarray]]:
    return {"prompt_mean": [], "guess": [], "sem_answer_mean": [], "probability": []}


def compute_low_high_span_means_and_directions_concat(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    low_conf_threshold: float,
    high_conf_threshold: float,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
    whole_concat_mode: bool,
    n_heads: Optional[int],
    d_head: Optional[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], set[str], set[str]]:
    if not whole_concat_mode and (n_heads is None or d_head is None):
        raise ValueError("n_heads and d_head are required for headwise direction computation.")

    low_ids: set[str] = set()
    high_ids: set[str] = set()
    layer_indices = np.asarray(ablate_layers)

    low_vectors = _init_span_buckets()
    high_vectors = _init_span_buckets()

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
            _validate_concat_field(resp0, ex_id, field_name)

        conf = float(resp0.get("verbalised_confidence"))
        is_low = conf <= low_conf_threshold
        is_high = conf >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        if not (is_low or is_high):
            continue

        emb_prompt = _validate_concat_field(resp0, ex_id, "embeddings_mean_prompt")
        emb_guess = _validate_concat_field(resp0, ex_id, "embeddings_guess")
        emb_sem_answer = _validate_concat_field(resp0, ex_id, "embeddings_mean_sem_answer")
        emb_prob = _validate_concat_field(resp0, ex_id, "embeddings_probability")

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

        if whole_concat_mode:
            prompt_selected = _as_layer_hidden(emb_prompt)[layer_indices, :]
            sem_answer_selected = _as_layer_hidden(emb_sem_answer)[layer_indices, :]
            guess_selected = np.stack(
                [_as_layer_hidden(tok_arr)[layer_indices, :] for tok_arr in emb_guess], axis=1
            )  # [L, T_guess, d_model]
            prob_selected = np.stack(
                [_as_layer_hidden(tok_arr)[layer_indices, :] for tok_arr in emb_prob], axis=1
            )  # [L, T_prob, d_model]
        else:
            prompt_selected = _reshape_layer_hidden_to_heads(
                emb_prompt, n_heads=int(n_heads), d_head=int(d_head)
            )[layer_indices, :, :]
            sem_answer_selected = _reshape_layer_hidden_to_heads(
                emb_sem_answer, n_heads=int(n_heads), d_head=int(d_head)
            )[layer_indices, :, :]
            guess_selected = np.stack(
                [
                    _reshape_layer_hidden_to_heads(tok_arr, n_heads=int(n_heads), d_head=int(d_head))[
                        layer_indices, :, :
                    ]
                    for tok_arr in emb_guess
                ],
                axis=1,
            )  # [L, T_guess, H, D]
            prob_selected = np.stack(
                [
                    _reshape_layer_hidden_to_heads(tok_arr, n_heads=int(n_heads), d_head=int(d_head))[
                        layer_indices, :, :
                    ]
                    for tok_arr in emb_prob
                ],
                axis=1,
            )  # [L, T_prob, H, D]

        if is_low:
            low_vectors["prompt_mean"].append(prompt_selected)
            low_vectors["guess"].append(guess_selected)
            low_vectors["sem_answer_mean"].append(sem_answer_selected)
            low_vectors["probability"].append(prob_selected)
        if is_high:
            high_vectors["prompt_mean"].append(prompt_selected)
            high_vectors["guess"].append(guess_selected)
            high_vectors["sem_answer_mean"].append(sem_answer_selected)
            high_vectors["probability"].append(prob_selected)

    if not low_vectors["probability"]:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_vectors["probability"]:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")

    mean_low: Dict[str, np.ndarray] = {}
    mean_high: Dict[str, np.ndarray] = {}
    direction: Dict[str, np.ndarray] = {}
    for span in ("prompt_mean", "guess", "sem_answer_mean", "probability"):
        low_mean = np.mean(np.stack(low_vectors[span], axis=0), axis=0).astype(np.float32)
        high_mean = np.mean(np.stack(high_vectors[span], axis=0), axis=0).astype(np.float32)
        mean_low[span] = low_mean
        mean_high[span] = high_mean
        direction[span] = (high_mean - low_mean).astype(np.float32)

    return mean_low, mean_high, direction, low_ids, high_ids


def normalize_direction_spans_to_unit_norm_budget(
    direction_by_span: Dict[str, np.ndarray],
    *,
    spans: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for span in spans:
        if span not in direction_by_span:
            raise ValueError(f"Cannot normalize missing span direction: span={span!r}")
        span_direction = direction_by_span[span]
        if span_direction.ndim == 4:
            num_layers, num_token_positions, num_heads, _ = span_direction.shape
            target_sum = float(num_layers * num_token_positions * num_heads)
        elif span_direction.ndim == 3:
            num_layers, num_token_positions, _ = span_direction.shape
            target_sum = float(num_layers * num_token_positions)
        else:
            raise ValueError(
                f"Expected direction[{span!r}] to have shape (layers, tokens, d) or "
                f"(layers, tokens, heads, d_head), got shape={span_direction.shape}."
            )

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


def _positions_and_vectors_for_mode(
    mode: str,
    *,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_delta: Dict[str, torch.Tensor],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
) -> Tuple[List[int], List[torch.Tensor]]:
    prompt_vec = layer_delta["prompt_mean"]
    guess_vecs = layer_delta["guess"]
    sem_answer_vec = layer_delta["sem_answer_mean"]
    prob_vecs = layer_delta["probability"]

    if mode == "probability_tokens_mean_replace":
        prob_positions = _absolute_prob_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=expected_probability_tokens,
            expected_confidence_tokens=expected_confidence_tokens,
            linguistic_confidence_prompt=linguistic_confidence_prompt,
        )
        if not prob_positions:
            return [], []
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
        return prob_positions, [prob_use[i] for i in range(n_pos)]

    if mode == "guess_tokens_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        if not guess_positions:
            return [], []
        if len(guess_positions) != guess_vecs.shape[0]:
            raise ValueError(f"Expected {guess_vecs.shape[0]} guess tokens, got {len(guess_positions)}.")
        return guess_positions, [guess_vecs[i] for i in range(len(guess_positions))]

    if mode == "all_pre_probability_tokens_mean_replace":
        positions = _absolute_pre_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
            expected_confidence_tokens=expected_confidence_tokens,
            linguistic_confidence_prompt=linguistic_confidence_prompt,
        )
        if positions is None:
            return [], []

        out_positions: List[int] = []
        out_vectors: List[torch.Tensor] = []
        for abs_pos in positions["prompt"]:
            out_positions.append(abs_pos)
            out_vectors.append(prompt_vec)
        if len(positions["guess"]) != guess_vecs.shape[0]:
            raise ValueError(f"Expected {guess_vecs.shape[0]} guess tokens, got {len(positions['guess'])}.")
        for i, abs_pos in enumerate(positions["guess"]):
            out_positions.append(abs_pos)
            out_vectors.append(guess_vecs[i])
        for abs_pos in positions["sem_answer"]:
            out_positions.append(abs_pos)
            out_vectors.append(sem_answer_vec)
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
                raise ValueError(f"Expected {prob_vecs.shape[0]} probability tokens, got {n_prob}.")
            prob_use = prob_vecs
        for i, abs_pos in enumerate(positions["probability"]):
            out_positions.append(abs_pos)
            out_vectors.append(prob_use[i])
        return out_positions, out_vectors

    if mode == "all_pre_guess_tokens_mean_replace":
        all_pre_guess_positions = _absolute_all_pre_guess_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        if not all_pre_guess_positions:
            return [], []
        out_positions: List[int] = []
        out_vectors: List[torch.Tensor] = []
        for abs_pos in all_pre_guess_positions[: prompt_len - 1]:
            out_positions.append(abs_pos)
            out_vectors.append(prompt_vec)
        guess_positions = all_pre_guess_positions[prompt_len - 1 :]
        if len(guess_positions) != guess_vecs.shape[0]:
            raise ValueError(f"Expected {guess_vecs.shape[0]} guess tokens, got {len(guess_positions)}.")
        for i, abs_pos in enumerate(guess_positions):
            out_positions.append(abs_pos)
            out_vectors.append(guess_vecs[i])
        return out_positions, out_vectors

    if mode == "guess_then_guess_probability_mean_replace":
        out_positions: List[int] = []
        out_vectors: List[torch.Tensor] = []
        guess_positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
        )
        if guess_positions:
            if len(guess_positions) != guess_vecs.shape[0]:
                raise ValueError(f"Expected {guess_vecs.shape[0]} guess tokens, got {len(guess_positions)}.")
            for i, abs_pos in enumerate(guess_positions):
                out_positions.append(abs_pos)
                out_vectors.append(guess_vecs[i])

        prob_positions = _absolute_prob_positions(
            prompt_len,
            decoded_tokens,
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
                    raise ValueError(f"Expected {prob_vecs.shape[0]} probability tokens, got {n_pos}.")
                prob_use = prob_vecs
            for i, abs_pos in enumerate(prob_positions):
                out_positions.append(abs_pos)
                out_vectors.append(prob_use[i])
        return out_positions, out_vectors

    raise ValueError(f"Unknown ablation mode for perturb hooks: {mode!r}")


def build_headwise_direction_perturb_hooks(
    layer_head_to_span_delta: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    *,
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    d_head: int,
    linguistic_confidence_prompt: bool = False,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    d_head = int(d_head)

    for layer in layer_head_to_span_delta:
        hook_name = f"blocks.{layer}.attn.hook_z"
        local_heads = [int(h) for h in selected_heads_by_layer.get(int(layer), [])]
        if not local_heads:
            raise ValueError(f"No selected heads for layer {layer}.")

        def _make_hook(layer_idx: int, heads_for_layer: List[int]):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_z activation with shape [batch, seq, heads, d_head], got {tuple(activation.shape)}."
                    )
                if activation.shape[3] != d_head:
                    raise ValueError(
                        f"hook_z d_head mismatch: activation has {activation.shape[3]}, expected {d_head}."
                    )
                for head_idx in heads_for_layer:
                    if not (0 <= head_idx < activation.shape[2]):
                        raise ValueError(
                            f"Head index {head_idx} out of range for hook_z activation with {activation.shape[2]} heads."
                        )
                    layer_delta = layer_head_to_span_delta[layer_idx][head_idx]
                    positions, vectors = _positions_and_vectors_for_mode(
                        mode,
                        prompt_len=prompt_len,
                        decoded_tokens=decoded_tokens_provider(),
                        layer_delta=layer_delta,
                        expected_guess_tokens=expected_guess_tokens,
                        expected_probability_tokens=expected_probability_tokens,
                        expected_confidence_tokens=expected_confidence_tokens,
                        linguistic_confidence_prompt=linguistic_confidence_prompt,
                    )
                    for abs_pos, vector in zip(positions, vectors):
                        if 0 <= abs_pos < activation.shape[1]:
                            if int(vector.numel()) != int(activation.shape[3]):
                                raise ValueError(
                                    f"Steering vector size {vector.numel()} does not match d_head {activation.shape[3]}."
                                )
                            activation[:, abs_pos, head_idx, :] += vector.to(
                                device=activation.device, dtype=activation.dtype
                            )
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(int(layer), local_heads)))
    return hooks


def build_concat_direction_perturb_hooks(
    layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]],
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    n_heads: int,
    d_head: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    linguistic_confidence_prompt: bool = False,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    n_heads = int(n_heads)
    d_head = int(d_head)

    for layer in layer_to_span_delta:
        hook_name = f"blocks.{layer}.attn.hook_z"

        def _make_hook(layer_idx: int):
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
                layer_delta = layer_to_span_delta[layer_idx]
                positions, vectors = _positions_and_vectors_for_mode(
                    mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens_provider(),
                    layer_delta=layer_delta,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    expected_confidence_tokens=expected_confidence_tokens,
                    linguistic_confidence_prompt=linguistic_confidence_prompt,
                )
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != n_heads * d_head:
                            raise ValueError(
                                f"Steering vector size {vector.numel()} does not match n_heads*d_head={n_heads*d_head}."
                            )
                        activation[:, abs_pos, :, :] += vector.to(
                            device=activation.device, dtype=activation.dtype
                        ).reshape(n_heads, d_head)
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


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


def greedy_generate_direction_perturbed_headwise(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_head_to_span_delta: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mode: str,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
    d_head: int,
    linguistic_confidence_prompt: bool = False,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_headwise_direction_perturb_hooks(
        layer_head_to_span_delta=layer_head_to_span_delta,
        selected_heads_by_layer=selected_heads_by_layer,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
        expected_confidence_tokens=expected_confidence_tokens,
        d_head=d_head,
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


def greedy_generate_direction_perturbed_concat(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]],
    mode: str,
    n_heads: int,
    d_head: int,
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

    hooks = build_concat_direction_perturb_hooks(
        layer_to_span_delta=layer_to_span_delta,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        n_heads=n_heads,
        d_head=d_head,
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


def _build_summary_json(
    *,
    non_none_modes: Sequence[str],
    units: Sequence[str],
    ablation_targets: Sequence[str],
    alphas: Sequence[float],
    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
    mode_responses_identical_true: Dict[str, Dict[str, Dict[str, Dict[float, int]]]],
    baseline_values_by_target: Dict[str, List[float]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for mode in non_none_modes:
        mode_payload: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float] | int]]]] = {}
        for unit in units:
            unit_payload: Dict[str, Dict[str, Dict[str, Optional[float] | int]]] = {}
            for target in ("low", "high"):
                if target not in ablation_targets:
                    continue
                alpha_payload: Dict[str, Dict[str, Optional[float] | int]] = {}
                for alpha in sorted(alphas):
                    values = mode_confidence_values.get(mode, {}).get(unit, {}).get(target, {}).get(alpha, [])
                    alpha_key = _format_alpha(alpha)
                    alpha_payload[alpha_key] = {
                        "alpha_value": float(alpha),
                        "mean_confidence": float(np.mean(values)) if values else None,
                        "sample_count": len(values),
                        "responses_identical_count": int(
                            mode_responses_identical_true.get(mode, {})
                            .get(unit, {})
                            .get(target, {})
                            .get(alpha, 0)
                        ),
                    }
                unit_payload[target] = alpha_payload
            mode_payload[unit] = unit_payload
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


def _plot_unit_mode_confidence(
    *,
    mode_name: str,
    unit: str,
    unit_payload: Dict[str, Dict[str, Dict[str, object]]],
    baseline_payload: Dict[str, Dict[str, object]],
    ablation_targets: Sequence[str],
    output_path: str,
) -> None:
    target_colors = {"high": "tab:blue", "low": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for target in ("high", "low"):
        if target not in ablation_targets:
            continue
        alpha_map = unit_payload.get(target, {})
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
    ax.set_title(f"{mode_name} ({unit}): confidence vs alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary_plots_from_json(
    *,
    summary_payload: Dict[str, object],
    units: Sequence[str],
    ablation_targets: Sequence[str],
    output_dir: str,
) -> None:
    baseline_payload = summary_payload.get("no_replacement_baseline", {})
    for mode_name, mode_payload in summary_payload.items():
        if mode_name == "no_replacement_baseline":
            continue
        if not isinstance(mode_payload, dict):
            continue
        for unit in units:
            unit_payload = mode_payload.get(unit, {})
            if not isinstance(unit_payload, dict):
                continue
            png_name = f"{mode_to_output_key(mode_name)}__{unit}_mean_confidence_vs_alpha.png"
            output_path = os.path.join(output_dir, png_name)
            _plot_unit_mode_confidence(
                mode_name=mode_name,
                unit=unit,
                unit_payload=unit_payload,
                baseline_payload=baseline_payload,
                ablation_targets=ablation_targets,
                output_path=output_path,
            )
            logging.info("Wrote %s", output_path)


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    model_n_heads: int,
    model_d_head: int,
    ablate_layers: Sequence[int],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    units: Sequence[str],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
    finished_at: str,
) -> None:
    lines = [
        "Headwise Mass Mean Probe Configuration",
        "======================================",
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
        f"ablation_targets={args.ablation_targets}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"ablate_heads_by_layer_spec={args.ablate_heads_by_layer}",
        f"ablate_heads_by_layer_resolved={format_layer_head_pairs(selected_heads_by_layer) if selected_heads_by_layer else 'ignored'}",
        f"whole_concat_mode={args.whole_concat_mode}",
        f"alpha={args.alpha}",
        "non_none_mode_behavior=additive_direction_perturbation",
        "direction_definition=high_mean_minus_low_mean",
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
        f"ablation_units={','.join(units)}",
        "",
        "[Mode Unit Confidence Means]",
    ]
    non_none_modes = [m for m in args.ablation_mode if m != "none"]
    for mode in non_none_modes:
        for unit in units:
            for target in args.ablation_targets:
                for alpha in args.alpha:
                    vals = mode_confidence_values[mode][unit][target][float(alpha)]
                    key = (
                        f"{mode}__{unit}__target_{target}__alpha_{_format_alpha(alpha)}"
                        "__mean_verbalised_confidence"
                    )
                    if vals:
                        lines.append(f"{key}={float(np.mean(vals)):.6f}")
                    else:
                        lines.append(f"{key}=None")
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headwise mass-mean direction probe inference (TransformerLens)."
    )
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
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
        "--ablate_heads_by_layer",
        type=str,
        default=None,
        help=(
            "Optional comma-separated <layer>.<head> list, e.g. '12.3,12.7,13.2'. "
            "When set, this selection takes precedence and --ablate_layers is ignored."
        ),
    )
    parser.add_argument(
        "--whole_concat_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, ignore --ablate_heads_by_layer and steer the whole concatenated pre-W_O attention activation.",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=ABLATION_MODES_DEFAULT,
        choices=ABLATION_MODES_DEFAULT,
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
        help="When linguistic prompt is used, expected token count for the confidence phrase.",
    )
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--normalize_span_directions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, normalize guess/probability directions so each span's summed per-position "
            "vector norms matches its unit budget."
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
        help="If true, use natural-language confidence prompt and phrase-to-probability parsing.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "headwise_mass_mean_probe/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline comparisons.")
    if args.normalize_span_directions:
        normalization_supported_modes = {
            "none",
            "probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
        }
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
    model_n_layers = int(model.cfg.n_layers)
    model_n_heads = int(model.cfg.n_heads)
    model_d_head = int(model.cfg.d_head)
    model_d_model = int(model.cfg.d_model)
    if model_n_heads * model_d_head != model_d_model:
        raise ValueError(
            f"Model shape mismatch: n_heads*d_head={model_n_heads*model_d_head}, d_model={model_d_model}."
        )

    ablate_layers_from_flag = parse_ablate_layers(args.ablate_layers, model_n_layers)
    if not ablate_layers_from_flag and not args.ablate_heads_by_layer:
        raise ValueError("No layers selected via --ablate_layers.")

    if args.whole_concat_mode:
        run_layers = sorted(ablate_layers_from_flag)
        selected_heads_by_layer: Dict[int, List[int]] = {}
        units = ["whole_concat"]
        logging.info(
            "whole_concat_mode=True: ignoring --ablate_heads_by_layer=%s",
            args.ablate_heads_by_layer,
        )
    elif args.ablate_heads_by_layer:
        selected_heads_by_layer = parse_layer_head_pairs(
            args.ablate_heads_by_layer, n_layers=model_n_layers, n_heads=model_n_heads
        )
        run_layers = sorted(selected_heads_by_layer.keys())
        if not run_layers:
            raise ValueError("No layers selected via --ablate_heads_by_layer.")
        logging.info(
            "Using --ablate_heads_by_layer selection; ignoring --ablate_layers=%s.",
            args.ablate_layers,
        )
        units = ["selected_layer_heads"]
    else:
        run_layers = sorted(ablate_layers_from_flag)
        selected_heads_by_layer = {layer: list(range(model_n_heads)) for layer in run_layers}
        logging.info(
            "No --ablate_heads_by_layer provided; using all heads across --ablate_layers=%s.",
            args.ablate_layers,
        )
        units = ["selected_layer_heads"]

    if not args.whole_concat_mode:
        missing_layer_heads = [layer for layer in run_layers if not selected_heads_by_layer.get(layer)]
        if missing_layer_heads:
            raise ValueError(
                "No selected heads provided for layers in this run: "
                + ",".join(str(layer) for layer in missing_layer_heads)
            )

    examples_h5 = load_examples_h5(Path(args.input_h5))
    _, _, direction_by_span, low_ids, high_ids = compute_low_high_span_means_and_directions_concat(
        examples_h5,
        ablate_layers=run_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_guess_tokens=args.expected_guess_tokens,
        whole_concat_mode=args.whole_concat_mode,
        n_heads=model_n_heads,
        d_head=model_d_head,
    )

    if args.normalize_span_directions:
        normalization_stats = normalize_direction_spans_to_unit_norm_budget(
            direction_by_span,
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

    logging.info(
        (
            "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), "
            "layers=%s, layer_heads=%s, units=%d, alphas=%s"
        ),
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        run_layers,
        format_layer_head_pairs(selected_heads_by_layer) if selected_heads_by_layer else "ignored",
        len(units),
        list(args.alpha),
    )

    out_path = resolve_output_json_path(args.output_json)
    results = {"train": {}, "validation": {}}
    mini_results = {"train": {}, "validation": {}}

    mode_confidence_values: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]] = {
        mode: {
            unit: {target: {float(alpha): [] for alpha in args.alpha} for target in args.ablation_targets}
            for unit in units
        }
        for mode in [m for m in args.ablation_mode if m != "none"]
    }
    mode_responses_identical_true: Dict[str, Dict[str, Dict[str, Dict[float, int]]]] = {
        mode: {
            unit: {target: {float(alpha): 0 for alpha in args.alpha} for target in args.ablation_targets}
            for unit in units
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

                        if args.whole_concat_mode:
                            layer_to_span_delta: Dict[int, Dict[str, torch.Tensor]] = {}
                            for layer_i, layer_idx in enumerate(run_layers):
                                layer_to_span_delta[int(layer_idx)] = {
                                    "prompt_mean": torch.tensor(
                                        sign * alpha * direction_by_span["prompt_mean"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                    "guess": torch.tensor(
                                        sign * alpha * direction_by_span["guess"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                    "sem_answer_mean": torch.tensor(
                                        sign * alpha * direction_by_span["sem_answer_mean"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                    "probability": torch.tensor(
                                        sign * alpha * direction_by_span["probability"][layer_i],
                                        device=device,
                                        dtype=torch_dtype,
                                    ),
                                }
                            response, decoded_tokens = greedy_generate_direction_perturbed_concat(
                                model=model,
                                local_prompt=local_prompt,
                                max_new_tokens=args.model_max_new_tokens,
                                layer_to_span_delta=layer_to_span_delta,
                                mode=mode,
                                n_heads=model_n_heads,
                                d_head=model_d_head,
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
                            unit_key = "whole_concat"
                            responses_identical = response == baseline_response
                            entry[key][unit_key] = {"response": response, "decoded_tokens": decoded_tokens}
                            mini_entry[key][unit_key] = {"response": response}
                            entry[key][unit_key]["responses_identical"] = responses_identical
                            mini_entry[key][unit_key]["responses_identical"] = responses_identical
                            if args.parse_mode_verbalised_confidence:
                                entry[key][unit_key]["verbalised_confidence"] = mode_confidence
                                mini_entry[key][unit_key]["verbalised_confidence"] = mode_confidence
                                if mode_confidence is not None:
                                    mode_confidence_values[mode][unit_key][target][float(alpha)].append(
                                        float(mode_confidence)
                                    )
                                if responses_identical:
                                    mode_responses_identical_true[mode][unit_key][target][float(alpha)] += 1
                                if mode_confidence is None or baseline_confidence is None:
                                    meets_none_confidence_direction = None
                                elif target == "low":
                                    meets_none_confidence_direction = mode_confidence > baseline_confidence
                                else:
                                    meets_none_confidence_direction = mode_confidence < baseline_confidence
                                entry[key][unit_key]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )
                                mini_entry[key][unit_key]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )
                        else:
                            layer_to_local_idx = {layer: i for i, layer in enumerate(run_layers)}
                            layer_head_to_span_delta: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
                            for layer_idx, head_indices in selected_heads_by_layer.items():
                                layer_i = layer_to_local_idx[int(layer_idx)]
                                layer_head_to_span_delta[int(layer_idx)] = {}
                                for head_idx in head_indices:
                                    layer_head_to_span_delta[int(layer_idx)][int(head_idx)] = {
                                        "prompt_mean": torch.tensor(
                                            sign * alpha * direction_by_span["prompt_mean"][layer_i, head_idx],
                                            device=device,
                                            dtype=torch_dtype,
                                        ),
                                        "guess": torch.tensor(
                                            sign * alpha * direction_by_span["guess"][layer_i, :, head_idx],
                                            device=device,
                                            dtype=torch_dtype,
                                        ),
                                        "sem_answer_mean": torch.tensor(
                                            sign * alpha * direction_by_span["sem_answer_mean"][layer_i, head_idx],
                                            device=device,
                                            dtype=torch_dtype,
                                        ),
                                        "probability": torch.tensor(
                                            sign * alpha * direction_by_span["probability"][layer_i, :, head_idx],
                                            device=device,
                                            dtype=torch_dtype,
                                        ),
                                    }
                            response, decoded_tokens = greedy_generate_direction_perturbed_headwise(
                                model=model,
                                local_prompt=local_prompt,
                                max_new_tokens=args.model_max_new_tokens,
                                layer_head_to_span_delta=layer_head_to_span_delta,
                                selected_heads_by_layer=selected_heads_by_layer,
                                mode=mode,
                                expected_guess_tokens=args.expected_guess_tokens,
                                expected_probability_tokens=args.expected_probability_tokens,
                                expected_confidence_tokens=args.expected_confidence_tokens,
                                d_head=model_d_head,
                                linguistic_confidence_prompt=args.linguistic_confidence_prompt,
                            )
                            mode_confidence = (
                                parse_mode_confidence_from_response(
                                    response, linguistic_prompt=args.linguistic_confidence_prompt
                                )
                                if args.parse_mode_verbalised_confidence
                                else None
                            )
                            unit_key = "selected_layer_heads"
                            responses_identical = response == baseline_response
                            entry[key][unit_key] = {"response": response, "decoded_tokens": decoded_tokens}
                            mini_entry[key][unit_key] = {"response": response}
                            entry[key][unit_key]["responses_identical"] = responses_identical
                            mini_entry[key][unit_key]["responses_identical"] = responses_identical
                            if args.parse_mode_verbalised_confidence:
                                entry[key][unit_key]["verbalised_confidence"] = mode_confidence
                                mini_entry[key][unit_key]["verbalised_confidence"] = mode_confidence
                                if mode_confidence is not None:
                                    mode_confidence_values[mode][unit_key][target][float(alpha)].append(
                                        float(mode_confidence)
                                    )
                                if responses_identical:
                                    mode_responses_identical_true[mode][unit_key][target][float(alpha)] += 1
                                if mode_confidence is None or baseline_confidence is None:
                                    meets_none_confidence_direction = None
                                elif target == "low":
                                    meets_none_confidence_direction = mode_confidence > baseline_confidence
                                else:
                                    meets_none_confidence_direction = mode_confidence < baseline_confidence
                                entry[key][unit_key]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )
                                mini_entry[key][unit_key]["meets_none_confidence_direction"] = (
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
        units=units,
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
        units=units,
        ablation_targets=args.ablation_targets,
        output_dir=os.path.dirname(out_path),
    )
    config_out_path = config_txt_path(out_path)
    write_config_txt(
        config_out_path,
        args=args,
        device=device,
        model_n_layers=model_n_layers,
        model_n_heads=model_n_heads,
        model_d_head=model_d_head,
        ablate_layers=run_layers,
        selected_heads_by_layer=selected_heads_by_layer,
        units=units,
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
