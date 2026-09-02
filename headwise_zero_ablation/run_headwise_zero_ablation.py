#!/usr/bin/env python3
"""Simultaneous head/MLP zero ablation on high- and low-confidence groups.

Zeros selected attention-head outputs (`hook_z`) and/or MLP subblock outputs
(`hook_mlp_out`) at mode-dependent token spans during greedy decoding. Example
grouping streams only `verbalised_confidence` from the processed H5 (no
embeddings are loaded). Aggregate metrics use a running mean rather than storing
every parsed confidence value.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_zero_ablation import run_blockwise_zero_ablation as blockwise_zero_mod
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
    _absolute_probability_value_positions,
    _completion_token_index_to_abs_pos,
    _is_expected_or_plus_two,
    build_subblock_zero_hooks,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    greedy_generate,
    load_eval_dataset,
    parse_ablate_layers,
    parse_guess_and_probability_indices,
    parse_mode_confidence_from_response,
    split_answerable_indices,
)
from layerwise_mean_ablation import run_mean_ablation as layerwise_mean_mod
from layerwise_mean_ablation.run_mean_ablation import (
    configure_prefix_tokens_for_model,
    load_hooked_transformer,
)


TRAIN_RATIO = 0.9
CONFIDENCE_GROUPS = ("low_confidence", "high_confidence")
NONE_MODE_CONFIDENCE_BUCKETS = ("eq_1", "lt_1")
NONE_MODE_CONFIDENCE_BUCKET_LABELS = {
    "eq_1": "none_eq_1",
    "lt_1": "none_lt_1",
}
ABLATION_UNIT_KEY = "selected_layer_heads"
_ATTN_UNIT_RE = re.compile(r"^a(\d+)\.h(\d+)$")
_ATTN_BLOCK_UNIT_RE = re.compile(r"^a(\d+)$")
_MLP_UNIT_RE = re.compile(r"^m(\d+)$")
_OLD_LAYER_HEAD_RE = re.compile(r"^\d+\.\d+$")
ABLATION_MODES_DEFAULT = [
    "none",
    "probability_tokens_zero_ablate",
    "probability_last_token_zero_ablate",
    "extended_probability_last_token_zero_ablate",
    "probability_pre_and_post_period_digit_zero_ablate",
    "probability_span_except_last_token_zero_ablate",
    "all_pre_probability_tokens_zero_ablate",
    "guess_tokens_zero_ablate",
    "all_pre_guess_tokens_zero_ablate",
    "guess_then_guess_probability_zero_ablate",
    "probability_value_zero_ablate",
    "current_generated_token_zero_ablate",
]


class RunningMean:
    """Welford running mean so aggregate stats do not store every sample."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        self.mean += (float(value) - self.mean) / self.n

    def value(self) -> Optional[float]:
        if self.n == 0:
            return None
        return float(self.mean)


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("headwise_zero_ablation") / "results"
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


def summary_json_path(mini_output_path: str) -> str:
    return os.path.join(os.path.dirname(mini_output_path), "summary.json")


def attach_output_log(run_root: str) -> str:
    output_log_path = os.path.join(run_root, "output.log")
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    return output_log_path


def mode_to_output_key(mode: str) -> str:
    if mode == "none":
        return "no_replacement"
    if mode in ABLATION_MODES_DEFAULT:
        return mode
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def _sync_prefix_tokens_for_model(model_name: str) -> None:
    configure_prefix_tokens_for_model(model_name)
    blockwise_zero_mod.GUESS_PREFIX_TOKENS = layerwise_mean_mod.GUESS_PREFIX_TOKENS
    blockwise_zero_mod.PROBABILITY_PREFIX_TOKENS = layerwise_mean_mod.PROBABILITY_PREFIX_TOKENS


def _open_h5_readonly(path: Path | str):
    """Open an H5 file for reading with a small chunk cache; disable locking when supported."""
    kwargs = {"rdcc_nbytes": 1024**2}
    try:
        return h5py.File(path, "r", locking=False, **kwargs)
    except TypeError:
        return h5py.File(path, "r", **kwargs)


def _id_column_to_index_map(dataset) -> Dict[str, int]:
    """Map encoded example IDs to row indices without materializing full rows."""
    if hasattr(dataset, "column_names") and "id" in getattr(dataset, "column_names", []):
        ids = dataset["id"]
        return {encode_example_id(ex_id): i for i, ex_id in enumerate(ids)}
    return {encode_example_id(ex["id"]): i for i, ex in enumerate(dataset)}


def _split_sample_targets(num_samples: int) -> Dict[str, int]:
    return {
        "train": round(num_samples * TRAIN_RATIO),
        "validation": round(num_samples * (1 - TRAIN_RATIO)),
    }


def _selected_groups_filled(
    selected: Dict[str, Dict[str, List[str]]],
    targets: Dict[str, int],
) -> bool:
    for group_name in CONFIDENCE_GROUPS:
        for split_name, target in targets.items():
            if len(selected[group_name][split_name]) < target:
                return False
    return True


def _read_verbalised_confidence_from_example_group(example_node, ex_id: str) -> float:
    if "responses" not in example_node:
        raise ValueError(f"Example {ex_id} must have exactly one response, got 0.")
    responses = example_node["responses"]
    if not isinstance(responses, h5py.Group):
        raise ValueError(f"Example {ex_id} responses is not a group.")
    length = int(responses.attrs.get("__len__", len(responses.keys())))
    if length != 1 or "0" not in responses:
        raise ValueError(f"Example {ex_id} must have exactly one response, got {length}.")
    response0 = responses["0"]
    if not isinstance(response0, h5py.Group):
        raise ValueError(f"Example {ex_id} responses/0 is not a dict.")
    dataset = response0.get("verbalised_confidence")
    if dataset is None or not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Example {ex_id} responses/0/verbalised_confidence is missing.")
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = np.asarray(value).reshape(-1)[0]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Example {ex_id} responses/0/verbalised_confidence is not a float."
        ) from exc


def collect_confidence_group_ids_streaming(
    path: Path | str,
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    split_id_to_index: Dict[str, Dict[str, int]],
    split_targets: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, List[str]]], int]:
    """Stream verbalised_confidence only; keep at most num_samples IDs per group.

    Stops once both confidence groups have enough train/validation IDs. Embeddings
    are never loaded.
    """
    selected: Dict[str, Dict[str, List[str]]] = {
        group_name: {split_name: [] for split_name in split_targets} for group_name in CONFIDENCE_GROUPS
    }
    seen_low = False
    seen_high = False
    with _open_h5_readonly(path) as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        h5_example_count = int(len(examples_group))
        for example_id in examples_group.keys():
            ex_id = str(example_id)
            conf = _read_verbalised_confidence_from_example_group(examples_group[example_id], ex_id)
            groups_for_example: List[str] = []
            if conf <= low_conf_threshold:
                seen_low = True
                groups_for_example.append("low_confidence")
            if conf >= high_conf_threshold:
                seen_high = True
                groups_for_example.append("high_confidence")
            if not groups_for_example:
                if seen_low and seen_high and _selected_groups_filled(selected, split_targets):
                    break
                continue
            split_name = None
            if ex_id in split_id_to_index.get("train", {}):
                split_name = "train"
            elif ex_id in split_id_to_index.get("validation", {}):
                split_name = "validation"
            if split_name is not None:
                target = int(split_targets.get(split_name, 0))
                for group_name in groups_for_example:
                    bucket = selected[group_name][split_name]
                    if len(bucket) < target:
                        bucket.append(ex_id)
            if seen_low and seen_high and _selected_groups_filled(selected, split_targets):
                break
    if not seen_low:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not seen_high:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    return selected, h5_example_count


def _absolute_extended_prob_last_token_only_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
) -> List[int]:
    """Absolute position for the post-period digit (`end_prob + 2`), once in-sequence."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    full_rel_positions = list(range(first_prob, end_prob + 1))
    if not _is_expected_or_plus_two(len(full_rel_positions), expected_probability_tokens):
        return []
    abs_pos = _completion_token_index_to_abs_pos(prompt_len, end_prob + 2)
    seq_len = prompt_len + len(decoded_tokens)
    if 0 <= abs_pos < seq_len:
        return [abs_pos]
    return []


def _mode_positions_provider_builder(
    mode: str,
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Callable[[int, Callable[[], List[str]]], Callable[[], List[int]]]:
    def _builder(prompt_len: int, decoded_tokens_provider: Callable[[], List[str]]) -> Callable[[], List[int]]:
        if mode == "probability_tokens_zero_ablate":
            return lambda: _absolute_prob_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "probability_last_token_zero_ablate":
            return lambda: _absolute_prob_last_token_only_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "extended_probability_last_token_zero_ablate":
            return lambda: _absolute_extended_prob_last_token_only_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "probability_pre_and_post_period_digit_zero_ablate":
            def _pre_and_post_period_digit_positions() -> List[int]:
                decoded_tokens = decoded_tokens_provider()
                return _absolute_prob_last_token_only_positions(
                    prompt_len,
                    decoded_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                ) + _absolute_extended_prob_last_token_only_positions(
                    prompt_len,
                    decoded_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )

            return _pre_and_post_period_digit_positions
        if mode == "probability_span_except_last_token_zero_ablate":
            return lambda: _absolute_prob_except_last_token_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "guess_tokens_zero_ablate":
            return lambda: _absolute_guess_span_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_guess_tokens=expected_guess_tokens,
            )
        if mode == "all_pre_probability_tokens_zero_ablate":
            return lambda: (
                []
                if (
                    pos_map := _absolute_pre_probability_positions(
                        prompt_len,
                        decoded_tokens_provider(),
                        expected_guess_tokens=expected_guess_tokens,
                        expected_probability_tokens=expected_probability_tokens,
                    )
                )
                is None
                else pos_map["prompt"] + pos_map["guess"] + pos_map["sem_answer"] + pos_map["probability"]
            )
        if mode == "all_pre_guess_tokens_zero_ablate":
            return lambda: _absolute_all_pre_guess_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_guess_tokens=expected_guess_tokens,
            )
        if mode == "guess_then_guess_probability_zero_ablate":
            return lambda: _absolute_guess_then_guess_probability_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "probability_value_zero_ablate":
            return lambda: _absolute_probability_value_positions(
                prompt_len,
                decoded_tokens_provider(),
            )
        if mode == "current_generated_token_zero_ablate":
            def _current_generated_positions() -> List[int]:
                current_abs_pos = prompt_len + len(decoded_tokens_provider()) - 1
                if current_abs_pos < 0:
                    return []
                return [current_abs_pos]

            return _current_generated_positions
        raise ValueError(f"Unknown ablation mode for provider builder: {mode!r}")

    return _builder


def parse_ablate_units(
    spec: str, *, n_layers: int, n_heads: int
) -> Tuple[Dict[int, List[int]], List[int], List[int]]:
    raw = (spec or "").strip()
    if not raw:
        raise ValueError(
            "--ablate_heads must be a non-empty comma-separated list of "
            "a<layer>, a<layer>.h<head>, and/or m<layer> tokens, e.g. 'a24,a24.h5,m30'."
        )
    heads_by_layer: Dict[int, Set[int]] = {}
    mlp_layers: Set[int] = set()
    attn_block_layers: Set[int] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if _OLD_LAYER_HEAD_RE.fullmatch(item):
            layer_str, head_str = item.split(".", 1)
            raise ValueError(
                f"Invalid unit {item!r}. The old <layer>.<head> format is no longer supported; "
                f"use a<layer>.h<head> for attention heads (e.g. a{layer_str}.h{head_str}), "
                f"a<layer> for whole attention blocks (e.g. a24), "
                f"or m<layer> for MLP subblocks (e.g. m30)."
            )
        attn_match = _ATTN_UNIT_RE.fullmatch(item)
        attn_block_match = _ATTN_BLOCK_UNIT_RE.fullmatch(item)
        mlp_match = _MLP_UNIT_RE.fullmatch(item)
        if attn_match:
            layer_idx = int(attn_match.group(1))
            head_idx = int(attn_match.group(2))
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}).")
            if head_idx < 0 or head_idx >= n_heads:
                raise ValueError(f"Head index {head_idx} out of range [0, {n_heads}).")
            heads_by_layer.setdefault(layer_idx, set()).add(head_idx)
            continue
        if attn_block_match:
            layer_idx = int(attn_block_match.group(1))
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}).")
            attn_block_layers.add(layer_idx)
            continue
        if mlp_match:
            layer_idx = int(mlp_match.group(1))
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}).")
            mlp_layers.add(layer_idx)
            continue
        raise ValueError(
            f"Invalid unit {item!r}. Expected a<layer> (e.g. a24), "
            f"a<layer>.h<head> (e.g. a24.h5), or m<layer> (e.g. m30)."
        )
    overlap = sorted(set(attn_block_layers) & set(heads_by_layer))
    if overlap:
        raise ValueError(
            "Cannot mix whole-attention-block a<layer> with per-head a<layer>.h<head> "
            "on the same layer(s): " + ",".join(f"a{layer}" for layer in overlap) + "."
        )
    if not heads_by_layer and not mlp_layers and not attn_block_layers:
        raise ValueError(
            "No valid a<layer>, a<layer>.h<head>, or m<layer> entries found in --ablate_heads."
        )
    return (
        {layer: sorted(heads) for layer, heads in sorted(heads_by_layer.items())},
        sorted(mlp_layers),
        sorted(attn_block_layers),
    )


def format_ablate_units(
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mlp_layers: Sequence[int],
    attn_block_layers: Sequence[int] = (),
) -> str:
    parts: List[str] = []
    for layer_idx in sorted(set(int(layer) for layer in attn_block_layers)):
        parts.append(f"a{layer_idx}")
    for layer_idx in sorted(selected_heads_by_layer):
        for head_idx in sorted(set(int(h) for h in selected_heads_by_layer[layer_idx])):
            parts.append(f"a{layer_idx}.h{head_idx}")
    for layer_idx in sorted(set(int(layer) for layer in mlp_layers)):
        parts.append(f"m{layer_idx}")
    return ",".join(parts)


def build_selected_layer_heads_zero_hooks(
    layer_indices: Sequence[int],
    *,
    positions_provider: Callable[[], List[int]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.attn.hook_z"
        heads_for_layer = [int(h) for h in selected_heads_by_layer.get(int(layer), [])]
        if not heads_for_layer:
            raise ValueError(f"No selected heads for layer {layer}.")

        def _make_hook(heads: List[int], *, local_layer: int) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if activation.ndim != 4:
                    raise ValueError(
                        f"Expected hook_z activation with shape [batch, seq, heads, d_head], "
                        f"got {tuple(activation.shape)}."
                    )
                n_heads_act = int(activation.shape[2])
                for head_idx in heads:
                    if not (0 <= head_idx < n_heads_act):
                        raise ValueError(
                            f"Head index {head_idx} out of range for hook_z at layer {local_layer} "
                            f"with {n_heads_act} heads."
                        )
                positions = positions_provider()
                if not positions:
                    return activation
                for abs_pos in positions:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, heads, :] = 0
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(heads_for_layer, local_layer=int(layer))))
    return hooks


def greedy_generate_selected_layer_heads_zero_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    mode: str,
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mlp_layers: Sequence[int],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    positions_provider = _mode_positions_provider_builder(
        mode,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )(prompt_len, _decoded_tokens_provider)
    head_layers = sorted(
        int(layer) for layer, heads in selected_heads_by_layer.items() if heads
    )
    hooks: List[Tuple[str, Callable]] = []
    if head_layers:
        hooks.extend(
            build_selected_layer_heads_zero_hooks(
                layer_indices=head_layers,
                positions_provider=positions_provider,
                selected_heads_by_layer=selected_heads_by_layer,
            )
        )
    if mlp_layers:
        hooks.extend(
            build_subblock_zero_hooks(
                mlp_layers,
                subblock="mlp",
                positions_provider=positions_provider,
            )
        )
    if not hooks:
        raise ValueError("No attention heads or MLP layers selected for zero ablation.")
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def _empty_mode_trackers(ablation_modes: Sequence[str]) -> Dict[str, object]:
    return {
        "confidence": {mode: RunningMean() for mode in ablation_modes},
        "delta": {mode: RunningMean() for mode in ablation_modes if mode != "none"},
        "identical": {mode: 0 for mode in ablation_modes if mode != "none"},
        "example_count": 0,
    }


def _none_mode_confidence_bucket(baseline_mode_confidence: Optional[float]) -> Optional[str]:
    if baseline_mode_confidence is None:
        return None
    if float(baseline_mode_confidence) == 1.0:
        return "eq_1"
    return "lt_1"


def _record_ablation_mode_metrics(
    trackers: Dict[str, object],
    *,
    mode_name: str,
    mode_confidence: Optional[float],
    confidence_delta: Optional[float],
    responses_identical: bool,
) -> None:
    if responses_identical:
        trackers["identical"][mode_name] += 1
    if mode_confidence is not None:
        trackers["confidence"][mode_name].update(float(mode_confidence))
    if confidence_delta is not None:
        trackers["delta"][mode_name].update(confidence_delta)


def _modes_summary_from_trackers(
    ablation_modes: Sequence[str],
    *,
    confidence: Dict[str, RunningMean],
    delta: Dict[str, RunningMean],
    identical: Dict[str, int],
) -> Dict[str, dict]:
    modes_out: Dict[str, dict] = {}
    for mode_name in ablation_modes:
        running = confidence[mode_name]
        entry: dict = {
            "mean_confidence": running.value(),
            "sample_count": running.n,
        }
        if mode_name != "none":
            delta_running = delta[mode_name]
            entry["mean_confidence_delta"] = delta_running.value()
            entry["confidence_delta_sample_count"] = delta_running.n
            entry["responses_identical_true"] = int(identical[mode_name])
        modes_out[mode_name] = entry
    return modes_out


def _append_mode_metric_lines(
    lines: List[str],
    *,
    ablation_modes: Sequence[str],
    confidence: Dict[str, RunningMean],
    delta: Dict[str, RunningMean],
    identical: Dict[str, int],
    skip_one_confidence_n: Optional[int] = None,
) -> None:
    for mode_name in ablation_modes:
        metric_key = f"{mode_name}__{ABLATION_UNIT_KEY}"
        running = confidence.get(mode_name)
        mode_mean = None if running is None else running.value()
        valid_count = 0 if running is None else running.n
        if mode_name == "none":
            if mode_mean is None:
                none_line = f"{metric_key}=None ({valid_count})"
            else:
                none_line = f"{metric_key}={mode_mean:.6f} ({valid_count})"
            if skip_one_confidence_n is not None:
                none_line += f" [skipped_one_confidence: {skip_one_confidence_n}]"
            lines.append(none_line)
            continue
        identical_n = int(identical.get(mode_name, 0))
        delta_running = delta.get(mode_name)
        delta_mean = None if delta_running is None else delta_running.value()
        if mode_mean is None:
            mean_str = "None"
        else:
            mean_str = f"{mode_mean:.6f}"
        if delta_mean is None:
            delta_str = "None"
        else:
            delta_str = f"{delta_mean:+.6f}"
        lines.append(
            f"{metric_key}={mean_str} ({valid_count}) "
            f"[mean_delta: {delta_str}] [responses_identical: {identical_n}]"
        )


def _empty_group_metrics(ablation_modes: Sequence[str]) -> Dict[str, object]:
    trackers = _empty_mode_trackers(ablation_modes)
    return {
        "confidence": trackers["confidence"],
        "delta": trackers["delta"],
        "identical": trackers["identical"],
        "evaluated_count": 0,
        "skipped_one_confidence": 0,
        "by_none_mode_confidence": {
            bucket: _empty_mode_trackers(ablation_modes) for bucket in NONE_MODE_CONFIDENCE_BUCKETS
        },
    }


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
    mlp_layers: Sequence[int],
    num_selected_layer_head_pairs: int,
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    evaluated_counts: Dict[str, int],
    skipped_one_confidence: Dict[str, int],
    mode_confidence: Dict[str, Dict[str, RunningMean]],
    mode_delta: Dict[str, Dict[str, RunningMean]],
    mode_identical: Dict[str, Dict[str, int]],
    by_none_mode_confidence: Dict[str, Dict[str, Dict[str, object]]],
    finished_at: str,
) -> None:
    lines = [
        "Headwise Zero Ablation Configuration",
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
        f"dataset={args.dataset}",
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
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"ablate_heads_spec={args.ablate_heads}",
        f"ablate_heads_resolved={format_ablate_units(selected_heads_by_layer, mlp_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"num_ablated_layer_head_pairs={num_selected_layer_head_pairs}",
        f"num_ablated_mlp_layers={len(mlp_layers)}",
        f"ablation_unit={ABLATION_UNIT_KEY}",
        f"skip_one_confidence={args.skip_one_confidence}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        f"low_conf_evaluated_count={evaluated_counts.get('low_confidence', 0)}",
        f"high_conf_evaluated_count={evaluated_counts.get('high_confidence', 0)}",
        f"low_conf_skipped_one_confidence_count={skipped_one_confidence.get('low_confidence', 0)}",
        f"high_conf_skipped_one_confidence_count={skipped_one_confidence.get('high_confidence', 0)}",
        "",
        "[Mode Confidence Metrics]",
        "Values below are running-mean verbalised confidence per group.",
        "Additional sections split each group by none-mode verbalised confidence",
        "(eq_1: exactly 1.0; lt_1: parsed and < 1.0).",
        "",
    ]
    for group_name in CONFIDENCE_GROUPS:
        lines.append(f"[{group_name}]")
        _append_mode_metric_lines(
            lines,
            ablation_modes=args.ablation_mode,
            confidence=mode_confidence.get(group_name, {}),
            delta=mode_delta.get(group_name, {}),
            identical=mode_identical.get(group_name, {}),
            skip_one_confidence_n=(
                int(skipped_one_confidence.get(group_name, 0)) if args.skip_one_confidence else None
            ),
        )
        lines.append("")
        group_buckets = by_none_mode_confidence.get(group_name, {})
        for bucket_name in NONE_MODE_CONFIDENCE_BUCKETS:
            bucket = group_buckets.get(bucket_name, {})
            bucket_label = NONE_MODE_CONFIDENCE_BUCKET_LABELS[bucket_name]
            example_count = int(bucket.get("example_count", 0)) if bucket else 0
            lines.append(f"[{group_name} / {bucket_label}]")
            lines.append(f"example_count={example_count}")
            if bucket:
                _append_mode_metric_lines(
                    lines,
                    ablation_modes=args.ablation_mode,
                    confidence=bucket["confidence"],
                    delta=bucket["delta"],
                    identical=bucket["identical"],
                )
            lines.append("")
    lines.extend(["[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_summary(
    *,
    run_root: str,
    args: argparse.Namespace,
    ablate_layers: Sequence[int],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mlp_layers: Sequence[int],
    num_selected_layer_head_pairs: int,
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    evaluated_counts: Dict[str, int],
    skipped_one_confidence: Dict[str, int],
    mode_confidence: Dict[str, Dict[str, RunningMean]],
    mode_delta: Dict[str, Dict[str, RunningMean]],
    mode_identical: Dict[str, Dict[str, int]],
    by_none_mode_confidence: Dict[str, Dict[str, Dict[str, object]]],
) -> dict:
    groups: Dict[str, dict] = {}
    for group_name in CONFIDENCE_GROUPS:
        selected_count = low_conf_count if group_name == "low_confidence" else high_conf_count
        modes_out = _modes_summary_from_trackers(
            args.ablation_mode,
            confidence=mode_confidence[group_name],
            delta=mode_delta[group_name],
            identical=mode_identical[group_name],
        )
        none_mode_buckets: Dict[str, dict] = {}
        group_buckets = by_none_mode_confidence.get(group_name, {})
        for bucket_name in NONE_MODE_CONFIDENCE_BUCKETS:
            bucket = group_buckets.get(bucket_name, {})
            none_mode_buckets[bucket_name] = {
                "example_count": int(bucket.get("example_count", 0)) if bucket else 0,
                "modes": _modes_summary_from_trackers(
                    args.ablation_mode,
                    confidence=bucket["confidence"],
                    delta=bucket["delta"],
                    identical=bucket["identical"],
                )
                if bucket
                else {},
            }
        groups[group_name] = {
            "selected_count": selected_count,
            "evaluated_count": int(evaluated_counts.get(group_name, 0)),
            "skipped_one_confidence_count": int(skipped_one_confidence.get(group_name, 0)),
            "modes": modes_out,
            "by_none_mode_confidence": none_mode_buckets,
        }
    return {
        "run_root": run_root,
        "dataset": args.dataset,
        "input_h5": args.input_h5,
        "h5_example_count": h5_example_count,
        "ablate_layers": list(ablate_layers),
        "ablate_heads": format_ablate_units(selected_heads_by_layer, mlp_layers),
        "ablate_mlp_layers": list(mlp_layers),
        "num_ablated_layer_head_pairs": num_selected_layer_head_pairs,
        "num_ablated_mlp_layers": len(mlp_layers),
        "ablation_modes": list(args.ablation_mode),
        "skip_one_confidence": bool(args.skip_one_confidence),
        "low_conf_threshold": args.low_conf_threshold,
        "high_conf_threshold": args.high_conf_threshold,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simultaneous head/MLP zero ablation on high- and low-confidence groups."
    )
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="Path to processed train/validation verbalised embedding H5 (confidence metadata only is read).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["trivia_qa", "squad", "bioasq", "nq", "svamp", "gsm8k"],
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_max_new_tokens", type=int, default=30)
    parser.add_argument("--brief_prompt", type=str, default="default", choices=["default", "chat"])
    parser.add_argument("--brief_always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_brief", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablate_layers", type=str, default="12-15")
    parser.add_argument(
        "--ablate_heads",
        type=str,
        default=None,
        help=(
            "Optional comma-separated unit list. Attention heads use a<layer>.h<head> "
            "(e.g. a24.h5); whole attention blocks use a<layer> (e.g. a24); "
            "MLP subblocks use m<layer> (e.g. m30). "
            "All listed units are zero-ablated simultaneously. "
            "Whole-block a<layer> units are not supported in this script. "
            "When set, this selection takes precedence and --ablate_layers is ignored."
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
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument("--parse_mode_verbalised_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip_one_confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, skip remaining ablation modes for examples whose none/baseline "
            "verbalised confidence is 1.0. Requires 'none' in --ablation_mode."
        ),
    )
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline comparison.")
    if args.skip_one_confidence and "none" not in args.ablation_mode:
        raise ValueError("--skip_one_confidence requires 'none' in --ablation_mode.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)
    os.makedirs(run_root, exist_ok=True)
    attach_output_log(run_root)

    _sync_prefix_tokens_for_model(args.model_name)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_eval_dataset(args.dataset, args.random_seed)
    random.seed(args.random_seed)
    if args.num_few_shot == 0:
        prompt_indices: List[int] = []
    else:
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

    split_targets = _split_sample_targets(args.num_samples)
    split_id_to_index = {
        "train": _id_column_to_index_map(train_ds),
        "validation": _id_column_to_index_map(val_ds),
    }
    logging.info("Streaming confidence groups from %s (no embeddings loaded).", args.input_h5)
    selected_ids_by_group, h5_example_count = collect_confidence_group_ids_streaming(
        args.input_h5,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        split_id_to_index=split_id_to_index,
        split_targets=split_targets,
    )
    used_ids = {
        ex_id
        for group_splits in selected_ids_by_group.values()
        for ids in group_splits.values()
        for ex_id in ids
    }
    split_id_to_index = {
        split_name: {ex_id: idx for ex_id, idx in id_map.items() if ex_id in used_ids}
        for split_name, id_map in split_id_to_index.items()
    }
    low_conf_count = sum(len(ids) for ids in selected_ids_by_group["low_confidence"].values())
    high_conf_count = sum(len(ids) for ids in selected_ids_by_group["high_confidence"].values())
    logging.info(
        "H5 has %d examples. selected low_conf=%d (<=%.3f), high_conf=%d (>=%.3f).",
        h5_example_count,
        low_conf_count,
        args.low_conf_threshold,
        high_conf_count,
        args.high_conf_threshold,
    )
    gc.collect()

    logging.info("Loading HookedTransformer: %s dtype=%s", args.model_name, args.dtype)
    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    model_n_layers = int(model.cfg.n_layers)
    model_n_heads = int(model.cfg.n_heads)
    model_d_head = int(model.cfg.d_head)
    model_d_model = int(model.cfg.d_model)
    if model_n_heads * model_d_head != model_d_model:
        logging.info(
            "n_heads*d_head=%d != d_model=%d (expected for Gemma-style independent head_dim).",
            model_n_heads * model_d_head,
            model_d_model,
        )

    ablate_layers_from_flag = parse_ablate_layers(args.ablate_layers, model_n_layers)
    if not ablate_layers_from_flag and not args.ablate_heads:
        raise ValueError("No layers selected via --ablate_layers.")

    if args.ablate_heads:
        selected_heads_by_layer, mlp_layers, attn_block_layers = parse_ablate_units(
            args.ablate_heads, n_layers=model_n_layers, n_heads=model_n_heads
        )
        if attn_block_layers:
            raise ValueError(
                "Whole-attention-block units a<layer> (e.g. a24) are not supported in "
                "headwise zero ablation; use a<layer>.h<head> for heads or m<layer> for MLP, "
                "or run componentwise mean ablation / mass-mean shift for whole attn blocks."
            )
        run_layers = sorted(set(selected_heads_by_layer.keys()) | set(mlp_layers))
        if not run_layers:
            raise ValueError("No units selected via --ablate_heads.")
        logging.info(
            "Using --ablate_heads selection; ignoring --ablate_layers=%s.",
            args.ablate_layers,
        )
    else:
        run_layers = sorted(ablate_layers_from_flag)
        selected_heads_by_layer = {layer: list(range(model_n_heads)) for layer in run_layers}
        mlp_layers = []
        attn_block_layers = []
        logging.info(
            "No --ablate_heads provided; using all heads across --ablate_layers=%s.",
            args.ablate_layers,
        )

    missing_layer_heads = [
        layer for layer, heads in selected_heads_by_layer.items() if not heads
    ]
    if missing_layer_heads:
        raise ValueError(
            "No selected heads provided for layers in this run: "
            + ",".join(str(layer) for layer in missing_layer_heads)
        )
    if not selected_heads_by_layer and not mlp_layers:
        raise ValueError("No attention heads or MLP layers selected for zero ablation.")
    num_selected_layer_head_pairs = sum(len(heads) for heads in selected_heads_by_layer.values())
    resolved_units = format_ablate_units(selected_heads_by_layer, mlp_layers)
    logging.info(
        "Ablating units=%s (%d layer-head pairs, %d mlp layers).",
        resolved_units,
        num_selected_layer_head_pairs,
        len(mlp_layers),
    )

    results_mini: Dict[str, Dict[str, dict]] = {
        group_name: {"train": {}, "validation": {}} for group_name in CONFIDENCE_GROUPS
    }
    group_metrics = {group_name: _empty_group_metrics(args.ablation_mode) for group_name in CONFIDENCE_GROUPS}

    for group_name in CONFIDENCE_GROUPS:
        metrics = group_metrics[group_name]
        logging.info(
            "Ablating group=%s (%d selected H5 ids); %d heads and %d MLP layers zeroed simultaneously.",
            group_name,
            sum(len(ids) for ids in selected_ids_by_group[group_name].values()),
            num_selected_layer_head_pairs,
            len(mlp_layers),
        )
        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = int(split_targets[split_name])
            id_to_index = split_id_to_index[split_name]
            selected_ids = selected_ids_by_group[group_name][split_name]
            if split_target > 0 and not selected_ids:
                logging.warning("No ablation target IDs available for %s / %s split.", group_name, split_name)
                continue
            logging.info(
                "Generating for %d examples (%s / %s split).",
                len(selected_ids),
                group_name,
                split_name,
            )

            for i, ex_id in enumerate(selected_ids):
                ds_idx = id_to_index.get(ex_id)
                if ds_idx is None:
                    raise ValueError(
                        f"Example id {ex_id} selected from H5 but missing in {split_name} split."
                    )
                example = eval_ds[int(ds_idx)]
                local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + example["question"]
                entry = {"question": example["question"]}

                baseline_response, _ = greedy_generate(
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
                skip_remaining_modes = (
                    args.skip_one_confidence
                    and baseline_mode_confidence is not None
                    and float(baseline_mode_confidence) == 1.0
                )
                if skip_remaining_modes:
                    metrics["skipped_one_confidence"] += 1
                    logging.info(
                        "[%s %s %d/%d] %s skipping remaining ablation modes: none confidence is 1.0",
                        group_name,
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                    )
                none_bucket = _none_mode_confidence_bucket(baseline_mode_confidence)
                bucket_metrics: Optional[Dict[str, object]] = None
                if none_bucket is not None:
                    bucket_metrics = metrics["by_none_mode_confidence"][none_bucket]
                    bucket_metrics["example_count"] += 1

                for mode_name in args.ablation_mode:
                    key = mode_to_output_key(mode_name)
                    if mode_name == "none":
                        entry[key] = {
                            "response": baseline_response,
                            "verbalised_confidence": baseline_mode_confidence,
                        }
                        if baseline_mode_confidence is not None and not skip_remaining_modes:
                            metrics["confidence"][mode_name].update(float(baseline_mode_confidence))
                        if bucket_metrics is not None and baseline_mode_confidence is not None:
                            bucket_metrics["confidence"][mode_name].update(
                                float(baseline_mode_confidence)
                            )
                        continue
                    if skip_remaining_modes:
                        continue

                    response, _ = greedy_generate_selected_layer_heads_zero_ablated(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=args.model_max_new_tokens,
                        mode=mode_name,
                        selected_heads_by_layer=selected_heads_by_layer,
                        mlp_layers=mlp_layers,
                        expected_guess_tokens=args.expected_guess_tokens,
                        expected_probability_tokens=args.expected_probability_tokens,
                    )
                    parsed_mode_confidence = (
                        parse_mode_confidence_from_response(response)
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    responses_identical = response == baseline_response
                    confidence_delta: Optional[float] = None
                    if parsed_mode_confidence is not None and baseline_mode_confidence is not None:
                        confidence_delta = float(parsed_mode_confidence) - float(
                            baseline_mode_confidence
                        )
                    _record_ablation_mode_metrics(
                        metrics,
                        mode_name=mode_name,
                        mode_confidence=parsed_mode_confidence,
                        confidence_delta=confidence_delta,
                        responses_identical=responses_identical,
                    )
                    if bucket_metrics is not None:
                        _record_ablation_mode_metrics(
                            bucket_metrics,
                            mode_name=mode_name,
                            mode_confidence=parsed_mode_confidence,
                            confidence_delta=confidence_delta,
                            responses_identical=responses_identical,
                        )
                    entry[key] = {
                        ABLATION_UNIT_KEY: {
                            "response": response,
                            "verbalised_confidence": parsed_mode_confidence,
                            "confidence_delta": confidence_delta,
                            "responses_identical": responses_identical,
                        }
                    }
                    logging.info(
                        "[%s %s %d/%d] %s %s/%s first line: %r",
                        group_name,
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                        key,
                        ABLATION_UNIT_KEY,
                        response[:120],
                    )

                results_mini[group_name][split_name][ex_id] = entry
                metrics["evaluated_count"] += 1

    evaluated_counts = {
        group_name: int(group_metrics[group_name]["evaluated_count"]) for group_name in CONFIDENCE_GROUPS
    }
    skipped_one_confidence = {
        group_name: int(group_metrics[group_name]["skipped_one_confidence"])
        for group_name in CONFIDENCE_GROUPS
    }
    mode_confidence = {
        group_name: group_metrics[group_name]["confidence"] for group_name in CONFIDENCE_GROUPS
    }
    mode_delta = {group_name: group_metrics[group_name]["delta"] for group_name in CONFIDENCE_GROUPS}
    mode_identical = {
        group_name: group_metrics[group_name]["identical"] for group_name in CONFIDENCE_GROUPS
    }
    by_none_mode_confidence = {
        group_name: group_metrics[group_name]["by_none_mode_confidence"]
        for group_name in CONFIDENCE_GROUPS
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_mini, f, ensure_ascii=False, indent=2)
    write_config_txt(
        config_txt_path(out_path),
        args=args,
        device=device,
        model_n_layers=model_n_layers,
        model_n_heads=model_n_heads,
        model_d_head=model_d_head,
        ablate_layers=run_layers,
        selected_heads_by_layer=selected_heads_by_layer,
        mlp_layers=mlp_layers,
        num_selected_layer_head_pairs=num_selected_layer_head_pairs,
        prompt_indices=prompt_indices,
        low_conf_count=low_conf_count,
        high_conf_count=high_conf_count,
        h5_example_count=h5_example_count,
        evaluated_counts=evaluated_counts,
        skipped_one_confidence=skipped_one_confidence,
        mode_confidence=mode_confidence,
        mode_delta=mode_delta,
        mode_identical=mode_identical,
        by_none_mode_confidence=by_none_mode_confidence,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    summary_path = summary_json_path(out_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            build_summary(
                run_root=run_root,
                args=args,
                ablate_layers=run_layers,
                selected_heads_by_layer=selected_heads_by_layer,
                mlp_layers=mlp_layers,
                num_selected_layer_head_pairs=num_selected_layer_head_pairs,
                low_conf_count=low_conf_count,
                high_conf_count=high_conf_count,
                h5_example_count=h5_example_count,
                evaluated_counts=evaluated_counts,
                skipped_one_confidence=skipped_one_confidence,
                mode_confidence=mode_confidence,
                mode_delta=mode_delta,
                mode_identical=mode_identical,
                by_none_mode_confidence=by_none_mode_confidence,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("Saved mini outputs to %s", out_path)
    logging.info("Wrote %s", config_txt_path(out_path))
    logging.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
