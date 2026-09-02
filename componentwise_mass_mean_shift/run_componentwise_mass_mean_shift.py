#!/usr/bin/env python3
"""Simultaneous componentwise mass-mean steering on high- and low-confidence groups.

Computes direction = high_mean - low_mean of selected attention heads, whole
attention blocks, and/or MLP subblocks, then additively steers those sites at
mode-dependent token spans during greedy decoding. --intervention_site selects
the patch location: input (Q/K/V post W_Q/W_K/W_V, attn ln1 from H5 res, and
MLP RMSNorm in) or output (hook_z / concat, hook_attn_out from H5 attn after
ln1_post on sandwich Gemma, and hook_mlp_out).

All units listed in --ablate_heads are steered together. Low-confidence examples
receive +alpha * direction; high-confidence examples receive -alpha * direction.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_zero_ablation.run_blockwise_zero_ablation import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT,
    construct_fewshot_prompt_from_indices,
    greedy_generate,
    load_eval_dataset,
    parse_ablate_layers,
    parse_mode_confidence_from_response,
    split_answerable_indices,
)
from componentwise_mean_ablation.run_componentwise_mean_ablation import (
    ABLATION_MODES_DEFAULT,
    INTERVENTION_SITES,
    MODES_NEEDING_PROBABILITY_EXTRA,
    _mlp_input_norm_module,
    _attn_input_norm_module,
    _positions_and_replacements_for_mode,
    _to_torch_means,
    stream_group_means_and_eval_ids,
)
from headwise_mean_ablation.run_headwise_mean_ablation import (
    resolve_n_key_value_heads,
    selected_kv_heads_by_layer,
)
from headwise_zero_ablation.run_headwise_zero_ablation import (
    ABLATION_UNIT_KEY,
    CONFIDENCE_GROUPS,
    NONE_MODE_CONFIDENCE_BUCKET_LABELS,
    NONE_MODE_CONFIDENCE_BUCKETS,
    RunningMean,
    _append_mode_metric_lines,
    _empty_group_metrics,
    _id_column_to_index_map,
    _modes_summary_from_trackers,
    _none_mode_confidence_bucket,
    _record_ablation_mode_metrics,
    _split_sample_targets,
    _sync_prefix_tokens_for_model,
    format_ablate_units,
    parse_ablate_units,
)
from layerwise_mean_ablation.run_mean_ablation import (
    load_hooked_transformer,
    validate_last_a_panl_and_pc_mode,
)
from mass_mean_probe.run_mass_mean_probe import _format_alpha


GROUP_STEERING_SIGN = {
    "low_confidence": 1.0,
    "high_confidence": -1.0,
}


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("componentwise_mass_mean_shift") / "results"
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


def alpha_key(alpha: float) -> str:
    return f"alpha_{_format_alpha(alpha)}"


def tracker_mode_name(mode: str, alpha: Optional[float] = None) -> str:
    if mode == "none":
        return "none"
    if alpha is None:
        raise ValueError("Non-none tracker modes require an alpha.")
    return f"{mode}__{alpha_key(alpha)}"


def expand_tracker_modes(ablation_modes: Sequence[str], alphas: Sequence[float]) -> List[str]:
    out: List[str] = []
    for mode in ablation_modes:
        if mode == "none":
            out.append("none")
            continue
        for alpha in alphas:
            out.append(tracker_mode_name(mode, alpha))
    return out


def _subtract_head_maps(
    high_map: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    low_map: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    *,
    component: str,
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    out: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
    for layer, high_heads in high_map.items():
        if int(layer) not in low_map:
            raise ValueError(f"Missing low-confidence {component} means for layer {layer}.")
        out[int(layer)] = {}
        low_heads = low_map[int(layer)]
        for head_idx, high_sources in high_heads.items():
            if int(head_idx) not in low_heads:
                raise ValueError(
                    f"Missing low-confidence {component} means for layer {layer} head {head_idx}."
                )
            low_sources = low_heads[int(head_idx)]
            entry: Dict[str, np.ndarray] = {}
            for source_name, high_arr in high_sources.items():
                if source_name not in low_sources:
                    continue
                low_arr = np.asarray(low_sources[source_name])
                high_np = np.asarray(high_arr)
                if high_np.shape != low_arr.shape:
                    raise ValueError(
                        f"{component} direction shape mismatch at layer {layer} head {head_idx} "
                        f"source {source_name}: high={high_np.shape} low={low_arr.shape}."
                    )
                entry[source_name] = (high_np - low_arr).astype(np.float32)
            if not entry:
                raise ValueError(
                    f"No overlapping {component} sources for layer {layer} head {head_idx}."
                )
            out[int(layer)][int(head_idx)] = entry
    return out


def _subtract_mlp_maps(
    high_map: Dict[int, Dict[str, np.ndarray]],
    low_map: Dict[int, Dict[str, np.ndarray]],
) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for layer, high_sources in high_map.items():
        if int(layer) not in low_map:
            raise ValueError(f"Missing low-confidence MLP means for layer {layer}.")
        low_sources = low_map[int(layer)]
        entry: Dict[str, np.ndarray] = {}
        for source_name, high_arr in high_sources.items():
            if source_name not in low_sources:
                continue
            low_arr = np.asarray(low_sources[source_name])
            high_np = np.asarray(high_arr)
            if high_np.shape != low_arr.shape:
                raise ValueError(
                    f"MLP direction shape mismatch at layer {layer} source {source_name}: "
                    f"high={high_np.shape} low={low_arr.shape}."
                )
            entry[source_name] = (high_np - low_arr).astype(np.float32)
        if not entry:
            raise ValueError(f"No overlapping MLP sources for layer {layer}.")
        out[int(layer)] = entry
    return out


def compute_high_minus_low_directions(
    packed_means: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    low_pack = packed_means["low_confidence"]
    high_pack = packed_means["high_confidence"]
    mlp_dir = _subtract_mlp_maps(high_pack["mlp"], low_pack["mlp"])  # type: ignore[arg-type]
    attn_block_dir = _subtract_mlp_maps(
        high_pack.get("attn_block", {}) or {},  # type: ignore[arg-type]
        low_pack.get("attn_block", {}) or {},  # type: ignore[arg-type]
    )
    if "z" in high_pack:
        z_dir = _subtract_head_maps(high_pack["z"], low_pack["z"], component="z")  # type: ignore[arg-type]
        return {
            "z": z_dir,
            "mlp": mlp_dir,
            "attn_block": attn_block_dir,
            "z_count": int(high_pack["z_count"]),
            "mlp_count": int(high_pack["mlp_count"]),
            "attn_block_count": int(high_pack.get("attn_block_count", 0)),
            "z_extra_count": min(int(high_pack["z_extra_count"]), int(low_pack["z_extra_count"])),
            "mlp_extra_count": min(int(high_pack["mlp_extra_count"]), int(low_pack["mlp_extra_count"])),
            "attn_block_extra_count": min(
                int(high_pack.get("attn_block_extra_count", 0)),
                int(low_pack.get("attn_block_extra_count", 0)),
            ),
            "intervention_site": "output",
        }
    q_dir = _subtract_head_maps(high_pack["q"], low_pack["q"], component="q")  # type: ignore[arg-type]
    k_dir = _subtract_head_maps(high_pack["k"], low_pack["k"], component="k")  # type: ignore[arg-type]
    v_dir = _subtract_head_maps(high_pack["v"], low_pack["v"], component="v")  # type: ignore[arg-type]
    return {
        "q": q_dir,
        "k": k_dir,
        "v": v_dir,
        "mlp": mlp_dir,
        "attn_block": attn_block_dir,
        "qkv_count": int(high_pack["qkv_count"]),
        "mlp_count": int(high_pack["mlp_count"]),
        "attn_block_count": int(high_pack.get("attn_block_count", 0)),
        "qkv_extra_count": min(int(high_pack["qkv_extra_count"]), int(low_pack["qkv_extra_count"])),
        "mlp_extra_count": min(int(high_pack["mlp_extra_count"]), int(low_pack["mlp_extra_count"])),
        "attn_block_extra_count": min(
            int(high_pack.get("attn_block_extra_count", 0)),
            int(low_pack.get("attn_block_extra_count", 0)),
        ),
        "intervention_site": "input",
    }


def build_qkv_input_direction_shift_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    selected_kv_heads: Dict[int, Sequence[int]],
    q_directions: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    k_directions: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    v_directions: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    scale = float(signed_alpha)

    def _make_component_hook(
        *,
        local_dirs: Dict[int, Dict[str, torch.Tensor]],
        heads: List[int],
        local_layer: int,
        component: str,
    ) -> Callable:
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            if activation.ndim != 4:
                raise ValueError(
                    f"Expected hook_{component} activation with shape "
                    f"[batch, seq, heads, d_head], got {tuple(activation.shape)}."
                )
            decoded_tokens = decoded_tokens_provider()
            for head_idx in heads:
                if not (0 <= head_idx < int(activation.shape[2])):
                    raise ValueError(
                        f"{component.upper()} head index {head_idx} out of range for layer {local_layer} "
                        f"with {int(activation.shape[2])} heads."
                    )
                head_dirs = local_dirs.get(head_idx)
                if head_dirs is None:
                    raise ValueError(
                        f"Missing {component} direction for layer {local_layer}, head {head_idx}."
                    )
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    source_means=head_dirs,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != int(activation.shape[3]):
                            raise ValueError(
                                f"{component} direction size {vector.numel()} != d_head {activation.shape[3]} "
                                f"at layer {local_layer} head {head_idx}."
                            )
                        activation[:, abs_pos, head_idx, :] = activation[:, abs_pos, head_idx, :] + (
                            scale * vector.to(device=activation.device, dtype=activation.dtype)
                        )
            return activation

        return hook_fn

    for layer, heads in selected_heads_by_layer.items():
        q_heads = [int(h) for h in heads]
        kv_heads = [int(h) for h in selected_kv_heads.get(int(layer), [])]
        if not q_heads:
            continue
        if not kv_heads:
            raise ValueError(f"No KV heads derived for query heads at layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.attn.hook_q",
                _make_component_hook(
                    local_dirs=q_directions[int(layer)],
                    heads=q_heads,
                    local_layer=int(layer),
                    component="q",
                ),
            )
        )
        hooks.append(
            (
                f"blocks.{int(layer)}.attn.hook_k",
                _make_component_hook(
                    local_dirs=k_directions[int(layer)],
                    heads=kv_heads,
                    local_layer=int(layer),
                    component="k",
                ),
            )
        )
        hooks.append(
            (
                f"blocks.{int(layer)}.attn.hook_v",
                _make_component_hook(
                    local_dirs=v_directions[int(layer)],
                    heads=kv_heads,
                    local_layer=int(layer),
                    component="v",
                ),
            )
        )
    return hooks


@contextmanager
def mlp_input_direction_shift_pre_hooks(
    model,
    *,
    mlp_layers: Sequence[int],
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    mlp_directions: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> Iterator[None]:
    handles = []
    scale = float(signed_alpha)

    def _make_pre_hook(layer_dirs: Dict[str, torch.Tensor], local_layer: int) -> Callable:
        def pre_hook(module, args):
            del module
            if not args:
                return None
            x = args[0]
            positions, vectors = _positions_and_replacements_for_mode(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens=decoded_tokens_provider(),
                source_means=layer_dirs,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if not positions:
                return None
            x = x.clone()
            for abs_pos, vector in zip(positions, vectors):
                if 0 <= abs_pos < x.shape[1]:
                    if int(vector.numel()) != int(x.shape[-1]):
                        raise ValueError(
                            f"MLP-input direction size {vector.numel()} != d_model {x.shape[-1]} "
                            f"at layer {local_layer}."
                        )
                    x[:, abs_pos, :] = x[:, abs_pos, :] + (
                        scale * vector.to(device=x.device, dtype=x.dtype)
                    )
            if len(args) == 1:
                return (x,)
            return (x, *args[1:])

        return pre_hook

    try:
        for layer in mlp_layers:
            module = _mlp_input_norm_module(model, int(layer))
            layer_dirs = mlp_directions.get(int(layer))
            if layer_dirs is None:
                raise ValueError(f"Missing MLP-input direction for layer {layer}.")
            handles.append(module.register_forward_pre_hook(_make_pre_hook(layer_dirs, int(layer))))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def attn_input_direction_shift_pre_hooks(
    model,
    *,
    attn_block_layers: Sequence[int],
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    attn_block_directions: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> Iterator[None]:
    handles = []
    scale = float(signed_alpha)

    def _make_pre_hook(layer_dirs: Dict[str, torch.Tensor], local_layer: int) -> Callable:
        def pre_hook(module, args):
            del module
            if not args:
                return None
            x = args[0]
            positions, vectors = _positions_and_replacements_for_mode(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens=decoded_tokens_provider(),
                source_means=layer_dirs,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if not positions:
                return None
            x = x.clone()
            for abs_pos, vector in zip(positions, vectors):
                if 0 <= abs_pos < x.shape[1]:
                    if int(vector.numel()) != int(x.shape[-1]):
                        raise ValueError(
                            f"Attn-block-input direction size {vector.numel()} != d_model {x.shape[-1]} "
                            f"at layer {local_layer}."
                        )
                    x[:, abs_pos, :] = x[:, abs_pos, :] + (
                        scale * vector.to(device=x.device, dtype=x.dtype)
                    )
            if len(args) == 1:
                return (x,)
            return (x, *args[1:])

        return pre_hook

    try:
        for layer in attn_block_layers:
            module = _attn_input_norm_module(model, int(layer))
            layer_dirs = attn_block_directions.get(int(layer))
            if layer_dirs is None:
                raise ValueError(f"Missing attn-block-input direction for layer {layer}.")
            handles.append(module.register_forward_pre_hook(_make_pre_hook(layer_dirs, int(layer))))
        yield
    finally:
        for handle in handles:
            handle.remove()


def build_z_output_direction_shift_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    z_directions: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    scale = float(signed_alpha)

    def _make_z_hook(
        *,
        local_dirs: Dict[int, Dict[str, torch.Tensor]],
        heads: List[int],
        local_layer: int,
    ) -> Callable:
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            if activation.ndim != 4:
                raise ValueError(
                    "Expected hook_z activation with shape "
                    f"[batch, seq, heads, d_head], got {tuple(activation.shape)}."
                )
            decoded_tokens = decoded_tokens_provider()
            for head_idx in heads:
                if not (0 <= head_idx < int(activation.shape[2])):
                    raise ValueError(
                        f"Z head index {head_idx} out of range for layer {local_layer} "
                        f"with {int(activation.shape[2])} heads."
                    )
                head_dirs = local_dirs.get(head_idx)
                if head_dirs is None:
                    raise ValueError(
                        f"Missing concat/hook_z direction for layer {local_layer}, head {head_idx}."
                    )
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    source_means=head_dirs,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != int(activation.shape[3]):
                            raise ValueError(
                                f"hook_z direction size {vector.numel()} != d_head {activation.shape[3]} "
                                f"at layer {local_layer} head {head_idx}."
                            )
                        activation[:, abs_pos, head_idx, :] = activation[:, abs_pos, head_idx, :] + (
                            scale * vector.to(device=activation.device, dtype=activation.dtype)
                        )
            return activation

        return hook_fn

    for layer, heads in selected_heads_by_layer.items():
        z_heads = [int(h) for h in heads]
        if not z_heads:
            continue
        layer_dirs = z_directions.get(int(layer))
        if layer_dirs is None:
            raise ValueError(f"Missing concat/hook_z direction for layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.attn.hook_z",
                _make_z_hook(
                    local_dirs=layer_dirs,
                    heads=z_heads,
                    local_layer=int(layer),
                ),
            )
        )
    return hooks


def build_mlp_output_direction_shift_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    mlp_layers: Sequence[int],
    mlp_directions: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    scale = float(signed_alpha)

    def _make_mlp_out_hook(layer_dirs: Dict[str, torch.Tensor], local_layer: int) -> Callable:
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            positions, vectors = _positions_and_replacements_for_mode(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens=decoded_tokens_provider(),
                source_means=layer_dirs,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if not positions:
                return activation
            for abs_pos, vector in zip(positions, vectors):
                if 0 <= abs_pos < activation.shape[1]:
                    if int(vector.numel()) != int(activation.shape[-1]):
                        raise ValueError(
                            f"MLP-output direction size {vector.numel()} != d_model {activation.shape[-1]} "
                            f"at layer {local_layer}."
                        )
                    activation[:, abs_pos, :] = activation[:, abs_pos, :] + (
                        scale * vector.to(device=activation.device, dtype=activation.dtype)
                    )
            return activation

        return hook_fn

    for layer in mlp_layers:
        layer_dirs = mlp_directions.get(int(layer))
        if layer_dirs is None:
            raise ValueError(f"Missing MLP-output direction for layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.hook_mlp_out",
                _make_mlp_out_hook(layer_dirs, int(layer)),
            )
        )
    return hooks


def build_attn_output_direction_shift_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    attn_block_layers: Sequence[int],
    attn_block_directions: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    scale = float(signed_alpha)

    def _make_attn_out_hook(layer_dirs: Dict[str, torch.Tensor], local_layer: int) -> Callable:
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            positions, vectors = _positions_and_replacements_for_mode(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens=decoded_tokens_provider(),
                source_means=layer_dirs,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if not positions:
                return activation
            for abs_pos, vector in zip(positions, vectors):
                if 0 <= abs_pos < activation.shape[1]:
                    if int(vector.numel()) != int(activation.shape[-1]):
                        raise ValueError(
                            f"Attn-block-output direction size {vector.numel()} != d_model "
                            f"{activation.shape[-1]} at layer {local_layer}."
                        )
                    activation[:, abs_pos, :] = activation[:, abs_pos, :] + (
                        scale * vector.to(device=activation.device, dtype=activation.dtype)
                    )
            return activation

        return hook_fn

    for layer in attn_block_layers:
        layer_dirs = attn_block_directions.get(int(layer))
        if layer_dirs is None:
            raise ValueError(f"Missing attn-block-output direction for layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.hook_attn_out",
                _make_attn_out_hook(layer_dirs, int(layer)),
            )
        )
    return hooks


def greedy_generate_componentwise_mass_mean_shifted(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    mode: str,
    model_name: str,
    intervention_site: str,
    selected_heads_by_layer: Dict[int, Sequence[int]],
    selected_kv_heads: Dict[int, Sequence[int]],
    mlp_layers: Sequence[int],
    attn_block_layers: Sequence[int],
    directions: Dict[str, object],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    signed_alpha: float,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks: List[Tuple[str, Callable]] = []
    if intervention_site == "output":
        if selected_heads_by_layer:
            hooks.extend(
                build_z_output_direction_shift_hooks(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    selected_heads_by_layer=selected_heads_by_layer,
                    z_directions=directions["z"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    signed_alpha=signed_alpha,
                )
            )
        if attn_block_layers:
            hooks.extend(
                build_attn_output_direction_shift_hooks(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    attn_block_layers=attn_block_layers,
                    attn_block_directions=directions["attn_block"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    signed_alpha=signed_alpha,
                )
            )
        if mlp_layers:
            hooks.extend(
                build_mlp_output_direction_shift_hooks(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    mlp_layers=mlp_layers,
                    mlp_directions=directions["mlp"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    signed_alpha=signed_alpha,
                )
            )
        if not hooks:
            raise ValueError(
                "No attention heads, attention blocks, or MLP layers selected for "
                "componentwise mass-mean shift."
            )
        return greedy_generate(
            model=model,
            local_prompt=local_prompt,
            max_new_tokens=max_new_tokens,
            fwd_hooks=hooks,
            decoded_tokens_buffer=decoded_tokens,
        )

    if selected_heads_by_layer:
        hooks.extend(
            build_qkv_input_direction_shift_hooks(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens_provider=_decoded_tokens_provider,
                selected_heads_by_layer=selected_heads_by_layer,
                selected_kv_heads=selected_kv_heads,
                q_directions=directions["q"],  # type: ignore[arg-type]
                k_directions=directions["k"],  # type: ignore[arg-type]
                v_directions=directions["v"],  # type: ignore[arg-type]
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
                signed_alpha=signed_alpha,
            )
        )
    if not hooks and not mlp_layers and not attn_block_layers:
        raise ValueError(
            "No attention heads, attention blocks, or MLP layers selected for "
            "componentwise mass-mean shift."
        )
    with ExitStack() as stack:
        if attn_block_layers:
            stack.enter_context(
                attn_input_direction_shift_pre_hooks(
                    model,
                    attn_block_layers=attn_block_layers,
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    attn_block_directions=directions["attn_block"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    signed_alpha=signed_alpha,
                )
            )
        if mlp_layers:
            stack.enter_context(
                mlp_input_direction_shift_pre_hooks(
                    model,
                    mlp_layers=mlp_layers,
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    mlp_directions=directions["mlp"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    signed_alpha=signed_alpha,
                )
            )
        return greedy_generate(
            model=model,
            local_prompt=local_prompt,
            max_new_tokens=max_new_tokens,
            fwd_hooks=hooks,
            decoded_tokens_buffer=decoded_tokens,
        )


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    model_n_heads: int,
    model_n_kv_heads: int,
    model_d_head: int,
    ablate_layers: Sequence[int],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mlp_layers: Sequence[int],
    attn_block_layers: Sequence[int],
    num_selected_layer_head_pairs: int,
    tracker_modes: Sequence[str],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    qkv_mean_counts: Dict[str, int],
    mlp_mean_counts: Dict[str, int],
    qkv_extra_counts: Dict[str, int],
    mlp_extra_counts: Dict[str, int],
    attn_block_mean_counts: Dict[str, int],
    attn_block_extra_counts: Dict[str, int],
    evaluated_counts: Dict[str, int],
    skipped_one_confidence: Dict[str, int],
    mode_confidence: Dict[str, Dict[str, RunningMean]],
    mode_delta: Dict[str, Dict[str, RunningMean]],
    mode_identical: Dict[str, Dict[str, int]],
    by_none_mode_confidence: Dict[str, Dict[str, Dict[str, object]]],
    finished_at: str,
) -> None:
    lines = [
        "Componentwise Mass-Mean Shift Configuration",
        "==========================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"model_n_heads={model_n_heads}",
        f"model_n_kv_heads={model_n_kv_heads}",
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
        f"intervention_site={args.intervention_site}",
        "direction_definition=high_mean_minus_low_mean",
        "non_none_mode_behavior=additive_direction_perturbation",
        "confidence_direction_expectation_for_low_targets=perturbed_confidence_gt_none",
        "confidence_direction_expectation_for_high_targets=perturbed_confidence_lt_none",
        f"alpha={list(args.alpha)}",
        f"ablation_mode={args.ablation_mode}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"ablate_heads_spec={args.ablate_heads}",
        f"ablate_heads_resolved={format_ablate_units(selected_heads_by_layer, mlp_layers, attn_block_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"num_ablated_layer_head_pairs={num_selected_layer_head_pairs}",
        f"num_ablated_mlp_layers={len(mlp_layers)}",
        f"num_ablated_attn_blocks={len(attn_block_layers)}",
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
        f"low_conf_qkv_mean_count={qkv_mean_counts.get('low_confidence', 0)}",
        f"high_conf_qkv_mean_count={qkv_mean_counts.get('high_confidence', 0)}",
        f"low_conf_mlp_mean_count={mlp_mean_counts.get('low_confidence', 0)}",
        f"high_conf_mlp_mean_count={mlp_mean_counts.get('high_confidence', 0)}",
        f"low_conf_qkv_extra_mean_count={qkv_extra_counts.get('low_confidence', 0)}",
        f"high_conf_qkv_extra_mean_count={qkv_extra_counts.get('high_confidence', 0)}",
        f"low_conf_mlp_extra_mean_count={mlp_extra_counts.get('low_confidence', 0)}",
        f"high_conf_mlp_extra_mean_count={mlp_extra_counts.get('high_confidence', 0)}",
        f"low_conf_attn_block_mean_count={attn_block_mean_counts.get('low_confidence', 0)}",
        f"high_conf_attn_block_mean_count={attn_block_mean_counts.get('high_confidence', 0)}",
        f"low_conf_attn_block_extra_mean_count={attn_block_extra_counts.get('low_confidence', 0)}",
        f"high_conf_attn_block_extra_mean_count={attn_block_extra_counts.get('high_confidence', 0)}",
        "",
        "[Mode Confidence Metrics]",
        "Values below are running-mean verbalised confidence per group.",
        "Low-confidence examples are steered with +alpha * (high_mean - low_mean);",
        "high-confidence examples are steered with -alpha * (high_mean - low_mean).",
        "Additional sections split each group by none-mode verbalised confidence",
        "(eq_1: exactly 1.0; lt_1: parsed and < 1.0).",
        "",
    ]
    for group_name in CONFIDENCE_GROUPS:
        lines.append(f"[{group_name}]")
        _append_mode_metric_lines(
            lines,
            ablation_modes=tracker_modes,
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
                    ablation_modes=tracker_modes,
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
    attn_block_layers: Sequence[int],
    num_selected_layer_head_pairs: int,
    tracker_modes: Sequence[str],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    qkv_mean_counts: Dict[str, int],
    mlp_mean_counts: Dict[str, int],
    attn_block_mean_counts: Dict[str, int],
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
            tracker_modes,
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
                    tracker_modes,
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
            "qkv_mean_count": int(qkv_mean_counts.get(group_name, 0)),
            "mlp_mean_count": int(mlp_mean_counts.get(group_name, 0)),
            "attn_block_mean_count": int(attn_block_mean_counts.get(group_name, 0)),
            "steering_sign": float(GROUP_STEERING_SIGN[group_name]),
            "modes": modes_out,
            "by_none_mode_confidence": none_mode_buckets,
        }
    return {
        "run_root": run_root,
        "dataset": args.dataset,
        "input_h5": args.input_h5,
        "h5_example_count": h5_example_count,
        "intervention_site": args.intervention_site,
        "direction_definition": "high_mean_minus_low_mean",
        "alpha": [float(a) for a in args.alpha],
        "ablate_layers": list(ablate_layers),
        "ablate_heads": format_ablate_units(selected_heads_by_layer, mlp_layers, attn_block_layers),
        "ablate_mlp_layers": list(mlp_layers),
        "ablate_attn_block_layers": list(attn_block_layers),
        "num_ablated_layer_head_pairs": num_selected_layer_head_pairs,
        "num_ablated_mlp_layers": len(mlp_layers),
        "num_ablated_attn_blocks": len(attn_block_layers),
        "ablation_modes": list(args.ablation_mode),
        "skip_one_confidence": bool(args.skip_one_confidence),
        "low_conf_threshold": args.low_conf_threshold,
        "high_conf_threshold": args.high_conf_threshold,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simultaneous componentwise mass-mean shift: steer selected head Q/K/V or hook_z, "
            "whole attention blocks (ln1 / hook_attn_out), and/or MLP RMSNorm inputs or "
            "hook_mlp_out along (high_mean - low_mean) with a list of alphas."
        )
    )
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help=(
            "Path to processed verbalised embedding H5. Input site uses q/k/v, res (whole attn), "
            "and res+attn (MLP); output site uses concat (hook_z), attn (hook_attn_out), "
            "and mlp under embeddings_*."
        ),
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
            "Whole-block a<layer> steers ln1 with H5 res (input) or hook_attn_out with H5 attn "
            "(output; after ln1_post on sandwich Gemma). "
            "All listed units are steered simultaneously at --intervention_site. "
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
    parser.add_argument(
        "--intervention_site",
        type=str,
        default="input",
        choices=list(INTERVENTION_SITES),
        help=(
            "Steer component inputs (Q/K/V, attn ln1 from H5 res, and MLP RMSNorm in) or outputs "
            "(hook_z, hook_attn_out from H5 attn, and hook_mlp_out)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        required=True,
        help="One or more real-valued alpha scale factors (perturbation strength along the direction).",
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
    if not args.alpha:
        raise ValueError("--alpha must include at least one scale factor.")
    validate_last_a_panl_and_pc_mode(args.model_name, args.ablation_mode)
    formatted_alphas = [_format_alpha(float(a)) for a in args.alpha]
    if len(set(formatted_alphas)) != len(formatted_alphas):
        raise ValueError(f"Duplicate --alpha values are not allowed: {args.alpha}")

    tracker_modes = expand_tracker_modes(args.ablation_mode, args.alpha)

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
    model_n_kv_heads = resolve_n_key_value_heads(model.cfg, model_n_heads)
    kv_heads_per_query_group = model_n_heads // model_n_kv_heads

    ablate_layers_from_flag = parse_ablate_layers(args.ablate_layers, model_n_layers)
    if not ablate_layers_from_flag and not args.ablate_heads:
        raise ValueError("No layers selected via --ablate_layers.")

    if args.ablate_heads:
        selected_heads_by_layer, mlp_layers, attn_block_layers = parse_ablate_units(
            args.ablate_heads, n_layers=model_n_layers, n_heads=model_n_heads
        )
        run_layers = sorted(
            set(selected_heads_by_layer.keys()) | set(mlp_layers) | set(attn_block_layers)
        )
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

    missing_layer_heads = [layer for layer, heads in selected_heads_by_layer.items() if not heads]
    if missing_layer_heads:
        raise ValueError(
            "No selected heads provided for layers in this run: "
            + ",".join(str(layer) for layer in missing_layer_heads)
        )
    if not selected_heads_by_layer and not mlp_layers and not attn_block_layers:
        raise ValueError(
            "No attention heads, attention blocks, or MLP layers selected for "
            "componentwise mass-mean shift."
        )

    selected_kv_heads = (
        selected_kv_heads_by_layer(
            selected_heads_by_layer,
            kv_heads_per_query_group=kv_heads_per_query_group,
        )
        if selected_heads_by_layer
        else {}
    )
    num_selected_layer_head_pairs = sum(len(heads) for heads in selected_heads_by_layer.values())
    resolved_units = format_ablate_units(selected_heads_by_layer, mlp_layers, attn_block_layers)
    logging.info(
        "Steering units=%s (%d layer-head pairs, %d attn blocks, %d mlp layers) at component %ss. alphas=%s",
        resolved_units,
        num_selected_layer_head_pairs,
        len(attn_block_layers),
        len(mlp_layers),
        args.intervention_site,
        [float(a) for a in args.alpha],
    )
    if MODES_NEEDING_PROBABILITY_EXTRA.intersection(args.ablation_mode):
        logging.info(
            "Modes %s use extra probability-span tokens when present in H5.",
            sorted(MODES_NEEDING_PROBABILITY_EXTRA.intersection(args.ablation_mode)),
        )

    packed_means, selected_ids_by_group, h5_example_count = stream_group_means_and_eval_ids(
        args.input_h5,
        selected_heads_by_layer=selected_heads_by_layer,
        mlp_layers=mlp_layers,
        attn_block_layers=attn_block_layers,
        n_heads=model_n_heads,
        n_kv_heads=model_n_kv_heads,
        d_head=model_d_head,
        d_model=model_d_model,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        split_id_to_index=split_id_to_index,
        split_targets=split_targets,
        intervention_site=args.intervention_site,
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
    qkv_mean_counts = {
        group_name: int(
            packed_means[group_name].get("z_count", packed_means[group_name].get("qkv_count", 0))
        )
        for group_name in CONFIDENCE_GROUPS
    }
    mlp_mean_counts = {
        group_name: int(packed_means[group_name]["mlp_count"]) for group_name in CONFIDENCE_GROUPS
    }
    qkv_extra_counts = {
        group_name: int(
            packed_means[group_name].get(
                "z_extra_count", packed_means[group_name].get("qkv_extra_count", 0)
            )
        )
        for group_name in CONFIDENCE_GROUPS
    }
    mlp_extra_counts = {
        group_name: int(packed_means[group_name]["mlp_extra_count"]) for group_name in CONFIDENCE_GROUPS
    }
    attn_block_mean_counts = {
        group_name: int(packed_means[group_name].get("attn_block_count", 0))
        for group_name in CONFIDENCE_GROUPS
    }
    attn_block_extra_counts = {
        group_name: int(packed_means[group_name].get("attn_block_extra_count", 0))
        for group_name in CONFIDENCE_GROUPS
    }
    logging.info(
        "H5 has %d examples. selected low_conf=%d (<=%.3f), high_conf=%d (>=%.3f). "
        "QKV/concat mean n=%s MLP mean n=%s attn_block mean n=%s site=%s.",
        h5_example_count,
        low_conf_count,
        args.low_conf_threshold,
        high_conf_count,
        args.high_conf_threshold,
        qkv_mean_counts,
        mlp_mean_counts,
        attn_block_mean_counts,
        args.intervention_site,
    )
    direction_packed = compute_high_minus_low_directions(packed_means)
    direction_torch = _to_torch_means(direction_packed, device=device, torch_dtype=torch_dtype)
    del packed_means, direction_packed
    gc.collect()

    results_mini: Dict[str, Dict[str, dict]] = {
        group_name: {"train": {}, "validation": {}} for group_name in CONFIDENCE_GROUPS
    }
    group_metrics = {group_name: _empty_group_metrics(tracker_modes) for group_name in CONFIDENCE_GROUPS}

    for group_name in CONFIDENCE_GROUPS:
        metrics = group_metrics[group_name]
        signed_base = float(GROUP_STEERING_SIGN[group_name])
        logging.info(
            "Steering group=%s (%d selected H5 ids) with sign=%+.1f * alpha * (high-low); "
            "%d heads, %d attn blocks, and %d MLP layers shifted simultaneously.",
            group_name,
            sum(len(ids) for ids in selected_ids_by_group[group_name].values()),
            signed_base,
            num_selected_layer_head_pairs,
            len(attn_block_layers),
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

                    alpha_entries: Dict[str, dict] = {}
                    for alpha in args.alpha:
                        signed_alpha = signed_base * float(alpha)
                        tracker_name = tracker_mode_name(mode_name, float(alpha))
                        response, _ = greedy_generate_componentwise_mass_mean_shifted(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            mode=mode_name,
                            model_name=args.model_name,
                            intervention_site=args.intervention_site,
                            selected_heads_by_layer=selected_heads_by_layer,
                            selected_kv_heads=selected_kv_heads,
                            mlp_layers=mlp_layers,
                            attn_block_layers=attn_block_layers,
                            directions=direction_torch,
                            expected_guess_tokens=args.expected_guess_tokens,
                            expected_probability_tokens=args.expected_probability_tokens,
                            signed_alpha=signed_alpha,
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
                            mode_name=tracker_name,
                            mode_confidence=parsed_mode_confidence,
                            confidence_delta=confidence_delta,
                            responses_identical=responses_identical,
                        )
                        if bucket_metrics is not None:
                            _record_ablation_mode_metrics(
                                bucket_metrics,
                                mode_name=tracker_name,
                                mode_confidence=parsed_mode_confidence,
                                confidence_delta=confidence_delta,
                                responses_identical=responses_identical,
                            )
                        a_key = alpha_key(float(alpha))
                        alpha_entries[a_key] = {
                            "response": response,
                            "verbalised_confidence": parsed_mode_confidence,
                            "confidence_delta": confidence_delta,
                            "responses_identical": responses_identical,
                            "alpha": float(alpha),
                            "signed_alpha": signed_alpha,
                        }
                        logging.info(
                            "[%s %s %d/%d] %s %s/%s/%s first line: %r",
                            group_name,
                            split_name,
                            i + 1,
                            len(selected_ids),
                            ex_id,
                            key,
                            ABLATION_UNIT_KEY,
                            a_key,
                            response[:120],
                        )
                    entry[key] = {ABLATION_UNIT_KEY: alpha_entries}

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
        model_n_kv_heads=model_n_kv_heads,
        model_d_head=model_d_head,
        ablate_layers=run_layers,
        selected_heads_by_layer=selected_heads_by_layer,
        mlp_layers=mlp_layers,
        attn_block_layers=attn_block_layers,
        num_selected_layer_head_pairs=num_selected_layer_head_pairs,
        tracker_modes=tracker_modes,
        prompt_indices=prompt_indices,
        low_conf_count=low_conf_count,
        high_conf_count=high_conf_count,
        h5_example_count=h5_example_count,
        qkv_mean_counts=qkv_mean_counts,
        mlp_mean_counts=mlp_mean_counts,
        qkv_extra_counts=qkv_extra_counts,
        mlp_extra_counts=mlp_extra_counts,
        attn_block_mean_counts=attn_block_mean_counts,
        attn_block_extra_counts=attn_block_extra_counts,
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
                attn_block_layers=attn_block_layers,
                num_selected_layer_head_pairs=num_selected_layer_head_pairs,
                tracker_modes=tracker_modes,
                low_conf_count=low_conf_count,
                high_conf_count=high_conf_count,
                h5_example_count=h5_example_count,
                qkv_mean_counts=qkv_mean_counts,
                mlp_mean_counts=mlp_mean_counts,
                attn_block_mean_counts=attn_block_mean_counts,
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
