#!/usr/bin/env python3
"""Simultaneous componentwise mean ablation on high- and low-confidence groups.

Mean-replaces selected attention heads, whole attention blocks, and/or MLP
subblocks at mode-dependent token spans during greedy decoding.
--intervention_site selects the patch location: input (Q/K/V post W_Q/W_K/W_V,
attn ln1 from H5 res, and MLP RMSNorm in) or output (hook_z / concat,
hook_attn_out from H5 attn after ln1_post on sandwich Gemma, and hook_mlp_out).

Means are computed separately for high- and low-confidence H5 examples, then
applied cross-group: high-confidence generations receive the low-confidence
mean, and vice versa. All units listed in --ablate_heads are patched together.
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

import h5py
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
    _absolute_prob_positions_at_row_indices,
    construct_fewshot_prompt_from_indices,
    greedy_generate,
    load_eval_dataset,
    parse_ablate_layers,
    parse_mode_confidence_from_response,
    split_answerable_indices,
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
    TRAIN_RATIO,
    _absolute_extended_prob_last_token_only_positions,
    _append_mode_metric_lines,
    _empty_group_metrics,
    _id_column_to_index_map,
    _modes_summary_from_trackers,
    _none_mode_confidence_bucket,
    _open_h5_readonly,
    _record_ablation_mode_metrics,
    _selected_groups_filled,
    _split_sample_targets,
    _sync_prefix_tokens_for_model,
    format_ablate_units,
    parse_ablate_units,
)
from layerwise_mean_ablation.run_mean_ablation import (
    PROBABILITY_ROW_INDEX_MODES,
    _absolute_probability_value_start_position,
    _as_layer_hidden,
    _is_expected_or_plus_two,
    load_hooked_transformer,
    probability_row_indices_for_mode,
    validate_last_a_panl_and_pc_mode,
)


OPPOSITE_GROUP = {
    "low_confidence": "high_confidence",
    "high_confidence": "low_confidence",
}
SCALAR_SOURCES = ("prompt_mean", "sem_answer_mean", "probability_value_mean")
LIST_SOURCES = ("guess", "probability")
ALL_SOURCES = SCALAR_SOURCES + LIST_SOURCES
SOURCE_TO_FIELD = {
    "prompt_mean": "embeddings_mean_prompt",
    "guess": "embeddings_guess",
    "sem_answer_mean": "embeddings_mean_sem_answer",
    "probability": "embeddings_probability",
    "probability_value_mean": "embeddings_mean_prob_val",
}
INTERVENTION_SITES = ("input", "output")
ABLATION_MODES_DEFAULT = [
    "none",
    "probability_tokens_mean_replace",
    "probability_last_token_mean_replace",
    "extended_probability_last_token_mean_replace",
    "probability_pre_and_post_period_digit_mean_replace",
    "probability_span_except_last_token_mean_replace",
    "last_a_mean_replace",
    "last_a_and_panl_mean_replace",
    "last_a_panl_and_pc_mean_replace",
    "panl_mean_replace",
    "pc_mean_replace",
    "all_pre_probability_tokens_mean_replace",
    "guess_tokens_mean_replace",
    "all_pre_guess_tokens_mean_replace",
    "guess_then_guess_probability_mean_replace",
    "probability_value_mean_replace",
    "current_generated_token_mean_replace",
]
MODES_NEEDING_PROBABILITY_EXTRA = frozenset(
    {
        "extended_probability_last_token_mean_replace",
        "probability_pre_and_post_period_digit_mean_replace",
    }
)


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("componentwise_mean_ablation") / "results"
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


def _h5_node_type(node) -> str:
    raw = node.attrs.get("__type__", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


def _read_layer_hidden_dataset(node) -> np.ndarray:
    if not isinstance(node, h5py.Dataset):
        raise ValueError("Expected an HDF5 dataset for an embedding tensor.")
    return _as_layer_hidden(np.asarray(node[()], dtype=np.float32))


def _read_field_component(
    r0: h5py.Group,
    field_name: str,
    component: str,
    *,
    expect_list: bool,
) -> np.ndarray | List[np.ndarray]:
    field = r0.get(field_name)
    if field is None or not isinstance(field, h5py.Group):
        raise ValueError(f"{field_name} is missing or not a group.")
    comp = field.get(component)
    if comp is None:
        raise ValueError(f"{field_name}/{component} is missing.")
    if isinstance(comp, h5py.Group):
        node_type = _h5_node_type(comp)
        if node_type == "none":
            raise ValueError(f"{field_name}/{component} is null.")
        if expect_list:
            if node_type not in ("list", "tuple"):
                raise ValueError(
                    f"{field_name}/{component} expected a list, got type {node_type!r}."
                )
            length = int(comp.attrs.get("__len__", len(comp.keys())))
            items: List[np.ndarray] = []
            for i in range(length):
                key = str(i)
                if key not in comp:
                    raise ValueError(f"{field_name}/{component}/{i} is missing.")
                items.append(_read_layer_hidden_dataset(comp[key]))
            return items
        raise ValueError(
            f"{field_name}/{component} expected a tensor dataset, got group type {node_type!r}."
        )
    if expect_list:
        raise ValueError(f"{field_name}/{component} expected a list group, got a dataset.")
    return _read_layer_hidden_dataset(comp)


def _add_sum(store: Dict[str, np.ndarray], key: str, value: np.ndarray) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if key not in store:
        store[key] = np.array(arr, dtype=np.float64, copy=True)
        return
    if store[key].shape != arr.shape:
        raise ValueError(
            f"Running-sum shape mismatch for {key}: {store[key].shape} vs {arr.shape}."
        )
    store[key] += arr


def _reshape_q_row(hidden: np.ndarray, *, layer: int, n_heads: int, d_head: int) -> np.ndarray:
    if hidden.ndim != 2:
        raise ValueError(f"Expected [layers, hidden] Q tensor, got {hidden.shape}.")
    if int(hidden.shape[0]) <= int(layer):
        raise ValueError(
            f"Q embeddings have {hidden.shape[0]} layers; need index {layer}."
        )
    row = hidden[int(layer)]
    expected = int(n_heads) * int(d_head)
    if int(row.size) != expected:
        raise ValueError(
            f"Q hidden dim {row.size} does not match n_heads*d_head={expected}."
        )
    return np.reshape(row, (int(n_heads), int(d_head)))


def _reshape_kv_row(
    hidden: np.ndarray, *, layer: int, n_kv_heads: int, d_head: int
) -> np.ndarray:
    if hidden.ndim != 2:
        raise ValueError(f"Expected [layers, hidden] KV tensor, got {hidden.shape}.")
    if int(hidden.shape[0]) <= int(layer):
        raise ValueError(
            f"KV embeddings have {hidden.shape[0]} layers; need index {layer}."
        )
    row = hidden[int(layer)]
    expected = int(n_kv_heads) * int(d_head)
    if int(row.size) != expected:
        raise ValueError(
            f"KV hidden dim {row.size} does not match n_kv_heads*d_head={expected}."
        )
    return np.reshape(row, (int(n_kv_heads), int(d_head)))


def _mlp_input_row(res_hidden: np.ndarray, attn_hidden: np.ndarray, *, layer: int, d_model: int) -> np.ndarray:
    if res_hidden.ndim != 2 or attn_hidden.ndim != 2:
        raise ValueError(
            f"Expected 2D res/attn, got res={res_hidden.shape} attn={attn_hidden.shape}."
        )
    if int(res_hidden.shape[0]) <= int(layer):
        raise ValueError(
            f"res embeddings have {res_hidden.shape[0]} rows; need index {layer} "
            "(HF res[L] is resid-pre of TransformerLens block L)."
        )
    if int(attn_hidden.shape[0]) <= int(layer):
        raise ValueError(
            f"attn embeddings have {attn_hidden.shape[0]} layers; need index {layer}."
        )
    res_row = res_hidden[int(layer)]
    attn_row = attn_hidden[int(layer)]
    if int(res_row.size) != int(d_model) or int(attn_row.size) != int(d_model):
        raise ValueError(
            f"MLP input hidden dim mismatch at layer {layer}: "
            f"res={res_row.size} attn={attn_row.size} d_model={d_model}."
        )
    return np.asarray(res_row, dtype=np.float32) + np.asarray(attn_row, dtype=np.float32)


def _d_model_row(
    hidden: np.ndarray, *, layer: int, d_model: int, component: str
) -> np.ndarray:
    if hidden.ndim != 2:
        raise ValueError(f"Expected 2D {component}, got {hidden.shape}.")
    if int(hidden.shape[0]) <= int(layer):
        raise ValueError(
            f"{component} embeddings have {hidden.shape[0]} layers; need index {layer}."
        )
    row = hidden[int(layer)]
    if int(row.size) != int(d_model):
        raise ValueError(
            f"{component} hidden dim mismatch at layer {layer}: "
            f"{component}={row.size} d_model={d_model}."
        )
    return np.asarray(row, dtype=np.float32)


def _split_source_token_list(
    raw,
    *,
    source_name: str,
    field_name: str,
    component: str,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    extra_tokens: Optional[List[np.ndarray]] = None
    if source_name == "guess":
        if not isinstance(raw, list) or len(raw) != expected_guess_tokens:
            raise ValueError(
                f"{field_name}/{component} len={0 if not isinstance(raw, list) else len(raw)}; "
                f"expected {expected_guess_tokens}."
            )
        return raw, None
    if source_name == "probability":
        if not isinstance(raw, list) or not _is_expected_or_plus_two(
            len(raw), expected_probability_tokens
        ):
            raise ValueError(
                f"{field_name}/{component} len={0 if not isinstance(raw, list) else len(raw)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
            )
        tokens = raw[:expected_probability_tokens]
        if len(raw) == expected_probability_tokens + 2:
            extra_tokens = raw[expected_probability_tokens:]
        return tokens, extra_tokens
    return [raw], None


def _extract_qkv_source_arrays(
    r0: h5py.Group,
    *,
    source_name: str,
    selected_heads_by_layer: Dict[int, Sequence[int]],
    selected_kv_heads: Dict[int, Sequence[int]],
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Optional[Dict[str, Dict[int, np.ndarray]]]]:
    field_name = SOURCE_TO_FIELD[source_name]
    expect_list = source_name in LIST_SOURCES
    q_raw = _read_field_component(r0, field_name, "q", expect_list=expect_list)
    k_raw = _read_field_component(r0, field_name, "k", expect_list=expect_list)
    v_raw = _read_field_component(r0, field_name, "v", expect_list=expect_list)

    extra: Optional[Dict[str, Dict[int, np.ndarray]]] = None
    extra_q_tokens: List[np.ndarray] = []
    extra_k_tokens: List[np.ndarray] = []
    extra_v_tokens: List[np.ndarray] = []
    if source_name == "guess":
        if not isinstance(q_raw, list) or len(q_raw) != expected_guess_tokens:
            raise ValueError(
                f"{field_name}/q len={0 if not isinstance(q_raw, list) else len(q_raw)}; "
                f"expected {expected_guess_tokens}."
            )
        if not isinstance(k_raw, list) or len(k_raw) != expected_guess_tokens:
            raise ValueError(f"{field_name}/k len mismatch; expected {expected_guess_tokens}.")
        if not isinstance(v_raw, list) or len(v_raw) != expected_guess_tokens:
            raise ValueError(f"{field_name}/v len mismatch; expected {expected_guess_tokens}.")
        q_tokens, k_tokens, v_tokens = q_raw, k_raw, v_raw
    elif source_name == "probability":
        if not isinstance(q_raw, list) or not _is_expected_or_plus_two(len(q_raw), expected_probability_tokens):
            raise ValueError(
                f"{field_name}/q len={0 if not isinstance(q_raw, list) else len(q_raw)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
            )
        if not isinstance(k_raw, list) or len(k_raw) != len(q_raw):
            raise ValueError(f"{field_name}/k length {0 if not isinstance(k_raw, list) else len(k_raw)} != q.")
        if not isinstance(v_raw, list) or len(v_raw) != len(q_raw):
            raise ValueError(f"{field_name}/v length {0 if not isinstance(v_raw, list) else len(v_raw)} != q.")
        q_tokens = q_raw[:expected_probability_tokens]
        k_tokens = k_raw[:expected_probability_tokens]
        v_tokens = v_raw[:expected_probability_tokens]
        if len(q_raw) == expected_probability_tokens + 2:
            extra = {
                "q": {},
                "k": {},
                "v": {},
            }
            extra_q_tokens = q_raw[expected_probability_tokens:]
            extra_k_tokens = k_raw[expected_probability_tokens:]
            extra_v_tokens = v_raw[expected_probability_tokens:]
    else:
        q_tokens, k_tokens, v_tokens = [q_raw], [k_raw], [v_raw]
        extra_q_tokens = extra_k_tokens = extra_v_tokens = []

    q_out: Dict[int, np.ndarray] = {}
    k_out: Dict[int, np.ndarray] = {}
    v_out: Dict[int, np.ndarray] = {}
    for layer, heads in selected_heads_by_layer.items():
        head_idx = np.asarray(list(heads), dtype=np.int64)
        q_sel = []
        for tok in q_tokens:
            q_sel.append(_reshape_q_row(tok, layer=layer, n_heads=n_heads, d_head=d_head)[head_idx])
        q_out[int(layer)] = np.stack(q_sel, axis=1) if source_name in LIST_SOURCES else q_sel[0]
        kv_idx = np.asarray(list(selected_kv_heads[int(layer)]), dtype=np.int64)
        k_sel = []
        v_sel = []
        for tok_k, tok_v in zip(k_tokens, v_tokens):
            k_sel.append(
                _reshape_kv_row(tok_k, layer=layer, n_kv_heads=n_kv_heads, d_head=d_head)[kv_idx]
            )
            v_sel.append(
                _reshape_kv_row(tok_v, layer=layer, n_kv_heads=n_kv_heads, d_head=d_head)[kv_idx]
            )
        k_out[int(layer)] = np.stack(k_sel, axis=1) if source_name in LIST_SOURCES else k_sel[0]
        v_out[int(layer)] = np.stack(v_sel, axis=1) if source_name in LIST_SOURCES else v_sel[0]
        if extra is not None:
            extra_q_sel = [
                _reshape_q_row(tok, layer=layer, n_heads=n_heads, d_head=d_head)[head_idx]
                for tok in extra_q_tokens
            ]
            extra_k_sel = [
                _reshape_kv_row(tok, layer=layer, n_kv_heads=n_kv_heads, d_head=d_head)[kv_idx]
                for tok in extra_k_tokens
            ]
            extra_v_sel = [
                _reshape_kv_row(tok, layer=layer, n_kv_heads=n_kv_heads, d_head=d_head)[kv_idx]
                for tok in extra_v_tokens
            ]
            extra["q"][int(layer)] = np.stack(extra_q_sel, axis=1)
            extra["k"][int(layer)] = np.stack(extra_k_sel, axis=1)
            extra["v"][int(layer)] = np.stack(extra_v_sel, axis=1)
    return q_out, k_out, v_out, extra


def _extract_z_source_arrays(
    r0: h5py.Group,
    *,
    source_name: str,
    selected_heads_by_layer: Dict[int, Sequence[int]],
    n_heads: int,
    d_head: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    field_name = SOURCE_TO_FIELD[source_name]
    expect_list = source_name in LIST_SOURCES
    z_raw = _read_field_component(r0, field_name, "concat", expect_list=expect_list)
    z_tokens, extra_tokens = _split_source_token_list(
        z_raw,
        source_name=source_name,
        field_name=field_name,
        component="concat",
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    extra: Optional[Dict[int, np.ndarray]] = None if extra_tokens is None else {}
    z_out: Dict[int, np.ndarray] = {}
    for layer, heads in selected_heads_by_layer.items():
        head_idx = np.asarray(list(heads), dtype=np.int64)
        z_sel = [
            _reshape_q_row(tok, layer=layer, n_heads=n_heads, d_head=d_head)[head_idx]
            for tok in z_tokens
        ]
        z_out[int(layer)] = np.stack(z_sel, axis=1) if source_name in LIST_SOURCES else z_sel[0]
        if extra is not None and extra_tokens is not None:
            extra_sel = [
                _reshape_q_row(tok, layer=layer, n_heads=n_heads, d_head=d_head)[head_idx]
                for tok in extra_tokens
            ]
            extra[int(layer)] = np.stack(extra_sel, axis=1)
    return z_out, extra


def _extract_d_model_source_arrays(
    r0: h5py.Group,
    *,
    source_name: str,
    layers: Sequence[int],
    d_model: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    component: str,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    field_name = SOURCE_TO_FIELD[source_name]
    expect_list = source_name in LIST_SOURCES
    raw = _read_field_component(r0, field_name, component, expect_list=expect_list)
    tokens, extra_tokens = _split_source_token_list(
        raw,
        source_name=source_name,
        field_name=field_name,
        component=component,
        expected_guess_tokens=expected_guess_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    extra: Optional[Dict[int, np.ndarray]] = None if extra_tokens is None else {}
    out: Dict[int, np.ndarray] = {}
    for layer in layers:
        rows = [
            _d_model_row(tok, layer=int(layer), d_model=d_model, component=component)
            for tok in tokens
        ]
        out[int(layer)] = np.stack(rows, axis=0) if source_name in LIST_SOURCES else rows[0]
        if extra is not None and extra_tokens is not None:
            extra_rows = [
                _d_model_row(tok, layer=int(layer), d_model=d_model, component=component)
                for tok in extra_tokens
            ]
            extra[int(layer)] = np.stack(extra_rows, axis=0)
    return out, extra


def _extract_mlp_source_arrays(
    r0: h5py.Group,
    *,
    source_name: str,
    mlp_layers: Sequence[int],
    d_model: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    intervention_site: str = "input",
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    field_name = SOURCE_TO_FIELD[source_name]
    expect_list = source_name in LIST_SOURCES
    if intervention_site == "output":
        return _extract_d_model_source_arrays(
            r0,
            source_name=source_name,
            layers=mlp_layers,
            d_model=d_model,
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
            component="mlp",
        )

    res_raw = _read_field_component(r0, field_name, "res", expect_list=expect_list)
    attn_raw = _read_field_component(r0, field_name, "attn", expect_list=expect_list)

    extra = None
    extra_res: List[np.ndarray] = []
    extra_attn: List[np.ndarray] = []
    if source_name == "guess":
        if not isinstance(res_raw, list) or len(res_raw) != expected_guess_tokens:
            raise ValueError(
                f"{field_name}/res len={0 if not isinstance(res_raw, list) else len(res_raw)}; "
                f"expected {expected_guess_tokens}."
            )
        if not isinstance(attn_raw, list) or len(attn_raw) != expected_guess_tokens:
            raise ValueError(f"{field_name}/attn len mismatch; expected {expected_guess_tokens}.")
        res_tokens, attn_tokens = res_raw, attn_raw
    elif source_name == "probability":
        if not isinstance(res_raw, list) or not _is_expected_or_plus_two(
            len(res_raw), expected_probability_tokens
        ):
            raise ValueError(
                f"{field_name}/res len={0 if not isinstance(res_raw, list) else len(res_raw)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
            )
        if not isinstance(attn_raw, list) or len(attn_raw) != len(res_raw):
            raise ValueError(
                f"{field_name}/attn length {0 if not isinstance(attn_raw, list) else len(attn_raw)} != res."
            )
        res_tokens = res_raw[:expected_probability_tokens]
        attn_tokens = attn_raw[:expected_probability_tokens]
        if len(res_raw) == expected_probability_tokens + 2:
            extra = {}
            extra_res = res_raw[expected_probability_tokens:]
            extra_attn = attn_raw[expected_probability_tokens:]
    else:
        res_tokens, attn_tokens = [res_raw], [attn_raw]

    out = {}
    for layer in mlp_layers:
        rows = [
            _mlp_input_row(res_tok, attn_tok, layer=int(layer), d_model=d_model)
            for res_tok, attn_tok in zip(res_tokens, attn_tokens)
        ]
        out[int(layer)] = np.stack(rows, axis=0) if source_name in LIST_SOURCES else rows[0]
        if extra is not None:
            extra_rows = [
                _mlp_input_row(res_tok, attn_tok, layer=int(layer), d_model=d_model)
                for res_tok, attn_tok in zip(extra_res, extra_attn)
            ]
            extra[int(layer)] = np.stack(extra_rows, axis=0)
    return out, extra


def stream_group_means_and_eval_ids(
    path: Path | str,
    *,
    selected_heads_by_layer: Dict[int, Sequence[int]],
    mlp_layers: Sequence[int],
    attn_block_layers: Sequence[int] = (),
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    d_model: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    split_id_to_index: Dict[str, Dict[str, int]],
    split_targets: Dict[str, int],
    intervention_site: str = "input",
    log_every: int = 100,
) -> Tuple[
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, List[str]]],
    int,
]:
    """Stream H5: accumulate per-group means at the chosen intervention site.

    Means use every high/low example with usable embeddings. Eval IDs are capped
    at the usual train/validation targets, matching headwise zero ablation.
    """
    if intervention_site not in INTERVENTION_SITES:
        raise ValueError(
            f"Unknown intervention_site {intervention_site!r}; expected one of {INTERVENTION_SITES}."
        )
    site_is_output = intervention_site == "output"
    selected_ids: Dict[str, Dict[str, List[str]]] = {
        group_name: {split_name: [] for split_name in split_targets} for group_name in CONFIDENCE_GROUPS
    }
    kv_heads_map = (
        selected_kv_heads_by_layer(
            selected_heads_by_layer,
            kv_heads_per_query_group=int(n_heads) // int(n_kv_heads),
        )
        if selected_heads_by_layer
        else {}
    )
    qkv_sums: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {
        group: {comp: {int(layer): {} for layer in selected_heads_by_layer} for comp in ("q", "k", "v")}
        for group in CONFIDENCE_GROUPS
    }
    z_sums: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        group: {int(layer): {} for layer in selected_heads_by_layer} for group in CONFIDENCE_GROUPS
    }
    mlp_sums: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        group: {int(layer): {} for layer in mlp_layers} for group in CONFIDENCE_GROUPS
    }
    attn_block_sums: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        group: {int(layer): {} for layer in attn_block_layers} for group in CONFIDENCE_GROUPS
    }
    qkv_counts = {group: 0 for group in CONFIDENCE_GROUPS}
    mlp_counts = {group: 0 for group in CONFIDENCE_GROUPS}
    attn_block_counts = {group: 0 for group in CONFIDENCE_GROUPS}
    qkv_extra_counts = {group: 0 for group in CONFIDENCE_GROUPS}
    mlp_extra_counts = {group: 0 for group in CONFIDENCE_GROUPS}
    attn_block_extra_counts = {group: 0 for group in CONFIDENCE_GROUPS}
    seen_low = False
    seen_high = False
    skipped_bad = 0

    with _open_h5_readonly(path) as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        h5_example_count = int(len(examples_group))
        logging.info(
            "Streaming %d examples from %s for cross-group %s means.",
            h5_example_count,
            path,
            intervention_site,
        )
        for i, example_id in enumerate(examples_group.keys()):
            ex_id = str(example_id)
            try:
                example_node = examples_group[example_id]
                responses = example_node.get("responses")
                if responses is None or not isinstance(responses, h5py.Group):
                    raise ValueError("responses is missing.")
                length = int(responses.attrs.get("__len__", len(responses.keys())))
                if length != 1 or "0" not in responses:
                    raise ValueError(f"must have exactly one response, got {length}.")
                r0 = responses["0"]
                if not isinstance(r0, h5py.Group):
                    raise ValueError("responses/0 is not a group.")
                conf_ds = r0.get("verbalised_confidence")
                if conf_ds is None or not isinstance(conf_ds, h5py.Dataset):
                    raise ValueError("verbalised_confidence is missing.")
                conf_val = conf_ds[()]
                if isinstance(conf_val, np.ndarray):
                    conf_val = np.asarray(conf_val).reshape(-1)[0]
                conf = float(conf_val)
            except (TypeError, ValueError, OSError, KeyError) as exc:
                skipped_bad += 1
                logging.warning("Skipping example %s: %s", ex_id, exc)
                continue

            groups_for_example: List[str] = []
            if conf <= low_conf_threshold:
                seen_low = True
                groups_for_example.append("low_confidence")
            if conf >= high_conf_threshold:
                seen_high = True
                groups_for_example.append("high_confidence")

            split_name = None
            if ex_id in split_id_to_index.get("train", {}):
                split_name = "train"
            elif ex_id in split_id_to_index.get("validation", {}):
                split_name = "validation"
            if split_name is not None:
                target = int(split_targets.get(split_name, 0))
                for group_name in groups_for_example:
                    bucket = selected_ids[group_name][split_name]
                    if len(bucket) < target:
                        bucket.append(ex_id)

            if not groups_for_example:
                if log_every > 0 and (i + 1) % log_every == 0:
                    logging.info("Progress %d/%d examples (mid-confidence skipped).", i + 1, h5_example_count)
                continue

            qkv_payload = None
            z_payload = None
            mlp_payload = None
            attn_block_payload = None
            if selected_heads_by_layer and site_is_output:
                try:
                    z_by_source: Dict[str, Dict[int, np.ndarray]] = {}
                    extra_z = None
                    for source_name in ALL_SOURCES:
                        z_out, extra = _extract_z_source_arrays(
                            r0,
                            source_name=source_name,
                            selected_heads_by_layer=selected_heads_by_layer,
                            n_heads=n_heads,
                            d_head=d_head,
                            expected_guess_tokens=expected_guess_tokens,
                            expected_probability_tokens=expected_probability_tokens,
                        )
                        z_by_source[source_name] = z_out
                        if extra is not None:
                            extra_z = extra
                    z_payload = (z_by_source, extra_z)
                except ValueError as exc:
                    logging.warning("Skipping concat/hook_z means for example %s: %s", ex_id, exc)
            elif selected_heads_by_layer:
                try:
                    q_by_source: Dict[str, Dict[int, np.ndarray]] = {}
                    k_by_source: Dict[str, Dict[int, np.ndarray]] = {}
                    v_by_source: Dict[str, Dict[int, np.ndarray]] = {}
                    extra_qkv = None
                    for source_name in ALL_SOURCES:
                        q_out, k_out, v_out, extra = _extract_qkv_source_arrays(
                            r0,
                            source_name=source_name,
                            selected_heads_by_layer=selected_heads_by_layer,
                            selected_kv_heads=kv_heads_map,
                            n_heads=n_heads,
                            n_kv_heads=n_kv_heads,
                            d_head=d_head,
                            expected_guess_tokens=expected_guess_tokens,
                            expected_probability_tokens=expected_probability_tokens,
                        )
                        q_by_source[source_name] = q_out
                        k_by_source[source_name] = k_out
                        v_by_source[source_name] = v_out
                        if extra is not None:
                            extra_qkv = extra
                    qkv_payload = (q_by_source, k_by_source, v_by_source, extra_qkv)
                except ValueError as exc:
                    logging.warning("Skipping QKV means for example %s: %s", ex_id, exc)

            if mlp_layers:
                try:
                    mlp_by_source: Dict[str, Dict[int, np.ndarray]] = {}
                    extra_mlp = None
                    for source_name in ALL_SOURCES:
                        mlp_out, extra = _extract_mlp_source_arrays(
                            r0,
                            source_name=source_name,
                            mlp_layers=mlp_layers,
                            d_model=d_model,
                            expected_guess_tokens=expected_guess_tokens,
                            expected_probability_tokens=expected_probability_tokens,
                            intervention_site=intervention_site,
                        )
                        mlp_by_source[source_name] = mlp_out
                        if extra is not None:
                            extra_mlp = extra
                    mlp_payload = (mlp_by_source, extra_mlp)
                except ValueError as exc:
                    mlp_label = "MLP-output" if site_is_output else "MLP-input"
                    logging.warning("Skipping %s means for example %s: %s", mlp_label, ex_id, exc)

            if attn_block_layers:
                attn_block_component = "attn" if site_is_output else "res"
                attn_block_label = (
                    "attn-block-output (H5 attn / hook_attn_out)"
                    if site_is_output
                    else "attn-block-input (H5 res / ln1)"
                )
                try:
                    attn_block_by_source: Dict[str, Dict[int, np.ndarray]] = {}
                    extra_attn_block = None
                    for source_name in ALL_SOURCES:
                        attn_out, extra = _extract_d_model_source_arrays(
                            r0,
                            source_name=source_name,
                            layers=attn_block_layers,
                            d_model=d_model,
                            expected_guess_tokens=expected_guess_tokens,
                            expected_probability_tokens=expected_probability_tokens,
                            component=attn_block_component,
                        )
                        attn_block_by_source[source_name] = attn_out
                        if extra is not None:
                            extra_attn_block = extra
                    attn_block_payload = (attn_block_by_source, extra_attn_block)
                except ValueError as exc:
                    logging.warning(
                        "Skipping %s means for example %s: %s",
                        attn_block_label,
                        ex_id,
                        exc,
                    )

            for group_name in groups_for_example:
                if z_payload is not None:
                    z_by_source, extra_z = z_payload
                    for source_name in ALL_SOURCES:
                        for layer in selected_heads_by_layer:
                            _add_sum(z_sums[group_name][int(layer)], source_name, z_by_source[source_name][int(layer)])
                    if extra_z is not None:
                        for layer in selected_heads_by_layer:
                            _add_sum(z_sums[group_name][int(layer)], "probability_extra", extra_z[int(layer)])
                        qkv_extra_counts[group_name] += 1
                    qkv_counts[group_name] += 1
                if qkv_payload is not None:
                    q_by_source, k_by_source, v_by_source, extra_qkv = qkv_payload
                    for source_name in ALL_SOURCES:
                        for layer in selected_heads_by_layer:
                            _add_sum(qkv_sums[group_name]["q"][int(layer)], source_name, q_by_source[source_name][int(layer)])
                            _add_sum(qkv_sums[group_name]["k"][int(layer)], source_name, k_by_source[source_name][int(layer)])
                            _add_sum(qkv_sums[group_name]["v"][int(layer)], source_name, v_by_source[source_name][int(layer)])
                    if extra_qkv is not None:
                        for layer in selected_heads_by_layer:
                            _add_sum(qkv_sums[group_name]["q"][int(layer)], "probability_extra", extra_qkv["q"][int(layer)])
                            _add_sum(qkv_sums[group_name]["k"][int(layer)], "probability_extra", extra_qkv["k"][int(layer)])
                            _add_sum(qkv_sums[group_name]["v"][int(layer)], "probability_extra", extra_qkv["v"][int(layer)])
                        qkv_extra_counts[group_name] += 1
                    qkv_counts[group_name] += 1
                if mlp_payload is not None:
                    mlp_by_source, extra_mlp = mlp_payload
                    for source_name in ALL_SOURCES:
                        for layer in mlp_layers:
                            _add_sum(mlp_sums[group_name][int(layer)], source_name, mlp_by_source[source_name][int(layer)])
                    if extra_mlp is not None:
                        for layer in mlp_layers:
                            _add_sum(mlp_sums[group_name][int(layer)], "probability_extra", extra_mlp[int(layer)])
                        mlp_extra_counts[group_name] += 1
                    mlp_counts[group_name] += 1
                if attn_block_payload is not None:
                    attn_block_by_source, extra_attn_block = attn_block_payload
                    for source_name in ALL_SOURCES:
                        for layer in attn_block_layers:
                            _add_sum(
                                attn_block_sums[group_name][int(layer)],
                                source_name,
                                attn_block_by_source[source_name][int(layer)],
                            )
                    if extra_attn_block is not None:
                        for layer in attn_block_layers:
                            _add_sum(
                                attn_block_sums[group_name][int(layer)],
                                "probability_extra",
                                extra_attn_block[int(layer)],
                            )
                        attn_block_extra_counts[group_name] += 1
                    attn_block_counts[group_name] += 1

            if log_every > 0 and (i + 1) % log_every == 0:
                logging.info(
                    "Progress %d/%d examples (qkv_counts=%s mlp_counts=%s attn_block_counts=%s bad=%d).",
                    i + 1,
                    h5_example_count,
                    qkv_counts,
                    mlp_counts,
                    attn_block_counts,
                    skipped_bad,
                )

    if not seen_low:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not seen_high:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if selected_heads_by_layer:
        for group_name in CONFIDENCE_GROUPS:
            if qkv_counts[group_name] == 0:
                if site_is_output:
                    raise ValueError(f"No usable concat/hook_z embeddings for {group_name} means.")
                raise ValueError(f"No usable Q/K/V embeddings for {group_name} means.")
    if mlp_layers:
        for group_name in CONFIDENCE_GROUPS:
            if mlp_counts[group_name] == 0:
                mlp_label = "mlp" if site_is_output else "res/attn"
                mlp_site = "MLP-output" if site_is_output else "MLP-input"
                raise ValueError(
                    f"No usable {mlp_label} embeddings for {group_name} {mlp_site} means."
                )
    if attn_block_layers:
        attn_block_component = "attn" if site_is_output else "res"
        attn_block_site = "attn-block-output" if site_is_output else "attn-block-input"
        for group_name in CONFIDENCE_GROUPS:
            if attn_block_counts[group_name] == 0:
                raise ValueError(
                    f"No usable {attn_block_component} embeddings for {group_name} {attn_block_site} means."
                )
    if not _selected_groups_filled(selected_ids, split_targets):
        logging.warning(
            "Eval ID targets were not fully filled: %s",
            {g: {s: len(ids) for s, ids in splits.items()} for g, splits in selected_ids.items()},
        )

    def _mean_or_none(sum_arr: Optional[np.ndarray], count: int) -> Optional[np.ndarray]:
        if sum_arr is None or count <= 0:
            return None
        return (sum_arr / float(count)).astype(np.float32)

    packed: Dict[str, Dict[str, object]] = {}
    for group_name in CONFIDENCE_GROUPS:
        q_means: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        k_means: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        v_means: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        z_means: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        if selected_heads_by_layer:
            n = qkv_counts[group_name]
            n_extra = qkv_extra_counts[group_name]
            if site_is_output:
                for layer, heads in selected_heads_by_layer.items():
                    z_layer = z_sums[group_name][int(layer)]
                    z_means[int(layer)] = {}
                    for local_i, head_idx in enumerate(heads):
                        entry: Dict[str, np.ndarray] = {}
                        for source_name in ALL_SOURCES:
                            arr = _mean_or_none(z_layer.get(source_name), n)
                            if arr is None:
                                raise ValueError(
                                    f"Missing concat/hook_z mean for {group_name} layer {layer} {source_name}."
                                )
                            entry[source_name] = arr[local_i]
                        extra_arr = _mean_or_none(z_layer.get("probability_extra"), n_extra)
                        if extra_arr is not None:
                            entry["probability_extra"] = extra_arr[local_i]
                        z_means[int(layer)][int(head_idx)] = entry
            else:
                for layer, heads in selected_heads_by_layer.items():
                    q_layer = qkv_sums[group_name]["q"][int(layer)]
                    k_layer = qkv_sums[group_name]["k"][int(layer)]
                    v_layer = qkv_sums[group_name]["v"][int(layer)]
                    kv_heads = list(kv_heads_map[int(layer)])
                    q_means[int(layer)] = {}
                    for local_i, head_idx in enumerate(heads):
                        entry = {}
                        for source_name in ALL_SOURCES:
                            arr = _mean_or_none(q_layer.get(source_name), n)
                            if arr is None:
                                raise ValueError(f"Missing Q mean for {group_name} layer {layer} {source_name}.")
                            entry[source_name] = arr[local_i]
                        extra_arr = _mean_or_none(q_layer.get("probability_extra"), n_extra)
                        if extra_arr is not None:
                            entry["probability_extra"] = extra_arr[local_i]
                        q_means[int(layer)][int(head_idx)] = entry
                    k_means[int(layer)] = {}
                    v_means[int(layer)] = {}
                    for local_i, kv_head in enumerate(kv_heads):
                        k_entry: Dict[str, np.ndarray] = {}
                        v_entry: Dict[str, np.ndarray] = {}
                        for source_name in ALL_SOURCES:
                            k_arr = _mean_or_none(k_layer.get(source_name), n)
                            v_arr = _mean_or_none(v_layer.get(source_name), n)
                            if k_arr is None or v_arr is None:
                                raise ValueError(f"Missing K/V mean for {group_name} layer {layer} {source_name}.")
                            k_entry[source_name] = k_arr[local_i]
                            v_entry[source_name] = v_arr[local_i]
                        k_extra = _mean_or_none(k_layer.get("probability_extra"), n_extra)
                        v_extra = _mean_or_none(v_layer.get("probability_extra"), n_extra)
                        if k_extra is not None:
                            k_entry["probability_extra"] = k_extra[local_i]
                        if v_extra is not None:
                            v_entry["probability_extra"] = v_extra[local_i]
                        k_means[int(layer)][int(kv_head)] = k_entry
                        v_means[int(layer)][int(kv_head)] = v_entry
        mlp_means: Dict[int, Dict[str, np.ndarray]] = {}
        if mlp_layers:
            n = mlp_counts[group_name]
            n_extra = mlp_extra_counts[group_name]
            mlp_label = "MLP-output" if site_is_output else "MLP-input"
            for layer in mlp_layers:
                layer_sums = mlp_sums[group_name][int(layer)]
                entry = {}
                for source_name in ALL_SOURCES:
                    arr = _mean_or_none(layer_sums.get(source_name), n)
                    if arr is None:
                        raise ValueError(
                            f"Missing {mlp_label} mean for {group_name} layer {layer} {source_name}."
                        )
                    entry[source_name] = arr
                extra_arr = _mean_or_none(layer_sums.get("probability_extra"), n_extra)
                if extra_arr is not None:
                    entry["probability_extra"] = extra_arr
                mlp_means[int(layer)] = entry
        attn_block_means: Dict[int, Dict[str, np.ndarray]] = {}
        if attn_block_layers:
            n = attn_block_counts[group_name]
            n_extra = attn_block_extra_counts[group_name]
            attn_block_label = "attn-block-output" if site_is_output else "attn-block-input"
            for layer in attn_block_layers:
                layer_sums = attn_block_sums[group_name][int(layer)]
                entry = {}
                for source_name in ALL_SOURCES:
                    arr = _mean_or_none(layer_sums.get(source_name), n)
                    if arr is None:
                        raise ValueError(
                            f"Missing {attn_block_label} mean for {group_name} layer {layer} {source_name}."
                        )
                    entry[source_name] = arr
                extra_arr = _mean_or_none(layer_sums.get("probability_extra"), n_extra)
                if extra_arr is not None:
                    entry["probability_extra"] = extra_arr
                attn_block_means[int(layer)] = entry
        packed_group: Dict[str, object] = {
            "mlp": mlp_means,
            "mlp_count": int(mlp_counts[group_name]),
            "mlp_extra_count": int(mlp_extra_counts[group_name]),
            "attn_block": attn_block_means,
            "attn_block_count": int(attn_block_counts[group_name]),
            "attn_block_extra_count": int(attn_block_extra_counts[group_name]),
            "intervention_site": intervention_site,
        }
        if site_is_output:
            packed_group["z"] = z_means
            packed_group["z_count"] = int(qkv_counts[group_name])
            packed_group["z_extra_count"] = int(qkv_extra_counts[group_name])
        else:
            packed_group["q"] = q_means
            packed_group["k"] = k_means
            packed_group["v"] = v_means
            packed_group["qkv_count"] = int(qkv_counts[group_name])
            packed_group["qkv_extra_count"] = int(qkv_extra_counts[group_name])
        packed[group_name] = packed_group
    logging.info(
        "Finished streaming means. attn_counts=%s mlp_counts=%s attn_block_counts=%s "
        "extra_attn=%s extra_mlp=%s extra_attn_block=%s skipped_bad=%d site=%s",
        qkv_counts,
        mlp_counts,
        attn_block_counts,
        qkv_extra_counts,
        mlp_extra_counts,
        attn_block_extra_counts,
        skipped_bad,
        intervention_site,
    )
    return packed, selected_ids, h5_example_count


def _to_torch_means(
    packed_group: Dict[str, object],
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[str, object]:
    def _convert_head_map(
        layer_map: Dict[int, Dict[int, Dict[str, np.ndarray]]]
    ) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
        out: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        for layer, heads in layer_map.items():
            out[int(layer)] = {}
            for head_idx, sources in heads.items():
                out[int(layer)][int(head_idx)] = {
                    name: torch.tensor(np.asarray(arr), device=device, dtype=torch_dtype)
                    for name, arr in sources.items()
                }
        return out

    def _convert_mlp_map(
        layer_map: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer, sources in layer_map.items():
            out[int(layer)] = {
                name: torch.tensor(np.asarray(arr), device=device, dtype=torch_dtype)
                for name, arr in sources.items()
            }
        return out

    converted = {
        "mlp": _convert_mlp_map(packed_group["mlp"]),  # type: ignore[arg-type]
        "mlp_count": packed_group["mlp_count"],
        "mlp_extra_count": packed_group["mlp_extra_count"],
        "attn_block": _convert_mlp_map(packed_group.get("attn_block", {}) or {}),  # type: ignore[arg-type]
        "attn_block_count": packed_group.get("attn_block_count", 0),
        "attn_block_extra_count": packed_group.get("attn_block_extra_count", 0),
        "intervention_site": packed_group.get("intervention_site", "input"),
    }
    if "z" in packed_group:
        converted["z"] = _convert_head_map(packed_group["z"])  # type: ignore[arg-type]
        converted["z_count"] = packed_group["z_count"]
        converted["z_extra_count"] = packed_group["z_extra_count"]
        return converted
    converted["q"] = _convert_head_map(packed_group["q"])  # type: ignore[arg-type]
    converted["k"] = _convert_head_map(packed_group["k"])  # type: ignore[arg-type]
    converted["v"] = _convert_head_map(packed_group["v"])  # type: ignore[arg-type]
    converted["qkv_count"] = packed_group["qkv_count"]
    converted["qkv_extra_count"] = packed_group["qkv_extra_count"]
    return converted


def _positions_and_replacements_for_mode(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens: List[str],
    source_means: Dict[str, torch.Tensor],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Tuple[List[int], List[torch.Tensor]]:
    prompt_mean = source_means["prompt_mean"]
    guess = source_means["guess"]
    sem_answer_mean = source_means["sem_answer_mean"]
    probability = source_means["probability"]
    probability_value_mean = source_means["probability_value_mean"]
    probability_extra = source_means.get("probability_extra")

    if mode == "probability_tokens_mean_replace":
        positions = _absolute_prob_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=expected_probability_tokens
        )
        n = min(len(positions), int(probability.shape[0]))
        return positions[:n], [probability[i] for i in range(n)]

    if mode in PROBABILITY_ROW_INDEX_MODES:
        row_indices = probability_row_indices_for_mode(mode, model_name)
        positions = _absolute_prob_positions_at_row_indices(
            prompt_len,
            decoded_tokens,
            row_indices,
            expected_probability_tokens=expected_probability_tokens,
        )
        if not positions:
            return [], []
        return positions, [probability[i] for i in row_indices]

    if mode == "probability_last_token_mean_replace":
        positions = _absolute_prob_last_token_only_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=expected_probability_tokens
        )
        if not positions:
            return [], []
        return [positions[0]], [probability[-1]]

    if mode == "extended_probability_last_token_mean_replace":
        positions = _absolute_extended_prob_last_token_only_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=expected_probability_tokens
        )
        if not positions or probability_extra is None or int(probability_extra.shape[0]) < 2:
            return [], []
        return [positions[0]], [probability_extra[1]]

    if mode == "probability_pre_and_post_period_digit_mean_replace":
        last_pos = _absolute_prob_last_token_only_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=expected_probability_tokens
        )
        extra_pos = _absolute_extended_prob_last_token_only_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=expected_probability_tokens
        )
        positions: List[int] = []
        vectors: List[torch.Tensor] = []
        if last_pos:
            positions.append(last_pos[0])
            vectors.append(probability[-1])
        if extra_pos and probability_extra is not None and int(probability_extra.shape[0]) >= 2:
            positions.append(extra_pos[0])
            vectors.append(probability_extra[1])
        return positions, vectors

    if mode == "probability_span_except_last_token_mean_replace":
        positions = _absolute_prob_except_last_token_positions(
            prompt_len, decoded_tokens, expected_probability_tokens=expected_probability_tokens
        )
        n = min(len(positions), max(0, int(probability.shape[0]) - 1))
        return positions[:n], [probability[i] for i in range(n)]

    if mode == "guess_tokens_mean_replace":
        positions = _absolute_guess_span_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=expected_guess_tokens
        )
        n = min(len(positions), int(guess.shape[0]))
        return positions[:n], [guess[i] for i in range(n)]

    if mode == "all_pre_guess_tokens_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=expected_guess_tokens
        )
        prompt_positions = _absolute_all_pre_guess_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=expected_guess_tokens
        )
        prompt_count = max(0, len(prompt_positions) - len(guess_positions))
        positions = []
        vectors = []
        for pos in prompt_positions[:prompt_count]:
            positions.append(pos)
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
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
        if spans is None:
            return [], []
        positions = []
        vectors = []
        for pos in spans["prompt"]:
            positions.append(pos)
            vectors.append(prompt_mean)
        n_guess = min(len(spans["guess"]), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(spans["guess"][i])
            vectors.append(guess[i])
        for pos in spans["sem_answer"]:
            positions.append(pos)
            vectors.append(sem_answer_mean)
        n_prob = min(len(spans["probability"]), int(probability.shape[0]))
        for i in range(n_prob):
            positions.append(spans["probability"][i])
            vectors.append(probability[i])
        return positions, vectors

    if mode == "guess_then_guess_probability_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len, decoded_tokens, expected_guess_tokens=expected_guess_tokens
        )
        all_positions = _absolute_guess_then_guess_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
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


def build_qkv_input_mean_replace_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    selected_kv_heads: Dict[int, Sequence[int]],
    q_means: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    k_means: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    v_means: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []

    def _make_component_hook(
        *,
        local_means: Dict[int, Dict[str, torch.Tensor]],
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
                head_means = local_means.get(head_idx)
                if head_means is None:
                    raise ValueError(
                        f"Missing {component} means for layer {local_layer}, head {head_idx}."
                    )
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    source_means=head_means,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != int(activation.shape[3]):
                            raise ValueError(
                                f"{component} replacement size {vector.numel()} != d_head {activation.shape[3]} "
                                f"at layer {local_layer} head {head_idx}."
                            )
                        activation[:, abs_pos, head_idx, :] = vector.to(
                            device=activation.device, dtype=activation.dtype
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
                    local_means=q_means[int(layer)],
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
                    local_means=k_means[int(layer)],
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
                    local_means=v_means[int(layer)],
                    heads=kv_heads,
                    local_layer=int(layer),
                    component="v",
                ),
            )
        )
    return hooks


def _mlp_input_norm_module(model, layer: int):
    block = model.blocks[int(layer)]
    if not hasattr(block, "ln2"):
        raise ValueError(
            f"Model block {layer} has no ln2 pre-MLP RMSNorm. "
            "Supported models expose TransformerLens ln2 "
            "(Mistral/Qwen post_attention_layernorm; Gemma-3 pre_feedforward_layernorm)."
        )
    return block.ln2


def _attn_input_norm_module(model, layer: int):
    block = model.blocks[int(layer)]
    if not hasattr(block, "ln1"):
        raise ValueError(
            f"Model block {layer} has no ln1 pre-attention RMSNorm. "
            "Supported models expose TransformerLens ln1 "
            "(HF input_layernorm on Mistral/Qwen/Gemma-3)."
        )
    return block.ln1


@contextmanager
def mlp_input_mean_replace_pre_hooks(
    model,
    *,
    mlp_layers: Sequence[int],
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    mlp_means: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Iterator[None]:
    handles = []

    def _make_pre_hook(layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
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
                source_means=layer_means,
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
                            f"MLP-input replacement size {vector.numel()} != d_model {x.shape[-1]} "
                            f"at layer {local_layer}."
                        )
                    x[:, abs_pos, :] = vector.to(device=x.device, dtype=x.dtype)
            if len(args) == 1:
                return (x,)
            return (x, *args[1:])

        return pre_hook

    try:
        for layer in mlp_layers:
            module = _mlp_input_norm_module(model, int(layer))
            layer_means = mlp_means.get(int(layer))
            if layer_means is None:
                raise ValueError(f"Missing MLP-input means for layer {layer}.")
            handles.append(module.register_forward_pre_hook(_make_pre_hook(layer_means, int(layer))))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def attn_input_mean_replace_pre_hooks(
    model,
    *,
    attn_block_layers: Sequence[int],
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    attn_block_means: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Iterator[None]:
    handles = []

    def _make_pre_hook(layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
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
                source_means=layer_means,
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
                            f"Attn-block-input replacement size {vector.numel()} != d_model {x.shape[-1]} "
                            f"at layer {local_layer}."
                        )
                    x[:, abs_pos, :] = vector.to(device=x.device, dtype=x.dtype)
            if len(args) == 1:
                return (x,)
            return (x, *args[1:])

        return pre_hook

    try:
        for layer in attn_block_layers:
            module = _attn_input_norm_module(model, int(layer))
            layer_means = attn_block_means.get(int(layer))
            if layer_means is None:
                raise ValueError(f"Missing attn-block-input means for layer {layer}.")
            handles.append(module.register_forward_pre_hook(_make_pre_hook(layer_means, int(layer))))
        yield
    finally:
        for handle in handles:
            handle.remove()


def build_z_output_mean_replace_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    z_means: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []

    def _make_z_hook(
        *,
        local_means: Dict[int, Dict[str, torch.Tensor]],
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
                head_means = local_means.get(head_idx)
                if head_means is None:
                    raise ValueError(
                        f"Missing concat/hook_z means for layer {local_layer}, head {head_idx}."
                    )
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    source_means=head_means,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        if int(vector.numel()) != int(activation.shape[3]):
                            raise ValueError(
                                f"hook_z replacement size {vector.numel()} != d_head {activation.shape[3]} "
                                f"at layer {local_layer} head {head_idx}."
                            )
                        activation[:, abs_pos, head_idx, :] = vector.to(
                            device=activation.device, dtype=activation.dtype
                        )
            return activation

        return hook_fn

    for layer, heads in selected_heads_by_layer.items():
        z_heads = [int(h) for h in heads]
        if not z_heads:
            continue
        layer_means = z_means.get(int(layer))
        if layer_means is None:
            raise ValueError(f"Missing concat/hook_z means for layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.attn.hook_z",
                _make_z_hook(
                    local_means=layer_means,
                    heads=z_heads,
                    local_layer=int(layer),
                ),
            )
        )
    return hooks


def build_mlp_output_mean_replace_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    mlp_layers: Sequence[int],
    mlp_means: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []

    def _make_mlp_out_hook(layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            positions, vectors = _positions_and_replacements_for_mode(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens=decoded_tokens_provider(),
                source_means=layer_means,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if not positions:
                return activation
            for abs_pos, vector in zip(positions, vectors):
                if 0 <= abs_pos < activation.shape[1]:
                    if int(vector.numel()) != int(activation.shape[-1]):
                        raise ValueError(
                            f"MLP-output replacement size {vector.numel()} != d_model {activation.shape[-1]} "
                            f"at layer {local_layer}."
                        )
                    activation[:, abs_pos, :] = vector.to(
                        device=activation.device, dtype=activation.dtype
                    )
            return activation

        return hook_fn

    for layer in mlp_layers:
        layer_means = mlp_means.get(int(layer))
        if layer_means is None:
            raise ValueError(f"Missing MLP-output means for layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.hook_mlp_out",
                _make_mlp_out_hook(layer_means, int(layer)),
            )
        )
    return hooks


def build_attn_output_mean_replace_hooks(
    *,
    mode: str,
    model_name: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    attn_block_layers: Sequence[int],
    attn_block_means: Dict[int, Dict[str, torch.Tensor]],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []

    def _make_attn_out_hook(layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            positions, vectors = _positions_and_replacements_for_mode(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens=decoded_tokens_provider(),
                source_means=layer_means,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
            if not positions:
                return activation
            for abs_pos, vector in zip(positions, vectors):
                if 0 <= abs_pos < activation.shape[1]:
                    if int(vector.numel()) != int(activation.shape[-1]):
                        raise ValueError(
                            f"Attn-block-output replacement size {vector.numel()} != d_model "
                            f"{activation.shape[-1]} at layer {local_layer}."
                        )
                    activation[:, abs_pos, :] = vector.to(
                        device=activation.device, dtype=activation.dtype
                    )
            return activation

        return hook_fn

    for layer in attn_block_layers:
        layer_means = attn_block_means.get(int(layer))
        if layer_means is None:
            raise ValueError(f"Missing attn-block-output means for layer {layer}.")
        hooks.append(
            (
                f"blocks.{int(layer)}.hook_attn_out",
                _make_attn_out_hook(layer_means, int(layer)),
            )
        )
    return hooks


def greedy_generate_componentwise_mean_ablated(
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
    source_means: Dict[str, object],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
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
                build_z_output_mean_replace_hooks(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    selected_heads_by_layer=selected_heads_by_layer,
                    z_means=source_means["z"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
            )
        if attn_block_layers:
            hooks.extend(
                build_attn_output_mean_replace_hooks(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    attn_block_layers=attn_block_layers,
                    attn_block_means=source_means["attn_block"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
            )
        if mlp_layers:
            hooks.extend(
                build_mlp_output_mean_replace_hooks(
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    mlp_layers=mlp_layers,
                    mlp_means=source_means["mlp"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
            )
        if not hooks:
            raise ValueError(
                "No attention heads, attention blocks, or MLP layers selected for "
                "componentwise mean ablation."
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
            build_qkv_input_mean_replace_hooks(
                mode=mode,
                model_name=model_name,
                prompt_len=prompt_len,
                decoded_tokens_provider=_decoded_tokens_provider,
                selected_heads_by_layer=selected_heads_by_layer,
                selected_kv_heads=selected_kv_heads,
                q_means=source_means["q"],  # type: ignore[arg-type]
                k_means=source_means["k"],  # type: ignore[arg-type]
                v_means=source_means["v"],  # type: ignore[arg-type]
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
        )
    if not hooks and not mlp_layers and not attn_block_layers:
        raise ValueError(
            "No attention heads, attention blocks, or MLP layers selected for "
            "componentwise mean ablation."
        )
    with ExitStack() as stack:
        if attn_block_layers:
            stack.enter_context(
                attn_input_mean_replace_pre_hooks(
                    model,
                    attn_block_layers=attn_block_layers,
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    attn_block_means=source_means["attn_block"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                )
            )
        if mlp_layers:
            stack.enter_context(
                mlp_input_mean_replace_pre_hooks(
                    model,
                    mlp_layers=mlp_layers,
                    mode=mode,
                    model_name=model_name,
                    prompt_len=prompt_len,
                    decoded_tokens_provider=_decoded_tokens_provider,
                    mlp_means=source_means["mlp"],  # type: ignore[arg-type]
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
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
        "Componentwise Mean Ablation Configuration",
        "========================================",
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
        "mean_protocol=cross_group",
        "mean_source_for_low_confidence=high_confidence",
        "mean_source_for_high_confidence=low_confidence",
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
        "Ablated generations use the opposite group's component mean at intervention_site.",
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
    attn_block_layers: Sequence[int],
    num_selected_layer_head_pairs: int,
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
            "qkv_mean_count": int(qkv_mean_counts.get(group_name, 0)),
            "mlp_mean_count": int(mlp_mean_counts.get(group_name, 0)),
            "attn_block_mean_count": int(attn_block_mean_counts.get(group_name, 0)),
            "mean_source_group": OPPOSITE_GROUP[group_name],
            "modes": modes_out,
            "by_none_mode_confidence": none_mode_buckets,
        }
    return {
        "run_root": run_root,
        "dataset": args.dataset,
        "input_h5": args.input_h5,
        "h5_example_count": h5_example_count,
        "intervention_site": args.intervention_site,
        "mean_protocol": "cross_group",
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
            "Simultaneous componentwise mean ablation: patch selected head Q/K/V or hook_z, "
            "whole attention blocks (ln1 / hook_attn_out), and/or MLP RMSNorm inputs or "
            "hook_mlp_out with the opposite confidence group's mean."
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
            "Whole-block a<layer> patches ln1 with H5 res (input) or hook_attn_out with H5 attn "
            "(output; after ln1_post on sandwich Gemma). "
            "All listed units are mean-ablated simultaneously at --intervention_site. "
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
            "Patch component inputs (Q/K/V, attn ln1 from H5 res, and MLP RMSNorm in) or outputs "
            "(hook_z, hook_attn_out from H5 attn, and hook_mlp_out)."
        ),
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
    validate_last_a_panl_and_pc_mode(args.model_name, args.ablation_mode)

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
            "componentwise mean ablation."
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
        "Ablating units=%s (%d layer-head pairs, %d attn blocks, %d mlp layers) at component %ss.",
        resolved_units,
        num_selected_layer_head_pairs,
        len(attn_block_layers),
        len(mlp_layers),
        args.intervention_site,
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
    group_torch_means = {
        group_name: _to_torch_means(packed_means[group_name], device=device, torch_dtype=torch_dtype)
        for group_name in CONFIDENCE_GROUPS
    }
    del packed_means
    gc.collect()

    results_mini: Dict[str, Dict[str, dict]] = {
        group_name: {"train": {}, "validation": {}} for group_name in CONFIDENCE_GROUPS
    }
    group_metrics = {group_name: _empty_group_metrics(args.ablation_mode) for group_name in CONFIDENCE_GROUPS}

    for group_name in CONFIDENCE_GROUPS:
        metrics = group_metrics[group_name]
        source_group = OPPOSITE_GROUP[group_name]
        source_means = group_torch_means[source_group]
        logging.info(
            "Ablating group=%s (%d selected H5 ids) with %s %s means; "
            "%d heads, %d attn blocks, and %d MLP layers patched simultaneously.",
            group_name,
            sum(len(ids) for ids in selected_ids_by_group[group_name].values()),
            source_group,
            args.intervention_site,
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

                    response, _ = greedy_generate_componentwise_mean_ablated(
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
                        source_means=source_means,
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
                            "mean_source_group": source_group,
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
        model_n_kv_heads=model_n_kv_heads,
        model_d_head=model_d_head,
        ablate_layers=run_layers,
        selected_heads_by_layer=selected_heads_by_layer,
        mlp_layers=mlp_layers,
        attn_block_layers=attn_block_layers,
        num_selected_layer_head_pairs=num_selected_layer_head_pairs,
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
