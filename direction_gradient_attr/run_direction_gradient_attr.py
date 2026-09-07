#!/usr/bin/env python3
"""Attribution patching of mass-mean (high−low) directions onto logit differences.

Approximates the effect of adding the cached mass-mean direction at each site
on a scalar logit difference ΔL, using Neel Nanda's attribution patching:

    ⟨∇_x ΔL, d_mass_mean⟩

evaluated at the predicting token (last position of the reconstructed prefix).

The input-site pass uses ``hook_q_input`` / ``hook_k_input`` / ``hook_v_input``
(copied resid-pre; direction from H5 ``res[L]``) and ``hook_mlp_in`` (cloned
resid-mid; direction from ``res[L] + attn[L]``). Fine head scores sum the three
input dots (GQA shares K/V across the query group).

With ``--attribute_component_outputs`` (default), a second independent
``run_with_hooks`` pass scores component *outputs* from the same prefixes and
ΔL: the attention residual addend at ``hook_attn_out`` (after ``ln1_post`` /
post-attn RMSNorm on sandwich Gemma-3; ``o_proj`` out on Mistral/Qwen) and
``hook_mlp_out`` (H5 ``mlp[L]``; after ``ln2_post`` / post-FFN norm on sandwich
Gemma-3). Fine attention reconstructs per-head addend pieces like
``gradient_based_attr`` (``z`` through ``W_O``, then a shared post-attn RMS
scale on Gemma) and scores each head with its own mass-mean direction
``d_h`` (H5 concat through ``W_O``, sandwich ``ln1_post`` applied per
example before averaging). That rebuild detaches ``hook_z``, so it cannot
share a graph with ``hook_q/k/v_input``. Coarse ``L*_attn`` is
``⟨∇_addend, d_attn⟩``; fine head ``h`` is ``⟨∇_post_h, d_h⟩``. Those
results nest under ``component_output/``.

Numeric runs use pre-period and post-period digits. Linguistic Confidence
(``--linguistic_confidence_prompt``, Mistral only) uses the variable-length
Confidence: span.

With ``--residual_stream``, a separate layer-level pass attributes the residual
stream instead of Q/K/V inputs and MLP: display 0 is ``blocks.0.hook_resid_pre``
(embedding; H5 ``res[0]``) and display ``k>=1`` is ``blocks.{k-1}.hook_resid_post``
(H5 ``res[k]``). There is no fine split and no ``component_output/`` tree.
Numeric residual runs write ``digit_vs_rest`` and ``high_vs_low`` with a
two-column pre/post-period heatmap; linguistic residual heatmaps are one column.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ans_gen.generate_answers_h5 import (
    CONFIDENCE_PROMPT,
    CONFIDENCE_PROMPT_LINGUISTIC,
)
from componentwise_mean_ablation.run_componentwise_mean_ablation import (
    _h5_node_type,
    _mlp_input_row,
    _read_layer_hidden_dataset,
)
from gradient_based_attr.run_gradient_based_attr import (
    DTYPE_MAP,
    GradExample,
    build_linguistic_contrast_specs,
    build_numeric_contrasts,
    contrast_span_index,
    fill_cohorts_from_generations,
    filter_examples_for_contrast,
    load_grad_example_cohorts,
    logit_diff_from_logits,
    prefix_generated_token_ids,
)
from headwise_mean_ablation.run_headwise_mean_ablation import resolve_n_key_value_heads
from layerwise_mean_ablation.run_mean_ablation import (
    greedy_generate,
    hook_name_for_display_layer,
    load_hooked_transformer,
)
from process_generations.process_generations_more_embs_from_h5 import (
    configure_prefix_tokens_for_model,
    parse_guess_and_probability_indices,
)
from direct_logit_attribution.run_direct_logit_attribution import (
    ANNOTATION_ROUNDING_DP,
    ContrastSpec,
    ExperimentResult,
    SUPPORTED_MODEL_NAMES,
    _h5_response0_group,
    _is_piece_token_decode,
    _is_verbalised_confidence_one,
    _open_h5_readonly,
    _read_decoded_tokens,
    _read_verbalised_confidence_scalar,
    aggregate_component_stats,
    attach_output_log,
    component_labels_coarse,
    component_labels_fine,
    ComponentStats,
    high_low_digit_sets,
    high_minus_low_rows,
    linguistic_first_tokens_for_model,
    invert_square_o_proj,
    parse_individual_ids,
    rank_by_mean,
    rank_by_value,
    recover_concat_from_o,
    resolve_all_digit_ids,
    resolve_linguistic_first_token_ids,
    resolve_single_piece_token_id,
    save_attributions_h5,
    validate_confidence_thresholds,
    write_bar_chart,
    write_distribution_plot,
    write_diverging_heatmap,
    write_experiment_outputs,
    write_high_minus_low_from_results,
    write_ranked_example_txt,
    write_ranked_high_minus_low_txt,
    write_ranked_mean_txt,
)

L_VS_UN_SPAN_INDEX = 6
COMPONENT_OUTPUT_SUBDIR = "component_output"
ATTN_OUT_RESUM_ATOL = 5e-2
RESIDUAL_STREAM_GRANULARITY = "residual"
NUMERIC_RESIDUAL_PERIODS = ("pre_period", "post_period")


# ---------------------------------------------------------------------------
# Mass-mean directions from cached H5 resid-pre (res[L]) and MLP-input rows
# ---------------------------------------------------------------------------


@dataclass
class SiteDirections:
    resid_pre: np.ndarray  # [n_layers, d_model] from H5 res[L]
    mlp: np.ndarray  # [n_layers, d_model] from res[L] + attn[L]
    n_high: int
    n_low: int
    attn_out: np.ndarray  # [n_layers, d_model] from H5 attn[L]
    mlp_out: np.ndarray  # [n_layers, d_model] from H5 mlp[L]
    attn_heads: np.ndarray | None = None  # [n_layers, n_heads, d_model] post-norm writes
    resid_stream: np.ndarray | None = None  # [n_layers+1, d_model] display layers


@dataclass
class AttnHeadWriteSpec:
    """Per-layer W_O / optional sandwich RMSNorm for H5 concat → residual writes."""

    n_heads: int
    d_head: int
    d_model: int
    w_o: np.ndarray  # [n_heads, d_head, d_model]
    b_o: np.ndarray | None
    rms_weight: np.ndarray | None  # [d_model]; None if no ln1_post
    rms_eps: float
    rms_use_one_plus: bool


def _probability_component_group(r0: h5py.Group, component: str) -> h5py.Group | None:
    field = r0.get("embeddings_probability")
    if field is None or not isinstance(field, h5py.Group):
        return None
    comp = field.get(component)
    if comp is None or not isinstance(comp, h5py.Group):
        return None
    if _h5_node_type(comp) == "none":
        return None
    return comp


def _probability_list_len(r0: h5py.Group, component: str = "res") -> int | None:
    comp = _probability_component_group(r0, component)
    if comp is None:
        return None
    return int(comp.attrs.get("__len__", len(comp.keys())))


def _read_probability_token_hidden(
    r0: h5py.Group, component: str, span_index: int
) -> np.ndarray:
    comp = _probability_component_group(r0, component)
    if comp is None:
        raise ValueError(f"embeddings_probability/{component} is missing.")
    key = str(int(span_index))
    if key not in comp:
        raise ValueError(f"embeddings_probability/{component}/{key} is missing.")
    ds = comp[key]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"embeddings_probability/{component}/{key} is not a dataset.")
    return _read_layer_hidden_dataset(ds)


def _resid_pre_all_layers(
    res_hidden: np.ndarray, *, n_layers: int, d_model: int
) -> np.ndarray:
    if res_hidden.ndim != 2:
        raise ValueError(f"Expected res [layers+1, d_model], got {res_hidden.shape}.")
    expected_rows = int(n_layers) + 1
    if int(res_hidden.shape[0]) != expected_rows:
        raise ValueError(
            f"res has {res_hidden.shape[0]} layer rows; expected n_layers+1={expected_rows} "
            "(res[0] is resid-pre of block 0)."
        )
    if int(res_hidden.shape[1]) != int(d_model):
        raise ValueError(
            f"res hidden dim {res_hidden.shape[1]} != d_model={d_model}."
        )
    return np.asarray(res_hidden[: int(n_layers)], dtype=np.float32)


def _resid_stream_all_layers(
    res_hidden: np.ndarray, *, n_layers: int, d_model: int
) -> np.ndarray:
    """Full display-layer residual stack: res[0]=embedding, res[k]=resid-post of block k-1."""
    if res_hidden.ndim != 2:
        raise ValueError(f"Expected res [layers+1, d_model], got {res_hidden.shape}.")
    expected_rows = int(n_layers) + 1
    if int(res_hidden.shape[0]) != expected_rows:
        raise ValueError(
            f"res has {res_hidden.shape[0]} layer rows; expected n_layers+1={expected_rows} "
            "(res[0] is embedding resid-pre; res[k] is resid-post of block k-1)."
        )
    if int(res_hidden.shape[1]) != int(d_model):
        raise ValueError(
            f"res hidden dim {res_hidden.shape[1]} != d_model={d_model}."
        )
    return np.asarray(res_hidden[:expected_rows], dtype=np.float32)


def _mlp_input_all_layers(
    res_hidden: np.ndarray, attn_hidden: np.ndarray, *, n_layers: int, d_model: int
) -> np.ndarray:
    rows = [
        _mlp_input_row(res_hidden, attn_hidden, layer=layer, d_model=d_model)
        for layer in range(int(n_layers))
    ]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _subblock_out_all_layers(
    hidden: np.ndarray, *, n_layers: int, d_model: int, name: str
) -> np.ndarray:
    if hidden.ndim != 2:
        raise ValueError(f"Expected {name} [layers, d_model], got {hidden.shape}.")
    if int(hidden.shape[0]) < int(n_layers):
        raise ValueError(
            f"{name} has {hidden.shape[0]} layer rows; need at least n_layers={n_layers}."
        )
    if int(hidden.shape[1]) != int(d_model):
        raise ValueError(
            f"{name} hidden dim {hidden.shape[1]} != d_model={d_model}."
        )
    return np.asarray(hidden[: int(n_layers)], dtype=np.float32)


def _concat_all_layers(
    hidden: np.ndarray, *, n_layers: int, concat_width: int, name: str
) -> np.ndarray:
    if hidden.ndim != 2:
        raise ValueError(f"Expected {name} [layers, concat_width], got {hidden.shape}.")
    if int(hidden.shape[0]) < int(n_layers):
        raise ValueError(
            f"{name} has {hidden.shape[0]} layer rows; need at least n_layers={n_layers}."
        )
    if int(hidden.shape[1]) != int(concat_width):
        raise ValueError(
            f"{name} width {hidden.shape[1]} != n_heads*d_head={concat_width}."
        )
    return np.asarray(hidden[: int(n_layers)], dtype=np.float32)


def _try_read_probability_token_hidden(
    r0: h5py.Group, component: str, span_index: int
) -> np.ndarray | None:
    if _probability_component_group(r0, component) is None:
        return None
    try:
        return _read_probability_token_hidden(r0, component, span_index)
    except (TypeError, ValueError, OSError, KeyError):
        return None


def _o_proj_weight_from_w_o(w_o: np.ndarray) -> np.ndarray:
    """TL ``W_O`` ``[n_heads, d_head, d_model]`` → HF ``o_proj.weight`` ``[d_model, n_heads*d_head]``."""
    n_heads, d_head, d_model = w_o.shape
    return np.transpose(w_o.reshape(n_heads * d_head, d_model), (1, 0))


def _maybe_invert_o_proj_weights(
    specs: Sequence[AttnHeadWriteSpec],
) -> list[np.ndarray]:
    invs: list[np.ndarray] = []
    for layer, spec in enumerate(specs):
        w = _o_proj_weight_from_w_o(spec.w_o)
        expected_in = int(spec.n_heads) * int(spec.d_head)
        if int(spec.d_model) != expected_in:
            raise ValueError(
                "concat embeddings are missing; recovering concat from o requires "
                "square W_O (d_model == n_heads*d_head), got "
                f"d_model={spec.d_model}, n_heads*d_head={expected_in} at layer {layer}."
            )
        invs.append(invert_square_o_proj(w, layer_idx=layer))
    return invs


def numpy_post_heads_from_concat(
    concat_layer: np.ndarray,
    spec: AttnHeadWriteSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-norm per-head residual writes and their reconstructed addend.

    Matches the live ``hook_z`` rebuild: ``z_h @ W_O[h]``, shared ``ln1_post``
    scale from ``sum_h write_h + b_O`` on sandwich Gemma, bias added only to the
    total.
    """
    expected = int(spec.n_heads) * int(spec.d_head)
    vec = np.asarray(concat_layer, dtype=np.float32)
    if vec.ndim != 1 or int(vec.shape[0]) != expected:
        raise ValueError(
            f"concat width {None if vec.ndim != 1 else int(vec.shape[0])} "
            f"!= n_heads*d_head={expected}."
        )
    z = vec.reshape(int(spec.n_heads), int(spec.d_head))
    writes = np.einsum("hd,hde->he", z, spec.w_o.astype(np.float32, copy=False))
    total_pre = writes.sum(axis=0)
    if spec.b_o is not None:
        total_pre = total_pre + spec.b_o.astype(np.float32, copy=False)
    if spec.rms_weight is None:
        return writes.astype(np.float32, copy=False), total_pre.astype(
            np.float32, copy=False
        )
    total_f = total_pre.astype(np.float64, copy=False)
    inv = 1.0 / np.sqrt(np.mean(np.square(total_f)) + float(spec.rms_eps))
    w = spec.rms_weight.astype(np.float64, copy=False)
    gamma = (1.0 + w) if spec.rms_use_one_plus else w
    scale = (inv * gamma).astype(np.float32)
    post_heads = writes * scale
    total_post = post_heads.sum(axis=0)
    if spec.b_o is not None:
        total_post = total_post + spec.b_o.astype(np.float32, copy=False) * scale
    return post_heads.astype(np.float32, copy=False), total_post.astype(
        np.float32, copy=False
    )


def _read_concat_or_o_hidden(
    r0: h5py.Group,
    span_index: int,
    specs: Sequence[AttnHeadWriteSpec],
    *,
    o_proj_invs: list[np.ndarray] | None,
    used_o_fallback: list[bool],
) -> np.ndarray | None:
    n_layers = len(specs)
    concat_width = int(specs[0].n_heads) * int(specs[0].d_head)
    raw = _try_read_probability_token_hidden(r0, "concat", span_index)
    if raw is not None:
        return _concat_all_layers(
            raw, n_layers=n_layers, concat_width=concat_width, name="concat"
        )
    o_raw = _try_read_probability_token_hidden(r0, "o", span_index)
    if o_raw is None:
        return None
    if o_proj_invs is None:
        return None
    o_h = _subblock_out_all_layers(
        o_raw, n_layers=n_layers, d_model=int(specs[0].d_model), name="o"
    )
    if not used_o_fallback[0]:
        used_o_fallback[0] = True
        logging.warning(
            "concat embeddings missing; recovering concat from o via inv(W_O)."
        )
    rows = [
        recover_concat_from_o(o_h[layer], o_proj_invs[layer])
        for layer in range(n_layers)
    ]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _post_heads_all_layers(
    concat_hidden: np.ndarray,
    attn_out: np.ndarray,
    specs: Sequence[AttnHeadWriteSpec],
    *,
    resum_warned: set[int],
    resum_fail_counts: dict[int, int],
    context: str,
) -> np.ndarray:
    n_layers = len(specs)
    n_heads = int(specs[0].n_heads)
    d_model = int(specs[0].d_model)
    out = np.zeros((n_layers, n_heads, d_model), dtype=np.float32)
    for layer, spec in enumerate(specs):
        heads, total = numpy_post_heads_from_concat(concat_hidden[layer], spec)
        out[layer] = heads
        delta = float(np.max(np.abs(total - attn_out[layer].astype(np.float32))))
        if delta > ATTN_OUT_RESUM_ATOL:
            resum_fail_counts[layer] = int(resum_fail_counts.get(layer, 0)) + 1
            if layer not in resum_warned:
                resum_warned.add(layer)
                logging.warning(
                    "Attn-addend H5 rebuild soft-fail %s layer=%d: "
                    "max |sum_h post_h (+ scaled b_O) - attn|=%.6g > atol=%g.",
                    context,
                    layer,
                    delta,
                    ATTN_OUT_RESUM_ATOL,
                )
    return out


def _token_is_l_or_un(decoded_tokens: Sequence[str], span_index: int) -> bool:
    parsed = parse_guess_and_probability_indices(
        list(decoded_tokens), linguistic_confidence_prompt=True
    )
    if parsed is None:
        return False
    _last_guess, first_prob, _end_prob = parsed
    abs_idx = int(first_prob) + int(span_index)
    if abs_idx < 0 or abs_idx >= len(decoded_tokens):
        return False
    token = decoded_tokens[abs_idx]
    return _is_piece_token_decode(token, "L") or _is_piece_token_decode(token, "Un")


def _direction_span_index(
    position: str,
    *,
    linguistic: bool,
    expected_probability_tokens: int,
    expected_confidence_tokens: int,
) -> int:
    if linguistic:
        if position == "first_value":
            return int(expected_confidence_tokens) - 1
        if position == "mistral_l_vs_un":
            return int(L_VS_UN_SPAN_INDEX)
        raise ValueError(f"Unknown linguistic position {position!r}.")
    expected_span = int(expected_probability_tokens) + 2
    if position == "pre_period":
        return expected_span - 3
    if position == "post_period":
        return expected_span - 1
    raise ValueError(f"Unknown numeric position {position!r}.")


def _read_site_arrays_at_index(
    r0: h5py.Group,
    span_index: int,
    *,
    n_layers: int,
    d_model: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_tok = _probability_list_len(r0, "res")
    if n_tok is None or int(span_index) >= int(n_tok):
        raise ValueError(
            f"probability span len={n_tok} cannot index span_index={span_index}."
        )
    res_h = _read_probability_token_hidden(r0, "res", span_index)
    attn_h = _read_probability_token_hidden(r0, "attn", span_index)
    mlp_h = _read_probability_token_hidden(r0, "mlp", span_index)
    resid_pre = _resid_pre_all_layers(res_h, n_layers=n_layers, d_model=d_model)
    mlp_in = _mlp_input_all_layers(res_h, attn_h, n_layers=n_layers, d_model=d_model)
    attn_out = _subblock_out_all_layers(
        attn_h, n_layers=n_layers, d_model=d_model, name="attn"
    )
    mlp_out = _subblock_out_all_layers(
        mlp_h, n_layers=n_layers, d_model=d_model, name="mlp"
    )
    return resid_pre, mlp_in, attn_out, mlp_out


def _read_resid_stream_at_index(
    r0: h5py.Group,
    span_index: int,
    *,
    n_layers: int,
    d_model: int,
) -> np.ndarray:
    n_tok = _probability_list_len(r0, "res")
    if n_tok is None or int(span_index) >= int(n_tok):
        raise ValueError(
            f"probability span len={n_tok} cannot index span_index={span_index}."
        )
    res_h = _read_probability_token_hidden(r0, "res", span_index)
    return _resid_stream_all_layers(res_h, n_layers=n_layers, d_model=d_model)


def residual_stream_labels(n_layers: int) -> list[str]:
    return [f"L{d}" for d in range(int(n_layers) + 1)]


def build_numeric_residual_contrasts(
    digit_ids: dict[str, tuple[int, str]],
    high_digits: set[str],
    low_digits: set[str],
) -> list[ContrastSpec]:
    """digit_vs_rest and high_vs_low at both periods (pre high_vs_low is 1 vs 0)."""
    dummy = np.zeros(1, dtype=np.float32)
    id0 = digit_ids["0"][0]
    id1 = digit_ids["1"][0]
    all_digit_ids = tuple(digit_ids[d][0] for d in "0123456789")
    high_ids = tuple(digit_ids[d][0] for d in sorted(high_digits))
    low_ids = tuple(digit_ids[d][0] for d in sorted(low_digits))
    return [
        ContrastSpec(
            position="pre_period",
            name="digit_vs_rest",
            u=dummy,
            pos_token_ids=(id0, id1),
            neg_token_ids=None,
        ),
        ContrastSpec(
            position="pre_period",
            name="high_vs_low",
            u=dummy,
            pos_token_ids=(id1,),
            neg_token_ids=(id0,),
        ),
        ContrastSpec(
            position="post_period",
            name="digit_vs_rest",
            u=dummy,
            pos_token_ids=all_digit_ids,
            neg_token_ids=None,
        ),
        ContrastSpec(
            position="post_period",
            name="high_vs_low",
            u=dummy,
            pos_token_ids=high_ids,
            neg_token_ids=low_ids,
        ),
    ]


def compute_mass_mean_directions(
    input_h5: str,
    *,
    positions: Sequence[str],
    span_index_by_position: dict[str, int],
    n_layers: int,
    d_model: int,
    high_conf_threshold: float,
    low_conf_threshold: float,
    exclude_conf_one_positions: Sequence[str],
    require_l_or_un_positions: Sequence[str],
    head_write_specs: Sequence[AttnHeadWriteSpec] | None = None,
    residual_stream: bool = False,
) -> dict[str, SiteDirections]:
    """Stream all high/low H5 examples; d = high_mean - low_mean at each site."""
    exclude_one = set(exclude_conf_one_positions)
    require_lu = set(require_l_or_un_positions)
    need_heads = head_write_specs is not None and not residual_stream
    if need_heads:
        assert head_write_specs is not None
        if len(head_write_specs) != int(n_layers):
            raise ValueError(
                f"head_write_specs has {len(head_write_specs)} layers; "
                f"expected n_layers={n_layers}."
            )
        for spec in head_write_specs:
            if int(spec.d_model) != int(d_model):
                raise ValueError(
                    f"Head-write spec d_model={spec.d_model} != d_model={d_model}."
                )
    o_proj_invs: list[np.ndarray] | None = None
    used_o_fallback = [False]
    resum_warned: set[int] = set()
    resum_fail_counts: dict[int, int] = {}
    skipped_no_head_src = 0
    sums: dict[str, dict[str, dict[str, np.ndarray]]] = {
        pos: {
            "high": {},
            "low": {},
        }
        for pos in positions
    }
    counts: dict[str, dict[str, int]] = {
        pos: {"high": 0, "low": 0} for pos in positions
    }
    skipped_bad = 0
    with _open_h5_readonly(input_h5) as f:
        examples = f.get("examples")
        if examples is None or not isinstance(examples, h5py.Group):
            raise ValueError(f"{input_h5} has no 'examples' group.")
        example_ids = list(examples.keys())
        for ex_id in tqdm(example_ids, desc="Streaming H5 mass-mean directions"):
            ex_group = examples[ex_id]
            if not isinstance(ex_group, h5py.Group):
                continue
            r0 = _h5_response0_group(ex_group)
            if r0 is None:
                continue
            conf = _read_verbalised_confidence_scalar(r0)
            if conf is None:
                continue
            conf_f = float(conf)
            groups: list[str] = []
            if conf_f >= high_conf_threshold:
                groups.append("high")
            if conf_f <= low_conf_threshold:
                groups.append("low")
            if not groups:
                continue
            is_one = _is_verbalised_confidence_one(conf_f)
            decoded_tokens: list[str] | None = None
            if require_lu:
                decoded_tokens = _read_decoded_tokens(r0)
            for pos in positions:
                if pos in exclude_one and is_one:
                    continue
                if pos in require_lu:
                    if not decoded_tokens:
                        continue
                    if not _token_is_l_or_un(decoded_tokens, span_index_by_position[pos]):
                        continue
                span_index = int(span_index_by_position[pos])
                try:
                    resid_stream_ex: np.ndarray | None = None
                    resid_pre: np.ndarray | None = None
                    mlp_in: np.ndarray | None = None
                    attn_out: np.ndarray | None = None
                    mlp_out: np.ndarray | None = None
                    attn_heads_ex: np.ndarray | None = None
                    if residual_stream:
                        resid_stream_ex = _read_resid_stream_at_index(
                            r0,
                            span_index,
                            n_layers=n_layers,
                            d_model=d_model,
                        )
                    else:
                        resid_pre, mlp_in, attn_out, mlp_out = _read_site_arrays_at_index(
                            r0,
                            span_index,
                            n_layers=n_layers,
                            d_model=d_model,
                        )
                        if head_write_specs is not None:
                            concat_h = _read_concat_or_o_hidden(
                                r0,
                                span_index,
                                head_write_specs,
                                o_proj_invs=o_proj_invs,
                                used_o_fallback=used_o_fallback,
                            )
                            if concat_h is None:
                                if (
                                    o_proj_invs is None
                                    and _try_read_probability_token_hidden(
                                        r0, "o", span_index
                                    )
                                    is not None
                                ):
                                    o_proj_invs = _maybe_invert_o_proj_weights(
                                        head_write_specs
                                    )
                                    concat_h = _read_concat_or_o_hidden(
                                        r0,
                                        span_index,
                                        head_write_specs,
                                        o_proj_invs=o_proj_invs,
                                        used_o_fallback=used_o_fallback,
                                    )
                            if concat_h is None:
                                skipped_no_head_src += 1
                                continue
                            attn_heads_ex = _post_heads_all_layers(
                                concat_h,
                                attn_out,
                                head_write_specs,
                                resum_warned=resum_warned,
                                resum_fail_counts=resum_fail_counts,
                                context=f"ex={ex_id} pos={pos}",
                            )
                except (TypeError, ValueError, OSError, KeyError) as exc:
                    skipped_bad += 1
                    logging.debug("Skip direction example %s pos=%s: %s", ex_id, pos, exc)
                    continue
                for group in groups:
                    bucket = sums[pos][group]
                    if residual_stream:
                        if resid_stream_ex is None:
                            raise RuntimeError("residual_stream example row is missing.")
                        if not bucket:
                            bucket["resid_stream"] = np.array(
                                resid_stream_ex, dtype=np.float64, copy=True
                            )
                        else:
                            bucket["resid_stream"] += resid_stream_ex
                    else:
                        if (
                            resid_pre is None
                            or mlp_in is None
                            or attn_out is None
                            or mlp_out is None
                        ):
                            raise RuntimeError("Site arrays missing for mass-mean bucket.")
                        if not bucket:
                            bucket["resid_pre"] = np.array(
                                resid_pre, dtype=np.float64, copy=True
                            )
                            bucket["mlp"] = np.array(mlp_in, dtype=np.float64, copy=True)
                            bucket["attn_out"] = np.array(
                                attn_out, dtype=np.float64, copy=True
                            )
                            bucket["mlp_out"] = np.array(
                                mlp_out, dtype=np.float64, copy=True
                            )
                            if attn_heads_ex is not None:
                                bucket["attn_heads"] = np.array(
                                    attn_heads_ex, dtype=np.float64, copy=True
                                )
                        else:
                            bucket["resid_pre"] += resid_pre
                            bucket["mlp"] += mlp_in
                            bucket["attn_out"] += attn_out
                            bucket["mlp_out"] += mlp_out
                            if attn_heads_ex is not None:
                                bucket["attn_heads"] += attn_heads_ex
                    counts[pos][group] += 1

    out: dict[str, SiteDirections] = {}
    for pos in positions:
        n_high = int(counts[pos]["high"])
        n_low = int(counts[pos]["low"])
        if n_high < 1:
            extra = (
                f" ({skipped_no_head_src} examples lacked concat and o)"
                if skipped_no_head_src
                else ""
            )
            raise ValueError(
                f"No high-confidence examples for mass-mean at {pos!r}.{extra}"
            )
        if n_low < 1:
            extra = (
                f" ({skipped_no_head_src} examples lacked concat and o)"
                if skipped_no_head_src
                else ""
            )
            raise ValueError(
                f"No low-confidence examples for mass-mean at {pos!r}.{extra}"
            )
        high = sums[pos]["high"]
        low = sums[pos]["low"]
        resid_stream = None
        zeros = np.zeros((int(n_layers), int(d_model)), dtype=np.float32)
        if residual_stream:
            resid_stream = (
                (high["resid_stream"] / n_high) - (low["resid_stream"] / n_low)
            ).astype(np.float32)
            resid_pre = resid_stream[: int(n_layers)]
            mlp = zeros
            attn_out = zeros
            mlp_out = zeros
            attn_heads = None
        else:
            resid_pre = ((high["resid_pre"] / n_high) - (low["resid_pre"] / n_low)).astype(
                np.float32
            )
            mlp = ((high["mlp"] / n_high) - (low["mlp"] / n_low)).astype(np.float32)
            attn_out = ((high["attn_out"] / n_high) - (low["attn_out"] / n_low)).astype(
                np.float32
            )
            mlp_out = ((high["mlp_out"] / n_high) - (low["mlp_out"] / n_low)).astype(
                np.float32
            )
            attn_heads = None
            if need_heads:
                if "attn_heads" not in high or "attn_heads" not in low:
                    raise ValueError(
                        f"No per-head concat/o writes for mass-mean at {pos!r}."
                    )
                attn_heads = (
                    (high["attn_heads"] / n_high) - (low["attn_heads"] / n_low)
                ).astype(np.float32)
                head_sum = attn_heads.sum(axis=1)
                completeness = float(np.max(np.abs(head_sum - attn_out)))
                logging.info(
                    "Mass-mean head completeness %s: max |sum_h d_h - d_attn|=%.6g "
                    "(n_heads=%d)",
                    pos,
                    completeness,
                    int(attn_heads.shape[1]),
                )
        out[pos] = SiteDirections(
            resid_pre=resid_pre,
            mlp=mlp,
            n_high=n_high,
            n_low=n_low,
            attn_out=attn_out,
            mlp_out=mlp_out,
            attn_heads=attn_heads,
            resid_stream=resid_stream,
        )
        logging.info(
            "Mass-mean direction %s: n_high=%d n_low=%d resid_pre=%s mlp=%s "
            "attn_out=%s mlp_out=%s attn_heads=%s resid_stream=%s",
            pos,
            n_high,
            n_low,
            tuple(resid_pre.shape),
            tuple(mlp.shape),
            tuple(attn_out.shape),
            tuple(mlp_out.shape),
            None if attn_heads is None else tuple(attn_heads.shape),
            None if resid_stream is None else tuple(resid_stream.shape),
        )
    if skipped_bad:
        logging.info("Direction stream skipped %d unreadable site rows.", skipped_bad)
    if skipped_no_head_src:
        logging.info(
            "Direction stream skipped %d examples missing concat and o for head writes.",
            skipped_no_head_src,
        )
    if resum_fail_counts:
        logging.info(
            "Attn-addend H5 rebuild soft-fail counts by layer: %s",
            dict(sorted(resum_fail_counts.items())),
        )
    del sums
    gc.collect()
    return out


def save_directions_npz(path: Path, directions: dict[str, SiteDirections]) -> None:
    payload: dict[str, np.ndarray] = {}
    for pos, d in directions.items():
        payload[f"{pos}__resid_pre"] = d.resid_pre
        payload[f"{pos}__mlp"] = d.mlp
        payload[f"{pos}__attn_out"] = d.attn_out
        payload[f"{pos}__mlp_out"] = d.mlp_out
        payload[f"{pos}__n_high"] = np.asarray(d.n_high, dtype=np.int64)
        payload[f"{pos}__n_low"] = np.asarray(d.n_low, dtype=np.int64)
        if d.attn_heads is not None:
            payload[f"{pos}__attn_heads"] = d.attn_heads
        if d.resid_stream is not None:
            payload[f"{pos}__resid_stream"] = d.resid_stream
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# TransformerLens site-gradient capture
# ---------------------------------------------------------------------------


def _retain_or_leaf(activation: torch.Tensor) -> torch.Tensor:
    """Keep a site on the autograd graph even when model weights are frozen.

    ``retain_grad()`` raises if ``requires_grad`` is False (the usual case after
    ``freeze_model``). Detach and re-enable grad so attribution patching can
    still read ``∂ΔL/∂x_site``.
    """
    if activation.requires_grad:
        activation.retain_grad()
        return activation
    return activation.detach().requires_grad_(True)


def _replace_last_hidden(full: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """Replace the last sequence position of ``full`` ``[B, S, D]`` with ``last`` ``[B, D]``."""
    if full.ndim != 3:
        raise ValueError(f"Expected rank-3 hidden tensor, got {tuple(full.shape)}")
    if last.ndim != 2:
        raise ValueError(f"Expected rank-2 last-pos tensor, got {tuple(last.shape)}")
    if full.shape[1] == 1:
        return last.unsqueeze(1)
    return torch.cat([full[:, :-1, :].detach(), last.unsqueeze(1)], dim=1)


def _head_writes_from_z(
    z_last: torch.Tensor,
    w_o: torch.Tensor,
    b_o: torch.Tensor | None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Per-head residual writes ``z_h @ W_O[h]`` and their sum (plus ``b_O``).

    ``z_last`` is ``[batch, n_heads, d_head]``; ``w_o`` is
    ``[n_heads, d_head, d_model]``.
    """
    writes = torch.einsum("bhd,hde->bhe", z_last.float(), w_o.float())
    heads = [writes[:, h, :] for h in range(int(writes.shape[1]))]
    total = writes.sum(dim=1)
    if b_o is not None:
        total = total + b_o.float()
    return heads, total


def _rms_norm_weight_eps(norm) -> tuple[torch.Tensor, float]:
    eps = float(getattr(norm, "eps", 1e-6))
    weight = getattr(norm, "w", None)
    if weight is None:
        weight = getattr(norm, "weight", None)
    if weight is None:
        raise ValueError("Post-attn RMSNorm has neither .w nor .weight.")
    return weight, eps


def _rms_norm_use_one_plus(
    norm, x: torch.Tensor, *, weight: torch.Tensor, eps: float
) -> bool:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    w = weight.float()
    with torch.no_grad():
        y = norm(x.detach()).float()
        pred_w = xf.detach() * (inv.detach() * w)
        pred_1pw = xf.detach() * (inv.detach() * (1.0 + w))
        return bool((pred_1pw - y).abs().mean() <= (pred_w - y).abs().mean())


def _shared_rmsnorm_scale(x: torch.Tensor, norm) -> torch.Tensor:
    """Per-token scale ``s`` such that ``norm(x) ≈ x * s`` (shared across heads)."""
    weight, eps = _rms_norm_weight_eps(norm)
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    w = weight.float()
    use_one_plus = _rms_norm_use_one_plus(norm, x, weight=weight, eps=eps)
    return inv * ((1.0 + w) if use_one_plus else w)


def extract_attn_head_write_specs(model) -> list[AttnHeadWriteSpec]:
    """Copy TL ``W_O`` / ``b_O`` / sandwich ``ln1_post`` into numpy for H5 concat."""
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    d_model = int(model.cfg.d_model)
    specs: list[AttnHeadWriteSpec] = []
    n_sandwich = 0
    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        w_o = attn.W_O.detach().float().cpu().numpy()
        if tuple(w_o.shape) != (n_heads, d_head, d_model):
            raise ValueError(
                f"Layer {layer} W_O shape {tuple(w_o.shape)} != "
                f"(n_heads, d_head, d_model)=({n_heads}, {d_head}, {d_model})."
            )
        b_o_t = getattr(attn, "b_O", None)
        b_o = None if b_o_t is None else b_o_t.detach().float().cpu().numpy()
        ln1_post = getattr(model.blocks[layer], "ln1_post", None)
        rms_weight = None
        rms_eps = 1e-6
        rms_use_one_plus = False
        if ln1_post is not None:
            n_sandwich += 1
            weight, rms_eps = _rms_norm_weight_eps(ln1_post)
            rms_weight = weight.detach().float().cpu().numpy()
            dummy = torch.ones(
                (1, d_model), device=weight.device, dtype=weight.dtype
            )
            rms_use_one_plus = _rms_norm_use_one_plus(
                ln1_post, dummy, weight=weight, eps=rms_eps
            )
        specs.append(
            AttnHeadWriteSpec(
                n_heads=n_heads,
                d_head=d_head,
                d_model=d_model,
                w_o=w_o,
                b_o=b_o,
                rms_weight=rms_weight,
                rms_eps=rms_eps,
                rms_use_one_plus=rms_use_one_plus,
            )
        )
    if n_sandwich and n_sandwich != n_layers:
        logging.warning(
            "ln1_post present on %d/%d layers; sandwich RMS applied only where present.",
            n_sandwich,
            n_layers,
        )
    logging.info(
        "Head-write specs: n_layers=%d n_heads=%d sandwich_ln1_post=%s "
        "rms_use_one_plus=%s",
        n_layers,
        n_heads,
        n_sandwich == n_layers,
        specs[0].rms_use_one_plus if specs else False,
    )
    return specs


class SiteInputCapture:
    """Retain grads of per-head residual copies and MLP-branch input.

    Requires ``cfg.use_split_qkv_input`` and ``cfg.use_hook_mlp_in`` so
    ``hook_q/k/v_input`` and ``hook_mlp_in`` run. MLP uses a clone of resid-mid,
    so the skip connection is not part of the MLP site.

    When ``capture_outputs`` is true, a *second* independent forward retains the
    attention residual addend at ``hook_attn_out`` (GBA-style per-head rebuild
    from ``hook_z`` when fine granularity is on; after ``ln1_post`` on sandwich
    Gemma-3) and ``hook_mlp_out`` (after ``ln2_post`` on sandwich Gemma-3). That
    rebuild detaches ``hook_z`` last-pos and cannot share a graph with input-site
    Q/K/V copies.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_model: int,
        capture_outputs: bool = False,
        need_fine: bool = False,
    ) -> None:
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.d_head = int(d_head)
        self.d_model = int(d_model)
        self.capture_outputs = bool(capture_outputs)
        self.need_fine = bool(need_fine)
        self.q: list[torch.Tensor | None] = [None] * self.n_layers
        self.k: list[torch.Tensor | None] = [None] * self.n_layers
        self.v: list[torch.Tensor | None] = [None] * self.n_layers
        self.mlp: list[torch.Tensor | None] = [None] * self.n_layers
        self.z_last: list[torch.Tensor | None] = [None] * self.n_layers
        self.attn_heads: list[list[torch.Tensor] | None] = [None] * self.n_layers
        self.attn_out: list[torch.Tensor | None] = [None] * self.n_layers
        self.mlp_out: list[torch.Tensor | None] = [None] * self.n_layers
        self._model = None
        self._attn_resum_warned: set[int] = set()

    def clear_captured(self) -> None:
        for seq in (
            self.q,
            self.k,
            self.v,
            self.mlp,
            self.z_last,
            self.attn_heads,
            self.attn_out,
            self.mlp_out,
        ):
            for i in range(len(seq)):
                seq[i] = None

    def iter_tensors(self):
        sequences = [self.q, self.k, self.v, self.mlp]
        if self.capture_outputs:
            sequences.extend([self.attn_out, self.mlp_out])
        for seq in sequences:
            for t in seq:
                if t is not None:
                    yield t
        if self.capture_outputs:
            for heads in self.attn_heads:
                if heads is None:
                    continue
                for h_t in heads:
                    yield h_t

    def zero_grads(self) -> None:
        for t in self.iter_tensors():
            if t.grad is not None:
                t.grad = None

    def _make_retain_hook(self, store_name: str, layer: int):
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            out = _retain_or_leaf(activation)
            getattr(self, store_name)[layer] = out
            return out

        return hook_fn

    @staticmethod
    def _embed_hook(activation: torch.Tensor, hook) -> torch.Tensor:
        del hook
        return activation.detach().requires_grad_(True)

    def _make_z_stash_hook(self, layer: int):
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            last = activation[:, -1]
            self.z_last[layer] = last.detach()
            detached_last = last.detach()
            if activation.shape[1] == 1:
                return detached_last.unsqueeze(1)
            return torch.cat([activation[:, :-1], detached_last.unsqueeze(1)], dim=1)

        return hook_fn

    def _make_attn_out_hook(self, layer: int):
        if not self.need_fine:
            return self._make_retain_hook("attn_out", layer)

        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            if self._model is None:
                raise RuntimeError("SiteInputCapture has no model; call register() first.")
            z_last = self.z_last[layer]
            if z_last is None:
                raise RuntimeError(f"Missing stashed hook_z last pos at layer {layer}.")
            attn = self._model.blocks[layer].attn
            z_leaf = z_last.detach().requires_grad_(True)
            heads, total_pre = _head_writes_from_z(
                z_leaf, attn.W_O, getattr(attn, "b_O", None)
            )
            orig_last = activation[:, -1].detach()
            ln1_post = getattr(self._model.blocks[layer], "ln1_post", None)
            if ln1_post is not None:
                scale = _shared_rmsnorm_scale(total_pre, ln1_post)
                post_heads = [(h_t * scale).to(dtype=activation.dtype) for h_t in heads]
            else:
                scale = None
                post_heads = [h_t.to(dtype=activation.dtype) for h_t in heads]
            for ph in post_heads:
                ph.retain_grad()
            total_post = post_heads[0]
            for h_t in post_heads[1:]:
                total_post = total_post + h_t
            b_o = getattr(attn, "b_O", None)
            if b_o is not None:
                bias = b_o.to(device=total_post.device, dtype=total_post.dtype)
                if scale is not None:
                    bias = (b_o.float() * scale).to(dtype=total_post.dtype)
                total_post = total_post + bias
            total_post.retain_grad()
            delta = (total_post.detach().float() - orig_last.float()).abs().max()
            if float(delta) > ATTN_OUT_RESUM_ATOL and layer not in self._attn_resum_warned:
                self._attn_resum_warned.add(layer)
                logging.warning(
                    "Attn-addend rebuild soft-fail layer=%d: max |recon-hook_attn_out|=%.6g "
                    "> atol=%g. Check W_O slicing / ln1_post RMS scale.",
                    layer,
                    float(delta),
                    ATTN_OUT_RESUM_ATOL,
                )
            self.attn_heads[layer] = post_heads
            self.attn_out[layer] = total_post
            return _replace_last_hidden(activation.detach(), total_post)

        return hook_fn

    def register(self, model) -> None:
        """Validate named hook points; capture happens in ``input_fwd_hooks`` / ``output_fwd_hooks``."""
        self._model = model
        self._attn_resum_warned = set()
        if not bool(getattr(model.cfg, "use_split_qkv_input", False)):
            raise ValueError(
                "use_split_qkv_input must be True so hook_q/k/v_input run on residual copies."
            )
        if not bool(getattr(model.cfg, "use_hook_mlp_in", False)):
            raise ValueError(
                "use_hook_mlp_in must be True so hook_mlp_in clones resid-mid off the skip."
            )
        hook_dict = getattr(model, "hook_dict", None)
        if not isinstance(hook_dict, dict) or not hook_dict:
            raise ValueError(
                "HookedTransformer is missing hook_dict; cannot attach site hooks."
            )
        needed = ["hook_embed"]
        for layer in range(self.n_layers):
            needed.extend(
                [
                    f"blocks.{layer}.hook_q_input",
                    f"blocks.{layer}.hook_k_input",
                    f"blocks.{layer}.hook_v_input",
                    f"blocks.{layer}.hook_mlp_in",
                ]
            )
            if self.capture_outputs:
                needed.append(f"blocks.{layer}.hook_attn_out")
                needed.append(f"blocks.{layer}.hook_mlp_out")
                if self.need_fine:
                    needed.append(f"blocks.{layer}.attn.hook_z")
        missing = [name for name in needed if name not in hook_dict]
        if missing:
            raise ValueError(
                "Model is missing TransformerLens hook points required for "
                f"attribution patching: {missing[:8]}."
            )

    def input_fwd_hooks(self) -> list[tuple[str, object]]:
        hooks: list[tuple[str, object]] = [("hook_embed", self._embed_hook)]
        for layer in range(self.n_layers):
            hooks.append(
                (f"blocks.{layer}.hook_q_input", self._make_retain_hook("q", layer))
            )
            hooks.append(
                (f"blocks.{layer}.hook_k_input", self._make_retain_hook("k", layer))
            )
            hooks.append(
                (f"blocks.{layer}.hook_v_input", self._make_retain_hook("v", layer))
            )
            hooks.append(
                (f"blocks.{layer}.hook_mlp_in", self._make_retain_hook("mlp", layer))
            )
        return hooks

    def output_fwd_hooks(self) -> list[tuple[str, object]]:
        if not self.capture_outputs:
            raise RuntimeError("output_fwd_hooks requires capture_outputs=True.")
        hooks: list[tuple[str, object]] = [("hook_embed", self._embed_hook)]
        for layer in range(self.n_layers):
            if self.need_fine:
                hooks.append(
                    (f"blocks.{layer}.attn.hook_z", self._make_z_stash_hook(layer))
                )
            hooks.append(
                (f"blocks.{layer}.hook_attn_out", self._make_attn_out_hook(layer))
            )
            hooks.append(
                (
                    f"blocks.{layer}.hook_mlp_out",
                    self._make_retain_hook("mlp_out", layer),
                )
            )
        return hooks

    def assert_input_sites_ready(self) -> None:
        missing = [
            i
            for i in range(self.n_layers)
            if self.q[i] is None
            or self.k[i] is None
            or self.v[i] is None
            or self.mlp[i] is None
        ]
        if missing:
            raise RuntimeError(
                f"Site hooks missed Q/K/V/MLP tensors at layers {missing[:8]}."
            )
        not_grad = [
            i
            for i in range(self.n_layers)
            if not self.q[i].requires_grad
            or not self.k[i].requires_grad
            or not self.v[i].requires_grad
            or not self.mlp[i].requires_grad
        ]
        if not_grad:
            raise RuntimeError(
                "Captured site tensors do not require grad at layers "
                f"{not_grad[:8]}. Attribution patching needs a differentiable "
                "path from hook_embed through hook_q/k/v_input and hook_mlp_in."
            )

    def assert_output_sites_ready(self) -> None:
        if not self.capture_outputs:
            raise RuntimeError("Output-site checks require capture_outputs=True.")
        missing = [
            i
            for i in range(self.n_layers)
            if self.attn_out[i] is None or self.mlp_out[i] is None
        ]
        if self.need_fine:
            missing.extend(
                i
                for i in range(self.n_layers)
                if self.attn_heads[i] is None
            )
        if missing:
            raise RuntimeError(
                f"Site hooks missed attn_out/mlp_out tensors at layers {missing[:8]}."
            )
        not_grad = [
            i
            for i in range(self.n_layers)
            if not self.attn_out[i].requires_grad
            or not self.mlp_out[i].requires_grad
        ]
        if not_grad:
            raise RuntimeError(
                "Captured output-site tensors do not require grad at layers "
                f"{not_grad[:8]}. Attribution patching needs a differentiable "
                "path through hook_attn_out / hook_mlp_out."
            )

    def remove(self) -> None:
        if self._model is not None:
            self._model.reset_hooks(including_permanent=True)
            self._model = None


def build_prefix_input_ids_tl(
    model,
    example: GradExample,
    abs_gen_index: int,
    *,
    prompt_prefix: str,
) -> torch.Tensor:
    """Prompt + completion prefix using sampled ids when available."""
    prompt_ids = model.to_tokens(prompt_prefix + example.question)
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("HookedTransformer has no tokenizer.")
    gen_ids = prefix_generated_token_ids(example, abs_gen_index, tokenizer)
    if not gen_ids:
        return prompt_ids
    gen = torch.tensor([gen_ids], device=prompt_ids.device, dtype=prompt_ids.dtype)
    return torch.cat([prompt_ids, gen], dim=1)


def _last_pos_grad(t: torch.Tensor | None, *, name: str, layer: int) -> torch.Tensor:
    if t is None:
        raise RuntimeError(f"Missing captured {name} at layer {layer}.")
    if t.grad is None:
        raise RuntimeError(f"Missing gradient for {name} at layer {layer}.")
    g = t.grad.float()
    if g.ndim < 2:
        raise RuntimeError(
            f"Expected {name} grad rank >= 2 at layer {layer}, got {tuple(g.shape)}."
        )
    if g.ndim == 2:
        return g
    return g[:, -1]


def _residual_input_dots(grad_last: torch.Tensor, direction: np.ndarray) -> np.ndarray:
    """grad_last [batch, n_units, d_model] · direction [d_model] -> [n_units]."""
    if grad_last.ndim != 3:
        raise ValueError(
            f"Expected [batch, units, d_model] grad, got {tuple(grad_last.shape)}."
        )
    g = grad_last[0]
    d = torch.as_tensor(direction, device=g.device, dtype=torch.float32)
    if d.ndim != 1:
        raise ValueError(f"Expected 1D residual direction, got {tuple(d.shape)}.")
    if g.shape[-1] != d.shape[0]:
        raise ValueError(
            f"Residual direction dim {int(d.shape[0])} != grad last dim {int(g.shape[-1])}."
        )
    return (g * d).sum(dim=-1).detach().cpu().numpy().astype(np.float32)


def _mlp_dot(grad_last: torch.Tensor, direction: np.ndarray) -> float:
    if grad_last.ndim != 2:
        raise ValueError(f"Expected [batch, d_model] MLP grad, got {tuple(grad_last.shape)}.")
    g = grad_last[0]
    d = torch.as_tensor(direction, device=g.device, dtype=torch.float32)
    if g.shape != d.shape:
        raise ValueError(f"MLP direction shape {tuple(d.shape)} != grad {tuple(g.shape)}.")
    return float((g * d).sum().detach().cpu().item())


def combined_head_scores(
    q_scores: np.ndarray,
    k_scores: np.ndarray,
    v_scores: np.ndarray,
    *,
    n_heads: int,
    n_kv_heads: int,
) -> np.ndarray:
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}.")
    group = int(n_heads) // int(n_kv_heads)
    out = np.zeros(int(n_heads), dtype=np.float32)
    for h in range(int(n_heads)):
        kv = h // group
        out[h] = float(q_scores[h]) + float(k_scores[kv]) + float(v_scores[kv])
    return out


def scores_from_capture(
    capture: SiteInputCapture,
    direction: SiteDirections,
    *,
    granularity: str,
) -> np.ndarray:
    n_layers = capture.n_layers
    n_heads = capture.n_heads
    n_kv = capture.n_kv_heads
    if granularity == "coarse":
        labels_n = 2 * n_layers
        row = np.zeros(labels_n, dtype=np.float32)
        for layer in range(n_layers):
            d_res = direction.resid_pre[layer]
            q_s = _residual_input_dots(
                _last_pos_grad(capture.q[layer], name="q_input", layer=layer),
                d_res,
            )
            k_s = _residual_input_dots(
                _last_pos_grad(capture.k[layer], name="k_input", layer=layer),
                d_res,
            )
            v_s = _residual_input_dots(
                _last_pos_grad(capture.v[layer], name="v_input", layer=layer),
                d_res,
            )
            heads = combined_head_scores(
                q_s, k_s, v_s, n_heads=n_heads, n_kv_heads=n_kv
            )
            mlp_s = _mlp_dot(
                _last_pos_grad(capture.mlp[layer], name="mlp", layer=layer),
                direction.mlp[layer],
            )
            row[2 * layer] = float(heads.sum())
            row[2 * layer + 1] = mlp_s
        return row
    if granularity != "fine":
        raise ValueError(f"Unknown granularity {granularity!r}")
    labels_n = n_layers * (n_heads + 1)
    row = np.zeros(labels_n, dtype=np.float32)
    col = 0
    for layer in range(n_layers):
        d_res = direction.resid_pre[layer]
        q_s = _residual_input_dots(
            _last_pos_grad(capture.q[layer], name="q_input", layer=layer),
            d_res,
        )
        k_s = _residual_input_dots(
            _last_pos_grad(capture.k[layer], name="k_input", layer=layer),
            d_res,
        )
        v_s = _residual_input_dots(
            _last_pos_grad(capture.v[layer], name="v_input", layer=layer),
            d_res,
        )
        heads = combined_head_scores(q_s, k_s, v_s, n_heads=n_heads, n_kv_heads=n_kv)
        mlp_s = _mlp_dot(
            _last_pos_grad(capture.mlp[layer], name="mlp", layer=layer),
            direction.mlp[layer],
        )
        row[col : col + n_heads] = heads
        col += n_heads
        row[col] = mlp_s
        col += 1
    return row


def _attn_head_mass_mean_dots(
    heads: list[torch.Tensor],
    d_heads: np.ndarray,
    *,
    layer: int,
) -> np.ndarray:
    """Fine scores ``⟨∇_post_h, d_h⟩`` with per-head mass-mean directions."""
    n_heads = len(heads)
    if d_heads.ndim != 2 or int(d_heads.shape[0]) != n_heads:
        raise ValueError(
            f"Layer {layer} head directions shape {tuple(d_heads.shape)} != "
            f"(n_heads, d_model)=({n_heads}, ...)."
        )
    out = np.zeros(n_heads, dtype=np.float32)
    for h, ph in enumerate(heads):
        gh = _last_pos_grad(ph, name=f"attn_head_{h}", layer=layer)[0]
        d = torch.as_tensor(d_heads[h], device=gh.device, dtype=torch.float32)
        if gh.shape != d.shape:
            raise ValueError(
                f"Layer {layer} head {h} grad shape {tuple(gh.shape)} != "
                f"mass-mean direction {tuple(d.shape)}."
            )
        out[h] = float((gh * d).sum().detach().cpu().item())
    return out


def scores_from_output_capture(
    capture: SiteInputCapture,
    direction: SiteDirections,
    *,
    granularity: str,
) -> np.ndarray:
    """Attention addend at ``hook_attn_out`` and ``hook_mlp_out``."""
    if not capture.capture_outputs:
        raise RuntimeError("Output-site scoring requires capture_outputs=True.")
    n_layers = capture.n_layers
    n_heads = capture.n_heads
    if granularity == "coarse":
        labels_n = 2 * n_layers
        row = np.zeros(labels_n, dtype=np.float32)
        for layer in range(n_layers):
            attn_s = _mlp_dot(
                _last_pos_grad(capture.attn_out[layer], name="attn_out", layer=layer),
                direction.attn_out[layer],
            )
            mlp_s = _mlp_dot(
                _last_pos_grad(capture.mlp_out[layer], name="mlp_out", layer=layer),
                direction.mlp_out[layer],
            )
            row[2 * layer] = attn_s
            row[2 * layer + 1] = mlp_s
        return row
    if granularity != "fine":
        raise ValueError(f"Unknown granularity {granularity!r}")
    if not capture.need_fine:
        raise RuntimeError("Fine output-site scoring requires need_fine=True.")
    if direction.attn_heads is None:
        raise RuntimeError(
            "Fine output-site scoring requires per-head mass-mean directions."
        )
    labels_n = n_layers * (n_heads + 1)
    row = np.zeros(labels_n, dtype=np.float32)
    col = 0
    for layer in range(n_layers):
        heads = capture.attn_heads[layer]
        if heads is None or len(heads) != n_heads:
            raise RuntimeError(
                f"Fine output attribution missing heads at layer {layer} "
                f"(got {None if heads is None else len(heads)})."
            )
        d_heads = direction.attn_heads[layer]
        if tuple(d_heads.shape) != (n_heads, capture.d_model):
            raise ValueError(
                f"Layer {layer} attn_heads direction shape {tuple(d_heads.shape)} != "
                f"(n_heads, d_model)=({n_heads}, {capture.d_model})."
            )
        head_scores = _attn_head_mass_mean_dots(heads, d_heads, layer=layer)
        mlp_s = _mlp_dot(
            _last_pos_grad(capture.mlp_out[layer], name="mlp_out", layer=layer),
            direction.mlp_out[layer],
        )
        row[col : col + n_heads] = head_scores
        col += n_heads
        row[col] = mlp_s
        col += 1
    return row


def freeze_model(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def enable_split_input_hooks(model) -> None:
    """Route residual copies through hook_q/k/v_input and MLP through hook_mlp_in."""
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)


class ResidualStreamCapture:
    """Retain grads of display-layer residual-stream sites without breaking the chain.

    Display 0 (``blocks.0.hook_resid_pre``) may become a grad leaf. Later display
    layers (``hook_resid_post`` of block k-1) only ``retain_grad`` on the live
    residual so earlier-layer dots still include indirect paths through ΔL.
    """

    def __init__(self, *, n_layers: int, d_model: int) -> None:
        self.n_layers = int(n_layers)
        self.n_display = int(n_layers) + 1
        self.d_model = int(d_model)
        self.resid: list[torch.Tensor | None] = [None] * self.n_display
        self._model = None

    def clear_captured(self) -> None:
        for i in range(len(self.resid)):
            self.resid[i] = None

    def iter_tensors(self):
        for t in self.resid:
            if t is not None:
                yield t

    def zero_grads(self) -> None:
        for t in self.iter_tensors():
            if t.grad is not None:
                t.grad = None

    def _make_retain_hook(self, display_layer: int, *, allow_leaf: bool):
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            del hook
            if allow_leaf:
                out = _retain_or_leaf(activation)
            else:
                if not activation.requires_grad:
                    raise RuntimeError(
                        f"Display layer {display_layer} residual does not require grad. "
                        "Detaching intermediate residual-stream sites would cut earlier "
                        "layers off from ΔL."
                    )
                activation.retain_grad()
                out = activation
            self.resid[display_layer] = out
            return out

        return hook_fn

    def register(self, model) -> None:
        self._model = model
        hook_dict = getattr(model, "hook_dict", None)
        if not isinstance(hook_dict, dict) or not hook_dict:
            raise ValueError(
                "HookedTransformer is missing hook_dict; cannot attach residual-stream hooks."
            )
        needed = [hook_name_for_display_layer(d) for d in range(self.n_display)]
        missing = [name for name in needed if name not in hook_dict]
        if missing:
            raise ValueError(
                "Model is missing TransformerLens hook points required for "
                f"residual-stream attribution: {missing[:8]}."
            )

    def fwd_hooks(self) -> list[tuple[str, object]]:
        hooks: list[tuple[str, object]] = []
        for display in range(self.n_display):
            hooks.append(
                (
                    hook_name_for_display_layer(display),
                    self._make_retain_hook(display, allow_leaf=display == 0),
                )
            )
        return hooks

    def assert_sites_ready(self) -> None:
        missing = [i for i in range(self.n_display) if self.resid[i] is None]
        if missing:
            raise RuntimeError(
                f"Residual-stream hooks missed tensors at display layers {missing[:8]}."
            )
        not_grad = [
            i for i in range(self.n_display) if not self.resid[i].requires_grad
        ]
        if not_grad:
            raise RuntimeError(
                "Captured residual-stream tensors do not require grad at display "
                f"layers {not_grad[:8]}."
            )

    def remove(self) -> None:
        if self._model is not None:
            self._model.reset_hooks(including_permanent=True)
            self._model = None


def scores_from_residual_capture(
    capture: ResidualStreamCapture,
    direction: SiteDirections,
) -> np.ndarray:
    if direction.resid_stream is None:
        raise RuntimeError("Residual-stream scoring requires resid_stream directions.")
    d_all = direction.resid_stream
    if d_all.ndim != 2 or int(d_all.shape[0]) != capture.n_display:
        raise ValueError(
            f"resid_stream direction shape {tuple(d_all.shape)} != "
            f"(n_display, d_model)=({capture.n_display}, {capture.d_model})."
        )
    if int(d_all.shape[1]) != capture.d_model:
        raise ValueError(
            f"resid_stream d_model {int(d_all.shape[1])} != capture d_model={capture.d_model}."
        )
    row = np.zeros(capture.n_display, dtype=np.float32)
    for display in range(capture.n_display):
        row[display] = _mlp_dot(
            _last_pos_grad(
                capture.resid[display], name="resid_stream", layer=display
            ),
            d_all[display],
        )
    return row


def _score_residual_pass(
    *,
    model,
    tokens: torch.Tensor,
    capture: ResidualStreamCapture,
    pos_contrasts: Sequence[ContrastSpec],
    direction: SiteDirections,
    position: str,
) -> dict[tuple[str, str], tuple[np.ndarray, float, float]]:
    out: dict[tuple[str, str], tuple[np.ndarray, float, float]] = {}
    capture.clear_captured()
    capture.zero_grads()
    with torch.enable_grad():
        logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=capture.fwd_hooks(),
            reset_hooks_end=True,
        )
        capture.assert_sites_ready()
        last_logits = logits[0, -1].float()
        for ci, contrast in enumerate(pos_contrasts):
            retain = ci < len(pos_contrasts) - 1
            ld = logit_diff_from_logits(
                last_logits, contrast.pos_token_ids, contrast.neg_token_ids
            )
            ld.backward(retain_graph=retain)
            ld_val = float(ld.detach().item())
            row = scores_from_residual_capture(capture, direction)
            out[(position, contrast.name)] = (row, 0.0, ld_val)
            capture.zero_grads()
    del logits, last_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_residual_attribution_for_example(
    *,
    model,
    example: GradExample,
    capture: ResidualStreamCapture,
    contrasts: Sequence[ContrastSpec],
    expected_span: int,
    prompt_prefix: str,
    directions: dict[str, SiteDirections],
) -> dict[tuple[str, str], tuple[np.ndarray, float, float]]:
    out: dict[tuple[str, str], tuple[np.ndarray, float, float]] = {}
    by_position: dict[str, list[ContrastSpec]] = {}
    for c in contrasts:
        by_position.setdefault(c.position, []).append(c)

    for position, pos_contrasts in by_position.items():
        if position not in directions:
            raise ValueError(f"No mass-mean direction computed for position {position!r}.")
        direction = directions[position]
        span_idx = contrast_span_index(pos_contrasts[0], example, expected_span)
        for extra in pos_contrasts[1:]:
            other = contrast_span_index(extra, example, expected_span)
            if other != span_idx:
                raise ValueError(
                    f"Contrasts at position {position!r} disagree on span index "
                    f"({span_idx} vs {other})."
                )
        abs_idx = example.first_prob_token_index + span_idx
        if abs_idx < 0 or abs_idx >= len(example.decoded_tokens):
            raise ValueError(
                f"Example {example.example_id}: abs index {abs_idx} out of range "
                f"for {position} (len={len(example.decoded_tokens)})."
            )
        tokens = build_prefix_input_ids_tl(
            model,
            example,
            abs_idx,
            prompt_prefix=prompt_prefix,
        )
        out.update(
            _score_residual_pass(
                model=model,
                tokens=tokens,
                capture=capture,
                pos_contrasts=pos_contrasts,
                direction=direction,
                position=position,
            )
        )
        del tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def run_residual_cohort_attributions(
    *,
    examples: Sequence[GradExample],
    model,
    capture: ResidualStreamCapture,
    contrasts: Sequence[ContrastSpec],
    expected_span: int,
    desc: str,
    prompt_prefix: str,
    directions: dict[str, SiteDirections],
) -> dict[tuple[str, str], ExperimentResult]:
    n_ex = len(examples)
    labels = residual_stream_labels(capture.n_layers)
    keys = [(c.position, c.name) for c in contrasts]
    attrs: dict[tuple[str, str], np.ndarray] = {
        key: np.zeros((n_ex, len(labels)), dtype=np.float32) for key in keys
    }
    embeds: dict[tuple[str, str], np.ndarray] = {
        key: np.zeros(n_ex, dtype=np.float32) for key in keys
    }
    scores: dict[tuple[str, str], np.ndarray] = {
        key: np.zeros(n_ex, dtype=np.float32) for key in keys
    }
    example_ids: list[str] = []
    skipped = 0
    ei = 0
    for ex in tqdm(examples, desc=desc):
        try:
            per = run_residual_attribution_for_example(
                model=model,
                example=ex,
                capture=capture,
                contrasts=contrasts,
                expected_span=expected_span,
                prompt_prefix=prompt_prefix,
                directions=directions,
            )
        except (ValueError, RuntimeError) as exc:
            skipped += 1
            logging.warning("Skipping example %s: %s", ex.example_id, exc)
            continue
        example_ids.append(ex.example_id)
        for key in keys:
            row, embed_attr, ld_val = per[key]
            attrs[key][ei] = row
            embeds[key][ei] = embed_attr
            scores[key][ei] = ld_val
        ei += 1

    if skipped:
        logging.info("%s skipped %d examples.", desc, skipped)
    if ei == 0:
        raise ValueError(f"No examples succeeded for cohort {desc!r}.")

    packed: dict[tuple[str, str], ExperimentResult] = {}
    for position, name in keys:
        key = (position, name)
        packed[key] = ExperimentResult(
            position=position,
            contrast_name=name,
            granularity=RESIDUAL_STREAM_GRANULARITY,
            labels=labels,
            example_ids=example_ids,
            attributions=attrs[key][:ei],
            embed_attributions=embeds[key][:ei],
            true_scores=scores[key][:ei],
            completeness_residuals=np.zeros(ei, dtype=np.float32),
        )
    return packed


def _score_site_pass(
    *,
    model,
    tokens: torch.Tensor,
    capture: SiteInputCapture,
    hooks: list[tuple[str, object]],
    pos_contrasts: Sequence[ContrastSpec],
    direction: SiteDirections,
    gran_list: Sequence[str],
    position: str,
    kind: str,
) -> dict[tuple[str, str, str], tuple[np.ndarray, float, float]]:
    """One independent forward+backward for input-site or output-site scores."""
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}.")
    out: dict[tuple[str, str, str], tuple[np.ndarray, float, float]] = {}
    capture.clear_captured()
    capture.zero_grads()
    with torch.enable_grad():
        logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=hooks,
            reset_hooks_end=True,
        )
        if kind == "input":
            capture.assert_input_sites_ready()
        else:
            capture.assert_output_sites_ready()
        last_logits = logits[0, -1].float()
        for ci, contrast in enumerate(pos_contrasts):
            retain = ci < len(pos_contrasts) - 1
            ld = logit_diff_from_logits(
                last_logits, contrast.pos_token_ids, contrast.neg_token_ids
            )
            ld.backward(retain_graph=retain)
            ld_val = float(ld.detach().item())
            for gran in gran_list:
                if kind == "input":
                    row = scores_from_capture(capture, direction, granularity=gran)
                else:
                    row = scores_from_output_capture(
                        capture, direction, granularity=gran
                    )
                out[(position, contrast.name, gran)] = (row, 0.0, ld_val)
            capture.zero_grads()
    del logits, last_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_attribution_for_example(
    *,
    model,
    tokenizer,
    capture: SiteInputCapture,
    example: GradExample,
    contrasts: Sequence[ContrastSpec],
    expected_span: int,
    gran_list: Sequence[str],
    prompt_prefix: str,
    directions: dict[str, SiteDirections],
) -> tuple[
    dict[tuple[str, str, str], tuple[np.ndarray, float, float]],
    dict[tuple[str, str, str], tuple[np.ndarray, float, float]],
]:
    del tokenizer
    input_out: dict[tuple[str, str, str], tuple[np.ndarray, float, float]] = {}
    output_out: dict[tuple[str, str, str], tuple[np.ndarray, float, float]] = {}
    by_position: dict[str, list[ContrastSpec]] = {}
    for c in contrasts:
        by_position.setdefault(c.position, []).append(c)

    for position, pos_contrasts in by_position.items():
        if position not in directions:
            raise ValueError(f"No mass-mean direction computed for position {position!r}.")
        direction = directions[position]
        span_idx = contrast_span_index(pos_contrasts[0], example, expected_span)
        for extra in pos_contrasts[1:]:
            other = contrast_span_index(extra, example, expected_span)
            if other != span_idx:
                raise ValueError(
                    f"Contrasts at position {position!r} disagree on span index "
                    f"({span_idx} vs {other})."
                )
        abs_idx = example.first_prob_token_index + span_idx
        if abs_idx < 0 or abs_idx >= len(example.decoded_tokens):
            raise ValueError(
                f"Example {example.example_id}: abs index {abs_idx} out of range "
                f"for {position} (len={len(example.decoded_tokens)})."
            )
        tokens = build_prefix_input_ids_tl(
            model,
            example,
            abs_idx,
            prompt_prefix=prompt_prefix,
        )
        input_out.update(
            _score_site_pass(
                model=model,
                tokens=tokens,
                capture=capture,
                hooks=capture.input_fwd_hooks(),
                pos_contrasts=pos_contrasts,
                direction=direction,
                gran_list=gran_list,
                position=position,
                kind="input",
            )
        )
        if capture.capture_outputs:
            output_out.update(
                _score_site_pass(
                    model=model,
                    tokens=tokens,
                    capture=capture,
                    hooks=capture.output_fwd_hooks(),
                    pos_contrasts=pos_contrasts,
                    direction=direction,
                    gran_list=gran_list,
                    position=position,
                    kind="output",
                )
            )
        del tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return input_out, output_out


def run_cohort_attributions(
    *,
    examples: Sequence[GradExample],
    model,
    tokenizer,
    capture: SiteInputCapture,
    contrasts: Sequence[ContrastSpec],
    expected_span: int,
    gran_list: Sequence[str],
    desc: str,
    prompt_prefix: str,
    directions: dict[str, SiteDirections],
) -> tuple[
    dict[tuple[str, str, str], ExperimentResult],
    dict[tuple[str, str, str], ExperimentResult],
]:
    n_ex = len(examples)
    n_layers = capture.n_layers
    n_heads = capture.n_heads
    label_map = {
        "coarse": component_labels_coarse(n_layers),
        "fine": component_labels_fine(n_layers, n_heads),
    }
    keys = [(c.position, c.name, g) for c in contrasts for g in gran_list]
    score_outputs = bool(capture.capture_outputs)

    def _empty_buffers() -> tuple[
        dict[tuple[str, str, str], np.ndarray],
        dict[tuple[str, str, str], np.ndarray],
        dict[tuple[str, str, str], np.ndarray],
    ]:
        attrs: dict[tuple[str, str, str], np.ndarray] = {}
        embeds: dict[tuple[str, str, str], np.ndarray] = {}
        scores: dict[tuple[str, str, str], np.ndarray] = {}
        for key in keys:
            n_comp = len(label_map[key[2]])
            attrs[key] = np.zeros((n_ex, n_comp), dtype=np.float32)
            embeds[key] = np.zeros(n_ex, dtype=np.float32)
            scores[key] = np.zeros(n_ex, dtype=np.float32)
        return attrs, embeds, scores

    attrs, embeds, scores = _empty_buffers()
    out_attrs = out_embeds = out_scores = None
    if score_outputs:
        out_attrs, out_embeds, out_scores = _empty_buffers()

    example_ids: list[str] = []
    skipped = 0
    ei = 0
    for ex in tqdm(examples, desc=desc):
        try:
            per_in, per_out = run_attribution_for_example(
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                example=ex,
                contrasts=contrasts,
                expected_span=expected_span,
                gran_list=gran_list,
                prompt_prefix=prompt_prefix,
                directions=directions,
            )
        except (ValueError, RuntimeError) as exc:
            skipped += 1
            logging.warning("Skipping example %s: %s", ex.example_id, exc)
            continue
        example_ids.append(ex.example_id)
        for key in keys:
            row, embed_attr, ld_val = per_in[key]
            attrs[key][ei] = row
            embeds[key][ei] = embed_attr
            scores[key][ei] = ld_val
            if score_outputs:
                assert out_attrs is not None and out_embeds is not None
                assert out_scores is not None
                row_o, embed_o, ld_o = per_out[key]
                out_attrs[key][ei] = row_o
                out_embeds[key][ei] = embed_o
                out_scores[key][ei] = ld_o
        ei += 1

    if skipped:
        logging.info("%s skipped %d examples.", desc, skipped)
    if ei == 0:
        raise ValueError(f"No examples succeeded for cohort {desc!r}.")

    def _pack(
        attr_buf: dict[tuple[str, str, str], np.ndarray],
        embed_buf: dict[tuple[str, str, str], np.ndarray],
        score_buf: dict[tuple[str, str, str], np.ndarray],
    ) -> dict[tuple[str, str, str], ExperimentResult]:
        packed: dict[tuple[str, str, str], ExperimentResult] = {}
        for position, name, gran in keys:
            key = (position, name, gran)
            packed[key] = ExperimentResult(
                position=position,
                contrast_name=name,
                granularity=gran,
                labels=label_map[gran],
                example_ids=example_ids,
                attributions=attr_buf[key][:ei],
                embed_attributions=embed_buf[key][:ei],
                true_scores=score_buf[key][:ei],
                completeness_residuals=np.zeros(ei, dtype=np.float32),
            )
        return packed

    input_results = _pack(attrs, embeds, scores)
    output_results: dict[tuple[str, str, str], ExperimentResult] = {}
    if score_outputs:
        assert out_attrs is not None and out_embeds is not None and out_scores is not None
        output_results = _pack(out_attrs, out_embeds, out_scores)
    return input_results, output_results


def _residual_period_component_labels(
    n_layers: int, periods: Sequence[str]
) -> list[str]:
    labels: list[str] = []
    n_display = int(n_layers) + 1
    if len(periods) == 1:
        return residual_stream_labels(n_layers)
    for display in range(n_display):
        for period in periods:
            labels.append(f"L{display}_{period}")
    return labels


def _residual_flatten_period_values(
    period_vectors: Sequence[np.ndarray],
) -> np.ndarray:
    """Layer-major, period-minor flatten of [n_display] vectors."""
    n_display = int(period_vectors[0].shape[0])
    out = np.zeros(n_display * len(period_vectors), dtype=np.float32)
    idx = 0
    for display in range(n_display):
        for vec in period_vectors:
            out[idx] = float(vec[display])
            idx += 1
    return out


def _save_residual_numeric_h5(
    path: Path,
    period_results: dict[str, ExperimentResult],
    *,
    contrast_name: str,
) -> None:
    first = next(iter(period_results.values()))
    with h5py.File(path, "w") as f:
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("component_labels", data=np.asarray(first.labels, dtype=dt))
        f.attrs["contrast_name"] = contrast_name
        f.attrs["granularity"] = RESIDUAL_STREAM_GRANULARITY
        f.attrs["position"] = ",".join(period_results.keys())
        for period, result in period_results.items():
            g = f.create_group(period)
            g.create_dataset("attributions", data=result.attributions)
            g.create_dataset("embed_attributions", data=result.embed_attributions)
            g.create_dataset("true_scores", data=result.true_scores)
            g.create_dataset(
                "example_ids", data=np.asarray(result.example_ids, dtype=dt)
            )


def _write_residual_experiment_outputs(
    exp_dir: Path,
    period_results: dict[str, ExperimentResult],
    examples_by_id: dict[str, GradExample],
    individual_ids: Sequence[str],
    *,
    contrast_name: str,
    col_labels: Sequence[str],
    n_layers: int,
    bar_chart_top_k: int,
) -> None:
    """Write residual-stream mean/example heatmaps (1 col linguistic, 2 col numeric)."""
    if not period_results:
        raise ValueError("period_results is empty.")
    periods = list(period_results.keys())
    if len(periods) != len(col_labels):
        raise ValueError(
            f"col_labels {list(col_labels)} length != n_periods={len(periods)}."
        )
    n_display = int(n_layers) + 1
    row_labels = [str(d) for d in range(n_display)]
    ranked_labels = _residual_period_component_labels(n_layers, periods)
    first = next(iter(period_results.values()))
    for result in period_results.values():
        if result.labels != first.labels:
            raise ValueError(
                f"Residual label mismatch for {contrast_name}: "
                f"{result.labels[:3]} vs {first.labels[:3]}"
            )
        if result.attributions.shape[1] != n_display:
            raise ValueError(
                f"{contrast_name} attributions width {result.attributions.shape[1]} "
                f"!= n_display={n_display}."
            )

    exp_dir.mkdir(parents=True, exist_ok=True)
    if len(period_results) == 1:
        save_attributions_h5(exp_dir / "attributions.h5", first)
    else:
        _save_residual_numeric_h5(
            exp_dir / "attributions.h5",
            period_results,
            contrast_name=contrast_name,
        )

    period_means = [
        np.mean(result.attributions, axis=0).astype(np.float32)
        for result in period_results.values()
    ]
    period_sems = [
        (
            np.std(result.attributions, axis=0, ddof=1)
            / np.sqrt(result.attributions.shape[0])
        ).astype(np.float32)
        if result.attributions.shape[0] > 1
        else np.zeros(n_display, dtype=np.float32)
        for result in period_results.values()
    ]
    mean_mat = np.stack(period_means, axis=1)
    flat_means = _residual_flatten_period_values(period_means)
    flat_sems = _residual_flatten_period_values(period_sems)
    stats = [
        ComponentStats(
            label=lab,
            mean=float(flat_means[i]),
            sem=float(flat_sems[i]),
            median=float(flat_means[i]),
            top10_fraction=0.0,
            sign_consistency=1.0,
        )
        for i, lab in enumerate(ranked_labels)
    ]
    # Prefer real per-example stats when a single cohort backs the heatmap.
    if len(period_results) == 1:
        stats = aggregate_component_stats(first.attributions, first.labels)
        ranked_labels = list(first.labels)
        flat_means = np.array([s.mean for s in stats], dtype=np.float32)
        flat_sems = np.array([s.sem for s in stats], dtype=np.float32)
    ranked = rank_by_mean(stats)

    if len(period_results) == 1:
        write_distribution_plot(
            exp_dir / "distribution_topk.png",
            first.attributions,
            first.labels,
            ranked[: max(bar_chart_top_k, 1)],
            top_k=bar_chart_top_k,
            title=(
                f"attr-patch residual {first.position} / {contrast_name}: "
                f"top-{bar_chart_top_k} by |mean|, ordered by mean"
            ),
        )

    mean_dir = exp_dir / "mean"
    mean_dir.mkdir(parents=True, exist_ok=True)
    n_note = ", ".join(
        f"{period} n={len(result.example_ids)}"
        for period, result in period_results.items()
    )
    write_diverging_heatmap(
        mean_dir / "heatmap.png",
        mean_mat,
        row_labels=row_labels,
        col_labels=list(col_labels),
        title=(
            f"Mean attr-patch residual — {contrast_name} ({n_note}; "
            "± normalised separately)"
        ),
        rounding_dp=ANNOTATION_ROUNDING_DP,
    )
    embed_parts = [
        float(np.mean(result.embed_attributions))
        for result in period_results.values()
        if result.embed_attributions.size
    ]
    embed_mean = float(np.mean(embed_parts)) if embed_parts else 0.0
    write_ranked_mean_txt(
        mean_dir / "ranked.txt", ranked, embed_mean=embed_mean, embed_sem=0.0
    )
    write_bar_chart(
        mean_dir / "bar_chart.png",
        [s.label for s in ranked],
        [s.mean for s in ranked],
        title=f"Mean attr-patch residual — {contrast_name}",
        top_k=bar_chart_top_k,
        errors=[s.sem for s in ranked],
    )

    id_to_row = {
        period: {eid: i for i, eid in enumerate(result.example_ids)}
        for period, result in period_results.items()
    }
    for eid in individual_ids:
        missing = [p for p, lookup in id_to_row.items() if eid not in lookup]
        if missing:
            raise ValueError(
                f"Individual example {eid} missing from residual periods {missing}."
            )
        ex = examples_by_id[eid]
        period_vals = [
            result.attributions[id_to_row[period][eid]]
            for period, result in period_results.items()
        ]
        mat = np.stack(period_vals, axis=1)
        ex_dir = exp_dir / f"example_{eid}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        write_diverging_heatmap(
            ex_dir / "heatmap.png",
            mat,
            row_labels=row_labels,
            col_labels=list(col_labels),
            title=(
                f"Example {eid} attr-patch residual — {contrast_name} "
                "(normalised to this example's own min/max; ± separately)"
            ),
            rounding_dp=ANNOTATION_ROUNDING_DP,
        )
        flat_vals = _residual_flatten_period_values(period_vals)
        ranked_pairs = rank_by_value(ranked_labels, flat_vals)
        first_period = next(iter(period_results))
        first_result = period_results[first_period]
        true_score = float(first_result.true_scores[id_to_row[first_period][eid]])
        if len(period_results) > 1:
            score_bits = [
                f"{period}={float(result.true_scores[id_to_row[period][eid]]):.6g}"
                for period, result in period_results.items()
            ]
            write_ranked_example_txt(
                ex_dir / "ranked.txt",
                ranked_pairs,
                example_id=eid,
                response=ex.response,
                confidence=ex.verbalised_confidence,
                embed_attr=0.0,
                true_score=true_score,
            )
            extra = ex_dir / "ranked.txt"
            text = extra.read_text(encoding="utf-8")
            extra.write_text(
                text.replace(
                    f"true_contrast_score={true_score:.6g}",
                    "true_contrast_scores " + " ".join(score_bits),
                    1,
                ),
                encoding="utf-8",
            )
        else:
            write_ranked_example_txt(
                ex_dir / "ranked.txt",
                ranked_pairs,
                example_id=eid,
                response=ex.response,
                confidence=ex.verbalised_confidence,
                embed_attr=float(
                    first_result.embed_attributions[id_to_row[first_period][eid]]
                ),
                true_score=true_score,
            )
        write_bar_chart(
            ex_dir / "bar_chart.png",
            [p[0] for p in ranked_pairs],
            [p[1] for p in ranked_pairs],
            title=f"Example {eid} attr-patch residual — {contrast_name}",
            top_k=bar_chart_top_k,
            errors=None,
        )


def _write_residual_high_minus_low(
    exp_dir: Path,
    high_period_results: dict[str, ExperimentResult],
    low_period_results: dict[str, ExperimentResult],
    *,
    contrast_name: str,
    col_labels: Sequence[str],
    n_layers: int,
    bar_chart_top_k: int,
) -> None:
    if set(high_period_results) != set(low_period_results):
        raise ValueError(
            f"High/low residual periods differ for {contrast_name}: "
            f"{sorted(high_period_results)} vs {sorted(low_period_results)}."
        )
    periods = list(high_period_results.keys())
    n_display = int(n_layers) + 1
    row_labels = [str(d) for d in range(n_display)]
    ranked_labels = _residual_period_component_labels(n_layers, periods)
    high_means = []
    low_means = []
    high_sems = []
    low_sems = []
    for period in periods:
        high = high_period_results[period]
        low = low_period_results[period]
        if high.labels != low.labels:
            raise ValueError(
                f"High/low label mismatch for residual {contrast_name}/{period}."
            )
        h_stats = aggregate_component_stats(high.attributions, high.labels)
        l_stats = aggregate_component_stats(low.attributions, low.labels)
        high_means.append(np.array([s.mean for s in h_stats], dtype=np.float32))
        low_means.append(np.array([s.mean for s in l_stats], dtype=np.float32))
        high_sems.append(np.array([s.sem for s in h_stats], dtype=np.float32))
        low_sems.append(np.array([s.sem for s in l_stats], dtype=np.float32))
    diff_mat = np.stack(high_means, axis=1) - np.stack(low_means, axis=1)
    flat_high = _residual_flatten_period_values(high_means)
    flat_low = _residual_flatten_period_values(low_means)
    flat_high_sem = _residual_flatten_period_values(high_sems)
    flat_low_sem = _residual_flatten_period_values(low_sems)
    if len(periods) == 1:
        ranked_labels = list(next(iter(high_period_results.values())).labels)
        flat_high = high_means[0]
        flat_low = low_means[0]
        flat_high_sem = high_sems[0]
        flat_low_sem = low_sems[0]
    ranked = high_minus_low_rows(
        ranked_labels, flat_high, flat_low, flat_high_sem, flat_low_sem
    )
    n_high_note = ", ".join(
        f"{p} n_high={len(high_period_results[p].example_ids)}" for p in periods
    )
    n_low_note = ", ".join(
        f"{p} n_low={len(low_period_results[p].example_ids)}" for p in periods
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    mean_dir = exp_dir / "mean"
    mean_dir.mkdir(parents=True, exist_ok=True)
    write_ranked_high_minus_low_txt(
        mean_dir / "ranked.txt",
        ranked,
        embed_diff=0.0,
        embed_high=0.0,
        embed_low=0.0,
        n_high=len(next(iter(high_period_results.values())).example_ids),
        n_low=len(next(iter(low_period_results.values())).example_ids),
    )
    extra = mean_dir / "ranked.txt"
    text = extra.read_text(encoding="utf-8")
    extra.write_text(
        text.replace(
            f"n_high={len(next(iter(high_period_results.values())).example_ids)}  "
            f"n_low={len(next(iter(low_period_results.values())).example_ids)}",
            f"{n_high_note}; {n_low_note}",
            1,
        ),
        encoding="utf-8",
    )
    write_diverging_heatmap(
        mean_dir / "heatmap.png",
        diff_mat.astype(np.float32),
        row_labels=row_labels,
        col_labels=list(col_labels),
        title=(
            f"High−low attr-patch residual — {contrast_name} "
            f"({n_high_note}; {n_low_note}; ± normalised separately)"
        ),
        rounding_dp=ANNOTATION_ROUNDING_DP,
    )
    write_bar_chart(
        mean_dir / "bar_chart.png",
        [r.label for r in ranked],
        [r.diff for r in ranked],
        title=f"High−low attr-patch residual — {contrast_name}",
        top_k=bar_chart_top_k,
        errors=[r.diff_sem for r in ranked],
    )


def _write_residual_cohort_dirs(
    *,
    run_root: Path,
    results: dict[tuple[str, str], ExperimentResult],
    examples_by_id: dict[str, GradExample],
    individual_ids: Sequence[str],
    dirname_suffix: str,
    n_layers: int,
    bar_chart_top_k: int,
    linguistic: bool,
) -> None:
    names: list[str] = []
    for _position, name in results:
        if name not in names:
            names.append(name)
    for name in names:
        period_results = {
            pos: result for (pos, n), result in results.items() if n == name
        }
        if linguistic:
            col_labels = ["residual"]
        else:
            missing = [p for p in NUMERIC_RESIDUAL_PERIODS if p not in period_results]
            if missing:
                raise ValueError(
                    f"Residual numeric contrast {name!r} missing periods {missing}."
                )
            period_results = {
                period: period_results[period] for period in NUMERIC_RESIDUAL_PERIODS
            }
            col_labels = list(NUMERIC_RESIDUAL_PERIODS)
        leaf = f"{name}{dirname_suffix}"
        exp_dir = run_root / leaf
        logging.info(
            "Writing residual %s (periods=%s)",
            leaf,
            ",".join(
                f"{p}:{len(r.example_ids)}" for p, r in period_results.items()
            ),
        )
        allowed: set[str] | None = None
        for result in period_results.values():
            ids = set(result.example_ids)
            allowed = ids if allowed is None else allowed & ids
        allowed_ids = allowed or set()
        contrast_ids = [eid for eid in individual_ids if eid in allowed_ids]
        if len(contrast_ids) < min(len(individual_ids), len(allowed_ids)):
            for eid in next(iter(period_results.values())).example_ids:
                if eid in allowed_ids and eid not in contrast_ids:
                    contrast_ids.append(eid)
                if len(contrast_ids) >= len(individual_ids):
                    break
        _write_residual_experiment_outputs(
            exp_dir,
            period_results,
            examples_by_id,
            contrast_ids[: len(individual_ids)],
            contrast_name=name,
            col_labels=col_labels,
            n_layers=n_layers,
            bar_chart_top_k=bar_chart_top_k,
        )


def _write_residual_high_minus_low_dirs(
    *,
    run_root: Path,
    high_results: dict[tuple[str, str], ExperimentResult],
    low_results: dict[tuple[str, str], ExperimentResult],
    n_layers: int,
    bar_chart_top_k: int,
    linguistic: bool,
) -> None:
    names: list[str] = []
    for _position, name in high_results:
        if name not in names:
            names.append(name)
    for name in names:
        high_period = {
            pos: result for (pos, n), result in high_results.items() if n == name
        }
        low_period = {
            pos: result for (pos, n), result in low_results.items() if n == name
        }
        if linguistic:
            col_labels = ["residual"]
        else:
            high_period = {
                period: high_period[period] for period in NUMERIC_RESIDUAL_PERIODS
            }
            low_period = {
                period: low_period[period] for period in NUMERIC_RESIDUAL_PERIODS
            }
            col_labels = list(NUMERIC_RESIDUAL_PERIODS)
        leaf = f"{name}__high_minus_low"
        logging.info("Writing residual high-minus-low ranking %s", leaf)
        _write_residual_high_minus_low(
            run_root / leaf,
            high_period,
            low_period,
            contrast_name=name,
            col_labels=col_labels,
            n_layers=n_layers,
            bar_chart_top_k=bar_chart_top_k,
        )


# ---------------------------------------------------------------------------
# Run scaffolding
# ---------------------------------------------------------------------------


def resolve_run_root(cli_output_dir: Optional[str]) -> Path:
    if cli_output_dir:
        root = Path(cli_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
    base = SCRIPT_DIR / "results"
    base.mkdir(parents=True, exist_ok=True)
    existing = [d for d in os.listdir(base) if (base / d).is_dir() and d.isdigit()]
    run_idx = max((int(d) for d in existing), default=0) + 1
    root = base / str(run_idx)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_readme(path: Path) -> None:
    path.write_text(
        "# Direction-gradient attribution (attribution patching) results\n\n"
        "Each component score is the first-order attribution-patching estimate of "
        "adding the mass-mean direction (high_conf mean − low_conf mean of cached "
        "H5 activations) at that site:\n\n"
        "`⟨∂ logit_diff / ∂ x_site[T], d_mass_mean⟩`\n\n"
        "The default tree at the run root is the **input-site** pass. Attention "
        "sites are per-head residual copies (`hook_q_input` / `hook_k_input` / "
        "`hook_v_input`; direction is H5 `res[L]`, resid-pre of block L). MLP is "
        "the cloned resid-mid branch (`hook_mlp_in`; direction is "
        "`res[L] + attn[L]`), so the skip is not part of the MLP site. Fine head "
        "scores sum the three input dots; under GQA the K/V term is shared across "
        "query heads in the group. "
        "Coarse `L*_attn` is the sum of those fine head scores. Embedding "
        "contribution is unused (written as 0).\n\n"
        "With `--attribute_component_outputs` (default), a second independent "
        "`run_with_hooks` pass from the same prefixes and logit differences is written "
        "under `component_output/` with the same `pre_period/` / `post_period/` (or "
        "linguistic) layout. Attention is the residual **addend** at `hook_attn_out` "
        "(after `ln1_post` / HF `post_attention_layernorm` on sandwich Gemma-3; "
        "`o_proj` out on Mistral/Qwen). Fine heads are rebuilt from `hook_z` through "
        "`W_O` with a shared post-attn RMS scale on Gemma, matching "
        "`gradient_based_attr`. Each fine head is scored with its own mass-mean "
        "direction `d_h` (high−low mean of H5 concat through `W_O`; sandwich Gemma "
        "applies that example's shared `ln1_post` scale before averaging). Coarse "
        "`L*_attn` is `⟨∇_addend, d_attn⟩`; fine head `h` is `⟨∇_post_h, d_h⟩`. "
        "MLP is `hook_mlp_out` with H5 `mlp[L]`. "
        "On sandwich Gemma-3, TransformerLens fires `hook_mlp_out` after "
        "`ln2_post` (HF `post_feedforward_layernorm`), matching "
        "`subblock_tokenwise_mean_ablation`. These two passes do not share an "
        "autograd graph: the output-site rebuild detaches `hook_z` last-pos, which "
        "would otherwise leave `hook_q/k/v_input` with no path to ΔL.\n\n"
        "Unlike Direct Logit Attribution this includes *indirect* paths through "
        "later layers. Completeness (sum of components = logit_diff) is not expected.\n\n"
        "Numeric Probability: experiments nest under `pre_period/` and `post_period/` "
        "as `{position}/{contrast}__{granularity}`. Post-period experiments (and the "
        "post-period mass-mean direction) drop samples whose verbalised_confidence "
        "is 1.0 and refill `--max_examples_for_mean` independently.\n\n"
        "Linguistic Confidence (`--linguistic_confidence_prompt`, Mistral only) writes "
        "experiments flat as `{contrast}__{granularity}`. `confidence_token_vs_rest` is "
        "attributed at the first phrase token (`expected_confidence_tokens - 1`). "
        "`L_vs_Un` uses span index 6 and keeps only examples whose decoded token there "
        "is `L` or `Un`. Other models raise.\n\n"
        "When `--confidence_split` is enabled (default), high- and low-confidence "
        "cohorts are written as `...__high_conf/` and `...__low_conf/`. "
        "`...__high_minus_low/` ranks components by signed "
        "(high_conf mean − low_conf mean) of these attribution scores.\n\n"
        "With `--residual_stream`, attribution patching is on the residual stream "
        "at display layers instead of Q/K/V inputs and MLP: display 0 is "
        "`blocks.0.hook_resid_pre` (embedding; H5 `res[0]`) and display `k>=1` is "
        "`blocks.{k-1}.hook_resid_post` (H5 `res[k]`). There is no coarse/fine split "
        "and no `component_output/` tree. Numeric residual experiments are written "
        "flat as `{contrast}/` (`digit_vs_rest` and `high_vs_low`; pre-period "
        "`high_vs_low` is 1 vs 0) with a two-column heatmap (pre_period, post_period) "
        "and layer rows `0 … n_layers`. The post-period column still drops "
        "verbalised_confidence 1.0. Linguistic residual heatmaps are one column "
        "(`residual`) under the same flat `{contrast}/` names.\n\n"
        "`--rerun_autoregressive` (default) greedy-decodes a new Guess:/Probability: "
        "completion from each H5 question for the live attribution pass and keeps the "
        "sampled token ids for prefix reconstruction (Mistral ``\\n`` is not 1:1 under "
        "``encode``). The H5 is then only a catalog of example ids, questions, and "
        "original verbalised_confidence labels for that pass. Mass-mean directions still "
        "come from cached H5 activations. `--no-rerun_autoregressive` reconstructs "
        "the prefix from stored `decoded_tokens` instead.\n",
        encoding="utf-8",
    )


def write_config_txt(
    path: Path,
    *,
    args: argparse.Namespace,
    input_h5_resolved: str,
    finished_at: str,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    d_model: int,
    high_digits: set[str] | None,
    low_digits: set[str] | None,
    first_token_ids: dict[str, tuple[int, str]] | None,
    l_vs_un_token_ids: dict[str, tuple[int, str]] | None,
    l_vs_un_skipped_not_l_or_un: int | None,
    actual_n: dict[str, int],
    direction_counts: dict[str, tuple[int, int]],
    span_index_by_position: dict[str, int],
) -> None:
    lines = [
        "Direction Gradient Attribution (Attribution Patching) Config",
        "===========================================================",
        "",
        f"finished_at={finished_at}",
        f"script={Path(__file__).resolve()}",
        f"input_h5={input_h5_resolved}",
        f"output_dir={args.output_dir}",
        f"device={args.device}",
        f"dtype={args.dtype}",
        f"model_name={args.model_name}",
        f"n_layers={n_layers}",
        f"n_heads={n_heads}",
        f"n_kv_heads={n_kv_heads}",
        f"d_head={d_head}",
        f"d_model={d_model}",
        "method=attribution_patching",
        "direction_definition=high_mean_minus_low_mean",
    ]
    if args.residual_stream:
        lines += [
            "residual_stream=True",
            "sites=display residual stream "
            "(0=blocks.0.hook_resid_pre; k>=1=blocks.{k-1}.hook_resid_post)",
            "resid_direction=h5_res[k] (display k; res[0]=embedding, res[k]=resid-post of block k-1)",
            "attribute_component_outputs=False",
            "output_pass=skipped",
            f"n_display_layers={n_layers + 1}",
        ]
    else:
        lines += [
            "residual_stream=False",
            "sites=hook_q_input,hook_k_input,hook_v_input,hook_mlp_in",
            "attn_direction=h5_res[L] (resid-pre of block L)",
            "mlp_direction=h5_res[L]+attn[L] (resid-mid, cloned onto hook_mlp_in)",
            "fine_head_score=q_input_dot + k_input_dot + v_input_dot (GQA shares K/V within query group)",
            f"attribute_component_outputs={args.attribute_component_outputs}",
            "output_pass=second_independent_forward",
            "output_sites=hook_attn_out,hook_mlp_out",
            "output_attn_direction=h5_attn[L] (hook_attn_out addend; after ln1_post / post_attention_layernorm on sandwich Gemma)",
            "output_mlp_direction=h5_mlp[L] (hook_mlp_out; after ln2_post / post_feedforward_layernorm on sandwich Gemma)",
            "output_fine_head_score=<grad_post_h, d_h>; d_h = high_mean(post_h)-low_mean(post_h) from H5 concat (o fallback) through W_O; sandwich Gemma applies per-example shared ln1_post before averaging",
            f"output_results_subdir={COMPONENT_OUTPUT_SUBDIR}",
        ]
    lines += [
        "",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_confidence_tokens={args.expected_confidence_tokens}",
        f"linguistic_confidence_prompt={args.linguistic_confidence_prompt}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"confidence_split={args.confidence_split}",
        f"rerun_autoregressive={args.rerun_autoregressive}",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"granularity={args.granularity}",
        f"max_examples_for_mean={args.max_examples_for_mean}",
        f"n_individual_examples={args.n_individual_examples}",
        f"bar_chart_top_k={args.bar_chart_top_k}",
        f"high_digits={None if high_digits is None else sorted(high_digits)}",
        f"low_digits={None if low_digits is None else sorted(low_digits)}",
        "",
        "[SpanIndices]",
    ]
    for pos, idx in span_index_by_position.items():
        lines.append(f"{pos}={idx}")
    lines += ["", "[MassMeanDirectionCounts]"]
    for pos, (n_high, n_low) in direction_counts.items():
        lines.append(f"{pos}_n_high={n_high}")
        lines.append(f"{pos}_n_low={n_low}")
    lines.append("")
    for k, v in actual_n.items():
        lines.append(f"{k}={v}")
    if args.linguistic_confidence_prompt:
        lines += ["", "[LinguisticFirstTokens]"]
        if first_token_ids is None:
            raise ValueError("linguistic run is missing resolved first_token_ids.")
        for piece, (tid, form) in first_token_ids.items():
            lines.append(f"piece={piece!r} token_id={tid} resolved_form={form!r}")
        lines += ["", "[LinguisticLvsUn]"]
        if l_vs_un_token_ids is None:
            raise ValueError("linguistic run is missing resolved L_vs_Un token ids.")
        for piece, (tid, form) in l_vs_un_token_ids.items():
            lines.append(f"piece={piece!r} token_id={tid} resolved_form={form!r}")
        lines.append(f"span_index={L_VS_UN_SPAN_INDEX}")
        skipped_n = (
            0 if l_vs_un_skipped_not_l_or_un is None else int(l_vs_un_skipped_not_l_or_un)
        )
        lines.append(f"l_vs_un_skipped_not_l_or_un={skipped_n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Attribution patching of mass-mean (high−low) residual-input "
            "directions (hook_q/k/v_input and hook_mlp_in), and optionally "
            "component-output directions (hook_attn_out addend and hook_mlp_out), onto "
            "verbalised-confidence logit differences. With --residual_stream, patch "
            "the residual stream at display layers instead."
        )
    )
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODEL_NAMES),
    )
    p.add_argument("--input_h5", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP),
    )
    p.add_argument("--expected_probability_tokens", type=int, default=5)
    p.add_argument(
        "--expected_confidence_tokens",
        type=int,
        default=5,
        help=(
            "When --linguistic_confidence_prompt, unextended Confidence: span length "
            "(prefix plus first phrase token). Stored spans may be longer."
        ),
    )
    p.add_argument(
        "--linguistic_confidence_prompt",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, attribute linguistic Confidence: (Mistral only). "
            "confidence_token_vs_rest at the first phrase token, plus L_vs_Un at "
            "span index 6 (kept only when that decoded token is L or Un)."
        ),
    )
    p.add_argument("--low_conf_threshold", type=float, default=0.1)
    p.add_argument("--high_conf_threshold", type=float, default=0.9)
    p.add_argument(
        "--confidence_split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--rerun_autoregressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), greedy-generate Guess:/Probability: from each H5 "
            "question and attribute that prefix using the sampled token ids. The H5 "
            "supplies example ids, questions, and verbalised_confidence cohort labels "
            "only. Mass-mean directions still come from cached H5 activations. "
            "If false, reconstruct from stored decoded_tokens."
        ),
    )
    p.add_argument(
        "--attribute_component_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), also score component outputs from the same "
            "prefixes and logit differences in a second independent forward: "
            "attention addend at hook_attn_out (GBA-style per-head rebuild "
            "after ln1_post on sandwich Gemma-3; H5 attn) and hook_mlp_out "
            "(H5 mlp; after ln2_post on sandwich Gemma-3). Fine heads use "
            "per-head mass-mean directions from H5 concat (o fallback). "
            "Written under component_output/. If false, input-site pass only."
        ),
    )
    p.add_argument(
        "--residual_stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, attribution-patch the residual stream at display layers "
            "(0=blocks.0.hook_resid_pre embedding; k>=1=blocks.{k-1}.hook_resid_post) "
            "with mass-mean directions from H5 res[k]. Layer-level only (no fine "
            "granularity, no component_output). Numeric runs write digit_vs_rest and "
            "high_vs_low (pre-period high_vs_low is 1 vs 0) with a two-column "
            "pre/post-period heatmap; linguistic writes one column."
        ),
    )
    p.add_argument(
        "--model_max_new_tokens",
        type=int,
        default=50,
        help="Max new tokens when --rerun_autoregressive (default: 50).",
    )
    p.add_argument(
        "--granularity",
        nargs="+",
        choices=["coarse", "fine"],
        default=["coarse", "fine"],
    )
    p.add_argument("--n_individual_examples", type=int, default=3)
    p.add_argument("--max_examples_for_mean", type=int, default=50)
    p.add_argument("--individual_example_indices", type=str, default=None)
    p.add_argument("--bar_chart_top_k", type=int, default=25)
    p.add_argument("--output_dir", type=str, default=None)
    return p


def _write_cohort_dirs(
    *,
    run_root: Path,
    results: dict[tuple[str, str, str], ExperimentResult],
    examples_by_id: dict[str, GradExample],
    individual_ids: Sequence[str],
    dirname_suffix: str,
    n_layers: int,
    n_heads: int,
    bar_chart_top_k: int,
    linguistic: bool,
) -> None:
    for (position, name, gran), result in results.items():
        leaf = f"{name}__{gran}{dirname_suffix}"
        if linguistic:
            exp_dir = run_root / leaf
            loc = leaf
        else:
            exp_dir = run_root / position / leaf
            loc = f"{position}/{leaf}"
        logging.info("Writing %s (n=%d)", loc, len(result.example_ids))
        contrast_ids = [eid for eid in individual_ids if eid in set(result.example_ids)]
        if len(contrast_ids) < min(len(individual_ids), len(result.example_ids)):
            for eid in result.example_ids:
                if eid not in contrast_ids:
                    contrast_ids.append(eid)
                if len(contrast_ids) >= len(individual_ids):
                    break
        write_experiment_outputs(
            exp_dir,
            result,
            examples_by_id,  # type: ignore[arg-type]
            contrast_ids[: len(individual_ids)],
            n_layers=n_layers,
            n_heads=n_heads,
            bar_chart_top_k=bar_chart_top_k,
            rounding_dp=ANNOTATION_ROUNDING_DP,
            method_label="attr-patch",
        )


def _write_high_minus_low_dirs(
    *,
    run_root: Path,
    high_results: dict[tuple[str, str, str], ExperimentResult],
    low_results: dict[tuple[str, str, str], ExperimentResult],
    n_layers: int,
    n_heads: int,
    bar_chart_top_k: int,
    linguistic: bool,
) -> None:
    for key, high_result in high_results.items():
        low_result = low_results.get(key)
        if low_result is None:
            raise ValueError(f"High-conf result {key} has no matching low-conf run.")
        position, contrast_name, gran = key
        leaf = f"{contrast_name}__{gran}__high_minus_low"
        if linguistic:
            exp_dir = run_root / leaf
            loc = leaf
        else:
            exp_dir = run_root / position / leaf
            loc = f"{position}/{leaf}"
        logging.info("Writing high-minus-low ranking %s", loc)
        write_high_minus_low_from_results(
            exp_dir,
            high_result,
            low_result,
            n_layers=n_layers,
            n_heads=n_heads,
            bar_chart_top_k=bar_chart_top_k,
            rounding_dp=ANNOTATION_ROUNDING_DP,
            method_label="attr-patch",
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_examples_for_mean <= args.n_individual_examples:
        raise ValueError(
            f"--max_examples_for_mean ({args.max_examples_for_mean}) must be strictly "
            f"greater than --n_individual_examples ({args.n_individual_examples})."
        )
    if args.n_individual_examples < 1:
        raise ValueError("--n_individual_examples must be >= 1")
    if args.bar_chart_top_k < 1:
        raise ValueError("--bar_chart_top_k must be >= 1")
    if args.model_max_new_tokens < 1:
        raise ValueError("--model_max_new_tokens must be >= 1")
    validate_confidence_thresholds(args.high_conf_threshold, args.low_conf_threshold)
    if args.linguistic_confidence_prompt:
        if args.expected_confidence_tokens < 1:
            raise ValueError("--expected_confidence_tokens must be >= 1")
        if args.model_name != "mistralai/Mistral-7B-Instruct-v0.1":
            raise ValueError(
                "--linguistic_confidence_prompt is currently only implemented for "
                "mistralai/Mistral-7B-Instruct-v0.1 "
                f"(got {args.model_name!r})."
            )
        linguistic_first_tokens_for_model(args.model_name)

    gran_list: list[str] = []
    for g in args.granularity:
        if g not in gran_list:
            gran_list.append(g)
    residual_stream = bool(args.residual_stream)
    if residual_stream:
        if gran_list not in (["coarse", "fine"], ["coarse"]):
            logging.warning(
                "Ignoring --granularity %s; residual-stream mode is layer-level only.",
                gran_list,
            )
        gran_list = [RESIDUAL_STREAM_GRANULARITY]
        if args.attribute_component_outputs:
            logging.info(
                "residual_stream=True: skipping component-output pass "
                "(--attribute_component_outputs ignored)."
            )
    args.granularity = gran_list

    input_h5 = str(Path(args.input_h5).resolve())
    if not os.path.isfile(input_h5):
        raise FileNotFoundError(f"--input_h5 not found: {input_h5}")

    run_root = resolve_run_root(args.output_dir)
    attach_output_log(run_root)
    logging.info("Run directory: %s", run_root)
    write_readme(run_root / "README.md")

    configure_prefix_tokens_for_model(args.model_name)

    dtype = DTYPE_MAP[args.dtype]
    logging.info("Loading HookedTransformer %s dtype=%s device=%s", args.model_name, args.dtype, args.device)
    model = load_hooked_transformer(args.model_name, device=args.device, torch_dtype=dtype)
    freeze_model(model)
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("HookedTransformer has no tokenizer.")

    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    d_model = int(model.cfg.d_model)
    n_kv_heads = resolve_n_key_value_heads(model.cfg, n_heads)
    logging.info(
        "TL cfg: n_layers=%d n_heads=%d n_kv_heads=%d d_head=%d d_model=%d",
        n_layers,
        n_heads,
        n_kv_heads,
        d_head,
        d_model,
    )

    high_digits: set[str] | None = None
    low_digits: set[str] | None = None
    first_token_ids: dict[str, tuple[int, str]] | None = None
    l_vs_un_token_ids: dict[str, tuple[int, str]] | None = None
    linguistic = bool(args.linguistic_confidence_prompt)
    if linguistic:
        pieces = linguistic_first_tokens_for_model(args.model_name)
        first_token_ids = resolve_linguistic_first_token_ids(tokenizer, pieces)
        for piece, (tid, form) in first_token_ids.items():
            logging.info("Linguistic first token %r -> id=%s form=%r", piece, tid, form)
        l_id, l_form = resolve_single_piece_token_id(tokenizer, "L")
        un_id, un_form = resolve_single_piece_token_id(tokenizer, "Un")
        l_vs_un_token_ids = {"L": (l_id, l_form), "Un": (un_id, un_form)}
        logging.info(
            "L vs Un: L id=%s form=%r; Un id=%s form=%r", l_id, l_form, un_id, un_form
        )
        expected_span = args.expected_confidence_tokens
        prompt_prefix = CONFIDENCE_PROMPT_LINGUISTIC
        contrasts = build_linguistic_contrast_specs(
            [tid for tid, _form in first_token_ids.values()],
            l_token_id=l_id,
            un_token_id=un_id,
            first_value_index=args.expected_confidence_tokens - 1,
            l_vs_un_index=L_VS_UN_SPAN_INDEX,
        )
        fill_post_period_excl_one = False
        positions = sorted({c.position for c in contrasts})
        exclude_conf_one_positions: list[str] = []
        require_l_or_un_positions = ["mistral_l_vs_un"] if "mistral_l_vs_un" in positions else []
    else:
        digit_ids = resolve_all_digit_ids(tokenizer)
        high_digits, low_digits = high_low_digit_sets(
            args.high_conf_threshold, args.low_conf_threshold
        )
        logging.info(
            "High digits %s  Low digits %s", sorted(high_digits), sorted(low_digits)
        )
        expected_span = args.expected_probability_tokens + 2
        prompt_prefix = CONFIDENCE_PROMPT
        if residual_stream:
            contrasts = build_numeric_residual_contrasts(
                digit_ids, high_digits, low_digits
            )
        else:
            contrasts = build_numeric_contrasts(digit_ids, high_digits, low_digits)
        fill_post_period_excl_one = True
        positions = ["pre_period", "post_period"]
        exclude_conf_one_positions = ["post_period"]
        require_l_or_un_positions = []

    span_index_by_position = {
        pos: _direction_span_index(
            pos,
            linguistic=linguistic,
            expected_probability_tokens=args.expected_probability_tokens,
            expected_confidence_tokens=args.expected_confidence_tokens,
        )
        for pos in positions
    }
    logging.info("Span indices: %s", span_index_by_position)

    need_output_fine = (
        (not residual_stream)
        and bool(args.attribute_component_outputs)
        and "fine" in gran_list
    )
    head_write_specs = (
        extract_attn_head_write_specs(model) if need_output_fine else None
    )
    directions = compute_mass_mean_directions(
        input_h5,
        positions=positions,
        span_index_by_position=span_index_by_position,
        n_layers=n_layers,
        d_model=d_model,
        high_conf_threshold=args.high_conf_threshold,
        low_conf_threshold=args.low_conf_threshold,
        exclude_conf_one_positions=exclude_conf_one_positions,
        require_l_or_un_positions=require_l_or_un_positions,
        head_write_specs=head_write_specs,
        residual_stream=residual_stream,
    )
    save_directions_npz(run_root / "directions.npz", directions)
    logging.info("Wrote %s", run_root / "directions.npz")

    cohorts = load_grad_example_cohorts(
        input_h5,
        expected_span=expected_span,
        max_examples=args.max_examples_for_mean,
        high_conf_threshold=args.high_conf_threshold,
        low_conf_threshold=args.low_conf_threshold,
        fill_confidence_splits=args.confidence_split,
        fill_post_period_excl_one=fill_post_period_excl_one,
        linguistic_confidence_prompt=linguistic,
        require_decoded_tokens=not args.rerun_autoregressive,
    )
    if args.rerun_autoregressive:
        logging.info(
            "Rerunning autoregressive greedy decode (max_new_tokens=%d).",
            args.model_max_new_tokens,
        )

        def _generate_one(question: str) -> tuple[str, list[str], list[int]]:
            return greedy_generate(
                model=model,
                local_prompt=prompt_prefix + question,
                max_new_tokens=args.model_max_new_tokens,
                fwd_hooks=None,
                return_generated_ids=True,
            )

        fill_cohorts_from_generations(
            cohorts,
            _generate_one,
            expected_span=expected_span,
            linguistic_confidence_prompt=linguistic,
        )
        if not cohorts.all:
            raise ValueError(
                "No usable examples remain after autoregressive generation "
                "(no parseable probability span of expected length)."
            )
    logging.info("Using %d all-examples (cap=%d)", len(cohorts.all), args.max_examples_for_mean)
    if len(cohorts.all) <= args.n_individual_examples:
        raise ValueError(
            f"All-examples cohort has {len(cohorts.all)} usable examples, which must be "
            f"strictly greater than --n_individual_examples ({args.n_individual_examples})."
        )
    if fill_post_period_excl_one:
        if not cohorts.all_excl_one:
            raise ValueError(
                "No usable post-period examples after excluding verbalised_confidence == 1.0."
            )
        if len(cohorts.all_excl_one) <= args.n_individual_examples:
            raise ValueError(
                f"Post-period all-examples cohort has {len(cohorts.all_excl_one)} examples "
                "after excluding confidence 1.0, which must be strictly greater than "
                f"--n_individual_examples ({args.n_individual_examples})."
            )

    requested_ids = parse_individual_ids(args.individual_example_indices)
    usable_ids_all = [ex.example_id for ex in cohorts.all]
    if requested_ids is not None:
        missing = [i for i in requested_ids if i not in set(usable_ids_all)]
        if missing:
            raise ValueError(
                f"--individual_example_indices not in usable all-examples set: {missing}"
            )
        individual_ids_all = requested_ids
    else:
        individual_ids_all = usable_ids_all[: args.n_individual_examples]

    individual_ids_high: list[str] | None = None
    individual_ids_low: list[str] | None = None
    if args.confidence_split:
        if not cohorts.high or not cohorts.low:
            raise ValueError("confidence_split enabled but high or low cohort is empty.")
        if len(cohorts.high) <= args.n_individual_examples:
            raise ValueError(
                f"High-confidence cohort has {len(cohorts.high)} examples; "
                f"need > {args.n_individual_examples}."
            )
        if len(cohorts.low) <= args.n_individual_examples:
            raise ValueError(
                f"Low-confidence cohort has {len(cohorts.low)} examples; "
                f"need > {args.n_individual_examples}."
            )
        if fill_post_period_excl_one and (
            not cohorts.high_excl_one or not cohorts.low_excl_one
        ):
            raise ValueError(
                "Post-period high or low cohort is empty after excluding confidence 1.0."
            )
        individual_ids_high = [ex.example_id for ex in cohorts.high][
            : args.n_individual_examples
        ]
        individual_ids_low = [ex.example_id for ex in cohorts.low][
            : args.n_individual_examples
        ]
        logging.info(
            "Using %d high-conf / %d low-conf examples",
            len(cohorts.high),
            len(cohorts.low),
        )

    if residual_stream and fill_post_period_excl_one:
        usable_ids_post = [ex.example_id for ex in cohorts.all_excl_one]
        if requested_ids is not None:
            missing = [i for i in requested_ids if i not in set(usable_ids_post)]
            if missing:
                raise ValueError(
                    "--individual_example_indices not in post-period (conf≠1.0) "
                    "set required for residual-stream two-column heatmaps: "
                    f"{missing}"
                )
            individual_ids_all = requested_ids
        else:
            individual_ids_all = usable_ids_post[: args.n_individual_examples]
        if args.confidence_split:
            individual_ids_high = [ex.example_id for ex in cohorts.high_excl_one][
                : args.n_individual_examples
            ]
            individual_ids_low = [ex.example_id for ex in cohorts.low_excl_one][
                : args.n_individual_examples
            ]

    l_vs_un_skipped_ids: set[str] = set()
    if residual_stream:
        capture = ResidualStreamCapture(n_layers=n_layers, d_model=d_model)
        capture.register(model)
        logging.info(
            "Residual-stream pass: display 0=%s, k>=1=blocks.{k-1}.hook_resid_post "
            "(%d display layers).",
            hook_name_for_display_layer(0),
            n_layers + 1,
        )
    else:
        enable_split_input_hooks(model)
        capture = SiteInputCapture(
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            d_model=d_model,
            capture_outputs=bool(args.attribute_component_outputs),
            need_fine="fine" in gran_list,
        )
        capture.register(model)
        logging.info(
            "Input-site pass: hook_q/k/v_input and hook_mlp_in via run_with_hooks "
            "on all %d layers (use_split_qkv_input, use_hook_mlp_in).",
            n_layers,
        )
        if capture.capture_outputs:
            logging.info(
                "Output-site pass: second independent forward for hook_attn_out "
                "(GBA-style addend%s) and hook_mlp_out; results under %s/.",
                " with per-head rebuild" if capture.need_fine else "",
                COMPONENT_OUTPUT_SUBDIR,
            )

    output_run_root = run_root / COMPONENT_OUTPUT_SUBDIR

    def _run_and_write_residual(
        *,
        cohort_examples: list[GradExample],
        post_period_examples: list[GradExample] | None,
        individual_ids: Sequence[str],
        dirname_suffix: str,
        cohort_label: str,
    ) -> dict[tuple[str, str], ExperimentResult]:
        if not isinstance(capture, ResidualStreamCapture):
            raise RuntimeError("Residual-stream run requires ResidualStreamCapture.")
        combined: dict[tuple[str, str], ExperimentResult] = {}
        examples_by_id: dict[str, GradExample] = {}
        if linguistic:
            for contrast in contrasts:
                filtered = filter_examples_for_contrast(
                    cohort_examples, contrast, skipped_ids=l_vs_un_skipped_ids
                )
                if not filtered:
                    need = (
                        "any usable span"
                        if contrast.span_index is None
                        else f"span_len > {contrast.span_index}"
                    )
                    raise ValueError(
                        f"No {cohort_label} examples with {need} for contrast "
                        f"{contrast.name!r}."
                    )
                if len(filtered) <= args.n_individual_examples:
                    raise ValueError(
                        f"{cohort_label} cohort for contrast {contrast.name!r} has "
                        f"{len(filtered)} examples after span filtering, which must be "
                        f"strictly greater than --n_individual_examples "
                        f"({args.n_individual_examples})."
                    )
                part = run_residual_cohort_attributions(
                    examples=filtered,
                    model=model,
                    capture=capture,
                    contrasts=[contrast],
                    expected_span=expected_span,
                    desc=f"{cohort_label}/{contrast.name}",
                    prompt_prefix=prompt_prefix,
                    directions=directions,
                )
                combined.update(part)
                for ex in filtered:
                    examples_by_id[ex.example_id] = ex
        else:
            if post_period_examples is None:
                raise ValueError("Numeric residual runs require post-period examples.")
            pre_contrasts = [c for c in contrasts if c.position == "pre_period"]
            post_contrasts = [c for c in contrasts if c.position == "post_period"]
            pre_results = run_residual_cohort_attributions(
                examples=cohort_examples,
                model=model,
                capture=capture,
                contrasts=pre_contrasts,
                expected_span=expected_span,
                desc=f"{cohort_label}/pre_period",
                prompt_prefix=prompt_prefix,
                directions=directions,
            )
            post_results = run_residual_cohort_attributions(
                examples=post_period_examples,
                model=model,
                capture=capture,
                contrasts=post_contrasts,
                expected_span=expected_span,
                desc=f"{cohort_label}/post_period",
                prompt_prefix=prompt_prefix,
                directions=directions,
            )
            combined = {**pre_results, **post_results}
            examples_by_id = {
                ex.example_id: ex
                for ex in list(cohort_examples) + list(post_period_examples)
            }
        _write_residual_cohort_dirs(
            run_root=run_root,
            results=combined,
            examples_by_id=examples_by_id,
            individual_ids=individual_ids,
            dirname_suffix=dirname_suffix,
            n_layers=n_layers,
            bar_chart_top_k=args.bar_chart_top_k,
            linguistic=linguistic,
        )
        return combined

    def _run_and_write(
        *,
        cohort_examples: list[GradExample],
        post_period_examples: list[GradExample] | None,
        individual_ids: Sequence[str],
        dirname_suffix: str,
        cohort_label: str,
    ) -> tuple[
        dict[tuple[str, str, str], ExperimentResult],
        dict[tuple[str, str, str], ExperimentResult],
    ]:
        combined: dict[tuple[str, str, str], ExperimentResult] = {}
        combined_out: dict[tuple[str, str, str], ExperimentResult] = {}
        examples_by_id: dict[str, GradExample] = {}
        if linguistic:
            for contrast in contrasts:
                filtered = filter_examples_for_contrast(
                    cohort_examples, contrast, skipped_ids=l_vs_un_skipped_ids
                )
                if not filtered:
                    need = (
                        "any usable span"
                        if contrast.span_index is None
                        else f"span_len > {contrast.span_index}"
                    )
                    raise ValueError(
                        f"No {cohort_label} examples with {need} for contrast "
                        f"{contrast.name!r}."
                    )
                if len(filtered) <= args.n_individual_examples:
                    raise ValueError(
                        f"{cohort_label} cohort for contrast {contrast.name!r} has "
                        f"{len(filtered)} examples after span filtering, which must be "
                        f"strictly greater than --n_individual_examples "
                        f"({args.n_individual_examples})."
                    )
                part, part_out = run_cohort_attributions(
                    examples=filtered,
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    contrasts=[contrast],
                    expected_span=expected_span,
                    gran_list=gran_list,
                    desc=f"{cohort_label}/{contrast.name}",
                    prompt_prefix=prompt_prefix,
                    directions=directions,
                )
                combined.update(part)
                combined_out.update(part_out)
                for ex in filtered:
                    examples_by_id[ex.example_id] = ex
        else:
            if post_period_examples is None:
                raise ValueError("Numeric runs require post-period examples.")
            pre_contrasts = [c for c in contrasts if c.position == "pre_period"]
            post_contrasts = [c for c in contrasts if c.position == "post_period"]
            pre_results, pre_out = run_cohort_attributions(
                examples=cohort_examples,
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                contrasts=pre_contrasts,
                expected_span=expected_span,
                gran_list=gran_list,
                desc=f"{cohort_label}/pre_period",
                prompt_prefix=prompt_prefix,
                directions=directions,
            )
            post_results, post_out = run_cohort_attributions(
                examples=post_period_examples,
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                contrasts=post_contrasts,
                expected_span=expected_span,
                gran_list=gran_list,
                desc=f"{cohort_label}/post_period",
                prompt_prefix=prompt_prefix,
                directions=directions,
            )
            combined = {**pre_results, **post_results}
            combined_out = {**pre_out, **post_out}
            examples_by_id = {
                ex.example_id: ex
                for ex in list(cohort_examples) + list(post_period_examples)
            }
        write_kwargs = dict(
            examples_by_id=examples_by_id,
            individual_ids=individual_ids,
            dirname_suffix=dirname_suffix,
            n_layers=n_layers,
            n_heads=n_heads,
            bar_chart_top_k=args.bar_chart_top_k,
            linguistic=linguistic,
        )
        _write_cohort_dirs(run_root=run_root, results=combined, **write_kwargs)
        if combined_out:
            _write_cohort_dirs(
                run_root=output_run_root, results=combined_out, **write_kwargs
            )
        return combined, combined_out

    if residual_stream:
        _run_and_write_residual(
            cohort_examples=cohorts.all,
            post_period_examples=(
                cohorts.all_excl_one if fill_post_period_excl_one else None
            ),
            individual_ids=individual_ids_all,
            dirname_suffix="",
            cohort_label="all-examples",
        )
        if args.confidence_split:
            high_resid = _run_and_write_residual(
                cohort_examples=cohorts.high,
                post_period_examples=(
                    cohorts.high_excl_one if fill_post_period_excl_one else None
                ),
                individual_ids=individual_ids_high or [],
                dirname_suffix="__high_conf",
                cohort_label="high-confidence",
            )
            low_resid = _run_and_write_residual(
                cohort_examples=cohorts.low,
                post_period_examples=(
                    cohorts.low_excl_one if fill_post_period_excl_one else None
                ),
                individual_ids=individual_ids_low or [],
                dirname_suffix="__low_conf",
                cohort_label="low-confidence",
            )
            _write_residual_high_minus_low_dirs(
                run_root=run_root,
                high_results=high_resid,
                low_results=low_resid,
                n_layers=n_layers,
                bar_chart_top_k=args.bar_chart_top_k,
                linguistic=linguistic,
            )
    else:
        all_input, all_output = _run_and_write(
            cohort_examples=cohorts.all,
            post_period_examples=(
                cohorts.all_excl_one if fill_post_period_excl_one else None
            ),
            individual_ids=individual_ids_all,
            dirname_suffix="",
            cohort_label="all-examples",
        )
        del all_input, all_output

        if args.confidence_split:
            high_input, high_output = _run_and_write(
                cohort_examples=cohorts.high,
                post_period_examples=(
                    cohorts.high_excl_one if fill_post_period_excl_one else None
                ),
                individual_ids=individual_ids_high or [],
                dirname_suffix="__high_conf",
                cohort_label="high-confidence",
            )
            low_input, low_output = _run_and_write(
                cohort_examples=cohorts.low,
                post_period_examples=(
                    cohorts.low_excl_one if fill_post_period_excl_one else None
                ),
                individual_ids=individual_ids_low or [],
                dirname_suffix="__low_conf",
                cohort_label="low-confidence",
            )
            hml_kwargs = dict(
                n_layers=n_layers,
                n_heads=n_heads,
                bar_chart_top_k=args.bar_chart_top_k,
                linguistic=linguistic,
            )
            _write_high_minus_low_dirs(
                run_root=run_root,
                high_results=high_input,
                low_results=low_input,
                **hml_kwargs,
            )
            if high_output:
                _write_high_minus_low_dirs(
                    run_root=output_run_root,
                    high_results=high_output,
                    low_results=low_output,
                    **hml_kwargs,
                )

    if linguistic:
        logging.info(
            "L_vs_Un skipped %d examples whose span-index-6 token was not L or Un.",
            len(l_vs_un_skipped_ids),
        )

    capture.remove()
    finished_at = datetime.now().isoformat(timespec="seconds")
    actual_n: dict[str, int] = {
        "actual_n_examples_all": len(cohorts.all),
        "actual_n_examples_high": len(cohorts.high) if args.confidence_split else 0,
        "actual_n_examples_low": len(cohorts.low) if args.confidence_split else 0,
    }
    if fill_post_period_excl_one:
        actual_n.update(
            {
                "actual_n_examples_all_post_period": len(cohorts.all_excl_one),
                "actual_n_examples_high_post_period": (
                    len(cohorts.high_excl_one) if args.confidence_split else 0
                ),
                "actual_n_examples_low_post_period": (
                    len(cohorts.low_excl_one) if args.confidence_split else 0
                ),
            }
        )
    direction_counts = {
        pos: (d.n_high, d.n_low) for pos, d in directions.items()
    }
    write_config_txt(
        run_root / "config.txt",
        args=args,
        input_h5_resolved=input_h5,
        finished_at=finished_at,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_model=d_model,
        high_digits=high_digits,
        low_digits=low_digits,
        first_token_ids=first_token_ids,
        l_vs_un_token_ids=l_vs_un_token_ids,
        l_vs_un_skipped_not_l_or_un=(
            len(l_vs_un_skipped_ids) if linguistic else None
        ),
        actual_n=actual_n,
        direction_counts=direction_counts,
        span_index_by_position=span_index_by_position,
    )
    logging.info("Finished. Results at %s", run_root)


if __name__ == "__main__":
    main()
