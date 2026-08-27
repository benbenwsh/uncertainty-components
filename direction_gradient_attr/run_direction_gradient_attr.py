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
scale on Gemma) and allocates H5 ``attn[L]`` across those pieces. That rebuild
detaches ``hook_z``, so it cannot share a graph with ``hook_q/k/v_input``.
Coarse ``L*_attn`` is ``⟨∇_addend, d_attn⟩``, equal to the sum of fine head
scores. Those results nest under ``component_output/``.

Numeric runs use pre-period and post-period digits. Linguistic Confidence
(``--linguistic_confidence_prompt``, Mistral only) uses the variable-length
Confidence: span.
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
from componentwise_input_mean_ablation.run_componentwise_input_mean_ablation import (
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
from layerwise_mean_ablation.run_mean_ablation import greedy_generate, load_hooked_transformer
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
    attach_output_log,
    component_labels_coarse,
    component_labels_fine,
    high_low_digit_sets,
    linguistic_first_tokens_for_model,
    parse_individual_ids,
    resolve_all_digit_ids,
    resolve_linguistic_first_token_ids,
    resolve_single_piece_token_id,
    validate_confidence_thresholds,
    write_experiment_outputs,
    write_high_minus_low_from_results,
)

L_VS_UN_SPAN_INDEX = 6
COMPONENT_OUTPUT_SUBDIR = "component_output"
ATTN_OUT_RESUM_ATOL = 5e-2
ATTN_HEAD_ALLOC_EPS = 1e-8


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
) -> dict[str, SiteDirections]:
    """Stream all high/low H5 examples; d = high_mean - low_mean at each site."""
    exclude_one = set(exclude_conf_one_positions)
    require_lu = set(require_l_or_un_positions)
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
                    resid_pre, mlp_in, attn_out, mlp_out = _read_site_arrays_at_index(
                        r0,
                        span_index,
                        n_layers=n_layers,
                        d_model=d_model,
                    )
                except (TypeError, ValueError, OSError, KeyError) as exc:
                    skipped_bad += 1
                    logging.debug("Skip direction example %s pos=%s: %s", ex_id, pos, exc)
                    continue
                for group in groups:
                    bucket = sums[pos][group]
                    if not bucket:
                        bucket["resid_pre"] = np.array(resid_pre, dtype=np.float64, copy=True)
                        bucket["mlp"] = np.array(mlp_in, dtype=np.float64, copy=True)
                        bucket["attn_out"] = np.array(attn_out, dtype=np.float64, copy=True)
                        bucket["mlp_out"] = np.array(mlp_out, dtype=np.float64, copy=True)
                    else:
                        bucket["resid_pre"] += resid_pre
                        bucket["mlp"] += mlp_in
                        bucket["attn_out"] += attn_out
                        bucket["mlp_out"] += mlp_out
                    counts[pos][group] += 1

    out: dict[str, SiteDirections] = {}
    for pos in positions:
        n_high = int(counts[pos]["high"])
        n_low = int(counts[pos]["low"])
        if n_high < 1:
            raise ValueError(f"No high-confidence examples for mass-mean at {pos!r}.")
        if n_low < 1:
            raise ValueError(f"No low-confidence examples for mass-mean at {pos!r}.")
        high = sums[pos]["high"]
        low = sums[pos]["low"]
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
        out[pos] = SiteDirections(
            resid_pre=resid_pre,
            mlp=mlp,
            n_high=n_high,
            n_low=n_low,
            attn_out=attn_out,
            mlp_out=mlp_out,
        )
        logging.info(
            "Mass-mean direction %s: n_high=%d n_low=%d resid_pre=%s mlp=%s "
            "attn_out=%s mlp_out=%s",
            pos,
            n_high,
            n_low,
            tuple(resid_pre.shape),
            tuple(mlp.shape),
            tuple(attn_out.shape),
            tuple(mlp_out.shape),
        )
    if skipped_bad:
        logging.info("Direction stream skipped %d unreadable site rows.", skipped_bad)
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


def _shared_rmsnorm_scale(x: torch.Tensor, norm) -> torch.Tensor:
    """Per-token scale ``s`` such that ``norm(x) ≈ x * s`` (shared across heads)."""
    eps = float(getattr(norm, "eps", 1e-6))
    weight = getattr(norm, "w", None)
    if weight is None:
        weight = getattr(norm, "weight", None)
    if weight is None:
        raise ValueError("Post-attn RMSNorm has neither .w nor .weight.")
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    w = weight.float()
    with torch.no_grad():
        y = norm(x.detach()).float()
        pred_w = xf.detach() * (inv.detach() * w)
        pred_1pw = xf.detach() * (inv.detach() * (1.0 + w))
        use_one_plus = (pred_1pw - y).abs().mean() <= (pred_w - y).abs().mean()
    return inv * ((1.0 + w) if bool(use_one_plus) else w)


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


def _attn_head_allocated_dots(
    heads: list[torch.Tensor],
    total: torch.Tensor,
    d_attn: np.ndarray,
    *,
    layer: int,
    eps: float = ATTN_HEAD_ALLOC_EPS,
) -> np.ndarray:
    """Fine scores ``⟨∇_post_h, (post_h / sum_h post_h) ⊙ d_attn⟩`` so they sum to coarse."""
    del total
    d = torch.as_tensor(d_attn, dtype=torch.float32)
    if d.ndim != 1:
        raise ValueError(f"Expected 1D attn-out direction, got {tuple(d.shape)}.")
    total_act = heads[0].detach().float()
    if total_act.ndim == 3:
        total_act = total_act[:, -1]
    total_act = total_act[0]
    for ph in heads[1:]:
        post = ph.detach().float()
        if post.ndim == 3:
            post = post[:, -1]
        total_act = total_act + post[0]
    d = d.to(device=total_act.device)
    out = np.zeros(len(heads), dtype=np.float32)
    for h, ph in enumerate(heads):
        gh = _last_pos_grad(ph, name=f"attn_head_{h}", layer=layer)[0]
        post = ph.detach().float()
        if post.ndim == 3:
            post = post[:, -1]
        post = post[0]
        d_h = post / (total_act + eps) * d
        if gh.shape != d_h.shape:
            raise ValueError(
                f"Layer {layer} head {h} grad shape {tuple(gh.shape)} != "
                f"allocated direction {tuple(d_h.shape)}."
            )
        out[h] = float((gh * d_h).sum().detach().cpu().item())
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
        head_scores = _attn_head_allocated_dots(
            heads,
            capture.attn_out[layer],
            direction.attn_out[layer],
            layer=layer,
        )
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
        "`gradient_based_attr`. The mixed H5 `attn[L]` direction is allocated across "
        "those pieces so coarse `L*_attn` equals the sum of fine head scores "
        "(`⟨∇_addend, d_attn⟩`). MLP is `hook_mlp_out` with H5 `mlp[L]`. "
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
        "sites=hook_q_input,hook_k_input,hook_v_input,hook_mlp_in",
        "attn_direction=h5_res[L] (resid-pre of block L)",
        "mlp_direction=h5_res[L]+attn[L] (resid-mid, cloned onto hook_mlp_in)",
        "fine_head_score=q_input_dot + k_input_dot + v_input_dot (GQA shares K/V within query group)",
        f"attribute_component_outputs={args.attribute_component_outputs}",
        "output_pass=second_independent_forward",
        "output_sites=hook_attn_out,hook_mlp_out",
        "output_attn_direction=h5_attn[L] (hook_attn_out addend; after ln1_post / post_attention_layernorm on sandwich Gemma)",
        "output_mlp_direction=h5_mlp[L] (hook_mlp_out; after ln2_post / post_feedforward_layernorm on sandwich Gemma)",
        "output_fine_head_score=GBA post-norm head split of d_attn (coarse attn = <grad_addend, d_attn> = sum of heads)",
        f"output_results_subdir={COMPONENT_OUTPUT_SUBDIR}",
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
            "verbalised-confidence logit differences."
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
            "(H5 mlp; after ln2_post on sandwich Gemma-3). Written under "
            "component_output/. If false, input-site pass only."
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

    l_vs_un_skipped_ids: set[str] = set()
    output_run_root = run_root / COMPONENT_OUTPUT_SUBDIR

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
