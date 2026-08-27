#!/usr/bin/env python3
"""Direct Logit Attribution (DLA) on verbalised-confidence tokens.

Numeric Probability: runs read processed H5 embeddings generated with
extend_probability_span and attribute logit contrasts for the pre-period and
post-period digits to attention/MLP residual writes (coarse) and individual
attention heads (fine).

Linguistic Confidence: runs (`--linguistic_confidence_prompt`) assume the input
H5 used a variable-length `--extend_probability_span` (phrase-dependent). They
attribute `confidence_token_vs_rest` at the first phrase token and, for Mistral,
also `L_vs_Un` at span index 6, keeping only examples whose decoded token there
is `L` or `Un`. Other models raise until those experiments are added.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_generations.process_generations_more_embs_from_h5 import (
    configure_prefix_tokens_for_model,
    parse_guess_and_probability_indices,
)

SUPPORTED_MODEL_NAMES = (
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-32B-Instruct",
)

# First subword of each allowed Confidence: phrase, keyed by --model_name.
# Gemma/Qwen lists are placeholders to fill in later.
LINGUISTIC_FIRST_TOKENS: dict[str, tuple[str, ...]] = {
    "mistralai/Mistral-7B-Instruct-v0.1": (
        "Almost",
        "High",
        "Ch",
        "Little",
        "Un",
        "Probably",
        "About",
        "Better",
        "L",
        "Very",
    ),
    "google/gemma-3-12b-it": (),
    "Qwen/Qwen2.5-32B-Instruct": (),
}

ANNOTATION_ROUNDING_DP = 2
HEAD_RESUM_ATOL = 1e-2
COMPLETENESS_ATOL = 1e-2
CONFIDENCE_ONE_ATOL = 1e-9
L_VS_UN_DECODED_WINDOW = 3
L_VS_UN_RESPONSE_LOG_CHARS = 160
POSITION_ALIGNMENT_ASSUMPTION = (
    "cached H5 embeddings_probability rows are already at the predicting residual "
    "for each span token (generation step i predicts decoded_tokens[i]); no -1 shift applied"
)
WEIGHT_LOAD_METHOD = "safetensors_selective_cpu"

# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------


@dataclass
class ModelAdapter:
    model_name: str
    rmsnorm_gain: str  # "plain_weight" | "one_plus_weight"
    w_u_source: str  # "lm_head" | "tied_embeddings"
    head_dim: int
    n_query_heads: int
    n_kv_heads: int
    hidden_size: int
    n_layers: int
    rms_norm_eps: float
    norm_path: tuple[str, ...]
    layers_path: tuple[str, ...]
    trust_remote_code: bool


ADAPTER_SPECS: dict[str, dict] = {
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "rmsnorm_gain": "plain_weight",
        "norm_path": ("model", "norm"),
        "norm_path_fallbacks": (),
        "layers_path": ("model", "layers"),
        "layers_path_fallbacks": (),
        "embed_path": ("model", "embed_tokens"),
        "embed_path_fallbacks": (),
        "lm_head_path": ("lm_head",),
        "lm_head_path_fallbacks": (),
        "rms_norm_eps": 1e-5,
        "trust_remote_code": False,
        "config_root": None,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "rmsnorm_gain": "plain_weight",
        "norm_path": ("model", "norm"),
        "norm_path_fallbacks": (),
        "layers_path": ("model", "layers"),
        "layers_path_fallbacks": (),
        "embed_path": ("model", "embed_tokens"),
        "embed_path_fallbacks": (),
        "lm_head_path": ("lm_head",),
        "lm_head_path_fallbacks": (),
        "rms_norm_eps": 1e-6,
        "trust_remote_code": True,
        "config_root": None,
    },
    "google/gemma-3-12b-it": {
        # HF Gemma3 multimodal checkpoints store text weights under
        # language_model.model.* (no leading "model." prefix in safetensors).
        "rmsnorm_gain": "one_plus_weight",
        "norm_path": ("language_model", "model", "norm"),
        "norm_path_fallbacks": (
            ("language_model", "norm"),
            ("model", "language_model", "model", "norm"),
            ("model", "language_model", "norm"),
        ),
        "layers_path": ("language_model", "model", "layers"),
        "layers_path_fallbacks": (
            ("language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("model", "language_model", "layers"),
        ),
        "embed_path": ("language_model", "model", "embed_tokens"),
        "embed_path_fallbacks": (
            ("language_model", "embed_tokens"),
            ("model", "language_model", "model", "embed_tokens"),
            ("model", "language_model", "embed_tokens"),
        ),
        # Gemma3-IT ties embeddings; no standalone lm_head in the shard map.
        "lm_head_path": ("lm_head",),
        "lm_head_path_fallbacks": (
            ("language_model", "lm_head"),
            ("model", "lm_head"),
            ("model", "language_model", "lm_head"),
        ),
        "rms_norm_eps": 1e-6,
        "trust_remote_code": True,
        "config_root": "text_config",
    },
}


def _path_to_weight_key(path: tuple[str, ...]) -> str:
    return ".".join(path) + ".weight"


def _weight_key_candidates(
    primary: tuple[str, ...], fallbacks: tuple[tuple[str, ...], ...] = ()
) -> list[str]:
    keys = [_path_to_weight_key(primary)]
    for fb in fallbacks:
        key = _path_to_weight_key(fb)
        if key not in keys:
            keys.append(key)
    return keys


def _o_proj_key_candidates(
    layers_path: tuple[str, ...],
    layers_fallbacks: tuple[tuple[str, ...], ...],
    layer_idx: int,
) -> list[str]:
    keys: list[str] = []
    for prefix in (layers_path, *layers_fallbacks):
        key = ".".join(prefix) + f".{layer_idx}.self_attn.o_proj.weight"
        if key not in keys:
            keys.append(key)
    return keys


def _post_attn_norm_key_candidates(
    layers_path: tuple[str, ...],
    layers_fallbacks: tuple[tuple[str, ...], ...],
    layer_idx: int,
) -> list[str]:
    keys: list[str] = []
    for prefix in (layers_path, *layers_fallbacks):
        key = ".".join(prefix) + f".{layer_idx}.post_attention_layernorm.weight"
        if key not in keys:
            keys.append(key)
    return keys


def _post_ff_norm_key_candidates(
    layers_path: tuple[str, ...],
    layers_fallbacks: tuple[tuple[str, ...], ...],
    layer_idx: int,
) -> list[str]:
    keys: list[str] = []
    for prefix in (layers_path, *layers_fallbacks):
        key = ".".join(prefix) + f".{layer_idx}.post_feedforward_layernorm.weight"
        if key not in keys:
            keys.append(key)
    return keys


def _layers_prefix_from_o_proj_key(key: str) -> tuple[str, ...]:
    marker = ".self_attn.o_proj.weight"
    if not key.endswith(marker):
        raise ValueError(f"Unexpected o_proj key: {key!r}")
    # model.layers.12.self_attn.o_proj.weight -> model.layers
    body = key[: -len(marker)]
    parts = body.split(".")
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(f"Could not parse layers prefix from o_proj key: {key!r}")
    return tuple(parts[:-1])


def _norm_path_from_weight_key(key: str) -> tuple[str, ...]:
    if not key.endswith(".weight"):
        raise ValueError(f"Unexpected norm key: {key!r}")
    return tuple(key[: -len(".weight")].split("."))


def _cfg_int(cfg, *names: str, default: int | None = None) -> int:
    for name in names:
        val = getattr(cfg, name, None)
        if val is not None:
            return int(val)
    if default is not None:
        return int(default)
    raise ValueError(f"None of {names} found on config {type(cfg).__name__}")


def _select_text_config(config, *, config_root: str | None):
    if config_root == "text_config":
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            raise ValueError(
                f"config_root=text_config but {type(config).__name__} has no text_config."
            )
        return text_cfg
    return config


def build_adapter_from_config(model_name: str) -> ModelAdapter:
    if model_name not in ADAPTER_SPECS:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Supported: {list(ADAPTER_SPECS)}"
        )
    spec = ADAPTER_SPECS[model_name]
    trust_remote_code = bool(spec["trust_remote_code"])
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    cfg = _select_text_config(config, config_root=spec.get("config_root"))

    n_query_heads = _cfg_int(cfg, "num_attention_heads", "n_head")
    n_kv_heads = _cfg_int(
        cfg,
        "num_key_value_heads",
        "n_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        default=n_query_heads,
    )
    hidden_size = _cfg_int(cfg, "hidden_size", "n_embd")
    n_layers = _cfg_int(cfg, "num_hidden_layers", "n_layer")

    head_dim_attr = getattr(cfg, "head_dim", None)
    if head_dim_attr is not None:
        head_dim = int(head_dim_attr)
    else:
        if hidden_size % n_query_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} not divisible by n_query_heads={n_query_heads} "
                "and config.head_dim is missing."
            )
        head_dim = hidden_size // n_query_heads
        if model_name == "google/gemma-3-12b-it":
            raise ValueError(
                "Gemma3 must expose config.head_dim; refusing hidden_size/n_heads fallback."
            )

    if n_query_heads % n_kv_heads != 0:
        raise ValueError(
            f"Invalid GQA: n_query_heads={n_query_heads} not divisible by n_kv_heads={n_kv_heads}."
        )

    # Prefer top-level tie flag; fall back to text_config.
    tie = bool(getattr(config, "tie_word_embeddings", False))
    if not tie:
        tie = bool(getattr(cfg, "tie_word_embeddings", False))
    w_u_source = "tied_embeddings" if tie else "lm_head"

    return ModelAdapter(
        model_name=model_name,
        rmsnorm_gain=str(spec["rmsnorm_gain"]),
        w_u_source=w_u_source,
        head_dim=head_dim,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        hidden_size=hidden_size,
        n_layers=n_layers,
        rms_norm_eps=float(spec["rms_norm_eps"]),
        norm_path=tuple(spec["norm_path"]),
        layers_path=tuple(spec["layers_path"]),
        trust_remote_code=trust_remote_code,
    )


@dataclass
class ModelTensors:
    adapter: ModelAdapter
    tokenizer: object
    gamma: np.ndarray  # [H]
    w_u: np.ndarray  # [H, V] columns = token directions
    o_proj_weights: list[np.ndarray]  # per layer [out, in] = [H, n_q * d]
    resolved_weight_keys: dict[str, str]
    # Gemma: per-layer post_attention_layernorm gain (1+w). None for other models.
    post_attn_norm_gammas: list[np.ndarray] | None = None
    # Gemma legacy H5: per-layer post_feedforward_layernorm gain (1+w).
    post_ff_norm_gammas: list[np.ndarray] | None = None
    # Per-layer inv(W_O) when fine DLA recovers concat from o. None otherwise.
    o_proj_inv_weights: list[np.ndarray] | None = None
    weight_load_method: str = WEIGHT_LOAD_METHOD


# ---------------------------------------------------------------------------
# Safetensors selective reader (CPU)
# ---------------------------------------------------------------------------


def _build_weight_map(model_name: str) -> dict[str, str]:
    """Map state-dict key -> absolute path of the safetensors shard containing it."""
    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    except Exception:
        index_path = None

    if index_path is not None:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                f"{model_name}: model.safetensors.index.json has empty/missing weight_map."
            )
        base = Path(index_path).parent
        # Ensure referenced shards are present in the hub cache.
        shard_names = sorted(set(weight_map.values()))
        shard_paths: dict[str, str] = {}
        for shard_name in shard_names:
            shard_paths[shard_name] = hf_hub_download(model_name, shard_name)
        return {key: shard_paths[shard] for key, shard in weight_map.items()}

    # Single-file checkpoint.
    try:
        single_path = hf_hub_download(model_name, "model.safetensors")
    except Exception as exc:
        raise RuntimeError(
            f"{model_name}: could not find model.safetensors.index.json or "
            f"model.safetensors in the Hugging Face cache/hub ({exc})."
        ) from exc
    with safe_open(single_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    if not keys:
        raise RuntimeError(f"{model_name}: {single_path} contains no tensors.")
    return {key: single_path for key in keys}


def _load_numpy_tensor(shard_path: str, key: str) -> np.ndarray:
    # Use PyTorch backend: many checkpoints store bfloat16, which NumPy cannot decode.
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"Key {key!r} not in shard {shard_path}")
        tensor = f.get_tensor(key)
    return tensor.detach().to(dtype=torch.float32).contiguous().numpy()


def load_first_matching_weight(
    weight_map: dict[str, str], candidates: Sequence[str], *, label: str
) -> tuple[np.ndarray, str]:
    tried: list[str] = []
    for key in candidates:
        tried.append(key)
        shard = weight_map.get(key)
        if shard is None:
            continue
        arr = _load_numpy_tensor(shard, key)
        logging.info("Loaded %s from key %s (shape=%s)", label, key, arr.shape)
        return arr, key
    present_sample = sorted(weight_map.keys())[:20]
    raise RuntimeError(
        f"Could not load {label}. Tried keys:\n  "
        + "\n  ".join(tried)
        + f"\nNone were present in the checkpoint weight map "
        f"({len(weight_map)} keys). Sample keys: {present_sample}"
    )


def invert_square_o_proj(w: np.ndarray, *, layer_idx: int) -> np.ndarray:
    """Invert a square o_proj / W_O matrix in float64, returning float32."""
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(
            f"o_proj.weight at layer {layer_idx} has shape {w.shape}; "
            "recovering concat from o requires a square invertible W_O."
        )
    try:
        inv = np.linalg.inv(w.astype(np.float64, copy=False))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"o_proj.weight at layer {layer_idx} is not invertible."
        ) from exc
    return inv.astype(np.float32, copy=False)


def maybe_invert_o_proj_for_concat_fallback(
    tensors: ModelTensors,
    examples: Sequence[ExampleData],
) -> None:
    """Invert W_O if any example lacks concat so fine DLA can recover it from o."""
    seen_ids: set[str] = set()
    unique_examples: list[ExampleData] = []
    for ex in examples:
        if ex.example_id in seen_ids:
            continue
        seen_ids.add(ex.example_id)
        unique_examples.append(ex)
    missing_concat = [ex for ex in unique_examples if ex.concat is None]
    if not missing_concat:
        return
    missing_o = [ex for ex in missing_concat if ex.o is None]
    if missing_o:
        raise ValueError(
            "Fine granularity requires concat embeddings, or o embeddings to recover "
            "concat. "
            f"{len(missing_o)} example(s) lack both (e.g. {missing_o[0].example_id})."
        )
    adapter = tensors.adapter
    expected_in = adapter.n_query_heads * adapter.head_dim
    if adapter.hidden_size != expected_in:
        raise ValueError(
            "concat embeddings are missing; recovering concat from o requires square "
            "W_O (hidden_size == n_query_heads*head_dim), got "
            f"hidden_size={adapter.hidden_size}, "
            f"n_query_heads*head_dim={expected_in}."
        )
    if len(tensors.o_proj_weights) != adapter.n_layers:
        raise ValueError(
            "Cannot recover concat from o: o_proj weights were not loaded for all "
            f"{adapter.n_layers} layers "
            f"(got {len(tensors.o_proj_weights)})."
        )
    logging.warning(
        "concat embeddings missing; recovering concat from o via inv(W_O) "
        "for %d example(s).",
        len(missing_concat),
    )
    invs: list[np.ndarray] = []
    for layer_idx, w in enumerate(tensors.o_proj_weights):
        logging.info(
            "Inverting o_proj W_O for layer %d/%d.",
            layer_idx + 1,
            adapter.n_layers,
        )
        invs.append(invert_square_o_proj(w, layer_idx=layer_idx))
    tensors.o_proj_inv_weights = invs


def load_model_tensors(
    model_name: str,
    *,
    need_o_proj: bool,
    legacy_gemma_pre_postnorm_h5: bool = False,
) -> ModelTensors:
    if model_name not in ADAPTER_SPECS:
        raise ValueError(
            f"Unsupported --model_name={model_name!r}. "
            f"Must be one of: {list(SUPPORTED_MODEL_NAMES)}"
        )
    spec = ADAPTER_SPECS[model_name]
    trust_remote_code = bool(spec["trust_remote_code"])

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        logging.warning(
            "Fast tokenizer load failed (%s). Falling back to use_fast=False.", exc
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, use_fast=False
        )

    adapter = build_adapter_from_config(model_name)
    logging.info(
        "Loading DLA weights for %s via selective safetensors on CPU "
        "(need_o_proj=%s, legacy_gemma_pre_postnorm_h5=%s).",
        model_name,
        need_o_proj,
        legacy_gemma_pre_postnorm_h5,
    )
    weight_map = _build_weight_map(model_name)
    resolved: dict[str, str] = {}

    norm_candidates = _weight_key_candidates(
        tuple(spec["norm_path"]), tuple(spec.get("norm_path_fallbacks", ()))
    )
    norm_weight, norm_key = load_first_matching_weight(
        weight_map, norm_candidates, label="final_norm.weight"
    )
    resolved["final_norm"] = norm_key
    adapter.norm_path = _norm_path_from_weight_key(norm_key)

    if adapter.rmsnorm_gain == "one_plus_weight":
        gamma = (1.0 + norm_weight).astype(np.float32, copy=False)
    elif adapter.rmsnorm_gain == "plain_weight":
        gamma = norm_weight.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown rmsnorm_gain={adapter.rmsnorm_gain!r}")

    embed_candidates = _weight_key_candidates(
        tuple(spec["embed_path"]), tuple(spec.get("embed_path_fallbacks", ()))
    )
    lm_head_candidates = _weight_key_candidates(
        tuple(spec["lm_head_path"]), tuple(spec.get("lm_head_path_fallbacks", ()))
    )
    if adapter.w_u_source == "tied_embeddings":
        w_u_candidates = embed_candidates + [
            k for k in lm_head_candidates if k not in embed_candidates
        ]
        w_u_label = "tied_embeddings / W_U"
    else:
        w_u_candidates = lm_head_candidates + [
            k for k in embed_candidates if k not in lm_head_candidates
        ]
        w_u_label = "lm_head / W_U"
    # Checkpoint stores [V, H]; DLA uses columns as token directions → transpose.
    w_u_raw, w_u_key = load_first_matching_weight(
        weight_map, w_u_candidates, label=w_u_label
    )
    if w_u_raw.ndim != 2:
        raise ValueError(f"W_U source {w_u_key!r} has shape {w_u_raw.shape}, expected 2D.")
    w_u = np.asarray(w_u_raw.T, dtype=np.float32)
    resolved["w_u"] = w_u_key
    if w_u.shape[0] != adapter.hidden_size:
        raise ValueError(
            f"W_U hidden dim {w_u.shape[0]} != adapter.hidden_size={adapter.hidden_size} "
            f"(key={w_u_key})."
        )

    o_proj_weights: list[np.ndarray] = []
    post_attn_norm_gammas: list[np.ndarray] | None = None
    post_ff_norm_gammas: list[np.ndarray] | None = None
    is_gemma = adapter.rmsnorm_gain == "one_plus_weight"
    # New Gemma H5 stores post-norm attn; fine DLA still needs post-attn gains to
    # map o_proj head slices onto that write. Legacy H5 stores pre-norm attn/mlp,
    # so both post-attn and post-FF gains are required even for coarse-only.
    need_post_attn_norm = is_gemma and (need_o_proj or legacy_gemma_pre_postnorm_h5)
    need_post_ff_norm = is_gemma and legacy_gemma_pre_postnorm_h5
    if need_post_attn_norm:
        post_attn_norm_gammas = []
    if need_post_ff_norm:
        post_ff_norm_gammas = []

    def _load_post_block_norm(
        layer_idx: int,
        *,
        kind: str,
        candidates_fn,
        out_list: list[np.ndarray],
        resolved_prefix: str,
    ) -> None:
        candidates = candidates_fn(
            tuple(spec["layers_path"]),
            tuple(spec.get("layers_path_fallbacks", ())),
            layer_idx,
        )
        pn_w, pn_key = load_first_matching_weight(
            weight_map,
            candidates,
            label=f"{kind} layer {layer_idx}",
        )
        if pn_w.ndim != 1 or pn_w.shape[0] != adapter.hidden_size:
            raise ValueError(
                f"{kind}.weight at layer {layer_idx} (key={pn_key}) has shape "
                f"{pn_w.shape}, expected [{adapter.hidden_size}]."
            )
        out_list.append(
            (1.0 + np.asarray(pn_w, dtype=np.float32)).astype(np.float32, copy=False)
        )
        resolved[f"{resolved_prefix}_L{layer_idx}"] = pn_key

    if need_o_proj:
        expected_in = adapter.n_query_heads * adapter.head_dim
        resolved_layers_path: tuple[str, ...] | None = None
        for layer_idx in range(adapter.n_layers):
            candidates = _o_proj_key_candidates(
                tuple(spec["layers_path"]),
                tuple(spec.get("layers_path_fallbacks", ())),
                layer_idx,
            )
            w, key = load_first_matching_weight(
                weight_map, candidates, label=f"o_proj layer {layer_idx}"
            )
            if w.ndim != 2:
                raise ValueError(f"o_proj.weight at layer {layer_idx} has shape {w.shape}")
            if w.shape[0] != adapter.hidden_size or w.shape[1] != expected_in:
                raise ValueError(
                    f"o_proj.weight at layer {layer_idx} (key={key}) has shape {w.shape}, "
                    f"expected [{adapter.hidden_size}, {expected_in}] "
                    f"(out=hidden, in=n_query_heads*head_dim)."
                )
            o_proj_weights.append(np.asarray(w, dtype=np.float32))
            resolved[f"o_proj_L{layer_idx}"] = key
            prefix = _layers_prefix_from_o_proj_key(key)
            if resolved_layers_path is None:
                resolved_layers_path = prefix
            elif resolved_layers_path != prefix:
                raise ValueError(
                    f"Inconsistent layers prefix across o_proj keys: "
                    f"{resolved_layers_path} vs {prefix}"
                )
            if need_post_attn_norm:
                assert post_attn_norm_gammas is not None
                _load_post_block_norm(
                    layer_idx,
                    kind="post_attention_layernorm",
                    candidates_fn=_post_attn_norm_key_candidates,
                    out_list=post_attn_norm_gammas,
                    resolved_prefix="post_attn_norm",
                )
            if need_post_ff_norm:
                assert post_ff_norm_gammas is not None
                _load_post_block_norm(
                    layer_idx,
                    kind="post_feedforward_layernorm",
                    candidates_fn=_post_ff_norm_key_candidates,
                    out_list=post_ff_norm_gammas,
                    resolved_prefix="post_ff_norm",
                )
        assert resolved_layers_path is not None
        adapter.layers_path = resolved_layers_path
    else:
        logging.info("Skipping o_proj weight load (coarse-only run).")
        if need_post_attn_norm or need_post_ff_norm:
            for layer_idx in range(adapter.n_layers):
                if need_post_attn_norm:
                    assert post_attn_norm_gammas is not None
                    _load_post_block_norm(
                        layer_idx,
                        kind="post_attention_layernorm",
                        candidates_fn=_post_attn_norm_key_candidates,
                        out_list=post_attn_norm_gammas,
                        resolved_prefix="post_attn_norm",
                    )
                if need_post_ff_norm:
                    assert post_ff_norm_gammas is not None
                    _load_post_block_norm(
                        layer_idx,
                        kind="post_feedforward_layernorm",
                        candidates_fn=_post_ff_norm_key_candidates,
                        out_list=post_ff_norm_gammas,
                        resolved_prefix="post_ff_norm",
                    )

    return ModelTensors(
        adapter=adapter,
        tokenizer=tokenizer,
        gamma=gamma,
        w_u=w_u,
        o_proj_weights=o_proj_weights,
        resolved_weight_keys=resolved,
        post_attn_norm_gammas=post_attn_norm_gammas,
        post_ff_norm_gammas=post_ff_norm_gammas,
        o_proj_inv_weights=None,
        weight_load_method=WEIGHT_LOAD_METHOD,
    )


# ---------------------------------------------------------------------------
# Digit tokens and contrasts
# ---------------------------------------------------------------------------


def _token_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _is_digit_token_decode(decoded: str, digit: str) -> bool:
    """True if decoded token content is exactly the digit, optionally with surrounding whitespace."""
    if decoded == digit:
        return True
    # Some tokenizers surface a leading space / SentencePiece underscore as whitespace.
    return decoded.strip() == digit and digit in decoded


def resolve_single_digit_token_id(tokenizer, digit: str) -> tuple[int, str]:
    """Resolve a digit character to a single tokenizer ID.

    Prefer a true single-token encode. For tokenizers like Mistral that encode
    ``\"0\"`` as ``[leading_empty_or_space, digit]``, accept the digit piece when
    decoding that id yields exactly the digit (fail loudly if ambiguous).
    """
    if len(digit) != 1 or digit not in "0123456789":
        raise ValueError(f"Expected a single digit char, got {digit!r}")

    errors: list[str] = []
    candidates = [digit, f" {digit}", f"\n{digit}", f"\t{digit}"]

    # 1) Exact single-token encode of a common string form.
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            decoded = _token_decode(tokenizer, ids[0])
            if _is_digit_token_decode(decoded, digit):
                return int(ids[0]), cand
            errors.append(
                f"{cand!r} -> single id {ids[0]} but decode={decoded!r} is not digit {digit!r}"
            )
        else:
            errors.append(f"{cand!r} -> {ids} ({len(ids)} tokens)")

    # 2) convert_tokens_to_ids on bare / SentencePiece / GPT-2 forms.
    #    Do NOT require encode(decode(id)) to be length-1: Mistral's encode(\"0\")
    #    always prepends an empty leading piece even when id 28734 alone is \"0\".
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in (digit, f"▁{digit}", f"Ġ{digit}", f" {digit}"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or (unk is not None and int(tid) == int(unk)):
            continue
        decoded = _token_decode(tokenizer, int(tid))
        if _is_digit_token_decode(decoded, digit):
            return int(tid), tok
        errors.append(f"token {tok!r} id={tid} decode={decoded!r} is not digit {digit!r}")

    # 3) Multi-token encode where a unique piece decodes to the digit and all
    #    other pieces decode to empty/whitespace only (Mistral: [\"\", \"0\"]).
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) < 2:
            continue
        digit_ids: list[int] = []
        other_ok = True
        parts: list[str] = []
        for tid in ids:
            decoded = _token_decode(tokenizer, tid)
            parts.append(decoded)
            if _is_digit_token_decode(decoded, digit):
                digit_ids.append(int(tid))
            elif decoded.strip() != "":
                other_ok = False
        if other_ok and len(set(digit_ids)) == 1:
            return digit_ids[0], f"{cand}#piece"
        errors.append(
            f"{cand!r} multi-token pieces={parts!r} digit_ids={digit_ids} other_ok={other_ok}"
        )

    raise ValueError(
        f"Could not resolve digit {digit!r} to a single tokenizer token. Tried:\n  "
        + "\n  ".join(errors)
    )


def resolve_all_digit_ids(tokenizer) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for d in "0123456789":
        out[d] = resolve_single_digit_token_id(tokenizer, d)
    return out


def _is_piece_token_decode(decoded: str, piece: str) -> bool:
    """True if decoded token content is exactly ``piece``, optionally with surrounding whitespace."""
    if decoded == piece:
        return True
    return decoded.strip() == piece and piece in decoded


def resolve_single_piece_token_id(tokenizer, piece: str) -> tuple[int, str]:
    """Resolve a linguistic first-token string to a single tokenizer ID.

    Same leading-empty-piece handling as ``resolve_single_digit_token_id``:
    Mistral often encodes ``\"High\"`` as ``[leading_empty_or_space, High]``.
    """
    if not piece:
        raise ValueError("Linguistic first-token piece must be non-empty.")

    errors: list[str] = []
    candidates = [piece, f" {piece}", f"\n{piece}", f"\t{piece}"]

    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            decoded = _token_decode(tokenizer, ids[0])
            if _is_piece_token_decode(decoded, piece):
                return int(ids[0]), cand
            errors.append(
                f"{cand!r} -> single id {ids[0]} but decode={decoded!r} is not piece {piece!r}"
            )
        else:
            errors.append(f"{cand!r} -> {ids} ({len(ids)} tokens)")

    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in (piece, f"▁{piece}", f"Ġ{piece}", f" {piece}"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or (unk is not None and int(tid) == int(unk)):
            continue
        decoded = _token_decode(tokenizer, int(tid))
        if _is_piece_token_decode(decoded, piece):
            return int(tid), tok
        errors.append(f"token {tok!r} id={tid} decode={decoded!r} is not piece {piece!r}")

    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) < 2:
            continue
        piece_ids: list[int] = []
        other_ok = True
        parts: list[str] = []
        for tid in ids:
            decoded = _token_decode(tokenizer, tid)
            parts.append(decoded)
            if _is_piece_token_decode(decoded, piece):
                piece_ids.append(int(tid))
            elif decoded.strip() != "":
                other_ok = False
        if other_ok and len(set(piece_ids)) == 1:
            return piece_ids[0], f"{cand}#piece"
        errors.append(
            f"{cand!r} multi-token pieces={parts!r} piece_ids={piece_ids} other_ok={other_ok}"
        )

    raise ValueError(
        f"Could not resolve linguistic first token {piece!r} to a single tokenizer token. Tried:\n  "
        + "\n  ".join(errors)
    )


def linguistic_first_tokens_for_model(model_name: str) -> tuple[str, ...]:
    if model_name not in LINGUISTIC_FIRST_TOKENS:
        raise ValueError(
            f"No LINGUISTIC_FIRST_TOKENS entry for --model_name={model_name!r}."
        )
    pieces = LINGUISTIC_FIRST_TOKENS[model_name]
    if not pieces:
        raise ValueError(
            "--linguistic_confidence_prompt requires a non-empty first-token list "
            f"for --model_name={model_name!r}. Fill LINGUISTIC_FIRST_TOKENS."
        )
    return pieces


def resolve_linguistic_first_token_ids(
    tokenizer, pieces: Sequence[str]
) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for piece in pieces:
        out[piece] = resolve_single_piece_token_id(tokenizer, piece)
    return out


def validate_confidence_thresholds(
    high_conf_threshold: float, low_conf_threshold: float
) -> None:
    if not (0.0 <= low_conf_threshold <= 1.0 and 0.0 <= high_conf_threshold <= 1.0):
        raise ValueError("Confidence thresholds must be in [0, 1].")
    if low_conf_threshold >= high_conf_threshold:
        raise ValueError(
            f"low_conf_threshold ({low_conf_threshold}) must be < "
            f"high_conf_threshold ({high_conf_threshold})."
        )


def high_low_digit_sets(
    high_conf_threshold: float, low_conf_threshold: float
) -> tuple[set[str], set[str]]:
    """Digit token sets for the high_vs_low contrast direction only.

    Example cohort splits (``__high_conf`` / ``__low_conf``) still use the raw
    verbalised-confidence thresholds and are unaffected by this mapping.

    Digit ``\"0\"`` is included in the low set: with the default thresholds
    this yields high={9}, low={0,1}.
    """
    validate_confidence_thresholds(high_conf_threshold, low_conf_threshold)
    high = {str(d) for d in range(int(high_conf_threshold * 10), 10)}
    low = {str(d) for d in range(0, int(low_conf_threshold * 10) + 1)}
    if not high:
        raise ValueError(
            f"high_conf_threshold={high_conf_threshold} produced empty high digit set."
        )
    if not low:
        raise ValueError(
            f"low_conf_threshold={low_conf_threshold} produced empty low digit set "
            "for the high_vs_low direction."
        )
    if high & low:
        raise ValueError(
            f"high and low digit sets overlap: high={sorted(high)} low={sorted(low)}. "
            "Adjust thresholds."
        )
    return high, low


def mean_columns(w_u: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    cols = w_u[:, list(token_ids)]
    return np.mean(cols, axis=1).astype(np.float32, copy=False)


def contrast_direction(
    w_u: np.ndarray,
    pos_ids: Sequence[int],
    neg_ids: Sequence[int] | None,
    *,
    vocab_size: int,
) -> np.ndarray:
    """pos mean minus neg mean. If neg_ids is None, neg = all tokens not in pos."""
    if not pos_ids:
        raise ValueError("pos_ids must be non-empty")
    u_pos = mean_columns(w_u, pos_ids)
    if neg_ids is None:
        n_pos = len(pos_ids)
        if vocab_size <= n_pos:
            raise ValueError("No negative tokens remain for contrast.")
        # mean_others = (sum_all - sum_pos) / (V - n_pos)
        sum_all = np.sum(w_u, axis=1, dtype=np.float64)
        sum_pos = np.sum(w_u[:, list(pos_ids)], axis=1, dtype=np.float64)
        u_neg = ((sum_all - sum_pos) / (vocab_size - n_pos)).astype(np.float32)
    else:
        if not neg_ids:
            raise ValueError("neg_ids must be non-empty when provided")
        u_neg = mean_columns(w_u, neg_ids)
    return (u_pos - u_neg).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ContrastSpec:
    position: str  # "pre_period" | "post_period" | "first_value" | "mistral_l_vs_un"
    name: str  # e.g. "digit_vs_rest" | "confidence_token_vs_rest" | "L_vs_Un"
    u: np.ndarray  # [H]
    pos_token_ids: tuple[int, ...]
    neg_token_ids: tuple[int, ...] | None  # None => all others
    span_index: int | None = None  # absolute embeddings_probability index when set


def build_contrasts(
    w_u: np.ndarray,
    digit_ids: dict[str, tuple[int, str]],
    high_digits: set[str],
    low_digits: set[str],
) -> list[ContrastSpec]:
    vocab_size = w_u.shape[1]
    id0 = digit_ids["0"][0]
    id1 = digit_ids["1"][0]
    all_digit_ids = [digit_ids[d][0] for d in "0123456789"]
    high_ids = [digit_ids[d][0] for d in sorted(high_digits)]
    low_ids = [digit_ids[d][0] for d in sorted(low_digits)]

    return [
        ContrastSpec(
            position="pre_period",
            name="digit_vs_rest",
            u=contrast_direction(w_u, [id0, id1], None, vocab_size=vocab_size),
            pos_token_ids=(id0, id1),
            neg_token_ids=None,
        ),
        ContrastSpec(
            position="pre_period",
            name="one_vs_zero",
            u=contrast_direction(w_u, [id1], [id0], vocab_size=vocab_size),
            pos_token_ids=(id1,),
            neg_token_ids=(id0,),
        ),
        ContrastSpec(
            position="post_period",
            name="digit_vs_rest",
            u=contrast_direction(w_u, all_digit_ids, None, vocab_size=vocab_size),
            pos_token_ids=tuple(all_digit_ids),
            neg_token_ids=None,
        ),
        ContrastSpec(
            position="post_period",
            name="high_vs_low",
            u=contrast_direction(w_u, high_ids, low_ids, vocab_size=vocab_size),
            pos_token_ids=tuple(high_ids),
            neg_token_ids=tuple(low_ids),
        ),
    ]


def build_linguistic_contrasts(
    w_u: np.ndarray,
    first_token_ids: Sequence[int],
    *,
    l_token_id: int,
    un_token_id: int,
    first_value_index: int,
    l_vs_un_index: int = 6,
) -> list[ContrastSpec]:
    vocab_size = w_u.shape[1]
    pos_ids = tuple(dict.fromkeys(int(i) for i in first_token_ids))
    if not pos_ids:
        raise ValueError("Linguistic first-token id list is empty after deduplication.")
    if first_value_index < 0:
        raise ValueError(f"first_value_index must be >= 0, got {first_value_index}")
    if l_vs_un_index < 0:
        raise ValueError(f"l_vs_un_index must be >= 0, got {l_vs_un_index}")
    return [
        ContrastSpec(
            position="first_value",
            name="confidence_token_vs_rest",
            u=contrast_direction(w_u, pos_ids, None, vocab_size=vocab_size),
            pos_token_ids=pos_ids,
            neg_token_ids=None,
            span_index=int(first_value_index),
        ),
        ContrastSpec(
            position="mistral_l_vs_un",
            name="L_vs_Un",
            u=contrast_direction(
                w_u, [int(l_token_id)], [int(un_token_id)], vocab_size=vocab_size
            ),
            pos_token_ids=(int(l_token_id),),
            neg_token_ids=(int(un_token_id),),
            span_index=int(l_vs_un_index),
        ),
    ]


def position_index(position: str, span_len: int, *, span_index: int | None = None) -> int:
    if span_index is not None:
        if span_index < 0:
            raise ValueError(f"span_index must be >= 0, got {span_index}")
        if span_index >= span_len:
            raise ValueError(
                f"span_index {span_index} is out of range for span length {span_len}."
            )
        return int(span_index)
    if position == "last_span":
        if span_len < 1:
            raise ValueError(
                f"Confidence span length {span_len} < 1; cannot locate last token."
            )
        return span_len - 1
    if span_len < 3:
        raise ValueError(f"Probability span length {span_len} < 3; cannot locate digits.")
    if position == "pre_period":
        return span_len - 3  # 3rd-last
    if position == "post_period":
        return span_len - 1  # last
    raise ValueError(f"Unknown position {position!r}")


# ---------------------------------------------------------------------------
# H5 I/O
# ---------------------------------------------------------------------------


def _tensor_to_numpy(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _as_layer_hidden(arr_like) -> np.ndarray:
    arr = _tensor_to_numpy(arr_like)
    if arr.ndim == 4:
        return np.asarray(arr[:, 0, -1, :], dtype=np.float32)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32)
    raise ValueError(f"Unsupported embedding tensor rank {arr.ndim}; expected 2 or 4.")


def _h5_response0_group(example_group: h5py.Group) -> h5py.Group | None:
    responses = example_group.get("responses")
    if responses is None or not isinstance(responses, h5py.Group):
        return None
    r0 = responses.get("0")
    if r0 is None or not isinstance(r0, h5py.Group):
        return None
    return r0


def _h5_list_length(group_obj: h5py.Group) -> int:
    return int(group_obj.attrs.get("__len__", len(group_obj.keys())))


def _is_verbalised_confidence_one(conf: float) -> bool:
    return float(conf) >= 1.0 - CONFIDENCE_ONE_ATOL


def _read_verbalised_confidence_scalar(r0: h5py.Group) -> float | None:
    ds = r0.get("verbalised_confidence")
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    try:
        value = ds[()]
        if isinstance(value, np.ndarray):
            return float(np.asarray(value).reshape(-1)[0])
        return float(value)
    except (TypeError, ValueError, OSError):
        return None


def _read_response_string(r0: h5py.Group) -> str:
    ds = r0.get("response")
    if ds is None:
        return ""
    data = ds[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            item = data.item()
            if isinstance(item, bytes):
                return item.decode("utf-8")
            return str(item)
        return str(data)
    return str(data)


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_h5_string(value.item())
    return str(value)


def _read_decoded_tokens(r0: h5py.Group) -> list[str] | None:
    node = r0.get("decoded_tokens")
    if node is None:
        return None
    if isinstance(node, h5py.Group):
        if node.attrs.get("__type__") == "none":
            return None
        length = _h5_list_length(node)
        if length <= 0:
            return None
        out: list[str] = []
        for i in range(length):
            item = node.get(str(i))
            if item is None or not isinstance(item, h5py.Dataset):
                return None
            out.append(_decode_h5_string(item[()]))
        return out
    if not isinstance(node, h5py.Dataset):
        return None
    data = node[()]
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            text = _decode_h5_string(data.item())
            return [text] if text else None
        tokens = [_decode_h5_string(x) for x in data.tolist()]
        return tokens if tokens else None
    text = _decode_h5_string(data)
    return [text] if text else None


def _get_list_embeddings(r0: h5py.Group, component: str) -> list[np.ndarray] | None:
    field = r0.get("embeddings_probability")
    if field is None or not isinstance(field, h5py.Group):
        return None
    comp_group = field.get(component)
    if comp_group is None:
        return None
    if isinstance(comp_group, h5py.Group) and comp_group.attrs.get("__type__") == "none":
        return None
    if not isinstance(comp_group, h5py.Group):
        return None
    length = _h5_list_length(comp_group)
    out: list[np.ndarray] = []
    for i in range(length):
        ds = comp_group.get(str(i))
        if ds is None or not isinstance(ds, h5py.Dataset):
            return None
        out.append(_as_layer_hidden(ds[()]))
    return out


@dataclass
class ExampleData:
    example_id: str
    verbalised_confidence: float
    response: str
    res: list[np.ndarray]  # per span token, [n_res_layers, H]
    attn: list[np.ndarray]  # per span token, [n_layers, H]
    mlp: list[np.ndarray]
    concat: list[np.ndarray] | None  # preferred for fine; optional if o is present
    o: list[np.ndarray] | None = None  # o_proj outs; concat fallback + fine diagnostics
    decoded_tokens: list[str] | None = None


def _decoded_token_window(
    tokens: Sequence[str], index: int, *, radius: int = L_VS_UN_DECODED_WINDOW
) -> list[str]:
    start = max(0, int(index) - radius)
    end = min(len(tokens), int(index) + radius + 1)
    return list(tokens[start:end])


def _l_vs_un_token_check(
    ex: ExampleData, span_index: int
) -> tuple[str | None, int | None, str | None]:
    """Return (token, first_prob_token_index, skip_reason). skip_reason is None if keep."""
    if not ex.decoded_tokens:
        return None, None, "missing decoded_tokens"
    parsed = parse_guess_and_probability_indices(
        ex.decoded_tokens, linguistic_confidence_prompt=True
    )
    if parsed is None:
        return None, None, "Confidence: prefix parse failed"
    _last_guess, first_prob, _end_prob = parsed
    abs_idx = int(first_prob) + int(span_index)
    if abs_idx < 0 or abs_idx >= len(ex.decoded_tokens):
        return (
            None,
            int(first_prob),
            f"mapped index {abs_idx} out of range (len={len(ex.decoded_tokens)})",
        )
    token = ex.decoded_tokens[abs_idx]
    if _is_piece_token_decode(token, "L") or _is_piece_token_decode(token, "Un"):
        return token, int(first_prob), None
    return token, int(first_prob), "token is not L or Un"


def _log_l_vs_un_skip(
    ex: ExampleData,
    *,
    span_index: int,
    token: str | None,
    first_prob_token_index: int | None,
    reason: str,
) -> None:
    abs_idx = (
        None
        if first_prob_token_index is None
        else int(first_prob_token_index) + int(span_index)
    )
    window: list[str] | None = None
    if ex.decoded_tokens is not None and abs_idx is not None:
        window = _decoded_token_window(ex.decoded_tokens, abs_idx)
    response_preview = ex.response[:L_VS_UN_RESPONSE_LOG_CHARS]
    logging.info(
        "Skipping L_vs_Un example %s: %s; span_index=%d token=%r "
        "first_prob_token_index=%s decoded_window=%r response=%r",
        ex.example_id,
        reason,
        span_index,
        token,
        first_prob_token_index,
        window,
        response_preview,
    )


def _open_h5_readonly(path: str):
    """Open an H5 file for reading; disable locking when supported (CephFS-friendly)."""
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _try_parse_example_data(
    *,
    ex_id: str,
    r0: h5py.Group,
    expected_span: int,
    span_mismatch_detail: str,
    require_head_source: bool,
    require_exact_span: bool = True,
    require_decoded_tokens: bool = False,
) -> ExampleData | None:
    conf = _read_verbalised_confidence_scalar(r0)
    if conf is None:
        return None
    res = _get_list_embeddings(r0, "res")
    attn = _get_list_embeddings(r0, "attn")
    mlp = _get_list_embeddings(r0, "mlp")
    concat = _get_list_embeddings(r0, "concat")
    o = _get_list_embeddings(r0, "o")
    if res is None or attn is None or mlp is None:
        return None
    if require_head_source and concat is None and o is None:
        return None
    span_len = len(res)
    if require_exact_span:
        if span_len != expected_span:
            raise ValueError(
                f"Example {ex_id}: embeddings_probability span length is {span_len}, "
                f"{span_mismatch_detail}"
            )
    elif span_len < expected_span:
        return None
    if len(attn) != span_len or len(mlp) != span_len:
        raise ValueError(
            f"Example {ex_id}: mismatched span lengths "
            f"res={len(res)} attn={len(attn)} mlp={len(mlp)}"
        )
    if concat is not None and len(concat) != span_len:
        raise ValueError(
            f"Example {ex_id}: mismatched concat span length "
            f"res={span_len} concat={len(concat)}"
        )
    if o is not None and len(o) != span_len:
        raise ValueError(
            f"Example {ex_id}: mismatched o span length "
            f"res={span_len} o={len(o)}"
        )
    decoded_tokens = _read_decoded_tokens(r0)
    if require_decoded_tokens and not decoded_tokens:
        raise ValueError(
            f"Example {ex_id}: missing responses/0/decoded_tokens. Reprocess with "
            "process_generations_more_embs_from_h5.py so decoded_tokens are stored."
        )
    return ExampleData(
        example_id=str(ex_id),
        verbalised_confidence=float(conf),
        response=_read_response_string(r0),
        res=res,
        attn=attn,
        mlp=mlp,
        concat=concat,
        o=o,
        decoded_tokens=decoded_tokens,
    )


@dataclass
class ExampleCohorts:
    all: list[ExampleData]
    high: list[ExampleData]
    low: list[ExampleData]
    all_excl_one: list[ExampleData]
    high_excl_one: list[ExampleData]
    low_excl_one: list[ExampleData]


def load_usable_example_cohorts(
    input_h5: str,
    *,
    expected_span: int,
    span_mismatch_detail: str,
    max_examples: int,
    require_head_source: bool,
    high_conf_threshold: float,
    low_conf_threshold: float,
    fill_confidence_splits: bool,
    require_exact_span: bool = True,
    fill_post_period_excl_one: bool = False,
    require_decoded_tokens: bool = False,
) -> ExampleCohorts:
    """Scan H5 once; fill all/high/low cohorts with independent caps at ``max_examples``.

    When ``fill_post_period_excl_one`` is true, also fill matching post-period
    cohorts that skip verbalised_confidence == 1.0, still capped independently.
    """
    examples_all: list[ExampleData] = []
    examples_high: list[ExampleData] = []
    examples_low: list[ExampleData] = []
    examples_all_excl_one: list[ExampleData] = []
    examples_high_excl_one: list[ExampleData] = []
    examples_low_excl_one: list[ExampleData] = []
    with _open_h5_readonly(input_h5) as f:
        examples = f.get("examples")
        if examples is None or not isinstance(examples, h5py.Group):
            raise ValueError(f"{input_h5} has no 'examples' group.")
        example_ids = sorted(examples.keys(), key=lambda x: (len(x), x))
        for ex_id in tqdm(example_ids, desc="Scanning H5 examples"):
            all_full = len(examples_all) >= max_examples
            high_full = (not fill_confidence_splits) or len(examples_high) >= max_examples
            low_full = (not fill_confidence_splits) or len(examples_low) >= max_examples
            all_excl_full = (
                (not fill_post_period_excl_one)
                or len(examples_all_excl_one) >= max_examples
            )
            high_excl_full = (
                (not fill_post_period_excl_one)
                or (not fill_confidence_splits)
                or len(examples_high_excl_one) >= max_examples
            )
            low_excl_full = (
                (not fill_post_period_excl_one)
                or (not fill_confidence_splits)
                or len(examples_low_excl_one) >= max_examples
            )
            if (
                all_full
                and high_full
                and low_full
                and all_excl_full
                and high_excl_full
                and low_excl_full
            ):
                break

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
            is_one = _is_verbalised_confidence_one(conf_f)
            want_all = not all_full
            want_high = (
                fill_confidence_splits
                and (not high_full)
                and conf_f >= high_conf_threshold
            )
            want_low = (
                fill_confidence_splits
                and (not low_full)
                and conf_f <= low_conf_threshold
            )
            want_all_excl = (
                fill_post_period_excl_one and (not all_excl_full) and (not is_one)
            )
            want_high_excl = (
                fill_post_period_excl_one
                and fill_confidence_splits
                and (not high_excl_full)
                and (not is_one)
                and conf_f >= high_conf_threshold
            )
            want_low_excl = (
                fill_post_period_excl_one
                and fill_confidence_splits
                and (not low_excl_full)
                and (not is_one)
                and conf_f <= low_conf_threshold
            )
            if not (
                want_all
                or want_high
                or want_low
                or want_all_excl
                or want_high_excl
                or want_low_excl
            ):
                continue

            parsed = _try_parse_example_data(
                ex_id=str(ex_id),
                r0=r0,
                expected_span=expected_span,
                span_mismatch_detail=span_mismatch_detail,
                require_head_source=require_head_source,
                require_exact_span=require_exact_span,
                require_decoded_tokens=require_decoded_tokens,
            )
            if parsed is None:
                continue

            if want_all:
                examples_all.append(parsed)
            if want_high:
                examples_high.append(parsed)
            if want_low:
                examples_low.append(parsed)
            if want_all_excl:
                examples_all_excl_one.append(parsed)
            if want_high_excl:
                examples_high_excl_one.append(parsed)
            if want_low_excl:
                examples_low_excl_one.append(parsed)

    if not examples_all:
        required = (
            "res/attn/mlp and concat or o" if require_head_source else "res/attn/mlp"
        )
        raise ValueError(
            f"No usable examples found in {input_h5} with required "
            f"{required} probability embeddings."
        )
    return ExampleCohorts(
        all=examples_all,
        high=examples_high,
        low=examples_low,
        all_excl_one=examples_all_excl_one,
        high_excl_one=examples_high_excl_one,
        low_excl_one=examples_low_excl_one,
    )


def load_usable_examples(
    input_h5: str,
    *,
    expected_probability_tokens: int,
    max_examples: int,
    require_head_source: bool,
) -> list[ExampleData]:
    """Backward-compatible wrapper: return the all-examples cohort only."""
    expected_span = expected_probability_tokens + 2
    cohorts = load_usable_example_cohorts(
        input_h5,
        expected_span=expected_span,
        span_mismatch_detail=(
            f"but expected exactly expected_probability_tokens+2="
            f"{expected_probability_tokens}+2={expected_span}. "
            "Input H5 must have been generated with extend_probability_span."
        ),
        max_examples=max_examples,
        require_head_source=require_head_source,
        high_conf_threshold=1.0,
        low_conf_threshold=0.0,
        fill_confidence_splits=False,
    )
    return cohorts.all


# ---------------------------------------------------------------------------
# DLA core
# ---------------------------------------------------------------------------


def rms(x: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float32)) + eps))


def apply_final_rmsnorm(h: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    inv = 1.0 / rms(h, eps)
    return (h.astype(np.float32) * inv) * gamma


def true_contrast_score(
    h: np.ndarray,
    gamma: np.ndarray,
    w_u: np.ndarray,
    eps: float,
    pos_ids: Sequence[int],
    neg_ids: Sequence[int] | None,
) -> float:
    """Logit contrast from final residual via adapter RMSNorm + W_U (no softcap)."""
    h_norm = apply_final_rmsnorm(h, gamma, eps)
    # Prefer contracting only the needed columns when the negative set is explicit.
    if neg_ids is not None:
        pos_mean = float(np.mean(h_norm @ w_u[:, list(pos_ids)]))
        neg_mean = float(np.mean(h_norm @ w_u[:, list(neg_ids)]))
        return pos_mean - neg_mean

    # neg = all tokens except pos: mean_neg = (sum_all - sum_pos) / (V - n_pos)
    # Avoid materialising full-vocab logits when possible via two matvecs.
    pos_logits = h_norm @ w_u[:, list(pos_ids)]
    pos_mean = float(np.mean(pos_logits))
    # sum over vocab of logits = h_norm @ sum_of_all_columns
    vocab_col_sum = np.sum(w_u, axis=1, dtype=np.float32)
    sum_all = float(np.dot(h_norm, vocab_col_sum))
    n_pos = len(pos_ids)
    v = w_u.shape[1]
    if v <= n_pos:
        raise ValueError("No negative tokens remain for contrast.")
    sum_pos = float(np.sum(pos_logits))
    neg_mean = (sum_all - sum_pos) / (v - n_pos)
    return pos_mean - neg_mean


def attr_vec(v: np.ndarray, u_tilde: np.ndarray, sigma: float) -> float:
    return float(np.dot(v.astype(np.float32), u_tilde) / sigma)


def apply_gemma_rmsnorm(
    x: np.ndarray, gamma: np.ndarray, eps: float
) -> np.ndarray:
    """Gemma-style RMSNorm: x * gamma / rms(x) with gamma = (1+w)."""
    x32 = x.astype(np.float32, copy=False)
    g32 = gamma.astype(np.float32, copy=False)
    if g32.shape != x32.shape:
        raise ValueError(
            f"apply_gemma_rmsnorm: gamma shape {g32.shape} != x shape {x32.shape}"
        )
    return x32 * g32 * np.float32(1.0 / rms(x32, eps))


def _scale_heads_with_shared_post_attn_norm(
    heads: list[np.ndarray],
    gamma: np.ndarray,
    eps: float,
) -> list[np.ndarray]:
    total = np.zeros_like(heads[0], dtype=np.float32)
    for h_vec in heads:
        total += h_vec
    scale = gamma.astype(np.float32) * np.float32(1.0 / rms(total, eps))
    return [h_vec * scale for h_vec in heads]


@dataclass
class HeadDecomposeResult:
    heads: list[np.ndarray]
    max_abs_vs_attn: float
    pre_norm_total: np.ndarray  # W_O(concat) before optional post-attn norm


def recover_concat_from_o(o_layer: np.ndarray, o_proj_inv: np.ndarray) -> np.ndarray:
    """Recover pre-W_O concat from cached o (o_proj output): concat = W_O^{-1} @ o."""
    return o_proj_inv @ o_layer.astype(np.float32, copy=False)


def decompose_heads(
    concat_layer: np.ndarray,
    o_proj_weight: np.ndarray,
    *,
    n_query_heads: int,
    head_dim: int,
    attn_reference: np.ndarray,
    context: str,
    post_attn_norm_gamma: np.ndarray | None = None,
    post_attn_norm_eps: float | None = None,
) -> HeadDecomposeResult:
    """Return per-head residual writes and re-sum error vs cached attn.

    When ``post_attn_norm_gamma`` is set (Gemma), apply the shared post-attention
    RMSNorm gain to each pre-norm head write so they match post-norm H5 attn:
    ``write_h = head_h * gamma / rms(sum_heads)``.

    Does not raise on re-sum mismatch; callers soft-warn using ``max_abs_vs_attn``.
    """
    expected_width = n_query_heads * head_dim
    if concat_layer.shape[-1] != expected_width:
        raise ValueError(
            f"{context}: concat width {concat_layer.shape[-1]} != "
            f"n_query_heads*head_dim={expected_width}"
        )
    # o_proj.weight: [out, in] = [H, n_q*d]
    # head h write: concat[h*d:(h+1)*d] @ weight[:, h*d:(h+1)*d].T
    heads: list[np.ndarray] = []
    total = np.zeros_like(attn_reference, dtype=np.float32)
    for h in range(n_query_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        z_h = concat_layer[sl].astype(np.float32)
        w_h = o_proj_weight[:, sl].astype(np.float32)  # [H, d]
        write = z_h @ w_h.T  # [H]
        heads.append(write)
        total += write

    pre_norm_total = total.copy()

    if post_attn_norm_gamma is not None:
        if post_attn_norm_eps is None:
            raise ValueError(f"{context}: post_attn_norm_eps required with post_attn_norm_gamma")
        if post_attn_norm_gamma.shape != total.shape:
            raise ValueError(
                f"{context}: post_attn_norm_gamma shape {post_attn_norm_gamma.shape} != "
                f"hidden {total.shape}"
            )
        inv_rms = 1.0 / rms(total, float(post_attn_norm_eps))
        scale = post_attn_norm_gamma.astype(np.float32) * np.float32(inv_rms)
        heads = [h_vec * scale for h_vec in heads]
        total = total * scale

    max_abs = float(np.max(np.abs(total - attn_reference.astype(np.float32))))
    return HeadDecomposeResult(
        heads=heads,
        max_abs_vs_attn=max_abs,
        pre_norm_total=pre_norm_total,
    )


def _log_head_resum_failure(
    *,
    example_id: str,
    layer: int,
    position: str,
    contrast_name: str,
    max_abs: float,
    atol: float,
    pre_norm_total: np.ndarray,
    attn: np.ndarray,
    o_layer: np.ndarray | None,
    post_attn_norm_gamma: np.ndarray | None,
    post_attn_norm_eps: float | None,
    applied_post_attn_in_decompose: bool,
) -> None:
    """Warn on head re-sum failure and log W_O(concat)/o/attn diagnostics when possible."""
    msg = (
        f"Head re-sum soft-fail ex={example_id} layer={layer} "
        f"({position}/{contrast_name}): max |sum_heads - attn|={max_abs} > atol={atol}. "
        "Check o_proj slicing / transposition"
        + (" / Gemma post_attention_layernorm" if applied_post_attn_in_decompose else "")
        + "."
    )
    if o_layer is None:
        logging.warning("%s o diagnostics skipped (no embeddings_probability/o).", msg)
        return

    o32 = o_layer.astype(np.float32, copy=False)
    if o32.shape != pre_norm_total.shape:
        logging.warning(
            "%s o diagnostics skipped (o shape %s != W_O(concat) shape %s).",
            msg,
            o32.shape,
            pre_norm_total.shape,
        )
        return

    wo_minus_o = float(np.max(np.abs(pre_norm_total - o32)))
    if applied_post_attn_in_decompose and post_attn_norm_gamma is not None:
        assert post_attn_norm_eps is not None
        o_post = apply_gemma_rmsnorm(o32, post_attn_norm_gamma, float(post_attn_norm_eps))
        post_o_minus_attn = float(np.max(np.abs(o_post - attn.astype(np.float32))))
        logging.warning(
            "%s ||W_O(concat)-o||_inf=%.6g ||post_norm(o)-attn||_inf=%.6g",
            msg,
            wo_minus_o,
            post_o_minus_attn,
        )
    else:
        o_minus_attn = float(np.max(np.abs(o32 - attn.astype(np.float32))))
        logging.warning(
            "%s ||W_O(concat)-o||_inf=%.6g ||o-attn||_inf=%.6g",
            msg,
            wo_minus_o,
            o_minus_attn,
        )


def component_labels_coarse(n_layers: int) -> list[str]:
    labels: list[str] = []
    for l in range(n_layers):
        labels.append(f"L{l}_attn")
        labels.append(f"L{l}_mlp")
    return labels


def component_labels_fine(n_layers: int, n_heads: int) -> list[str]:
    labels: list[str] = []
    for l in range(n_layers):
        for h in range(n_heads):
            labels.append(f"L{l}_H{h}")
        labels.append(f"L{l}_mlp_block")
    return labels


def labels_to_heatmap_matrix(
    labels: list[str],
    values: np.ndarray,
    *,
    granularity: str,
    n_layers: int,
    n_heads: int,
) -> tuple[np.ndarray, list[str]]:
    """Map flat component vector to [n_layers, n_cols] heatmap (+ col labels)."""
    by_label = {lab: float(values[i]) for i, lab in enumerate(labels)}
    if granularity == "coarse":
        col_labels = ["attn", "mlp"]
        mat = np.zeros((n_layers, 2), dtype=np.float32)
        for l in range(n_layers):
            mat[l, 0] = by_label[f"L{l}_attn"]
            mat[l, 1] = by_label[f"L{l}_mlp"]
        return mat, col_labels
    if granularity == "fine":
        col_labels = [f"H{h}" for h in range(n_heads)] + ["mlp_block"]
        mat = np.zeros((n_layers, n_heads + 1), dtype=np.float32)
        for l in range(n_layers):
            for h in range(n_heads):
                mat[l, h] = by_label[f"L{l}_H{h}"]
            mat[l, n_heads] = by_label[f"L{l}_mlp_block"]
        return mat, col_labels
    raise ValueError(f"Unknown granularity {granularity!r}")


@dataclass
class ExperimentResult:
    position: str
    contrast_name: str
    granularity: str
    labels: list[str]
    example_ids: list[str]
    attributions: np.ndarray  # [n_ex, n_comp]
    embed_attributions: np.ndarray  # [n_ex]
    true_scores: np.ndarray  # [n_ex]
    completeness_residuals: np.ndarray  # [n_ex]
    head_resum_failure_count: int = 0
    completeness_soft_failure_count: int = 0


def run_dla_for_experiment(
    examples: Sequence[ExampleData],
    *,
    contrast: ContrastSpec,
    granularity: str,
    tensors: ModelTensors,
    head_resum_atol: float,
    completeness_atol: float,
    legacy_gemma_pre_postnorm_h5: bool = False,
) -> ExperimentResult:
    adapter = tensors.adapter
    u_tilde = (tensors.gamma * contrast.u).astype(np.float32)
    n_layers = adapter.n_layers
    n_heads = adapter.n_query_heads

    if granularity == "coarse":
        labels = component_labels_coarse(n_layers)
    elif granularity == "fine":
        labels = component_labels_fine(n_layers, n_heads)
    else:
        raise ValueError(f"Unknown granularity {granularity!r}")

    if legacy_gemma_pre_postnorm_h5:
        if tensors.post_attn_norm_gammas is None or tensors.post_ff_norm_gammas is None:
            raise ValueError(
                "legacy_gemma_pre_postnorm_h5 requires post_attn_norm_gammas and "
                "post_ff_norm_gammas to be loaded."
            )
        if len(tensors.post_attn_norm_gammas) != n_layers:
            raise ValueError(
                f"Expected {n_layers} post_attn_norm_gammas, got "
                f"{len(tensors.post_attn_norm_gammas)}."
            )
        if len(tensors.post_ff_norm_gammas) != n_layers:
            raise ValueError(
                f"Expected {n_layers} post_ff_norm_gammas, got "
                f"{len(tensors.post_ff_norm_gammas)}."
            )

    n_ex = len(examples)
    n_comp = len(labels)
    attrs = np.zeros((n_ex, n_comp), dtype=np.float32)
    embed_attrs = np.zeros(n_ex, dtype=np.float32)
    true_scores = np.zeros(n_ex, dtype=np.float32)
    residuals = np.zeros(n_ex, dtype=np.float32)
    example_ids: list[str] = []
    head_resum_failure_count = 0
    completeness_soft_failure_count = 0

    for ei, ex in enumerate(tqdm(examples, desc=f"{contrast.position}/{contrast.name}/{granularity}")):
        example_ids.append(ex.example_id)
        span_len = len(ex.res)
        pos_idx = position_index(
            contrast.position, span_len, span_index=contrast.span_index
        )

        res_tok = ex.res[pos_idx]  # [n_res_layers, H]
        attn_tok = ex.attn[pos_idx]  # [n_layers, H]
        mlp_tok = ex.mlp[pos_idx]
        concat_tok = None
        o_tok = None
        if granularity == "fine":
            if len(tensors.o_proj_weights) != n_layers:
                raise ValueError(
                    f"Fine granularity requires o_proj weights for all {n_layers} layers, "
                    f"but only {len(tensors.o_proj_weights)} were loaded."
                )
            if tensors.post_attn_norm_gammas is not None and len(
                tensors.post_attn_norm_gammas
            ) != n_layers:
                raise ValueError(
                    f"Fine granularity Gemma path requires post_attention_layernorm "
                    f"gains for all {n_layers} layers, but only "
                    f"{len(tensors.post_attn_norm_gammas)} were loaded."
                )
            if ex.concat is not None:
                concat_tok = ex.concat[pos_idx]
            else:
                if ex.o is None:
                    raise ValueError(
                        f"Example {ex.example_id}: fine granularity requires concat "
                        "embeddings, or o embeddings to recover concat, but neither "
                        "were loaded."
                    )
                n_inv = (
                    0
                    if tensors.o_proj_inv_weights is None
                    else len(tensors.o_proj_inv_weights)
                )
                if n_inv != n_layers:
                    raise ValueError(
                        f"Fine granularity concat-from-o fallback requires inv(W_O) "
                        f"for all {n_layers} layers, but only {n_inv} were loaded."
                    )
            if ex.o is not None:
                if ex.o[pos_idx].shape[0] != n_layers:
                    raise ValueError(
                        f"Example {ex.example_id}: o layers {ex.o[pos_idx].shape[0]} != {n_layers}"
                    )
                o_tok = ex.o[pos_idx]

        if attn_tok.shape[0] != n_layers or mlp_tok.shape[0] != n_layers:
            raise ValueError(
                f"Example {ex.example_id}: attn/mlp layers "
                f"{attn_tok.shape[0]}/{mlp_tok.shape[0]} != adapter n_layers={n_layers}"
            )
        if concat_tok is not None and concat_tok.shape[0] != n_layers:
            raise ValueError(
                f"Example {ex.example_id}: concat layers {concat_tok.shape[0]} != {n_layers}"
            )

        # Final residual: last entry of res stack (embedding + all layers).
        h = res_tok[-1].astype(np.float32)
        if h.shape[0] != adapter.hidden_size:
            raise ValueError(
                f"Example {ex.example_id}: hidden dim {h.shape[0]} != {adapter.hidden_size}"
            )

        sigma = rms(h, adapter.rms_norm_eps)
        if sigma <= 0:
            raise ValueError(f"Example {ex.example_id}: non-positive RMS sigma={sigma}")

        block_sum = np.zeros_like(h, dtype=np.float32)
        row = np.zeros(n_comp, dtype=np.float32)

        if granularity == "coarse":
            for l in range(n_layers):
                a = attn_tok[l].astype(np.float32)
                m = mlp_tok[l].astype(np.float32)
                if legacy_gemma_pre_postnorm_h5:
                    assert tensors.post_attn_norm_gammas is not None
                    assert tensors.post_ff_norm_gammas is not None
                    a = apply_gemma_rmsnorm(
                        a, tensors.post_attn_norm_gammas[l], adapter.rms_norm_eps
                    )
                    m = apply_gemma_rmsnorm(
                        m, tensors.post_ff_norm_gammas[l], adapter.rms_norm_eps
                    )
                row[2 * l] = attr_vec(a, u_tilde, sigma)
                row[2 * l + 1] = attr_vec(m, u_tilde, sigma)
                block_sum += a + m
        else:
            col = 0
            for l in range(n_layers):
                a = attn_tok[l].astype(np.float32)
                m = mlp_tok[l].astype(np.float32)
                o_layer = None if o_tok is None else o_tok[l].astype(np.float32)
                if concat_tok is not None:
                    concat_layer = concat_tok[l].astype(np.float32)
                else:
                    if o_layer is None:
                        raise ValueError(
                            f"Example {ex.example_id}: fine granularity requires concat "
                            "or o embeddings to recover concat, but neither were loaded."
                        )
                    assert tensors.o_proj_inv_weights is not None
                    concat_layer = recover_concat_from_o(
                        o_layer, tensors.o_proj_inv_weights[l]
                    )
                if legacy_gemma_pre_postnorm_h5:
                    # H5 attn is pre-norm: re-sum check without post-attn norm, then
                    # scale heads to residual writes; MLP needs post-FF norm.
                    decomp = decompose_heads(
                        concat_layer,
                        tensors.o_proj_weights[l],
                        n_query_heads=n_heads,
                        head_dim=adapter.head_dim,
                        attn_reference=a,
                        context=f"ex={ex.example_id} layer={l}",
                        post_attn_norm_gamma=None,
                        post_attn_norm_eps=None,
                    )
                    if decomp.max_abs_vs_attn > head_resum_atol:
                        head_resum_failure_count += 1
                        assert tensors.post_attn_norm_gammas is not None
                        _log_head_resum_failure(
                            example_id=ex.example_id,
                            layer=l,
                            position=contrast.position,
                            contrast_name=contrast.name,
                            max_abs=decomp.max_abs_vs_attn,
                            atol=head_resum_atol,
                            pre_norm_total=decomp.pre_norm_total,
                            attn=a,
                            o_layer=o_layer,
                            post_attn_norm_gamma=tensors.post_attn_norm_gammas[l],
                            post_attn_norm_eps=adapter.rms_norm_eps,
                            applied_post_attn_in_decompose=False,
                        )
                    assert tensors.post_attn_norm_gammas is not None
                    assert tensors.post_ff_norm_gammas is not None
                    heads = _scale_heads_with_shared_post_attn_norm(
                        decomp.heads,
                        tensors.post_attn_norm_gammas[l],
                        adapter.rms_norm_eps,
                    )
                    m = apply_gemma_rmsnorm(
                        m, tensors.post_ff_norm_gammas[l], adapter.rms_norm_eps
                    )
                    a_write = np.zeros_like(a, dtype=np.float32)
                    for h_vec in heads:
                        a_write += h_vec
                else:
                    post_gamma = (
                        None
                        if tensors.post_attn_norm_gammas is None
                        else tensors.post_attn_norm_gammas[l]
                    )
                    post_eps = (
                        adapter.rms_norm_eps
                        if tensors.post_attn_norm_gammas is not None
                        else None
                    )
                    decomp = decompose_heads(
                        concat_layer,
                        tensors.o_proj_weights[l],
                        n_query_heads=n_heads,
                        head_dim=adapter.head_dim,
                        attn_reference=a,
                        context=f"ex={ex.example_id} layer={l}",
                        post_attn_norm_gamma=post_gamma,
                        post_attn_norm_eps=post_eps,
                    )
                    if decomp.max_abs_vs_attn > head_resum_atol:
                        head_resum_failure_count += 1
                        _log_head_resum_failure(
                            example_id=ex.example_id,
                            layer=l,
                            position=contrast.position,
                            contrast_name=contrast.name,
                            max_abs=decomp.max_abs_vs_attn,
                            atol=head_resum_atol,
                            pre_norm_total=decomp.pre_norm_total,
                            attn=a,
                            o_layer=o_layer,
                            post_attn_norm_gamma=post_gamma,
                            post_attn_norm_eps=post_eps,
                            applied_post_attn_in_decompose=post_gamma is not None,
                        )
                    heads = decomp.heads
                    a_write = a
                for h_vec in heads:
                    row[col] = attr_vec(h_vec, u_tilde, sigma)
                    col += 1
                row[col] = attr_vec(m, u_tilde, sigma)
                col += 1
                block_sum += a_write + m

        h_embed = h - block_sum
        embed_attr = attr_vec(h_embed, u_tilde, sigma)
        score = true_contrast_score(
            h,
            tensors.gamma,
            tensors.w_u,
            adapter.rms_norm_eps,
            contrast.pos_token_ids,
            contrast.neg_token_ids,
        )
        recon = float(np.sum(row) + embed_attr)
        residual = recon - score
        if abs(residual) > completeness_atol:
            completeness_soft_failure_count += 1
            logging.warning(
                "Completeness soft-fail for example %s "
                "(%s/%s/%s): sum(components)+embed=%.6f, true_score=%.6f, "
                "|residual|=%.6g > atol=%s.",
                ex.example_id,
                contrast.position,
                contrast.name,
                granularity,
                recon,
                score,
                abs(residual),
                completeness_atol,
            )

        attrs[ei] = row
        embed_attrs[ei] = embed_attr
        true_scores[ei] = score
        residuals[ei] = residual

    if granularity == "fine" or head_resum_failure_count or completeness_soft_failure_count:
        logging.info(
            "%s/%s/%s soft-failure counts: head_resum=%d completeness=%d (n_examples=%d)",
            contrast.position,
            contrast.name,
            granularity,
            head_resum_failure_count,
            completeness_soft_failure_count,
            n_ex,
        )

    return ExperimentResult(
        position=contrast.position,
        contrast_name=contrast.name,
        granularity=granularity,
        labels=labels,
        example_ids=example_ids,
        attributions=attrs,
        embed_attributions=embed_attrs,
        true_scores=true_scores,
        completeness_residuals=residuals,
        head_resum_failure_count=head_resum_failure_count,
        completeness_soft_failure_count=completeness_soft_failure_count,
    )


# ---------------------------------------------------------------------------
# Aggregation and plots
# ---------------------------------------------------------------------------


@dataclass
class ComponentStats:
    label: str
    mean: float
    sem: float
    median: float
    top10_fraction: float
    sign_consistency: float


def aggregate_component_stats(attrs: np.ndarray, labels: Sequence[str]) -> list[ComponentStats]:
    n_ex, n_comp = attrs.shape
    top_k = min(10, n_comp)
    # per-example top-10 by |attr|
    top10_counts = np.zeros(n_comp, dtype=np.float64)
    for i in range(n_ex):
        order = np.argsort(-np.abs(attrs[i]))
        for j in order[:top_k]:
            top10_counts[j] += 1.0

    stats: list[ComponentStats] = []
    for c, lab in enumerate(labels):
        col = attrs[:, c]
        mean = float(np.mean(col))
        sem = float(np.std(col, ddof=1) / np.sqrt(n_ex)) if n_ex > 1 else 0.0
        median = float(np.median(col))
        if mean == 0.0:
            sign_consistency = float(np.mean(col == 0.0))
        else:
            sign_consistency = float(np.mean(np.sign(col) == np.sign(mean)))
        stats.append(
            ComponentStats(
                label=lab,
                mean=mean,
                sem=sem,
                median=median,
                top10_fraction=float(top10_counts[c] / n_ex),
                sign_consistency=sign_consistency,
            )
        )
    return stats


def rank_by_mean(stats: Sequence[ComponentStats]) -> list[ComponentStats]:
    return sorted(stats, key=lambda s: s.mean, reverse=True)


def rank_by_value(labels: Sequence[str], values: np.ndarray) -> list[tuple[str, float]]:
    pairs = [(lab, float(values[i])) for i, lab in enumerate(labels)]
    return sorted(pairs, key=lambda p: p[1], reverse=True)


def write_diverging_heatmap(
    path: Path,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    rounding_dp: int,
) -> None:
    """Blue=positive, red=negative; signs normalised separately; exact 0 uncoloured."""
    n_rows, n_cols = matrix.shape
    pos_max = float(np.max(matrix)) if matrix.size else 0.0
    neg_min = float(np.min(matrix)) if matrix.size else 0.0
    pos_max = max(pos_max, 0.0)
    neg_min = min(neg_min, 0.0)

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            v = float(matrix[r, c])
            if v == 0.0:
                continue
            if v > 0.0:
                alpha = (v / pos_max) if pos_max > 0 else 0.0
                rgba[r, c] = (0.10, 0.35, 1.0, alpha)
            else:
                alpha = (abs(v) / abs(neg_min)) if neg_min < 0 else 0.0
                rgba[r, c] = (1.0, 0.15, 0.15, alpha)

    fig_w = max(8.0, 0.55 * n_cols + 2.0)
    fig_h = max(6.0, 0.28 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    fmt = f"{{:.{rounding_dp}f}}"
    for r in range(n_rows):
        for c in range(n_cols):
            ax.text(
                c,
                r,
                fmt.format(float(matrix[r, c])),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(row_labels))
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(list(col_labels), rotation=45, ha="left")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Component")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_bar_chart(
    path: Path,
    ordered_labels: Sequence[str],
    ordered_values: Sequence[float],
    *,
    title: str,
    top_k: int,
    errors: Sequence[float] | None = None,
) -> None:
    n = len(ordered_labels)
    if n == 0:
        return
    k = min(top_k, n)
    if 2 * k >= n:
        sel_idx = list(range(n))
        omitted = 0
    else:
        sel_idx = list(range(k)) + list(range(n - k, n))
        omitted = n - len(sel_idx)

    labs = [ordered_labels[i] for i in sel_idx]
    vals = [ordered_values[i] for i in sel_idx]
    errs = [errors[i] for i in sel_idx] if errors is not None else None
    colors = ["#1f4fd6" if v >= 0 else "#d62728" for v in vals]

    fig_w = max(10.0, 0.35 * len(labs) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    x = np.arange(len(labs))
    ax.bar(x, vals, color=colors, yerr=errs, capsize=2 if errs else 0, ecolor="black")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Attribution")
    full_title = title
    if omitted > 0:
        full_title = f"{title} (omitted {omitted} middle components)"
    ax.set_title(full_title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_distribution_plot(
    path: Path,
    attrs: np.ndarray,
    labels: Sequence[str],
    stats: Sequence[ComponentStats],
    *,
    top_k: int,
    title: str,
) -> None:
    ranked = rank_by_mean(stats)
    k = min(top_k, len(ranked))
    chosen = ranked[:k]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    data = [attrs[:, label_to_idx[s.label]] for s in chosen]
    names = [s.label for s in chosen]

    fig_w = max(10.0, 0.4 * k + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4c72b0")
        body.set_alpha(0.7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Per-example attribution")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


_FINE_HEAD_LABEL_RE = re.compile(r"^L(\d+)_H(\d+)$")
_FINE_MLP_LABEL_RE = re.compile(r"^L(\d+)_mlp_block$")


def _dla_label_to_ablate_unit(label: str) -> Optional[str]:
    """Map fine DLA labels to ablation-unit tokens (a<layer>.h<head> / m<layer>)."""
    m = _FINE_HEAD_LABEL_RE.fullmatch(label)
    if m:
        return f"a{m.group(1)}.h{m.group(2)}"
    m = _FINE_MLP_LABEL_RE.fullmatch(label)
    if m:
        return f"m{m.group(1)}"
    return None


def _ablate_summary_lines(labels: Sequence[str], *, k: int = 10) -> Optional[list[str]]:
    """Return top_10/bottom_10 lines if every label is a fine DLA component."""
    units = [_dla_label_to_ablate_unit(lab) for lab in labels]
    if not units or any(u is None for u in units):
        return None
    n = min(k, len(units))
    return [
        f"top_10: {','.join(units[:n])}",
        f"bottom_10: {','.join(units[-n:])}",
    ]


def write_ranked_mean_txt(
    path: Path,
    ranked: Sequence[ComponentStats],
    *,
    embed_mean: float,
    embed_sem: float,
) -> None:
    lines = [
        "Ranked components (most positive → most negative)",
        "format: label  mean  SEM  median  top10_frac  sign_consistency",
        "",
        f"embedding_contribution  mean={embed_mean:.6g}  SEM={embed_sem:.6g}",
        "",
    ]
    for s in ranked:
        lines.append(
            f"{s.label}\t{s.mean:.6g}\t{s.sem:.6g}\t{s.median:.6g}\t"
            f"{s.top10_fraction:.4f}\t{s.sign_consistency:.4f}"
        )
    summary = _ablate_summary_lines([s.label for s in ranked])
    if summary:
        lines.append("")
        lines.extend(summary)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ranked_example_txt(
    path: Path,
    ranked_pairs: Sequence[tuple[str, float]],
    *,
    example_id: str,
    response: str,
    confidence: float,
    embed_attr: float,
    true_score: float,
) -> None:
    lines = [
        f"example_id={example_id}",
        f"verbalised_confidence={confidence}",
        f"response={response}",
        f"embedding_attribution={embed_attr:.6g}",
        f"true_contrast_score={true_score:.6g}",
        "",
        "Ranked components (most positive → most negative)",
        "format: label  attribution",
        "",
    ]
    for lab, val in ranked_pairs:
        lines.append(f"{lab}\t{val:.6g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_attributions_h5(path: Path, result: ExperimentResult) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("attributions", data=result.attributions)
        f.create_dataset("embed_attributions", data=result.embed_attributions)
        f.create_dataset("true_scores", data=result.true_scores)
        f.create_dataset("completeness_residuals", data=result.completeness_residuals)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("component_labels", data=np.asarray(result.labels, dtype=dt))
        f.create_dataset("example_ids", data=np.asarray(result.example_ids, dtype=dt))
        f.attrs["position"] = result.position
        f.attrs["contrast_name"] = result.contrast_name
        f.attrs["granularity"] = result.granularity
        f.attrs["head_resum_failure_count"] = int(result.head_resum_failure_count)
        f.attrs["completeness_soft_failure_count"] = int(
            result.completeness_soft_failure_count
        )


def write_experiment_outputs(
    exp_dir: Path,
    result: ExperimentResult,
    examples_by_id: dict[str, ExampleData],
    individual_ids: Sequence[str],
    *,
    n_layers: int,
    n_heads: int,
    bar_chart_top_k: int,
    rounding_dp: int,
    method_label: str = "DLA",
) -> dict:
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_attributions_h5(exp_dir / "attributions.h5", result)

    stats = aggregate_component_stats(result.attributions, result.labels)
    ranked = rank_by_mean(stats)
    means = np.array([s.mean for s in stats], dtype=np.float32)
    sems_by_label = {s.label: s.sem for s in stats}

    # Distribution plot (top-k by |mean|)
    by_abs = sorted(stats, key=lambda s: abs(s.mean), reverse=True)
    top_abs_labels = {s.label for s in by_abs[: max(bar_chart_top_k, 1)]}
    # Re-order those by mean for the violin
    dist_stats = [s for s in ranked if s.label in top_abs_labels]
    write_distribution_plot(
        exp_dir / "distribution_topk.png",
        result.attributions,
        result.labels,
        dist_stats if dist_stats else ranked[:bar_chart_top_k],
        top_k=bar_chart_top_k,
        title=(
            f"{method_label} {result.position} / {result.contrast_name} / {result.granularity}: "
            f"top-{bar_chart_top_k} by |mean|, ordered by mean"
        ),
    )

    # mean/
    mean_dir = exp_dir / "mean"
    mean_dir.mkdir(parents=True, exist_ok=True)
    mean_mat, col_labels = labels_to_heatmap_matrix(
        result.labels,
        means,
        granularity=result.granularity,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    write_diverging_heatmap(
        mean_dir / "heatmap.png",
        mean_mat,
        row_labels=[str(l) for l in range(n_layers)],
        col_labels=col_labels,
        title=(
            f"Mean {method_label} — {result.position} / {result.contrast_name} / {result.granularity} "
            f"(n={len(result.example_ids)}; ± normalised separately)"
        ),
        rounding_dp=rounding_dp,
    )
    embed_mean = float(np.mean(result.embed_attributions))
    embed_sem = (
        float(np.std(result.embed_attributions, ddof=1) / np.sqrt(len(result.embed_attributions)))
        if len(result.embed_attributions) > 1
        else 0.0
    )
    write_ranked_mean_txt(
        mean_dir / "ranked.txt", ranked, embed_mean=embed_mean, embed_sem=embed_sem
    )
    ordered_labels = [s.label for s in ranked]
    ordered_values = [s.mean for s in ranked]
    ordered_errors = [sems_by_label[lab] for lab in ordered_labels]
    write_bar_chart(
        mean_dir / "bar_chart.png",
        ordered_labels,
        ordered_values,
        title=f"Mean {method_label} — {result.position}/{result.contrast_name}/{result.granularity}",
        top_k=bar_chart_top_k,
        errors=ordered_errors,
    )

    # individual examples
    id_to_row = {eid: i for i, eid in enumerate(result.example_ids)}
    for eid in individual_ids:
        if eid not in id_to_row:
            raise ValueError(f"Individual example {eid} not in experiment example set.")
        row_i = id_to_row[eid]
        ex = examples_by_id[eid]
        vals = result.attributions[row_i]
        ex_dir = exp_dir / f"example_{eid}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        mat, col_labels = labels_to_heatmap_matrix(
            result.labels,
            vals,
            granularity=result.granularity,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        write_diverging_heatmap(
            ex_dir / "heatmap.png",
            mat,
            row_labels=[str(l) for l in range(n_layers)],
            col_labels=col_labels,
            title=(
                f"Example {eid} {method_label} — {result.position}/{result.contrast_name}/{result.granularity} "
                "(normalised to this example's own min/max; ± separately)"
            ),
            rounding_dp=rounding_dp,
        )
        ranked_pairs = rank_by_value(result.labels, vals)
        write_ranked_example_txt(
            ex_dir / "ranked.txt",
            ranked_pairs,
            example_id=eid,
            response=ex.response,
            confidence=ex.verbalised_confidence,
            embed_attr=float(result.embed_attributions[row_i]),
            true_score=float(result.true_scores[row_i]),
        )
        write_bar_chart(
            ex_dir / "bar_chart.png",
            [p[0] for p in ranked_pairs],
            [p[1] for p in ranked_pairs],
            title=f"Example {eid} {method_label} — {result.position}/{result.contrast_name}/{result.granularity}",
            top_k=bar_chart_top_k,
            errors=None,
        )

    return {
        "n_examples": len(result.example_ids),
        "n_components": len(result.labels),
        "completeness_residual_mean_abs": float(np.mean(np.abs(result.completeness_residuals))),
        "completeness_residual_max_abs": float(np.max(np.abs(result.completeness_residuals))),
        "head_resum_failure_count": int(result.head_resum_failure_count),
        "completeness_soft_failure_count": int(result.completeness_soft_failure_count),
        "embed_mean": embed_mean,
    }


# ---------------------------------------------------------------------------
# High-confidence minus low-confidence ranking
# ---------------------------------------------------------------------------


@dataclass
class HighMinusLowRow:
    label: str
    diff: float
    high_mean: float
    low_mean: float
    high_sem: float
    low_sem: float

    @property
    def diff_sem(self) -> float:
        return float(np.sqrt(self.high_sem**2 + self.low_sem**2))


_EMBED_MEAN_RE = re.compile(
    r"embedding_contribution\s+mean=([^\s]+)\s+SEM=([^\s]+)"
)
_COARSE_ATTN_RE = re.compile(r"^L(\d+)_attn$")
_COARSE_MLP_RE = re.compile(r"^L(\d+)_mlp$")


def _sem_from_column(col: np.ndarray) -> float:
    n = int(col.shape[0])
    if n <= 1:
        return 0.0
    return float(np.std(col, ddof=1) / np.sqrt(n))


def infer_layout_from_labels(labels: Sequence[str]) -> tuple[str, int, int]:
    """Return (granularity, n_layers, n_heads) from DLA component labels."""
    labs = list(labels)
    if not labs:
        raise ValueError("Cannot infer layout from empty component label list.")
    if any(_COARSE_ATTN_RE.fullmatch(lab) for lab in labs):
        layers = []
        for lab in labs:
            m = _COARSE_ATTN_RE.fullmatch(lab)
            if m:
                layers.append(int(m.group(1)))
            m = _COARSE_MLP_RE.fullmatch(lab)
            if m:
                layers.append(int(m.group(1)))
        n_layers = max(layers) + 1 if layers else 0
        return "coarse", n_layers, 0
    n_heads = sum(1 for lab in labs if _FINE_HEAD_LABEL_RE.fullmatch(lab) and lab.startswith("L0_"))
    n_mlp = sum(1 for lab in labs if _FINE_MLP_LABEL_RE.fullmatch(lab))
    if n_mlp < 1:
        raise ValueError(f"Could not infer fine layout from labels (first={labs[0]!r}).")
    n_heads = n_heads if n_heads > 0 else (len(labs) // n_mlp) - 1
    return "fine", n_mlp, n_heads


def _decode_h5_label_array(raw) -> list[str]:
    out: list[str] = []
    for x in np.asarray(raw).tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def load_mean_vector_from_experiment_dir(
    exp_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, float, float, int]:
    """Load (labels, means, sems, embed_mean, embed_sem, n_examples).

    Prefers ``attributions.h5``; falls back to ``mean/ranked.txt``.
    """
    h5_path = exp_dir / "attributions.h5"
    if h5_path.is_file():
        with h5py.File(h5_path, "r") as f:
            attrs = np.asarray(f["attributions"], dtype=np.float64)
            labels = _decode_h5_label_array(f["component_labels"][()])
            embed = np.asarray(f["embed_attributions"], dtype=np.float64)
        if attrs.ndim != 2 or attrs.shape[1] != len(labels):
            raise ValueError(
                f"{h5_path}: attributions shape {attrs.shape} != n_labels={len(labels)}"
            )
        n = int(attrs.shape[0])
        means = np.mean(attrs, axis=0).astype(np.float32)
        sems = np.array(
            [_sem_from_column(attrs[:, c]) for c in range(attrs.shape[1])],
            dtype=np.float32,
        )
        embed_mean = float(np.mean(embed)) if embed.size else 0.0
        embed_sem = _sem_from_column(embed) if embed.size else 0.0
        return labels, means, sems, embed_mean, embed_sem, n

    ranked_path = exp_dir / "mean" / "ranked.txt"
    if not ranked_path.is_file():
        raise FileNotFoundError(
            f"{exp_dir} has neither attributions.h5 nor mean/ranked.txt"
        )
    labels: list[str] = []
    means_l: list[float] = []
    sems_l: list[float] = []
    embed_mean = 0.0
    embed_sem = 0.0
    for line in ranked_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Ranked") or stripped.startswith("format:"):
            continue
        if stripped.startswith("top_10:") or stripped.startswith("bottom_10:"):
            continue
        embed_m = _EMBED_MEAN_RE.search(stripped)
        if embed_m:
            embed_mean = float(embed_m.group(1))
            embed_sem = float(embed_m.group(2))
            continue
        parts = stripped.split("\t")
        if len(parts) < 3:
            continue
        labels.append(parts[0])
        means_l.append(float(parts[1]))
        sems_l.append(float(parts[2]))
    if not labels:
        raise ValueError(f"No component rows parsed from {ranked_path}")
    return (
        labels,
        np.asarray(means_l, dtype=np.float32),
        np.asarray(sems_l, dtype=np.float32),
        embed_mean,
        embed_sem,
        0,
    )


def high_minus_low_rows(
    labels: Sequence[str],
    high_means: np.ndarray,
    low_means: np.ndarray,
    high_sems: np.ndarray,
    low_sems: np.ndarray,
) -> list[HighMinusLowRow]:
    if not (len(labels) == len(high_means) == len(low_means) == len(high_sems) == len(low_sems)):
        raise ValueError("high/low mean vectors and labels must have the same length.")
    rows = [
        HighMinusLowRow(
            label=str(lab),
            diff=float(high_means[i] - low_means[i]),
            high_mean=float(high_means[i]),
            low_mean=float(low_means[i]),
            high_sem=float(high_sems[i]),
            low_sem=float(low_sems[i]),
        )
        for i, lab in enumerate(labels)
    ]
    return sorted(rows, key=lambda r: r.diff, reverse=True)


def write_ranked_high_minus_low_txt(
    path: Path,
    ranked: Sequence[HighMinusLowRow],
    *,
    embed_diff: float,
    embed_high: float,
    embed_low: float,
    n_high: int,
    n_low: int,
) -> None:
    lines = [
        "Ranked components (high_conf mean - low_conf mean; most positive → most negative)",
        "format: label  diff  high_mean  low_mean  high_sem  low_sem  diff_sem",
        "",
        f"n_high={n_high}  n_low={n_low}",
        (
            f"embedding_contribution  diff={embed_diff:.6g}  "
            f"high_mean={embed_high:.6g}  low_mean={embed_low:.6g}"
        ),
        "",
    ]
    for r in ranked:
        lines.append(
            f"{r.label}\t{r.diff:.6g}\t{r.high_mean:.6g}\t{r.low_mean:.6g}\t"
            f"{r.high_sem:.6g}\t{r.low_sem:.6g}\t{r.diff_sem:.6g}"
        )
    summary = _ablate_summary_lines([r.label for r in ranked])
    if summary:
        lines.append("")
        lines.extend(summary)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_high_minus_low_outputs(
    exp_dir: Path,
    *,
    labels: Sequence[str],
    high_means: np.ndarray,
    low_means: np.ndarray,
    high_sems: np.ndarray,
    low_sems: np.ndarray,
    position: str,
    contrast_name: str,
    granularity: str,
    n_layers: int,
    n_heads: int,
    bar_chart_top_k: int,
    rounding_dp: int,
    embed_high: float = 0.0,
    embed_low: float = 0.0,
    n_high: int = 0,
    n_low: int = 0,
    method_label: str = "DLA",
) -> list[HighMinusLowRow]:
    """Write signed high-minus-low ranking, heatmap, and bar chart."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    mean_dir = exp_dir / "mean"
    mean_dir.mkdir(parents=True, exist_ok=True)

    diffs = np.asarray(high_means, dtype=np.float32) - np.asarray(low_means, dtype=np.float32)
    ranked = high_minus_low_rows(labels, high_means, low_means, high_sems, low_sems)
    write_ranked_high_minus_low_txt(
        mean_dir / "ranked.txt",
        ranked,
        embed_diff=float(embed_high - embed_low),
        embed_high=embed_high,
        embed_low=embed_low,
        n_high=n_high,
        n_low=n_low,
    )
    mat, col_labels = labels_to_heatmap_matrix(
        list(labels),
        diffs,
        granularity=granularity,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    write_diverging_heatmap(
        mean_dir / "heatmap.png",
        mat,
        row_labels=[str(l) for l in range(n_layers)],
        col_labels=col_labels,
        title=(
            f"High−low {method_label} — {position} / {contrast_name} / {granularity} "
            f"(n_high={n_high}, n_low={n_low}; ± normalised separately)"
        ),
        rounding_dp=rounding_dp,
    )
    write_bar_chart(
        mean_dir / "bar_chart.png",
        [r.label for r in ranked],
        [r.diff for r in ranked],
        title=(
            f"High−low {method_label} — {position}/{contrast_name}/{granularity}"
        ),
        top_k=bar_chart_top_k,
        errors=[r.diff_sem for r in ranked],
    )
    return ranked


def write_high_minus_low_from_results(
    exp_dir: Path,
    high: ExperimentResult,
    low: ExperimentResult,
    *,
    n_layers: int,
    n_heads: int,
    bar_chart_top_k: int,
    rounding_dp: int,
    method_label: str = "DLA",
) -> list[HighMinusLowRow]:
    if high.labels != low.labels:
        raise ValueError(
            f"High/low label mismatch for {high.position}/{high.contrast_name}/"
            f"{high.granularity}: {high.labels[:3]} vs {low.labels[:3]}"
        )
    high_stats = aggregate_component_stats(high.attributions, high.labels)
    low_stats = aggregate_component_stats(low.attributions, low.labels)
    high_means = np.array([s.mean for s in high_stats], dtype=np.float32)
    low_means = np.array([s.mean for s in low_stats], dtype=np.float32)
    high_sems = np.array([s.sem for s in high_stats], dtype=np.float32)
    low_sems = np.array([s.sem for s in low_stats], dtype=np.float32)
    embed_high = float(np.mean(high.embed_attributions)) if high.embed_attributions.size else 0.0
    embed_low = float(np.mean(low.embed_attributions)) if low.embed_attributions.size else 0.0
    return write_high_minus_low_outputs(
        exp_dir,
        labels=high.labels,
        high_means=high_means,
        low_means=low_means,
        high_sems=high_sems,
        low_sems=low_sems,
        position=high.position,
        contrast_name=high.contrast_name,
        granularity=high.granularity,
        n_layers=n_layers,
        n_heads=n_heads,
        bar_chart_top_k=bar_chart_top_k,
        rounding_dp=rounding_dp,
        embed_high=embed_high,
        embed_low=embed_low,
        n_high=len(high.example_ids),
        n_low=len(low.example_ids),
        method_label=method_label,
    )


def _high_minus_low_leaf_name(high_name: str) -> str | None:
    """Map ``...__high_conf`` directory name to ``...__high_minus_low``."""
    if high_name.endswith("__high_conf"):
        return high_name[: -len("__high_conf")] + "__high_minus_low"
    return None


def backfill_high_minus_low_results(
    results_root: Path,
    *,
    bar_chart_top_k: int = 25,
    rounding_dp: int = ANNOTATION_ROUNDING_DP,
    method_label: str = "DLA",
) -> int:
    """Write ``__high_minus_low`` siblings for every high/low pair under ``results_root``."""
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    n_written = 0
    for high_dir in sorted(results_root.rglob("*__high_conf")):
        if not high_dir.is_dir():
            continue
        low_dir = high_dir.parent / (high_dir.name[: -len("__high_conf")] + "__low_conf")
        if not low_dir.is_dir():
            logging.warning("Skipping %s: matching __low_conf dir not found.", high_dir)
            continue
        leaf = _high_minus_low_leaf_name(high_dir.name)
        if leaf is None:
            continue
        out_dir = high_dir.parent / leaf
        try:
            h_labels, h_means, h_sems, h_embed, _h_embed_sem, n_high = (
                load_mean_vector_from_experiment_dir(high_dir)
            )
            l_labels, l_means, l_sems, l_embed, _l_embed_sem, n_low = (
                load_mean_vector_from_experiment_dir(low_dir)
            )
        except (OSError, ValueError, FileNotFoundError) as exc:
            logging.warning("Skipping %s: failed to load means (%s).", high_dir, exc)
            continue
        if h_labels != l_labels:
            if set(h_labels) != set(l_labels):
                logging.warning("Skipping %s: high/low label sets differ.", high_dir)
                continue
            l_index = {lab: i for i, lab in enumerate(l_labels)}
            l_means = np.array([l_means[l_index[lab]] for lab in h_labels], dtype=np.float32)
            l_sems = np.array([l_sems[l_index[lab]] for lab in h_labels], dtype=np.float32)
        gran, n_layers, n_heads = infer_layout_from_labels(h_labels)
        parent_name = high_dir.parent.name
        position = parent_name if parent_name in {"pre_period", "post_period"} else parent_name
        contrast_part = high_dir.name[: -len(f"__{gran}__high_conf")]
        if not contrast_part:
            contrast_part = high_dir.name.split("__")[0]
        write_high_minus_low_outputs(
            out_dir,
            labels=h_labels,
            high_means=h_means,
            low_means=l_means,
            high_sems=h_sems,
            low_sems=l_sems,
            position=position,
            contrast_name=contrast_part,
            granularity=gran,
            n_layers=n_layers,
            n_heads=n_heads,
            bar_chart_top_k=bar_chart_top_k,
            rounding_dp=rounding_dp,
            embed_high=h_embed,
            embed_low=l_embed,
            n_high=n_high,
            n_low=n_low,
            method_label=method_label,
        )
        n_written += 1
        logging.info("Wrote high-minus-low ranking to %s", out_dir)
    return n_written


# ---------------------------------------------------------------------------
# Run scaffolding / config
# ---------------------------------------------------------------------------


def resolve_run_root(cli_output_dir: Optional[str]) -> Path:
    if cli_output_dir:
        root = Path(cli_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
    base = SCRIPT_DIR / "results"
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        d for d in os.listdir(base) if (base / d).is_dir() and d.isdigit()
    ]
    run_idx = max((int(d) for d in existing), default=0) + 1
    root = base / str(run_idx)
    root.mkdir(parents=True, exist_ok=True)
    return root


def attach_output_log(run_root: Path) -> Path:
    log_path = run_root / "output.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
        force=True,
    )
    return log_path


def write_readme(path: Path) -> None:
    path.write_text(
        "# Direct Logit Attribution results\n\n"
        "DLA attributes logit contrasts only along **direct** paths from residual-stream "
        "components to the unembedding. A component that acts by changing a *downstream* "
        "component's behaviour is invisible here.\n\n"
        "Treat these rankings as a **filter for candidate components**, not a complete "
        "circuit analysis.\n\n"
        "Model weights for this run were loaded selectively from safetensors shards on CPU "
        "(final norm, unembedding, and o_proj only when fine granularity is requested) — "
        "the full model is not loaded.\n\n"
        "Numeric Probability: runs nest experiments under `pre_period/` and `post_period/` "
        "as `{position}/{contrast}__{granularity}`. Post-period experiments drop samples "
        "whose verbalised_confidence is 1.0 and refill `--max_examples_for_mean` "
        "independently. The post-period `high_vs_low` low digit set includes `0`.\n\n"
        "Linguistic Confidence: runs (`--linguistic_confidence_prompt`) assume a "
        "variable-length extended Confidence: span. They write experiments flat under "
        "the run root as `{contrast}__{granularity}`. `confidence_token_vs_rest` is "
        "attributed at the first phrase token (`expected_confidence_tokens - 1`). "
        "Mistral additionally runs `L_vs_Un` at span index 6 (examples shorter than "
        "7 span tokens, or whose decoded token there is not `L` or `Un`, are skipped "
        "for that contrast). Other models are not yet supported for linguistic DLA.\n\n"
        "When `--confidence_split` is enabled (default), each experiment is also run on "
        "high- and low-confidence example cohorts (`...__high_conf/`, `...__low_conf/`), "
        "capped independently at `--max_examples_for_mean`. Post-period high/low splits "
        "also exclude verbalised_confidence 1.0. Disable with `--no-confidence_split`.\n\n"
        "When confidence splits are present, `{contrast}__{granularity}__high_minus_low/` "
        "ranks components by signed (high_conf mean − low_conf mean), not |difference|, "
        "with top-10 / bottom-10 lists (and a heatmap / bar chart of that difference).\n",
        encoding="utf-8",
    )


def write_config_txt(
    path: Path,
    *,
    args: argparse.Namespace,
    adapter: ModelAdapter,
    input_h5_resolved: str,
    digit_ids: dict[str, tuple[int, str]] | None,
    high_digits: set[str] | None,
    low_digits: set[str] | None,
    first_token_ids: dict[str, tuple[int, str]] | None,
    l_vs_un_token_ids: dict[str, tuple[int, str]] | None,
    individual_ids_all: Sequence[str],
    individual_ids_high: Sequence[str] | None,
    individual_ids_low: Sequence[str] | None,
    actual_n_examples_all: int,
    actual_n_examples_high: int | None,
    actual_n_examples_low: int | None,
    post_period_exclude_confidence_one: bool,
    actual_n_examples_all_post_period: int | None,
    actual_n_examples_high_post_period: int | None,
    actual_n_examples_low_post_period: int | None,
    l_vs_un_skipped_not_l_or_un: int | None,
    completeness_summaries: dict,
    finished_at: str,
    weight_load_method: str,
    resolved_weight_keys: dict[str, str],
    need_o_proj: bool,
) -> None:
    lines = [
        "Direct Logit Attribution Config",
        "================================",
        "",
        "[Run]",
        f"finished_at={finished_at}",
        f"script={Path(__file__).resolve()}",
        f"input_h5={input_h5_resolved}",
        f"output_dir={args.output_dir}",
        f"dtype=float32",
        f"device={args.device}",
        f"weight_load_method={weight_load_method}",
        f"need_o_proj={need_o_proj}",
        "",
        "[ModelAdapter]",
        f"model_name={adapter.model_name}",
    ]
    for k, v in asdict(adapter).items():
        if k == "model_name":
            continue
        lines.append(f"{k}={v}")
    lines += [
        "",
        "[ResolvedWeightKeys]",
    ]
    for k, v in sorted(resolved_weight_keys.items()):
        lines.append(f"{k}={v}")
    lines += [
        "",
        "[Alignment]",
        f"position_alignment_assumption={POSITION_ALIGNMENT_ASSUMPTION}",
        "",
        "[CLI]",
        f"linguistic_confidence_prompt={args.linguistic_confidence_prompt}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_confidence_tokens={args.expected_confidence_tokens}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"confidence_split={args.confidence_split}",
        f"post_period_exclude_confidence_one={post_period_exclude_confidence_one}",
        f"legacy_gemma_pre_postnorm_h5={args.legacy_gemma_pre_postnorm_h5}",
        f"granularity={args.granularity}",
        f"n_individual_examples={args.n_individual_examples}",
        f"max_examples_for_mean={args.max_examples_for_mean}",
        f"individual_example_indices={args.individual_example_indices}",
        f"bar_chart_top_k={args.bar_chart_top_k}",
    ]
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
        lines.append("span_index=6")
        skipped_n = (
            0 if l_vs_un_skipped_not_l_or_un is None else int(l_vs_un_skipped_not_l_or_un)
        )
        lines.append(f"l_vs_un_skipped_not_l_or_un={skipped_n}")
    else:
        if digit_ids is None or high_digits is None or low_digits is None:
            raise ValueError("numeric run is missing digit_ids/high_digits/low_digits.")
        lines += [
            "",
            "[Digits]",
            f"high_digits={sorted(high_digits)}",
            f"low_digits={sorted(low_digits)}",
        ]
        for d in "0123456789":
            tid, form = digit_ids[d]
            lines.append(f"digit_{d}_token_id={tid}")
            lines.append(f"digit_{d}_resolved_form={form!r}")
    lines += [
        "",
        "[Examples]",
        f"actual_n_examples_all={actual_n_examples_all}",
        f"individual_example_ids_all={list(individual_ids_all)}",
        f"actual_n_examples_high={actual_n_examples_high}",
        f"individual_example_ids_high="
        f"{None if individual_ids_high is None else list(individual_ids_high)}",
        f"actual_n_examples_low={actual_n_examples_low}",
        f"individual_example_ids_low="
        f"{None if individual_ids_low is None else list(individual_ids_low)}",
        f"actual_n_examples_all_post_period={actual_n_examples_all_post_period}",
        f"actual_n_examples_high_post_period={actual_n_examples_high_post_period}",
        f"actual_n_examples_low_post_period={actual_n_examples_low_post_period}",
        "",
        "[Numerics]",
        f"annotation_rounding_dp={ANNOTATION_ROUNDING_DP}",
        f"head_resum_atol={HEAD_RESUM_ATOL}",
        f"completeness_atol={COMPLETENESS_ATOL}",
        f"confidence_one_atol={CONFIDENCE_ONE_ATOL}",
        "",
        "[CompletenessResiduals]",
    ]
    for key, summary in sorted(completeness_summaries.items()):
        lines.append(
            f"{key}: mean_abs={summary['completeness_residual_mean_abs']:.6g} "
            f"max_abs={summary['completeness_residual_max_abs']:.6g} "
            f"head_resum_soft_fail={summary.get('head_resum_failure_count', 0)} "
            f"completeness_soft_fail={summary.get('completeness_soft_failure_count', 0)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_individual_ids(raw: str | None) -> list[str] | None:
    if raw is None or str(raw).strip() == "":
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return parts


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Direct Logit Attribution on verbalised-confidence tokens."
    )
    p.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=None,
        choices=list(SUPPORTED_MODEL_NAMES),
        help="Must be one of the three supported models (not required for --backfill_results_root).",
    )
    p.add_argument(
        "--input_h5",
        type=str,
        required=False,
        default=None,
        help=(
            "Processed verbalised embeddings H5. Numeric runs must use "
            "extend_probability_span. Linguistic runs assume a variable-length "
            "extended Confidence: span. Not required for --backfill_results_root."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Retained for config compatibility. Weight I/O is always selective "
            "safetensors on CPU; DLA math is float32 NumPy on CPU. Default: cpu."
        ),
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32"],
        help="DLA computation dtype (forced float32 on CPU).",
    )
    p.add_argument(
        "--linguistic_confidence_prompt",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, attribute linguistic Confidence: DLA (Mistral only). "
            "Assumes a variable-length extended span: confidence_token_vs_rest at "
            "the first phrase token, plus L_vs_Un at span index 6 (kept only when "
            "decoded_tokens at that index is L or Un). If false (default), "
            "run numeric Probability: digit DLA."
        ),
    )
    p.add_argument(
        "--expected_probability_tokens",
        type=int,
        default=5,
        help="Numeric-only: stored probability span length before the +2 extension.",
    )
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
        "--low_conf_threshold",
        type=float,
        default=0.1,
        help=(
            "Low verbalised-confidence cutoff for example cohorts. On numeric runs, "
            "also maps digits into the high_vs_low contrast direction (digit '0' is "
            "included in that direction's low set)."
        ),
    )
    p.add_argument(
        "--high_conf_threshold",
        type=float,
        default=0.9,
        help=(
            "High verbalised-confidence cutoff for example cohorts. On numeric runs, "
            "also maps digits into the high_vs_low contrast direction."
        ),
    )
    p.add_argument(
        "--confidence_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run DLA on high- and low-confidence example cohorts (default: True).",
    )
    p.add_argument(
        "--legacy_gemma_pre_postnorm_h5",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Gemma-only: treat H5 attn/mlp as pre–post-block-norm activations and "
            "apply post_attention_layernorm / post_feedforward_layernorm before DLA "
            "(default: False; use for older Gemma H5s)."
        ),
    )
    p.add_argument(
        "--granularity",
        nargs="+",
        choices=["coarse", "fine"],
        default=["coarse", "fine"],
        help=(
            "One or both of: coarse fine. Fine uses concat when present, otherwise "
            "recovers concat from o via inv(W_O)."
        ),
    )
    p.add_argument("--n_individual_examples", type=int, default=3)
    p.add_argument("--max_examples_for_mean", type=int, default=100)
    p.add_argument(
        "--individual_example_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated H5 example IDs for per-example outputs "
            "(all-examples cohort only; high/low cohorts always use first-N)."
        ),
    )
    p.add_argument("--bar_chart_top_k", type=int, default=25)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override auto-incremented results/<N>/ directory.",
    )
    p.add_argument(
        "--backfill_results_root",
        type=str,
        default=None,
        help=(
            "If set, skip DLA and write __high_minus_low rankings under this directory "
            "from existing __high_conf / __low_conf experiment outputs."
        ),
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.backfill_results_root:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        n_written = backfill_high_minus_low_results(
            Path(args.backfill_results_root),
            bar_chart_top_k=args.bar_chart_top_k,
            rounding_dp=ANNOTATION_ROUNDING_DP,
            method_label="DLA",
        )
        logging.info(
            "Backfill finished: wrote %d high-minus-low experiment dirs under %s",
            n_written,
            args.backfill_results_root,
        )
        return

    if not args.model_name:
        raise ValueError("--model_name is required unless --backfill_results_root is set.")
    if not args.input_h5:
        raise ValueError("--input_h5 is required unless --backfill_results_root is set.")

    if args.max_examples_for_mean <= args.n_individual_examples:
        raise ValueError(
            f"--max_examples_for_mean ({args.max_examples_for_mean}) must be strictly "
            f"greater than --n_individual_examples ({args.n_individual_examples})."
        )
    if args.n_individual_examples < 1:
        raise ValueError("--n_individual_examples must be >= 1")
    if args.bar_chart_top_k < 1:
        raise ValueError("--bar_chart_top_k must be >= 1")
    if args.legacy_gemma_pre_postnorm_h5 and args.model_name != "google/gemma-3-12b-it":
        raise ValueError(
            "--legacy_gemma_pre_postnorm_h5 is only valid with "
            "--model_name=google/gemma-3-12b-it "
            f"(got {args.model_name!r})."
        )
    if args.linguistic_confidence_prompt and args.expected_confidence_tokens < 1:
        raise ValueError("--expected_confidence_tokens must be >= 1")
    validate_confidence_thresholds(args.high_conf_threshold, args.low_conf_threshold)
    if args.linguistic_confidence_prompt:
        if args.model_name != "mistralai/Mistral-7B-Instruct-v0.1":
            raise ValueError(
                "--linguistic_confidence_prompt DLA is currently only implemented for "
                "mistralai/Mistral-7B-Instruct-v0.1 "
                f"(got {args.model_name!r})."
            )
        linguistic_first_tokens_for_model(args.model_name)

    # Deduplicate granularity while preserving order
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

    need_o_proj = "fine" in gran_list
    require_head_source = need_o_proj
    if need_o_proj:
        logging.info(
            "Fine granularity requested; concat embeddings are used when present, "
            "otherwise concat is recovered from o via inv(W_O)."
        )
    else:
        logging.info(
            "Coarse-only run; concat embeddings and o_proj weights are optional/skipped."
        )
    if args.legacy_gemma_pre_postnorm_h5:
        logging.info(
            "Legacy Gemma H5 mode: applying post-attn/post-FF norms to cached attn/mlp."
        )

    logging.info(
        "Loading selective safetensors weights for %s on CPU (device arg=%s).",
        args.model_name,
        args.device,
    )
    tensors = load_model_tensors(
        args.model_name,
        need_o_proj=need_o_proj,
        legacy_gemma_pre_postnorm_h5=args.legacy_gemma_pre_postnorm_h5,
    )
    adapter = tensors.adapter
    logging.info("Adapter: %s", asdict(adapter))
    logging.info("Resolved weight keys: %s", tensors.resolved_weight_keys)

    digit_ids: dict[str, tuple[int, str]] | None = None
    high_digits: set[str] | None = None
    low_digits: set[str] | None = None
    first_token_ids: dict[str, tuple[int, str]] | None = None
    l_vs_un_token_ids: dict[str, tuple[int, str]] | None = None
    if args.linguistic_confidence_prompt:
        configure_prefix_tokens_for_model(args.model_name)
        pieces = linguistic_first_tokens_for_model(args.model_name)
        first_token_ids = resolve_linguistic_first_token_ids(tensors.tokenizer, pieces)
        for piece, (tid, form) in first_token_ids.items():
            logging.info("Linguistic first token %r -> id=%s form=%r", piece, tid, form)
        l_id, l_form = resolve_single_piece_token_id(tensors.tokenizer, "L")
        un_id, un_form = resolve_single_piece_token_id(tensors.tokenizer, "Un")
        l_vs_un_token_ids = {"L": (l_id, l_form), "Un": (un_id, un_form)}
        logging.info("L vs Un: L id=%s form=%r; Un id=%s form=%r", l_id, l_form, un_id, un_form)
        expected_span = args.expected_confidence_tokens
        span_mismatch_detail = (
            f"but expected at least expected_confidence_tokens="
            f"{args.expected_confidence_tokens} (variable-length extended phrase span)."
        )
        contrasts = build_linguistic_contrasts(
            tensors.w_u,
            [tid for tid, _form in first_token_ids.values()],
            l_token_id=l_id,
            un_token_id=un_id,
            first_value_index=args.expected_confidence_tokens - 1,
        )
    else:
        digit_ids = resolve_all_digit_ids(tensors.tokenizer)
        for d, (tid, form) in digit_ids.items():
            logging.info("Digit %s -> id=%s form=%r", d, tid, form)
        high_digits, low_digits = high_low_digit_sets(
            args.high_conf_threshold, args.low_conf_threshold
        )
        logging.info(
            "High digits %s  Low digits %s", sorted(high_digits), sorted(low_digits)
        )
        expected_span = args.expected_probability_tokens + 2
        span_mismatch_detail = (
            f"but expected exactly expected_probability_tokens+2="
            f"{args.expected_probability_tokens}+2={expected_span}. "
            "Input H5 must have been generated with extend_probability_span."
        )
        contrasts = build_contrasts(tensors.w_u, digit_ids, high_digits, low_digits)

    fill_post_period_excl_one = not args.linguistic_confidence_prompt
    cohorts = load_usable_example_cohorts(
        input_h5,
        expected_span=expected_span,
        span_mismatch_detail=span_mismatch_detail,
        max_examples=args.max_examples_for_mean,
        require_head_source=require_head_source,
        high_conf_threshold=args.high_conf_threshold,
        low_conf_threshold=args.low_conf_threshold,
        fill_confidence_splits=args.confidence_split,
        require_exact_span=not args.linguistic_confidence_prompt,
        fill_post_period_excl_one=fill_post_period_excl_one,
        require_decoded_tokens=args.linguistic_confidence_prompt,
    )
    examples_all = cohorts.all
    actual_n_all = len(examples_all)
    logging.info(
        "Using %d all-examples (cap=%d)", actual_n_all, args.max_examples_for_mean
    )
    if actual_n_all <= args.n_individual_examples:
        raise ValueError(
            f"All-examples cohort has {actual_n_all} usable examples, which must be "
            f"strictly greater than --n_individual_examples "
            f"({args.n_individual_examples})."
        )

    requested_ids = parse_individual_ids(args.individual_example_indices)
    usable_ids_all = [ex.example_id for ex in examples_all]
    usable_set_all = set(usable_ids_all)
    if requested_ids is not None:
        missing = [i for i in requested_ids if i not in usable_set_all]
        if missing:
            raise ValueError(
                f"--individual_example_indices not in usable all-examples set: {missing}"
            )
        if len(requested_ids) > actual_n_all:
            raise ValueError("More individual IDs requested than usable all-examples.")
        individual_ids_all = requested_ids
    else:
        individual_ids_all = usable_ids_all[: args.n_individual_examples]
    logging.info("Individual example IDs (all): %s", individual_ids_all)

    individual_ids_high: list[str] | None = None
    individual_ids_low: list[str] | None = None
    actual_n_high: int | None = None
    actual_n_low: int | None = None
    actual_n_all_post: int | None = None
    actual_n_high_post: int | None = None
    actual_n_low_post: int | None = None
    if fill_post_period_excl_one:
        if not cohorts.all_excl_one:
            raise ValueError(
                "No usable post-period examples after excluding "
                "verbalised_confidence == 1.0."
            )
        actual_n_all_post = len(cohorts.all_excl_one)
        logging.info(
            "Using %d post-period all-examples excluding confidence 1.0 (cap=%d)",
            actual_n_all_post,
            args.max_examples_for_mean,
        )
        if actual_n_all_post <= args.n_individual_examples:
            raise ValueError(
                f"Post-period all-examples cohort has {actual_n_all_post} usable "
                "examples after excluding verbalised_confidence == 1.0, which must "
                f"be strictly greater than --n_individual_examples "
                f"({args.n_individual_examples})."
            )
    if args.confidence_split:
        if not cohorts.high:
            raise ValueError(
                "confidence_split enabled but high-confidence cohort is empty "
                f"(verbalised_confidence >= {args.high_conf_threshold})."
            )
        if not cohorts.low:
            raise ValueError(
                "confidence_split enabled but low-confidence cohort is empty "
                f"(verbalised_confidence <= {args.low_conf_threshold})."
            )
        actual_n_high = len(cohorts.high)
        actual_n_low = len(cohorts.low)
        if actual_n_high <= args.n_individual_examples:
            raise ValueError(
                f"High-confidence cohort has {actual_n_high} usable examples, which "
                f"must be strictly greater than --n_individual_examples "
                f"({args.n_individual_examples})."
            )
        if actual_n_low <= args.n_individual_examples:
            raise ValueError(
                f"Low-confidence cohort has {actual_n_low} usable examples, which "
                f"must be strictly greater than --n_individual_examples "
                f"({args.n_individual_examples})."
            )
        individual_ids_high = [ex.example_id for ex in cohorts.high][
            : args.n_individual_examples
        ]
        individual_ids_low = [ex.example_id for ex in cohorts.low][
            : args.n_individual_examples
        ]
        logging.info(
            "Using %d high-conf / %d low-conf examples (cap=%d each)",
            actual_n_high,
            actual_n_low,
            args.max_examples_for_mean,
        )
        logging.info("Individual example IDs (high_conf): %s", individual_ids_high)
        logging.info("Individual example IDs (low_conf): %s", individual_ids_low)
        if fill_post_period_excl_one:
            if not cohorts.high_excl_one:
                raise ValueError(
                    "confidence_split enabled but post-period high-confidence cohort "
                    "is empty (verbalised_confidence >= "
                    f"{args.high_conf_threshold} and != 1.0)."
                )
            if not cohorts.low_excl_one:
                raise ValueError(
                    "confidence_split enabled but post-period low-confidence cohort "
                    "is empty (verbalised_confidence <= "
                    f"{args.low_conf_threshold} and != 1.0)."
                )
            actual_n_high_post = len(cohorts.high_excl_one)
            actual_n_low_post = len(cohorts.low_excl_one)
            if actual_n_high_post <= args.n_individual_examples:
                raise ValueError(
                    f"Post-period high-confidence cohort has {actual_n_high_post} "
                    "usable examples after excluding verbalised_confidence == 1.0, "
                    "which must be strictly greater than --n_individual_examples "
                    f"({args.n_individual_examples})."
                )
            if actual_n_low_post <= args.n_individual_examples:
                raise ValueError(
                    f"Post-period low-confidence cohort has {actual_n_low_post} "
                    "usable examples after excluding verbalised_confidence == 1.0, "
                    "which must be strictly greater than --n_individual_examples "
                    f"({args.n_individual_examples})."
                )
            logging.info(
                "Using %d post-period high-conf / %d post-period low-conf examples "
                "excluding confidence 1.0 (cap=%d each)",
                actual_n_high_post,
                actual_n_low_post,
                args.max_examples_for_mean,
            )

    if need_o_proj:
        fallback_examples = list(examples_all)
        if args.confidence_split:
            fallback_examples.extend(cohorts.high)
            fallback_examples.extend(cohorts.low)
        if fill_post_period_excl_one:
            fallback_examples.extend(cohorts.all_excl_one)
            if args.confidence_split:
                fallback_examples.extend(cohorts.high_excl_one)
                fallback_examples.extend(cohorts.low_excl_one)
        maybe_invert_o_proj_for_concat_fallback(tensors, fallback_examples)

    l_vs_un_skipped_ids: set[str] = set()

    def _examples_for_contrast(
        examples: list[ExampleData], contrast: ContrastSpec
    ) -> list[ExampleData]:
        kept: list[ExampleData] = []
        for ex in examples:
            if contrast.span_index is not None and len(ex.res) <= contrast.span_index:
                continue
            if contrast.name == "L_vs_Un":
                if contrast.span_index is None:
                    raise ValueError("L_vs_Un contrast is missing span_index.")
                token, first_prob, reason = _l_vs_un_token_check(ex, contrast.span_index)
                if reason is not None:
                    if ex.example_id not in l_vs_un_skipped_ids:
                        l_vs_un_skipped_ids.add(ex.example_id)
                        _log_l_vs_un_skip(
                            ex,
                            span_index=contrast.span_index,
                            token=token,
                            first_prob_token_index=first_prob,
                            reason=reason,
                        )
                    continue
            kept.append(ex)
        return kept

    def _individual_ids_for_filtered(
        filtered: list[ExampleData], preferred_ids: Sequence[str]
    ) -> list[str]:
        usable_ids = [ex.example_id for ex in filtered]
        usable_set = set(usable_ids)
        chosen = [i for i in preferred_ids if i in usable_set]
        if len(chosen) < args.n_individual_examples:
            for eid in usable_ids:
                if eid not in chosen:
                    chosen.append(eid)
                if len(chosen) >= args.n_individual_examples:
                    break
        return chosen[: args.n_individual_examples]

    def _run_cohort_experiments(
        *,
        cohort_examples: list[ExampleData],
        post_period_examples: list[ExampleData] | None,
        individual_ids: Sequence[str],
        dirname_suffix: str,
        cohort_label: str,
    ) -> None:
        for contrast in contrasts:
            examples = (
                post_period_examples
                if contrast.position == "post_period" and post_period_examples is not None
                else cohort_examples
            )
            filtered = _examples_for_contrast(examples, contrast)
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
            contrast_individual_ids = _individual_ids_for_filtered(
                filtered, individual_ids
            )
            examples_by_id = {ex.example_id: ex for ex in filtered}
            for gran in gran_list:
                leaf = f"{contrast.name}__{gran}{dirname_suffix}"
                if args.linguistic_confidence_prompt:
                    exp_name = leaf
                    exp_dir = run_root / leaf
                else:
                    exp_name = f"{contrast.position}/{leaf}"
                    exp_dir = run_root / contrast.position / leaf
                logging.info(
                    "Running experiment %s (n=%d, span_index=%s)",
                    exp_name,
                    len(filtered),
                    contrast.span_index,
                )
                result = run_dla_for_experiment(
                    filtered,
                    contrast=contrast,
                    granularity=gran,
                    tensors=tensors,
                    head_resum_atol=HEAD_RESUM_ATOL,
                    completeness_atol=COMPLETENESS_ATOL,
                    legacy_gemma_pre_postnorm_h5=args.legacy_gemma_pre_postnorm_h5,
                )
                summary = write_experiment_outputs(
                    exp_dir,
                    result,
                    examples_by_id,
                    contrast_individual_ids,
                    n_layers=adapter.n_layers,
                    n_heads=adapter.n_query_heads,
                    bar_chart_top_k=args.bar_chart_top_k,
                    rounding_dp=ANNOTATION_ROUNDING_DP,
                )
                completeness_summaries[exp_name] = summary
                cohort_results[dirname_suffix][
                    (contrast.position, contrast.name, gran)
                ] = result
                logging.info(
                    "%s done: max completeness |residual|=%.3g "
                    "(head_resum_soft_fail=%d completeness_soft_fail=%d)",
                    exp_name,
                    summary["completeness_residual_max_abs"],
                    result.head_resum_failure_count,
                    result.completeness_soft_failure_count,
                )

    completeness_summaries: dict = {}
    cohort_results: dict[str, dict[tuple[str, str, str], ExperimentResult]] = {
        "": {},
        "__high_conf": {},
        "__low_conf": {},
    }
    _run_cohort_experiments(
        cohort_examples=examples_all,
        post_period_examples=(
            cohorts.all_excl_one if fill_post_period_excl_one else None
        ),
        individual_ids=individual_ids_all,
        dirname_suffix="",
        cohort_label="all-examples",
    )
    if args.confidence_split:
        _run_cohort_experiments(
            cohort_examples=cohorts.high,
            post_period_examples=(
                cohorts.high_excl_one if fill_post_period_excl_one else None
            ),
            individual_ids=individual_ids_high or [],
            dirname_suffix="__high_conf",
            cohort_label="high-confidence",
        )
        _run_cohort_experiments(
            cohort_examples=cohorts.low,
            post_period_examples=(
                cohorts.low_excl_one if fill_post_period_excl_one else None
            ),
            individual_ids=individual_ids_low or [],
            dirname_suffix="__low_conf",
            cohort_label="low-confidence",
        )
        for key, high_result in cohort_results["__high_conf"].items():
            low_result = cohort_results["__low_conf"].get(key)
            if low_result is None:
                raise ValueError(
                    f"High-conf result {key} has no matching low-conf experiment."
                )
            position, contrast_name, gran = key
            leaf = f"{contrast_name}__{gran}__high_minus_low"
            if args.linguistic_confidence_prompt:
                exp_dir = run_root / leaf
                exp_name = leaf
            else:
                exp_dir = run_root / position / leaf
                exp_name = f"{position}/{leaf}"
            logging.info("Writing high-minus-low ranking %s", exp_name)
            write_high_minus_low_from_results(
                exp_dir,
                high_result,
                low_result,
                n_layers=adapter.n_layers,
                n_heads=adapter.n_query_heads,
                bar_chart_top_k=args.bar_chart_top_k,
                rounding_dp=ANNOTATION_ROUNDING_DP,
                method_label="DLA",
            )

    if args.linguistic_confidence_prompt:
        logging.info(
            "L_vs_Un skipped %d examples whose span-index-6 token was not L or Un.",
            len(l_vs_un_skipped_ids),
        )

    finished_at = datetime.now().isoformat(timespec="seconds")
    write_config_txt(
        run_root / "config.txt",
        args=args,
        adapter=adapter,
        input_h5_resolved=input_h5,
        digit_ids=digit_ids,
        high_digits=high_digits,
        low_digits=low_digits,
        first_token_ids=first_token_ids,
        l_vs_un_token_ids=l_vs_un_token_ids,
        individual_ids_all=individual_ids_all,
        individual_ids_high=individual_ids_high,
        individual_ids_low=individual_ids_low,
        actual_n_examples_all=actual_n_all,
        actual_n_examples_high=actual_n_high,
        actual_n_examples_low=actual_n_low,
        post_period_exclude_confidence_one=fill_post_period_excl_one,
        actual_n_examples_all_post_period=actual_n_all_post,
        actual_n_examples_high_post_period=actual_n_high_post,
        actual_n_examples_low_post_period=actual_n_low_post,
        l_vs_un_skipped_not_l_or_un=(
            len(l_vs_un_skipped_ids) if args.linguistic_confidence_prompt else None
        ),
        completeness_summaries=completeness_summaries,
        finished_at=finished_at,
        weight_load_method=tensors.weight_load_method,
        resolved_weight_keys=tensors.resolved_weight_keys,
        need_o_proj=need_o_proj,
    )
    logging.info("Finished. Results at %s", run_root)


if __name__ == "__main__":
    main()
