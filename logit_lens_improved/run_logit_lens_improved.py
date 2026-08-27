"""
Improved logit-lens runner over processed H5 embeddings.

Supports:
- selecting embedding families to include
- per-example top-k and bottom-k tables for n examples
- mean-embedding top-k and bottom-k tables over all examples
- optional high/low confidence split means
- optional subblock mode (attn/mlp rows only)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from datetime import datetime
import json
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

EMBEDDING_CHOICES = (
    "mean_prompt",
    "guess_tokens",
    "mean_sem_answer",
    "probability_tokens",
    "mean_prob_val",
)
EPS = 1e-12
WEIGHT_LOAD_METHOD = "safetensors_selective_cpu"

SUPPORTED_MODEL_NAMES = (
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-32B-Instruct",
)

# Hardcoded logit-lens extraction specs (selective safetensors; no full-model load).
LOGIT_LENS_SPECS: dict[str, dict] = {
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "trust_remote_code": False,
        "rmsnorm_gain": "plain_weight",
        "norm_path": ("model", "norm"),
        "norm_path_fallbacks": (),
        "embed_path": ("model", "embed_tokens"),
        "embed_path_fallbacks": (),
        "lm_head_path": ("lm_head",),
        "lm_head_path_fallbacks": (),
        "softcap": None,
        "rms_norm_eps": 1e-5,
        "config_root": None,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "trust_remote_code": True,
        "rmsnorm_gain": "plain_weight",
        "norm_path": ("model", "norm"),
        "norm_path_fallbacks": (),
        "embed_path": ("model", "embed_tokens"),
        "embed_path_fallbacks": (),
        "lm_head_path": ("lm_head",),
        "lm_head_path_fallbacks": (),
        "softcap": None,
        "rms_norm_eps": 1e-6,
        "config_root": None,
    },
    "google/gemma-3-12b-it": {
        "trust_remote_code": True,
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
        "softcap": None,
        "rms_norm_eps": 1e-6,
        "config_root": "text_config",
    },
}


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cpu")


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _as_layer_hidden(arr_like: np.ndarray) -> np.ndarray:
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


def _h5_list_length(group_obj: h5py.Group) -> int:
    return int(group_obj.attrs.get("__len__", len(group_obj.keys())))


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


def _bias_key_candidates(weight_keys: Sequence[str]) -> list[str]:
    keys: list[str] = []
    for weight_key in weight_keys:
        if not weight_key.endswith(".weight"):
            continue
        bias_key = weight_key[: -len(".weight")] + ".bias"
        if bias_key not in keys:
            keys.append(bias_key)
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


def _select_text_config(config, *, config_root: str | None):
    if config_root == "text_config":
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            raise ValueError(
                f"config_root=text_config but {type(config).__name__} has no text_config."
            )
        return text_cfg
    return config


def _tie_word_embeddings(model_name_or_path: str, spec: dict) -> bool:
    trust_remote_code = bool(spec["trust_remote_code"])
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    cfg = _select_text_config(config, config_root=spec.get("config_root"))
    tie = bool(getattr(config, "tie_word_embeddings", False))
    if not tie:
        tie = bool(getattr(cfg, "tie_word_embeddings", False))
    return tie


def _build_weight_map(model_name: str) -> dict[str, str]:
    """Map state-dict key -> absolute path of the safetensors shard containing it.

    Only downloads the index (or single-file checkpoint). Individual shards are
    resolved lazily via ``_resolve_shard_path`` when a key is actually loaded.
    """
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
        # Store "model_name::shard_name" so we can download only needed shards later.
        return {
            key: f"{model_name}::{shard}"
            for key, shard in weight_map.items()
        }

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


def _resolve_shard_path(shard_ref: str) -> str:
    """Resolve a weight-map entry to a local safetensors path."""
    if "::" in shard_ref:
        model_name, shard_name = shard_ref.split("::", 1)
        return hf_hub_download(model_name, shard_name)
    return shard_ref


def _load_torch_tensor(shard_path: str, key: str) -> torch.Tensor:
    # Use PyTorch backend: many checkpoints store bfloat16, which NumPy cannot decode.
    resolved = _resolve_shard_path(shard_path)
    with safe_open(resolved, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"Key {key!r} not in shard {resolved}")
        tensor = f.get_tensor(key)
    return tensor.detach().to(dtype=torch.float32).contiguous()


def _load_first_matching_weight(
    weight_map: dict[str, str], candidates: Sequence[str], *, label: str
) -> tuple[torch.Tensor, str]:
    tried: list[str] = []
    for key in candidates:
        tried.append(key)
        shard = weight_map.get(key)
        if shard is None:
            continue
        tensor = _load_torch_tensor(shard, key)
        print(f"Loaded {label} from key {key} (shape={tuple(tensor.shape)})")
        return tensor, key
    present_sample = sorted(weight_map.keys())[:20]
    raise RuntimeError(
        f"Could not load {label}. Tried keys:\n  "
        + "\n  ".join(tried)
        + f"\nNone were present in the checkpoint weight map "
        f"({len(weight_map)} keys). Sample keys: {present_sample}"
    )


def _load_unembedding(model_name_or_path: str, device: torch.device) -> tuple:
    """Load tokenizer + final RMSNorm / W_U via selective safetensors on CPU."""
    if model_name_or_path not in LOGIT_LENS_SPECS:
        raise ValueError(
            f"Unsupported model_name_or_path={model_name_or_path!r}. "
            f"Supported: {list(LOGIT_LENS_SPECS)}"
        )
    spec = LOGIT_LENS_SPECS[model_name_or_path]
    trust_remote_code = bool(spec["trust_remote_code"])

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        print(
            f"WARNING: fast tokenizer load failed ({exc}). "
            "Falling back to slow tokenizer (use_fast=False)."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

    print(
        f"Loading logit-lens weights for {model_name_or_path} via "
        f"{WEIGHT_LOAD_METHOD} (device={device})."
    )
    weight_map = _build_weight_map(model_name_or_path)

    norm_candidates = _weight_key_candidates(
        tuple(spec["norm_path"]), tuple(spec.get("norm_path_fallbacks", ()))
    )
    norm_raw, _norm_key = _load_first_matching_weight(
        weight_map, norm_candidates, label="final_norm.weight"
    )
    rmsnorm_gain = str(spec["rmsnorm_gain"])
    if rmsnorm_gain == "one_plus_weight":
        norm_weight = (1.0 + norm_raw).to(dtype=torch.float32)
    elif rmsnorm_gain == "plain_weight":
        norm_weight = norm_raw.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown rmsnorm_gain={rmsnorm_gain!r}")

    tie = _tie_word_embeddings(model_name_or_path, spec)
    embed_candidates = _weight_key_candidates(
        tuple(spec["embed_path"]), tuple(spec.get("embed_path_fallbacks", ()))
    )
    lm_head_candidates = _weight_key_candidates(
        tuple(spec["lm_head_path"]), tuple(spec.get("lm_head_path_fallbacks", ()))
    )
    if tie:
        w_u_candidates = embed_candidates + [
            k for k in lm_head_candidates if k not in embed_candidates
        ]
        w_u_label = "tied_embeddings / W_U"
    else:
        w_u_candidates = lm_head_candidates + [
            k for k in embed_candidates if k not in lm_head_candidates
        ]
        w_u_label = "lm_head / W_U"
    unembed_w, w_u_key = _load_first_matching_weight(
        weight_map, w_u_candidates, label=w_u_label
    )
    if unembed_w.ndim != 2:
        raise ValueError(
            f"W_U source {w_u_key!r} has shape {tuple(unembed_w.shape)}, expected 2D."
        )

    unembed_b: torch.Tensor | None = None
    for bias_key in _bias_key_candidates(w_u_candidates):
        shard = weight_map.get(bias_key)
        if shard is None:
            continue
        unembed_b = _load_torch_tensor(shard, bias_key)
        print(f"Loaded lm_head bias from key {bias_key} (shape={tuple(unembed_b.shape)})")
        break

    # Materialize on CPU first, then move only the lens tensors to the target device.
    unembed_w = unembed_w.to(device=device, dtype=torch.float32)
    if unembed_b is not None:
        unembed_b = unembed_b.to(device=device, dtype=torch.float32)
    norm_weight = norm_weight.to(device=device, dtype=torch.float32)

    norm_eps = float(spec["rms_norm_eps"])
    softcap = spec["softcap"]
    softcap = float(softcap) if softcap is not None else None

    return tokenizer, unembed_w, unembed_b, norm_weight, norm_eps, softcap


def _load_gemma_post_block_norm_gammas(
    model_name_or_path: str,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Load per-layer post-attn / post-FF gains (1+w) for legacy Gemma H5 subblock lensing."""
    if model_name_or_path not in LOGIT_LENS_SPECS:
        raise ValueError(f"Unsupported model_name_or_path={model_name_or_path!r}.")
    spec = LOGIT_LENS_SPECS[model_name_or_path]
    if "layers_path" not in spec:
        raise ValueError(
            f"{model_name_or_path} has no layers_path; cannot load post-block norms."
        )
    trust_remote_code = bool(spec["trust_remote_code"])
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    cfg = _select_text_config(config, config_root=spec.get("config_root"))
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    hidden_size = int(getattr(cfg, "hidden_size"))
    eps = float(spec["rms_norm_eps"])
    weight_map = _build_weight_map(model_name_or_path)
    layers_path = tuple(spec["layers_path"])
    layers_fallbacks = tuple(spec.get("layers_path_fallbacks", ()))

    post_attn: list[np.ndarray] = []
    post_ff: list[np.ndarray] = []
    for layer_idx in range(n_layers):
        attn_w, _ = _load_first_matching_weight(
            weight_map,
            _post_attn_norm_key_candidates(layers_path, layers_fallbacks, layer_idx),
            label=f"post_attention_layernorm layer {layer_idx}",
        )
        ff_w, _ = _load_first_matching_weight(
            weight_map,
            _post_ff_norm_key_candidates(layers_path, layers_fallbacks, layer_idx),
            label=f"post_feedforward_layernorm layer {layer_idx}",
        )
        if attn_w.ndim != 1 or int(attn_w.shape[0]) != hidden_size:
            raise ValueError(
                f"post_attention_layernorm layer {layer_idx} shape {tuple(attn_w.shape)} "
                f"!= [{hidden_size}]"
            )
        if ff_w.ndim != 1 or int(ff_w.shape[0]) != hidden_size:
            raise ValueError(
                f"post_feedforward_layernorm layer {layer_idx} shape {tuple(ff_w.shape)} "
                f"!= [{hidden_size}]"
            )
        post_attn.append((1.0 + attn_w.numpy()).astype(np.float32, copy=False))
        post_ff.append((1.0 + ff_w.numpy()).astype(np.float32, copy=False))
    return post_attn, post_ff, eps


def _apply_gemma_rmsnorm_np(
    x: np.ndarray, gamma: np.ndarray, eps: float
) -> np.ndarray:
    """Gemma-style RMSNorm on a 1D vector: x * gamma / rms(x)."""
    x32 = np.asarray(x, dtype=np.float32)
    g32 = np.asarray(gamma, dtype=np.float32)
    if x32.shape != g32.shape:
        raise ValueError(
            f"_apply_gemma_rmsnorm_np: x shape {x32.shape} != gamma shape {g32.shape}"
        )
    inv_rms = 1.0 / float(np.sqrt(np.mean(np.square(x32)) + eps))
    return (x32 * g32 * np.float32(inv_rms)).astype(np.float32, copy=False)


def _apply_legacy_gemma_post_block_norms(
    arr: np.ndarray,
    *,
    component: str,
    post_attn_gammas: Sequence[np.ndarray],
    post_ff_gammas: Sequence[np.ndarray],
    eps: float,
) -> np.ndarray:
    """Apply per-layer post-attn (attn) or post-FF (mlp) norms to [n_layers, H]."""
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.ndim != 2:
        raise ValueError(f"Expected [n_layers, H], got shape {out.shape}")
    n_layers = out.shape[0]
    if component == "attn":
        if len(post_attn_gammas) != n_layers:
            raise ValueError(
                f"post_attn_gammas length {len(post_attn_gammas)} != n_layers {n_layers}"
            )
        for l in range(n_layers):
            out[l] = _apply_gemma_rmsnorm_np(out[l], post_attn_gammas[l], eps)
    elif component == "mlp":
        if len(post_ff_gammas) != n_layers:
            raise ValueError(
                f"post_ff_gammas length {len(post_ff_gammas)} != n_layers {n_layers}"
            )
        for l in range(n_layers):
            out[l] = _apply_gemma_rmsnorm_np(out[l], post_ff_gammas[l], eps)
    else:
        raise ValueError(f"Legacy post-block norms only apply to attn/mlp, got {component!r}")
    return out


def _apply_rmsnorm(hidden: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
    hidden_sq_mean = hidden.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(hidden_sq_mean + norm_eps)
    return (hidden * inv_rms) * norm_weight


def _apply_logit_softcap(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)


def _probs_from_hidden(
    hidden: torch.Tensor,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    softcap: float | None = None,
) -> torch.Tensor:
    hidden = _apply_rmsnorm(hidden, norm_weight=norm_weight, norm_eps=norm_eps)
    logits = hidden @ w_u.T
    if b_u is not None:
        logits = logits + b_u
    logits = _apply_logit_softcap(logits, softcap)
    return torch.softmax(logits, dim=-1)


def _topn_for_distribution(dist: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if top_k >= dist.shape[0]:
        idx = np.argsort(-dist)
    else:
        idx = np.argpartition(-dist, top_k - 1)[:top_k]
        idx = idx[np.argsort(-dist[idx])]
    vals = dist[idx]
    return idx.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)


def _bottomn_for_distribution(dist: np.ndarray, bottom_k: int) -> tuple[np.ndarray, np.ndarray]:
    if bottom_k >= dist.shape[0]:
        idx = np.argsort(dist)
    else:
        idx = np.argpartition(dist, bottom_k - 1)[:bottom_k]
        idx = idx[np.argsort(dist[idx])]
    vals = dist[idx]
    return idx.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)


def _format_token_decoded(tokenizer, token_id: int) -> str:
    token = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return token if token else "<empty>"


def _disable_table_mathtext(table) -> None:
    """Stop matplotlib from parsing ``$...$`` in cell text (tokens like ``$$`` crash savefig)."""
    for cell in table.get_celld().values():
        text = cell.get_text()
        if hasattr(text, "set_parse_math"):
            text.set_parse_math(False)
        else:
            raw = text.get_text()
            if "$" in raw:
                text.set_text(raw.replace("$", r"\$"))


def _get_run_base_dir(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    k = 1
    while (results_dir / str(k)).exists():
        k += 1
    run_base = results_dir / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _get_component_group(emb_field: h5py.Group, component: str, *, subblock_mode: bool):
    if not isinstance(emb_field, h5py.Group):
        return None

    if subblock_mode:
        return emb_field.get(component)

    if "res" in emb_field:
        return emb_field.get("res")
    return emb_field


def _get_single_embedding(
    r0: h5py.Group,
    field_name: str,
    component: str,
    *,
    subblock_mode: bool,
) -> np.ndarray | None:
    field = r0.get(field_name)
    if field is None or not isinstance(field, h5py.Group):
        return None
    comp_group = _get_component_group(field, component, subblock_mode=subblock_mode)
    if comp_group is None:
        return None
    if not isinstance(comp_group, h5py.Dataset):
        return None
    return _as_layer_hidden(comp_group[()])


def _get_list_embeddings(
    r0: h5py.Group,
    field_name: str,
    component: str,
    *,
    subblock_mode: bool,
) -> list[np.ndarray] | None:
    field = r0.get(field_name)
    if field is None or not isinstance(field, h5py.Group):
        return None
    comp_group = _get_component_group(field, component, subblock_mode=subblock_mode)
    if comp_group is None or not isinstance(comp_group, h5py.Group):
        return None
    length = _h5_list_length(comp_group)
    out: list[np.ndarray] = []
    for i in range(length):
        ds = comp_group.get(str(i))
        if ds is None or not isinstance(ds, h5py.Dataset):
            return None
        out.append(_as_layer_hidden(ds[()]))
    return out


def _extract_example_token_positions(
    r0: h5py.Group,
    include_embeddings: set[str],
    component: str,
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    subblock_mode: bool,
) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []

    if "mean_prompt" in include_embeddings:
        arr = _get_single_embedding(
            r0,
            "embeddings_mean_prompt",
            component,
            subblock_mode=subblock_mode,
        )
        if arr is None:
            raise ValueError("Missing embeddings_mean_prompt")
        rows.append(("mean_prompt", arr))

    if "guess_tokens" in include_embeddings:
        guess_list = _get_list_embeddings(
            r0,
            "embeddings_guess",
            component,
            subblock_mode=subblock_mode,
        )
        if guess_list is None:
            raise ValueError("Missing embeddings_guess")
        if len(guess_list) != expected_guess_tokens:
            raise ValueError(
                f"Expected {expected_guess_tokens} guess token embeddings, found {len(guess_list)}."
            )
        rows.extend((f"guess_{i}", arr) for i, arr in enumerate(guess_list))

    if "mean_sem_answer" in include_embeddings:
        arr = _get_single_embedding(
            r0,
            "embeddings_mean_sem_answer",
            component,
            subblock_mode=subblock_mode,
        )
        if arr is None:
            raise ValueError("Missing embeddings_mean_sem_answer")
        rows.append(("mean_sem_answer", arr))

    if "probability_tokens" in include_embeddings:
        prob_list = _get_list_embeddings(
            r0,
            "embeddings_probability",
            component,
            subblock_mode=subblock_mode,
        )
        if prob_list is None:
            raise ValueError("Missing embeddings_probability")
        if len(prob_list) != expected_probability_tokens:
            raise ValueError(
                f"Expected {expected_probability_tokens} probability token embeddings, found {len(prob_list)}."
            )
        rows.extend((f"prob_{i}", arr) for i, arr in enumerate(prob_list))

    if "mean_prob_val" in include_embeddings:
        arr = _get_single_embedding(
            r0,
            "embeddings_mean_prob_val",
            component,
            subblock_mode=subblock_mode,
        )
        if arr is None:
            raise ValueError("Missing embeddings_mean_prob_val")
        rows.append(("mean_prob_val", arr))

    return rows


def _canonical_token_labels(
    include_embeddings: set[str],
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> list[str]:
    labels: list[str] = []
    if "mean_prompt" in include_embeddings:
        labels.append("mean_prompt")
    if "guess_tokens" in include_embeddings:
        labels.extend(f"guess_{i}" for i in range(expected_guess_tokens))
    if "mean_sem_answer" in include_embeddings:
        labels.append("mean_sem_answer")
    if "probability_tokens" in include_embeddings:
        labels.extend(f"prob_{i}" for i in range(expected_probability_tokens))
    if "mean_prob_val" in include_embeddings:
        labels.append("mean_prob_val")
    return labels


def _save_topk_table_png(
    out_path: Path,
    tokenizer,
    top_ids: np.ndarray,
    top_vals: np.ndarray,
    *,
    rank_label: str = "Top",
    shade_body: bool = True,
) -> None:
    n_rows, top_k = top_ids.shape
    col_labels = ["Row"]
    for k in range(top_k):
        col_labels.append(f"{rank_label}-{k + 1} token")
        col_labels.append(f"{rank_label}-{k + 1} prob")

    rows = []
    for row_idx in range(n_rows):
        row = [f"layer_{row_idx}"]
        for k in range(top_k):
            tok_id = int(top_ids[row_idx, k])
            prob = float(top_vals[row_idx, k])
            row.append(_format_token_decoded(tokenizer, tok_id))
            row.append(f"{prob:.6f}")
        rows.append(row)

    fig_w = max(11.0, 2.1 * len(col_labels))
    fig_h = max(6.0, 0.4 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    for c in range(len(col_labels)):
        header_cell = table[(0, c)]
        header_cell.set_facecolor((0.85, 0.9, 1.0, 1.0))
        header_cell.set_text_props(weight="bold")

    if shade_body:
        for r in range(n_rows):
            alpha = float(np.clip(top_vals[r, 0], 0.0, 1.0))
            for c in range(1, len(col_labels)):
                cell = table[(r + 1, c)]
                cell.set_facecolor((0.1, 0.3, 1.0, alpha))

    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_topk_table_png_subblocks(
    out_path: Path,
    tokenizer,
    top_ids_attn: np.ndarray,
    top_vals_attn: np.ndarray,
    top_ids_mlp: np.ndarray,
    top_vals_mlp: np.ndarray,
    *,
    rank_label: str = "Top",
) -> None:
    n_layers, top_k = top_ids_attn.shape
    if top_ids_mlp.shape != (n_layers, top_k):
        raise ValueError("attn/mlp top-k shapes must match in subblock mode.")

    col_labels = ["Row"]
    for k in range(top_k):
        col_labels.append(f"{rank_label}-{k + 1} token")
        col_labels.append(f"{rank_label}-{k + 1} prob")

    rows = []
    row_types: list[str] = []
    for layer_idx in range(n_layers):
        row_attn = [f"layer_{layer_idx}_attn"]
        row_mlp = [f"layer_{layer_idx}_mlp"]
        for k in range(top_k):
            tok_id_a = int(top_ids_attn[layer_idx, k])
            prob_a = float(top_vals_attn[layer_idx, k])
            row_attn.append(_format_token_decoded(tokenizer, tok_id_a))
            row_attn.append(f"{prob_a:.6f}")

            tok_id_m = int(top_ids_mlp[layer_idx, k])
            prob_m = float(top_vals_mlp[layer_idx, k])
            row_mlp.append(_format_token_decoded(tokenizer, tok_id_m))
            row_mlp.append(f"{prob_m:.6f}")

        rows.append(row_attn)
        row_types.append("attn")
        rows.append(row_mlp)
        row_types.append("mlp")

    fig_w = max(11.0, 2.1 * len(col_labels))
    fig_h = max(6.0, 0.35 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.15)

    for c in range(len(col_labels)):
        header_cell = table[(0, c)]
        header_cell.set_facecolor((0.9, 0.9, 0.9, 1.0))
        header_cell.set_text_props(weight="bold")

    attn_color = (1.0, 0.82, 0.82, 1.0)
    mlp_color = (0.82, 0.88, 1.0, 1.0)
    for row_idx, row_type in enumerate(row_types):
        color = attn_color if row_type == "attn" else mlp_color
        for c in range(1, len(col_labels)):
            table[(row_idx + 1, c)].set_facecolor(color)

    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_topk(
    hidden: np.ndarray,
    top_k: int,
    device: torch.device,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    softcap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = _probs_from_hidden(
            hidden_t,
            w_u=w_u,
            b_u=b_u,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            softcap=softcap,
        ).detach().cpu().numpy()

    n_rows = probs.shape[0]
    top_ids = np.zeros((n_rows, top_k), dtype=np.int64)
    top_vals = np.zeros((n_rows, top_k), dtype=np.float32)
    bottom_ids = np.zeros((n_rows, top_k), dtype=np.int64)
    bottom_vals = np.zeros((n_rows, top_k), dtype=np.float32)
    for i in range(n_rows):
        ids_i, vals_i = _topn_for_distribution(probs[i], top_k)
        top_ids[i] = ids_i
        top_vals[i] = vals_i
        bot_ids_i, bot_vals_i = _bottomn_for_distribution(probs[i], top_k)
        bottom_ids[i] = bot_ids_i
        bottom_vals[i] = bot_vals_i
    return top_ids, top_vals, bottom_ids, bottom_vals


def _write_config_txt(
    run_dir: Path,
    args: argparse.Namespace,
    valid_examples: int,
    n_layers: int,
    hidden_dim: int,
    vocab_size: int,
    token_labels: list[str],
    elapsed_seconds: float,
    *,
    examples_high_dir: Path | None = None,
    examples_low_dir: Path | None = None,
    high_confidence_examples_written: int | None = None,
    low_confidence_examples_written: int | None = None,
) -> None:
    lines = [
        "Logit lens improved analysis",
        "=" * 72,
        f"Model: {args.model_name_or_path}",
        f"Device: {args.device if args.device else 'auto'}",
        f"Input H5: {args.input_h5}",
        f"Output dir: {run_dir}",
        f"n_examples: {args.n_examples}",
        f"top_k: {args.top_k}",
        "bottom_k: same as top_k (bottom-k tables always written)",
        f"include_embeddings: {','.join(args.include_embeddings)}",
        f"expected_guess_tokens: {args.expected_guess_tokens}",
        f"expected_probability_tokens: {args.expected_probability_tokens}",
        f"extend_probability_span: {args.extend_probability_span}",
        f"probability_token_budget: {args.expected_probability_tokens + (2 if args.extend_probability_span else 0)}",
        f"split_confidence_groups: {args.split_confidence_groups}",
        f"low_conf_threshold: {args.low_conf_threshold}",
        f"high_conf_threshold: {args.high_conf_threshold}",
        f"subblock_mode: {args.subblock_mode}",
        f"legacy_gemma_pre_postnorm_h5: {args.legacy_gemma_pre_postnorm_h5}",
        f"Valid examples: {valid_examples}",
        f"Total token positions: {len(token_labels)}",
        f"Token labels: {', '.join(token_labels)}",
        f"Total layers: {n_layers}",
        f"Hidden dim: {hidden_dim}",
        f"Vocab size: {vocab_size}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
    ]
    if args.split_confidence_groups:
        lines.extend(
            [
                f"high_confidence_examples_written: {high_confidence_examples_written}",
                f"low_confidence_examples_written: {low_confidence_examples_written}",
                f"examples_high_dir: {examples_high_dir}",
                f"examples_low_dir: {examples_low_dir}",
            ]
        )
    lines.extend(
        [
            "",
            f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )
    with open(run_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run improved logit-lens analysis.")
    parser.add_argument("--input_h5", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results base directory. Defaults to logit_lens_improved/results/<run_id>/.",
    )
    parser.add_argument("--n_examples", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        choices=list(SUPPORTED_MODEL_NAMES),
        help="Supported: Mistral-7B-Instruct-v0.1, gemma-3-12b-it, Qwen2.5-32B-Instruct.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Torch device for logit-lens math (e.g. cpu or cuda:0). "
            "Weight I/O is always selective safetensors on CPU. Default: cpu."
        ),
    )
    parser.add_argument(
        "--include_embeddings",
        nargs="+",
        choices=EMBEDDING_CHOICES,
        default=list(EMBEDDING_CHOICES),
        help="Embedding families to include. Default: all.",
    )
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--expected_probability_tokens",
        type=int,
        default=7,
        help=(
            "Expected Probability: span length as written at process time "
            "(before optional +2 from --extend_probability_span)."
        ),
    )
    parser.add_argument(
        "--extend_probability_span",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, treat stored probability span length as "
            "expected_probability_tokens + 2 (for H5s processed with "
            "--extend_probability_span)."
        ),
    )
    parser.add_argument(
        "--split_confidence_groups",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--subblock_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--legacy_gemma_pre_postnorm_h5",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Gemma subblock-only: treat H5 attn/mlp as pre–post-block-norm and apply "
            "post_attention_layernorm / post_feedforward_layernorm before logit lens "
            "(default: False; use for older Gemma H5s)."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_examples <= 0:
        raise ValueError("--n_examples must be >= 1")
    if args.top_k <= 0:
        raise ValueError("--top_k must be >= 1")
    if args.expected_guess_tokens <= 0:
        raise ValueError("--expected_guess_tokens must be >= 1")
    if args.expected_probability_tokens <= 0:
        raise ValueError("--expected_probability_tokens must be >= 1")
    if args.low_conf_threshold > args.high_conf_threshold:
        raise ValueError("--low_conf_threshold must be <= --high_conf_threshold")
    if args.legacy_gemma_pre_postnorm_h5:
        if args.model_name_or_path != "google/gemma-3-12b-it":
            raise ValueError(
                "--legacy_gemma_pre_postnorm_h5 is only valid with "
                "--model_name_or_path=google/gemma-3-12b-it "
                f"(got {args.model_name_or_path!r})."
            )
        if not args.subblock_mode:
            raise ValueError(
                "--legacy_gemma_pre_postnorm_h5 requires --subblock_mode "
                "(residual embeddings do not need post-block norms)."
            )


def _iter_example_ids(examples_group: h5py.Group) -> Iterable[str]:
    for ex_id in examples_group.keys():
        yield str(ex_id)


def main():
    start_time = time.perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    include_set = set(args.include_embeddings)
    probability_token_budget = args.expected_probability_tokens + (
        2 if args.extend_probability_span else 0
    )
    token_labels = _canonical_token_labels(
        include_set,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=probability_token_budget,
    )
    if not token_labels:
        raise ValueError("No token positions selected; include at least one embedding family.")

    device = _resolve_device(args.device)
    tokenizer, w_u, b_u, norm_weight, norm_eps, softcap = _load_unembedding(
        args.model_name_or_path, device
    )

    post_attn_gammas: list[np.ndarray] | None = None
    post_ff_gammas: list[np.ndarray] | None = None
    legacy_block_norm_eps: float | None = None
    if args.legacy_gemma_pre_postnorm_h5:
        print(
            "Legacy Gemma H5 mode: applying post-attn/post-FF norms to cached attn/mlp."
        )
        post_attn_gammas, post_ff_gammas, legacy_block_norm_eps = (
            _load_gemma_post_block_norm_gammas(args.model_name_or_path)
        )

    results_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results"
    run_base = _get_run_base_dir(results_dir)
    figures_dir = run_base / "figures"
    mean_overall_dir = figures_dir / "mean" / "overall"
    mean_high_dir = figures_dir / "mean" / "high_confidence"
    mean_low_dir = figures_dir / "mean" / "low_confidence"
    examples_dir: Path | None = None
    examples_high_dir: Path | None = None
    examples_low_dir: Path | None = None
    if args.split_confidence_groups:
        examples_high_dir = figures_dir / "examples" / "high_confidence"
        examples_low_dir = figures_dir / "examples" / "low_confidence"
        examples_high_dir.mkdir(parents=True, exist_ok=True)
        examples_low_dir.mkdir(parents=True, exist_ok=True)
    else:
        examples_dir = figures_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
    mean_overall_dir.mkdir(parents=True, exist_ok=True)
    if args.split_confidence_groups:
        mean_high_dir.mkdir(parents=True, exist_ok=True)
        mean_low_dir.mkdir(parents=True, exist_ok=True)

    components = ("attn", "mlp") if args.subblock_mode else ("res",)

    per_example: list[dict[str, dict[str, np.ndarray]]] = []
    per_example_high: list[dict[str, dict[str, np.ndarray]]] = []
    per_example_low: list[dict[str, dict[str, np.ndarray]]] = []
    sums_overall: dict[str, dict[str, np.ndarray]] = {}
    sums_high: dict[str, dict[str, np.ndarray]] = {}
    sums_low: dict[str, dict[str, np.ndarray]] = {}
    count_overall = 0
    count_high = 0
    count_low = 0
    valid_examples = 0
    n_layers: int | None = None
    hidden_dim: int | None = None

    with h5py.File(args.input_h5, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {args.input_h5}")
        examples_group = h5_file["examples"]

        for ex_id in tqdm(_iter_example_ids(examples_group), desc="Reading examples"):
            ex_group = examples_group[ex_id]
            r0 = _h5_response0_group(ex_group)
            if r0 is None:
                continue

            confidence = _read_verbalised_confidence_scalar(r0)
            if confidence is None:
                continue

            try:
                example_rows_by_component: dict[str, list[tuple[str, np.ndarray]]] = {}
                for comp in components:
                    rows = _extract_example_token_positions(
                        r0,
                        include_set,
                        comp,
                        expected_guess_tokens=args.expected_guess_tokens,
                        expected_probability_tokens=probability_token_budget,
                        subblock_mode=args.subblock_mode,
                    )
                    row_labels = [label for label, _ in rows]
                    if row_labels != token_labels:
                        raise ValueError(
                            f"Unexpected token label order for example {ex_id}, component {comp}: {row_labels}"
                        )
                    example_rows_by_component[comp] = rows
            except ValueError:
                continue

            # Validate and initialize shape metadata.
            for comp in components:
                for label, arr in example_rows_by_component[comp]:
                    if n_layers is None or hidden_dim is None:
                        n_layers = int(arr.shape[0])
                        hidden_dim = int(arr.shape[1])
                    if arr.shape != (n_layers, hidden_dim):
                        raise ValueError(
                            f"Shape mismatch for example {ex_id} ({comp}/{label}): "
                            f"expected {(n_layers, hidden_dim)}, got {arr.shape}."
                        )

            valid_examples += 1
            count_overall += 1
            if confidence >= args.high_conf_threshold:
                count_high += 1
            if confidence <= args.low_conf_threshold:
                count_low += 1

            ex_store: dict[str, dict[str, np.ndarray]] = {}
            for comp in components:
                ex_store[comp] = {}
                for label, arr in example_rows_by_component[comp]:
                    arr_f32 = np.asarray(arr, dtype=np.float32)
                    if args.legacy_gemma_pre_postnorm_h5:
                        assert post_attn_gammas is not None
                        assert post_ff_gammas is not None
                        assert legacy_block_norm_eps is not None
                        arr_f32 = _apply_legacy_gemma_post_block_norms(
                            arr_f32,
                            component=comp,
                            post_attn_gammas=post_attn_gammas,
                            post_ff_gammas=post_ff_gammas,
                            eps=legacy_block_norm_eps,
                        )
                    ex_store[comp][label] = arr_f32
                    sums_overall.setdefault(comp, {}).setdefault(
                        label,
                        np.zeros_like(arr_f32, dtype=np.float64),
                    )
                    sums_overall[comp][label] += arr_f32

                    if confidence >= args.high_conf_threshold:
                        sums_high.setdefault(comp, {}).setdefault(
                            label,
                            np.zeros_like(arr_f32, dtype=np.float64),
                        )
                        sums_high[comp][label] += arr_f32
                    if confidence <= args.low_conf_threshold:
                        sums_low.setdefault(comp, {}).setdefault(
                            label,
                            np.zeros_like(arr_f32, dtype=np.float64),
                        )
                        sums_low[comp][label] += arr_f32

            if args.split_confidence_groups:
                if (
                    confidence >= args.high_conf_threshold
                    and len(per_example_high) < args.n_examples
                ):
                    per_example_high.append(ex_store)
                if confidence <= args.low_conf_threshold and len(per_example_low) < args.n_examples:
                    per_example_low.append(ex_store)
            elif len(per_example) < args.n_examples:
                per_example.append(ex_store)

    if valid_examples == 0:
        raise ValueError("No valid examples found in the provided input_h5.")
    if args.split_confidence_groups:
        if len(per_example_high) < args.n_examples:
            raise ValueError(
                f"n_examples ({args.n_examples}) is greater than high-confidence valid examples "
                f"collected ({len(per_example_high)}); need confidence >= {args.high_conf_threshold}."
            )
        if len(per_example_low) < args.n_examples:
            raise ValueError(
                f"n_examples ({args.n_examples}) is greater than low-confidence valid examples "
                f"collected ({len(per_example_low)}); need confidence <= {args.low_conf_threshold}."
            )
    elif args.n_examples > valid_examples:
        raise ValueError(
            f"n_examples ({args.n_examples}) is greater than valid examples in file ({valid_examples})."
        )
    if n_layers is None or hidden_dim is None:
        raise ValueError("Failed to infer hidden dimensions from valid examples.")

    if args.split_confidence_groups:
        if count_high == 0:
            raise ValueError(
                f"No high-confidence examples found at threshold >= {args.high_conf_threshold}."
            )
        if count_low == 0:
            raise ValueError(
                f"No low-confidence examples found at threshold <= {args.low_conf_threshold}."
            )

    def write_tables_for_store(store: dict[str, dict[str, np.ndarray]], out_dir: Path, prefix: str) -> None:
        bottom_prefix = prefix.replace("topk_table", "bottomk_table", 1)
        if args.subblock_mode:
            for label in token_labels:
                top_ids_attn, top_vals_attn, bottom_ids_attn, bottom_vals_attn = _compute_topk(
                    store["attn"][label],
                    top_k=args.top_k,
                    device=device,
                    w_u=w_u,
                    b_u=b_u,
                    norm_weight=norm_weight,
                    norm_eps=norm_eps,
                    softcap=softcap,
                )
                top_ids_mlp, top_vals_mlp, bottom_ids_mlp, bottom_vals_mlp = _compute_topk(
                    store["mlp"][label],
                    top_k=args.top_k,
                    device=device,
                    w_u=w_u,
                    b_u=b_u,
                    norm_weight=norm_weight,
                    norm_eps=norm_eps,
                    softcap=softcap,
                )
                out_path = out_dir / f"{prefix}_{label}.png"
                _save_topk_table_png_subblocks(
                    out_path,
                    tokenizer=tokenizer,
                    top_ids_attn=top_ids_attn,
                    top_vals_attn=top_vals_attn,
                    top_ids_mlp=top_ids_mlp,
                    top_vals_mlp=top_vals_mlp,
                )
                bottom_path = out_dir / f"{bottom_prefix}_{label}.png"
                _save_topk_table_png_subblocks(
                    bottom_path,
                    tokenizer=tokenizer,
                    top_ids_attn=bottom_ids_attn,
                    top_vals_attn=bottom_vals_attn,
                    top_ids_mlp=bottom_ids_mlp,
                    top_vals_mlp=bottom_vals_mlp,
                    rank_label="Bottom",
                )
        else:
            for label in token_labels:
                top_ids, top_vals, bottom_ids, bottom_vals = _compute_topk(
                    store["res"][label],
                    top_k=args.top_k,
                    device=device,
                    w_u=w_u,
                    b_u=b_u,
                    norm_weight=norm_weight,
                    norm_eps=norm_eps,
                    softcap=softcap,
                )
                out_path = out_dir / f"{prefix}_{label}.png"
                _save_topk_table_png(
                    out_path,
                    tokenizer=tokenizer,
                    top_ids=top_ids,
                    top_vals=top_vals,
                )
                bottom_path = out_dir / f"{bottom_prefix}_{label}.png"
                _save_topk_table_png(
                    bottom_path,
                    tokenizer=tokenizer,
                    top_ids=bottom_ids,
                    top_vals=bottom_vals,
                    rank_label="Bottom",
                    shade_body=False,
                )

    def write_per_example_tables(
        stores: list[dict[str, dict[str, np.ndarray]]],
        parent_dir: Path,
    ) -> None:
        for i, store in enumerate(stores):
            ex_dir = parent_dir / f"example_{i + 1}"
            ex_dir.mkdir(parents=True, exist_ok=True)
            write_tables_for_store(store, ex_dir, prefix="topk_table")

    # Per-example tables (one subdirectory per example).
    if args.split_confidence_groups:
        assert examples_high_dir is not None and examples_low_dir is not None
        write_per_example_tables(per_example_high, examples_high_dir)
        write_per_example_tables(per_example_low, examples_low_dir)
    else:
        assert examples_dir is not None
        write_per_example_tables(per_example, examples_dir)

    # Mean over all valid examples.
    mean_overall: dict[str, dict[str, np.ndarray]] = {}
    for comp in components:
        mean_overall[comp] = {}
        for label in token_labels:
            mean_overall[comp][label] = np.asarray(
                sums_overall[comp][label] / max(float(count_overall), EPS),
                dtype=np.float32,
            )
    write_tables_for_store(mean_overall, mean_overall_dir, prefix="topk_table_mean")

    # Optional confidence groups.
    if args.split_confidence_groups:
        mean_high: dict[str, dict[str, np.ndarray]] = {}
        mean_low: dict[str, dict[str, np.ndarray]] = {}
        for comp in components:
            mean_high[comp] = {}
            mean_low[comp] = {}
            for label in token_labels:
                mean_high[comp][label] = np.asarray(
                    sums_high[comp][label] / max(float(count_high), EPS),
                    dtype=np.float32,
                )
                mean_low[comp][label] = np.asarray(
                    sums_low[comp][label] / max(float(count_low), EPS),
                    dtype=np.float32,
                )

        write_tables_for_store(
            mean_high,
            mean_high_dir,
            prefix="topk_table_mean_high_confidence",
        )
        write_tables_for_store(
            mean_low,
            mean_low_dir,
            prefix="topk_table_mean_low_confidence",
        )

    elapsed = time.perf_counter() - start_time
    _write_config_txt(
        run_dir=run_base,
        args=args,
        valid_examples=valid_examples,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        vocab_size=int(w_u.shape[0]),
        token_labels=token_labels,
        elapsed_seconds=elapsed,
        examples_high_dir=examples_high_dir,
        examples_low_dir=examples_low_dir,
        high_confidence_examples_written=(
            len(per_example_high) if args.split_confidence_groups else None
        ),
        low_confidence_examples_written=(
            len(per_example_low) if args.split_confidence_groups else None
        ),
    )
    print(f"Finished. Outputs written to: {run_base}")


if __name__ == "__main__":
    main()
