"""
Attention-OV logit lens over processed H5 residual embeddings.

For each residual row i (row 0 is the input embedding; row k >= 1 is the
residual after block k - 1):

- ``None``: the usual final-norm logit lens of that vector.
- ``attn_j`` for j = i .. last block: pre-norm, V, 100% attention on this
  vector (KV heads repeated onto the query heads, then O), Gemma
  post-attention RMSNorm when the model has one, then the same final-norm
  logit lens. With ``--gain`` (default), the lens sees the source residual
  plus that attention output. With ``--no-gain``, it sees the attention
  output alone.

Means are averaged in residual space first, matching the improved logit lens.
With ``--ultimate`` (default), each residual row also averages the full softmax
of the plain lens and every later attention read. ``--ultimate_detailed`` keeps
the per-attention tables as well.
Probabilities in this script's tables are percentages rounded to 2 decimal places.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from direction_analysis.h5_probability_directions import read_decoded_tokens
from logit_lens_improved.run_logit_lens_improved import (
    EMBEDDING_CHOICES,
    EPS,
    LOGIT_LENS_SPECS,
    SUPPORTED_MODEL_NAMES,
    _apply_rmsnorm,
    _build_weight_map,
    _canonical_token_labels,
    _disable_table_mathtext,
    _extract_example_token_positions,
    _format_token_decoded,
    _get_run_base_dir,
    _h5_response0_group,
    _iter_example_ids,
    _load_first_matching_weight,
    _load_torch_tensor,
    _load_unembedding,
    _probability_span_len,
    _probs_from_hidden,
    _read_verbalised_confidence_scalar,
    _resolve_device,
    _select_text_config,
)
from process_generations.process_generations_more_embs_from_h5 import (
    L_VS_UN_SPAN_INDEX,
    configure_prefix_tokens_for_model,
    linguistic_span_token_is_l_or_un,
)

LENS_ROW_CHUNK = 64
RESIDUAL_COMPONENT = "res"

# Per-block attention paths. Final-norm / unembedding paths stay in LOGIT_LENS_SPECS.
ATTN_BLOCK_SPECS: dict[str, dict] = {
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "layers_path": ("model", "layers"),
        "layers_path_fallbacks": (),
        "post_attention_layernorm": False,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "layers_path": ("model", "layers"),
        "layers_path_fallbacks": (),
        "post_attention_layernorm": False,
    },
    "google/gemma-3-12b-it": {
        "layers_path": ("language_model", "model", "layers"),
        "layers_path_fallbacks": (
            ("language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("model", "language_model", "layers"),
        ),
        "post_attention_layernorm": True,
    },
}


@dataclass
class ModelArch:
    model_name: str
    n_query_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    n_blocks: int
    rmsnorm_gain: str
    use_post_attention_norm: bool
    layers_prefix: tuple[str, ...]


@dataclass
class AttnOVBlock:
    layer_idx: int
    input_norm: torch.Tensor
    v_weight: torch.Tensor
    v_bias: torch.Tensor | None
    o_weight: torch.Tensor
    o_bias: torch.Tensor | None
    post_attn_norm: torch.Tensor | None
    norm_eps: float


@dataclass
class TopKEntry:
    rank: int
    token_id: int
    token: str
    probability_percent: float


@dataclass
class DetailRow:
    row: str
    attention_block: int | None
    topk: list[TopKEntry]
    top1_probability: float
    l2_norm: float
    l2_norm_raw: float


@dataclass
class HiddenJob:
    out_dir: Path
    label: str
    hidden: np.ndarray


@dataclass
class ResidualCorpus:
    token_labels: list[str]
    n_layers: int
    hidden_dim: int
    valid_examples: int
    count_overall: int
    count_high: int
    count_low: int
    sums_overall: dict[str, np.ndarray]
    sums_high: dict[str, np.ndarray]
    sums_low: dict[str, np.ndarray]
    per_example: list[dict[str, np.ndarray]]
    per_example_high: list[dict[str, np.ndarray]]
    per_example_low: list[dict[str, np.ndarray]]
    skipped_l_vs_un: int


def _cfg_int(cfg, *names: str, default: int | None = None) -> int:
    for name in names:
        val = getattr(cfg, name, None)
        if val is not None:
            return int(val)
    if default is not None:
        return int(default)
    raise ValueError(f"None of {names} found on config {type(cfg).__name__}")


def _probability_percent(prob: float) -> float:
    """Percentage rounded to 2 decimal places, shared by PNG text and JSON."""
    return float(f"{float(prob) * 100.0:.2f}")


def _l2_norm_3sf(value: float) -> float:
    """L2 norm rounded to 3 significant figures, shared by PNG text and JSON."""
    return float(f"{float(value):.3g}")


def _layer_weight_key(prefix: tuple[str, ...], layer_idx: int, leaf: str) -> str:
    return f"{'.'.join(prefix)}.{layer_idx}.{leaf}"


def _norm_gamma(raw: torch.Tensor, gain: str) -> torch.Tensor:
    if gain == "one_plus_weight":
        return (1.0 + raw).to(dtype=torch.float32)
    if gain == "plain_weight":
        return raw.to(dtype=torch.float32)
    raise ValueError(f"Unknown rmsnorm_gain={gain!r}")


def _read_model_arch(model_name: str, weight_map: dict[str, str]) -> ModelArch:
    if model_name not in LOGIT_LENS_SPECS or model_name not in ATTN_BLOCK_SPECS:
        raise ValueError(
            f"Unsupported model_name_or_path={model_name!r}. "
            f"Supported: {list(SUPPORTED_MODEL_NAMES)}"
        )
    lens_spec = LOGIT_LENS_SPECS[model_name]
    block_spec = ATTN_BLOCK_SPECS[model_name]
    trust_remote_code = bool(lens_spec["trust_remote_code"])
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    cfg = _select_text_config(config, config_root=lens_spec.get("config_root"))

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
    n_blocks = _cfg_int(cfg, "num_hidden_layers", "n_layer")
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

    tried: list[str] = []
    layers_prefix: tuple[str, ...] | None = None
    for prefix in (tuple(block_spec["layers_path"]), *tuple(block_spec["layers_path_fallbacks"])):
        key = _layer_weight_key(prefix, 0, "self_attn.v_proj.weight")
        tried.append(key)
        if key in weight_map:
            layers_prefix = prefix
            break
    if layers_prefix is None:
        raise RuntimeError(
            "Could not resolve attention-block weight prefix. Tried keys:\n  "
            + "\n  ".join(tried)
        )
    print(f"Attention weights prefix: {'.'.join(layers_prefix)}")

    return ModelArch(
        model_name=model_name,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        n_blocks=n_blocks,
        rmsnorm_gain=str(lens_spec["rmsnorm_gain"]),
        use_post_attention_norm=bool(block_spec["post_attention_layernorm"]),
        layers_prefix=layers_prefix,
    )


def _load_optional_bias(
    weight_map: dict[str, str], weight_key: str, *, expected_numel: int
) -> torch.Tensor | None:
    if not weight_key.endswith(".weight"):
        return None
    bias_key = weight_key[: -len(".weight")] + ".bias"
    shard = weight_map.get(bias_key)
    if shard is None:
        return None
    bias = _load_torch_tensor(shard, bias_key)
    print(f"Loaded bias from key {bias_key} (shape={tuple(bias.shape)})")
    if tuple(bias.shape) != (expected_numel,):
        raise ValueError(
            f"Bias {bias_key} has shape {tuple(bias.shape)}, expected {(expected_numel,)}."
        )
    return bias


def load_attn_ov_block(
    weight_map: dict[str, str],
    arch: ModelArch,
    layer_idx: int,
    device: torch.device,
    norm_eps: float,
) -> AttnOVBlock:
    """Load one block's pre-norm, V, O, and optional Gemma post-attention norm."""
    prefix = arch.layers_prefix
    v_key = _layer_weight_key(prefix, layer_idx, "self_attn.v_proj.weight")
    o_key = _layer_weight_key(prefix, layer_idx, "self_attn.o_proj.weight")
    in_key = _layer_weight_key(prefix, layer_idx, "input_layernorm.weight")

    v_weight, _ = _load_first_matching_weight(weight_map, [v_key], label=f"layer {layer_idx} v_proj")
    o_weight, _ = _load_first_matching_weight(weight_map, [o_key], label=f"layer {layer_idx} o_proj")
    in_raw, _ = _load_first_matching_weight(
        weight_map, [in_key], label=f"layer {layer_idx} input_layernorm"
    )
    input_norm = _norm_gamma(in_raw, arch.rmsnorm_gain)

    n_v = arch.n_kv_heads * arch.head_dim
    n_o_in = arch.n_query_heads * arch.head_dim
    if tuple(v_weight.shape) != (n_v, arch.hidden_size):
        raise ValueError(
            f"Layer {layer_idx} v_proj shape {tuple(v_weight.shape)} != {(n_v, arch.hidden_size)}."
        )
    if tuple(o_weight.shape) != (arch.hidden_size, n_o_in):
        raise ValueError(
            f"Layer {layer_idx} o_proj shape {tuple(o_weight.shape)} != "
            f"{(arch.hidden_size, n_o_in)}."
        )
    if tuple(input_norm.shape) != (arch.hidden_size,):
        raise ValueError(
            f"Layer {layer_idx} input_layernorm shape {tuple(input_norm.shape)} != "
            f"{(arch.hidden_size,)}."
        )

    v_bias = _load_optional_bias(weight_map, v_key, expected_numel=n_v)
    o_bias = _load_optional_bias(weight_map, o_key, expected_numel=arch.hidden_size)

    post_attn_norm: torch.Tensor | None = None
    if arch.use_post_attention_norm:
        post_key = _layer_weight_key(prefix, layer_idx, "post_attention_layernorm.weight")
        post_raw, _ = _load_first_matching_weight(
            weight_map, [post_key], label=f"layer {layer_idx} post_attention_layernorm"
        )
        post_attn_norm = _norm_gamma(post_raw, arch.rmsnorm_gain)
        if tuple(post_attn_norm.shape) != (arch.hidden_size,):
            raise ValueError(
                f"Layer {layer_idx} post_attention_layernorm shape {tuple(post_attn_norm.shape)} "
                f"!= {(arch.hidden_size,)}."
            )

    def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=torch.float32)

    return AttnOVBlock(
        layer_idx=layer_idx,
        input_norm=_move(input_norm),
        v_weight=_move(v_weight),
        v_bias=_move(v_bias),
        o_weight=_move(o_weight),
        o_bias=_move(o_bias),
        post_attn_norm=_move(post_attn_norm),
        norm_eps=float(norm_eps),
    )


def apply_attn_ov(
    hidden: torch.Tensor,
    block: AttnOVBlock,
    arch: ModelArch,
) -> torch.Tensor:
    """100% self-attention OV: pre-norm, V, repeat KV heads, O, optional post-norm.

    ``hidden`` is ``[N, H]`` or ``[H]``. Q, K, RoPE, and QK-norm are unused because
    every query head attends only to this vector. The returned tensor is the
    attention output, not the residual plus that output.
    """
    squeeze = hidden.ndim == 1
    if squeeze:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim != 2:
        raise ValueError(f"apply_attn_ov expected rank 1 or 2, got {hidden.ndim}.")

    normed = _apply_rmsnorm(hidden, block.input_norm, block.norm_eps)
    values = F.linear(normed, block.v_weight, block.v_bias)
    values = values.view(values.shape[0], arch.n_kv_heads, arch.head_dim)
    n_rep = arch.n_query_heads // arch.n_kv_heads
    if n_rep != 1:
        values = values.repeat_interleave(n_rep, dim=1)
    values = values.reshape(values.shape[0], arch.n_query_heads * arch.head_dim)
    out = F.linear(values, block.o_weight, block.o_bias)
    if block.post_attn_norm is not None:
        out = _apply_rmsnorm(out, block.post_attn_norm, block.norm_eps)
    if squeeze:
        return out.squeeze(0)
    return out


def _topk_from_probs(probs: torch.Tensor, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    vals, ids = torch.topk(probs, k=top_k, dim=-1)
    return (
        ids.detach().cpu().numpy().astype(np.int64, copy=False),
        vals.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _topk_from_distributions(dists: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k of probability rows ``[N, V]`` without holding every row on device at once."""
    n_rows = int(dists.shape[0])
    top_ids = np.zeros((n_rows, top_k), dtype=np.int64)
    top_vals = np.zeros((n_rows, top_k), dtype=np.float32)
    for start in range(0, n_rows, LENS_ROW_CHUNK):
        end = min(start + LENS_ROW_CHUNK, n_rows)
        chunk = torch.as_tensor(np.ascontiguousarray(dists[start:end]), dtype=torch.float32)
        vals, ids = torch.topk(chunk, k=top_k, dim=-1)
        top_ids[start:end] = ids.numpy().astype(np.int64, copy=False)
        top_vals[start:end] = vals.numpy().astype(np.float32, copy=False)
        del chunk
    return top_ids, top_vals


def logit_lens_topk(
    hidden: np.ndarray,
    top_k: int,
    device: torch.device,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    softcap: float | None,
    scatter: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Final-norm logit lens. ``hidden`` is ``[N, H]``; returns ids/probs ``[N, K]``.

    ``scatter(start, probs)`` receives each chunk of full softmax rows when set.
    """
    if hidden.ndim != 2:
        raise ValueError(f"logit_lens_topk expected [N, H], got {hidden.shape}.")
    n_rows = int(hidden.shape[0])
    top_ids = np.zeros((n_rows, top_k), dtype=np.int64)
    top_vals = np.zeros((n_rows, top_k), dtype=np.float32)
    for start in range(0, n_rows, LENS_ROW_CHUNK):
        end = min(start + LENS_ROW_CHUNK, n_rows)
        hidden_t = torch.as_tensor(hidden[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = _probs_from_hidden(
                hidden_t,
                w_u=w_u,
                b_u=b_u,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                softcap=softcap,
            )
            ids_chunk, vals_chunk = _topk_from_probs(probs, top_k)
            if scatter is not None:
                scatter(start, probs.detach().to(dtype=torch.float32).cpu().numpy())
        top_ids[start:end] = ids_chunk
        top_vals[start:end] = vals_chunk
        del hidden_t, probs
    return top_ids, top_vals


def ov_logit_lens_topk(
    hidden: np.ndarray,
    block: AttnOVBlock,
    arch: ModelArch,
    top_k: int,
    device: torch.device,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    softcap: float | None,
    *,
    gain: bool,
    scatter: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OV-transform ``hidden`` ``[N, H]``, then the final-norm logit lens.

    When ``gain`` is true, the lens input is the source residual plus the
    attention output. Otherwise it is the attention output alone.
    The returned L2 norm is always the attention output alone, before the
    final RMSNorm, whether or not the residual is added into the lens.
    """
    if hidden.ndim != 2:
        raise ValueError(f"ov_logit_lens_topk expected [N, H], got {hidden.shape}.")
    n_rows = int(hidden.shape[0])
    top_ids = np.zeros((n_rows, top_k), dtype=np.int64)
    top_vals = np.zeros((n_rows, top_k), dtype=np.float32)
    l2_norms = np.zeros((n_rows,), dtype=np.float32)
    for start in range(0, n_rows, LENS_ROW_CHUNK):
        end = min(start + LENS_ROW_CHUNK, n_rows)
        hidden_t = torch.as_tensor(hidden[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            attn_out = apply_attn_ov(hidden_t, block, arch)
            lens_input = hidden_t + attn_out if gain else attn_out
            norm_chunk = torch.linalg.vector_norm(attn_out, ord=2, dim=-1)
            probs = _probs_from_hidden(
                lens_input,
                w_u=w_u,
                b_u=b_u,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                softcap=softcap,
            )
            ids_chunk, vals_chunk = _topk_from_probs(probs, top_k)
            if scatter is not None:
                scatter(start, probs.detach().to(dtype=torch.float32).cpu().numpy())
        top_ids[start:end] = ids_chunk
        top_vals[start:end] = vals_chunk
        l2_norms[start:end] = norm_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        del hidden_t, attn_out, lens_input, norm_chunk, probs
    return top_ids, top_vals, l2_norms


def _detail_row(
    label: str,
    attention_block: int | None,
    token_ids: np.ndarray,
    probs: np.ndarray,
    l2_norm: float,
    tokenizer,
) -> DetailRow:
    entries: list[TopKEntry] = []
    for rank, (token_id, prob) in enumerate(zip(token_ids, probs), start=1):
        tid = int(token_id)
        entries.append(
            TopKEntry(
                rank=rank,
                token_id=tid,
                token=_format_token_decoded(tokenizer, tid),
                probability_percent=_probability_percent(float(prob)),
            )
        )
    raw_norm = float(l2_norm)
    return DetailRow(
        row=label,
        attention_block=attention_block,
        topk=entries,
        top1_probability=float(probs[0]),
        l2_norm=_l2_norm_3sf(raw_norm),
        l2_norm_raw=raw_norm,
    )


def detail_rows_for_source(
    source_layer: int,
    n_blocks: int,
    plain_ids: np.ndarray,
    plain_vals: np.ndarray,
    plain_l2: float,
    block_ids: list[np.ndarray],
    block_vals: list[np.ndarray],
    block_l2s: list[np.ndarray],
    tokenizer,
) -> list[DetailRow]:
    """``None`` plus ``attn_j`` for every block ``j >= source_layer``."""
    rows = [
        _detail_row("None", None, plain_ids, plain_vals, plain_l2, tokenizer),
    ]
    if source_layer < n_blocks:
        for block_idx in range(source_layer, n_blocks):
            rows.append(
                _detail_row(
                    f"attn_{block_idx}",
                    block_idx,
                    block_ids[block_idx][source_layer],
                    block_vals[block_idx][source_layer],
                    float(block_l2s[block_idx][source_layer]),
                    tokenizer,
                )
            )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _detail_payload(position: str, source_layer: int, rows: list[DetailRow]) -> dict:
    return {
        "position": position,
        "source_layer": source_layer,
        "rows": [
            {
                "row": row.row,
                "attention_block": row.attention_block,
                "l2_norm": row.l2_norm,
                "topk": [
                    {
                        "rank": entry.rank,
                        "token_id": entry.token_id,
                        "token": entry.token,
                        "probability_percent": entry.probability_percent,
                    }
                    for entry in row.topk
                ],
            }
            for row in rows
        ],
    }


def _save_detail_table_png(path: Path, rows: list[DetailRow]) -> None:
    if not rows:
        raise ValueError("Detail table has no rows.")
    top_k = len(rows[0].topk)
    col_labels = ["Row"]
    for k in range(top_k):
        col_labels.append(f"Top-{k + 1} token")
        col_labels.append(f"Top-{k + 1} prob (%)")
    col_labels.append("L2 norm")
    norm_col = len(col_labels) - 1

    cell_text: list[list[str]] = []
    for row in rows:
        cells = [row.row]
        for entry in row.topk:
            cells.append(entry.token)
            cells.append(f"{entry.probability_percent:.2f}%")
        cells.append(f"{row.l2_norm:.3g}")
        cell_text.append(cells)

    fig_w = max(11.0, 2.1 * len(col_labels))
    fig_h = max(6.0, 0.4 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
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

    raw_norms = [row.l2_norm_raw for row in rows]
    norm_lo = min(raw_norms)
    norm_hi = max(raw_norms)
    norm_span = norm_hi - norm_lo
    for r, row in enumerate(rows):
        alpha = float(np.clip(row.top1_probability, 0.0, 1.0))
        for c in range(1, norm_col):
            table[(r + 1, c)].set_facecolor((0.1, 0.3, 1.0, alpha))
        if norm_span == 0.0:
            red_t = 1.0
        else:
            red_t = (row.l2_norm_raw - norm_lo) / norm_span
        table[(r + 1, norm_col)].set_facecolor((1.0, 1.0 - 0.85 * red_t, 1.0 - 0.85 * red_t))

    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_token_cell(token_ids: np.ndarray, tokenizer) -> dict:
    return {
        "tokens": [_format_token_decoded(tokenizer, int(token_id)) for token_id in token_ids],
        "token_ids": [int(token_id) for token_id in token_ids],
    }


def summary_cells_for_position(
    n_layers: int,
    n_blocks: int,
    plain_ids: np.ndarray,
    block_ids: list[np.ndarray],
    tokenizer,
) -> tuple[list[str], list[str], list[list[dict | None]]]:
    """None column plus the attention triangle. Cell (i, attn_j) is filled when j >= i."""
    row_labels = [f"layer_{i}" for i in range(n_layers)]
    column_labels = ["None", *[f"attn_{j}" for j in range(n_blocks)]]
    cells: list[list[dict | None]] = []
    for source_layer in range(n_layers):
        row: list[dict | None] = [_summary_token_cell(plain_ids[source_layer], tokenizer)]
        for block_idx in range(n_blocks):
            if block_idx < source_layer:
                row.append(None)
                continue
            row.append(_summary_token_cell(block_ids[block_idx][source_layer], tokenizer))
        cells.append(row)
    return row_labels, column_labels, cells


def _save_summary_table_png(
    path: Path,
    row_labels: list[str],
    column_labels: list[str],
    cells: list[list[dict | None]],
) -> None:
    n_rows = len(row_labels)
    n_cols = len(column_labels)
    header = ["Layer", *column_labels]
    cell_text: list[list[str]] = []
    for row_label, row in zip(row_labels, cells):
        rendered = [row_label]
        for cell in row:
            if cell is None:
                rendered.append("")
            else:
                rendered.append(", ".join(cell["tokens"]))
        cell_text.append(rendered)

    top_k = 1
    for row in cells:
        for cell in row:
            if cell is not None:
                top_k = max(top_k, len(cell["tokens"]))
    fig_w = max(10.0, 0.55 * n_cols * top_k + 1.5)
    fig_h = max(6.0, 0.38 * (n_rows + 1) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=header,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7 if n_cols <= 40 else 5)
    if hasattr(table, "auto_set_column_width"):
        table.auto_set_column_width(list(range(len(header))))
    table.scale(1.0, 1.4)

    for c in range(len(header)):
        header_cell = table[(0, c)]
        header_cell.set_text_props(weight="bold")

    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _comparison_topk_entries(token_ids: np.ndarray, probs: np.ndarray, tokenizer) -> list[dict]:
    entries: list[dict] = []
    for rank, (token_id, prob) in enumerate(zip(token_ids, probs), start=1):
        tid = int(token_id)
        entries.append(
            {
                "rank": rank,
                "token_id": tid,
                "token": _format_token_decoded(tokenizer, tid),
                "probability_percent": _probability_percent(float(prob)),
            }
        )
    return entries


def _ultimate_comparison_rows(
    plain_ids: np.ndarray,
    plain_vals: np.ndarray,
    ultimate_ids: np.ndarray,
    ultimate_vals: np.ndarray,
    tokenizer,
) -> list[dict]:
    rows: list[dict] = []
    n_layers = int(plain_ids.shape[0])
    for layer in range(n_layers):
        rows.append(
            {
                "layer": layer,
                "lens": "vanilla",
                "row": f"layer_{layer} vanilla",
                "topk": _comparison_topk_entries(plain_ids[layer], plain_vals[layer], tokenizer),
                "top1_probability": float(plain_vals[layer, 0]),
            }
        )
        rows.append(
            {
                "layer": layer,
                "lens": "ultimate",
                "row": f"layer_{layer} ultimate",
                "topk": _comparison_topk_entries(
                    ultimate_ids[layer], ultimate_vals[layer], tokenizer
                ),
                "top1_probability": float(ultimate_vals[layer, 0]),
            }
        )
    return rows


def _save_ultimate_compact_png(path: Path, rows: list[dict]) -> None:
    n_layers = len(rows) // 2
    col_labels = ["Layer", "vanilla logit lens", "ultimate attn lens"]
    cell_text: list[list[str]] = []
    top1 = np.zeros((n_layers, 2), dtype=np.float32)
    for layer in range(n_layers):
        vanilla = rows[2 * layer]
        ultimate = rows[2 * layer + 1]
        cell_text.append(
            [
                f"layer_{layer}",
                ", ".join(entry["token"] for entry in vanilla["topk"]),
                ", ".join(entry["token"] for entry in ultimate["topk"]),
            ]
        )
        top1[layer, 0] = vanilla["top1_probability"]
        top1[layer, 1] = ultimate["top1_probability"]

    top_k = max(1, len(rows[0]["topk"]))
    fig_w = max(12.0, 1.8 * top_k + 4.0)
    fig_h = max(6.0, 0.4 * n_layers + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
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
    for layer in range(n_layers):
        for col in range(2):
            alpha = float(np.clip(top1[layer, col], 0.0, 1.0))
            table[(layer + 1, col + 1)].set_facecolor((0.1, 0.3, 1.0, alpha))
    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_ultimate_detail_png(path: Path, rows: list[dict]) -> None:
    top_k = len(rows[0]["topk"])
    col_labels = ["Row"]
    for k in range(top_k):
        col_labels.append(f"Top-{k + 1} token")
        col_labels.append(f"Top-{k + 1} prob (%)")
    cell_text: list[list[str]] = []
    for row in rows:
        cells = [row["row"]]
        for entry in row["topk"]:
            cells.append(entry["token"])
            cells.append(f"{entry['probability_percent']:.2f}%")
        cell_text.append(cells)

    fig_w = max(11.0, 2.1 * len(col_labels))
    fig_h = max(6.0, 0.35 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
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
    for r, row in enumerate(rows):
        alpha = float(np.clip(row["top1_probability"], 0.0, 1.0))
        for c in range(1, len(col_labels)):
            table[(r + 1, c)].set_facecolor((0.1, 0.3, 1.0, alpha))
    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_ultimate_comparison(
    pos_dir: Path,
    label: str,
    plain_ids: np.ndarray,
    plain_vals: np.ndarray,
    ultimate_ids: np.ndarray,
    ultimate_vals: np.ndarray,
    tokenizer,
) -> None:
    rows = _ultimate_comparison_rows(
        plain_ids, plain_vals, ultimate_ids, ultimate_vals, tokenizer
    )
    _save_ultimate_compact_png(pos_dir / "vanilla_vs_ultimate.png", rows)
    _save_ultimate_detail_png(pos_dir / "vanilla_vs_ultimate_detail.png", rows)
    _write_json(
        pos_dir / "vanilla_vs_ultimate.json",
        {
            "position": label,
            "rows": [
                {
                    "layer": row["layer"],
                    "lens": row["lens"],
                    "row": row["row"],
                    "topk": row["topk"],
                }
                for row in rows
            ],
        },
    )


def write_position_outputs(
    out_dir: Path,
    label: str,
    n_layers: int,
    n_blocks: int,
    plain_ids: np.ndarray,
    plain_vals: np.ndarray,
    plain_l2: np.ndarray,
    block_ids: list[np.ndarray],
    block_vals: list[np.ndarray],
    block_l2s: list[np.ndarray],
    tokenizer,
    *,
    write_attention_details: bool,
    ultimate_ids: np.ndarray | None = None,
    ultimate_vals: np.ndarray | None = None,
) -> None:
    """Write per-position tables. Attention details are optional; ultimate comparison is optional."""
    pos_dir = out_dir / label
    pos_dir.mkdir(parents=True, exist_ok=True)
    if write_attention_details:
        for source_layer in range(n_layers):
            rows = detail_rows_for_source(
                source_layer,
                n_blocks,
                plain_ids[source_layer],
                plain_vals[source_layer],
                float(plain_l2[source_layer]),
                block_ids,
                block_vals,
                block_l2s,
                tokenizer,
            )
            stem = pos_dir / f"layer_{source_layer}"
            _save_detail_table_png(stem.with_suffix(".png"), rows)
            _write_json(stem.with_suffix(".json"), _detail_payload(label, source_layer, rows))

        row_labels, column_labels, cells = summary_cells_for_position(
            n_layers, n_blocks, plain_ids, block_ids, tokenizer
        )
        _save_summary_table_png(pos_dir / "summary.png", row_labels, column_labels, cells)
        _write_json(
            pos_dir / "summary.json",
            {
                "position": label,
                "row_labels": row_labels,
                "column_labels": column_labels,
                "cells": cells,
            },
        )
    if ultimate_ids is not None and ultimate_vals is not None:
        _write_ultimate_comparison(
            pos_dir,
            label,
            plain_ids,
            plain_vals,
            ultimate_ids,
            ultimate_vals,
            tokenizer,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an attention-OV logit lens on residual embeddings."
    )
    parser.add_argument("--input_h5", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results base directory. Defaults to attn_logit_lens/results/<run_id>/.",
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
        "--linguistic_confidence_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, treat the H5 as a linguistic Confidence: run. "
            "Does not change the probability-token budget; use with "
            "--l_vs_un to apply the DLA L/Un example filter."
        ),
    )
    parser.add_argument(
        "--l_vs_un",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true together with --linguistic_confidence_prompt, keep only examples "
            "whose Confidence: span token at index 6 decodes as L or Un (same as DLA). "
            "Ignored when --linguistic_confidence_prompt is off."
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
        "--gain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), each attn_j logit lens reads the source residual "
            "plus that block's attention output. If false (--no-gain), it reads "
            "the attention output alone. The None rows are unchanged."
        ),
    )
    parser.add_argument(
        "--ultimate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), average the full softmax of the plain logit lens "
            "and every later attention read at each residual row, then report "
            "that average beside the plain lens. If false (--no-ultimate), "
            "write only the per-attention tables."
        ),
    )
    parser.add_argument(
        "--ultimate_detailed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true together with --ultimate, also write the per-attention "
            "layer tables and summary. Ignored when --ultimate is off."
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


def _read_residual_corpus(args: argparse.Namespace, token_labels: list[str]) -> ResidualCorpus:
    include_set = set(args.include_embeddings)
    probability_token_budget = args.expected_probability_tokens + (
        2 if args.extend_probability_span else 0
    )
    apply_l_vs_un = bool(args.linguistic_confidence_prompt) and bool(args.l_vs_un)

    per_example: list[dict[str, np.ndarray]] = []
    per_example_high: list[dict[str, np.ndarray]] = []
    per_example_low: list[dict[str, np.ndarray]] = []
    sums_overall: dict[str, np.ndarray] = {}
    sums_high: dict[str, np.ndarray] = {}
    sums_low: dict[str, np.ndarray] = {}
    count_overall = 0
    count_high = 0
    count_low = 0
    valid_examples = 0
    skipped_l_vs_un = 0
    n_layers: int | None = None
    hidden_dim: int | None = None

    try:
        h5_file = h5py.File(args.input_h5, "r", locking=False)
    except TypeError:
        h5_file = h5py.File(args.input_h5, "r")
    with h5_file:
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
                rows = _extract_example_token_positions(
                    r0,
                    include_set,
                    RESIDUAL_COMPONENT,
                    expected_guess_tokens=args.expected_guess_tokens,
                    expected_probability_tokens=probability_token_budget,
                    subblock_mode=False,
                )
                row_labels = [label for label, _ in rows]
                if row_labels != token_labels:
                    raise ValueError(
                        f"Unexpected token label order for example {ex_id}: {row_labels}"
                    )
            except ValueError:
                continue

            for label, arr in rows:
                if n_layers is None or hidden_dim is None:
                    n_layers = int(arr.shape[0])
                    hidden_dim = int(arr.shape[1])
                if arr.shape != (n_layers, hidden_dim):
                    raise ValueError(
                        f"Shape mismatch for example {ex_id} ({label}): "
                        f"expected {(n_layers, hidden_dim)}, got {arr.shape}."
                    )

            if apply_l_vs_un:
                span_len = _probability_span_len(
                    r0, RESIDUAL_COMPONENT, subblock_mode=False
                )
                decoded = read_decoded_tokens(r0)
                if (
                    span_len is None
                    or span_len <= L_VS_UN_SPAN_INDEX
                    or not decoded
                    or not linguistic_span_token_is_l_or_un(decoded)
                ):
                    skipped_l_vs_un += 1
                    continue

            valid_examples += 1
            count_overall += 1
            if confidence >= args.high_conf_threshold:
                count_high += 1
            if confidence <= args.low_conf_threshold:
                count_low += 1

            ex_store: dict[str, np.ndarray] = {}
            for label, arr in rows:
                arr_f32 = np.asarray(arr, dtype=np.float32)
                ex_store[label] = arr_f32
                sums_overall.setdefault(label, np.zeros_like(arr_f32, dtype=np.float64))
                sums_overall[label] += arr_f32
                if confidence >= args.high_conf_threshold:
                    sums_high.setdefault(label, np.zeros_like(arr_f32, dtype=np.float64))
                    sums_high[label] += arr_f32
                if confidence <= args.low_conf_threshold:
                    sums_low.setdefault(label, np.zeros_like(arr_f32, dtype=np.float64))
                    sums_low[label] += arr_f32

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
    if apply_l_vs_un:
        print(
            f"L_vs_Un skipped {skipped_l_vs_un} examples whose span-index-"
            f"{L_VS_UN_SPAN_INDEX} token was not L or Un."
        )
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
        if count_high == 0:
            raise ValueError(
                f"No high-confidence examples found at threshold >= {args.high_conf_threshold}."
            )
        if count_low == 0:
            raise ValueError(
                f"No low-confidence examples found at threshold <= {args.low_conf_threshold}."
            )
    elif args.n_examples > valid_examples:
        raise ValueError(
            f"n_examples ({args.n_examples}) is greater than valid examples in file ({valid_examples})."
        )
    if n_layers is None or hidden_dim is None:
        raise ValueError("Failed to infer hidden dimensions from valid examples.")

    return ResidualCorpus(
        token_labels=token_labels,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        valid_examples=valid_examples,
        count_overall=count_overall,
        count_high=count_high,
        count_low=count_low,
        sums_overall=sums_overall,
        sums_high=sums_high,
        sums_low=sums_low,
        per_example=per_example,
        per_example_high=per_example_high,
        per_example_low=per_example_low,
        skipped_l_vs_un=skipped_l_vs_un,
    )


def _mean_store(
    sums: dict[str, np.ndarray], count: int, token_labels: list[str]
) -> dict[str, np.ndarray]:
    return {
        label: np.asarray(sums[label] / max(float(count), EPS), dtype=np.float32)
        for label in token_labels
    }


def _jobs_for_store(store: dict[str, np.ndarray], out_dir: Path, token_labels: list[str]) -> list[HiddenJob]:
    return [HiddenJob(out_dir=out_dir, label=label, hidden=store[label]) for label in token_labels]


def _run_lenses(
    jobs: list[HiddenJob],
    arch: ModelArch,
    weight_map: dict[str, str],
    device: torch.device,
    top_k: int,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    softcap: float | None,
    *,
    gain: bool,
    ultimate: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return plain ids/vals/l2, per-block ids/vals/l2, and optional ultimate top-k."""
    if not jobs:
        raise ValueError("No residual vectors to lens.")
    n_jobs = len(jobs)
    n_layers = int(jobs[0].hidden.shape[0])
    hidden_dim = int(jobs[0].hidden.shape[1])
    for job in jobs:
        if job.hidden.shape != (n_layers, hidden_dim):
            raise ValueError(
                f"Job {job.label} shape {job.hidden.shape} != {(n_layers, hidden_dim)}."
            )

    print(f"Plain logit lens on {n_jobs * n_layers} residual rows.")
    vocab = int(w_u.shape[0])
    ultimate_sum: np.ndarray | None = None
    ultimate_count: np.ndarray | None = None
    if ultimate:
        ultimate_sum = np.zeros((n_jobs, n_layers, vocab), dtype=np.float32)
        ultimate_count = np.zeros((n_jobs, n_layers), dtype=np.int32)
        print(
            "Accumulating ultimate softmax "
            f"({n_jobs} positions x {n_layers} layers x {vocab} vocab)."
        )

    def _scatter_plain(start: int, probs: np.ndarray) -> None:
        assert ultimate_sum is not None and ultimate_count is not None
        end = start + int(probs.shape[0])
        ultimate_sum.reshape(-1, vocab)[start:end] += probs
        ultimate_count.reshape(-1)[start:end] += 1

    plain_batch = np.stack([job.hidden for job in jobs], axis=0).reshape(n_jobs * n_layers, hidden_dim)
    plain_ids, plain_vals = logit_lens_topk(
        plain_batch,
        top_k=top_k,
        device=device,
        w_u=w_u,
        b_u=b_u,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        softcap=softcap,
        scatter=_scatter_plain if ultimate else None,
    )
    plain_l2 = np.linalg.norm(plain_batch, axis=-1).astype(np.float32, copy=False)
    plain_ids = plain_ids.reshape(n_jobs, n_layers, top_k)
    plain_vals = plain_vals.reshape(n_jobs, n_layers, top_k)
    plain_l2 = plain_l2.reshape(n_jobs, n_layers)
    del plain_batch

    block_ids: list[np.ndarray] = []
    block_vals: list[np.ndarray] = []
    block_l2s: list[np.ndarray] = []
    for block_idx in tqdm(range(arch.n_blocks), desc="Attention blocks"):
        n_sources = block_idx + 1

        def _scatter_block(start: int, probs: np.ndarray, n_src: int = n_sources) -> None:
            assert ultimate_sum is not None and ultimate_count is not None
            for local, row in enumerate(range(start, start + int(probs.shape[0]))):
                job = row // n_src
                source = row % n_src
                ultimate_sum[job, source] += probs[local]
                ultimate_count[job, source] += 1

        source_batch = np.stack(
            [job.hidden[:n_sources] for job in jobs], axis=0
        ).reshape(n_jobs * n_sources, hidden_dim)
        block = load_attn_ov_block(
            weight_map, arch, block_idx, device=device, norm_eps=norm_eps
        )
        try:
            ids, vals, l2s = ov_logit_lens_topk(
                source_batch,
                block,
                arch,
                top_k=top_k,
                device=device,
                w_u=w_u,
                b_u=b_u,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                softcap=softcap,
                gain=gain,
                scatter=_scatter_block if ultimate else None,
            )
        finally:
            del block
            if device.type == "cuda":
                torch.cuda.empty_cache()
        block_ids.append(ids.reshape(n_jobs, n_sources, top_k))
        block_vals.append(vals.reshape(n_jobs, n_sources, top_k))
        block_l2s.append(l2s.reshape(n_jobs, n_sources))
        del source_batch, ids, vals, l2s

    ultimate_ids: np.ndarray | None = None
    ultimate_vals: np.ndarray | None = None
    if ultimate_sum is not None and ultimate_count is not None:
        if int(ultimate_count.min()) < 1:
            raise ValueError("Ultimate softmax count is zero for at least one residual row.")
        ultimate_sum /= ultimate_count.astype(np.float32)[:, :, None]
        flat_ids, flat_vals = _topk_from_distributions(
            ultimate_sum.reshape(n_jobs * n_layers, vocab), top_k
        )
        ultimate_ids = flat_ids.reshape(n_jobs, n_layers, top_k)
        ultimate_vals = flat_vals.reshape(n_jobs, n_layers, top_k)
        del ultimate_sum, flat_ids, flat_vals

    return (
        plain_ids,
        plain_vals,
        plain_l2,
        block_ids,
        block_vals,
        block_l2s,
        ultimate_ids,
        ultimate_vals,
    )


def _write_config_txt(
    run_dir: Path,
    args: argparse.Namespace,
    corpus: ResidualCorpus,
    arch: ModelArch,
    vocab_size: int,
    elapsed_seconds: float,
    *,
    examples_high_dir: Path | None,
    examples_low_dir: Path | None,
) -> None:
    lines = [
        "Attention OV logit lens",
        "=" * 72,
        f"Model: {args.model_name_or_path}",
        f"Device: {args.device if args.device else 'cpu'}",
        f"Input H5: {args.input_h5}",
        f"Output dir: {run_dir}",
        f"n_examples: {args.n_examples}",
        f"top_k: {args.top_k}",
        f"include_embeddings: {','.join(args.include_embeddings)}",
        f"expected_guess_tokens: {args.expected_guess_tokens}",
        f"expected_probability_tokens: {args.expected_probability_tokens}",
        f"extend_probability_span: {args.extend_probability_span}",
        f"probability_token_budget: {args.expected_probability_tokens + (2 if args.extend_probability_span else 0)}",
        f"linguistic_confidence_prompt: {args.linguistic_confidence_prompt}",
        f"l_vs_un: {args.l_vs_un}",
        f"split_confidence_groups: {args.split_confidence_groups}",
        f"low_conf_threshold: {args.low_conf_threshold}",
        f"high_conf_threshold: {args.high_conf_threshold}",
        f"gain: {args.gain}",
        f"ultimate: {args.ultimate}",
        f"ultimate_detailed: {args.ultimate_detailed}",
        f"Valid examples: {corpus.valid_examples}",
        f"Total token positions: {len(corpus.token_labels)}",
        f"Token labels: {', '.join(corpus.token_labels)}",
        f"Residual rows: {corpus.n_layers}",
        f"Attention blocks: {arch.n_blocks}",
        f"Hidden dim: {corpus.hidden_dim}",
        f"Query heads: {arch.n_query_heads}",
        f"KV heads: {arch.n_kv_heads}",
        f"Head dim: {arch.head_dim}",
        f"Post-attention RMSNorm: {arch.use_post_attention_norm}",
        "OV read: input RMSNorm, V, 100% attention (repeat KV heads), O, "
        "optional post-attention RMSNorm, then final RMSNorm and unembedding.",
        (
            "Lens input for attn rows is the source residual plus the attention output."
            if args.gain
            else "Lens input for attn rows is the attention output alone."
        ),
        "Probabilities are percentages rounded to 2 decimal places.",
        (
            "Ultimate lens averages the full softmax of the plain lens and every later attention read."
            if args.ultimate
            else "Ultimate lens is off."
        ),
        f"Vocab size: {vocab_size}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
    ]
    if args.split_confidence_groups:
        lines.extend(
            [
                f"high_confidence_examples_written: {len(corpus.per_example_high)}",
                f"low_confidence_examples_written: {len(corpus.per_example_low)}",
                f"examples_high_dir: {examples_high_dir}",
                f"examples_low_dir: {examples_low_dir}",
            ]
        )
    lines.extend(["", f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    with open(run_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    start_time = time.perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    include_set = set(args.include_embeddings)
    probability_token_budget = args.expected_probability_tokens + (
        2 if args.extend_probability_span else 0
    )
    if args.l_vs_un and not args.linguistic_confidence_prompt:
        print(
            "Warning: ignoring --l_vs_un because --linguistic_confidence_prompt is off.",
            file=sys.stderr,
        )
    if args.ultimate_detailed and not args.ultimate:
        print(
            "Warning: ignoring --ultimate_detailed because --ultimate is off.",
            file=sys.stderr,
        )
        args.ultimate_detailed = False
    if bool(args.linguistic_confidence_prompt) and bool(args.l_vs_un):
        configure_prefix_tokens_for_model(args.model_name_or_path)
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
    vocab_size = int(w_u.shape[0])
    if args.top_k > vocab_size:
        raise ValueError(f"--top_k ({args.top_k}) exceeds vocab size ({vocab_size}).")

    weight_map = _build_weight_map(args.model_name_or_path)
    arch = _read_model_arch(args.model_name_or_path, weight_map)

    corpus = _read_residual_corpus(args, token_labels)
    if corpus.n_layers != arch.n_blocks + 1:
        raise ValueError(
            f"H5 residual rows ({corpus.n_layers}) != num_hidden_layers + 1 "
            f"({arch.n_blocks + 1}). Row 0 must be the input embedding."
        )
    if corpus.hidden_dim != arch.hidden_size:
        raise ValueError(
            f"H5 hidden dim ({corpus.hidden_dim}) != config hidden_size ({arch.hidden_size})."
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
        mean_high_dir.mkdir(parents=True, exist_ok=True)
        mean_low_dir.mkdir(parents=True, exist_ok=True)
    else:
        examples_dir = figures_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
    mean_overall_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[HiddenJob] = []
    jobs.extend(
        _jobs_for_store(
            _mean_store(corpus.sums_overall, corpus.count_overall, token_labels),
            mean_overall_dir,
            token_labels,
        )
    )
    if args.split_confidence_groups:
        assert examples_high_dir is not None and examples_low_dir is not None
        jobs.extend(
            _jobs_for_store(
                _mean_store(corpus.sums_high, corpus.count_high, token_labels),
                mean_high_dir,
                token_labels,
            )
        )
        jobs.extend(
            _jobs_for_store(
                _mean_store(corpus.sums_low, corpus.count_low, token_labels),
                mean_low_dir,
                token_labels,
            )
        )
        for i, store in enumerate(corpus.per_example_high):
            ex_dir = examples_high_dir / f"example_{i + 1}"
            jobs.extend(_jobs_for_store(store, ex_dir, token_labels))
        for i, store in enumerate(corpus.per_example_low):
            ex_dir = examples_low_dir / f"example_{i + 1}"
            jobs.extend(_jobs_for_store(store, ex_dir, token_labels))
    else:
        assert examples_dir is not None
        for i, store in enumerate(corpus.per_example):
            ex_dir = examples_dir / f"example_{i + 1}"
            jobs.extend(_jobs_for_store(store, ex_dir, token_labels))

    plain_ids, plain_vals, plain_l2, block_ids, block_vals, block_l2s, ultimate_ids, ultimate_vals = _run_lenses(
        jobs,
        arch,
        weight_map,
        device,
        top_k=args.top_k,
        w_u=w_u,
        b_u=b_u,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        softcap=softcap,
        gain=bool(args.gain),
        ultimate=bool(args.ultimate),
    )

    write_attention_details = (not args.ultimate) or bool(args.ultimate_detailed)
    for job_idx, job in enumerate(tqdm(jobs, desc="Writing tables")):
        write_position_outputs(
            job.out_dir,
            job.label,
            n_layers=corpus.n_layers,
            n_blocks=arch.n_blocks,
            plain_ids=plain_ids[job_idx],
            plain_vals=plain_vals[job_idx],
            plain_l2=plain_l2[job_idx],
            block_ids=[ids[job_idx] for ids in block_ids],
            block_vals=[vals[job_idx] for vals in block_vals],
            block_l2s=[norms[job_idx] for norms in block_l2s],
            tokenizer=tokenizer,
            write_attention_details=write_attention_details,
            ultimate_ids=None if ultimate_ids is None else ultimate_ids[job_idx],
            ultimate_vals=None if ultimate_vals is None else ultimate_vals[job_idx],
        )

    elapsed = time.perf_counter() - start_time
    _write_config_txt(
        run_dir=run_base,
        args=args,
        corpus=corpus,
        arch=arch,
        vocab_size=vocab_size,
        elapsed_seconds=elapsed,
        examples_high_dir=examples_high_dir,
        examples_low_dir=examples_low_dir,
    )
    print(f"Finished. Outputs written to: {run_base}")


if __name__ == "__main__":
    main()
