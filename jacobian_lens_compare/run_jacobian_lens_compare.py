"""Compare the public Jacobian lens with this project's logit lens and ultimate attention lens.

One generated continuation is read three ways:

- ``library``: the fitted Jacobian lens and its logit-lens baseline (``compute_slice``).
  The interactive flag writes both pages. The static table is the unmasked logit lens
  at the last prompt token.
- ``logit``: the residual logit lens from ``logit_lens_improved``. Row 0 is the
  embedding. The last row is ``hidden_states[-1]``, which already includes the
  final RMSNorm, so that row is not normalized again.
- ``attn``: ultimate attention-OV lens with gain. Each earlier residual row averages
  the plain lens with every later 100% attention read (residual plus attention
  output). The last row has no later block, so it is only the plain lens.

The library's layer 0 is the first block output, not the embedding.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jlens
from jlens.vis import SliceData, build_page, compute_slice

from attn_logit_lens.run_attn_logit_lens import (
    AttnOVBlock,
    apply_attn_ov,
    logit_lens_topk,
)
from logit_lens_improved.run_logit_lens_improved import (
    _disable_table_mathtext,
    _format_token_decoded,
    _get_run_base_dir,
    _probs_from_hidden,
)

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_CHOICES = ("library", "logit", "attn")
_HF_MODEL_NAME = re.compile(r'^hf_model_name:\s*"([^"]+)"\s*$', re.MULTILINE)


class _IdentityTransportLens:
    """Same fitted layers as a Jacobian lens, with transport left as the identity.

    ``compute_slice`` has no logit-lens switch. It calls ``transport`` whenever
    the layer is in ``jacobians``, so keeping those keys and returning the
    residual unchanged is the library logit lens.
    """

    def __init__(self, lens) -> None:
        self.jacobians = lens.jacobians
        self.source_layers = lens.source_layers

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        return residual


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model id.")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt text.")
    parser.add_argument(
        "--lenses",
        nargs="+",
        required=True,
        choices=LENS_CHOICES,
        help="Up to three of: library, logit, attn.",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an interactive HTML page for each selected lens (default: on).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Parent directory. A new numbered run folder is created inside it.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.lenses:
        raise ValueError("Pass at least one --lenses value.")
    if len(args.lenses) > 3:
        raise ValueError(f"--lenses accepts at most 3 values, got {args.lenses}.")
    if len(set(args.lenses)) != len(args.lenses):
        raise ValueError(f"--lenses values must be unique, got {args.lenses}.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1.")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1.")


def _resolve_lens_file(model_name: str) -> str:
    """Return the Hub path of the fitted lens for ``model_name``."""
    files = HfApi().list_repo_files(LENS_REPO, revision=LENS_REVISION)
    configs = [path for path in files if path.endswith("config.yaml") and "/jlens/" in path]
    found: dict[str, str] = {}
    for config_path in configs:
        local = hf_hub_download(LENS_REPO, config_path, revision=LENS_REVISION)
        match = _HF_MODEL_NAME.search(Path(local).read_text(encoding="utf-8"))
        if match is None:
            continue
        parent = str(Path(config_path).parent).replace("\\", "/")
        pts = [
            path
            for path in files
            if path.endswith(".pt") and str(Path(path).parent).replace("\\", "/") == parent
        ]
        pts.sort(key=lambda path: (0 if "jacobian_lens" in Path(path).name else 1, path))
        if not pts:
            continue
        found[match.group(1)] = pts[0]
    filename = found.get(model_name)
    if filename is None:
        known = "\n".join(f"  {name}" for name in sorted(found))
        raise ValueError(
            f"{model_name!r} has no fitted lens in {LENS_REPO}@{LENS_REVISION}.\n"
            f"Available models:\n{known}"
        )
    return filename


def _input_ids_from_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"]
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected a single prompt batch, got shape {tuple(input_ids.shape)}.")
    return input_ids.to(device)


def _load_hf_model(model_name: str, device: torch.device):
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "token": token}
    if device.type == "cuda":
        kwargs["device_map"] = str(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    input_ids = _input_ids_from_prompt(tokenizer, prompt, device)
    prompt_len = int(input_ids.shape[1])
    with torch.no_grad():
        sequences = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_ids = sequences[0].detach().to("cpu")
    if int(full_ids.shape[0]) < prompt_len:
        raise RuntimeError(
            f"Generation is shorter than the prompt ({full_ids.shape[0]} < {prompt_len})."
        )
    return input_ids[0].detach().to("cpu"), full_ids, prompt_len


def _stack_hidden_states(hidden_states) -> np.ndarray:
    if not hidden_states:
        raise RuntimeError("Model forward returned no hidden states.")
    rows = []
    for layer in hidden_states:
        tensor = layer[0] if isinstance(layer, (tuple, list)) else layer
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise RuntimeError(f"Hidden state rank {tensor.ndim}, expected [seq, hidden] or [1, seq, hidden].")
        rows.append(tensor.detach().float().cpu().numpy())
    stacked = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return stacked


def _text_config(model):
    if hasattr(model.config, "get_text_config"):
        return model.config.get_text_config()
    return model.config


def _gemma_offset_norm(module: torch.nn.Module) -> bool:
    return "gemma" in type(module).__name__.lower()


def _norm_gamma(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    weight = module.weight.detach().float()
    if _gemma_offset_norm(module):
        weight = 1.0 + weight
    return weight.to(device=device, dtype=torch.float32)


def _norm_eps(module: torch.nn.Module) -> float:
    eps = getattr(module, "variance_epsilon", None)
    if eps is None:
        eps = getattr(module, "eps", 1e-6)
    return float(eps)


def _linear_weight(linear: torch.nn.Module, device: torch.device) -> torch.Tensor:
    return linear.weight.detach().to(device=device, dtype=torch.float32)


def _linear_bias(linear: torch.nn.Module, device: torch.device) -> torch.Tensor | None:
    if getattr(linear, "bias", None) is None:
        return None
    return linear.bias.detach().to(device=device, dtype=torch.float32)


def _cfg_int(cfg, *names: str, default: int | None = None) -> int:
    for name in names:
        value = getattr(cfg, name, None)
        if value is not None:
            return int(value)
    if default is not None:
        return int(default)
    raise ValueError(f"None of {names} found on {type(cfg).__name__}.")


def _readout_params(lens_model, device: torch.device):
    text = _text_config(lens_model._hf_model)
    softcap = getattr(text, "final_logit_softcapping", None)
    if softcap is not None and float(softcap) == 0.0:
        softcap = None
    elif softcap is not None:
        softcap = float(softcap)
    norm = lens_model._final_norm
    lm_head = lens_model._lm_head
    return {
        "w_u": _linear_weight(lm_head, device),
        "b_u": _linear_bias(lm_head, device),
        "norm_weight": _norm_gamma(norm, device),
        "norm_eps": _norm_eps(norm),
        "softcap": softcap,
        "n_blocks": int(len(lens_model.layers)),
        "text_config": text,
        "use_post_attention_norm": all(
            hasattr(layer, "post_feedforward_layernorm") and hasattr(layer, "post_attention_layernorm")
            for layer in lens_model.layers
        ),
    }


def _attention_arch(lens_model, params: dict) -> SimpleNamespace:
    """Query/KV head sizes for the OV lens, checked against block 0's V projection."""
    text = params["text_config"]
    n_query = _cfg_int(text, "num_attention_heads", "n_head")
    n_kv = _cfg_int(
        text,
        "num_key_value_heads",
        "n_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        default=n_query,
    )
    if n_query % n_kv != 0:
        raise RuntimeError(f"n_query_heads={n_query} is not divisible by n_kv_heads={n_kv}.")
    layer0 = lens_model.layers[0]
    attn = getattr(layer0, "self_attn", None)
    if attn is None or not hasattr(attn, "v_proj"):
        raise RuntimeError(
            "The attention lens needs self_attn.v_proj/o_proj "
            "(Llama, Mistral, Qwen, Gemma, OLMo)."
        )
    v_out = int(attn.v_proj.weight.shape[0])
    if v_out % n_kv != 0:
        raise RuntimeError(f"v_proj out features {v_out} are not divisible by n_kv_heads={n_kv}.")
    head_dim = v_out // n_kv
    config_head_dim = getattr(text, "head_dim", None)
    if config_head_dim is not None and int(config_head_dim) != head_dim:
        raise RuntimeError(
            f"config.head_dim={int(config_head_dim)} but v_proj implies head_dim={head_dim}."
        )
    o_in = int(attn.o_proj.weight.shape[1])
    if o_in != n_query * head_dim:
        raise RuntimeError(
            f"o_proj in_features {o_in} != n_query_heads * head_dim ({n_query * head_dim})."
        )
    params["n_query_heads"] = n_query
    params["n_kv_heads"] = n_kv
    params["head_dim"] = head_dim
    return SimpleNamespace(n_query_heads=n_query, n_kv_heads=n_kv, head_dim=head_dim)


def _ov_block(layer, layer_idx: int, params: dict, device: torch.device) -> AttnOVBlock:
    attn = getattr(layer, "self_attn", None)
    if attn is None or not hasattr(attn, "v_proj") or not hasattr(attn, "o_proj"):
        raise RuntimeError(
            f"Layer {layer_idx} has no self_attn.v_proj/o_proj. "
            "The attention lens supports the Llama-style block layout "
            "(Llama, Mistral, Qwen, Gemma, OLMo)."
        )
    if not hasattr(layer, "input_layernorm"):
        raise RuntimeError(f"Layer {layer_idx} has no input_layernorm.")
    v_weight = _linear_weight(attn.v_proj, device)
    expected = params["n_kv_heads"] * params["head_dim"]
    if int(v_weight.shape[0]) != expected:
        raise RuntimeError(
            f"Layer {layer_idx} v_proj out features {int(v_weight.shape[0])} != {expected}."
        )
    post = None
    if params["use_post_attention_norm"]:
        post = _norm_gamma(layer.post_attention_layernorm, device)
    return AttnOVBlock(
        layer_idx=layer_idx,
        input_norm=_norm_gamma(layer.input_layernorm, device),
        v_weight=v_weight,
        v_bias=_linear_bias(attn.v_proj, device),
        o_weight=_linear_weight(attn.o_proj, device),
        o_bias=_linear_bias(attn.o_proj, device),
        post_attn_norm=post,
        norm_eps=_norm_eps(layer.input_layernorm),
    )


def _residual_row_label(row: int, n_rows: int) -> str:
    if row == 0:
        return "row_0_embedding"
    if row == n_rows - 1:
        return f"row_{row}_final_norm"
    return f"row_{row}_after_block_{row - 1}"


def _library_row_label(layer: int, n_layers: int) -> str:
    if layer == n_layers - 1:
        return f"layer_{layer}_final_block"
    return f"layer_{layer}_block_output"


def _save_labeled_table(path: Path, tokenizer, labels: Sequence[str], top_ids: np.ndarray, top_vals: np.ndarray) -> None:
    n_rows, top_k = top_ids.shape
    if len(labels) != n_rows:
        raise ValueError(f"{len(labels)} labels for {n_rows} rows.")
    col_labels = ["Row"]
    for k in range(top_k):
        col_labels.append(f"Top-{k + 1} token")
        col_labels.append(f"Top-{k + 1} prob")
    cells = []
    for row_idx, label in enumerate(labels):
        row = [label]
        for k in range(top_k):
            row.append(_format_token_decoded(tokenizer, int(top_ids[row_idx, k])))
            row.append(f"{float(top_vals[row_idx, k]):.6f}")
        cells.append(row)
    fig_w = max(12.0, 2.2 * len(col_labels))
    fig_h = max(6.0, 0.4 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    for col in range(len(col_labels)):
        header = table[(0, col)]
        header.set_facecolor((0.85, 0.9, 1.0, 1.0))
        header.set_text_props(weight="bold")
    for row_idx in range(n_rows):
        alpha = float(np.clip(top_vals[row_idx, 0], 0.0, 1.0))
        for col in range(1, len(col_labels)):
            table[(row_idx + 1, col)].set_facecolor((0.1, 0.3, 1.0, alpha))
    _disable_table_mathtext(table)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _topk_records(tokenizer, token_ids: np.ndarray, probs: np.ndarray) -> list[dict]:
    records = []
    for rank, (token_id, prob) in enumerate(zip(token_ids, probs), start=1):
        tid = int(token_id)
        records.append(
            {
                "rank": rank,
                "token_id": tid,
                "token": _format_token_decoded(tokenizer, tid),
                "probability": float(prob),
            }
        )
    return records


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_table_pair(
    table_dir: Path,
    stem: str,
    tokenizer,
    labels: Sequence[str],
    top_ids: np.ndarray,
    top_vals: np.ndarray,
    payload: dict,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    _save_labeled_table(table_dir / f"{stem}.png", tokenizer, labels, top_ids, top_vals)
    _write_json(table_dir / f"{stem}.json", payload)


OWN_INDEXING = (
    "Row 0 is the input embedding. Row k for 0 < k < n-1 is the residual after "
    "block k-1, before the model's final RMSNorm. The last row is "
    "hidden_states[-1] from output_hidden_states, which already passed through "
    "that norm, so the logit lens does not normalize it again."
)


def _forward_hidden(model, full_ids: torch.Tensor, device: torch.device) -> np.ndarray:
    input_ids = full_ids.to(device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is None:
        raise RuntimeError("Forward pass did not return hidden_states.")
    return _stack_hidden_states(hidden)


def _selected_rows(hidden: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """``hidden`` is ``[n_rows, seq, H]``. Return ``[n_positions, n_rows, H]``."""
    return np.stack([hidden[:, pos, :] for pos in positions], axis=0)


def _logit_topk_positions(
    hidden: np.ndarray,
    positions: Sequence[int],
    top_k: int,
    device: torch.device,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _selected_rows(hidden, positions)
    n_pos, n_rows, _hidden_dim = rows.shape
    norm_rows = np.ones((n_pos, n_rows), dtype=bool)
    norm_rows[:, -1] = False
    top_ids, top_vals = logit_lens_topk(
        rows.reshape(n_pos * n_rows, _hidden_dim),
        top_k=top_k,
        device=device,
        w_u=params["w_u"],
        b_u=params["b_u"],
        norm_weight=params["norm_weight"],
        norm_eps=params["norm_eps"],
        softcap=params["softcap"],
        norm_rows=norm_rows.reshape(-1),
    )
    return top_ids.reshape(n_pos, n_rows, top_k), top_vals.reshape(n_pos, n_rows, top_k)


def _ultimate_topk_positions(
    lens_model,
    hidden: np.ndarray,
    positions: Sequence[int],
    top_k: int,
    device: torch.device,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Ultimate gain=True top-k. Full softmaxes are dropped after each position."""
    n_rows = int(hidden.shape[0])
    n_blocks = int(params["n_blocks"])
    if n_rows != n_blocks + 1:
        raise RuntimeError(
            f"Expected {n_blocks + 1} residual rows (embedding plus each block, "
            f"last row post-norm), got {n_rows}."
        )
    arch = _attention_arch(lens_model, params)
    top_ids = np.zeros((len(positions), n_rows, top_k), dtype=np.int64)
    top_vals = np.zeros((len(positions), n_rows, top_k), dtype=np.float32)
    apply_norm_plain = torch.ones(n_rows, dtype=torch.bool, device=device)
    apply_norm_plain[-1] = False
    for out_idx, pos in enumerate(positions):
        hidden_t = torch.as_tensor(hidden[:, pos, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            total = _probs_from_hidden(
                hidden_t,
                w_u=params["w_u"],
                b_u=params["b_u"],
                norm_weight=params["norm_weight"],
                norm_eps=params["norm_eps"],
                softcap=params["softcap"],
                apply_norm=apply_norm_plain,
            )
            count = torch.ones(n_rows, dtype=torch.float32, device=device)
            for block_idx, layer in enumerate(lens_model.layers):
                block = _ov_block(layer, block_idx, params, device)
                try:
                    n_src = block_idx + 1
                    sources = hidden_t[:n_src]
                    attn_out = apply_attn_ov(sources, block, arch)
                    ov_probs = _probs_from_hidden(
                        sources + attn_out,
                        w_u=params["w_u"],
                        b_u=params["b_u"],
                        norm_weight=params["norm_weight"],
                        norm_eps=params["norm_eps"],
                        softcap=params["softcap"],
                        apply_norm=True,
                    )
                    total[:n_src] = total[:n_src] + ov_probs
                    count[:n_src] = count[:n_src] + 1
                finally:
                    del block
            average = total / count[:, None]
            vals, ids = torch.topk(average, k=top_k, dim=-1)
        top_ids[out_idx] = ids.detach().cpu().numpy().astype(np.int64, copy=False)
        top_vals[out_idx] = vals.detach().cpu().numpy().astype(np.float32, copy=False)
        del hidden_t, total, count, average, vals, ids
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return top_ids, top_vals


def _slice_from_topk(
    tokenizer,
    full_ids: Sequence[int],
    top_ids: np.ndarray,
    top_vals: np.ndarray,
    vocab_size: int,
) -> SliceData:
    """Build a page from stored top-k ids only.

    Ranks are exact for tokens inside each cell's top-k. Other tracked-token
    ranks stay -1 because the full distribution is not kept.
    """
    del top_vals
    seq_len, n_rows, top_k = top_ids.shape
    top_ids_i32 = top_ids.astype(np.int32, copy=False)
    top_ranks = np.broadcast_to(np.arange(top_k, dtype=np.int32), top_ids_i32.shape).copy()
    tracked = sorted({int(token_id) for token_id in top_ids_i32.reshape(-1)})
    index_of = {token_id: idx for idx, token_id in enumerate(tracked)}
    rank_tensor = np.full((seq_len, n_rows, len(tracked)), -1, dtype=np.int32)
    for pos in range(seq_len):
        for row in range(n_rows):
            for rank, token_id in enumerate(top_ids_i32[pos, row]):
                slot = index_of[int(token_id)]
                if rank_tensor[pos, row, slot] < 0:
                    rank_tensor[pos, row, slot] = rank
    context_ids = [int(token_id) for token_id in full_ids]
    context_strs = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in context_ids
    ]
    vocab_ids = set(tracked) | set(context_ids)
    vocab_fragment = {
        token_id: tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in vocab_ids
    }
    return SliceData(
        seq_len=seq_len,
        layers=list(range(n_rows)),
        context_token_ids=context_ids,
        context_token_strs=context_strs,
        top_ids=top_ids_i32,
        top_ranks=top_ranks,
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor,
        vocab_fragment=vocab_fragment,
        vocab_size=int(vocab_size),
        pinned_token_ids=[],
        ctx_offset=0,
    )


def _write_page(path: Path, slice_data: SliceData, prompt: str, title: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page, _, _ = build_page(
            slice_data,
            prompt,
            title=title,
            description=description,
            mode="embed",
        )
        path.write_text(page, encoding="utf-8")
        print(f"Wrote {path}")
        return
    except RuntimeError as exc:
        print(f"Self-contained page failed ({exc}). Writing a fetch-mode page instead.")
    out_dir = path.with_suffix("")
    page, _, _ = build_page(
        slice_data,
        prompt,
        title=title,
        description=description,
        mode="fetch",
        out_dir=out_dir,
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {out_dir / 'index.html'}")


def _own_table_payload(
    lens_name: str,
    tokenizer,
    labels: Sequence[str],
    top_ids: np.ndarray,
    top_vals: np.ndarray,
    prompt_pos: int,
    n_rows: int,
    extra: dict,
) -> dict:
    rows = []
    for row, label in enumerate(labels):
        rows.append(
            {
                "row": label,
                "layer_index": row,
                "final_norm_applied": row != n_rows - 1,
                "topk": _topk_records(tokenizer, top_ids[row], top_vals[row]),
            }
        )
    return {
        "lens": lens_name,
        "position": "last_prompt_token",
        "position_index": prompt_pos,
        "indexing": OWN_INDEXING,
        "rows": rows,
        **extra,
    }


def _run_own_lenses(
    args: argparse.Namespace,
    lens_model,
    tokenizer,
    full_ids: torch.Tensor,
    prompt_len: int,
    full_text: str,
    device: torch.device,
    table_dir: Path,
    page_dir: Path,
) -> None:
    hidden = _forward_hidden(lens_model._hf_model, full_ids, device)
    n_rows, seq_len, _hidden_dim = hidden.shape
    if seq_len != int(full_ids.shape[0]):
        raise RuntimeError(
            f"Hidden sequence length {seq_len} != generated length {int(full_ids.shape[0])}."
        )
    prompt_pos = prompt_len - 1
    params = _readout_params(lens_model, device)
    if n_rows != params["n_blocks"] + 1:
        raise RuntimeError(
            f"output_hidden_states has {n_rows} rows; expected {params['n_blocks'] + 1} "
            "(embedding, each block, and the post-norm final state as the last row)."
        )
    positions = list(range(seq_len)) if args.interactive else [prompt_pos]
    labels = [_residual_row_label(row, n_rows) for row in range(n_rows)]
    selected = set(args.lenses)

    if "logit" in selected:
        top_ids, top_vals = _logit_topk_positions(hidden, positions, args.top_k, device, params)
        local = positions.index(prompt_pos)
        _write_table_pair(
            table_dir,
            "logit_last_prompt",
            tokenizer,
            labels,
            top_ids[local],
            top_vals[local],
            _own_table_payload(
                "logit",
                tokenizer,
                labels,
                top_ids[local],
                top_vals[local],
                prompt_pos,
                n_rows,
                {},
            ),
        )
        if args.interactive:
            _write_page(
                page_dir / "logit.html",
                _slice_from_topk(tokenizer, full_ids.tolist(), top_ids, top_vals, int(params["w_u"].shape[0])),
                full_text,
                title="Project logit lens",
                description=OWN_INDEXING,
            )

    if "attn" in selected:
        top_ids, top_vals = _ultimate_topk_positions(
            lens_model, hidden, positions, args.top_k, device, params
        )
        local = positions.index(prompt_pos)
        note = (
            "ultimate=True and gain=True. Each row averages the plain logit lens "
            "with every later attention read of that residual (pre-norm, V, repeat "
            "KV heads, O, optional Gemma post-attention norm, then the residual plus "
            "that output). The last row has no later block."
        )
        _write_table_pair(
            table_dir,
            "attn_ultimate_last_prompt",
            tokenizer,
            labels,
            top_ids[local],
            top_vals[local],
            _own_table_payload(
                "attn_ultimate_gain",
                tokenizer,
                labels,
                top_ids[local],
                top_vals[local],
                prompt_pos,
                n_rows,
                {"ultimate": True, "gain": True, "note": note},
            ),
        )
        if args.interactive:
            _write_page(
                page_dir / "attn_ultimate.html",
                _slice_from_topk(tokenizer, full_ids.tolist(), top_ids, top_vals, int(params["w_u"].shape[0])),
                full_text,
                title="Project ultimate attention lens (gain)",
                description=OWN_INDEXING + " " + note,
            )


def _last_prompt_index(lens_model, prompt_text: str, full_text: str, max_seq_len: int, fallback: int) -> int:
    prompt_ids = lens_model.encode(prompt_text, max_length=max_seq_len)[0].tolist()
    full_ids = lens_model.encode(full_text, max_length=max_seq_len)[0].tolist()
    if prompt_ids and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids) - 1
    matched = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        matched += 1
    if matched > 0:
        print(
            "Library re-encode of the prompt is not a prefix of the full text. "
            f"Using the longest matching prefix ({matched} tokens) for the last prompt token."
        )
        return matched - 1
    if 0 <= fallback < len(full_ids):
        print(
            "Library re-encode does not share a prefix with the prompt. "
            f"Falling back to generated prompt index {fallback}."
        )
        return fallback
    raise RuntimeError("Could not locate the last prompt token in the library encoding.")


def _library_logit_table(
    lens,
    lens_model,
    tokenizer,
    full_text: str,
    prompt_pos: int,
    top_k: int,
    max_seq_len: int,
    table_dir: Path,
) -> None:
    n_layers = int(lens_model.n_layers)
    layers = list(range(n_layers))
    lens_logits, _model_logits, _input_ids = lens.apply(
        lens_model,
        full_text,
        layers=layers,
        positions=[prompt_pos],
        max_seq_len=max_seq_len,
        use_jacobian=False,
    )
    labels = [_library_row_label(layer, n_layers) for layer in layers]
    top_ids = np.zeros((n_layers, top_k), dtype=np.int64)
    top_vals = np.zeros((n_layers, top_k), dtype=np.float32)
    for row, layer in enumerate(layers):
        probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
        vals, ids = torch.topk(probs, k=min(top_k, int(probs.shape[0])))
        top_ids[row] = ids.cpu().numpy()
        top_vals[row] = vals.cpu().numpy()
    rows = []
    for row, (layer, label) in enumerate(zip(layers, labels)):
        rows.append(
            {
                "row": label,
                "layer_index": layer,
                "final_norm_applied": True,
                "topk": _topk_records(tokenizer, top_ids[row], top_vals[row]),
            }
        )
    _write_table_pair(
        table_dir,
        "library_logit_lens_last_prompt",
        tokenizer,
        labels,
        top_ids,
        top_vals,
        {
            "lens": "library_logit",
            "position": "last_prompt_token",
            "position_index": prompt_pos,
            "indexing": (
                "Library layer i is the output of transformer block i. "
                "The lens unembed applies the final RMSNorm. There is no embedding row. "
                "Layer 0 here corresponds to project residual row 1."
            ),
            "rows": rows,
        },
    )


def _run_library(
    args: argparse.Namespace,
    lens_model,
    tokenizer,
    prompt_text: str,
    full_text: str,
    full_ids: torch.Tensor,
    prompt_len: int,
    table_dir: Path,
    page_dir: Path,
    lens_filename: str,
) -> None:
    max_seq_len = max(int(full_ids.shape[0]) + 32, 512)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=lens_filename, revision=LENS_REVISION
    )
    prompt_pos = _last_prompt_index(
        lens_model, prompt_text, full_text, max_seq_len, fallback=prompt_len - 1
    )
    encoded = lens_model.encode(full_text, max_length=max_seq_len)[0].tolist()
    if encoded != full_ids.tolist():
        print(
            "Library encode of the decoded generation differs from the generated token ids "
            f"({len(encoded)} vs {int(full_ids.shape[0])} tokens). "
            "compute_slice lenses the decoded text, as requested."
        )
    _library_logit_table(
        lens, lens_model, tokenizer, full_text, prompt_pos, args.top_k, max_seq_len, table_dir
    )
    if not args.interactive:
        return
    logit_lens = _IdentityTransportLens(lens)
    descriptions = {
        "library_j_lens.html": (
            "Jacobian lens. Layer i is block i, not the embedding. "
            "The bottom fitted layer is transported with J = I and then final-normalized."
        ),
        "library_logit_lens.html": (
            "Library logit lens (Jacobian transport disabled). "
            "Layer i is block i, not the embedding. The unembed applies the final RMSNorm."
        ),
    }
    slices = {
        "library_j_lens.html": compute_slice(
            lens_model, lens, full_text, top_n=args.top_k, max_seq_len=max_seq_len, mask_display=True
        ),
        "library_logit_lens.html": compute_slice(
            lens_model,
            logit_lens,
            full_text,
            top_n=args.top_k,
            max_seq_len=max_seq_len,
            mask_display=True,
        ),
    }
    for name, slice_data in slices.items():
        _write_page(
            page_dir / name,
            slice_data,
            full_text,
            title=name.removesuffix(".html").replace("_", " "),
            description=descriptions[name],
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} was requested, but CUDA is not available.")

    lens_filename = _resolve_lens_file(args.model)
    print(f"Lens file: {LENS_REPO}@{LENS_REVISION}:{lens_filename}")
    tokenizer, hf_model = _load_hf_model(args.model, device)
    _prompt_ids, full_ids, prompt_len = _generate(
        hf_model, tokenizer, args.prompt, args.max_new_tokens, device
    )
    prompt_text = tokenizer.decode(full_ids[:prompt_len], skip_special_tokens=False)
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    print(f"Prompt tokens: {prompt_len}. Generated sequence: {int(full_ids.shape[0])}.")

    # Wrap after generation so add_bos_token does not change the chat-template ids.
    lens_model = jlens.from_hf(hf_model, tokenizer)
    run_dir = _get_run_base_dir(Path(args.output_dir))
    table_dir = run_dir / "tables"
    page_dir = run_dir / "interactive"
    (run_dir / "generation.txt").write_text(full_text, encoding="utf-8")

    selected = set(args.lenses)
    if "logit" in selected or "attn" in selected:
        _run_own_lenses(
            args, lens_model, tokenizer, full_ids, prompt_len, full_text, device, table_dir, page_dir
        )
    if "library" in selected:
        _run_library(
            args,
            lens_model,
            tokenizer,
            prompt_text,
            full_text,
            full_ids,
            prompt_len,
            table_dir,
            page_dir,
            lens_filename,
        )

    config = {
        "model": args.model,
        "prompt": args.prompt,
        "lenses": list(args.lenses),
        "interactive": bool(args.interactive),
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "device": args.device,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_filename": lens_filename,
        "prompt_len": prompt_len,
        "sequence_len": int(full_ids.shape[0]),
        "last_prompt_token_id": int(full_ids[prompt_len - 1]),
        "project_row_0": "embedding",
        "library_layer_0": "first block output",
        "final_row": "hidden_states[-1] already includes the final RMSNorm",
    }
    _write_json(run_dir / "config.json", config)
    print(f"Finished. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
