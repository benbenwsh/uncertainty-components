"""
Train tuned-lens affine probes for every layer using guess token position 5.

Input:
- HDF5 files produced by process_generations_verbalised_embeddings_h5.py
- We read embeddings_guess[5] from responses[0]

For each example with hidden states h_l at every layer l:
- Teacher distribution: final-layer logits (h_{L-1} through lm_head)
- Student distribution at each layer l:
    z_l = W_l h_l + b_l
    logits_l = lm_head(z_l)
- Objective: KL( teacher || student_l ), averaged over layers and batch.

The script writes:
- numbered run folder under output_dir/tuned_lens_guess_tok_5/<id>/
- output.log, metrics_summary.json, by-layer metric plots
- per-layer folders with config.txt and train_val_loss.png
- top-k table PNGs for a few validation examples
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_TOKEN_INDEX = 5
EPS = 1e-12


def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_h5_node(node):
    """Recursively read object written by native HDF5 writer."""
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "S":
                return np.array([x.decode("utf-8") for x in value], dtype=object).tolist()
            if value.dtype.kind == "O":
                return [_decode_h5_string(x) for x in value.tolist()]
            return value
        return _decode_h5_string(value)

    node_type = node.attrs.get("__type__", "")
    if isinstance(node_type, bytes):
        node_type = node_type.decode("utf-8")
    if node_type == "none":
        return None
    if node_type in ("list", "tuple"):
        length = int(node.attrs.get("__len__", len(node.keys())))
        items = [_read_h5_node(node[str(i)]) for i in range(length)]
        return tuple(items) if node_type == "tuple" else items

    out = {}
    for key in node.keys():
        out[key] = _read_h5_node(node[key])
    return out


def _read_token_stack_ds(example_group, token_idx: int):
    """
    Fast-path loader for:
      examples/<id>/responses/0/embeddings_guess/<token_idx>
    """
    responses = example_group.get("responses")
    if responses is None or not isinstance(responses, h5py.Group):
        return None
    resp0 = responses.get("0")
    if resp0 is None or not isinstance(resp0, h5py.Group):
        return None
    emb_guess = resp0.get("embeddings_guess")
    if emb_guess is None or not isinstance(emb_guess, h5py.Group):
        return None
    ds = emb_guess.get(str(token_idx))
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    return ds


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _stack_guess_token(emb) -> np.ndarray:
    arr = _tensor_to_numpy(emb)
    return arr[:, 0, -1, :]


def _iter_token_stacks_from_h5(path: str, token_idx: int, max_examples: int | None = None):
    emitted = 0
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        desc = f"Streaming token {token_idx} ({Path(path).name})"
        for example_id in tqdm(examples_group.keys(), desc=desc, leave=False):
            example_id = str(example_id)
            example_group = examples_group[example_id]

            ds = _read_token_stack_ds(example_group, token_idx)
            if ds is not None and ds.ndim == 4:
                stack = np.asarray(ds[:, 0, -1, :], dtype=np.float32)
                if stack.ndim == 2:
                    yield example_id, stack
                    emitted += 1
                    if max_examples is not None and emitted >= max_examples:
                        return
                    continue

            # Fallback for legacy nested object structures.
            example_data = _read_h5_node(example_group)
            if not isinstance(example_data, dict):
                continue
            responses = example_data.get("responses", [])
            if not responses:
                continue
            emb_guess = responses[0].get("embeddings_guess")
            if emb_guess is None:
                continue
            if not isinstance(emb_guess, (list, tuple)) or len(emb_guess) <= token_idx:
                continue
            try:
                stack = _stack_guess_token(emb_guess[token_idx]).astype(np.float32, copy=False)
            except Exception:
                continue
            if stack.ndim != 2:
                continue
            yield example_id, stack
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return


def _load_token_dataset(path: str, token_idx: int, max_examples: int | None = None):
    stacks = []
    example_ids = []
    n_layers = None
    hidden_dim = None
    for example_id, stack in _iter_token_stacks_from_h5(path, token_idx, max_examples=max_examples):
        if n_layers is None:
            n_layers, hidden_dim = int(stack.shape[0]), int(stack.shape[1])
        if stack.shape != (n_layers, hidden_dim):
            continue
        stacks.append(stack)
        example_ids.append(example_id)
    if not stacks:
        raise ValueError(f"No valid examples found in: {path}")
    data = np.stack(stacks, axis=0).astype(np.float32, copy=False)  # [N, L, H]
    return data, example_ids, int(n_layers), int(hidden_dim)


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_unembedding(model_name_or_path: str, device: torch.device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as exc:
        logging.warning(
            "Fast tokenizer load failed (%s). Falling back to use_fast=False.",
            exc,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if not hasattr(model, "lm_head"):
        raise ValueError("Model has no lm_head; cannot train tuned lens.")
    w_u = model.lm_head.weight.detach().to(device=device, dtype=torch.float32)  # [V, H]
    b_u = getattr(model.lm_head, "bias", None)
    if b_u is not None:
        b_u = b_u.detach().to(device=device, dtype=torch.float32)
    return tokenizer, w_u, b_u


class PerLayerAffineTunedLens(nn.Module):
    """Per-layer affine maps hidden->hidden: z_l = W_l h_l + b_l."""

    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        eye = torch.eye(hidden_dim, dtype=torch.float32).unsqueeze(0).repeat(n_layers, 1, 1)
        self.weight = nn.Parameter(eye)  # [L, H, H], indexed as (out, in)
        self.bias = nn.Parameter(torch.zeros(n_layers, hidden_dim, dtype=torch.float32))  # [L, H]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, H]
        transformed = torch.einsum("blh,loh->blo", hidden_states, self.weight)
        transformed = transformed + self.bias.unsqueeze(0)
        return transformed


def _compute_logits(hidden: torch.Tensor, w_u: torch.Tensor, b_u: torch.Tensor | None) -> torch.Tensor:
    logits = hidden @ w_u.T
    if b_u is not None:
        logits = logits + b_u
    return logits


def _student_logits_all_layers(
    lens: PerLayerAffineTunedLens,
    hidden_states: torch.Tensor,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
) -> torch.Tensor:
    # hidden_states: [B, L, H]
    transformed = lens(hidden_states)  # [B, L, H]
    logits = torch.einsum("blh,vh->blv", transformed, w_u)  # [B, L, V]
    if b_u is not None:
        logits = logits + b_u.view(1, 1, -1)
    return logits


def _kl_per_layer(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    # student_logits: [B, L, V], teacher_probs: [B, V]
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    target = teacher_probs.unsqueeze(1).expand_as(student_log_probs)
    kl = torch.sum(target * (torch.log(target.clamp_min(EPS)) - student_log_probs), dim=-1)  # [B, L]
    return kl.mean(dim=0)  # [L]


def _cross_entropy_per_layer(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    target = teacher_probs.unsqueeze(1).expand_as(student_log_probs)
    ce = -torch.sum(target * student_log_probs, dim=-1)  # [B, L]
    return ce.mean(dim=0)  # [L]


def _cosine_per_layer(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    student_probs = F.softmax(student_logits, dim=-1)  # [B, L, V]
    target = teacher_probs.unsqueeze(1).expand_as(student_probs)
    cos = F.cosine_similarity(student_probs, target, dim=-1)  # [B, L]
    return cos.mean(dim=0)  # [L]


def _plot_metric_by_layer(
    layer_numbers: list[int],
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, values, "o-", markersize=4)
    ax.set_xlabel("Layer number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_train_val_curve(
    steps: np.ndarray,
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_vals, "o-", label="Train", markersize=3)
    ax.plot(steps, val_vals, "s-", label="Validation", markersize=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL divergence")
    ax.set_title("Train vs validation KL loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_token(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        token = tokenizer.decode([int(token_id)])
    return f"{token} | {tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)}"


def _save_topk_table_png(
    out_path: Path,
    tokenizer,
    top_ids: np.ndarray,   # [L, K]
    top_vals: np.ndarray,  # [L, K]
) -> None:
    n_layers, top_k = top_ids.shape
    col_labels = ["Layer"]
    for k in range(top_k):
        col_labels.append(f"Top-{k + 1} token")
        col_labels.append(f"Top-{k + 1} prob")

    rows = []
    for layer_idx in range(n_layers):
        row = [f"layer_{layer_idx + 1}"]
        for k in range(top_k):
            tok_id = int(top_ids[layer_idx, k])
            prob = float(top_vals[layer_idx, k])
            row.append(_format_token(tokenizer, tok_id))
            row.append(f"{prob:.6f}")
        rows.append(row)

    fig_w = max(11.0, 2.1 * len(col_labels))
    fig_h = max(6.0, 0.4 * n_layers + 2.0)
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

    # Header style.
    for c in range(len(col_labels)):
        header_cell = table[(0, c)]
        header_cell.set_facecolor((0.85, 0.9, 1.0, 1.0))
        header_cell.set_text_props(weight="bold")

    # Data row styling: alpha equals row top-1 probability.
    for r in range(n_layers):
        alpha = float(np.clip(top_vals[r, 0], 0.0, 1.0))
        for c in range(1, len(col_labels)):  # leave "Layer" column plain
            cell = table[(r + 1, c)]
            cell.set_facecolor((0.1, 0.3, 1.0, alpha))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "tuned_lens_guess_tok_5"
    base.mkdir(parents=True, exist_ok=True)
    run_id = 1
    while (base / str(run_id)).exists():
        run_id += 1
    run_base = base / str(run_id)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _write_layer_config_txt(
    layer_dir: Path,
    args: argparse.Namespace,
    layer_idx: int,
    n_layers: int,
    hidden_dim: int,
    vocab_size: int,
    n_train: int,
    n_val: int,
    final_train_kl: float,
    final_val_kl: float,
    eval_kl: float,
    eval_ce: float,
    eval_cos: float,
) -> None:
    lines = [
        "Tuned lens probe config",
        "=" * 60,
        f"Target token index: {TARGET_TOKEN_INDEX}",
        f"Layer index (0-based): {layer_idx}",
        f"Layer folder index (1-based): {layer_idx + 1}",
        f"Total layers: {n_layers}",
        f"Hidden dim: {hidden_dim}",
        f"Vocab size: {vocab_size}",
        "",
        f"Model: {args.model_name_or_path}",
        f"Device: {args.device if args.device else 'auto'}",
        f"Train path: {args.train_path}",
        f"Val path: {args.val_path}",
        f"Train examples: {n_train}",
        f"Val examples: {n_val}",
        "",
        "Optimization",
        "-" * 40,
        "Optimizer: SGD (Nesterov momentum)",
        f"Steps: {args.steps}",
        f"Batch size: {args.batch_size}",
        f"Learning rate (start): {args.lr}",
        f"Momentum: {args.momentum}",
        f"Weight decay: {args.weight_decay}",
        f"Max grad norm: {args.max_grad_norm}",
        f"Seed: {args.seed}",
        "LR schedule: linear decay to 0",
        "",
        "Metrics",
        "-" * 40,
        f"Final train KL (batch estimate): {final_train_kl:.8f}",
        f"Final val KL (batch estimate): {final_val_kl:.8f}",
        f"Validation KL (full set): {eval_kl:.8f}",
        f"Validation cross-entropy (full set): {eval_ce:.8f}",
        f"Validation cosine similarity (full set): {eval_cos:.8f}",
        "",
        f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(layer_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def _sample_batch(data: torch.Tensor, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
    n = int(data.shape[0])
    if batch_size >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=batch_size, replace=False)
    idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
    return data.index_select(0, idx_t)


@torch.no_grad()
def _evaluate_full(
    lens: PerLayerAffineTunedLens,
    val_data: torch.Tensor,  # [N, L, H] on CPU
    batch_size: int,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    device: torch.device,
) -> dict:
    n_val = int(val_data.shape[0])
    n_layers = int(val_data.shape[1])

    kl_sum = torch.zeros(n_layers, dtype=torch.float64)
    ce_sum = torch.zeros(n_layers, dtype=torch.float64)
    cos_sum = torch.zeros(n_layers, dtype=torch.float64)
    seen = 0

    for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        batch = val_data[start:end].to(device=device, dtype=torch.float32)
        teacher_logits = _compute_logits(batch[:, -1, :], w_u, b_u)  # [B, V]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_logits = _student_logits_all_layers(lens, batch, w_u, b_u)  # [B, L, V]

        kl = _kl_per_layer(student_logits, teacher_probs).to(dtype=torch.float64)
        ce = _cross_entropy_per_layer(student_logits, teacher_probs).to(dtype=torch.float64)
        cos = _cosine_per_layer(student_logits, teacher_probs).to(dtype=torch.float64)

        bsz = end - start
        kl_sum += kl * bsz
        ce_sum += ce * bsz
        cos_sum += cos * bsz
        seen += bsz

    if seen == 0:
        raise ValueError("Validation set is empty.")

    kl_mean = (kl_sum / seen).cpu().numpy()
    ce_mean = (ce_sum / seen).cpu().numpy()
    cos_mean = (cos_sum / seen).cpu().numpy()
    return {"kl_by_layer": kl_mean, "ce_by_layer": ce_mean, "cos_by_layer": cos_mean}


@torch.no_grad()
def _collect_topk_for_examples(
    lens: PerLayerAffineTunedLens,
    val_data: torch.Tensor,  # [N, L, H]
    top_k: int,
    n_examples: int,
    tokenizer,
    w_u: torch.Tensor,
    b_u: torch.Tensor | None,
    device: torch.device,
    run_base: Path,
) -> None:
    n_examples = min(n_examples, int(val_data.shape[0]))
    if n_examples <= 0:
        return
    for ex_idx in range(n_examples):
        sample = val_data[ex_idx : ex_idx + 1].to(device=device, dtype=torch.float32)
        student_logits = _student_logits_all_layers(lens, sample, w_u, b_u).squeeze(0)  # [L, V]
        probs = F.softmax(student_logits, dim=-1)
        top_vals, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        top_vals_np = top_vals.detach().cpu().numpy().astype(np.float32, copy=False)
        top_ids_np = top_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        out_path = run_base / f"topk_table_example_{ex_idx + 1}.png"
        _save_topk_table_png(out_path, tokenizer, top_ids_np, top_vals_np)


def main():
    parser = argparse.ArgumentParser(
        description="Train tuned-lens affine probes for all layers using embeddings_guess[5] from H5."
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Run dir: output_dir/tuned_lens_guess_tok_5/<id>/",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--n_eval_examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_examples_per_split",
        type=int,
        default=None,
        help="Optional cap of examples loaded from each split.",
    )
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if args.top_k <= 0:
        raise ValueError("--top_k must be >= 1")
    if args.n_eval_examples <= 0:
        raise ValueError("--n_eval_examples must be >= 1")
    if not (0.0 <= args.momentum < 1.0):
        raise ValueError("--momentum must satisfy 0 <= momentum < 1")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0")

    start_time = time.perf_counter()
    run_base = _get_run_base_dir(Path(args.output_dir))
    log_path = run_base / "output.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Run directory: %s", run_base)
    logger.info("Loading H5 datasets for embeddings_guess[%d]...", TARGET_TOKEN_INDEX)

    train_np, train_ids, n_layers_tr, hidden_dim_tr = _load_token_dataset(
        args.train_path,
        token_idx=TARGET_TOKEN_INDEX,
        max_examples=args.max_examples_per_split,
    )
    val_np, val_ids, n_layers_val, hidden_dim_val = _load_token_dataset(
        args.val_path,
        token_idx=TARGET_TOKEN_INDEX,
        max_examples=args.max_examples_per_split,
    )
    if (n_layers_tr, hidden_dim_tr) != (n_layers_val, hidden_dim_val):
        raise ValueError(
            "Train/val shape mismatch: "
            f"train=({n_layers_tr}, {hidden_dim_tr}) vs val=({n_layers_val}, {hidden_dim_val})"
        )
    n_layers = n_layers_tr
    hidden_dim = hidden_dim_tr
    logger.info(
        "Loaded train=%d val=%d examples | layers=%d hidden_dim=%d",
        train_np.shape[0],
        val_np.shape[0],
        n_layers,
        hidden_dim,
    )

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Loading model/unembedding: %s", args.model_name_or_path)
    tokenizer, w_u, b_u = _load_unembedding(args.model_name_or_path, device)
    vocab_size, model_hidden_dim = int(w_u.shape[0]), int(w_u.shape[1])
    if model_hidden_dim != hidden_dim:
        raise ValueError(
            f"Hidden dim mismatch between H5 and model lm_head: {hidden_dim} vs {model_hidden_dim}"
        )
    logger.info("Model vocab_size=%d hidden_dim=%d", vocab_size, model_hidden_dim)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng_train = np.random.default_rng(args.seed)
    rng_val = np.random.default_rng(args.seed + 1)

    train_data = torch.from_numpy(train_np).to(dtype=torch.float32)
    val_data = torch.from_numpy(val_np).to(dtype=torch.float32)

    lens = PerLayerAffineTunedLens(n_layers=n_layers, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.SGD(
        lens.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: max(0.0, 1.0 - (step_idx / float(max(args.steps, 1)))),
    )

    train_kl_history = np.zeros((args.steps, n_layers), dtype=np.float32)
    val_kl_history = np.zeros((args.steps, n_layers), dtype=np.float32)
    total_train_history = np.zeros((args.steps,), dtype=np.float32)
    total_val_history = np.zeros((args.steps,), dtype=np.float32)

    logger.info("Starting training for %d steps...", args.steps)
    for step in range(args.steps):
        lens.train()
        batch_train = _sample_batch(train_data, args.batch_size, rng_train).to(device=device, dtype=torch.float32)

        teacher_logits = _compute_logits(batch_train[:, -1, :], w_u, b_u)
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
        student_logits = _student_logits_all_layers(lens, batch_train, w_u, b_u)
        layer_kl = _kl_per_layer(student_logits, teacher_probs)
        loss = layer_kl.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(lens.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        train_kl_np = layer_kl.detach().cpu().numpy().astype(np.float32, copy=False)
        train_kl_history[step] = train_kl_np
        total_train_history[step] = float(train_kl_np.mean())

        # Validation batch estimate each step (for train-vs-val curves).
        lens.eval()
        with torch.no_grad():
            batch_val = _sample_batch(val_data, args.batch_size, rng_val).to(device=device, dtype=torch.float32)
            val_teacher_logits = _compute_logits(batch_val[:, -1, :], w_u, b_u)
            val_teacher_probs = F.softmax(val_teacher_logits, dim=-1)
            val_student_logits = _student_logits_all_layers(lens, batch_val, w_u, b_u)
            val_layer_kl = _kl_per_layer(val_student_logits, val_teacher_probs)
            val_kl_np = val_layer_kl.detach().cpu().numpy().astype(np.float32, copy=False)
            val_kl_history[step] = val_kl_np
            total_val_history[step] = float(val_kl_np.mean())

        if (step + 1) % max(1, args.log_every) == 0 or step == 0 or (step + 1) == args.steps:
            current_lr = float(optimizer.param_groups[0]["lr"])
            logger.info(
                "step=%d/%d train_kl=%.6f val_kl=%.6f lr=%.8f",
                step + 1,
                args.steps,
                float(total_train_history[step]),
                float(total_val_history[step]),
                current_lr,
            )

    logger.info("Training complete. Running full validation metrics...")
    lens.eval()
    eval_metrics = _evaluate_full(
        lens=lens,
        val_data=val_data,
        batch_size=max(1, args.batch_size),
        w_u=w_u,
        b_u=b_u,
        device=device,
    )
    kl_by_layer = eval_metrics["kl_by_layer"]
    ce_by_layer = eval_metrics["ce_by_layer"]
    cos_by_layer = eval_metrics["cos_by_layer"]

    layer_numbers = list(range(1, n_layers + 1))
    _plot_metric_by_layer(
        layer_numbers,
        kl_by_layer,
        title="Validation KL divergence by layer",
        ylabel="KL divergence",
        out_path=run_base / "kl_by_layer.png",
    )
    _plot_metric_by_layer(
        layer_numbers,
        ce_by_layer,
        title="Validation cross-entropy by layer",
        ylabel="Cross-entropy",
        out_path=run_base / "cross_entropy_by_layer.png",
    )
    _plot_metric_by_layer(
        layer_numbers,
        cos_by_layer,
        title="Validation cosine similarity by layer",
        ylabel="Cosine similarity",
        out_path=run_base / "cosine_similarity_by_layer.png",
    )

    steps_axis = np.arange(1, args.steps + 1, dtype=np.int32)
    for layer_idx in range(n_layers):
        layer_dir = run_base / f"layer_{layer_idx + 1}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        _plot_train_val_curve(
            steps=steps_axis,
            train_vals=train_kl_history[:, layer_idx],
            val_vals=val_kl_history[:, layer_idx],
            out_path=layer_dir / "train_val_loss.png",
        )
        _write_layer_config_txt(
            layer_dir=layer_dir,
            args=args,
            layer_idx=layer_idx,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            n_train=int(train_data.shape[0]),
            n_val=int(val_data.shape[0]),
            final_train_kl=float(train_kl_history[-1, layer_idx]),
            final_val_kl=float(val_kl_history[-1, layer_idx]),
            eval_kl=float(kl_by_layer[layer_idx]),
            eval_ce=float(ce_by_layer[layer_idx]),
            eval_cos=float(cos_by_layer[layer_idx]),
        )

    logger.info(
        "Generating top-k tables for %d validation example(s)...",
        args.n_eval_examples,
    )
    _collect_topk_for_examples(
        lens=lens,
        val_data=val_data,
        top_k=args.top_k,
        n_examples=args.n_eval_examples,
        tokenizer=tokenizer,
        w_u=w_u,
        b_u=b_u,
        device=device,
        run_base=run_base,
    )

    summary = {
        "run_dir": str(run_base),
        "model_name_or_path": args.model_name_or_path,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "target_token_index": TARGET_TOKEN_INDEX,
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "vocab_size": int(vocab_size),
        "n_train_examples": int(train_data.shape[0]),
        "n_val_examples": int(val_data.shape[0]),
        "train_example_ids_preview": train_ids[: min(10, len(train_ids))],
        "val_example_ids_preview": val_ids[: min(10, len(val_ids))],
        "optimizer": "SGD",
        "nesterov": True,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr_start": float(args.lr),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "seed": int(args.seed),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "final_train_kl_mean": float(total_train_history[-1]),
        "final_val_kl_mean": float(total_val_history[-1]),
        "kl_by_layer": kl_by_layer.tolist(),
        "cross_entropy_by_layer": ce_by_layer.tolist(),
        "cosine_similarity_by_layer": cos_by_layer.tolist(),
    }
    with open(run_base / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved metrics summary: %s", run_base / "metrics_summary.json")
    logger.info("Saved execution log: %s", log_path)
    logger.info("Done. Outputs under %s", run_base)


if __name__ == "__main__":
    main()
