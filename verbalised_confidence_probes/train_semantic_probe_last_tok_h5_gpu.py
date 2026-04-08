"""
Train independent binary probes for tok_5_guess across all layers using PyTorch.

Compared with train_semantic_probe_last_tok_h5.py:
- uses GPU when available (or CPU fallback),
- trains all per-layer probes in parallel (batched layer dimension),
- keeps output layout compatible:
  results/sem_probe_last_tok_all_layers/<run_id>/tok_5_guess/layer_k/

I/O note:
This script supports multiple HDF5 reader workers. Readers are read-only and
safe in practice, but heavy worker/process fanout on a shared filesystem can
still bottleneck on storage bandwidth/metadata.
"""

from __future__ import annotations

import argparse
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

try:
    from verbalised_confidence_probes.train_multitoken_verbalised_confidence_probe import (
        _tensor_to_numpy,
    )
except ImportError:
    from train_multitoken_verbalised_confidence_probe import _tensor_to_numpy

REF_TOKEN_INDEX = 5
TARGET_TOKEN_POS = 5
NUM_GUESS_TOKENS = 6


def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_h5_node(node):
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


def _stack_guess_token(emb) -> np.ndarray:
    arr = _tensor_to_numpy(emb)
    return arr[:, 0, -1, :]


def _example_stacks(emb_guess: list) -> list[np.ndarray] | None:
    if len(emb_guess) != NUM_GUESS_TOKENS:
        return None
    stacks = []
    for t in range(NUM_GUESS_TOKENS):
        s = _stack_guess_token(emb_guess[t])
        stacks.append(s)
    n_layers = stacks[0].shape[0]
    hidden_dim = stacks[0].shape[1]
    for s in stacks:
        if s.shape != (n_layers, hidden_dim):
            return None
    return stacks


def _read_token_stack_ds(example_group, token_idx: int):
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


class H5PairIterableDataset(IterableDataset):
    """
    Yields (ref_last[D], h_all_layers[L,D]) per valid example for one token position.
    """

    def __init__(self, h5_path: str, token_pos: int, use_swmr: bool = False):
        super().__init__()
        self.h5_path = h5_path
        self.token_pos = token_pos
        self.use_swmr = use_swmr

    def _iter_ids(self, keys: list[str]) -> Iterable[str]:
        worker = get_worker_info()
        if worker is None:
            yield from keys
            return
        # Deterministic sharding: each worker gets its modulo slice.
        for idx in range(worker.id, len(keys), worker.num_workers):
            yield keys[idx]

    def __iter__(self):
        with h5py.File(self.h5_path, "r", swmr=self.use_swmr) as h5_file:
            if "examples" not in h5_file:
                raise ValueError(f"HDF5 file has no 'examples' group: {self.h5_path}")
            examples_group = h5_file["examples"]
            keys = list(examples_group.keys())

            for example_id in self._iter_ids(keys):
                example_group = examples_group[example_id]

                ref_ds = _read_token_stack_ds(example_group, REF_TOKEN_INDEX)
                h_ds = _read_token_stack_ds(example_group, self.token_pos)

                if ref_ds is None or h_ds is None:
                    example_data = _read_h5_node(example_group)
                    if not isinstance(example_data, dict):
                        continue
                    responses = example_data.get("responses", [])
                    if not responses:
                        continue
                    emb_guess = responses[0].get("embeddings_guess")
                    if emb_guess is None:
                        continue
                    stacks = _example_stacks(emb_guess)
                    if stacks is None:
                        continue
                    n_layers = stacks[0].shape[0]
                    ref = stacks[REF_TOKEN_INDEX][n_layers - 1, :].astype(np.float32, copy=False)
                    h_all = stacks[self.token_pos].astype(np.float32, copy=False)
                    yield ref, h_all
                    continue

                if ref_ds.ndim != 4 or h_ds.ndim != 4:
                    continue
                if ref_ds.shape[0] != h_ds.shape[0] or ref_ds.shape[-1] != h_ds.shape[-1]:
                    continue
                n_layers = int(ref_ds.shape[0])
                ref = np.asarray(ref_ds[n_layers - 1, 0, -1, :], dtype=np.float32)
                h_all = np.asarray(h_ds[:, 0, -1, :], dtype=np.float32)
                yield ref, h_all


def _augment_batch(
    refs: torch.Tensor,
    hs: torch.Tensor,
    n_negatives: int,
    noise_rel: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    refs: [B,D], hs: [B,L,D]
    Returns:
      X: [B*(1+n_neg), L, 2D]
      y: [B*(1+n_neg)]
    """
    bsz, n_layers, dim = hs.shape
    refs_dev = refs.to(device=device, dtype=torch.float32, non_blocking=True)
    hs_dev = hs.to(device=device, dtype=torch.float32, non_blocking=True)

    ref_pos = refs_dev.unsqueeze(1).expand(-1, n_layers, -1)
    x_pos = torch.cat([ref_pos, hs_dev], dim=-1)

    if n_negatives <= 0:
        y = torch.ones(bsz, device=device, dtype=torch.float32)
        return x_pos, y

    norms = torch.linalg.norm(refs_dev, dim=1, keepdim=True)
    sigma = noise_rel * norms / math.sqrt(max(dim, 1))
    sigma = sigma.unsqueeze(1)  # [B,1,1]
    noise = torch.randn(
        bsz, n_negatives, dim, device=device, dtype=torch.float32
    ) * sigma
    ref_neg = refs_dev.unsqueeze(1) + noise  # [B,N,D]
    ref_neg = ref_neg.unsqueeze(2).expand(-1, -1, n_layers, -1)  # [B,N,L,D]
    hs_neg = hs_dev.unsqueeze(1).expand(-1, n_negatives, -1, -1)  # [B,N,L,D]
    x_neg = torch.cat([ref_neg, hs_neg], dim=-1).reshape(
        bsz * n_negatives, n_layers, 2 * dim
    )

    x_pos_flat = x_pos.reshape(bsz, n_layers, 2 * dim)
    x_all = torch.cat([x_pos_flat, x_neg], dim=0)
    y = torch.cat(
        [
            torch.ones(bsz, device=device, dtype=torch.float32),
            torch.zeros(bsz * n_negatives, device=device, dtype=torch.float32),
        ],
        dim=0,
    )
    return x_all, y


class BatchedLayerLinearProbe(nn.Module):
    """
    Independent linear classifier per layer, trained in one batched module.
    """

    def __init__(self, n_layers: int, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_layers, input_dim))
        self.bias = nn.Parameter(torch.zeros(n_layers))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,L,F] -> logits: [N,L]
        return (x * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias.unsqueeze(0)


def _classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_pred = (prob >= 0.5).astype(np.int8)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        ll = float(log_loss(y_true, prob, labels=[0, 1]))
    except Exception:
        ll = float("nan")
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y_true, prob))
        except Exception:
            auc = float("nan")
    return {"accuracy": acc, "roc_auc": auc, "f1": f1, "log_loss": ll}


def _evaluate_all_layers(
    model: BatchedLayerLinearProbe,
    loader: DataLoader,
    n_layers: int,
    n_negatives: int,
    noise_rel: float,
    device: torch.device,
) -> tuple[list[dict], int]:
    model.eval()
    ys = [[] for _ in range(n_layers)]
    ps = [[] for _ in range(n_layers)]
    n_rows = 0
    with torch.no_grad():
        for refs, hs in tqdm(loader, desc="Eval", leave=False):
            Xb, yb = _augment_batch(refs, hs, n_negatives, noise_rel, device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_np = yb.detach().cpu().numpy().astype(np.int8)
            n_rows += int(y_np.shape[0])
            for layer_idx in range(n_layers):
                ys[layer_idx].append(y_np)
                ps[layer_idx].append(probs[:, layer_idx])

    metrics = []
    for layer_idx in range(n_layers):
        y_true = np.concatenate(ys[layer_idx], axis=0)
        prob = np.concatenate(ps[layer_idx], axis=0)
        metrics.append(_classification_metrics(y_true, prob))
    return metrics, n_rows


def write_semantic_probe_config_txt(
    run_dir: Path,
    args: argparse.Namespace,
    token_pos: int,
    layer_idx: int,
    n_train_rows: int,
    n_val_rows: int,
    n_layers: int,
    hidden_dim: int,
    metrics: dict,
    train_path: str,
    val_path: str,
) -> None:
    lines = [
        f"Semantic entailment probe (single token: tok_{token_pos}_guess, all layers)",
        "=" * 60,
        "Model: torch independent linear probes (one per layer, trained batched)",
        f"Device: {args.device}",
        f"Reference token index (guess span): {REF_TOKEN_INDEX}",
        f"Probe token position: {token_pos}",
        f"Probe layer index (0-based): {layer_idx}",
        f"Layer folder uses 1-based index: {layer_idx + 1}",
        f"Feature dim: {2 * hidden_dim} (concat ref + h)",
        f"Negatives per example: {args.n_negatives}",
        f"noise_rel: {args.noise_rel}",
        f"random_seed: {args.random_seed}",
        f"epochs: {args.epochs}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"batch_size: {args.batch_size}",
        f"num_workers: {args.num_workers}",
        f"prefetch_factor: {args.prefetch_factor}",
        f"persistent_workers: {args.persistent_workers}",
        f"use_swmr: {args.use_swmr}",
        f"Train path: {train_path}",
        f"Val path: {val_path}",
        "",
        "I/O note: many parallel readers on one shared H5 can saturate filesystem throughput.",
        "",
        "Data (rows after augmentation: 1 + n_negatives per example)",
        "-" * 40,
        f"Training rows: {n_train_rows}",
        f"Validation rows: {n_val_rows}",
        f"Total layers in stack: {n_layers}",
        "",
        "Metrics",
        "-" * 40,
        "Train:",
        f"  Accuracy: {metrics['train']['accuracy']:.6f}",
        (
            f"  ROC-AUC:  {metrics['train']['roc_auc']:.6f}"
            if not np.isnan(metrics["train"]["roc_auc"])
            else "  ROC-AUC:  nan"
        ),
        f"  F1:       {metrics['train']['f1']:.6f}",
        (
            f"  Log loss: {metrics['train']['log_loss']:.6f}"
            if not np.isnan(metrics["train"]["log_loss"])
            else "  Log loss: nan"
        ),
        "Validation:",
        f"  Accuracy: {metrics['val']['accuracy']:.6f}",
        (
            f"  ROC-AUC:  {metrics['val']['roc_auc']:.6f}"
            if not np.isnan(metrics["val"]["roc_auc"])
            else "  ROC-AUC:  nan"
        ),
        f"  F1:       {metrics['val']['f1']:.6f}",
        (
            f"  Log loss: {metrics['val']['log_loss']:.6f}"
            if not np.isnan(metrics["val"]["log_loss"])
            else "  Log loss: nan"
        ),
        "",
        f"Written at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(run_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def _plot_metrics_by_layer(
    layer_numbers: list[int],
    train_metrics: list[float],
    val_metrics: list[float],
    metric_name: str,
    metric_label: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    tr = np.ma.masked_invalid(np.asarray(train_metrics, dtype=np.float64))
    va = np.ma.masked_invalid(np.asarray(val_metrics, dtype=np.float64))
    ax.plot(layer_numbers, tr, "o-", label="Train", markersize=4)
    ax.plot(layer_numbers, va, "s-", label="Validation", markersize=4)
    ax.set_xlabel("Layer number")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{metric_name}_by_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "sem_probe_last_tok_all_layers"
    base.mkdir(parents=True, exist_ok=True)
    k = 1
    while (base / str(k)).exists():
        k += 1
    run_base = base / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _detect_shape(train_path: str, token_pos: int, use_swmr: bool) -> tuple[int, int]:
    ds = H5PairIterableDataset(train_path, token_pos=token_pos, use_swmr=use_swmr)
    for ref, h_all in ds:
        return int(h_all.shape[0]), int(ref.shape[0])
    raise ValueError(f"No valid training examples in {train_path}")


def _worker_init_fn(_worker_id: int):
    # h5py file handles are created in worker __iter__, so no shared handle here.
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Train tok_5_guess semantic probes over all layers using GPU-batched PyTorch (H5 input)"
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Run dir: output_dir/sem_probe_last_tok_all_layers/<id>/tok_5_guess/",
    )
    parser.add_argument("--n_negatives", type=int, default=3)
    parser.add_argument(
        "--noise_rel",
        type=float,
        default=0.5,
        help="Gaussian noise scale: sigma = noise_rel * ||ref|| / sqrt(d)",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=2000, help="Alias for compatibility")
    parser.add_argument("--plot", action="store_true", help="Save by-layer metric plots")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--no_save_model", action="store_false", dest="save_model")

    # GPU/optimization controls
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument(
        "--use_swmr",
        action="store_true",
        help="Open HDF5 with swmr=True for read-side safety on some setups",
    )
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_base = _get_run_base_dir(Path(args.output_dir))
    token_dir = run_base / f"tok_{TARGET_TOKEN_POS}_guess"
    token_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {token_dir}")
    print("Reading HDF5 with streaming dataset (read-only readers per worker/process).")

    n_layers, hidden_dim = _detect_shape(args.train_path, TARGET_TOKEN_POS, args.use_swmr)
    model = BatchedLayerLinearProbe(n_layers=n_layers, input_dim=2 * hidden_dim).to(device)

    tr_dataset = H5PairIterableDataset(
        args.train_path, token_pos=TARGET_TOKEN_POS, use_swmr=args.use_swmr
    )
    va_dataset = H5PairIterableDataset(
        args.val_path, token_pos=TARGET_TOKEN_POS, use_swmr=args.use_swmr
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_init_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(tr_dataset, **loader_kwargs)
    eval_train_loader = DataLoader(tr_dataset, **loader_kwargs)
    val_loader = DataLoader(va_dataset, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    pos_weight = torch.tensor([float(max(args.n_negatives, 1))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        n_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for refs, hs in progress:
            Xb, yb = _augment_batch(refs, hs, args.n_negatives, args.noise_rel, device)
            logits = model(Xb)
            y_expand = yb.unsqueeze(1).expand(-1, n_layers)
            loss = criterion(logits, y_expand)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1
            progress.set_postfix(loss=f"{running_loss / max(n_steps, 1):.4f}")
        print(f"Epoch {epoch + 1}: mean_loss={running_loss / max(n_steps, 1):.6f}")

    train_layer_metrics, n_train_rows = _evaluate_all_layers(
        model, eval_train_loader, n_layers, args.n_negatives, args.noise_rel, device
    )
    val_layer_metrics, n_val_rows = _evaluate_all_layers(
        model, val_loader, n_layers, args.n_negatives, args.noise_rel, device
    )

    layer_numbers: list[int] = []
    train_acc, train_auc, train_f1 = [], [], []
    val_acc, val_auc, val_f1 = [], [], []

    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx + 1}"
        layer_dir = token_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "train": train_layer_metrics[layer_idx],
            "val": val_layer_metrics[layer_idx],
        }
        layer_numbers.append(layer_idx + 1)
        train_acc.append(metrics["train"]["accuracy"])
        train_auc.append(metrics["train"]["roc_auc"])
        train_f1.append(metrics["train"]["f1"])
        val_acc.append(metrics["val"]["accuracy"])
        val_auc.append(metrics["val"]["roc_auc"])
        val_f1.append(metrics["val"]["f1"])

        if args.save_model:
            payload = {
                "model_type": "torch_batched_layer_linear",
                "layer_idx": layer_idx,
                "token_pos": TARGET_TOKEN_POS,
                "ref_token_index": REF_TOKEN_INDEX,
                "n_negatives": args.n_negatives,
                "noise_rel": args.noise_rel,
                "random_seed": args.random_seed,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": str(device),
                "metrics": metrics,
                "weight": model.weight[layer_idx].detach().cpu().numpy(),
                "bias": float(model.bias[layer_idx].detach().cpu().item()),
                "feature_dim": 2 * hidden_dim,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
            }
            with open(layer_dir / "semantic_probe.pkl", "wb") as f:
                pickle.dump(payload, f)

            write_semantic_probe_config_txt(
                layer_dir,
                args,
                TARGET_TOKEN_POS,
                layer_idx,
                n_train_rows,
                n_val_rows,
                n_layers,
                hidden_dim,
                metrics,
                str(args.train_path),
                str(args.val_path),
            )
            print(
                f"Saved {layer_dir / 'semantic_probe.pkl'} "
                f"val_acc={metrics['val']['accuracy']:.4f}"
            )

    if args.plot and layer_numbers:
        _plot_metrics_by_layer(
            layer_numbers,
            train_acc,
            val_acc,
            "accuracy",
            "Accuracy",
            token_dir,
        )
        _plot_metrics_by_layer(
            layer_numbers,
            train_auc,
            val_auc,
            "auc",
            "ROC-AUC",
            token_dir,
        )
        _plot_metrics_by_layer(
            layer_numbers,
            train_f1,
            val_f1,
            "f1",
            "F1",
            token_dir,
        )

    print(f"Done. Outputs under {run_base}")


if __name__ == "__main__":
    main()
