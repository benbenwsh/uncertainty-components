"""
Train binary logistic probes across all layers for each guess token position 0..5.

For each token index n, output lives under tok_n_guess/. Reference remains the last-layer
embedding at guess token index REF_TOKEN_INDEX (5). The probe uses layer-ell embedding at
token n paired with that reference (concat(ref, h)). Synthetic negatives corrupt ref with
isotropic Gaussian noise (same h, label 0).

Loads verbalised-embeddings native HDF5 examples (embeddings_guess only).

Per-token RNG streams (train/test) use offsets from random_seed so negative sampling differs
across token folders.

Output layout: output_dir/sem_probe_mult_toks_all_layers/<run_id>/tok_n_guess/ and, when
plotting, combined metric PNGs at the run_id level (accuracy_all_tokens.png, etc.).

Use --plots_only --run_dir <existing_run> to regenerate per-token and combined plots from
saved semantic_probe.pkl files without retraining. --more_graphs (default off) adds AUC and
F1 plots in addition to accuracy.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm

try:
    from verbalised_confidence_probes.train_multitoken_verbalised_confidence_probe import (
        _tensor_to_numpy,
    )
except ImportError:
    from train_multitoken_verbalised_confidence_probe import _tensor_to_numpy

REF_TOKEN_INDEX = 5
NUM_GUESS_TOKENS = 6  # token positions 0..5 inclusive


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
    Fast-path loader: returns HDF5 dataset for embeddings_guess[token_idx] if present.
    Structure expected from native writer:
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


def _iter_layer_pairs_from_h5(
    path: str,
    token_pos: int,
    layer_idx: int,
):
    """
    Yield (ref_vec, h_vec, n_layers, hidden_dim) per valid example for one layer.

    Reads only:
    - ref token stack at last layer (REF_TOKEN_INDEX, -1)
    - probe token stack at requested layer (token_pos, layer_idx)
    """
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        for _example_id in tqdm(examples_group.keys(), desc="Streaming examples", leave=False):
            example_group = examples_group[_example_id]

            ref_ds = _read_token_stack_ds(example_group, REF_TOKEN_INDEX)
            h_ds = _read_token_stack_ds(example_group, token_pos)
            if ref_ds is None or h_ds is None:
                # Slow-path fallback for unexpected legacy structure.
                example_data = _read_h5_node(example_group)
                if not isinstance(example_data, dict):
                    continue
                responses = example_data.get("responses", [])
                if not responses:
                    continue
                r0 = responses[0]
                emb_guess = r0.get("embeddings_guess")
                if emb_guess is None or len(emb_guess) != NUM_GUESS_TOKENS:
                    continue
                stacks = _example_stacks(emb_guess)
                if stacks is None:
                    continue
                n_layers = stacks[0].shape[0]
                hidden_dim = stacks[0].shape[1]
                if layer_idx < 0 or layer_idx >= n_layers:
                    continue
                ref = stacks[REF_TOKEN_INDEX][n_layers - 1, :].astype(np.float32, copy=False)
                h = stacks[token_pos][layer_idx, :].astype(np.float32, copy=False)
                yield ref, h, n_layers, hidden_dim
                continue

            if ref_ds.ndim != 4 or h_ds.ndim != 4:
                continue
            n_layers = int(ref_ds.shape[0])
            hidden_dim = int(ref_ds.shape[-1])
            if h_ds.shape[0] != n_layers or h_ds.shape[-1] != hidden_dim:
                continue
            if layer_idx < 0 or layer_idx >= n_layers:
                continue

            # Shape convention: [layers, batch, seq, dim].
            ref = np.asarray(ref_ds[n_layers - 1, 0, -1, :], dtype=np.float32)
            h = np.asarray(h_ds[layer_idx, 0, -1, :], dtype=np.float32)
            yield ref, h, n_layers, hidden_dim


def _stack_guess_token(emb) -> np.ndarray:
    arr = _tensor_to_numpy(emb)
    return arr[:, 0, -1, :]


def _example_stacks(emb_guess: list) -> list[np.ndarray] | None:
    assert len(emb_guess) == NUM_GUESS_TOKENS, "Length mismatch"
    stacks = []
    for t in range(NUM_GUESS_TOKENS):
        s = _stack_guess_token(emb_guess[t])
        stacks.append(s)

    n_layers = stacks[0].shape[0]
    hidden_dim = stacks[0].shape[1]
    for s in stacks:
        assert s.shape == (n_layers, hidden_dim), "Shape mismatch"

    return stacks


def build_xy_for_layer_token(
    h5_path: str,
    token_pos: int,
    layer_idx: int,
    n_negatives: int,
    noise_rel: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    n_layers = None
    hidden_dim = None

    for ref, h, nl, hd in _iter_layer_pairs_from_h5(h5_path, token_pos, layer_idx):
        if n_layers is None:
            n_layers = nl
            hidden_dim = hd
        x_pos = np.concatenate([ref, h], axis=0)
        if (np.array_equal(ref, h)):
            print(f"layer_idx: {layer_idx}, token_pos: {token_pos}, these two should be last")
            if layer_idx != n_layers - 1 or token_pos != REF_TOKEN_INDEX:
                raise ValueError(f"layer_idx: {layer_idx}, token_pos: {token_pos}, these two should be last layer and ref token")
        X_list.append(x_pos)
        y_list.append(1.0)

        norm = np.linalg.norm(ref)
        for _ in range(n_negatives):
            ref_fake = rng.normal(0.0, 1.0, size=ref.shape).astype(np.float32, copy=False)
            fake_norm = np.linalg.norm(ref_fake)
            if fake_norm > 0:
                ref_fake = ref_fake * (norm / fake_norm)
            else:
                ref_fake = np.zeros_like(ref)
            X_list.append(np.concatenate([ref_fake, h], axis=0))
            y_list.append(0.0)

    if not X_list or n_layers is None or hidden_dim is None:
        raise ValueError("No valid examples in train or test data.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int8)
    return X, y, n_layers, hidden_dim


def _classification_metrics(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    try:
        proba = model.predict_proba(X)[:, 1]
        ll = float(log_loss(y, proba, labels=[0, 1]))
    except Exception:
        ll = float("nan")
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y, proba))
        except Exception:
            auc = float("nan")
    return {"accuracy": acc, "roc_auc": auc, "f1": f1, "log_loss": ll}


def train_semantic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    max_iter: int,
) -> tuple[LogisticRegression, dict]:
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    metrics = {
        "train": _classification_metrics(model, X_train, y_train),
        "test": _classification_metrics(model, X_test, y_test),
    }
    return model, metrics


def write_semantic_probe_config_txt(
    run_dir: Path,
    args: argparse.Namespace,
    token_pos: int,
    layer_idx: int,
    n_train_rows: int,
    n_test_rows: int,
    n_layers: int,
    hidden_dim: int,
    metrics: dict,
    train_path: str,
    test_path: str,
) -> None:
    lines = [
        f"Semantic entailment probe (multitoken: tok_{token_pos}_guess, all layers)",
        "=" * 60,
        "Model: sklearn.linear_model.LogisticRegression (class_weight=balanced)",
        f"Reference token index (guess span): {REF_TOKEN_INDEX}",
        f"Probe token position: {token_pos}",
        f"Probe layer index (0-based): {layer_idx}",
        f"Layer folder uses 1-based index: {layer_idx + 1}",
        f"Feature dim: {2 * hidden_dim} (concat ref + h)",
        f"Negatives per example: {args.n_negatives}",
        f"noise_rel: {args.noise_rel}",
        f"random_seed: {args.random_seed}",
        f"max_iter: {args.max_iter}",
        f"Train path: {train_path}",
        f"Test path: {test_path}",
        "",
        "Data (rows after augmentation: 1 + n_negatives per example)",
        "-" * 40,
        f"Training rows: {n_train_rows}",
        f"Test rows: {n_test_rows}",
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
        "Test:",
        f"  Accuracy: {metrics['test']['accuracy']:.6f}",
        (
            f"  ROC-AUC:  {metrics['test']['roc_auc']:.6f}"
            if not np.isnan(metrics["test"]["roc_auc"])
            else "  ROC-AUC:  nan"
        ),
        f"  F1:       {metrics['test']['f1']:.6f}",
        (
            f"  Log loss: {metrics['test']['log_loss']:.6f}"
            if not np.isnan(metrics["test"]["log_loss"])
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
    test_metrics: list[float],
    metric_name: str,
    metric_label: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    tr = np.ma.masked_invalid(np.asarray(train_metrics, dtype=np.float64))
    va = np.ma.masked_invalid(np.asarray(test_metrics, dtype=np.float64))
    ax.plot(layer_numbers, tr, "o-", label="Train", markersize=4)
    ax.plot(layer_numbers, va, "s-", label="Test", markersize=4)
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


def _parse_tok_guess_name(token_name: str) -> tuple[int | None, str | None]:
    match = re.fullmatch(r"tok_(\d+)_guess", token_name)
    if match is None:
        return None, None
    return int(match.group(1)), "guess"


def _token_sort_key_guess(token_name: str) -> tuple:
    token_pos, _ = _parse_tok_guess_name(token_name)
    if token_pos is None:
        return (float("inf"), token_name)
    return (token_pos, token_name)


def _style_for_token_order(order_idx: int, total_tokens: int) -> dict:
    if total_tokens <= 1:
        return {"linewidth": 2.2, "alpha": 1.0}
    frac = order_idx / float(total_tokens - 1)
    return {"linewidth": 1.5 + 1.7 * frac, "alpha": 0.7 + 0.3 * frac}


def _marker_for_token_pos(token_pos: int) -> str:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    return marker_cycle[token_pos % len(marker_cycle)]


def _sorted_semantic_token_items(all_token_metrics: dict) -> list[tuple[str, dict, int]]:
    items: list[tuple[str, dict, int]] = []
    for token_name, metrics_dict in all_token_metrics.items():
        token_pos = metrics_dict.get("token_pos")
        if token_pos is None:
            tp, _ = _parse_tok_guess_name(token_name)
            token_pos = tp if tp is not None else -1
        items.append((token_name, metrics_dict, int(token_pos)))
    return sorted(items, key=lambda x: x[2])


def _load_semantic_metrics_from_run_dir(run_base: Path) -> dict:
    """Rebuild per-layer metric lists from tok_*_guess/layer_*/semantic_probe.pkl."""
    all_token_metrics: dict = {}
    token_dirs = [d for d in run_base.iterdir() if d.is_dir() and d.name.startswith("tok_")]
    for token_dir in sorted(token_dirs, key=lambda p: _token_sort_key_guess(p.name)):
        token_name = token_dir.name
        token_pos, _ = _parse_tok_guess_name(token_name)
        if token_pos is None:
            logging.warning("Skipping unexpected directory name: %s", token_dir)
            continue

        layer_pkls = sorted(
            token_dir.glob("layer_*/semantic_probe.pkl"),
            key=lambda p: int(p.parent.name.split("_")[-1]) if p.parent.name.split("_")[-1].isdigit() else float("inf"),
        )
        if not layer_pkls:
            logging.warning("No semantic_probe.pkl under %s", token_dir)
            continue

        layer_numbers: list[int] = []
        train_acc, train_auc, train_f1 = [], [], []
        test_acc, test_auc, test_f1 = [], [], []

        for layer_pkl in layer_pkls:
            try:
                with open(layer_pkl, "rb") as f:
                    payload = pickle.load(f)
            except Exception as exc:
                logging.warning("Failed to read %s: %s", layer_pkl, exc)
                continue
            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            if not isinstance(metrics, dict):
                logging.warning("Missing metrics in %s", layer_pkl)
                continue
            tr = metrics.get("train", {})
            va = metrics.get("test", {})
            try:
                layer_num = int(layer_pkl.parent.name.split("_")[-1])
                layer_numbers.append(layer_num)
                train_acc.append(float(tr["accuracy"]))
                train_auc.append(float(tr["roc_auc"]))
                train_f1.append(float(tr["f1"]))
                test_acc.append(float(va["accuracy"]))
                test_auc.append(float(va["roc_auc"]))
                test_f1.append(float(va["f1"]))
            except (KeyError, TypeError, ValueError) as exc:
                logging.warning("Invalid metrics in %s: %s", layer_pkl, exc)
                continue

        if not layer_numbers:
            logging.warning("No valid metrics for %s", token_dir)
            continue

        order = np.argsort(np.array(layer_numbers))
        layer_numbers = [layer_numbers[i] for i in order]
        train_acc = [train_acc[i] for i in order]
        train_auc = [train_auc[i] for i in order]
        train_f1 = [train_f1[i] for i in order]
        test_acc = [test_acc[i] for i in order]
        test_auc = [test_auc[i] for i in order]
        test_f1 = [test_f1[i] for i in order]

        all_token_metrics[token_name] = {
            "train": [train_acc, train_auc, train_f1],
            "test": [test_acc, test_auc, test_f1],
            "layers": layer_numbers,
            "token_pos": token_pos,
        }
    return all_token_metrics


def _plot_all_metrics_by_layer_semantic(
    token_dir: Path,
    layer_numbers: list[int],
    train_acc: list[float],
    train_auc: list[float],
    train_f1: list[float],
    test_acc: list[float],
    test_auc: list[float],
    test_f1: list[float],
    more_graphs: bool,
) -> None:
    metrics_config = [
        ("accuracy", "Accuracy", train_acc, test_acc),
        ("auc", "ROC-AUC", train_auc, test_auc),
        ("f1", "F1", train_f1, test_f1),
    ]
    if not more_graphs:
        metrics_config = metrics_config[:1]
    for metric_name, metric_label, tr, va in metrics_config:
        _plot_metrics_by_layer(layer_numbers, tr, va, metric_name, metric_label, token_dir)


def _plot_metrics_all_tokens_semantic(
    run_base: Path,
    all_token_metrics: dict,
    more_graphs: bool,
) -> None:
    """Combined plots at run root: Blues for train, Oranges for test; token order via width/alpha."""
    token_items = _sorted_semantic_token_items(all_token_metrics)
    if not token_items:
        return
    n_tok = len(token_items)
    blues = plt.cm.Blues(np.linspace(0.45, 0.85, n_tok))
    oranges = plt.cm.Oranges(np.linspace(0.45, 0.85, n_tok))

    metric_specs = [
        (0, "accuracy", "Accuracy"),
        (1, "auc", "ROC-AUC"),
        (2, "f1", "F1"),
    ]
    if not more_graphs:
        metric_specs = metric_specs[:1]

    for metric_idx, fname, title_metric in metric_specs:
        fig, ax = plt.subplots(figsize=(11, 6))
        for order_idx, (token_name, metrics_dict, token_pos) in enumerate(token_items):
            style = _style_for_token_order(order_idx, n_tok)
            layer_numbers = metrics_dict["layers"]
            train_vals = metrics_dict["train"][metric_idx]
            test_vals = metrics_dict["test"][metric_idx]
            marker = _marker_for_token_pos(token_pos)
            ax.plot(
                layer_numbers,
                train_vals,
                label=f"{token_name} (Train)",
                marker=marker,
                markersize=4,
                color=blues[order_idx],
                linestyle="-",
                linewidth=style["linewidth"],
                alpha=style["alpha"],
            )
            ax.plot(
                layer_numbers,
                test_vals,
                label=f"{token_name} (Test)",
                marker=marker,
                markersize=4,
                color=oranges[order_idx],
                linestyle="--",
                linewidth=style["linewidth"],
                alpha=max(0.6, style["alpha"] - 0.05),
            )
        ax.set_xlabel("Layer number")
        ax.set_ylabel(title_metric)
        ax.set_title(f"{title_metric} by layer — all guess tokens (train=blue, test=orange)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = run_base / f"{fname}_all_tokens.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "sem_probe_mult_toks_all_layers"
    base.mkdir(parents=True, exist_ok=True)
    k = 1
    while (base / str(k)).exists():
        k += 1
    run_base = base / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def main():
    parser = argparse.ArgumentParser(
        description="Train semantic entailment logistic probes for tok_0..5_guess across all layers (H5 input)"
    )
    parser.add_argument("--train_path", type=str, required=False)
    parser.add_argument("--test_path", type=str, required=False)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Run dir: output_dir/sem_probe_mult_toks_all_layers/<id>/tok_n_guess/; combined PNGs at <id>/",
    )
    parser.add_argument("--n_negatives", type=int, default=3)
    parser.add_argument(
        "--noise_rel",
        type=float,
        default=0.5,
        help="Gaussian noise scale: sigma = noise_rel * ||ref|| / sqrt(d)",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--plot", action="store_true", help="Save by-layer metric plots and combined all-tokens plots")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--no_save_model", action="store_false", dest="save_model")
    parser.add_argument(
        "--more_graphs",
        action="store_true",
        default=False,
        help="If set, also generate AUC and F1 plots (default: accuracy only).",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Regenerate metric plots from an existing run_dir using saved semantic_probe.pkl files",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing run directory (e.g. results/sem_probe_mult_toks_all_layers/2) with --plots_only",
    )
    args = parser.parse_args()

    if not args.plots_only and (not args.train_path or not args.test_path):
        parser.error("--train_path and --test_path are required unless --plots_only is enabled.")
    if args.plots_only and not args.run_dir:
        parser.error("--run_dir is required when --plots_only is enabled.")

    if args.plots_only:
        run_base = Path(args.run_dir)
        if not run_base.exists() or not run_base.is_dir():
            parser.error(f"--run_dir does not exist or is not a directory: {run_base}")
        output_log_path = run_base / "output_plots_only.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(output_log_path, mode="w"),
                logging.StreamHandler(),
            ],
        )
        logging.info("Plots-only mode. Rebuilding from %s", run_base)
        all_token_metrics = _load_semantic_metrics_from_run_dir(run_base)
        if not all_token_metrics:
            logging.error("No token metrics found under %s", run_base)
            return
        for token_name, metrics_dict in sorted(all_token_metrics.items(), key=lambda x: _token_sort_key_guess(x[0])):
            token_dir = run_base / token_name
            tr, va = metrics_dict["train"], metrics_dict["test"]
            logging.info("Regenerating per-token plots for %s", token_name)
            _plot_all_metrics_by_layer_semantic(
                token_dir,
                metrics_dict["layers"],
                tr[0],
                tr[1],
                tr[2],
                va[0],
                va[1],
                va[2],
                more_graphs=args.more_graphs,
            )
        logging.info("Regenerating combined run-level plots...")
        _plot_metrics_all_tokens_semantic(run_base, all_token_metrics, more_graphs=args.more_graphs)
        logging.info("Done. Plots under %s", run_base)
        return

    run_base = _get_run_base_dir(Path(args.output_dir))
    print(f"Run base directory: {run_base}")

    print("Streaming HDF5 examples lazily per layer (all guess token positions 0..5)...")

    all_token_metrics: dict = {}

    for token_pos in range(NUM_GUESS_TOKENS):
        token_dir = run_base / f"tok_{token_pos}_guess"
        token_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- tok_{token_pos}_guess -> {token_dir}")

        rng_train = np.random.default_rng(args.random_seed + token_pos)
        rng_test = np.random.default_rng(args.random_seed + 1 + token_pos)

        n_layers = None
        for _ref, _h, nl, _hd in _iter_layer_pairs_from_h5(args.train_path, token_pos, 0):
            n_layers = nl
            break
        if n_layers is None:
            raise ValueError(
                f"No valid training examples for token_pos={token_pos} "
                f"(need >= {NUM_GUESS_TOKENS} guess tokens)."
            )

        layer_numbers: list[int] = []
        train_acc, train_auc, train_f1 = [], [], []
        test_acc, test_auc, test_f1 = [], [], []

        for layer_idx in range(n_layers):
            layer_name = f"layer_{layer_idx + 1}"
            layer_dir = token_dir / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)

            print(f"Building train data for tok_{token_pos}_guess layer {layer_idx + 1}...")
            X_tr, y_tr, nl_tr, hd_tr = build_xy_for_layer_token(
                args.train_path,
                token_pos,
                layer_idx,
                args.n_negatives,
                args.noise_rel,
                rng_train,
            )
            print(f"Building test data for tok_{token_pos}_guess layer {layer_idx + 1}...")
            X_va, y_va, nl_va, hd_va = build_xy_for_layer_token(
                args.test_path,
                token_pos,
                layer_idx,
                args.n_negatives,
                args.noise_rel,
                rng_test,
            )

            if X_tr.size == 0 or X_va.size == 0 or y_tr.size == 0 or y_va.size == 0:
                raise ValueError(f"Empty train or test data at {token_dir.name} {layer_name}")

            if nl_tr != n_layers or nl_va != n_layers or hd_tr != hd_va:
                raise ValueError(f"Layer/dim mismatch at {token_dir.name} {layer_name}")

            model, metrics = train_semantic_probe(
                X_tr,
                y_tr,
                X_va,
                y_va,
                random_state=args.random_seed,
                max_iter=args.max_iter,
            )

            layer_numbers.append(layer_idx + 1)
            train_acc.append(metrics["train"]["accuracy"])
            train_auc.append(metrics["train"]["roc_auc"])
            train_f1.append(metrics["train"]["f1"])
            test_acc.append(metrics["test"]["accuracy"])
            test_auc.append(metrics["test"]["roc_auc"])
            test_f1.append(metrics["test"]["f1"])

            if args.save_model:
                with open(layer_dir / "semantic_probe.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "model": model,
                            "metrics": metrics,
                            "layer_idx": layer_idx,
                            "token_pos": token_pos,
                            "ref_token_index": REF_TOKEN_INDEX,
                            "n_negatives": args.n_negatives,
                            "noise_rel": args.noise_rel,
                            "random_seed": args.random_seed,
                            "max_iter": args.max_iter,
                        },
                        f,
                    )
                write_semantic_probe_config_txt(
                    layer_dir,
                    args,
                    token_pos,
                    layer_idx,
                    len(y_tr),
                    len(y_va),
                    nl_tr,
                    hd_tr,
                    metrics,
                    str(args.train_path),
                    str(args.test_path),
                )
                print(
                    f"Saved {layer_dir / 'semantic_probe.pkl'} "
                    f"test_acc={metrics['test']['accuracy']:.4f}"
                )

        all_token_metrics[f"tok_{token_pos}_guess"] = {
            "train": [train_acc, train_auc, train_f1],
            "test": [test_acc, test_auc, test_f1],
            "layers": layer_numbers,
            "token_pos": token_pos,
        }

        if args.plot and layer_numbers:
            _plot_all_metrics_by_layer_semantic(
                token_dir,
                layer_numbers,
                train_acc,
                train_auc,
                train_f1,
                test_acc,
                test_auc,
                test_f1,
                more_graphs=args.more_graphs,
            )

    if args.plot and all_token_metrics:
        _plot_metrics_all_tokens_semantic(run_base, all_token_metrics, more_graphs=args.more_graphs)

    print(f"\nDone. Outputs under {run_base}")


if __name__ == "__main__":
    main()
