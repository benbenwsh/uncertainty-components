"""
Train binary logistic probes across all layers for one guess token position.

This variant only probes token position 4 ("tok_4_guess").
Target remains: whether a hidden state at (layer, token 4) pairs with the true
last-layer reference at guess token index 4.

Loads verbalised-embeddings pickles (embeddings_guess only). Features are
concat(ref, h) where ref is last-layer embedding at token 4 and h is
layer-ell embedding at token 4. Synthetic negatives corrupt ref with isotropic
Gaussian noise (same h, label 0).
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

try:
    from verbalised_confidence_probes.train_multitoken_verbalised_confidence_probe import (
        _tensor_to_numpy,
    )
except ImportError:
    from train_multitoken_verbalised_confidence_probe import _tensor_to_numpy

REF_TOKEN_INDEX = 5
TARGET_TOKEN_POS = 5
NUM_GUESS_TOKENS = 6  # token positions 0..5 inclusive


def _load_pickle_batches_guess_only(path: str) -> dict:
    data = {}
    with open(path, "rb") as f:
        while True:
            print(f"Length of data: {len(data)}")
            try:
                batch = pickle.load(f)
            except EOFError:
                break
            if not isinstance(batch, dict):
                print(f"WARNING: Skipping non-dict batch: {type(batch)}")
                continue
            for example_id, example_data in batch.items():
                responses = example_data.get("responses", [])
                if not responses:
                    continue
                r = responses[0]
                emb_guess = r.get("embeddings_guess")
                if emb_guess is None:
                    continue
                data[example_id] = {"responses": [{"embeddings_guess": emb_guess}]}
    return data


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
    data: dict,
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

    for _eid, example_data in data.items():
        responses = example_data.get("responses", [])
        if not responses:
            continue
        assert len(responses) == 1, "Each example should have one response"
        emb_guess = responses[0].get("embeddings_guess")
        assert emb_guess is not None, "embeddings_guess should not be None"
        stacks = _example_stacks(emb_guess)
        assert stacks is not None, "stacks should not be None"

        if n_layers is None:
            n_layers = stacks[0].shape[0]
            hidden_dim = stacks[0].shape[1]
        ref = stacks[REF_TOKEN_INDEX][n_layers - 1, :].astype(np.float32, copy=False)
        h = stacks[token_pos][layer_idx, :].astype(np.float32, copy=False)
        x_pos = np.concatenate([ref, h], axis=0)
        X_list.append(x_pos)
        y_list.append(1.0)

        d = ref.size
        norm = np.linalg.norm(ref)
        sigma = noise_rel * norm / np.sqrt(max(d, 1))
        for _ in range(n_negatives):
            noise = rng.normal(0.0, sigma, size=ref.shape)
            ref_fake = ref + noise
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
        "Semantic entailment probe (single token: tok_4_guess, all layers)",
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


def main():
    parser = argparse.ArgumentParser(
        description="Train semantic entailment logistic probes for tok_4_guess across all layers"
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Run dir: output_dir/sem_probe_last_tok_all_layers/<id>/tok_4_guess/",
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
    parser.add_argument("--plot", action="store_true", help="Save by-layer metric plots")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--no_save_model", action="store_false", dest="save_model")
    args = parser.parse_args()

    run_base = _get_run_base_dir(Path(args.output_dir))
    token_dir = run_base / f"tok_{TARGET_TOKEN_POS}_guess"
    token_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {token_dir}")

    print("Loading pickles...")
    train_data = _load_pickle_batches_guess_only(args.train_path)
    test_data = _load_pickle_batches_guess_only(args.test_path)
    print(f"Train examples: {len(train_data)}, test examples: {len(test_data)}")

    rng_train = np.random.default_rng(args.random_seed)
    rng_test = np.random.default_rng(args.random_seed + 1)

    n_layers = None
    for _eid, example_data in train_data.items():
        assert len(example_data["responses"]) == 1, "Each example should have one response"
        emb_guess = example_data["responses"][0].get("embeddings_guess")
        stacks = _example_stacks(emb_guess)
        if stacks:
            n_layers = stacks[0].shape[0]
            break
    if n_layers is None:
        print(f"ERROR: No valid training examples (need >= {NUM_GUESS_TOKENS} guess tokens).")
        return

    layer_numbers: list[int] = []
    train_acc, train_auc, train_f1 = [], [], []
    test_acc, test_auc, test_f1 = [], [], []

    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx + 1}"
        layer_dir = token_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building train data for layer {layer_idx + 1}...")
        X_tr, y_tr, nl_tr, hd_tr = build_xy_for_layer_token(
            train_data,
            TARGET_TOKEN_POS,
            layer_idx,
            args.n_negatives,
            args.noise_rel,
            rng_train,
        )
        print(f"Building test data for layer {layer_idx + 1}...")
        X_va, y_va, nl_va, hd_va = build_xy_for_layer_token(
            test_data,
            TARGET_TOKEN_POS,
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
                        "token_pos": TARGET_TOKEN_POS,
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
                TARGET_TOKEN_POS,
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

    if args.plot and layer_numbers:
        _plot_metrics_by_layer(
            layer_numbers,
            train_acc,
            test_acc,
            "accuracy",
            "Accuracy",
            token_dir,
        )
        _plot_metrics_by_layer(
            layer_numbers,
            train_auc,
            test_auc,
            "auc",
            "ROC-AUC",
            token_dir,
        )
        _plot_metrics_by_layer(
            layer_numbers,
            train_f1,
            test_f1,
            "f1",
            "F1",
            token_dir,
        )

    print(f"Done. Outputs under {run_base}")


if __name__ == "__main__":
    main()
