"""
Train linear probes for multiple token positions across all layers to predict verbalised confidence.

HDF5 variant of train_multitoken_verbalised_confidence_probe.py: loads native HDF5 examples
(produced by semantic_uncertainty/process_generations_verbalised_embeddings_h5.py) instead of
pickle files. Training, outputs, and directory layout match the pickle script:
results/mult_toks_all_layers/<run_id>/tok_n_guess/ and tok_n_probability/ directories.

Usage:
    python h5_train_multitoken_verbalised_confidence_probe.py \\
        --train_path path/to/train_verbalised_embeddings.h5 \\
        --val_path path/to/validation_verbalised_embeddings.h5 \\
        [--output_dir ./results] \\
        [--model_type ridge] \\
        [--alpha 1.0] \\
        [--plot]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from verbalised_confidence_probes.train_verbalised_confidence_probe import (
        plot_results,
        train_verbalised_confidence_probe,
        write_config_txt,
    )
except ImportError:
    print("Importing from train_verbalised_confidence_probe.py this way")
    from train_verbalised_confidence_probe import (
        plot_results,
        train_verbalised_confidence_probe,
        write_config_txt,
    )


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


def _load_h5_examples_multitoken(path: str) -> dict:
    """
    Load HDF5 file with root group ``examples``. Each example should have
    responses[0] with verbalised_confidence, embeddings_guess, embeddings_probability
    (same semantics as the pickle multitoken loader).
    """
    data: dict = {}
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        for example_id in examples_group.keys():
            example_id = str(example_id)
            example_group = examples_group[example_id]
            example_data = _read_h5_node(example_group)
            if not isinstance(example_data, dict):
                logging.warning("Skipping example %s: not a dict after read", example_id)
                continue
            responses = example_data.get("responses", [])
            if not responses:
                continue
            r = responses[0]
            if not isinstance(r, dict):
                logging.warning("Skipping example %s: responses[0] not a dict", example_id)
                continue
            vc = r.get("verbalised_confidence")
            emb_guess = r.get("embeddings_guess")
            emb_prob = r.get("embeddings_probability")
            if vc is None or emb_guess is None or emb_prob is None:
                logging.warning("ERROR: Missing data for example %s", example_id)
                continue
            data[example_id] = {
                "responses": [
                    {
                        "verbalised_confidence": vc,
                        "embeddings_guess": emb_guess,
                        "embeddings_probability": emb_prob,
                    }
                ]
            }
    return data


def _tensor_to_numpy(obj):
    """Convert tensor or array-like to numpy."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def load_multitoken_verbalised_confidence_data(train_path, val_path):
    """
    Load HDF5 train/val files and extract verbalised confidence and embeddings for all token positions.

    Returns:
        dict: {
            'guess': {
                token_pos: (X_train, X_val, y_train, y_val, n_layers)
            },
            'probability': {
                token_pos: (X_train, X_val, y_train, y_val, n_layers)
            }
        }
    """
    train_data = _load_h5_examples_multitoken(train_path)
    val_data = _load_h5_examples_multitoken(val_path)

    def extract_token_positions_and_labels(data_dict, embedding_type):
        """
        Extract embeddings for all token positions from embeddings_guess or embeddings_probability.

        Returns:
            dict: {token_pos: (X_array, y_array)} where X_array shape is (n_examples, n_layers, hidden_dim)
        """
        max_token_pos = -1
        n_layers = None
        hidden_dim = None

        for example_id, example_data in data_dict.items():
            responses = example_data.get("responses", [])
            if len(responses) == 0:
                continue
            response = responses[0]
            emb_list = response.get(f"embeddings_{embedding_type}")
            if emb_list is None or len(emb_list) == 0:
                continue

            first_emb = _tensor_to_numpy(emb_list[0])
            if n_layers is None:
                n_layers = first_emb.shape[0]
                hidden_dim = (
                    first_emb.shape[3]
                    if len(first_emb.shape) >= 4
                    else (first_emb.shape[2] if len(first_emb.shape) >= 3 else first_emb.shape[1])
                )

            max_token_pos = max(max_token_pos, len(emb_list) - 1)

        if max_token_pos < 0:
            return {}

        token_data = {pos: {"X": [], "y": []} for pos in range(max_token_pos + 1)}

        for example_id, example_data in data_dict.items():
            responses = example_data.get("responses", [])
            if len(responses) == 0:
                continue
            response = responses[0]
            verbalised_confidence = response.get("verbalised_confidence")
            emb_list = response.get(f"embeddings_{embedding_type}")

            if verbalised_confidence is None or emb_list is None:
                continue

            for token_pos, emb in enumerate(emb_list):
                if token_pos > max_token_pos:
                    break

                emb_array = _tensor_to_numpy(emb)
                layer_embs = emb_array[:, 0, -1, :]

                if layer_embs.shape[0] != n_layers or layer_embs.shape[1] != hidden_dim:
                    logging.warning(
                        "Shape mismatch for example %s, token %s: expected (%s, %s), got %s",
                        example_id,
                        token_pos,
                        n_layers,
                        hidden_dim,
                        layer_embs.shape,
                    )
                    continue

                token_data[token_pos]["X"].append(layer_embs)
                token_data[token_pos]["y"].append(float(verbalised_confidence))

        result = {}
        for token_pos in range(max_token_pos + 1):
            if len(token_data[token_pos]["X"]) == 0:
                continue
            X_array = np.array(token_data[token_pos]["X"])
            y_array = np.array(token_data[token_pos]["y"])
            result[token_pos] = (X_array, y_array, n_layers)

        return result

    train_guess = extract_token_positions_and_labels(train_data, "guess")
    train_prob = extract_token_positions_and_labels(train_data, "probability")
    val_guess = extract_token_positions_and_labels(val_data, "guess")
    val_prob = extract_token_positions_and_labels(val_data, "probability")

    result = {"guess": {}, "probability": {}}

    for embedding_type, train_dict, val_dict in [
        ("guess", train_guess, val_guess),
        ("probability", train_prob, val_prob),
    ]:
        common_positions = set(train_dict.keys()) & set(val_dict.keys())
        if len(common_positions) == 0:
            logging.warning("No common token positions found for %s embeddings", embedding_type)
            continue

        for token_pos in sorted(common_positions):
            X_train, y_train, n_layers_train = train_dict[token_pos]
            X_val, y_val, n_layers_val = val_dict[token_pos]

            if n_layers_train != n_layers_val:
                logging.warning(
                    "Layer count mismatch for %s token %s: train=%s, val=%s",
                    embedding_type,
                    token_pos,
                    n_layers_train,
                    n_layers_val,
                )
                continue

            if len(X_train) == 0 or len(X_val) == 0:
                logging.warning("No examples for %s token %s", embedding_type, token_pos)
                continue

            result[embedding_type][token_pos] = (X_train, X_val, y_train, y_val, n_layers_train)
            logging.info(
                "Loaded %s token %s: %s train, %s val, %s layers",
                embedding_type,
                token_pos,
                len(X_train),
                len(X_val),
                n_layers_train,
            )

    return result


def _get_run_base_dir(output_dir: Path) -> Path:
    """Return output_dir / mult_toks_all_layers / k where k is the first unused 1, 2, 3, ..."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mult_toks_dir = output_dir / "mult_toks_all_layers"
    mult_toks_dir.mkdir(parents=True, exist_ok=True)
    k = 1
    while (mult_toks_dir / str(k)).exists():
        k += 1
    run_base = mult_toks_dir / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _parse_token_dir_name(token_name: str):
    """Parse token directory name like tok_3_guess or tok_2_probability."""
    match = re.fullmatch(r"tok_(\d+)_(guess|probability|prob)", token_name)
    if match is None:
        return None, None
    token_pos = int(match.group(1))
    embedding_type = "probability" if match.group(2) == "prob" else match.group(2)
    return token_pos, embedding_type


def _token_sort_key(token_name: str):
    token_pos, embedding_type = _parse_token_dir_name(token_name)
    type_order = 0 if embedding_type == "guess" else 1
    if token_pos is None:
        return (2, float("inf"), token_name)
    return (type_order, token_pos, token_name)


def _style_for_token_order(order_idx: int, total_tokens: int):
    """Monotonic style progression to visualize token order."""
    if total_tokens <= 1:
        return {"linewidth": 2.2, "alpha": 1.0}
    frac = order_idx / float(total_tokens - 1)
    return {
        "linewidth": 1.5 + 1.7 * frac,
        "alpha": 0.7 + 0.3 * frac,
    }


def _sorted_token_items(all_token_metrics, embedding_type: str):
    items = []
    for token_name, metrics_dict in all_token_metrics.items():
        token_pos, token_type = _parse_token_dir_name(token_name)
        if token_type == embedding_type and token_pos is not None:
            items.append((token_name, metrics_dict, token_pos))
    return sorted(items, key=lambda x: x[2])


def _marker_for_token_pos(token_pos: int) -> str:
    """Return a deterministic marker by token index."""
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    return marker_cycle[token_pos % len(marker_cycle)]


def _plot_group_lines(ax, token_items, metric_idx: int, cmap_name: str, split_mode: str = "both"):
    if len(token_items) == 0:
        return
    color_map = plt.get_cmap(cmap_name)
    colors = color_map(np.linspace(0.45, 0.85, len(token_items)))
    for order_idx, (token_name, metrics_dict, token_pos) in enumerate(token_items):
        style = _style_for_token_order(order_idx, len(token_items))
        layer_numbers = metrics_dict["layers"]
        train_vals = metrics_dict["train"][metric_idx]
        val_vals = metrics_dict["val"][metric_idx]
        color = colors[order_idx]
        marker = _marker_for_token_pos(token_pos)

        if split_mode in ("both", "train"):
            ax.plot(
                layer_numbers,
                train_vals,
                label=f"{token_name} (Train)",
                marker=marker,
                markersize=4,
                color=color,
                linestyle="-",
                linewidth=style["linewidth"],
                alpha=style["alpha"],
            )
        if split_mode in ("both", "val"):
            ax.plot(
                layer_numbers,
                val_vals,
                label=f"{token_name} (Val)",
                marker=marker,
                markersize=4,
                color=color,
                linestyle="--",
                linewidth=style["linewidth"],
                alpha=max(0.6, style["alpha"] - 0.05),
            )


def _load_all_token_metrics_from_run_dir(run_base: Path):
    """Load saved per-layer metrics from tok_*/layer_*/verbalised_confidence_probe.pkl."""
    all_token_metrics = {}
    token_dirs = [d for d in run_base.iterdir() if d.is_dir() and d.name.startswith("tok_")]
    for token_dir in sorted(token_dirs, key=lambda p: _token_sort_key(p.name)):
        token_name = token_dir.name
        token_pos, embedding_type = _parse_token_dir_name(token_name)
        if token_pos is None:
            logging.warning("Skipping directory with unexpected token name: %s", token_dir)
            continue

        layer_pkls = sorted(
            token_dir.glob("layer_*/verbalised_confidence_probe.pkl"),
            key=lambda p: int(p.parent.name.split("_")[-1]) if p.parent.name.split("_")[-1].isdigit() else float("inf"),
        )
        if len(layer_pkls) == 0:
            logging.warning("No layer probe pickles found under %s", token_dir)
            continue

        layer_numbers = []
        train_mse, train_mae, train_r2 = [], [], []
        val_mse, val_mae, val_r2 = [], [], []

        for layer_pkl in layer_pkls:
            try:
                with open(layer_pkl, "rb") as f:
                    payload = pickle.load(f)
            except Exception as exc:
                logging.warning("Failed to read %s: %s", layer_pkl, exc)
                continue

            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            if not isinstance(metrics, dict):
                logging.warning("Missing metrics in %s; skipping", layer_pkl)
                continue

            train_metrics = metrics.get("train", {})
            val_metrics = metrics.get("val", {})
            try:
                layer_num = int(layer_pkl.parent.name.split("_")[-1])
                layer_numbers.append(layer_num)
                train_mse.append(float(train_metrics["mse"]))
                train_mae.append(float(train_metrics["mae"]))
                train_r2.append(float(train_metrics["r2"]))
                val_mse.append(float(val_metrics["mse"]))
                val_mae.append(float(val_metrics["mae"]))
                val_r2.append(float(val_metrics["r2"]))
            except (KeyError, TypeError, ValueError) as exc:
                logging.warning("Invalid metrics format in %s: %s", layer_pkl, exc)
                continue

        if len(layer_numbers) == 0:
            logging.warning("No valid metrics recovered for %s", token_dir)
            continue

        order = np.argsort(np.array(layer_numbers))
        layer_numbers = [layer_numbers[i] for i in order]
        train_mse = [train_mse[i] for i in order]
        train_mae = [train_mae[i] for i in order]
        train_r2 = [train_r2[i] for i in order]
        val_mse = [val_mse[i] for i in order]
        val_mae = [val_mae[i] for i in order]
        val_r2 = [val_r2[i] for i in order]

        all_token_metrics[token_name] = {
            "train": [train_mse, train_mae, train_r2],
            "val": [val_mse, val_mae, val_r2],
            "layers": layer_numbers,
            "token_pos": token_pos,
            "embedding_type": embedding_type,
        }
    return all_token_metrics


def _plot_metrics_by_layer(layer_numbers, train_metrics, val_metrics, metric_name, metric_label, output_dir: Path):
    """Plot one metric (train and val) vs layer number and save to output_dir."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, train_metrics, "o-", label="Train", markersize=4)
    ax.plot(layer_numbers, val_metrics, "s-", label="Validation", markersize=4)
    ax.set_xlabel("Layer number")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"{metric_name}_by_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", out_path)


def _plot_all_metrics_by_layer(output_dir: Path, layer_numbers, train_metrics_list, val_metrics_list, more_graphs: bool = False):
    """Create 3 graphs: MSE, MAE, R² vs layer number for a single token position."""
    metrics_config = [
        ("mse", "MSE", train_metrics_list[0], val_metrics_list[0]),
        ("mae", "MAE", train_metrics_list[1], val_metrics_list[1]),
        ("r2", "R²", train_metrics_list[2], val_metrics_list[2]),
    ]
    if not more_graphs:
        metrics_config = metrics_config[:1]
    for metric_name, metric_label, train_vals, val_vals in metrics_config:
        _plot_metrics_by_layer(layer_numbers, train_vals, val_vals, metric_name, metric_label, output_dir)


def _plot_metrics_all_tokens(run_base: Path, all_token_metrics, more_graphs: bool = False):
    """
    Create cross-token graphs for guess/probability/combined families.
    For each family and each metric, save:
    - both splits in one chart
    - train-only chart
    - val-only chart

    Token order is encoded via increasing line width/opacity by token index.
    """
    metrics_config = [
        ("mse", "MSE", 0),
        ("mae", "MAE", 1),
        ("r2", "R²", 2),
    ]
    if not more_graphs:
        metrics_config = metrics_config[:1]

    guess_token_items = _sorted_token_items(all_token_metrics, "guess")
    prob_token_items = _sorted_token_items(all_token_metrics, "probability")

    if len(guess_token_items) > 0:
        for metric_name, metric_label, metric_idx in metrics_config:
            fig, ax = plt.subplots(figsize=(10, 6))
            _plot_group_lines(ax, guess_token_items, metric_idx, cmap_name="Blues", split_mode="both")
            ax.set_xlabel("Layer number")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by layer - Guess tokens")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = run_base / f"{metric_name}_all_tokens_guess.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved %s", out_path)

            for split_mode, split_label in [("train", "Train"), ("val", "Val")]:
                fig, ax = plt.subplots(figsize=(10, 6))
                _plot_group_lines(ax, guess_token_items, metric_idx, cmap_name="Blues", split_mode=split_mode)
                ax.set_xlabel("Layer number")
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} by layer - Guess tokens ({split_label} only)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_path = run_base / f"{metric_name}_all_tokens_guess_{split_mode}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logging.info("Saved %s", out_path)

    if len(prob_token_items) > 0:
        for metric_name, metric_label, metric_idx in metrics_config:
            fig, ax = plt.subplots(figsize=(10, 6))
            _plot_group_lines(ax, prob_token_items, metric_idx, cmap_name="Oranges", split_mode="both")
            ax.set_xlabel("Layer number")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by layer - Probability tokens")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = run_base / f"{metric_name}_all_tokens_probability.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved %s", out_path)

            for split_mode, split_label in [("train", "Train"), ("val", "Val")]:
                fig, ax = plt.subplots(figsize=(10, 6))
                _plot_group_lines(ax, prob_token_items, metric_idx, cmap_name="Oranges", split_mode=split_mode)
                ax.set_xlabel("Layer number")
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} by layer - Probability tokens ({split_label} only)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_path = run_base / f"{metric_name}_all_tokens_probability_{split_mode}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logging.info("Saved %s", out_path)

    if len(guess_token_items) > 0 or len(prob_token_items) > 0:
        for metric_name, metric_label, metric_idx in metrics_config:
            fig, ax = plt.subplots(figsize=(11, 6))
            _plot_group_lines(ax, guess_token_items, metric_idx, cmap_name="Blues", split_mode="both")
            _plot_group_lines(ax, prob_token_items, metric_idx, cmap_name="Oranges", split_mode="both")
            ax.set_xlabel("Layer number")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by layer - Combined guess + probability tokens")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = run_base / f"{metric_name}_all_tokens_combined.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved %s", out_path)

            for split_mode, split_label in [("train", "Train"), ("val", "Val")]:
                fig, ax = plt.subplots(figsize=(11, 6))
                _plot_group_lines(ax, guess_token_items, metric_idx, cmap_name="Blues", split_mode=split_mode)
                _plot_group_lines(ax, prob_token_items, metric_idx, cmap_name="Oranges", split_mode=split_mode)
                ax.set_xlabel("Layer number")
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} by layer - Combined guess + probability tokens ({split_label} only)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_path = run_base / f"{metric_name}_all_tokens_combined_{split_mode}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logging.info("Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train verbalised-confidence probes for multiple token positions across all layers (HDF5 input)"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=False,
        help="Path to train HDF5 file (from process_generations_verbalised_embeddings_h5.py)",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=False,
        help="Path to validation HDF5 file (from process_generations_verbalised_embeddings_h5.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results); run dir is output_dir/mult_toks_all_layers/<run_id>/",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        choices=["ridge", "linear"],
        help="Type of regression model (default: ridge)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength for Ridge (default: 1.0)",
    )
    parser.add_argument("--plot", default=True, action="store_true", help="Save train/val regression plots per layer")
    parser.add_argument("--save_model", default=True, action="store_true", help="Save trained probes to pickle")
    parser.add_argument(
        "--more_graphs",
        action="store_true",
        default=False,
        help="If set, also generate MAE/R2 plots and per-layer train/val regression plots (default: only MSE plots).",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="If set, regenerate metric plots from an existing run_dir without retraining",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing run directory (e.g., results/mult_toks_all_layers/7) used with --plots_only",
    )
    args = parser.parse_args()

    if not args.plots_only and (not args.train_path or not args.val_path):
        parser.error("--train_path and --val_path are required unless --plots_only is enabled.")
    if args.plots_only and not args.run_dir:
        parser.error("--run_dir is required when --plots_only is enabled.")

    if args.plots_only:
        run_base = Path(args.run_dir)
        if not run_base.exists() or not run_base.is_dir():
            parser.error(f"--run_dir does not exist or is not a directory: {run_base}")
        output_log_path = run_base / "output_plots_only.log"
    else:
        run_base = _get_run_base_dir(Path(args.output_dir))
        output_log_path = run_base / "output.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Run directory: %s", run_base)
    if args.plots_only:
        logging.info("Plots-only mode enabled. Rebuilding plots from %s", run_base)
        all_token_metrics = _load_all_token_metrics_from_run_dir(run_base)
        if len(all_token_metrics) == 0:
            logging.error("No token metrics found in %s", run_base)
            return

        for token_name, metrics_dict in sorted(all_token_metrics.items(), key=lambda x: _token_sort_key(x[0])):
            token_dir = run_base / token_name
            logging.info("Regenerating per-token graphs for %s", token_name)
            _plot_all_metrics_by_layer(
                token_dir,
                metrics_dict["layers"],
                metrics_dict["train"],
                metrics_dict["val"],
                more_graphs=args.more_graphs,
            )

        logging.info("Regenerating cross-token graphs...")
        _plot_metrics_all_tokens(run_base, all_token_metrics, more_graphs=args.more_graphs)
        logging.info("Done. Regenerated plots in %s", run_base)
        return

    logging.info("Loading HDF5 data from %s and %s", args.train_path, args.val_path)

    data_dict = load_multitoken_verbalised_confidence_data(args.train_path, args.val_path)

    if len(data_dict["guess"]) == 0 and len(data_dict["probability"]) == 0:
        logging.error("No valid data loaded. Exiting.")
        return

    all_token_metrics = {}

    for embedding_type in ["guess", "probability"]:
        if embedding_type not in data_dict or len(data_dict[embedding_type]) == 0:
            logging.info("No data for %s embeddings, skipping...", embedding_type)
            continue

        logging.info("\n%s", "=" * 60)
        logging.info("Processing %s embeddings", embedding_type)
        logging.info("%s", "=" * 60)

        for token_pos in sorted(data_dict[embedding_type].keys()):
            X_train, X_val, y_train, y_val, n_layers = data_dict[embedding_type][token_pos]

            token_dir_name = f"tok_{token_pos}_{embedding_type}"
            token_dir = run_base / token_dir_name
            token_dir.mkdir(parents=True, exist_ok=True)

            logging.info("\nTraining probes for %s (%s layers)", token_dir_name, n_layers)

            layer_numbers = []
            train_mse, train_mae, train_r2 = [], [], []
            val_mse, val_mae, val_r2 = [], [], []

            for layer_idx in range(n_layers):
                layer_name = f"layer_{layer_idx + 1}"
                layer_dir = token_dir / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)

                X_train_l = X_train[:, layer_idx, :]
                X_val_l = X_val[:, layer_idx, :]

                model, metrics = train_verbalised_confidence_probe(
                    X_train_l,
                    y_train,
                    X_val_l,
                    y_val,
                    model_type=args.model_type,
                    alpha=args.alpha,
                    verbose=False,
                )

                layer_numbers.append(layer_idx + 1)
                train_mse.append(metrics["train"]["mse"])
                train_mae.append(metrics["train"]["mae"])
                train_r2.append(metrics["train"]["r2"])
                val_mse.append(metrics["val"]["mse"])
                val_mae.append(metrics["val"]["mae"])
                val_r2.append(metrics["val"]["r2"])

                if args.plot and args.more_graphs:
                    plot_results(y_train, model.predict(X_train_l), "train", str(layer_dir))
                    plot_results(y_val, model.predict(X_val_l), "val", str(layer_dir))

                if args.save_model:
                    model_path = layer_dir / "verbalised_confidence_probe.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(
                            {
                                "model": model,
                                "metrics": metrics,
                                "layer_idx": layer_idx,
                                "token_pos": token_pos,
                                "embedding_type": embedding_type,
                                "model_type": args.model_type,
                                "alpha": args.alpha if args.model_type == "ridge" else None,
                            },
                            f,
                        )

                    layer_args = argparse.Namespace(layer_idx=layer_idx, alpha=args.alpha)
                    write_config_txt(
                        layer_dir,
                        layer_args,
                        args.model_type,
                        n_train=len(X_train_l),
                        n_val=len(X_val_l),
                        metrics=metrics,
                        train_path=str(args.train_path),
                        val_path=str(args.val_path),
                    )
                    logging.debug("Saved model and config to %s", layer_dir)

            logging.info("Generating per-token graphs for %s...", token_dir_name)
            _plot_all_metrics_by_layer(
                token_dir,
                layer_numbers,
                [train_mse, train_mae, train_r2],
                [val_mse, val_mae, val_r2],
                more_graphs=args.more_graphs,
            )

            all_token_metrics[token_dir_name] = {
                "train": [train_mse, train_mae, train_r2],
                "val": [val_mse, val_mae, val_r2],
                "layers": layer_numbers,
                "token_pos": token_pos,
                "embedding_type": embedding_type,
            }

    if len(all_token_metrics) > 0:
        logging.info("\nGenerating cross-token graphs...")
        _plot_metrics_all_tokens(run_base, all_token_metrics, more_graphs=args.more_graphs)

    logging.info("\nDone. Trained probes saved in %s", run_base)


if __name__ == "__main__":
    main()
