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


def _embeddings_h5_key(embedding_type: str) -> str:
    if embedding_type == "guess":
        return "embeddings_guess"
    if embedding_type == "probability":
        return "embeddings_probability"
    raise ValueError(f"Unknown embedding_type: {embedding_type}")


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
        v = ds[()]
        if isinstance(v, np.ndarray):
            return float(np.asarray(v).reshape(-1)[0])
        return float(v)
    except (TypeError, ValueError, OSError):
        return None


def _read_token_embedding_dataset(
    r0: h5py.Group, embedding_type: str, token_pos: int
) -> h5py.Dataset | None:
    key = _embeddings_h5_key(embedding_type)
    emb_grp = r0.get(key)
    if emb_grp is None or not isinstance(emb_grp, h5py.Group):
        return None
    ds = emb_grp.get(str(token_pos))
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    return ds


def _embedding_list_len(emb_grp: h5py.Group) -> int:
    return int(emb_grp.attrs.get("__len__", len(emb_grp.keys())))


def _try_verbalised_row_fast(
    example_group: h5py.Group,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    example_id: str,
) -> tuple[float, np.ndarray, int, int] | None:
    r0 = _h5_response0_group(example_group)
    if r0 is None:
        return None
    vc = _read_verbalised_confidence_scalar(r0)
    if vc is None:
        return None
    ds = _read_token_embedding_dataset(r0, embedding_type, token_pos)
    if ds is None or ds.ndim != 4:
        return None
    n_layers = int(ds.shape[0])
    hidden_dim = int(ds.shape[-1])
    if layer_idx < 0 or layer_idx >= n_layers:
        return None
    h = np.asarray(ds[layer_idx, 0, -1, :], dtype=np.float64)
    if h.shape[0] != hidden_dim:
        logging.warning(
            "Hidden dim mismatch for example %s: expected %s, got %s",
            example_id,
            hidden_dim,
            h.shape[0],
        )
        return None
    return (vc, h, n_layers, hidden_dim)


def _try_verbalised_row_slow(
    example_group: h5py.Group,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    example_id: str,
) -> tuple[float, np.ndarray, int, int] | None:
    try:
        example_data = _read_h5_node(example_group)
    except Exception as exc:
        logging.warning("Failed to read example %s: %s", example_id, exc)
        return None
    if not isinstance(example_data, dict):
        return None
    responses = example_data.get("responses", [])
    if not responses:
        return None
    response = responses[0]
    if not isinstance(response, dict):
        return None
    verbalised_confidence = response.get("verbalised_confidence")
    emb_list = response.get(f"embeddings_{embedding_type}")
    if verbalised_confidence is None or emb_list is None or len(emb_list) <= token_pos:
        return None
    emb_array = _tensor_to_numpy(emb_list[token_pos])
    layer_embs = emb_array[:, 0, -1, :]
    n_layers, hidden_dim = layer_embs.shape[0], layer_embs.shape[1]
    if layer_idx < 0 or layer_idx >= n_layers:
        return None
    h = np.asarray(layer_embs[layer_idx], dtype=np.float64)
    return (float(verbalised_confidence), h, n_layers, hidden_dim)


def _extract_verbalised_row_for_layer(
    example_group: h5py.Group,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    example_id: str,
) -> tuple[float, np.ndarray, int, int] | None:
    row = _try_verbalised_row_fast(
        example_group, embedding_type, token_pos, layer_idx, example_id
    )
    if row is not None:
        return row
    return _try_verbalised_row_slow(
        example_group, embedding_type, token_pos, layer_idx, example_id
    )


def _tensor_to_numpy(obj):
    """Convert tensor or array-like to numpy."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _scan_token_positions_one_file(path: str, embedding_type: str) -> set[int]:
    """Token indices that appear in at least one example with verbalised confidence and embedding list."""
    positions: set[int] = set()
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        for _example_id in h5_file["examples"].keys():
            example_group = h5_file["examples"][_example_id]
            r0 = _h5_response0_group(example_group)
            if r0 is None:
                continue
            if _read_verbalised_confidence_scalar(r0) is None:
                continue
            emb_grp = r0.get(_embeddings_h5_key(embedding_type))
            if emb_grp is None or not isinstance(emb_grp, h5py.Group):
                continue
            n_tok = _embedding_list_len(emb_grp)
            if n_tok <= 0:
                continue
            positions.update(range(n_tok))
    return positions


def scan_common_token_positions(train_path: str, val_path: str) -> dict[str, list[int]]:
    """Train ∩ val token indices per embedding kind (metadata only, no embedding tensor reads)."""
    out: dict[str, list[int]] = {}
    for embedding_type in ("guess", "probability"):
        common = _scan_token_positions_one_file(train_path, embedding_type) & _scan_token_positions_one_file(
            val_path, embedding_type
        )
        if not common:
            logging.warning("No common token positions found for %s embeddings", embedding_type)
            out[embedding_type] = []
        else:
            out[embedding_type] = sorted(common)
    return out


def _peek_n_layers_hidden_dim(path: str, embedding_type: str, token_pos: int) -> tuple[int, int] | None:
    """First 4D token embedding dataset shape along examples (shape read only)."""
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            return None
        for _example_id in h5_file["examples"].keys():
            example_group = h5_file["examples"][_example_id]
            r0 = _h5_response0_group(example_group)
            if r0 is None:
                continue
            ds = _read_token_embedding_dataset(r0, embedding_type, token_pos)
            if ds is None or ds.ndim != 4:
                continue
            return int(ds.shape[0]), int(ds.shape[-1])
    return None


def build_xy_verbalised_for_layer(
    h5_path: str,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Stream one HDF5 file and build X, y for a single layer (hidden states only).

    Returns:
        X: (n_samples, hidden_dim), y: (n_samples,), n_layers, hidden_dim.
        If no valid rows, returns (empty 2D array with second dim 0), empty y, 0, 0.
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    n_layers: int | None = None
    hidden_dim: int | None = None

    with h5py.File(h5_path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {h5_path}")
        examples_group = h5_file["examples"]
        for example_id in examples_group.keys():
            example_id_str = str(example_id)
            example_group = examples_group[example_id]
            row = _extract_verbalised_row_for_layer(
                example_group, embedding_type, token_pos, layer_idx, example_id_str
            )
            if row is None:
                continue
            y_val, h_vec, nl, hd = row
            if n_layers is None:
                n_layers, hidden_dim = nl, hd
            elif nl != n_layers or h_vec.shape[0] != hidden_dim:
                logging.warning(
                    "Shape mismatch for example %s, token %s, layer %s: expected nl=%s hd=%s, got nl=%s h.shape=%s",
                    example_id_str,
                    token_pos,
                    layer_idx,
                    n_layers,
                    hidden_dim,
                    nl,
                    h_vec.shape,
                )
                continue
            X_list.append(h_vec)
            y_list.append(y_val)

    if not X_list or n_layers is None or hidden_dim is None:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64), 0, 0

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float64)
    return X, y, n_layers, hidden_dim


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

    logging.info("Scanning HDF5 metadata (token positions) from %s and %s", args.train_path, args.val_path)
    common_by_kind = scan_common_token_positions(args.train_path, args.val_path)

    if not common_by_kind.get("guess") and not common_by_kind.get("probability"):
        logging.error("No valid data (no common token positions). Exiting.")
        return

    all_token_metrics = {}

    for embedding_type in ["guess", "probability"]:
        token_positions = common_by_kind.get(embedding_type) or []
        if not token_positions:
            logging.info("No data for %s embeddings, skipping...", embedding_type)
            continue

        logging.info("\n%s", "=" * 60)
        logging.info("Processing %s embeddings", embedding_type)
        logging.info("%s", "=" * 60)

        for token_pos in token_positions:
            peek_tr = _peek_n_layers_hidden_dim(args.train_path, embedding_type, token_pos)
            peek_va = _peek_n_layers_hidden_dim(args.val_path, embedding_type, token_pos)
            if peek_tr is None or peek_va is None:
                logging.warning(
                    "Could not read layer/hidden shape for %s token %s (train=%s val=%s); skipping token.",
                    embedding_type,
                    token_pos,
                    peek_tr,
                    peek_va,
                )
                continue
            n_layers_tr, hidden_dim_tr = peek_tr
            n_layers_va, hidden_dim_va = peek_va
            if n_layers_tr != n_layers_va or hidden_dim_tr != hidden_dim_va:
                logging.warning(
                    "Layer/hidden mismatch for %s token %s: train (%s, %s) vs val (%s, %s); skipping token.",
                    embedding_type,
                    token_pos,
                    n_layers_tr,
                    hidden_dim_tr,
                    n_layers_va,
                    hidden_dim_va,
                )
                continue
            n_layers, hidden_dim = n_layers_tr, hidden_dim_tr

            token_dir_name = f"tok_{token_pos}_{embedding_type}"
            token_dir = run_base / token_dir_name
            token_dir.mkdir(parents=True, exist_ok=True)

            logging.info("\nTraining probes for %s (%s layers, hidden_dim=%s)", token_dir_name, n_layers, hidden_dim)

            layer_numbers = []
            train_mse, train_mae, train_r2 = [], [], []
            val_mse, val_mae, val_r2 = [], [], []

            for layer_idx in range(n_layers):
                layer_name = f"layer_{layer_idx + 1}"
                layer_dir = token_dir / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)

                X_train, y_train, nl_tr, hd_tr = build_xy_verbalised_for_layer(
                    args.train_path, embedding_type, token_pos, layer_idx
                )
                X_val, y_val, nl_va, hd_va = build_xy_verbalised_for_layer(
                    args.val_path, embedding_type, token_pos, layer_idx
                )

                if nl_tr != n_layers or nl_va != n_layers or hd_tr != hidden_dim or hd_va != hidden_dim:
                    logging.warning(
                        "Streaming build shape mismatch for %s token %s layer %s: skipping layer.",
                        embedding_type,
                        token_pos,
                        layer_idx + 1,
                    )
                    continue

                if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                    logging.warning(
                        "No examples for %s token %s layer %s (train=%s val=%s); skipping layer.",
                        embedding_type,
                        token_pos,
                        layer_idx + 1,
                        X_train.shape[0],
                        X_val.shape[0],
                    )
                    continue

                X_train_l = X_train
                X_val_l = X_val

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

                del X_train, y_train, X_val, y_val, X_train_l, X_val_l, model

            if not layer_numbers:
                logging.warning("No layers trained for %s; skipping per-token and aggregate metrics for this token.", token_dir_name)
                continue

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
