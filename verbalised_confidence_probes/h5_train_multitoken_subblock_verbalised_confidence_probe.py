"""
Train linear probes for multiple token positions across all layers at subblock level.

This HDF5 trainer learns separate probes for attention and MLP activations for each
token position and each layer to predict verbalised confidence.
Outputs are written to:
results/mult_toks_subblocks/<run_id>/tok_n_guess/ and tok_n_probability/ directories.
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

try:
    from verbalised_confidence_probes.train_verbalised_confidence_probe import (
        train_verbalised_confidence_probe,
    )
except ImportError:
    print("Importing from train_verbalised_confidence_probe.py this way")
    from train_verbalised_confidence_probe import train_verbalised_confidence_probe


_COMPONENTS = ("attn", "mlp")


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


def _embedding_list_len(emb_grp: h5py.Group) -> int:
    return int(emb_grp.attrs.get("__len__", len(emb_grp.keys())))


def _unwrap_component_list(emb_field, *, component: str, new_h5_format: bool, ex_id: str, field_name: str):
    if not new_h5_format:
        return emb_field
    if not isinstance(emb_field, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be dict-like when --new_h5_format is set."
        )
    if "res" not in emb_field:
        raise ValueError(f"Example {ex_id} responses/0/{field_name} missing 'res' key.")
    if component not in emb_field:
        raise ValueError(f"Example {ex_id} responses/0/{field_name} missing '{component}' key.")
    if emb_field[component] is None:
        raise ValueError(f"Example {ex_id} responses/0/{field_name}/{component} is null.")
    comp_list = emb_field[component]
    if not isinstance(comp_list, list):
        raise ValueError(f"Example {ex_id} responses/0/{field_name}/{component} must be a list.")
    return comp_list


def _embedding_token_list_group(
    r0: h5py.Group,
    embedding_type: str,
    component: str,
    *,
    new_h5_format: bool,
    strict: bool = False,
    example_id: str = "",
    path: str = "",
) -> h5py.Group | None:
    field_name = _embeddings_h5_key(embedding_type)
    emb_field = r0.get(field_name)
    if emb_field is None or not isinstance(emb_field, h5py.Group):
        if strict:
            raise ValueError(f"{path}: example {example_id} responses/0/{field_name} must be an HDF5 group.")
        return None
    if not new_h5_format:
        return emb_field
    if emb_field.get("res") is None:
        if strict:
            raise ValueError(f"{path}: example {example_id} responses/0/{field_name} missing 'res' group.")
        return None
    comp_grp = emb_field.get(component)
    if comp_grp is None or not isinstance(comp_grp, h5py.Group):
        if strict:
            raise ValueError(
                f"{path}: example {example_id} responses/0/{field_name} missing '{component}' group with token stacks."
            )
        return None
    return comp_grp


def _read_token_embedding_dataset(
    r0: h5py.Group, embedding_type: str, token_pos: int, component: str, *, new_h5_format: bool
) -> h5py.Dataset | None:
    emb_grp = _embedding_token_list_group(
        r0,
        embedding_type,
        component,
        new_h5_format=new_h5_format,
    )
    if emb_grp is None:
        return None
    ds = emb_grp.get(str(token_pos))
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    return ds


def _try_verbalised_row_fast(
    example_group: h5py.Group,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    example_id: str,
    component: str,
    *,
    new_h5_format: bool,
) -> tuple[float, np.ndarray, int, int] | None:
    r0 = _h5_response0_group(example_group)
    if r0 is None:
        return None
    confidence = _read_verbalised_confidence_scalar(r0)
    if confidence is None:
        return None
    ds = _read_token_embedding_dataset(
        r0, embedding_type, token_pos, component, new_h5_format=new_h5_format
    )
    if ds is None or ds.ndim != 4:
        return None
    n_layers = int(ds.shape[0])
    hidden_dim = int(ds.shape[-1])
    if layer_idx < 0 or layer_idx >= n_layers:
        return None
    h = np.asarray(ds[layer_idx, 0, -1, :], dtype=np.float64)
    if h.shape[0] != hidden_dim:
        logging.warning(
            "Hidden dim mismatch for example %s (%s): expected %s, got %s",
            example_id,
            component,
            hidden_dim,
            h.shape[0],
        )
        return None
    return (confidence, h, n_layers, hidden_dim)


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _try_verbalised_row_slow(
    example_group: h5py.Group,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    example_id: str,
    component: str,
    *,
    new_h5_format: bool,
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
    confidence = response.get("verbalised_confidence")
    field_name = f"embeddings_{embedding_type}"
    emb_list = _unwrap_component_list(
        response.get(field_name),
        component=component,
        new_h5_format=new_h5_format,
        ex_id=example_id,
        field_name=field_name,
    )
    if confidence is None or emb_list is None or len(emb_list) <= token_pos:
        return None
    emb_array = _tensor_to_numpy(emb_list[token_pos])
    layer_embs = emb_array[:, 0, -1, :]
    n_layers, hidden_dim = layer_embs.shape[0], layer_embs.shape[1]
    if layer_idx < 0 or layer_idx >= n_layers:
        return None
    h = np.asarray(layer_embs[layer_idx], dtype=np.float64)
    return (float(confidence), h, n_layers, hidden_dim)


def _extract_verbalised_row_for_layer(
    example_group: h5py.Group,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    example_id: str,
    component: str,
    *,
    new_h5_format: bool,
) -> tuple[float, np.ndarray, int, int] | None:
    row = _try_verbalised_row_fast(
        example_group,
        embedding_type,
        token_pos,
        layer_idx,
        example_id,
        component,
        new_h5_format=new_h5_format,
    )
    if row is not None:
        return row
    return _try_verbalised_row_slow(
        example_group,
        embedding_type,
        token_pos,
        layer_idx,
        example_id,
        component,
        new_h5_format=new_h5_format,
    )


def _validate_h5_embedding_lengths(
    path: str,
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    new_h5_format: bool,
) -> None:
    checked_examples = 0
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        for example_id in h5_file["examples"].keys():
            example_id_str = str(example_id)
            example_group = h5_file["examples"][example_id]
            r0 = _h5_response0_group(example_group)
            if r0 is None:
                continue
            if _read_verbalised_confidence_scalar(r0) is None:
                continue
            checked_examples += 1

            for embedding_type, expected_tokens in (
                ("guess", expected_guess_tokens),
                ("probability", expected_probability_tokens),
            ):
                for component in _COMPONENTS:
                    emb_grp = _embedding_token_list_group(
                        r0,
                        embedding_type,
                        component,
                        new_h5_format=new_h5_format,
                        strict=True,
                        example_id=example_id_str,
                        path=path,
                    )
                    n_tokens = _embedding_list_len(emb_grp)
                    if n_tokens != expected_tokens:
                        raise ValueError(
                            f"{path}: example {example_id_str} {_embeddings_h5_key(embedding_type)}/{component} "
                            f"len={n_tokens}; expected {expected_tokens}."
                        )
    logging.info("Validated token lengths for %s examples in %s", checked_examples, path)


def scan_common_token_positions(*, expected_guess_tokens: int, expected_probability_tokens: int) -> dict[str, list[int]]:
    return {
        "guess": list(range(expected_guess_tokens)),
        "probability": list(range(expected_probability_tokens)),
    }


def _peek_n_layers_hidden_dim(
    path: str, embedding_type: str, token_pos: int, component: str, *, new_h5_format: bool
) -> tuple[int, int] | None:
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            return None
        for example_id in h5_file["examples"].keys():
            example_group = h5_file["examples"][example_id]
            r0 = _h5_response0_group(example_group)
            if r0 is None:
                continue
            ds = _read_token_embedding_dataset(
                r0,
                embedding_type,
                token_pos,
                component,
                new_h5_format=new_h5_format,
            )
            if ds is None or ds.ndim != 4:
                continue
            return int(ds.shape[0]), int(ds.shape[-1])
    return None


def build_xy_verbalised_for_layer(
    h5_path: str,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    component: str,
    *,
    new_h5_format: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    x_list: list[np.ndarray] = []
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
                example_group,
                embedding_type,
                token_pos,
                layer_idx,
                example_id_str,
                component,
                new_h5_format=new_h5_format,
            )
            if row is None:
                continue
            y_val, h_vec, nl, hd = row
            if n_layers is None:
                n_layers, hidden_dim = nl, hd
            elif nl != n_layers or h_vec.shape[0] != hidden_dim:
                logging.warning(
                    "Shape mismatch for example %s, token %s, layer %s (%s): expected nl=%s hd=%s, got nl=%s h.shape=%s",
                    example_id_str,
                    token_pos,
                    layer_idx,
                    component,
                    n_layers,
                    hidden_dim,
                    nl,
                    h_vec.shape,
                )
                continue
            x_list.append(h_vec)
            y_list.append(y_val)

    if not x_list or n_layers is None or hidden_dim is None:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64), 0, 0

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.float64)
    return x, y, n_layers, hidden_dim


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = output_dir / "mult_toks_subblocks"
    sub_dir.mkdir(parents=True, exist_ok=True)
    k = 1
    while (sub_dir / str(k)).exists():
        k += 1
    run_base = sub_dir / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _write_layer_config_txt(
    layer_dir: Path,
    args,
    *,
    embedding_type: str,
    token_pos: int,
    layer_idx: int,
    n_train_attn: int,
    n_test_attn: int,
    n_train_mlp: int,
    n_test_mlp: int,
    metrics_attn: dict,
    metrics_mlp: dict,
) -> None:
    lines = [
        "Verbalised confidence subblock probes - training configuration and results",
        "=" * 72,
        f"Model type: {args.model_type}",
        f"Layer index (0-based): {layer_idx}",
        f"Embedding type: {embedding_type}",
        f"Token position: {token_pos}",
        f"Alpha (regularization): {args.alpha}",
        f"Train path: {args.train_path}",
        f"Test path: {args.test_path}",
        "",
        "Data",
        "-" * 40,
        f"ATTN samples (train / test): {n_train_attn} / {n_test_attn}",
        f"MLP samples (train / test): {n_train_mlp} / {n_test_mlp}",
        "",
        "Final metrics - attention subblock",
        "-" * 40,
        f"Train MSE:  {metrics_attn['train']['mse']:.6f}",
        f"Train MAE:  {metrics_attn['train']['mae']:.6f}",
        f"Train R2:   {metrics_attn['train']['r2']:.6f}",
        f"Test MSE:   {metrics_attn['test']['mse']:.6f}",
        f"Test MAE:   {metrics_attn['test']['mae']:.6f}",
        f"Test R2:    {metrics_attn['test']['r2']:.6f}",
        "",
        "Final metrics - MLP subblock",
        "-" * 40,
        f"Train MSE:  {metrics_mlp['train']['mse']:.6f}",
        f"Train MAE:  {metrics_mlp['train']['mae']:.6f}",
        f"Train R2:   {metrics_mlp['train']['r2']:.6f}",
        f"Test MSE:   {metrics_mlp['test']['mse']:.6f}",
        f"Test MAE:   {metrics_mlp['test']['mae']:.6f}",
        f"Test R2:    {metrics_mlp['test']['r2']:.6f}",
        "",
        f"Trained at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(layer_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def _parse_token_dir_name(token_name: str):
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
    if total_tokens <= 1:
        return {"linewidth": 2.2, "alpha": 1.0}
    frac = order_idx / float(total_tokens - 1)
    return {"linewidth": 1.5 + 1.7 * frac, "alpha": 0.7 + 0.3 * frac}


def _marker_for_token_pos(token_pos: int) -> str:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    return marker_cycle[token_pos % len(marker_cycle)]


def _plot_token_metrics_by_layer(token_dir: Path, token_pos: int, layer_numbers: list[int], metrics_dict, metric_name: str):
    metric_idx = {"mse": 0, "mae": 1, "r2": 2}[metric_name]
    metric_label = {"mse": "MSE", "mae": "MAE", "r2": "R2"}[metric_name]
    marker = _marker_for_token_pos(token_pos)

    fig, ax = plt.subplots(figsize=(10, 5))
    for component, component_color in (("attn", "red"), ("mlp", "blue")):
        train_vals = metrics_dict[component]["train"][metric_idx]
        test_vals = metrics_dict[component]["test"][metric_idx]
        ax.plot(
            layer_numbers,
            train_vals,
            label=f"{component.upper()} (Train)",
            color=component_color,
            linestyle="-",
            marker=marker,
            markersize=4,
            linewidth=2.1,
            alpha=0.9,
        )
        ax.plot(
            layer_numbers,
            test_vals,
            label=f"{component.upper()} (Test)",
            color=component_color,
            linestyle="--",
            marker=marker,
            markersize=4,
            linewidth=1.9,
            alpha=0.75,
        )

    ax.set_xlabel("Layer number")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by layer - token {token_pos}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = token_dir / f"{metric_name}_by_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", out_path)


def _load_all_token_metrics_from_run_dir(run_base: Path):
    all_token_metrics = {}
    token_dirs = [d for d in run_base.iterdir() if d.is_dir() and d.name.startswith("tok_")]
    for token_dir in sorted(token_dirs, key=lambda p: _token_sort_key(p.name)):
        token_name = token_dir.name
        token_pos, embedding_type = _parse_token_dir_name(token_name)
        if token_pos is None:
            logging.warning("Skipping directory with unexpected token name: %s", token_dir)
            continue

        layer_dirs = sorted(
            [d for d in token_dir.glob("layer_*") if d.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else float("inf"),
        )
        if not layer_dirs:
            continue

        layers = []
        out = {
            "attn": {"train": [[], [], []], "test": [[], [], []]},
            "mlp": {"train": [[], [], []], "test": [[], [], []]},
        }
        for layer_dir in layer_dirs:
            attn_pkl = layer_dir / "verbalised_confidence_probe_attn.pkl"
            mlp_pkl = layer_dir / "verbalised_confidence_probe_mlp.pkl"
            if not attn_pkl.exists() or not mlp_pkl.exists():
                continue
            try:
                with open(attn_pkl, "rb") as f:
                    attn_payload = pickle.load(f)
                with open(mlp_pkl, "rb") as f:
                    mlp_payload = pickle.load(f)
                layer_num = int(layer_dir.name.split("_")[-1])
                layers.append(layer_num)
                for comp, payload in (("attn", attn_payload), ("mlp", mlp_payload)):
                    metrics = payload["metrics"]
                    out[comp]["train"][0].append(float(metrics["train"]["mse"]))
                    out[comp]["train"][1].append(float(metrics["train"]["mae"]))
                    out[comp]["train"][2].append(float(metrics["train"]["r2"]))
                    out[comp]["test"][0].append(float(metrics["test"]["mse"]))
                    out[comp]["test"][1].append(float(metrics["test"]["mae"]))
                    out[comp]["test"][2].append(float(metrics["test"]["r2"]))
            except Exception as exc:
                logging.warning("Skipping invalid layer payload under %s: %s", layer_dir, exc)
                continue

        if not layers:
            continue
        order = np.argsort(np.array(layers))
        layers = [layers[i] for i in order]
        for comp in _COMPONENTS:
            for split in ("train", "test"):
                for idx in range(3):
                    arr = out[comp][split][idx]
                    out[comp][split][idx] = [arr[i] for i in order]

        all_token_metrics[token_name] = {
            "layers": layers,
            "token_pos": token_pos,
            "embedding_type": embedding_type,
            "attn": out["attn"],
            "mlp": out["mlp"],
        }
    return all_token_metrics


def _sorted_token_items(all_token_metrics, embedding_type: str):
    items = []
    for token_name, metrics_dict in all_token_metrics.items():
        token_pos, token_type = _parse_token_dir_name(token_name)
        if token_type == embedding_type and token_pos is not None:
            items.append((token_name, metrics_dict, token_pos))
    return sorted(items, key=lambda x: x[2])


def _plot_group_lines(ax, token_items, metric_idx: int, split: str):
    if len(token_items) == 0:
        return
    colors_attn = plt.get_cmap("Reds")(np.linspace(0.45, 0.85, len(token_items)))
    colors_mlp = plt.get_cmap("Blues")(np.linspace(0.45, 0.85, len(token_items)))
    for order_idx, (token_name, metrics_dict, token_pos) in enumerate(token_items):
        style = _style_for_token_order(order_idx, len(token_items))
        marker = _marker_for_token_pos(token_pos)
        layer_numbers = metrics_dict["layers"]
        ax.plot(
            layer_numbers,
            metrics_dict["attn"][split][metric_idx],
            label=f"{token_name} ATTN ({split.capitalize()})",
            marker=marker,
            markersize=4,
            color=colors_attn[order_idx],
            linestyle="-",
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
        ax.plot(
            layer_numbers,
            metrics_dict["mlp"][split][metric_idx],
            label=f"{token_name} MLP ({split.capitalize()})",
            marker=marker,
            markersize=4,
            color=colors_mlp[order_idx],
            linestyle="--",
            linewidth=style["linewidth"],
            alpha=max(0.6, style["alpha"] - 0.05),
        )


def _plot_metrics_all_tokens(run_base: Path, all_token_metrics):
    metric_names = [("mse", "MSE"), ("mae", "MAE"), ("r2", "R2")]
    guess_token_items = _sorted_token_items(all_token_metrics, "guess")
    prob_token_items = _sorted_token_items(all_token_metrics, "probability")

    for metric_idx, (metric_name, metric_label) in enumerate(metric_names):
        for split in ("train", "test"):
            if guess_token_items:
                fig, ax = plt.subplots(figsize=(12, 6))
                _plot_group_lines(ax, guess_token_items, metric_idx, split=split)
                ax.set_xlabel("Layer number")
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} by layer - Guess tokens ({split.capitalize()})")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_path = run_base / f"{metric_name}_all_tokens_guess_{split}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logging.info("Saved %s", out_path)

            if prob_token_items:
                fig, ax = plt.subplots(figsize=(12, 6))
                _plot_group_lines(ax, prob_token_items, metric_idx, split=split)
                ax.set_xlabel("Layer number")
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} by layer - Probability tokens ({split.capitalize()})")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_path = run_base / f"{metric_name}_all_tokens_probability_{split}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logging.info("Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train verbalised-confidence probes for multiple token positions across all layers (HDF5 subblocks)."
    )
    parser.add_argument("--train_path", type=str, required=False, help="Path to train HDF5 file.")
    parser.add_argument("--test_path", type=str, required=False, help="Path to test HDF5 file.")
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, expect embeddings_guess/embeddings_probability to include component sub-groups "
            "('attn'/'mlp') and a 'res' sub-group."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results); run dir is output_dir/mult_toks_subblocks/<run_id>/",
    )
    parser.add_argument("--model_type", type=str, default="ridge", choices=["ridge", "linear"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--more_graphs", action="store_true", default=False, help="If set, also generate MAE/R2 plots.")
    parser.add_argument("--save_model", default=True, action="store_true", help="Save trained probes to pickle.")
    parser.add_argument("--plots_only", action="store_true", help="Regenerate metric plots from an existing run_dir.")
    parser.add_argument("--run_dir", type=str, default=None, help="Existing run directory used with --plots_only.")
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
    else:
        run_base = _get_run_base_dir(Path(args.output_dir))
        output_log_path = run_base / "output.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(output_log_path, mode="w"), logging.StreamHandler()],
    )

    logging.info("Run directory: %s", run_base)
    if args.plots_only:
        all_token_metrics = _load_all_token_metrics_from_run_dir(run_base)
        if not all_token_metrics:
            logging.error("No token metrics found in %s", run_base)
            return
        for token_name, metrics_dict in sorted(all_token_metrics.items(), key=lambda x: _token_sort_key(x[0])):
            token_dir = run_base / token_name
            token_pos = metrics_dict["token_pos"]
            metric_names = [("mse",), ("mae",), ("r2",)] if args.more_graphs else [("mse",)]
            for metric_tuple in metric_names:
                metric_name = metric_tuple[0]
                _plot_token_metrics_by_layer(token_dir, token_pos, metrics_dict["layers"], metrics_dict, metric_name)
        _plot_metrics_all_tokens(run_base, all_token_metrics)
        logging.info("Done. Regenerated plots in %s", run_base)
        return

    logging.info("expected_guess_tokens=%s", args.expected_guess_tokens)
    logging.info("expected_probability_tokens=%s", args.expected_probability_tokens)
    logging.info("new_h5_format=%s", args.new_h5_format)
    _validate_h5_embedding_lengths(
        args.train_path,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
        new_h5_format=args.new_h5_format,
    )
    _validate_h5_embedding_lengths(
        args.test_path,
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
        new_h5_format=args.new_h5_format,
    )
    token_positions_by_kind = scan_common_token_positions(
        expected_guess_tokens=args.expected_guess_tokens,
        expected_probability_tokens=args.expected_probability_tokens,
    )

    all_token_metrics = {}
    for embedding_type in ("guess", "probability"):
        token_positions = token_positions_by_kind.get(embedding_type) or []
        if not token_positions:
            continue
        for token_pos in token_positions:
            peek = {}
            for component in _COMPONENTS:
                train_shape = _peek_n_layers_hidden_dim(
                    args.train_path,
                    embedding_type,
                    token_pos,
                    component,
                    new_h5_format=args.new_h5_format,
                )
                test_shape = _peek_n_layers_hidden_dim(
                    args.test_path,
                    embedding_type,
                    token_pos,
                    component,
                    new_h5_format=args.new_h5_format,
                )
                if train_shape is None or test_shape is None or train_shape != test_shape:
                    logging.warning(
                        "Shape unavailable/mismatch for %s token %s component %s (train=%s test=%s); skipping token.",
                        embedding_type,
                        token_pos,
                        component,
                        train_shape,
                        test_shape,
                    )
                    peek = {}
                    break
                peek[component] = train_shape
            if not peek:
                continue

            n_layers = peek["attn"][0]
            token_dir_name = f"tok_{token_pos}_{embedding_type}"
            token_dir = run_base / token_dir_name
            token_dir.mkdir(parents=True, exist_ok=True)

            metrics_by_component = {
                "attn": {"train": [[], [], []], "test": [[], [], []]},
                "mlp": {"train": [[], [], []], "test": [[], [], []]},
            }
            layer_numbers = []

            for layer_idx in range(n_layers):
                layer_dir = token_dir / f"layer_{layer_idx + 1}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                layer_payloads = {}
                skip_layer = False
                for component in _COMPONENTS:
                    x_train, y_train, nl_tr, hd_tr = build_xy_verbalised_for_layer(
                        args.train_path,
                        embedding_type,
                        token_pos,
                        layer_idx,
                        component,
                        new_h5_format=args.new_h5_format,
                    )
                    x_test, y_test, nl_te, hd_te = build_xy_verbalised_for_layer(
                        args.test_path,
                        embedding_type,
                        token_pos,
                        layer_idx,
                        component,
                        new_h5_format=args.new_h5_format,
                    )
                    if (
                        nl_tr != n_layers
                        or nl_te != n_layers
                        or hd_tr != peek[component][1]
                        or hd_te != peek[component][1]
                        or x_train.shape[0] == 0
                        or x_test.shape[0] == 0
                    ):
                        logging.warning(
                            "Skipping layer %s for %s token %s component %s due to missing/shape mismatch.",
                            layer_idx + 1,
                            embedding_type,
                            token_pos,
                            component,
                        )
                        skip_layer = True
                        break
                    model, metrics = train_verbalised_confidence_probe(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        model_type=args.model_type,
                        alpha=args.alpha,
                        verbose=False,
                    )
                    layer_payloads[component] = {
                        "model": model,
                        "metrics": metrics,
                        "x_train_n": int(x_train.shape[0]),
                        "x_test_n": int(x_test.shape[0]),
                        "layer_idx": layer_idx,
                        "token_pos": token_pos,
                        "embedding_type": embedding_type,
                        "component": component,
                        "model_type": args.model_type,
                        "alpha": args.alpha if args.model_type == "ridge" else None,
                    }
                if skip_layer or len(layer_payloads) != 2:
                    continue

                layer_numbers.append(layer_idx + 1)
                for component in _COMPONENTS:
                    payload = layer_payloads[component]
                    metrics = payload["metrics"]
                    metrics_by_component[component]["train"][0].append(metrics["train"]["mse"])
                    metrics_by_component[component]["train"][1].append(metrics["train"]["mae"])
                    metrics_by_component[component]["train"][2].append(metrics["train"]["r2"])
                    metrics_by_component[component]["test"][0].append(metrics["test"]["mse"])
                    metrics_by_component[component]["test"][1].append(metrics["test"]["mae"])
                    metrics_by_component[component]["test"][2].append(metrics["test"]["r2"])
                    if args.save_model:
                        out_path = layer_dir / f"verbalised_confidence_probe_{component}.pkl"
                        with open(out_path, "wb") as f:
                            pickle.dump(payload, f)

                _write_layer_config_txt(
                    layer_dir,
                    args,
                    embedding_type=embedding_type,
                    token_pos=token_pos,
                    layer_idx=layer_idx,
                    n_train_attn=layer_payloads["attn"]["x_train_n"],
                    n_test_attn=layer_payloads["attn"]["x_test_n"],
                    n_train_mlp=layer_payloads["mlp"]["x_train_n"],
                    n_test_mlp=layer_payloads["mlp"]["x_test_n"],
                    metrics_attn=layer_payloads["attn"]["metrics"],
                    metrics_mlp=layer_payloads["mlp"]["metrics"],
                )

            if not layer_numbers:
                continue

            metric_names = ["mse", "mae", "r2"] if args.more_graphs else ["mse"]
            for metric_name in metric_names:
                _plot_token_metrics_by_layer(token_dir, token_pos, layer_numbers, metrics_by_component, metric_name)

            all_token_metrics[token_dir_name] = {
                "layers": layer_numbers,
                "token_pos": token_pos,
                "embedding_type": embedding_type,
                "attn": metrics_by_component["attn"],
                "mlp": metrics_by_component["mlp"],
            }

    if all_token_metrics:
        _plot_metrics_all_tokens(run_base, all_token_metrics)
    logging.info("Done. Trained probes saved in %s", run_base)


if __name__ == "__main__":
    main()
