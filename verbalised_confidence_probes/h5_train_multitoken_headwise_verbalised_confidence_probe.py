"""
Train headwise linear probes on concatenated probability-token concat embeddings.

For each (layer, head) pair, learns a probe on activations from
embeddings_probability/concat across all probability prefix tokens.
Outputs are written to: results/headwise/<run_id>/layer_<L>/head_<H>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mass_mean_probe.run_mass_mean_probe import parse_ablate_layers

try:
    from verbalised_confidence_probes.train_verbalised_confidence_probe import (
        train_verbalised_confidence_probe,
    )
except ImportError:
    from train_verbalised_confidence_probe import train_verbalised_confidence_probe


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


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


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


def _as_layer_hidden(arr_like: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 4:
        return arr[:, 0, -1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected embedding tensor shape: {arr.shape}; expected 4D or 2D.")


def _reshape_layer_hidden_to_heads(arr_like: np.ndarray, *, n_heads: int, d_head: int) -> np.ndarray:
    arr = _as_layer_hidden(arr_like)
    if arr.ndim != 2:
        raise ValueError(f"Expected [layers, d_model] after _as_layer_hidden, got shape {arr.shape}.")
    if arr.shape[-1] != n_heads * d_head:
        raise ValueError(
            f"Hidden dim {arr.shape[-1]} does not match n_heads*d_head={n_heads * d_head}."
        )
    return arr.reshape(arr.shape[0], n_heads, d_head)


def _is_expected_or_plus_two(actual_len: int, expected_len: int) -> bool:
    return actual_len in (expected_len, expected_len + 2)


def _validate_concat_field(resp0: dict, ex_id: str, field_name: str):
    field = resp0.get(field_name)
    if not isinstance(field, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be an object with key 'concat'."
        )
    if "concat" not in field:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} missing key 'concat'. "
            "Please regenerate processed H5 with --collect_concat_embeddings."
        )
    value = field.get("concat")
    if value is None:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name}/concat is null. "
            "Please regenerate processed H5 with --collect_concat_embeddings."
        )
    return value


def _read_concat_probability_list_fast(r0: h5py.Group) -> list[np.ndarray] | None:
    emb_field = r0.get("embeddings_probability")
    if not isinstance(emb_field, h5py.Group):
        return None
    concat_grp = emb_field.get("concat")
    if concat_grp is None or not isinstance(concat_grp, h5py.Group):
        return None
    n_tokens = int(concat_grp.attrs.get("__len__", len(concat_grp.keys())))
    out: list[np.ndarray] = []
    for token_idx in range(n_tokens):
        ds = concat_grp.get(str(token_idx))
        if ds is None or not isinstance(ds, h5py.Dataset):
            return None
        out.append(np.asarray(ds[()], dtype=np.float64))
    return out


def _read_concat_probability_list_slow(example_group: h5py.Group) -> list[np.ndarray] | None:
    try:
        example_data = _read_h5_node(example_group)
    except Exception:
        return None
    if not isinstance(example_data, dict):
        return None
    responses = example_data.get("responses", [])
    if not responses or not isinstance(responses[0], dict):
        return None
    resp0 = responses[0]
    try:
        emb_prob = _validate_concat_field(resp0, "unknown", "embeddings_probability")
    except ValueError:
        return None
    if not isinstance(emb_prob, list):
        return None
    return [_tensor_to_numpy(tok) for tok in emb_prob]


def _normalize_probability_concat_list(
    emb_prob: list[np.ndarray],
    *,
    expected_probability_tokens: int,
    example_id: str,
    path: str,
) -> list[np.ndarray] | None:
    if not _is_expected_or_plus_two(len(emb_prob), expected_probability_tokens):
        logging.warning(
            "%s: example %s embeddings_probability/concat len=%s; expected %s or %s. Skipping.",
            path,
            example_id,
            len(emb_prob),
            expected_probability_tokens,
            expected_probability_tokens + 2,
        )
        return None
    return emb_prob[:expected_probability_tokens]


def _precompute_prob_heads(
    emb_prob: list[np.ndarray],
    *,
    n_heads: int,
    d_head: int,
) -> np.ndarray:
    token_heads = [
        _reshape_layer_hidden_to_heads(tok_arr, n_heads=n_heads, d_head=d_head) for tok_arr in emb_prob
    ]
    return np.stack(token_heads, axis=0)


def _extract_example_row_fast(
    example_group: h5py.Group,
    *,
    n_heads: int,
    d_head: int,
    expected_probability_tokens: int,
    example_id: str,
    path: str,
) -> tuple[float, np.ndarray] | None:
    r0 = _h5_response0_group(example_group)
    if r0 is None:
        return None
    confidence = _read_verbalised_confidence_scalar(r0)
    if confidence is None:
        return None
    emb_prob = _read_concat_probability_list_fast(r0)
    if emb_prob is None:
        return None
    emb_prob = _normalize_probability_concat_list(
        emb_prob,
        expected_probability_tokens=expected_probability_tokens,
        example_id=example_id,
        path=path,
    )
    if emb_prob is None:
        return None
    try:
        prob_heads = _precompute_prob_heads(emb_prob, n_heads=n_heads, d_head=d_head)
    except ValueError as exc:
        logging.warning("%s: example %s invalid concat embeddings: %s", path, example_id, exc)
        return None
    return confidence, prob_heads


def _extract_example_row_slow(
    example_group: h5py.Group,
    *,
    n_heads: int,
    d_head: int,
    expected_probability_tokens: int,
    example_id: str,
    path: str,
) -> tuple[float, np.ndarray] | None:
    r0 = _h5_response0_group(example_group)
    confidence = _read_verbalised_confidence_scalar(r0) if r0 is not None else None
    if confidence is None:
        return None
    emb_prob = _read_concat_probability_list_slow(example_group)
    if emb_prob is None:
        return None
    emb_prob = _normalize_probability_concat_list(
        emb_prob,
        expected_probability_tokens=expected_probability_tokens,
        example_id=example_id,
        path=path,
    )
    if emb_prob is None:
        return None
    try:
        prob_heads = _precompute_prob_heads(emb_prob, n_heads=n_heads, d_head=d_head)
    except ValueError as exc:
        logging.warning("%s: example %s invalid concat embeddings: %s", path, example_id, exc)
        return None
    return confidence, prob_heads


def load_probability_concat_examples(
    h5_path: str,
    *,
    n_heads: int,
    d_head: int,
    expected_probability_tokens: int,
) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = []
    with h5py.File(h5_path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {h5_path}")
        examples_group = h5_file["examples"]
        for example_id in examples_group.keys():
            example_id_str = str(example_id)
            example_group = examples_group[example_id]
            row = _extract_example_row_fast(
                example_group,
                n_heads=n_heads,
                d_head=d_head,
                expected_probability_tokens=expected_probability_tokens,
                example_id=example_id_str,
                path=h5_path,
            )
            if row is None:
                row = _extract_example_row_slow(
                    example_group,
                    n_heads=n_heads,
                    d_head=d_head,
                    expected_probability_tokens=expected_probability_tokens,
                    example_id=example_id_str,
                    path=h5_path,
                )
            if row is not None:
                rows.append(row)
    return rows


def _validate_h5_concat_probability_lengths(
    path: str,
    *,
    expected_probability_tokens: int,
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
            emb_prob = _read_concat_probability_list_fast(r0)
            if emb_prob is None:
                emb_prob = _read_concat_probability_list_slow(example_group)
            if emb_prob is None:
                raise ValueError(
                    f"{path}: example {example_id_str} missing embeddings_probability/concat. "
                    "Please regenerate processed H5 with --collect_concat_embeddings."
                )
            if not _is_expected_or_plus_two(len(emb_prob), expected_probability_tokens):
                raise ValueError(
                    f"{path}: example {example_id_str} embeddings_probability/concat len={len(emb_prob)}; "
                    f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
                )
            checked_examples += 1
    logging.info("Validated concat probability token lengths for %s examples in %s", checked_examples, path)


def build_xy_for_layer_head(
    cached_rows: list[tuple[float, np.ndarray]],
    layer_idx: int,
    head_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for confidence, prob_heads in cached_rows:
        if layer_idx < 0 or layer_idx >= prob_heads.shape[1]:
            continue
        if head_idx < 0 or head_idx >= prob_heads.shape[2]:
            continue
        feat = prob_heads[:, layer_idx, head_idx, :].reshape(-1)
        x_list.append(feat)
        y_list.append(confidence)
    if not x_list:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=np.float64)


def _get_run_base_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = output_dir / "headwise"
    sub_dir.mkdir(parents=True, exist_ok=True)
    run_id = 1
    while (sub_dir / str(run_id)).exists():
        run_id += 1
    run_base = sub_dir / str(run_id)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def _layer_head_key(layer_idx: int, head_idx: int) -> str:
    return f"{layer_idx}.{head_idx}"


def _write_head_config_txt(
    head_dir: Path,
    args,
    *,
    layer_idx: int,
    head_idx: int,
    feature_dim: int,
    n_train: int,
    n_test: int,
    metrics: dict,
) -> None:
    lines = [
        "Verbalised confidence headwise probes - training configuration and results",
        "=" * 72,
        f"Probe id: {_layer_head_key(layer_idx, head_idx)}",
        f"Model type: {args.model_type}",
        f"Model name: {args.model_name}",
        f"Layer index (0-based): {layer_idx}",
        f"Head index (0-based): {head_idx}",
        f"Feature dim (probability tokens * d_head): {feature_dim}",
        f"Expected probability tokens: {args.expected_probability_tokens}",
        f"Alpha (regularization): {args.alpha}",
        f"Train path: {args.train_path}",
        f"Test path: {args.test_path}",
        "",
        "Data",
        "-" * 40,
        f"Samples (train / test): {n_train} / {n_test}",
        "",
        "Final metrics",
        "-" * 40,
        f"Train MSE:  {metrics['train']['mse']:.6f}",
        f"Test MSE:   {metrics['test']['mse']:.6f}",
        "",
        f"Trained at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    with open(head_dir / "config.txt", "w") as f:
        f.write("\n".join(lines))


def _build_mse_matrix(
    summary_heads: dict,
    *,
    layers: list[int],
    n_heads: int,
    split: str,
) -> np.ndarray:
    matrix = np.full((len(layers), n_heads), np.nan, dtype=np.float64)
    layer_to_row = {layer: idx for idx, layer in enumerate(layers)}
    for key, entry in summary_heads.items():
        layer_idx = int(entry["layer"])
        head_idx = int(entry["head"])
        if layer_idx not in layer_to_row or head_idx < 0 or head_idx >= n_heads:
            continue
        row = layer_to_row[layer_idx]
        matrix[row, head_idx] = float(entry[f"{split}_mse"])
    return matrix


def _write_mse_heatmap(
    path: Path,
    *,
    layers: list[int],
    n_heads: int,
    mse_matrix: np.ndarray,
    title: str,
) -> None:
    n_rows, n_cols = mse_matrix.shape
    finite = mse_matrix[np.isfinite(mse_matrix)]
    if finite.size > 0:
        min_mse = float(np.min(finite))
        max_mse = float(np.max(finite))
        if max_mse > min_mse:
            alpha_grid = np.clip((max_mse - mse_matrix) / (max_mse - min_mse), 0.0, 1.0)
        else:
            alpha_grid = np.ones_like(mse_matrix)
    else:
        alpha_grid = np.zeros_like(mse_matrix)
    alpha_grid = np.nan_to_num(alpha_grid, nan=0.0)

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.1
    rgba[:, :, 1] = 0.3
    rgba[:, :, 2] = 1.0
    rgba[:, :, 3] = alpha_grid

    fig, ax = plt.subplots(figsize=(max(10.0, 0.45 * n_cols), max(6.0, 0.45 * n_rows)))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    for row in range(n_rows):
        for col in range(n_cols):
            val = mse_matrix[row, col]
            text = "NA" if not np.isfinite(val) else f"{val:.4f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=7, color="black")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([str(head) for head in range(n_heads)])
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.35, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", path)


def _write_summary_and_heatmaps(
    run_base: Path,
    summary_heads: dict,
    *,
    layers: list[int],
    n_heads: int,
    expected_probability_tokens: int,
    model_name: str,
) -> None:
    summary = {
        "metadata": {
            "layers": layers,
            "n_heads": n_heads,
            "expected_probability_tokens": expected_probability_tokens,
            "model_name": model_name,
        },
        "heads": summary_heads,
    }
    summary_path = run_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved %s", summary_path)

    train_matrix = _build_mse_matrix(summary_heads, layers=layers, n_heads=n_heads, split="train")
    test_matrix = _build_mse_matrix(summary_heads, layers=layers, n_heads=n_heads, split="test")
    _write_mse_heatmap(
        run_base / "train_mse_heatmap.png",
        layers=layers,
        n_heads=n_heads,
        mse_matrix=train_matrix,
        title="Train MSE by layer and head (darker blue = lower MSE)",
    )
    _write_mse_heatmap(
        run_base / "test_mse_heatmap.png",
        layers=layers,
        n_heads=n_heads,
        mse_matrix=test_matrix,
        title="Test MSE by layer and head (darker blue = lower MSE)",
    )


def _load_model_head_metadata(model_name: str) -> tuple[int, int, int]:
    logging.info("Loading model config: %s", model_name)
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None)))
    n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None)))
    hidden_size = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None)))
    if n_layers is None or n_heads is None or hidden_size is None:
        raise ValueError(f"Could not resolve layer/head/hidden metadata from config for {model_name}.")
    if n_heads <= 0 or hidden_size % n_heads != 0:
        raise ValueError(
            f"Invalid head configuration for {model_name}: hidden_size={hidden_size}, n_heads={n_heads}."
        )
    d_head = hidden_size // n_heads
    if n_heads * d_head != hidden_size:
        raise ValueError(
            f"Model shape mismatch: n_heads*d_head={n_heads * d_head}, hidden_size={hidden_size}."
        )
    return n_layers, n_heads, d_head


def main():
    parser = argparse.ArgumentParser(
        description="Train headwise verbalised-confidence probes on probability concat embeddings (HDF5)."
    )
    parser.add_argument("--train_path", type=str, required=False, help="Path to train HDF5 file.")
    parser.add_argument("--test_path", type=str, required=False, help="Path to test HDF5 file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results); run dir is output_dir/headwise/<run_id>/",
    )
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--layers",
        type=str,
        default="10-16",
        help="Inclusive layer range '10-16' or comma list '10,12,14' (zero-indexed).",
    )
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--model_type", type=str, default="ridge", choices=["ridge", "linear"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--save_model", default=True, action="store_true", help="Save trained probes to pickle.")
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Regenerate heatmaps from an existing run_dir summary.json.",
    )
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
        summary_path = run_base / "summary.json"
        if not summary_path.exists():
            logging.error("No summary.json found in %s", run_base)
            return
        with open(summary_path) as f:
            summary = json.load(f)
        metadata = summary.get("metadata", {})
        layers = [int(x) for x in metadata.get("layers", [])]
        n_heads = int(metadata.get("n_heads", 0))
        if not layers or n_heads <= 0:
            logging.error("summary.json metadata missing layers or n_heads")
            return
        _write_summary_and_heatmaps(
            run_base,
            summary.get("heads", {}),
            layers=layers,
            n_heads=n_heads,
            expected_probability_tokens=int(metadata.get("expected_probability_tokens", 7)),
            model_name=str(metadata.get("model_name", args.model_name)),
        )
        logging.info("Done. Regenerated heatmaps in %s", run_base)
        return

    model_n_layers, n_heads, d_head = _load_model_head_metadata(args.model_name)
    probe_layers = parse_ablate_layers(args.layers, model_n_layers)
    feature_dim = args.expected_probability_tokens * d_head

    logging.info("model_n_layers=%s n_heads=%s d_head=%s feature_dim=%s", model_n_layers, n_heads, d_head, feature_dim)
    logging.info("probe_layers=%s", probe_layers)
    logging.info("expected_probability_tokens=%s", args.expected_probability_tokens)

    _validate_h5_concat_probability_lengths(
        args.train_path,
        expected_probability_tokens=args.expected_probability_tokens,
    )
    _validate_h5_concat_probability_lengths(
        args.test_path,
        expected_probability_tokens=args.expected_probability_tokens,
    )

    train_rows = load_probability_concat_examples(
        args.train_path,
        n_heads=n_heads,
        d_head=d_head,
        expected_probability_tokens=args.expected_probability_tokens,
    )
    test_rows = load_probability_concat_examples(
        args.test_path,
        n_heads=n_heads,
        d_head=d_head,
        expected_probability_tokens=args.expected_probability_tokens,
    )
    if not train_rows or not test_rows:
        raise ValueError("No training or test examples with verbalised confidence and concat probability embeddings.")

    logging.info("Loaded %s train examples and %s test examples", len(train_rows), len(test_rows))

    summary_heads: dict[str, dict] = {}

    for layer_idx in probe_layers:
        layer_dir = run_base / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for head_idx in range(n_heads):
            x_train, y_train = build_xy_for_layer_head(train_rows, layer_idx, head_idx)
            x_test, y_test = build_xy_for_layer_head(test_rows, layer_idx, head_idx)
            if x_train.shape[0] == 0 or x_test.shape[0] == 0:
                logging.warning(
                    "Skipping layer %s head %s due to missing train/test samples.",
                    layer_idx,
                    head_idx,
                )
                continue
            if x_train.shape[1] != feature_dim or x_test.shape[1] != feature_dim:
                logging.warning(
                    "Skipping layer %s head %s due to feature dim mismatch (train=%s test=%s expected=%s).",
                    layer_idx,
                    head_idx,
                    x_train.shape[1],
                    x_test.shape[1],
                    feature_dim,
                )
                continue

            model, metrics = train_verbalised_confidence_probe(
                x_train,
                y_train,
                x_test,
                y_test,
                model_type=args.model_type,
                alpha=args.alpha,
                verbose=False,
            )

            head_dir = layer_dir / f"head_{head_idx}"
            head_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": model,
                "metrics": metrics,
                "x_train_n": int(x_train.shape[0]),
                "x_test_n": int(x_test.shape[0]),
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "probe_id": _layer_head_key(layer_idx, head_idx),
                "feature_dim": feature_dim,
                "expected_probability_tokens": args.expected_probability_tokens,
                "model_type": args.model_type,
                "alpha": args.alpha if args.model_type == "ridge" else None,
                "model_name": args.model_name,
            }
            if args.save_model:
                with open(head_dir / "verbalised_confidence_probe.pkl", "wb") as f:
                    pickle.dump(payload, f)

            _write_head_config_txt(
                head_dir,
                args,
                layer_idx=layer_idx,
                head_idx=head_idx,
                feature_dim=feature_dim,
                n_train=int(x_train.shape[0]),
                n_test=int(x_test.shape[0]),
                metrics=metrics,
            )

            summary_heads[_layer_head_key(layer_idx, head_idx)] = {
                "layer": layer_idx,
                "head": head_idx,
                "train_mse": float(metrics["train"]["mse"]),
                "test_mse": float(metrics["test"]["mse"]),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
            }

    if not summary_heads:
        raise ValueError("No headwise probes were trained successfully.")

    _write_summary_and_heatmaps(
        run_base,
        summary_heads,
        layers=probe_layers,
        n_heads=n_heads,
        expected_probability_tokens=args.expected_probability_tokens,
        model_name=args.model_name,
    )
    logging.info("Done. Trained probes saved in %s", run_base)


if __name__ == "__main__":
    main()
