#!/usr/bin/env python3
"""Streaming probability-span attn/mlp means from verbalised-embeddings H5.

Avoids ``load_examples_h5`` (which materializes the entire file). Only reads
``verbalised_confidence`` and ``embeddings_probability/{attn,mlp}`` token tensors,
and accumulates a running sum for the mean-source confidence group.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np


def _h5_list_length(group_obj: h5py.Group) -> int:
    return int(group_obj.attrs.get("__len__", len(group_obj.keys())))


def _is_expected_or_plus_two(actual_len: int, expected_len: int) -> bool:
    return actual_len in (expected_len, expected_len + 2)


def _read_verbalised_confidence(r0: h5py.Group) -> Optional[float]:
    ds = r0.get("verbalised_confidence")
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    try:
        value = ds[()]
        if isinstance(value, np.ndarray):
            return float(np.asarray(value).reshape(-1)[0])
        return float(value)
    except (TypeError, ValueError, OSError):
        return None


def _probability_component_group(
    r0: h5py.Group,
    component: str,
) -> Optional[h5py.Group]:
    field = r0.get("embeddings_probability")
    if field is None or not isinstance(field, h5py.Group):
        return None
    comp = field.get(component)
    if comp is None or not isinstance(comp, h5py.Group):
        return None
    return comp


def _as_layer_hidden(arr_like: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 4:
        return arr[:, 0, -1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected embedding tensor shape: {arr.shape}; expected 4D or 2D.")


def _read_component_probability_span(
    tok_group: h5py.Group,
    *,
    layer_idx: np.ndarray,
    expected_probability_tokens: int,
    ex_id: str,
    component: str,
) -> np.ndarray:
    """Return ``(n_selected_layers, T, d_model)`` for one example/component."""
    n_tok = _h5_list_length(tok_group)
    if not _is_expected_or_plus_two(n_tok, expected_probability_tokens):
        raise ValueError(
            f"Example {ex_id} embeddings_probability/{component} len={n_tok}; "
            f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
        )
    selected_tokens = []
    d_model: Optional[int] = None
    n_selected = int(layer_idx.shape[0])
    for t in range(expected_probability_tokens):
        key = str(t)
        if key not in tok_group:
            raise ValueError(
                f"Example {ex_id} missing embeddings_probability/{component}/{t}."
            )
        ds = tok_group[key]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/{component}/{t} is not a dataset."
            )
        hidden = _as_layer_hidden(np.asarray(ds[()], dtype=np.float32))
        if hidden.shape[0] <= int(np.max(layer_idx)):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/{component}/{t} has "
                f"{hidden.shape[0]} layers; need index {int(np.max(layer_idx))}."
            )
        row = hidden[layer_idx, :]
        if row.shape[0] != n_selected:
            raise ValueError(
                f"Example {ex_id} token {t} selected-layer count {row.shape[0]} != {n_selected}."
            )
        if d_model is None:
            d_model = int(row.shape[-1])
        elif int(row.shape[-1]) != d_model:
            raise ValueError(
                f"Example {ex_id} token {t} hidden dim {row.shape[-1]} != {d_model}."
            )
        selected_tokens.append(row)
    return np.stack(selected_tokens, axis=1).astype(np.float32, copy=False)


def compute_probability_subblock_group_means_streaming(
    input_h5: Path | str,
    *,
    ablate_layers: Sequence[int],
    ablate_subblocks: Sequence[str],
    expected_probability_tokens: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    log_every: int = 50,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], set[str], set[str], int]:
    """Stream H5 and compute probability-span attn/mlp means for one confidence group.

    Returns
    -------
    means_by_component :
        ``{component: {"probability": (n_selected_layers, T, d_model)}}``.
    low_ids, high_ids :
        Example id sets that met the confidence thresholds.
    h5_example_count :
        Total examples present under ``examples/``.
    """
    path = Path(input_h5)
    components = list(ablate_subblocks)
    if not components:
        raise ValueError("ablate_subblocks must be non-empty.")
    layer_idx = np.asarray(ablate_layers, dtype=np.int64)
    if layer_idx.ndim != 1 or layer_idx.size == 0:
        raise ValueError(f"ablate_layers must be a non-empty 1D sequence, got {ablate_layers!r}.")

    sums: Dict[str, Optional[np.ndarray]] = {c: None for c in components}
    count = 0
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    h5_example_count = 0
    used = 0
    skipped_mid = 0
    skipped_bad = 0

    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        ex_group = h5_file["examples"]
        example_ids = list(ex_group.keys())
        h5_example_count = len(example_ids)
        logging.info(
            "Streaming %d examples from %s (probability %s means only)",
            h5_example_count,
            path,
            "+".join(components),
        )

        for i, ex_id in enumerate(example_ids):
            ex_id_str = str(ex_id)
            try:
                ex_node = ex_group[ex_id]
                responses = ex_node.get("responses")
                if responses is None or not isinstance(responses, h5py.Group):
                    skipped_bad += 1
                    continue
                r0 = responses.get("0")
                if r0 is None or not isinstance(r0, h5py.Group):
                    skipped_bad += 1
                    continue

                conf = _read_verbalised_confidence(r0)
                if conf is None:
                    skipped_bad += 1
                    continue
                is_low = conf <= low_conf_threshold
                is_high = conf >= high_conf_threshold
                if is_low:
                    low_ids.add(ex_id_str)
                if is_high:
                    high_ids.add(ex_id_str)
                use_for_mean = is_low if mean_from_low_confidence else is_high
                if not use_for_mean:
                    if not (is_low or is_high):
                        skipped_mid += 1
                    continue

                for component in components:
                    tok_group = _probability_component_group(r0, component)
                    if tok_group is None:
                        raise ValueError(
                            f"Example {ex_id_str} missing embeddings_probability/{component}."
                        )
                    span = _read_component_probability_span(
                        tok_group,
                        layer_idx=layer_idx,
                        expected_probability_tokens=expected_probability_tokens,
                        ex_id=ex_id_str,
                        component=component,
                    )
                    if sums[component] is None:
                        sums[component] = np.zeros(span.shape, dtype=np.float64)
                    elif sums[component].shape != span.shape:
                        raise ValueError(
                            f"Example {ex_id_str} {component} span shape {span.shape} != "
                            f"{sums[component].shape}."
                        )
                    sums[component] += span.astype(np.float64, copy=False)
                count += 1
                used += 1
            except Exception as exc:
                skipped_bad += 1
                logging.warning("Skipping example %s: %s", ex_id_str, exc)
                continue

            if log_every > 0 and (i + 1) % log_every == 0:
                logging.info(
                    "Progress %d/%d examples (used=%d low=%d high=%d mid=%d bad=%d)",
                    i + 1,
                    h5_example_count,
                    used,
                    len(low_ids),
                    len(high_ids),
                    skipped_mid,
                    skipped_bad,
                )

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    if count == 0:
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(
            f"No {source_name} examples usable at threshold {operator} {threshold}."
        )

    means_by_component: Dict[str, Dict[str, np.ndarray]] = {}
    for component in components:
        acc = sums[component]
        if acc is None:
            raise ValueError(f"No usable probability embeddings for component '{component}'.")
        means_by_component[component] = {
            "probability": (acc / float(count)).astype(np.float32),
        }
    logging.info(
        "Probability means from %s count=%d shape=%s (scanned=%d mid_skipped=%d bad_skipped=%d)",
        source_name,
        count,
        tuple(means_by_component[components[0]]["probability"].shape),
        h5_example_count,
        skipped_mid,
        skipped_bad,
    )
    return means_by_component, low_ids, high_ids, h5_example_count
