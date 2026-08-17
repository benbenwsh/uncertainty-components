#!/usr/bin/env python3
"""Streaming probability-span mass-mean directions from verbalised-embeddings H5.

Avoids ``load_examples_h5`` (which materializes the entire file, including unused
attn/mlp components). Only reads ``verbalised_confidence`` and
``embeddings_probability[/res]`` token tensors, and accumulates running means.
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


def _check_probability_span_length(
    n_tok: int,
    expected_probability_tokens: int,
    *,
    allow_at_least_expected: bool,
    loc: str,
    ex_id: str,
) -> None:
    if allow_at_least_expected:
        if n_tok < expected_probability_tokens:
            raise ValueError(
                f"Example {ex_id} {loc} len={n_tok}; "
                f"expected at least {expected_probability_tokens}."
            )
        return
    if not _is_expected_or_plus_two(n_tok, expected_probability_tokens):
        raise ValueError(
            f"Example {ex_id} {loc} len={n_tok}; "
            f"expected {expected_probability_tokens} or {expected_probability_tokens + 2}."
        )


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


def _probability_token_group(
    r0: h5py.Group,
    *,
    new_h5_format: bool,
) -> Optional[h5py.Group]:
    """Return the list-group of probability token embeddings (res only when new format)."""
    field = r0.get("embeddings_probability")
    if field is None or not isinstance(field, h5py.Group):
        return None
    if not new_h5_format:
        return field
    res = field.get("res")
    if res is None or not isinstance(res, h5py.Group):
        return None
    return res


def _read_token_resid_post(
    token_ds: h5py.Dataset,
    *,
    n_layers: Optional[int],
) -> Tuple[np.ndarray, int]:
    """Load one token tensor and select resid_post rows (skip embedding index 0).

    Returns ``(selected, n_layers)`` with shape ``(n_layers, d_model)``.
    """
    if token_ds.ndim == 4:
        n_stream = int(token_ds.shape[0])
        d_model = int(token_ds.shape[-1])
        # Slice only resid_post sites: indices 1 .. n_stream-1
        if n_stream < 2:
            raise ValueError(f"Residual stream length {n_stream} is too short.")
        inferred_n_layers = n_stream - 1
        if n_layers is not None and inferred_n_layers != n_layers:
            raise ValueError(
                f"Layer count mismatch: expected {n_layers}, got {inferred_n_layers}."
            )
        # Read only needed rows: [1:, 0, -1, :]
        selected = np.asarray(token_ds[1:, 0, -1, :], dtype=np.float32)
        return selected, inferred_n_layers

    if token_ds.ndim == 2:
        n_stream = int(token_ds.shape[0])
        if n_stream < 2:
            raise ValueError(f"Residual stream length {n_stream} is too short.")
        inferred_n_layers = n_stream - 1
        if n_layers is not None and inferred_n_layers != n_layers:
            raise ValueError(
                f"Layer count mismatch: expected {n_layers}, got {inferred_n_layers}."
            )
        selected = np.asarray(token_ds[1:, :], dtype=np.float32)
        return selected, inferred_n_layers

    raise ValueError(f"Unsupported probability embedding rank {token_ds.ndim}; expected 2 or 4.")


def _read_probability_span(
    tok_group: h5py.Group,
    *,
    expected_probability_tokens: int,
    n_layers: Optional[int],
    ex_id: str,
    allow_at_least_expected: bool = False,
) -> Tuple[np.ndarray, int]:
    """Return ``(n_layers, T, d_model)`` for one example's probability span."""
    n_tok = _h5_list_length(tok_group)
    _check_probability_span_length(
        n_tok,
        expected_probability_tokens,
        allow_at_least_expected=allow_at_least_expected,
        loc="embeddings_probability",
        ex_id=ex_id,
    )
    use_n = expected_probability_tokens
    selected_tokens = []
    inferred_n_layers = n_layers
    d_model: Optional[int] = None
    for t in range(use_n):
        key = str(t)
        if key not in tok_group:
            raise ValueError(f"Example {ex_id} missing embeddings_probability token index {t}.")
        ds = tok_group[key]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"Example {ex_id} embeddings_probability/{t} is not a dataset.")
        row, inferred_n_layers = _read_token_resid_post(ds, n_layers=inferred_n_layers)
        if d_model is None:
            d_model = int(row.shape[-1])
        elif int(row.shape[-1]) != d_model:
            raise ValueError(
                f"Example {ex_id} token {t} hidden dim {row.shape[-1]} != {d_model}."
            )
        selected_tokens.append(row)
    stacked = np.stack(selected_tokens, axis=1).astype(np.float32, copy=False)
    assert inferred_n_layers is not None
    return stacked, inferred_n_layers


SUBBLOCK_COMPONENTS: Tuple[str, ...] = ("attn", "mlp")


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


def _read_token_subblock(
    token_ds: h5py.Dataset,
    *,
    n_layers: Optional[int],
) -> Tuple[np.ndarray, int]:
    """Load one attn/mlp token tensor. Uses all layer rows (no embedding skip).

    Returns ``(selected, n_layers)`` with shape ``(n_layers, d_model)``.
    """
    if token_ds.ndim == 4:
        inferred_n_layers = int(token_ds.shape[0])
        if inferred_n_layers < 1:
            raise ValueError("Subblock residual length is empty.")
        if n_layers is not None and inferred_n_layers != n_layers:
            raise ValueError(
                f"Layer count mismatch: expected {n_layers}, got {inferred_n_layers}."
            )
        selected = np.asarray(token_ds[:, 0, -1, :], dtype=np.float32)
        return selected, inferred_n_layers

    if token_ds.ndim == 2:
        inferred_n_layers = int(token_ds.shape[0])
        if inferred_n_layers < 1:
            raise ValueError("Subblock residual length is empty.")
        if n_layers is not None and inferred_n_layers != n_layers:
            raise ValueError(
                f"Layer count mismatch: expected {n_layers}, got {inferred_n_layers}."
            )
        selected = np.asarray(token_ds[:, :], dtype=np.float32)
        return selected, inferred_n_layers

    raise ValueError(f"Unsupported subblock embedding rank {token_ds.ndim}; expected 2 or 4.")


def _read_subblock_probability_span(
    tok_group: h5py.Group,
    *,
    expected_probability_tokens: int,
    n_layers: Optional[int],
    ex_id: str,
    component: str,
    allow_at_least_expected: bool = False,
) -> Tuple[np.ndarray, int]:
    """Return ``(n_layers, T, d_model)`` for one example's attn or mlp probability span."""
    n_tok = _h5_list_length(tok_group)
    _check_probability_span_length(
        n_tok,
        expected_probability_tokens,
        allow_at_least_expected=allow_at_least_expected,
        loc=f"embeddings_probability/{component}",
        ex_id=ex_id,
    )
    selected_tokens = []
    inferred_n_layers = n_layers
    d_model: Optional[int] = None
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
        row, inferred_n_layers = _read_token_subblock(ds, n_layers=inferred_n_layers)
        if d_model is None:
            d_model = int(row.shape[-1])
        elif int(row.shape[-1]) != d_model:
            raise ValueError(
                f"Example {ex_id} {component} token {t} hidden dim {row.shape[-1]} != {d_model}."
            )
        selected_tokens.append(row)
    stacked = np.stack(selected_tokens, axis=1).astype(np.float32, copy=False)
    assert inferred_n_layers is not None
    return stacked, inferred_n_layers


def compute_probability_mass_mean_direction_streaming(
    input_h5: Path | str,
    *,
    expected_probability_tokens: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    new_h5_format: bool = True,
    log_every: int = 50,
    allow_at_least_expected: bool = False,
) -> Tuple[np.ndarray, set[str], set[str], int, int]:
    """Stream H5 and compute probability-span mass-mean direction.

    Returns
    -------
    direction :
        ``(n_layers, T, d_model)`` float32, ``high_mean - low_mean``.
    low_ids, high_ids :
        Example id sets that met the confidence thresholds.
    n_layers :
        Number of TransformerLens residual-post layers (excludes embedding).
    h5_example_count :
        Total examples present under ``examples/``.
    """
    path = Path(input_h5)
    sum_low: Optional[np.ndarray] = None  # float64 accumulator
    sum_high: Optional[np.ndarray] = None
    count_low = 0
    count_high = 0
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    n_layers: Optional[int] = None
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
            "Streaming %d examples from %s (probability span only; new_h5_format=%s)",
            h5_example_count,
            path,
            new_h5_format,
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
                if not (is_low or is_high):
                    skipped_mid += 1
                    continue

                tok_group = _probability_token_group(r0, new_h5_format=new_h5_format)
                if tok_group is None:
                    raise ValueError(
                        f"Example {ex_id_str} missing embeddings_probability"
                        f"{'/res' if new_h5_format else ''}."
                    )
                span, n_layers = _read_probability_span(
                    tok_group,
                    expected_probability_tokens=expected_probability_tokens,
                    n_layers=n_layers,
                    ex_id=ex_id_str,
                    allow_at_least_expected=allow_at_least_expected,
                )
                # span: (n_layers, T, d)
                if is_low:
                    if sum_low is None:
                        sum_low = np.zeros(span.shape, dtype=np.float64)
                    sum_low += span.astype(np.float64, copy=False)
                    count_low += 1
                if is_high:
                    if sum_high is None:
                        sum_high = np.zeros(span.shape, dtype=np.float64)
                    sum_high += span.astype(np.float64, copy=False)
                    count_high += 1
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
                    count_low,
                    count_high,
                    skipped_mid,
                    skipped_bad,
                )

    if n_layers is None or sum_low is None or sum_high is None:
        raise ValueError(
            "Failed to build probability directions: need at least one low- and one "
            "high-confidence example with usable embeddings_probability."
        )
    if count_low == 0:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if count_high == 0:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")

    mean_low = (sum_low / float(count_low)).astype(np.float32)
    mean_high = (sum_high / float(count_high)).astype(np.float32)
    direction = (mean_high - mean_low).astype(np.float32)
    logging.info(
        "Direction shape=%s from low=%d high=%d (scanned=%d mid_skipped=%d bad_skipped=%d)",
        tuple(direction.shape),
        count_low,
        count_high,
        h5_example_count,
        skipped_mid,
        skipped_bad,
    )
    return direction, low_ids, high_ids, int(n_layers), h5_example_count


def compute_probability_subblock_mass_mean_directions_streaming(
    input_h5: Path | str,
    *,
    expected_probability_tokens: int,
    low_conf_threshold: float,
    high_conf_threshold: float,
    components: Sequence[str] = SUBBLOCK_COMPONENTS,
    log_every: int = 50,
    allow_at_least_expected: bool = False,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str], int, int]:
    """Stream H5 and compute attn/mlp probability-span mass-mean directions.

    Returns
    -------
    direction_by_component :
        ``{component: (n_layers, T, d_model)}`` float32, ``high_mean - low_mean``.
    low_ids, high_ids :
        Example id sets that met the confidence thresholds.
    n_layers :
        Number of attn/mlp layer rows (no embedding skip).
    h5_example_count :
        Total examples present under ``examples/``.
    """
    path = Path(input_h5)
    comps = list(components)
    if not comps:
        raise ValueError("components must be non-empty.")
    sum_low: Dict[str, Optional[np.ndarray]] = {c: None for c in comps}
    sum_high: Dict[str, Optional[np.ndarray]] = {c: None for c in comps}
    count_low = 0
    count_high = 0
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    n_layers: Optional[int] = None
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
            "Streaming %d examples from %s (probability %s mass-mean directions)",
            h5_example_count,
            path,
            "+".join(comps),
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
                if not (is_low or is_high):
                    skipped_mid += 1
                    continue

                spans: Dict[str, np.ndarray] = {}
                for component in comps:
                    tok_group = _probability_component_group(r0, component)
                    if tok_group is None:
                        raise ValueError(
                            f"Example {ex_id_str} missing embeddings_probability/{component}."
                        )
                    span, n_layers = _read_subblock_probability_span(
                        tok_group,
                        expected_probability_tokens=expected_probability_tokens,
                        n_layers=n_layers,
                        ex_id=ex_id_str,
                        component=component,
                        allow_at_least_expected=allow_at_least_expected,
                    )
                    spans[component] = span

                if is_low:
                    for component, span in spans.items():
                        if sum_low[component] is None:
                            sum_low[component] = np.zeros(span.shape, dtype=np.float64)
                        elif sum_low[component].shape != span.shape:
                            raise ValueError(
                                f"Example {ex_id_str} {component} span shape {span.shape} != "
                                f"{sum_low[component].shape}."
                            )
                        sum_low[component] += span.astype(np.float64, copy=False)
                    count_low += 1
                if is_high:
                    for component, span in spans.items():
                        if sum_high[component] is None:
                            sum_high[component] = np.zeros(span.shape, dtype=np.float64)
                        elif sum_high[component].shape != span.shape:
                            raise ValueError(
                                f"Example {ex_id_str} {component} span shape {span.shape} != "
                                f"{sum_high[component].shape}."
                            )
                        sum_high[component] += span.astype(np.float64, copy=False)
                    count_high += 1
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
                    count_low,
                    count_high,
                    skipped_mid,
                    skipped_bad,
                )

    if n_layers is None or count_low == 0 or count_high == 0:
        raise ValueError(
            "Failed to build subblock probability directions: need at least one low- and one "
            "high-confidence example with usable embeddings_probability attn/mlp."
        )
    if count_low == 0:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if count_high == 0:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")

    direction_by_component: Dict[str, np.ndarray] = {}
    for component in comps:
        acc_low = sum_low[component]
        acc_high = sum_high[component]
        if acc_low is None or acc_high is None:
            raise ValueError(f"No usable probability embeddings for component '{component}'.")
        mean_low = (acc_low / float(count_low)).astype(np.float32)
        mean_high = (acc_high / float(count_high)).astype(np.float32)
        direction_by_component[component] = (mean_high - mean_low).astype(np.float32)

    logging.info(
        "Subblock directions shape=%s from low=%d high=%d (scanned=%d mid_skipped=%d bad_skipped=%d)",
        tuple(direction_by_component[comps[0]].shape),
        count_low,
        count_high,
        h5_example_count,
        skipped_mid,
        skipped_bad,
    )
    return direction_by_component, low_ids, high_ids, int(n_layers), h5_example_count
