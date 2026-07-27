"""
Process HDF5 generations into verbalised-confidence embedding HDF5.

Input:
  HDF5 produced by generate_answers_with_confidence_h5.py, with examples under
  /examples/<example_id>. This script processes only `most_likely_answer` per
  example.

Output:
  HDF5 + JSON with one processed response per example containing:
    - response (original model completion string)
    - verbalised_confidence
    - embeddings_mean_prompt (default mode)
    - embeddings_guess
    - embeddings_mean_sem_answer (default mode)
    - embeddings_probability
    - embeddings_mean_prob_val
    - embeddings_prompt_k_tokens (tokenwise K mode)
    - embeddings_sem_answer_k_tokens (tokenwise K mode)

  Also writes a companion *_summary.json with the same structure but embedding
  fields reduced to shape metadata only (no value previews).

  With --balance (default), also writes a balanced copy under a `balanced/`
  subdirectory, capping each verbalised_confidence bin at --balance_cap.
  Use --balance_from_h5 to balance an existing verbalised embeddings HDF5
  without re-running processing.
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from process_generations_tok_bef_gen import (
    parse_probability_from_response,
)

_EMBEDDING_KEYS = frozenset(
    {
        "embeddings_mean_prompt",
        "embeddings_guess",
        "embeddings_mean_sem_answer",
        "embeddings_probability",
        "embeddings_mean_prob_val",
        "embeddings_prompt_k_tokens",
        "embeddings_sem_answer_k_tokens",
    }
)

# Gemma-3 token alternatives (fixed length; each inner list = allowed tokens at that position)
GEMMA_GUESS_PREFIX_TOKENS = [
    ["\n", "\n\n"],
    ["Guess"],
    [":"],
]
GEMMA_PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    ["Probability", " Probability"],
    [":"],
    [" "],
]

# Qwen2.5 token alternatives (from ans_gen/generated_answers/3_32B_200 decoded tokens)
QWEN_GUESS_PREFIX_TOKENS = [
    [" Guess", "Guess"],
    [":"],
]
QWEN_PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    [" Probability"],
    [":"],
    [" "],
]

# Mistral-7B-Instruct-v0.1 (from ans_gen/generated_answers/1_svamp_mistral)
MISTRAL_GUESS_PREFIX_TOKENS = [
    ["\n"],
    ["\n"],
    ["Gu"],
    ["ess"],
    [":"],
]
MISTRAL_PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    ["Pro"],
    ["b"],
    ["ability"],
    [":"],
    [""],  # space before number decodes as empty string
]

# Active tables; set via configure_prefix_tokens_for_model(model_name).
GUESS_PREFIX_TOKENS: list[list[str]] = GEMMA_GUESS_PREFIX_TOKENS
PROBABILITY_PREFIX_TOKENS: list[list[str]] = GEMMA_PROBABILITY_PREFIX_TOKENS


def configure_prefix_tokens_for_model(model_name: str) -> None:
    """Set GUESS/PROBABILITY_PREFIX_TOKENS from exact model_name (case-sensitive)."""
    global GUESS_PREFIX_TOKENS, PROBABILITY_PREFIX_TOKENS
    if model_name == "google/gemma-3-12b-it":
        GUESS_PREFIX_TOKENS = GEMMA_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = GEMMA_PROBABILITY_PREFIX_TOKENS
    elif model_name == "Qwen/Qwen2.5-32B-Instruct":
        GUESS_PREFIX_TOKENS = QWEN_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = QWEN_PROBABILITY_PREFIX_TOKENS
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        GUESS_PREFIX_TOKENS = MISTRAL_GUESS_PREFIX_TOKENS
        PROBABILITY_PREFIX_TOKENS = MISTRAL_PROBABILITY_PREFIX_TOKENS
    else:
        raise ValueError(
            f"Unsupported model_name for Guess/Probability token parsing: {model_name!r}. "
            "Supported: 'google/gemma-3-12b-it', 'Qwen/Qwen2.5-32B-Instruct', "
            "'mistralai/Mistral-7B-Instruct-v0.1'."
        )


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _token_index_for_char_offset(decoded_tokens: list, char_offset: int) -> int:
    cumulative = 0
    for i, tok in enumerate(decoded_tokens):
        cumulative += len(tok)
        if cumulative > char_offset:
            return i
    return max(0, len(decoded_tokens) - 1)


def _match_token_prefix(
    decoded_tokens: list,
    prefix_tokens: list[list[str]],
    *,
    start: int = 0,
) -> int | None:
    """Return start index of first match of prefix_tokens (2D alts) at/after `start`, else None."""
    prefix_len = len(prefix_tokens)
    if prefix_len == 0:
        return None
    for i in range(start, len(decoded_tokens) - prefix_len + 1):
        if all(decoded_tokens[i + j] in prefix_tokens[j] for j in range(prefix_len)):
            return i
    return None


def parse_guess_and_probability_indices(decoded_tokens: list) -> tuple[int, int, int] | None:
    """
    Returns:
      - last_guess_token_index: first token index of semantic answer
      - first_prob_token_index: token index at "\\n" before "Probability:" (first occurrence)
      - end_prob_token_index: first token index of probability value
    """
    guess_start = _match_token_prefix(decoded_tokens, GUESS_PREFIX_TOKENS, start=0)
    if guess_start is None:
        return None

    last_guess_token_index = guess_start + len(GUESS_PREFIX_TOKENS)

    prob_start = _match_token_prefix(
        decoded_tokens, PROBABILITY_PREFIX_TOKENS, start=last_guess_token_index
    )
    if prob_start is None:
        return None

    first_prob_token_index = prob_start
    end_prob_token_index = prob_start + len(PROBABILITY_PREFIX_TOKENS)

    if (
        last_guess_token_index <= 0
        or last_guess_token_index >= len(decoded_tokens)
        or end_prob_token_index >= len(decoded_tokens)  # ensure a number token after the whitespace
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None

    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def _first_probability_value_char_span(full_str: str) -> tuple[int, int] | None:
    """Return [start,end] char span for the first numeric value after ``Probability:``."""
    if not full_str:
        return None
    matches = list(
        re.finditer(r"probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?", full_str, re.IGNORECASE)
    )
    if not matches:
        matches = list(re.finditer(r"probability\s*:\s*(\d+(?:[.,]\d+)?)", full_str, re.IGNORECASE))
    if not matches:
        return None
    match = matches[0]
    return match.start(1), match.end(1) - 1 # inclusive


def _probability_value_token_span(decoded_tokens: list, full_str: str) -> tuple[int, int] | None:
    """Map probability numeric char span to [start,end] token indices."""
    span = _first_probability_value_char_span(full_str)
    if span is None:
        return None
    char_start, char_end = span
    token_start = _token_index_for_char_offset(decoded_tokens, char_start)
    token_end = _token_index_for_char_offset(decoded_tokens, char_end)
    if token_start > token_end:
        return None
    return token_start, token_end


def _mean_across_tokens(token_embeddings: list[np.ndarray]) -> np.ndarray:
    """Average token embeddings across token dimension.

    Input is conceptually (n_tokens, *embedding_shape), output is (*embedding_shape).
    """
    if not token_embeddings:
        raise ValueError("Cannot mean over empty token embedding list")
    stacked = np.stack([_tensor_to_numpy(e) for e in token_embeddings], axis=0)
    return np.mean(stacked, axis=0)


def _last_token_position(arr: np.ndarray) -> np.ndarray:
    """Return final sequence position as singleton seq length."""
    arr = _tensor_to_numpy(arr)
    if arr.ndim < 4 or arr.shape[2] < 1:
        raise ValueError(f"Expected embedding rank>=4 with seq axis, got shape {arr.shape}")
    return arr[:, :, -1:, :]


def _compute_embeddings_from_source(
    *,
    source_embeddings,
    decoded_tokens: list,
    last_guess_token_index: int,
    first_prob_token_index: int,
    end_prob_token_index: int,
    prob_value_start_token_index: int,
    prob_value_end_token_index: int,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    example_id,
    source_label: str,
    extend_probability_span: bool = False,
    attention_score_tokenwise_k_mode: bool = False,
) -> dict:
    if source_embeddings is None:
        raise ValueError(f"{source_label}: missing source embeddings")
    if len(source_embeddings) != len(decoded_tokens):
        raise ValueError(
            f"{source_label}: len(source_embeddings)={len(source_embeddings)} != len(decoded_tokens)={len(decoded_tokens)}"
        )

    # Prompt mean: use source_embeddings[0] (embedding row for the first generation step).
    # That tensor is [layers, batch, prompt_seq_len, hidden_dim]; shape[2] is the prompt
    # sequence length. Mean over axis 2 after dropping the final
    # position (prompt_only_embeddings = ...[:, :, :-1, :]).
    prompt_plus_first_gen = _tensor_to_numpy(source_embeddings[0])
    if prompt_plus_first_gen.ndim != 4:
        raise ValueError(
            f"{source_label}: prompt_plus_first_gen has wrong shape: {prompt_plus_first_gen.shape}"
        )
    if prompt_plus_first_gen.shape[2] <= 1:
        raise ValueError(f"{source_label}: prompt has zero tokens")
    prompt_only_embeddings = prompt_plus_first_gen[:, :, :-1, :]
    embeddings_mean_prompt = np.mean(prompt_only_embeddings, axis=2, keepdims=True)
    embeddings_prompt_k_tokens = [
        prompt_only_embeddings[:, :, i : i + 1, :] for i in range(prompt_only_embeddings.shape[2])
    ]

    # Guess token embeddings:
    # - start from the last sequence position of source_embeddings[0] (index shape[2]-1
    #   along the prompt sequence axis),
    # - add remaining guess-token position embeddings.
    guess_token_indices = range(0, last_guess_token_index)
    embeddings_guess = []
    for token_idx in guess_token_indices:
        emb = _tensor_to_numpy(source_embeddings[token_idx])
        if emb.ndim != 4:
            raise ValueError(
                f"{source_label}: source_embeddings[{token_idx}] has wrong rank: {emb.shape}"
            )
        if token_idx == 0:
            if emb.shape[2] <= 1:
                raise ValueError(
                    f"{source_label}: source_embeddings[0] has seq_len <= 1: {emb.shape}"
                )
            embeddings_guess.append(_last_token_position(emb))
            continue

        if emb.shape[2] != 1:
            raise ValueError(
                f"{source_label}: source_embeddings[{token_idx}] expected seq_len=1, got shape {emb.shape}"
            )
        embeddings_guess.append(emb)

    if len(embeddings_guess) == 0:
        raise ValueError(f"{source_label}: empty guess token embedding list")
    if len(embeddings_guess) != expected_guess_tokens:
        raise ValueError(
            f"{source_label}: got {len(embeddings_guess)} guess embeddings, expected {expected_guess_tokens}"
        )

    # Semantic answer mean over (last_guess_token_index, first_prob_token_index), exclusive-exclusive.
    sem_answer_slice_start = last_guess_token_index
    sem_answer_slice_end = first_prob_token_index
    sem_answer_token_embeddings = source_embeddings[sem_answer_slice_start:sem_answer_slice_end]
    if len(sem_answer_token_embeddings) == 0:
        raise ValueError(
            f"{source_label}: empty semantic-answer token window ({last_guess_token_index},{first_prob_token_index})"
        )
    embeddings_mean_sem_answer = _mean_across_tokens(sem_answer_token_embeddings)
    embeddings_sem_answer_k_tokens = []
    for token_idx in range(sem_answer_slice_start, sem_answer_slice_end):
        emb = _tensor_to_numpy(source_embeddings[token_idx])
        if emb.ndim != 4:
            raise ValueError(
                f"{source_label}: source_embeddings[{token_idx}] has wrong rank: {emb.shape}"
            )
        if token_idx == 0:
            if emb.shape[2] <= 1:
                raise ValueError(
                    f"{source_label}: source_embeddings[0] has seq_len <= 1: {emb.shape}"
                )
            embeddings_sem_answer_k_tokens.append(_last_token_position(emb))
            continue
        if emb.shape[2] != 1:
            raise ValueError(
                f"{source_label}: source_embeddings[{token_idx}] expected seq_len=1, got shape {emb.shape}"
            )
        embeddings_sem_answer_k_tokens.append(emb)

    # Probability span as in prior script, with optional fixed +2 extension.
    prob_span_end_index = end_prob_token_index + (2 if extend_probability_span else 0)
    effective_expected_probability_tokens = (
        expected_probability_tokens + 2 if extend_probability_span else expected_probability_tokens
    )
    embeddings_probability = source_embeddings[first_prob_token_index:prob_span_end_index+1]
    if len(embeddings_probability) != effective_expected_probability_tokens:
        raise ValueError(
            f"{source_label}: got {len(embeddings_probability)} probability embeddings, expected {effective_expected_probability_tokens}"
        )
    embeddings_probability = [_tensor_to_numpy(e) for e in embeddings_probability]
    if (
        prob_value_start_token_index <= 0
        or prob_value_end_token_index >= len(source_embeddings)
        or prob_value_start_token_index > prob_value_end_token_index
    ):
        raise ValueError(
            f"{source_label}: invalid probability value span "
            f"[{prob_value_start_token_index}, {prob_value_end_token_index}]"
        )
    prob_value_token_embeddings = source_embeddings[prob_value_start_token_index:prob_value_end_token_index+1]
    if len(prob_value_token_embeddings) == 0:
        raise ValueError(f"{source_label}: empty probability-value token window")
    embeddings_mean_prob_val = _mean_across_tokens(prob_value_token_embeddings)

    if attention_score_tokenwise_k_mode:
        if source_label != "k":
            raise ValueError(
                f"{source_label}: tokenwise K prompt/sem-answer mode is only valid for source_label='k'"
            )
        return {
            "embeddings_prompt_k_tokens": [_tensor_to_numpy(e) for e in embeddings_prompt_k_tokens],
            "embeddings_guess": [_tensor_to_numpy(e) for e in embeddings_guess],
            "embeddings_sem_answer_k_tokens": [
                _tensor_to_numpy(e) for e in embeddings_sem_answer_k_tokens
            ],
            "embeddings_probability": embeddings_probability,
            "embeddings_mean_prob_val": _tensor_to_numpy(embeddings_mean_prob_val),
        }

    return {
        "embeddings_mean_prompt": _tensor_to_numpy(embeddings_mean_prompt),
        "embeddings_guess": [_tensor_to_numpy(e) for e in embeddings_guess],
        "embeddings_mean_sem_answer": _tensor_to_numpy(embeddings_mean_sem_answer),
        "embeddings_probability": embeddings_probability,
        "embeddings_mean_prob_val": _tensor_to_numpy(embeddings_mean_prob_val),
    }


def _write_ndarray_dataset(group: h5py.Group, name: str, arr: np.ndarray) -> None:
    if arr.ndim == 0:
        group.create_dataset(name, data=arr)
        return
    init_shape = (0,) + tuple(arr.shape[1:])
    max_shape = (None,) + tuple(arr.shape[1:])
    ds = group.create_dataset(
        name,
        shape=init_shape,
        maxshape=max_shape,
        dtype=arr.dtype,
        chunks=True,
    )
    ds.resize(arr.shape)
    ds[...] = arr


def _write_h5_node(group: h5py.Group, name: str, obj) -> None:
    if obj is None:
        none_group = group.create_group(name)
        none_group.attrs["__type__"] = "none"
        return

    if isinstance(obj, dict):
        sub = group.create_group(name)
        sub.attrs["__type__"] = "dict"
        for k, v in obj.items():
            _write_h5_node(sub, str(k), v)
        return

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            sub = group.create_group(name)
            sub.attrs["__type__"] = "tuple" if isinstance(obj, tuple) else "list"
            sub.attrs["__len__"] = 0
            return
        if all(isinstance(x, str) for x in obj):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.asarray(list(obj), dtype=dt))
            return
        sub = group.create_group(name)
        sub.attrs["__type__"] = "tuple" if isinstance(obj, tuple) else "list"
        sub.attrs["__len__"] = len(obj)
        for i, item in enumerate(obj):
            _write_h5_node(sub, str(i), item)
        return

    if isinstance(obj, str):
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=np.asarray(obj, dtype=dt))
        return

    if isinstance(obj, (bool, int, float, np.bool_, np.integer, np.floating)):
        group.create_dataset(name, data=obj)
        return

    arr = _tensor_to_numpy(obj)
    if arr.dtype == np.dtype("O"):
        _write_h5_node(group, name, arr.tolist())
        return
    _write_ndarray_dataset(group, name, arr)


def _decode_h5_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_h5_scalar(value.item())
    return value


def _read_h5_node(node):
    if isinstance(node, h5py.Dataset):
        data = node[()]
        decoded = _decode_h5_scalar(data)
        if isinstance(decoded, np.ndarray):
            # Decode unicode/bytes arrays for token lists.
            if decoded.dtype.kind in {"S", "U", "O"}:
                return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in decoded.tolist()]
            return decoded
        return decoded

    if isinstance(node, h5py.Group):
        node_type = node.attrs.get("__type__")
        if isinstance(node_type, bytes):
            node_type = node_type.decode("utf-8")

        if node_type == "none":
            return None
        if node_type in {"list", "tuple"}:
            length = int(node.attrs.get("__len__", 0))
            items = [_read_h5_node(node[str(i)]) for i in range(length)]
            return tuple(items) if node_type == "tuple" else items

        # For dict-tagged groups and untagged container groups, recurse as dict.
        return {k: _read_h5_node(v) for k, v in node.items()}

    raise TypeError(f"Unsupported HDF5 node type: {type(node)}")


def _first_and_last_layer_values_for_list_of_arrays(arr_list, n: int = 5, *, shape_only: bool = False):
    if not arr_list:
        return {"summary": "empty list"}
    arr0 = _tensor_to_numpy(arr_list[0])
    arr_last = _tensor_to_numpy(arr_list[-1])
    out = {
        "length": len(arr_list),
        "first_elem_shape": list(arr0.shape) if hasattr(arr0, "shape") else None,
        "last_elem_shape": list(arr_last.shape) if hasattr(arr_last, "shape") else None,
    }
    if shape_only:
        return out
    try:
        if arr0.ndim >= 1 and arr0.shape[0] > 0:
            first_layer = np.asarray(arr0[0]).ravel()[:n].tolist()
            last_layer = np.asarray(arr0[-1]).ravel()[:n].tolist()
            out["first_elem_first_layer"] = first_layer
            out["first_elem_last_layer"] = last_layer
    except (IndexError, AttributeError, TypeError, ValueError):
        pass
    return out


def convert_for_json(obj, parent_key=None, in_embedding_field=False, shape_only: bool = False):
    if parent_key in _EMBEDDING_KEYS:
        in_embedding_field = True
    elif in_embedding_field and parent_key in {"res", "attn", "mlp", "q", "k", "v", "o"}:
        in_embedding_field = True

    if isinstance(obj, np.ndarray):
        if in_embedding_field:
            out = {"shape": list(obj.shape)}
            if not shape_only:
                out["preview"] = obj.ravel()[:5].tolist()
            return out
        return obj.tolist()
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            if in_embedding_field:
                arr = _tensor_to_numpy(obj)
                out = {"shape": list(arr.shape)}
                if not shape_only:
                    out["preview"] = arr.ravel()[:5].tolist()
                return out
            return obj.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {
            k: convert_for_json(v, parent_key=k, in_embedding_field=in_embedding_field, shape_only=shape_only)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        if in_embedding_field and obj and hasattr(obj[0], "shape"):
            return _first_and_last_layer_values_for_list_of_arrays(
                [_tensor_to_numpy(x) for x in obj], 5, shape_only=shape_only
            )
        return [
            convert_for_json(x, parent_key, in_embedding_field=in_embedding_field, shape_only=shape_only)
            for x in obj
        ]
    return obj


def process_example(
    example_id,
    example: dict,
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
    collect_attn_block_embeddings: bool,
    collect_mlp_block_embeddings: bool,
    collect_qkvo_embeddings: bool,
    attention_score_tokenwise_k_mode: bool,
    extend_probability_span: bool,
) -> dict | None:
    most_likely = example.get("most_likely_answer")
    question = example.get("question")
    if not isinstance(most_likely, dict):
        logging.warning("Skipping example %s: missing most_likely_answer dict", example_id)
        return None

    response_str = most_likely.get("response")
    decoded_tokens = most_likely.get("decoded_tokens")
    all_embeddings = most_likely.get("all_embeddings")
    all_attn_embeddings = most_likely.get("all_attn_embeddings")
    all_mlp_embeddings = most_likely.get("all_mlp_embeddings")
    all_q_embeddings = most_likely.get("all_q_embeddings")
    all_k_embeddings = most_likely.get("all_k_embeddings")
    all_v_embeddings = most_likely.get("all_v_embeddings")
    all_o_embeddings = most_likely.get("all_o_embeddings")

    if response_str is None or decoded_tokens is None or all_embeddings is None:
        logging.warning(
            "Skipping example %s: missing one of response/decoded_tokens/all_embeddings",
            example_id,
        )
        return None
    full_str = "".join(decoded_tokens)
    prob = parse_probability_from_response(full_str)
    if prob is None:
        logging.warning(
            "Skipping example %s: could not parse probability from response. response=%r",
            example_id,
            response_str,
        )
        return None

    indices = parse_guess_and_probability_indices(decoded_tokens)
    if indices is None:
        logging.warning(
            "Skipping example %s: could not parse Guess/Probability token spans. response=%r",
            example_id,
            response_str,
        )
        return None
    last_guess_token_index, first_prob_token_index, end_prob_token_index = indices
    prob_value_token_span = _probability_value_token_span(decoded_tokens, full_str)
    if prob_value_token_span is None:
        logging.warning(
            "Skipping example %s: could not map probability value token span. response=%r",
            example_id,
            response_str,
        )
        return None
    prob_value_start_token_index, prob_value_end_token_index = prob_value_token_span
    if not (prob_value_start_token_index <= end_prob_token_index <= prob_value_end_token_index):
        logging.warning(
            "Skipping example %s: value token span [%d,%d] does not include end_prob_token_index=%d. response=%r",
            example_id,
            prob_value_start_token_index,
            prob_value_end_token_index,
            end_prob_token_index,
            response_str,
        )
        return None
    if extend_probability_span and end_prob_token_index + 2 >= len(decoded_tokens):
        logging.warning(
            "Skipping example %s: need 2 extra probability tokens after index %d, but decoded_tokens has len=%d. response=%r",
            example_id,
            end_prob_token_index,
            len(decoded_tokens),
            response_str,
        )
        return None

    try:
        res_processed = _compute_embeddings_from_source(
            source_embeddings=all_embeddings,
            decoded_tokens=decoded_tokens,
            last_guess_token_index=last_guess_token_index,
            first_prob_token_index=first_prob_token_index,
            end_prob_token_index=end_prob_token_index,
            prob_value_start_token_index=prob_value_start_token_index,
            prob_value_end_token_index=prob_value_end_token_index,
            expected_guess_tokens=expected_guess_tokens,
            expected_probability_tokens=expected_probability_tokens,
            example_id=example_id,
            source_label="res",
            extend_probability_span=extend_probability_span,
        )
    except ValueError as exc:
        logging.warning("Skipping example %s: %s", example_id, exc)
        return None

    attn_processed = None
    if collect_attn_block_embeddings:
        if all_attn_embeddings is None:
            logging.error(
                "Example %s missing all_attn_embeddings while --collect_attn_block_embeddings is enabled; setting attn outputs to null.",
                example_id,
            )
        else:
            try:
                attn_processed = _compute_embeddings_from_source(
                    source_embeddings=all_attn_embeddings,
                    decoded_tokens=decoded_tokens,
                    last_guess_token_index=last_guess_token_index,
                    first_prob_token_index=first_prob_token_index,
                    end_prob_token_index=end_prob_token_index,
                    prob_value_start_token_index=prob_value_start_token_index,
                    prob_value_end_token_index=prob_value_end_token_index,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    example_id=example_id,
                    source_label="attn",
                    extend_probability_span=extend_probability_span,
                )
            except ValueError as exc:
                logging.error(
                    "Example %s has invalid all_attn_embeddings: %s. Setting attn outputs to null.",
                    example_id,
                    exc,
                )

    mlp_processed = None
    if collect_mlp_block_embeddings:
        if all_mlp_embeddings is None:
            logging.error(
                "Example %s missing all_mlp_embeddings while --collect_mlp_block_embeddings is enabled; setting mlp outputs to null.",
                example_id,
            )
        else:
            try:
                mlp_processed = _compute_embeddings_from_source(
                    source_embeddings=all_mlp_embeddings,
                    decoded_tokens=decoded_tokens,
                    last_guess_token_index=last_guess_token_index,
                    first_prob_token_index=first_prob_token_index,
                    end_prob_token_index=end_prob_token_index,
                    prob_value_start_token_index=prob_value_start_token_index,
                    prob_value_end_token_index=prob_value_end_token_index,
                    expected_guess_tokens=expected_guess_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                    example_id=example_id,
                    source_label="mlp",
                    extend_probability_span=extend_probability_span,
                )
            except ValueError as exc:
                logging.error(
                    "Example %s has invalid all_mlp_embeddings: %s. Setting mlp outputs to null.",
                    example_id,
                    exc,
                )

    optional_sources = {
        "q": {
            "enabled": collect_qkvo_embeddings,
            "source_embeddings": all_q_embeddings,
            "missing_log": (
                "Example %s missing all_q_embeddings while --collect_qkvo_embeddings is enabled; "
                "setting q outputs to null."
            ),
            "invalid_log": "Example %s has invalid all_q_embeddings: %s. Setting q outputs to null.",
            "processed": None,
        },
        "k": {
            "enabled": collect_qkvo_embeddings,
            "source_embeddings": all_k_embeddings,
            "missing_log": (
                "Example %s missing all_k_embeddings while --collect_qkvo_embeddings is enabled; "
                "setting k outputs to null."
            ),
            "invalid_log": "Example %s has invalid all_k_embeddings: %s. Setting k outputs to null.",
            "processed": None,
        },
        "v": {
            "enabled": collect_qkvo_embeddings,
            "source_embeddings": all_v_embeddings,
            "missing_log": (
                "Example %s missing all_v_embeddings while --collect_qkvo_embeddings is enabled; "
                "setting v outputs to null."
            ),
            "invalid_log": "Example %s has invalid all_v_embeddings: %s. Setting v outputs to null.",
            "processed": None,
        },
        "o": {
            "enabled": collect_qkvo_embeddings,
            "source_embeddings": all_o_embeddings,
            "missing_log": (
                "Example %s missing all_o_embeddings while --collect_qkvo_embeddings is enabled; "
                "setting o outputs to null."
            ),
            "invalid_log": "Example %s has invalid all_o_embeddings: %s. Setting o outputs to null.",
            "processed": None,
        },
    }

    k_tokenwise_processed = None
    if attention_score_tokenwise_k_mode:
        if all_k_embeddings is None:
            logging.error(
                "Example %s missing all_k_embeddings while --attention_score_tokenwise_k_mode is enabled; skipping example.",
                example_id,
            )
            return None
        try:
            k_tokenwise_processed = _compute_embeddings_from_source(
                source_embeddings=all_k_embeddings,
                decoded_tokens=decoded_tokens,
                last_guess_token_index=last_guess_token_index,
                first_prob_token_index=first_prob_token_index,
                end_prob_token_index=end_prob_token_index,
                prob_value_start_token_index=prob_value_start_token_index,
                prob_value_end_token_index=prob_value_end_token_index,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
                example_id=example_id,
                source_label="k",
                extend_probability_span=extend_probability_span,
                attention_score_tokenwise_k_mode=True,
            )
        except ValueError as exc:
            logging.error(
                "Example %s has invalid all_k_embeddings for --attention_score_tokenwise_k_mode: %s. Skipping example.",
                example_id,
                exc,
            )
            return None
    for source_label, config in optional_sources.items():
        if not config["enabled"]:
            continue
        if config["source_embeddings"] is None:
            logging.error(config["missing_log"], example_id)
            continue
        try:
            config["processed"] = _compute_embeddings_from_source(
                source_embeddings=config["source_embeddings"],
                decoded_tokens=decoded_tokens,
                last_guess_token_index=last_guess_token_index,
                first_prob_token_index=first_prob_token_index,
                end_prob_token_index=end_prob_token_index,
                prob_value_start_token_index=prob_value_start_token_index,
                prob_value_end_token_index=prob_value_end_token_index,
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
                example_id=example_id,
                source_label=source_label,
                extend_probability_span=extend_probability_span,
            )
        except ValueError as exc:
            logging.error(config["invalid_log"], example_id, exc)

    if attention_score_tokenwise_k_mode:
        processed_response = {
            "response": response_str,
            "decoded_tokens": decoded_tokens,
            "verbalised_confidence": float(prob),
            "embeddings_prompt_k_tokens": {"k": k_tokenwise_processed["embeddings_prompt_k_tokens"]},
            "embeddings_guess": {"res": res_processed["embeddings_guess"]},
            "embeddings_sem_answer_k_tokens": {
                "k": k_tokenwise_processed["embeddings_sem_answer_k_tokens"]
            },
            "embeddings_probability": {"res": res_processed["embeddings_probability"]},
            "embeddings_mean_prob_val": {"res": res_processed["embeddings_mean_prob_val"]},
        }
    else:
        processed_response = {
            "response": response_str,
            "verbalised_confidence": float(prob),
            "embeddings_mean_prompt": {"res": res_processed["embeddings_mean_prompt"]},
            "embeddings_guess": {"res": res_processed["embeddings_guess"]},
            "embeddings_mean_sem_answer": {"res": res_processed["embeddings_mean_sem_answer"]},
            "embeddings_probability": {"res": res_processed["embeddings_probability"]},
            "embeddings_mean_prob_val": {"res": res_processed["embeddings_mean_prob_val"]},
        }
        if collect_attn_block_embeddings:
            processed_response["embeddings_mean_prompt"]["attn"] = (
                None if attn_processed is None else attn_processed["embeddings_mean_prompt"]
            )
            processed_response["embeddings_guess"]["attn"] = (
                None if attn_processed is None else attn_processed["embeddings_guess"]
            )
            processed_response["embeddings_mean_sem_answer"]["attn"] = (
                None if attn_processed is None else attn_processed["embeddings_mean_sem_answer"]
            )
            processed_response["embeddings_probability"]["attn"] = (
                None if attn_processed is None else attn_processed["embeddings_probability"]
            )
            processed_response["embeddings_mean_prob_val"]["attn"] = (
                None if attn_processed is None else attn_processed["embeddings_mean_prob_val"]
            )
        if collect_mlp_block_embeddings:
            processed_response["embeddings_mean_prompt"]["mlp"] = (
                None if mlp_processed is None else mlp_processed["embeddings_mean_prompt"]
            )
            processed_response["embeddings_guess"]["mlp"] = (
                None if mlp_processed is None else mlp_processed["embeddings_guess"]
            )
            processed_response["embeddings_mean_sem_answer"]["mlp"] = (
                None if mlp_processed is None else mlp_processed["embeddings_mean_sem_answer"]
            )
            processed_response["embeddings_probability"]["mlp"] = (
                None if mlp_processed is None else mlp_processed["embeddings_probability"]
            )
            processed_response["embeddings_mean_prob_val"]["mlp"] = (
                None if mlp_processed is None else mlp_processed["embeddings_mean_prob_val"]
            )
    for source_label, config in optional_sources.items():
        if not config["enabled"]:
            continue
        source_processed = config["processed"]
        if not attention_score_tokenwise_k_mode:
            processed_response["embeddings_mean_prompt"][source_label] = (
                None if source_processed is None else source_processed["embeddings_mean_prompt"]
            )
            processed_response["embeddings_mean_sem_answer"][source_label] = (
                None if source_processed is None else source_processed["embeddings_mean_sem_answer"]
            )
        processed_response["embeddings_guess"][source_label] = (
            None if source_processed is None else source_processed["embeddings_guess"]
        )
        processed_response["embeddings_probability"][source_label] = (
            None if source_processed is None else source_processed["embeddings_probability"]
        )
        processed_response["embeddings_mean_prob_val"][source_label] = (
            None if source_processed is None else source_processed["embeddings_mean_prob_val"]
        )

    return {
        "question": question,
        "responses": [processed_response],
    }


def iter_h5_examples(path: Path):
    with h5py.File(path, "r") as input_h5:
        if "examples" not in input_h5:
            raise ValueError(f"Input HDF5 missing 'examples' group: {path}")
        examples_group = input_h5["examples"]
        for example_id in examples_group.keys():
            yield example_id, _read_h5_node(examples_group[example_id])


def _read_verbalised_confidence_from_h5_example(example_node) -> float | None:
    """Read verbalised_confidence from a processed example H5 group (responses/0/...)."""
    if "responses" not in example_node:
        return None
    responses = example_node["responses"]
    if "0" not in responses:
        return None
    response0 = responses["0"]
    if "verbalised_confidence" not in response0:
        return None
    value = _decode_h5_scalar(response0["verbalised_confidence"][()])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def balance_dataset(src_h5: Path, out_dir: Path, *, balance_cap: int) -> int:
    """
    Cap examples per verbalised_confidence bin (rounded to 2 d.p.) at ``balance_cap``.

    Writes `{stem}.h5`, `{stem}.json`, `{stem}_summary.json`, and `samples.txt`
    under `out_dir`. Kept examples preserve original (unrounded) confidence.
    Returns the number of kept examples.
    """
    if balance_cap < 1:
        raise ValueError(f"balance_cap must be >= 1, got {balance_cap}")
    if not src_h5.exists():
        raise FileNotFoundError(f"Source HDF5 not found: {src_h5}")

    bins: dict[float, list[str]] = defaultdict(list)
    n_skip = 0
    with h5py.File(src_h5, "r") as input_h5:
        if "examples" not in input_h5:
            raise ValueError(f"Input HDF5 missing 'examples' group: {src_h5}")
        examples_group = input_h5["examples"]
        for example_id in examples_group.keys():
            conf = _read_verbalised_confidence_from_h5_example(examples_group[example_id])
            if conf is None:
                n_skip += 1
                logging.warning(
                    "Example %s missing/invalid verbalised_confidence; skipping for balance.",
                    example_id,
                )
                continue
            bins[round(conf, 2)].append(str(example_id))

    if not bins:
        raise ValueError(f"No examples with verbalised_confidence found in {src_h5}")

    bin_counts = [len(ids) for ids in bins.values()]
    keep_ids: set[str] = set()
    for rounded, ids in bins.items():
        kept_for_bin = ids[:balance_cap]
        keep_ids.update(kept_for_bin)
        logging.info(
            "Balance bin %.2f: count=%d kept=%d%s",
            rounded,
            len(ids),
            len(kept_for_bin),
            " (capped)" if len(ids) > balance_cap else "",
        )

    logging.info(
        "Balance cap=%d over %d bins; keeping %d / %d examples (skipped %d)",
        balance_cap,
        len(bins),
        len(keep_ids),
        sum(bin_counts),
        n_skip,
    )
    if not keep_ids:
        raise ValueError(f"Balance produced zero kept examples from {src_h5}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_h5.stem
    h5_path = out_dir / f"{stem}.h5"
    json_path = out_dir / f"{stem}.json"
    json_summary_path = out_dir / f"{stem}_summary.json"
    samples_txt_path = out_dir / "samples.txt"

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["format"] = "native_examples_v1"
        h5_file.require_group("examples")

    n_kept = 0
    first_item = True
    with (
        h5py.File(h5_path, "a") as out_h5,
        open(json_path, "w", encoding="utf-8") as json_file,
        open(json_summary_path, "w", encoding="utf-8") as json_summary_file,
    ):
        out_examples = out_h5["examples"]
        json_file.write("{\n")
        json_summary_file.write("{\n")
        for example_id, example in iter_h5_examples(src_h5):
            eid = str(example_id)
            if eid not in keep_ids:
                continue

            n_kept += 1
            if eid in out_examples:
                del out_examples[eid]
            _write_h5_node(out_examples, eid, example)

            if not first_item:
                json_file.write(",\n")
                json_summary_file.write(",\n")
            json_file.write(f'  "{example_id}": ')
            json_summary_file.write(f'  "{example_id}": ')
            json_str = json.dumps(convert_for_json(example), indent=2)
            json_summary_str = json.dumps(convert_for_json(example, shape_only=True), indent=2)
            indented = "\n".join("    " + line if line.strip() else line for line in json_str.split("\n"))
            json_summary_indented = "\n".join(
                "    " + line if line.strip() else line for line in json_summary_str.split("\n")
            )
            json_file.write(indented)
            json_summary_file.write(json_summary_indented)
            first_item = False

        json_file.write("\n}")
        json_summary_file.write("\n}")

    with open(samples_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{n_kept} samples\n")

    logging.info("Wrote balanced %s", h5_path)
    logging.info("Wrote balanced %s", json_path)
    logging.info("Wrote balanced %s", json_summary_path)
    logging.info("Wrote balanced %s", samples_txt_path)
    return n_kept


def write_config_txt(run_dir: Path, args, input_path: Path, out_base: str, output_paths: dict[str, Path]) -> Path:
    config_path = run_dir / "config.txt"
    lines = [
        f"script: {Path(__file__).name}",
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"input_path: {input_path.resolve()}",
        f"output_base: {out_base}",
        f"run_dir: {run_dir.resolve()}",
        "arguments:",
    ]
    for key, value in sorted(vars(args).items()):
        lines.append(f"  {key}: {value}")
    lines.append("outputs:")
    for key, path in output_paths.items():
        lines.append(f"  {key}: {path.resolve()}")

    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write("\n".join(lines) + "\n")
    return config_path


def append_total_duration_to_config(config_path: Path, total_duration_seconds: float) -> None:
    with open(config_path, "a", encoding="utf-8") as config_file:
        config_file.write(f"total_duration_seconds: {total_duration_seconds:.2f}\n")


def main():
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Process HDF5 generations into verbalised-confidence embedding HDF5."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-12b-it",
        help=(
            "Model name used to select Guess/Probability prefix token tables. "
            "Supported: google/gemma-3-12b-it, Qwen/Qwen2.5-32B-Instruct, "
            "mistralai/Mistral-7B-Instruct-v0.1."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to train_generations.h5 or validation_generations.h5",
    )
    parser.add_argument(
        "--output_dir",
        default="./processed_generations_more_h5",
        help="Output directory for HDF5 and JSON (default: ./processed_generations_h5)",
    )
    parser.add_argument(
        "--output_suffix",
        default="_verbalised_embeddings",
        help="Suffix for output filenames (default: _verbalised_embeddings)",
    )
    parser.add_argument(
        "--expected_guess_tokens",
        type=int,
        default=3,
        help="Expected number of stored guess embeddings (list length).",
    )
    parser.add_argument(
        "--expected_probability_tokens",
        type=int,
        default=5,
        help="Expected number of stored probability embeddings (list length).",
    )
    parser.add_argument(
        "--collect_attn_block_embeddings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include attn subblock outputs under each embedding field as key 'attn'.",
    )
    parser.add_argument(
        "--collect_mlp_block_embeddings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include mlp subblock outputs under each embedding field as key 'mlp'.",
    )
    parser.add_argument(
        "--collect_qkvo_embeddings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include q/k/v/o_proj outputs under each embedding field as keys 'q', 'k', 'v', 'o'.",
    )
    parser.add_argument(
        "--attention_score_tokenwise_k_mode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Store tokenwise K prompt/semantic-answer fields for downstream attention "
            "score analysis (`embeddings_prompt_k_tokens`, `embeddings_sem_answer_k_tokens`)."
        ),
    )
    parser.add_argument(
        "--extend_probability_span",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Extend embeddings_probability by 2 tokens past the first probability-value token "
            "(for example, 7 to 9 with default --expected_probability_tokens)."
        ),
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help=(
            "If set, stop after this many successfully processed examples "
            "(rejected examples do not count)."
        ),
    )
    parser.add_argument(
        "--balance",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "After processing, write a balanced copy under balanced/, capping each "
            "verbalised_confidence bin at --balance_cap (default: True). "
            "Ignored when --balance_from_h5 is set."
        ),
    )
    parser.add_argument(
        "--balance_cap",
        type=int,
        default=None,
        help=(
            "Max examples to keep per verbalised_confidence bin (rounded to 2 d.p.). "
            "Required when balancing (--balance or --balance_from_h5)."
        ),
    )
    parser.add_argument(
        "--balance_from_h5",
        default=None,
        help=(
            "Path to an existing verbalised embeddings HDF5 to balance. "
            "Skips normal processing; writes outputs under <parent>/balanced/."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        configure_prefix_tokens_for_model(args.model_name)
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    will_balance = args.balance_from_h5 is not None or args.balance
    if will_balance and args.balance_cap is None:
        logging.error("--balance_cap is required when balancing")
        sys.exit(1)
    if args.balance_cap is not None and args.balance_cap < 1:
        logging.error("--balance_cap must be >= 1, got %s", args.balance_cap)
        sys.exit(1)
    if args.max_examples is not None and args.max_examples < 1:
        logging.error("--max_examples must be >= 1, got %s", args.max_examples)
        sys.exit(1)

    if args.balance_from_h5 is not None:
        src_h5 = Path(args.balance_from_h5)
        if not src_h5.exists():
            logging.error("Balance source HDF5 not found: %s", src_h5)
            sys.exit(1)
        out_dir = src_h5.parent / "balanced"
        logging.info("Balance-only mode: balancing %s into %s", src_h5, out_dir)
        try:
            n_kept = balance_dataset(src_h5, out_dir, balance_cap=args.balance_cap)
        except (FileNotFoundError, ValueError) as exc:
            logging.error("%s", exc)
            sys.exit(1)
        total_elapsed = time.perf_counter() - total_start
        logging.info("Balanced %d examples in %.2fs", n_kept, total_elapsed)
        return

    if args.input is None:
        logging.error("Either --input or --balance_from_h5 is required")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        sys.exit(1)

    stem = input_path.stem
    if stem.endswith("_generations"):
        base = stem.replace("_generations", "")
    else:
        raise ValueError(f"Input file does not end with '_generations': {input_path}")

    out_base = f"{base}{args.output_suffix}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_number = max((int(n) for n in existing), default=0) + 1
    run_dir = output_dir / str(run_number)
    run_dir.mkdir(parents=True, exist_ok=True)
    h5_path = run_dir / f"{out_base}.h5"
    json_path = run_dir / f"{out_base}.json"
    json_summary_path = run_dir / f"{out_base}_summary.json"
    samples_txt_path = run_dir / "samples.txt"
    config_path = write_config_txt(
        run_dir=run_dir,
        args=args,
        input_path=input_path,
        out_base=out_base,
        output_paths={
            "h5": h5_path,
            "json": json_path,
            "json_summary": json_summary_path,
            "samples_txt": samples_txt_path,
        },
    )

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["format"] = "native_examples_v1"
        h5_file.require_group("examples")

    output_log_path = run_dir / "output.log"
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    file_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    logging.info("Files will be saved to %s", run_dir)
    logging.info("Wrote %s", config_path)

    n_ok = 0
    n_reject = 0
    first_item = True
    t_start = time.perf_counter()

    with (
        h5py.File(h5_path, "a") as out_h5,
        open(json_path, "w") as json_file,
        open(json_summary_path, "w") as json_summary_file,
    ):
        out_examples = out_h5["examples"]
        json_file.write("{\n")
        json_summary_file.write("{\n")
        for example_id, example in iter_h5_examples(input_path):
            out = process_example(
                example_id,
                example,
                expected_guess_tokens=args.expected_guess_tokens,
                expected_probability_tokens=args.expected_probability_tokens,
                collect_attn_block_embeddings=args.collect_attn_block_embeddings,
                collect_mlp_block_embeddings=args.collect_mlp_block_embeddings,
                collect_qkvo_embeddings=args.collect_qkvo_embeddings,
                attention_score_tokenwise_k_mode=args.attention_score_tokenwise_k_mode,
                extend_probability_span=args.extend_probability_span,
            )
            if out is None:
                n_reject += 1
                continue

            n_ok += 1
            if str(example_id) in out_examples:
                del out_examples[str(example_id)]
            _write_h5_node(out_examples, str(example_id), out)

            if not first_item:
                json_file.write(",\n")
                json_summary_file.write(",\n")
            json_file.write(f'  "{example_id}": ')
            json_summary_file.write(f'  "{example_id}": ')
            json_str = json.dumps(convert_for_json(out), indent=2)
            json_summary_str = json.dumps(convert_for_json(out, shape_only=True), indent=2)
            indented = "\n".join("    " + line if line.strip() else line for line in json_str.split("\n"))
            json_summary_indented = "\n".join(
                "    " + line if line.strip() else line for line in json_summary_str.split("\n")
            )
            json_file.write(indented)
            json_summary_file.write(json_summary_indented)
            first_item = False

            if (n_ok % 10) == 0:
                logging.info("Processed %d examples", n_ok)

            if args.max_examples is not None and n_ok >= args.max_examples:
                logging.info(
                    "Reached max_examples=%d successful examples; early exit.",
                    args.max_examples,
                )
                break

        json_file.write("\n}")
        json_summary_file.write("\n}")

    if n_ok == 0:
        logging.error("No valid examples processed from input file")
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    logging.info("Wrote %s", h5_path)
    logging.info("Wrote %s", json_path)
    logging.info("Wrote %s", json_summary_path)
    logging.info("Processed %d valid and rejected %d examples in %.2fs", n_ok, n_reject, elapsed)

    with open(samples_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{n_ok} samples\n")
    logging.info("Wrote %s", samples_txt_path)

    if args.balance:
        balanced_dir = run_dir / "balanced"
        logging.info("Balancing dataset into %s", balanced_dir)
        try:
            n_kept = balance_dataset(h5_path, balanced_dir, balance_cap=args.balance_cap)
        except (FileNotFoundError, ValueError) as exc:
            logging.error("%s", exc)
            sys.exit(1)
        logging.info("Balanced dataset kept %d examples", n_kept)

    total_elapsed = time.perf_counter() - total_start
    append_total_duration_to_config(config_path, total_elapsed)
    logging.info("Updated %s with total_duration_seconds=%.2f", config_path, total_elapsed)


if __name__ == "__main__":
    main()

