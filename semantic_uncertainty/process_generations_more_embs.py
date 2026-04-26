"""
Process streamed pickle generations into verbalised-confidence embedding HDF5.

Input:
  Pickle stream produced by generate_answers_with_confidence.py, where each
  batch is a dict: example_id -> example dict. This script processes only
  `most_likely_answer` per example.

Output:
  HDF5 + JSON with one processed response per example containing:
    - verbalised_confidence
    - embeddings_mean_prompt
    - embeddings_guess
    - embeddings_mean_sem_answer
    - embeddings_probability
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from uncertainty.utils import utils
from process_generations_tok_bef_gen import (
    parse_probability_from_response,
)

utils.setup_logger()

_EMBEDDING_KEYS = frozenset(
    {
        "embeddings_mean_prompt",
        "embeddings_guess",
        "embeddings_mean_sem_answer",
        "embeddings_probability",
    }
)

GUESS_PREFIX = "\n\nGuess:"
PROBABILITY_MARKER = "\nProbability:"


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


def parse_guess_and_probability_indices(decoded_tokens: list) -> tuple[int, int, int] | None:
    """
    Returns:
      - last_guess_token_index: first token index of semantic answer
      - first_prob_token_index: token index at "\\n" before "Probability:"
      - end_prob_token_index: first token index of probability value
    """
    full_str = "".join(decoded_tokens)
    if not full_str.startswith(GUESS_PREFIX):
        return None

    last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1

    rfind_start = full_str.rfind(PROBABILITY_MARKER)
    if rfind_start < 0:
        return None

    first_prob_token_index = _token_index_for_char_offset(decoded_tokens, rfind_start)
    prob_whitespace_token_index = _token_index_for_char_offset(
        decoded_tokens, rfind_start + len(PROBABILITY_MARKER) - 1
    ) + 1
    if prob_whitespace_token_index >= len(decoded_tokens):
        return None
    if decoded_tokens[prob_whitespace_token_index].strip() != "":
        return None
    end_prob_token_index = prob_whitespace_token_index + 1

    if (
        last_guess_token_index <= 0
        or last_guess_token_index >= len(decoded_tokens)
        or end_prob_token_index >= len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None

    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def _first_minus_last_position(arr: np.ndarray) -> np.ndarray:
    """Return first-seq-position minus last-seq-position, preserving seq dim of 1."""
    if arr.ndim < 4 or arr.shape[2] < 1:
        raise ValueError(f"Expected embedding rank>=4 with seq axis, got shape {arr.shape}")
    return arr[:, :, 0:1, :] - arr[:, :, -1:, :]


def _mean_first_minus_last_across_tokens(token_embeddings: list[np.ndarray]) -> np.ndarray:
    """Apply first-minus-last per token embedding, then average over token dimension."""
    if not token_embeddings:
        raise ValueError("Cannot mean over empty token embedding list")
    transformed = [_first_minus_last_position(_tensor_to_numpy(e)) for e in token_embeddings]
    stacked = np.stack(transformed, axis=0)
    return np.mean(stacked, axis=0)


def _last_token_position(arr: np.ndarray) -> np.ndarray:
    """Return final sequence position as singleton seq length."""
    arr = _tensor_to_numpy(arr)
    if arr.ndim < 4 or arr.shape[2] < 1:
        raise ValueError(f"Expected embedding rank>=4 with seq axis, got shape {arr.shape}")
    return arr[:, :, -1:, :]


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


def _first_and_last_layer_values_for_list_of_arrays(arr_list, n: int = 5):
    if not arr_list:
        return {"summary": "empty list"}
    arr0 = _tensor_to_numpy(arr_list[0])
    arr_last = _tensor_to_numpy(arr_list[-1])
    out = {
        "length": len(arr_list),
        "first_elem_shape": list(arr0.shape) if hasattr(arr0, "shape") else None,
        "last_elem_shape": list(arr_last.shape) if hasattr(arr_last, "shape") else None,
    }
    try:
        if arr0.ndim >= 1 and arr0.shape[0] > 0:
            first_layer = np.asarray(arr0[0]).ravel()[:n].tolist()
            last_layer = np.asarray(arr0[-1]).ravel()[:n].tolist()
            out["first_elem_first_layer"] = first_layer
            out["first_elem_last_layer"] = last_layer
    except (IndexError, AttributeError, TypeError, ValueError):
        pass
    return out


def convert_for_json(obj, parent_key=None):
    if isinstance(obj, np.ndarray):
        if parent_key in _EMBEDDING_KEYS:
            return {"shape": list(obj.shape), "preview": obj.ravel()[:5].tolist()}
        return obj.tolist()
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            if parent_key in _EMBEDDING_KEYS:
                arr = _tensor_to_numpy(obj)
                return {"shape": list(arr.shape), "preview": arr.ravel()[:5].tolist()}
            return obj.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v, parent_key=k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if parent_key in _EMBEDDING_KEYS and obj and hasattr(obj[0], "shape"):
            return _first_and_last_layer_values_for_list_of_arrays([_tensor_to_numpy(x) for x in obj], 5)
        return [convert_for_json(x, parent_key) for x in obj]
    return obj


def process_example(example_id, example: dict) -> dict | None:
    most_likely = example.get("most_likely_answer")
    question = example.get("question")
    if not isinstance(most_likely, dict):
        logging.warning("Skipping example %s: missing most_likely_answer dict", example_id)
        return None

    response_str = most_likely.get("response")
    decoded_tokens = most_likely.get("decoded_tokens")
    all_embeddings = most_likely.get("all_embeddings")

    if response_str is None or decoded_tokens is None or all_embeddings is None:
        logging.warning(
            "Skipping example %s: missing one of response/decoded_tokens/all_embeddings",
            example_id,
        )
        return None
    if len(all_embeddings) != len(decoded_tokens):
        logging.warning(
            "Skipping example %s: len(all_embeddings)=%d != len(decoded_tokens)=%d",
            example_id,
            len(all_embeddings),
            len(decoded_tokens),
        )
        return None

    full_str = "".join(decoded_tokens)
    prob = parse_probability_from_response(full_str)
    if prob is None:
        logging.warning("Skipping example %s: could not parse probability from response", example_id)
        return None

    indices = parse_guess_and_probability_indices(decoded_tokens)
    if indices is None:
        logging.warning("Skipping example %s: could not parse Guess/Probability token spans", example_id)
        return None
    last_guess_token_index, first_prob_token_index, end_prob_token_index = indices

    # Guess span: first generated token through token before Probability marker.
    guess_token_indices = range(0, first_prob_token_index)
    embeddings_guess = []
    for token_idx in guess_token_indices:
        if token_idx == 0:
            continue
        embeddings_guess.append(_last_token_position(all_embeddings[token_idx - 1]))

    # Prompt mean: use all available pre-generation token embeddings from all_embeddings[:-1].
    prompt_token_embeddings = all_embeddings[:-1]
    if len(prompt_token_embeddings) == 0:
        logging.warning("Skipping example %s: empty prompt-token embedding list", example_id)
        return None
    embeddings_mean_prompt = _mean_first_minus_last_across_tokens(prompt_token_embeddings)

    # Semantic answer mean over (last_guess_token_index, first_prob_token_index), exclusive-exclusive.
    sem_answer_slice_start = last_guess_token_index + 1
    sem_answer_slice_end = first_prob_token_index
    sem_answer_token_embeddings = all_embeddings[sem_answer_slice_start:sem_answer_slice_end]
    if len(sem_answer_token_embeddings) == 0:
        logging.warning(
            "Skipping example %s: empty semantic-answer token window (%d,%d)",
            example_id,
            last_guess_token_index,
            first_prob_token_index,
        )
        return None
    embeddings_mean_sem_answer = _mean_first_minus_last_across_tokens(sem_answer_token_embeddings)

    # Probability span as in prior script.
    embeddings_probability = all_embeddings[first_prob_token_index : end_prob_token_index + 1]
    embeddings_probability = [_tensor_to_numpy(e) for e in embeddings_probability]

    processed_response = {
        "verbalised_confidence": float(prob),
        "embeddings_mean_prompt": _tensor_to_numpy(embeddings_mean_prompt),
        "embeddings_guess": [_tensor_to_numpy(e) for e in embeddings_guess],
        "embeddings_mean_sem_answer": _tensor_to_numpy(embeddings_mean_sem_answer),
        "embeddings_probability": embeddings_probability,
    }

    return {
        "question": question,
        "responses": [processed_response],
    }


def iter_pickle_examples(path: Path):
    with open(path, "rb") as input_file:
        batch_num = 0
        while True:
            try:
                batch = pickle.load(input_file)
                batch_num += 1
                if not isinstance(batch, dict):
                    logging.warning("Skipping non-dict batch %d", batch_num)
                    continue
                for example_id, example in batch.items():
                    yield example_id, example
            except EOFError:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Process pickle generations into verbalised-confidence embedding HDF5."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to train_generations.pkl or validation_generations.pkl",
    )
    parser.add_argument(
        "--output_dir",
        default="./processed_generations_h5",
        help="Output directory for HDF5 and JSON (default: ./processed_generations_h5)",
    )
    parser.add_argument(
        "--output_suffix",
        default="_verbalised_embeddings",
        help="Suffix for output filenames (default: _verbalised_embeddings)",
    )
    args = parser.parse_args()

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

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["format"] = "native_examples_v1"
        h5_file.require_group("examples")

    output_log_path = run_dir / "output.log"
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    logging.info("Files will be saved to %s", run_dir)

    n_ok = 0
    n_reject = 0
    first_item = True
    t_start = time.perf_counter()

    with h5py.File(h5_path, "a") as out_h5, open(json_path, "w") as json_file:
        out_examples = out_h5["examples"]
        json_file.write("{\n")
        for example_id, example in iter_pickle_examples(input_path):
            out = process_example(example_id, example)
            if out is None:
                n_reject += 1
                continue

            n_ok += 1
            if str(example_id) in out_examples:
                del out_examples[str(example_id)]
            _write_h5_node(out_examples, str(example_id), out)

            if not first_item:
                json_file.write(",\n")
            json_file.write(f'  "{example_id}": ')
            json_str = json.dumps(convert_for_json(out), indent=2)
            indented = "\n".join("    " + line if line.strip() else line for line in json_str.split("\n"))
            json_file.write(indented)
            first_item = False

            if (n_ok % 10) == 0:
                logging.info("Processed %d examples", n_ok)

        json_file.write("\n}")

    if n_ok == 0:
        logging.error("No valid examples processed from pickle file")
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    logging.info("Wrote %s", h5_path)
    logging.info("Wrote %s", json_path)
    logging.info("Processed %d valid and rejected %d examples in %.2fs", n_ok, n_reject, elapsed)

    samples_txt_path = run_dir / "samples.txt"
    with open(samples_txt_path, "w") as f:
        f.write(f"{n_ok} samples\n")
    logging.info("Wrote %s", samples_txt_path)


if __name__ == "__main__":
    main()
