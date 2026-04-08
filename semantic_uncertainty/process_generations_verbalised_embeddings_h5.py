"""
Process generations HDF5 for verbalised-confidence embedding subsets.

Reads a generations HDF5 file where each response is a dict with 'response',
'all_embeddings', and 'decoded_tokens'. Validates "Guess: <guess> \\nProbability:
<number>" structure, extracts two embedding subsets (guess span and probability
span), and writes HDF5 + JSON.
JSON file only contains one example, because of streaming

Input: HDF5 with group "examples", where each key is an example id
(e.g. train_generations.h5 from generate_answers_with_confidence_h5.py).
Tuple responses or missing keys are skipped with a log.

Usage:
  python -m semantic_uncertainty.process_generations_verbalised_embeddings_h5 --input train_generations.h5
  python -m semantic_uncertainty.process_generations_verbalised_embeddings_h5 --input validation_generations.h5 --output_dir ./out

Output:
  - HDF5: verbalised_confidence, embeddings_guess (list of arrays), embeddings_probability (list of arrays).
  - JSON: same structure with embeddings truncated for inspection.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from uncertainty.utils import utils

# Reuse parsing from tok_bef_gen script
from process_generations_tok_bef_gen import (
    REJECT_KEYWORD,
    parse_probability_from_response,
    # TODO: why is this not used?
    prompt_user_for_probability,
)

utils.setup_logger()

# Keys whose values are embedding (lists of) arrays: truncate in JSON
_EMBEDDING_KEYS = frozenset({"embeddings_guess", "embeddings_probability"})

# Debug: log first N examples with full detail
_DEBUG_FIRST_N = 3


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

    # default dict-like group
    out = {}
    for k in node.keys():
        out[k] = _read_h5_node(node[k])
    return out


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


def load_examples_h5(path: Path):
    """Yield (example_id, example_dict) from native HDF5 examples group."""
    total_node_read_time = 0.0
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        examples_group = h5_file["examples"]
        for example_id in examples_group.keys():
            t0 = time.perf_counter()
            obj = _read_h5_node(examples_group[example_id])
            elapsed = time.perf_counter() - t0
            total_node_read_time += elapsed
            if not isinstance(obj, dict):
                logging.warning(f"Skipping non-dict example {example_id}")
                continue
            yield example_id, obj
    logging.info(
        "Total time spent reading H5 nodes from %s: %.2fs",
        path,
        total_node_read_time,
    )


def _print_decoded_tokens_neatly(decoded_tokens: list, example_id) -> None:
    """Print decoded_tokens with index and repr for parse-failure debugging."""
    logging.debug(f"[{example_id}] decoded_tokens (len={len(decoded_tokens)}):")
    for i, t in enumerate(decoded_tokens):
        logging.debug(f"  {i:3d}: {repr(t)}")


def _tensor_to_numpy(obj):
    """Convert tensor or array-like to numpy. For lists of tensors, convert each element."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def normalise_response_item(item) -> dict | None:
    """
    Normalise a response item to a common shape for verbalised-embedding processing.

    Only dicts with 'response', 'all_embeddings', and 'decoded_tokens' are accepted.
    Tuple items or missing keys return None (caller should log skip).
    """
    if not isinstance(item, dict):
        return None
    response = item.get("response")
    all_embeddings = item.get("all_embeddings")
    decoded_tokens = item.get("decoded_tokens")
    if response is None or all_embeddings is None or decoded_tokens is None:
        return None
    return {
        "response": response,
        "all_embeddings": all_embeddings,
        "decoded_tokens": decoded_tokens,
    }


def _token_index_for_char_offset(decoded_tokens: list, char_offset: int) -> int:
    """
    Return the smallest token index i such that the cumulative length (num of chars) of
    decoded_tokens[0..i+1] (exclusive end) is strictly greater than char_offset.
    """
    cumulative = 0
    for i, tok in enumerate(decoded_tokens):
        cumulative += len(tok)
        if cumulative > char_offset:
            return i
    return max(0, len(decoded_tokens) - 1)


GUESS_PREFIX = "\n\nGuess:"
PROBABILITY_MARKER = "\nProbability:"


def parse_guess_and_probability_indices(
    decoded_tokens: list,
    full_str: str,
    example_id,
) -> tuple[int, int, int] | None:
    """
    Compute token indices for the two embedding subsets.

    Guess span: token 0 through the token right after the token containing
    "\\n\\nGuess:". Probability span: tokens for the literal last
    "\\nProbability:" plus one following token.

    Returns (last_guess_token_index, first_prob_token_index, end_prob_token_index)
    or None on failure.
    """
    if not full_str.startswith(GUESS_PREFIX):
        return None

    last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1

    rfind_start = full_str.rfind(PROBABILITY_MARKER)
    if rfind_start < 0:
        return None

    first_prob_token_index = _token_index_for_char_offset(decoded_tokens, rfind_start)
    end_prob_token_index = _token_index_for_char_offset(
        decoded_tokens, rfind_start + len(PROBABILITY_MARKER) - 1
    ) + 1

    if (
        last_guess_token_index <= 0
        or last_guess_token_index >= len(decoded_tokens)
        or end_prob_token_index >= len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None

    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def prompt_user_on_structure_failure(
    example_id,
    full_str: str,
    decoded_tokens: list,
    response_str: str,
) -> float | str:
    """
    Interrupt on structure failure: print details and let user enter probability or reject.
    Returns float in [0,1] or REJECT_KEYWORD.
    """
    logging.warning(
        f"\n[Example id: {example_id}] Response does not match 'Guess: ... \\nProbability: <number>'."
    )
    logging.warning("-" * 60)
    logging.warning("full_str (from decoded_tokens):")
    logging.warning(full_str)
    logging.warning("-" * 60)
    _print_decoded_tokens_neatly(decoded_tokens, example_id)
    logging.warning("-" * 60)
    logging.warning("response (original):")
    logging.warning(response_str)
    logging.warning("-" * 60)
    line = input(f"Enter a number in [0, 1] or '{REJECT_KEYWORD}' to exclude this example: ")
    if line == REJECT_KEYWORD:
        return REJECT_KEYWORD
    try:
        value = float(line.replace(",", "."))
        if 0 <= value <= 1:
            return value
    except ValueError:
        pass
    logging.warning("Invalid input. Enter a number in [0, 1] or 'reject'.")
    return prompt_user_on_structure_failure(example_id, full_str, decoded_tokens, response_str)


def _first_and_last_layer_values_for_list_of_arrays(arr_list, n: int = 5):
    """
    For a list of arrays (each shape [layers, batch, seq, dim]), return a compact summary
    for JSON: e.g. first element first/last layer and last element first/last layer.
    """
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
    """Convert numpy/torch to JSON; for embedding list keys show compact summary."""
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


def process_example(
    example_id,
    example: dict,
    debug_first_n: int,
    example_index: int = 0,
    prompt_on_parse_failure: bool = True,
) -> dict | None:
    """
    Process one example: only dict items with all_embeddings and decoded_tokens.
    Validates structure, computes embedding subsets. On parse failure: if prompt_on_parse_failure
    then prompt user (accept/reject); else skip the response (auto-reject).
    Returns dict with question, context, responses or None if rejected / no processable responses.
    example_index: 0-based index of this example in the run (used for debug logging first N examples).
    """
    print("Processing example", example_id)
    most_likely = example.get("most_likely_answer")
    responses_list = example.get("responses") or []
    items = []
    if most_likely is not None:
        items.append(most_likely)
    items.extend(responses_list)

    processed = []
    for item in items:
        norm = normalise_response_item(item)
        if norm is None:
            logging.debug(f"Skipping item (tuple or missing all_embeddings/decoded_tokens) in example {example_id}")
            continue

        all_embeddings = norm["all_embeddings"]
        decoded_tokens = norm["decoded_tokens"]
        response_str = norm["response"]

        if len(all_embeddings) != len(decoded_tokens):
            logging.error(
                f"len(all_embeddings)={len(all_embeddings)} != len(decoded_tokens)={len(decoded_tokens)} "
                f"(example_id={example_id})"
            )
            sys.exit(1)

        full_str = "".join(decoded_tokens)

        prob = parse_probability_from_response(full_str)

        has_guess = "guess:" in full_str.lower()
        has_prob = "probability:" in full_str.lower()
        if not has_guess or not has_prob or prob is None:
            if not prompt_on_parse_failure:
                logging.debug(
                    f"Skipping response for example {example_id}: parse failed (no Guess/Probability or no number), --no_prompt set."
                )
                _print_decoded_tokens_neatly(decoded_tokens, example_id)
                continue
            user_val = prompt_user_on_structure_failure(example_id, full_str, decoded_tokens, response_str)
            if user_val == REJECT_KEYWORD:
                return None
            prob = user_val
        else:
            prob = float(prob)

        indices = parse_guess_and_probability_indices(decoded_tokens, full_str, example_id)
        if indices is None:
            if not prompt_on_parse_failure:
                logging.debug(
                    f"Skipping response for example {example_id}: cannot compute token indices, --no_prompt set."
                )
                _print_decoded_tokens_neatly(decoded_tokens, example_id)
                continue
            user_val = prompt_user_on_structure_failure(example_id, full_str, decoded_tokens, response_str)
            if user_val == REJECT_KEYWORD:
                return None
            prob = user_val
            indices = parse_guess_and_probability_indices(decoded_tokens, full_str, example_id)
            if indices is None:
                logging.debug(
                    f"Skipping response for example {example_id}: structure invalid, "
                    "cannot compute token indices for embedding subsets (user provided probability)."
                )
                _print_decoded_tokens_neatly(decoded_tokens, example_id)
                continue

        last_guess_token_index, first_prob_token_index, end_prob_token_index = indices

        embeddings_guess = all_embeddings[0 : last_guess_token_index + 1]
        embeddings_probability = all_embeddings[first_prob_token_index : end_prob_token_index + 1]

        embeddings_guess_np = [_tensor_to_numpy(e) for e in embeddings_guess]
        embeddings_probability_np = [_tensor_to_numpy(e) for e in embeddings_probability]

        debug = example_index < debug_first_n
        if debug:
            logging.debug(f"[{example_id}] full_str (first 120 chars)={repr(full_str[:120])!r}")
            logging.debug(
                f"[{example_id}] last_guess_token_index={last_guess_token_index} "
                f"first_prob_token_index={first_prob_token_index} end_prob_token_index={end_prob_token_index}"
            )
            logging.debug(
                f"[{example_id}] len(embeddings_guess)={len(embeddings_guess_np)} "
                f"len(embeddings_probability)={len(embeddings_probability_np)}"
            )
            if embeddings_guess_np:
                logging.debug(f"[{example_id}] embeddings_guess[0].shape={embeddings_guess_np[0].shape}")
            if embeddings_probability_np:
                logging.debug(f"[{example_id}] embeddings_probability[0].shape={embeddings_probability_np[0].shape}")

        processed.append(
            {
                "verbalised_confidence": float(prob),
                "embeddings_guess": embeddings_guess_np,
                "embeddings_probability": embeddings_probability_np,
                "response": response_str,
                "decoded_tokens": decoded_tokens,
            }
        )

    # Check that the embedding lengths are consistent
    if len(processed) >= 2:
        guess_lens = [len(r["embeddings_guess"]) for r in processed]
        prob_lens = [len(r["embeddings_probability"]) for r in processed]
        if len(set(guess_lens)) != 1 or len(set(prob_lens)) != 1:
            logging.error(
                f"Error: example {example_id} has inconsistent embedding lengths: "
                f"embeddings_guess lengths={guess_lens}, embeddings_probability lengths={prob_lens}"
            )
            sys.exit(1)

    if not processed:
        logging.debug(f"No processable responses for example {example_id} (all skipped or tuple/missing keys).")
        return None

    return {
        "question": example.get("question"),
        "context": example.get("context"),
        "responses": processed,
    }


def main():
    logging.info("Starting process_generations_verbalised_embeddings_h5.py...")
    parser = argparse.ArgumentParser(
        description="Process generations HDF5 for verbalised-confidence embedding subsets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to train_generations.h5 or validation_generations.h5 (dict items with all_embeddings, decoded_tokens)",
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
    parser.add_argument(
        "--debug_first_n",
        type=int,
        default=3,
        help="Number of first examples to log with full debug (default: 3)",
    )
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="If set, do not prompt on parse failure; automatically skip responses that fail parsing.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
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

    # Initialize HDF5 output
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["format"] = "native_examples_v1"
        h5_file.require_group("examples")

    # Set up file logging to output.log in run directory
    output_log_path = run_dir / "output.log"
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Files will be saved to {run_dir}")

    n_ok = 0
    n_reject = 0
    first_item = True
    example_index = 0
    json_file = open(json_path, "w")
    json_file.write("{\n")

    try:
        logging.info(f"Reading input file: {input_path}")
        with h5py.File(h5_path, "a") as out_h5:
            out_examples = out_h5["examples"]
            for example_id, example in load_examples_h5(input_path):
                logging.info("Processing example %s", example_id) # Example is of type dict
                t0 = time.perf_counter()
                out = process_example(
                    example_id,
                    example,
                    args.debug_first_n,
                    example_index,
                    prompt_on_parse_failure=not args.no_prompt,
                )
                if out is None:
                    logging.warning("Skipping example %s", example_id)
                    n_reject += 1
                    example_index += 1
                    continue

                n_ok += 1
                example_index += 1

                processed_example = {
                    "responses": [
                        {
                            "verbalised_confidence": r["verbalised_confidence"],
                            "embeddings_guess": r["embeddings_guess"],
                            "embeddings_probability": r["embeddings_probability"],
                        }
                        for r in out["responses"]
                    ]
                }
                if str(example_id) in out_examples:
                    del out_examples[str(example_id)]
                _write_h5_node(out_examples, str(example_id), processed_example)

                if not first_item:
                    json_file.write(",\n")
                json_file.write(f'  "{example_id}": ')
                json_str = json.dumps(convert_for_json(out), indent=2)
                indented = "\n".join(
                    "    " + line if line.strip() else line for line in json_str.split("\n")
                )
                json_file.write(indented)
                first_item = False

                elapsed = time.perf_counter() - t0
                if (n_ok % 10) == 0:
                    logging.info("Processed %d examples (last in %.3fs)", n_ok, elapsed)

        if n_ok == 0:
            logging.error("No valid examples processed from HDF5 file")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error reading HDF5 file: {e}")
        sys.exit(1)
    finally:
        json_file.write("\n}")
        json_file.close()

    logging.info(f"Wrote {h5_path}")
    logging.info(f"Wrote {json_path}")

    samples_txt_path = run_dir / "samples.txt"
    with open(samples_txt_path, "w") as f:
        f.write(f"{n_ok} samples\n")
    logging.info(f"Wrote {samples_txt_path}")


if __name__ == "__main__":
    main()
