"""
Process generations pickle for verbalised-confidence embedding subsets.

Reads a generations pickle where each response is a dict with 'response', 'all_embeddings',
and 'decoded_tokens'. Validates "Guess: <guess> \\nProbability: <number>" structure,
extracts two embedding subsets (guess span and probability span), and writes pickle + JSON.

Input: pickle with dict items containing all_embeddings and decoded_tokens (e.g. most_likely_answer
from generate_answers_with_confidence.py). Tuple responses or missing keys are skipped with a log.

Usage:
  python -m semantic_uncertainty.process_generations_verbalised_embeddings --input train_generations.pkl
  python -m semantic_uncertainty.process_generations_verbalised_embeddings --input validation_generations.pkl --output_dir ./out

Output:
  - Pickle: verbalised_confidence, embeddings_guess (list of arrays), embeddings_probability (list of arrays).
  - JSON: same structure with embeddings truncated for inspection.
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

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
_EMBEDDING_KEYS = frozenset({'embeddings_guess', 'embeddings_probability'})

# Debug: log first N examples with full detail
_DEBUG_FIRST_N = 3


def _print_decoded_tokens_neatly(decoded_tokens: list, example_id) -> None:
    """Print decoded_tokens with index and repr for parse-failure debugging."""
    logging.debug(f"[{example_id}] decoded_tokens (len={len(decoded_tokens)}):")
    for i, t in enumerate(decoded_tokens):
        logging.debug(f"  {i:3d}: {repr(t)}")


def _tensor_to_numpy(obj):
    """Convert tensor or array-like to numpy. For lists of tensors, convert each element."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
        return obj.cpu().numpy()
    if hasattr(obj, 'numpy'):
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
    response = item.get('response')
    all_embeddings = item.get('all_embeddings')
    decoded_tokens = item.get('decoded_tokens')
    if response is None or all_embeddings is None or decoded_tokens is None:
        return None
    return {
        'response': response,
        'all_embeddings': all_embeddings,
        'decoded_tokens': decoded_tokens,
    }


def _token_index_for_char_offset(decoded_tokens: list, char_offset: int) -> int:
    """
    Return the smallest token index i such that the cumulative length of
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

    Guess span: tokens for the literal prefix "\\n\\nGuess:" only (token 0 through
    the token containing the colon). Probability span: tokens for the literal
    last "\\nProbability:" only (not the number after it).

    Returns (last_guess_token_index, first_prob_token_index, end_prob_token_index)
    or None on failure.
    """
    if not full_str.startswith(GUESS_PREFIX):
        return None

    last_guess_token_index = _token_index_for_char_offset(
        decoded_tokens, len(GUESS_PREFIX) - 1
    )

    rfind_start = full_str.rfind(PROBABILITY_MARKER)
    if rfind_start < 0:
        return None

    first_prob_token_index = _token_index_for_char_offset(decoded_tokens, rfind_start)
    end_prob_token_index = _token_index_for_char_offset(
        decoded_tokens, rfind_start + len(PROBABILITY_MARKER) - 1
    )

    if last_guess_token_index < 0 or end_prob_token_index < first_prob_token_index:
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
    logging.warning(f"\n[Example id: {example_id}] Response does not match 'Guess: ... \\nProbability: <number>'.")
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
    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
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
        if parent_key in _EMBEDDING_KEYS and obj and hasattr(obj[0], 'shape'):
            return _first_and_last_layer_values_for_list_of_arrays(
                [_tensor_to_numpy(x) for x in obj], 5
            )
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
    most_likely = example.get('most_likely_answer')
    responses_list = example.get('responses') or []
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

        all_embeddings = norm['all_embeddings']
        decoded_tokens = norm['decoded_tokens']
        response_str = norm['response']

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
            user_val = prompt_user_on_structure_failure(
                example_id, full_str, decoded_tokens, response_str
            )
            if user_val == REJECT_KEYWORD:
                return None
            prob = user_val
        else:
            prob = float(prob)

        indices = parse_guess_and_probability_indices(
            decoded_tokens, full_str, example_id
        )
        if indices is None:
            if not prompt_on_parse_failure:
                logging.debug(
                    f"Skipping response for example {example_id}: cannot compute token indices, --no_prompt set."
                )
                _print_decoded_tokens_neatly(decoded_tokens, example_id)
                continue
            user_val = prompt_user_on_structure_failure(
                example_id, full_str, decoded_tokens, response_str
            )
            if user_val == REJECT_KEYWORD:
                return None
            prob = user_val
            indices = parse_guess_and_probability_indices(
                decoded_tokens, full_str, example_id
            )
            if indices is None:
                logging.debug(
                    f"Skipping response for example {example_id}: structure invalid, "
                    "cannot compute token indices for embedding subsets (user provided probability)."
                )
                _print_decoded_tokens_neatly(decoded_tokens, example_id)
                continue

        last_guess_token_index, first_prob_token_index, end_prob_token_index = indices

        embeddings_guess = all_embeddings[0 : last_guess_token_index + 1]
        embeddings_probability = all_embeddings[
            first_prob_token_index : end_prob_token_index + 1
        ]

        embeddings_guess_np = [_tensor_to_numpy(e) for e in embeddings_guess]
        embeddings_probability_np = [_tensor_to_numpy(e) for e in embeddings_probability]

        debug = example_index < debug_first_n
        if debug:
            logging.debug(
                f"[{example_id}] full_str (first 120 chars)={repr(full_str[:120])!r}"
            )
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

        processed.append({
            'verbalised_confidence': float(prob),
            'embeddings_guess': embeddings_guess_np,
            'embeddings_probability': embeddings_probability_np,
            'response': response_str,
            'decoded_tokens': decoded_tokens,
        })

    # Check that the embedding lengths are consistent
    if len(processed) >= 2:
        guess_lens = [len(r['embeddings_guess']) for r in processed]
        prob_lens = [len(r['embeddings_probability']) for r in processed]
        if len(set(guess_lens)) != 1 or len(set(prob_lens)) != 1:
            print(
                f"Error: example {example_id} has inconsistent embedding lengths: "
                f"embeddings_guess lengths={guess_lens}, embeddings_probability lengths={prob_lens}"
            )
            sys.exit(1)

    if not processed:
        logging.debug(f"No processable responses for example {example_id} (all skipped or tuple/missing keys).")
        return None

    return {
        'question': example.get('question'),
        'context': example.get('context'),
        'responses': processed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process generations pickle for verbalised-confidence embedding subsets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to train_generations.pkl or validation_generations.pkl (dict items with all_embeddings, decoded_tokens)",
    )
    parser.add_argument(
        "--output_dir",
        default="./processed_generations",
        help="Output directory for pickle and JSON (default: current dir)",
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
    pickle_path = run_dir / f"{out_base}.pkl"
    json_path = run_dir / f"{out_base}.json"
    
    # Set up file logging to output.log in run directory (similar to generate_answers_with_confidence.py)
    output_log_path = run_dir / 'output.log'
    file_handler = logging.FileHandler(output_log_path, mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    logging.info(f'Files will be saved to {run_dir}')

    n_ok = 0
    n_reject = 0
    first_item = True
    example_index = 0

    pickle_file = open(pickle_path, "wb")
    json_file = open(json_path, "w")
    json_file.write("{\n")

    try:
        with open(input_path, "rb") as input_file:
            logging.info(f"Reading input file: {input_path}")
            batch_num = 0
            while True:
                try:
                    t0 = time.perf_counter()
                    input_batch = pickle.load(input_file)
                    load_elapsed = time.perf_counter() - t0
                    if not isinstance(input_batch, dict):
                        logging.warning(f"Skipping non-dict batch {batch_num}")
                        continue

                    batch_num += 1
                    logging.info(f"Processing input batch {batch_num} with {len(input_batch)} examples... (loaded in {load_elapsed:.2f}s)")

                    batch_pickle_result = {}
                    batch_json_result = {}

                    for example_id, example in input_batch.items():
                        out = process_example(
                            example_id,
                            example,
                            args.debug_first_n,
                            example_index,
                            prompt_on_parse_failure=not args.no_prompt,
                        )
                        if out is None:
                            n_reject += 1
                            example_index += 1
                            continue

                        n_ok += 1
                        example_index += 1

                        batch_pickle_result[example_id] = {
                            'responses': [
                                {
                                    'verbalised_confidence': r['verbalised_confidence'],
                                    'embeddings_guess': r['embeddings_guess'],
                                    'embeddings_probability': r['embeddings_probability'],
                                }
                                for r in out['responses']
                            ]
                        }
                        batch_json_result[example_id] = out

                    if batch_pickle_result:
                        pickle.dump(batch_pickle_result, pickle_file)
                        for eid, example_data in batch_json_result.items():
                            if not first_item:
                                json_file.write(",\n")
                            json_file.write(f'  "{eid}": ')
                            json_str = json.dumps(convert_for_json(example_data), indent=2)
                            indented = "\n".join(
                                "    " + line if line.strip() else line
                                for line in json_str.split("\n")
                            )
                            json_file.write(indented)
                            first_item = False
                        batch_pickle_result.clear()
                        batch_json_result.clear()

                except EOFError:
                    break

            if n_ok == 0:
                logging.error("No valid examples processed from pickle file")
                sys.exit(1)

    except Exception as e:
        logging.error(f"Error reading pickle file: {e}")
        sys.exit(1)

    pickle_file.close()
    json_file.write("\n}")
    json_file.close()
    logging.info(f"Wrote {pickle_path}")
    logging.info(f"Wrote {json_path}")

    samples_txt_path = run_dir / "samples.txt"
    with open(samples_txt_path, "w") as f:
        f.write(f"{n_ok} samples\n")
    logging.info(f"Wrote {samples_txt_path}")


if __name__ == "__main__":
    main()
