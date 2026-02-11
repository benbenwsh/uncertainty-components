"""
Process train_generations.pkl or validation_generations.pkl for linear probe training.

Reads a single generations pickle, extracts verbalised confidence (probability in [0,1])
from each response string via rule-based parsing, supports interactive fallback and
rejecting examples on parse failure. Writes JSON + pickle in a linear-probe-friendly format.

Usage:
  python -m semantic_uncertainty.process_generations_for_linear_probe --input train_generations.pkl
  python -m semantic_uncertainty.process_generations_for_linear_probe --input validation_generations.pkl --output_dir ./out

Output:
  - Pickle: full data (embeddings as numpy arrays). Use this for training.
  - JSON: same structure with arrays as nested lists (for inspection); may be large.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REJECT_KEYWORD = 'reject'


def parse_probability_from_response(response_str: str) -> float | None:
    """
    Extract probability in [0, 1] from a response string.

    Expected format: 'Guess: <...>\\nProbability:<number>' (case-insensitive, flexible whitespace).
    Allows . or , as decimal separator. If number > 1 (e.g. 80 or 80%), treats as percentage.
    Returns None if parsing fails.
    """
    if not response_str or not isinstance(response_str, str):
        return None
    # Case-insensitive find "probability:" then take the next number
    match = re.search(r'probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?', response_str, re.IGNORECASE)
    if not match:
        # Fallback: any number in [0,1] or percentage-like after "probability"
        match = re.search(r'probability\s*:\s*(\d+(?:[.,]\d+)?)', response_str, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip().replace(',', '.')
    try:
        value = float(raw)
    except ValueError:
        return None
    # Percentage: e.g. 80 -> 0.8
    if value > 1:
        value = value / 100.0
    if value < 0 or value > 1:
        return None
    return value


def normalise_response_item(item) -> dict | None:
    """
    Normalise a response item (tuple or dict) to a common shape.

    Tuple: (response_str, token_log_likelihoods, embedding, acc) -> embedding ignored; others None.
    Dict: response, emb_sec_last_token, emb_tok_bef_gen, decoded_tokens.

    Returns dict with keys: response, emb_sec_last_token, emb_tok_bef_gen, decoded_tokens.
    Returns None if item is invalid.
    """
    if isinstance(item, dict):
        return {
            'response': item.get('response'),
            'emb_sec_last_token': item.get('emb_sec_last_token'),
            'emb_tok_bef_gen': item.get('emb_tok_bef_gen'),
            'decoded_tokens': item.get('decoded_tokens'),
        }
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return {
            'response': item[0],
            'emb_sec_last_token': None,
            'emb_tok_bef_gen': None,
            'decoded_tokens': None,
        }
    return None


def prompt_user_for_probability(response_str: str, example_id) -> float | str:
    """
    Prompt user to enter probability or reject the example.

    Returns float in [0,1] or REJECT_KEYWORD if user rejects the example.
    """
    print(f"\n[Example id: {example_id}] Could not parse probability from response:", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(response_str, file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(
        f"Enter a number in [0, 1] or '{REJECT_KEYWORD}' to exclude this example: ",
        end="",
        file=sys.stderr,
    )
    line = sys.stdin.readline().strip().lower()
    if line == REJECT_KEYWORD:
        return REJECT_KEYWORD
    try:
        value = float(line.replace(",", "."))
        if 0 <= value <= 1:
            return value
    except ValueError:
        pass
    print("Invalid input. Enter a number in [0, 1] or 'reject'.", file=sys.stderr)
    return prompt_user_for_probability(response_str, example_id)


# Keys whose values are embeddings: show only first 5 values in JSON for readability.
_EMBEDDING_KEYS = frozenset({'emb_sec_last_token', 'emb_tok_bef_gen'})


def _first_n_values(obj, n: int = 5):
    """Flatten array-like to 1D and return first n values (for JSON readability)."""
    if isinstance(obj, np.ndarray):
        flat = obj.ravel()
        return flat[:n].tolist()
    if hasattr(obj, 'tolist') and hasattr(obj, 'reshape'):
        try:
            flat = np.asarray(obj).ravel()
            return flat[:n].tolist()
        except (TypeError, ValueError):
            pass
    if hasattr(obj, 'tolist'):
        try:
            lst = obj.tolist()
        except (TypeError, ValueError):
            return obj
    else:
        lst = obj
    # Recursive flatten for nested lists
    out = []
    def flatten(x):
        if isinstance(x, (int, float)):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for e in x:
                flatten(e)
    flatten(lst)
    return out[:n]


def convert_for_json(obj, parent_key=None):
    """Convert numpy/torch arrays to lists for JSON; for embedding keys show only first 5 values."""
    is_embedding = parent_key in _EMBEDDING_KEYS
    if isinstance(obj, np.ndarray):
        return _first_n_values(obj, 5) if is_embedding else obj.tolist()
    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            lst = obj.tolist()
            if is_embedding:
                return _first_n_values(obj, 5)
            return lst
        except (TypeError, ValueError):
            pass
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v, parent_key=k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(x, parent_key) for x in obj]
    return obj


def process_example(example_id, example: dict, *, dry_run: bool) -> dict | None:
    """
    Process one example: normalise response items, parse probability for each.

    On parse failure, if not dry_run, prompts user; if user rejects, returns None.
    Returns dict with question, context, responses (list of processed items) or None if rejected.
    Note: pickle output will only include verbalised_confidence, emb_sec_last_token, emb_tok_bef_gen.
    JSON output includes all fields (question, context, response, decoded_tokens) with truncated embeddings.
    """
    most_likely = example.get('most_likely_answer')
    responses_list = example.get('responses') or []
    # Build list: most_likely first, then each entry in responses
    items = []
    if most_likely is not None:
        items.append(most_likely)
    items.extend(responses_list)

    processed = []
    for item in items:
        norm = normalise_response_item(item)
        if norm is None:
            continue
        response_str = norm.get('response')
        if response_str is None:
            continue
        prob = parse_probability_from_response(response_str)
        if prob is None:
            if dry_run:
                processed.append({
                    'verbalised_confidence': None,
                    'emb_sec_last_token': norm.get('emb_sec_last_token'),
                    'emb_tok_bef_gen': norm.get('emb_tok_bef_gen'),
                    'response': response_str,
                    'decoded_tokens': norm.get('decoded_tokens'),
                })
                continue
            user_val = prompt_user_for_probability(response_str, example_id)
            if user_val == REJECT_KEYWORD:
                return None
            prob = user_val
        processed.append({
            'verbalised_confidence': float(prob),
            'emb_sec_last_token': norm.get('emb_sec_last_token'),
            'emb_tok_bef_gen': norm.get('emb_tok_bef_gen'),
            'response': response_str,
            'decoded_tokens': norm.get('decoded_tokens'),
        })

    return {
        'question': example.get('question'),
        'context': example.get('context'),
        'responses': processed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process generations pickle for linear probe (extract verbalised confidence)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to train_generations.pkl or validation_generations.pkl",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for pickle and JSON (default: current dir)",
    )
    parser.add_argument(
        "--output_suffix",
        default="_linear_probe",
        help="Suffix for output filenames (default: _linear_probe)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only run parser on all examples; print stats; do not write files or prompt",
    )
    parser.add_argument(
        "--dry_run_n",
        type=int,
        default=None,
        help="If set with --dry_run, only process first N examples",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "rb") as f:
        generations = pickle.load(f)

    if not isinstance(generations, dict):
        print("Error: expected generations to be a dict keyed by example id", file=sys.stderr)
        sys.exit(1)

    # Derive base name: train_generations.pkl -> train, validation_generations.pkl -> validation
    stem = input_path.stem  # e.g. "train_generations" or "validation_generations"
    if stem.endswith("_generations"):
        base = stem.replace("_generations", "")
    else:
        base = stem
    out_base = f"{base}{args.output_suffix}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use a new numbered subdir each run (1, 2, 3, ...)
    existing = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_number = max((int(n) for n in existing), default=0) + 1
    run_dir = output_dir / str(run_number)
    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / f"{out_base}.pkl"
    json_path = run_dir / f"{out_base}.json"

    n_ok = 0
    n_reject = 0
    n_parse_fail = 0
    limit = args.dry_run_n if args.dry_run and args.dry_run_n is not None else None
    result = {}
    keys = list(generations.keys())
    if limit is not None:
        keys = keys[:limit]

    for example_id in keys:
        example = generations[example_id]
        out = process_example(example_id, example, dry_run=args.dry_run)
        if out is None:
            n_reject += 1
            continue
        # Count parse failures in this example (dry run only)
        if args.dry_run:
            for r in out["responses"]:
                if r.get("verbalised_confidence") is None:
                    n_parse_fail += 1
        n_ok += 1
        result[example_id] = out

    if args.dry_run:
        print(
            f"Dry run: {n_ok} examples kept, {n_reject} rejected, "
            f"{n_parse_fail} response(s) with parse failure (would prompt)."
        )
        return

    # Create minimal version for pickle: only verbalised_confidence, emb_sec_last_token, emb_tok_bef_gen
    pickle_result = {}
    for example_id, example_data in result.items():
        pickle_result[example_id] = {
            'responses': [
                {
                    'verbalised_confidence': r['verbalised_confidence'],
                    'emb_sec_last_token': r['emb_sec_last_token'],
                    'emb_tok_bef_gen': r['emb_tok_bef_gen'],
                }
                for r in example_data['responses']
            ]
        }

    # Save pickle (minimal data: only embeddings and confidence)
    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_result, f)
    print(f"Wrote {pickle_path}")

    # Save JSON (full data with truncated embeddings)
    jsonable = convert_for_json(result)
    with open(json_path, "w") as f:
        json.dump(jsonable, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
