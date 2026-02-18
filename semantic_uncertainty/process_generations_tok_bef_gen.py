"""
Process train_generations.pkl or validation_generations.pkl to extract emb_tok_bef_gen embeddings.

Reads a single generations pickle, extracts verbalised confidence (probability in [0,1])
from each response string via rule-based parsing, supports interactive fallback and
rejecting examples on parse failure. Writes JSON + pickle with emb_tok_bef_gen embeddings
for probe training.

Usage:
  python -m semantic_uncertainty.process_generations_tok_bef_gen --input train_generations.pkl
  python -m semantic_uncertainty.process_generations_tok_bef_gen --input validation_generations.pkl --output_dir ./out

Output:
  - Pickle: full data (embeddings as numpy arrays). Use this for training.
  - JSON: same structure with arrays as nested lists (for inspection); may be large.
"""

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

REJECT_KEYWORD = 'reject'


def parse_probability_from_response(response_str: str) -> float | None:
    """
    Extract probability in [0, 1] from a response string.

    Expected format: 'Guess: <...>\\nProbability:<number>' (case-insensitive, flexible whitespace).
    Uses the **last** occurrence of "probability:" so that if the guess text contains
    "Probability: ...", we still take the final number. Allows . or , as decimal separator.
    If number > 1 (e.g. 80 or 80%), treats as percentage. Returns None if parsing fails.
    """
    if not response_str or not isinstance(response_str, str):
        return None
    # Case-insensitive: use last occurrence of "probability:" then take the next number
    matches = list(re.finditer(r'probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?', response_str, re.IGNORECASE))
    if not matches:
        matches = list(re.finditer(r'probability\s*:\s*(\d+(?:[.,]\d+)?)', response_str, re.IGNORECASE))
    if not matches:
        return None
    match = matches[-1]
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
    Dict: response, emb_tok_bef_gen, decoded_tokens.

    Returns dict with keys: response, emb_tok_bef_gen, decoded_tokens.
    Returns None if item is invalid.
    """
    if isinstance(item, dict):
        return {
            'response': item.get('response'),
            'emb_tok_bef_gen': item.get('emb_tok_bef_gen'),
            'decoded_tokens': item.get('decoded_tokens'),
        }
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return {
            'response': item[0],
            'emb_tok_bef_gen': None,
            'decoded_tokens': None,
        }
    return None


def prompt_user_for_probability(response_str: str, example_id) -> float | str:
    """
    Prompt user to enter probability or reject the example.

    Returns float in [0,1] or REJECT_KEYWORD if user rejects the example.
    """
    print(f"\n[Example id: {example_id}] Could not parse probability from response:")
    print("-" * 60)
    # TODO: maybe add question in here in the future
    print(response_str)
    print("-" * 60)
    line = input(f"Enter a number in [0, 1] or '{REJECT_KEYWORD}' to exclude this example: ")
    if line == REJECT_KEYWORD:
        return REJECT_KEYWORD
    try:
        value = float(line.replace(",", "."))
        if 0 <= value <= 1:
            return value
    except ValueError:
        pass
    print("Invalid input. Enter a number in [0, 1] or 'reject'.")
    return prompt_user_for_probability(response_str, example_id)


# Keys whose values are embeddings: show only first 5 values in JSON for readability.
_EMBEDDING_KEYS = frozenset({'emb_tok_bef_gen'})


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

# Only for visualisation in json
def _first_and_last_layer_values(obj, n: int = 5):
    """
    Extract first n values from first layer [0] and last layer [-1] of embeddings.
    Returns dict with 'first_layer' and 'last_layer' keys.
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(obj, np.ndarray):
            if hasattr(obj, 'numpy'):
                arr = obj.numpy()
            elif hasattr(obj, 'cpu'):
                arr = obj.cpu().numpy()
            else:
                arr = np.asarray(obj)
        else:
            arr = obj
        
        # Handle different shapes: [layers, ...] or [layers, batch, hidden_dim] etc.
        if arr.ndim == 0:
            # Scalar - return as is
            return {'first_layer': float(arr), 'last_layer': float(arr)}
        
        # Get first layer [0] and last layer [-1]
        first_layer = arr[0] if arr.shape[0] > 0 else arr
        last_layer = arr[-1] if arr.shape[0] > 0 else arr
        
        # Flatten and take first n values
        first_flat = np.asarray(first_layer).ravel()[:n].tolist()
        last_flat = np.asarray(last_layer).ravel()[:n].tolist()
        
        return {
            'first_layer': first_flat,
            'last_layer': last_flat
        }
    except (IndexError, AttributeError, TypeError, ValueError) as e:
        # Fallback to original behavior if extraction fails
        return {'first_layer': _first_n_values(obj, n), 'last_layer': _first_n_values(obj, n)}


def convert_for_json(obj, parent_key=None):
    """Convert numpy/torch arrays to lists for JSON; for embedding keys show first 5 values from first and last layers."""
    is_embedding = parent_key in _EMBEDDING_KEYS
    if isinstance(obj, np.ndarray):
        return _first_and_last_layer_values(obj, 5) if is_embedding else obj.tolist()
    if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            if is_embedding:
                return _first_and_last_layer_values(obj, 5)
            lst = obj.tolist()
            return lst
        except (TypeError, ValueError):
            # If tolist fails but it's an embedding, try direct extraction
            if is_embedding:
                return _first_and_last_layer_values(obj, 5)
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


def process_example(example_id, example: dict) -> dict | None:
    """
    Process one example: normalise response items, parse probability for each.

    On parse failure, prompts user; if user rejects, returns None.
    Returns dict with question, context, responses (list of processed items) or None if rejected.
    Note: pickle output will only include verbalised_confidence, emb_tok_bef_gen.
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
            user_val = prompt_user_for_probability(response_str, example_id)
            if user_val == REJECT_KEYWORD:
                return None
            prob = user_val
        processed.append({
            'verbalised_confidence': float(prob),
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
        description="Process generations pickle to extract emb_tok_bef_gen embeddings (extract verbalised confidence)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to train_generations.pkl or validation_generations.pkl",
    )
    parser.add_argument(
        "--output_dir",
        default="./processed_generations",
        help="Output directory for pickle and JSON (default: current dir)",
    )
    parser.add_argument(
        "--output_suffix",
        default="_linear_probe",
        help="Suffix for output filenames (default: _linear_probe)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
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
    first_item = True  # Track first JSON entry for comma handling
    
    # Open output files for streaming writes
    # Note: Pickle output will contain appended batch dicts (for RAM efficiency).
    # To read: use the same streaming logic as input reading (read batches and merge).
    pickle_file = open(pickle_path, "wb")
    json_file = open(json_path, "w")
    json_file.write("{\n")  # Start JSON object
    
    # Stream read and process pickle file batch by batch
    try:
        with open(input_path, "rb") as input_file:
            print(f"Reading input file: {input_path}")
            batch_num = 0
            while True:
                try:
                    # Read one batch from input file
                    t0 = time.perf_counter()
                    input_batch = pickle.load(input_file)
                    load_elapsed = time.perf_counter() - t0
                    if not isinstance(input_batch, dict):
                        print(f"Warning: skipping non-dict batch {batch_num}")
                        continue
                    
                    batch_num += 1
                    print(f"Processing input batch {batch_num} with {len(input_batch)} examples... (loaded in {load_elapsed:.2f}s)")
                    
                    # Process each example in this batch immediately
                    batch_pickle_result = {}
                    batch_json_result = {}
                    
                    for example_id, example in input_batch.items():
                        out = process_example(example_id, example)
                        if out is None:
                            n_reject += 1
                            continue
                        
                        n_ok += 1
                        
                        # Add to batch for pickle (minimal data)
                        batch_pickle_result[example_id] = {
                            'responses': [
                                {
                                    'verbalised_confidence': r['verbalised_confidence'],
                                    'emb_tok_bef_gen': r['emb_tok_bef_gen'],
                                }
                                for r in out['responses']
                            ]
                        }
                        # Add to batch for JSON (full data)
                        batch_json_result[example_id] = out
                    
                    # Write processed batch immediately to output files
                    if batch_pickle_result:
                        # Append batch to pickle file
                        pickle.dump(batch_pickle_result, pickle_file)
                        
                        # Write batch to JSON file (with proper comma handling)
                        for eid, example_data in batch_json_result.items():
                            if not first_item:
                                json_file.write(",\n")
                            json_file.write(f'  "{eid}": ')
                            # Convert to JSON string with proper indentation
                            json_str = json.dumps(convert_for_json(example_data), indent=2)
                            # Add extra indentation for nested content
                            indented = "\n".join("    " + line if line.strip() else line 
                                                 for line in json_str.split("\n"))
                            json_file.write(indented)
                            first_item = False
                        
                        # Clear batches from memory immediately
                        batch_pickle_result.clear()
                        batch_json_result.clear()
                        
                except EOFError:
                    # End of file - no more batches
                    break
            
            if n_ok == 0:
                print("Error: no valid examples processed from pickle file")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        sys.exit(1)
    
    # Close files
    pickle_file.close()
    json_file.write("\n}")  # Close JSON object
    json_file.close()
    print(f"Wrote {pickle_path}")
    print(f"Wrote {json_path}")

    # Write sample count to a txt file
    samples_txt_path = run_dir / "samples.txt"
    with open(samples_txt_path, "w") as f:
        f.write(f"{n_ok} samples\n")
    print(f"Wrote {samples_txt_path}")


if __name__ == "__main__":
    main()
