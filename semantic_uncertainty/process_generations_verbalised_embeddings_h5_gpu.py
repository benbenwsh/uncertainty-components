"""
GPU variant of process_generations_verbalised_embeddings_h5.py.

Reads native HDF5 examples, parses verbalised confidence format, and slices
embedding subsets on GPU (when available). Writes processed native HDF5 + JSON
+ samples.txt into processed_generations_h5.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    from semantic_uncertainty.process_generations_verbalised_embeddings_h5 import (
        REJECT_KEYWORD,
        _print_decoded_tokens_neatly,
        _write_h5_node,
        convert_for_json,
        load_examples_h5,
        normalise_response_item,
        parse_guess_and_probability_indices,
        prompt_user_on_structure_failure,
    )
except ImportError:
    from process_generations_verbalised_embeddings_h5 import (
        REJECT_KEYWORD,
        _print_decoded_tokens_neatly,
        _write_h5_node,
        convert_for_json,
        load_examples_h5,
        normalise_response_item,
        parse_guess_and_probability_indices,
        prompt_user_on_structure_failure,
    )

from process_generations_tok_bef_gen import parse_probability_from_response
from uncertainty.utils import utils

utils.setup_logger()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _embedding_to_numpy_gpu(embedding, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    """Move one embedding to device and return numpy output."""
    arr = np.asarray(embedding)
    tensor = torch.as_tensor(arr, device=device)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().to("cpu", dtype=tensor.dtype).numpy()


def process_example_gpu(
    example_id,
    example: dict,
    debug_first_n: int,
    example_index: int,
    prompt_on_parse_failure: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict | None:
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
            logging.debug(
                f"Skipping item (tuple or missing all_embeddings/decoded_tokens) in example {example_id}"
            )
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
            user_val = prompt_user_on_structure_failure(
                example_id, full_str, decoded_tokens, response_str
            )
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
            user_val = prompt_user_on_structure_failure(
                example_id, full_str, decoded_tokens, response_str
            )
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

        # GPU-accelerated conversion path.
        embeddings_guess_np = [_embedding_to_numpy_gpu(e, device, dtype) for e in embeddings_guess]
        embeddings_probability_np = [
            _embedding_to_numpy_gpu(e, device, dtype) for e in embeddings_probability
        ]

        if example_index < debug_first_n:
            logging.debug(
                f"[{example_id}] guess_len={len(embeddings_guess_np)} prob_len={len(embeddings_probability_np)}"
            )
            if embeddings_guess_np:
                logging.debug(
                    f"[{example_id}] embeddings_guess[0].shape={embeddings_guess_np[0].shape}"
                )
            if embeddings_probability_np:
                logging.debug(
                    f"[{example_id}] embeddings_probability[0].shape={embeddings_probability_np[0].shape}"
                )

        processed.append(
            {
                "verbalised_confidence": float(prob),
                "embeddings_guess": embeddings_guess_np,
                "embeddings_probability": embeddings_probability_np,
                "response": response_str,
                "decoded_tokens": decoded_tokens,
            }
        )

    if len(processed) >= 2:
        guess_lens = [len(r["embeddings_guess"]) for r in processed]
        prob_lens = [len(r["embeddings_probability"]) for r in processed]
        if len(set(guess_lens)) != 1 or len(set(prob_lens)) != 1:
            print(
                f"Error: example {example_id} has inconsistent embedding lengths: "
                f"embeddings_guess lengths={guess_lens}, embeddings_probability lengths={prob_lens}"
            )
            sys.exit(1)

    if not processed:
        return None

    return {
        "question": example.get("question"),
        "context": example.get("context"),
        "responses": processed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process generations HDF5 with GPU-accelerated embedding conversion."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="./processed_generations_h5")
    parser.add_argument("--output_suffix", default="_verbalised_embeddings")
    parser.add_argument("--debug_first_n", type=int, default=3)
    parser.add_argument("--no_prompt", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for embedding conversion.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Floating dtype used on device for embeddings.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    logging.info("Using device=%s dtype=%s", device, dtype)

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    stem = input_path.stem
    if not stem.endswith("_generations"):
        raise ValueError(f"Input file does not end with '_generations': {input_path}")
    base = stem.replace("_generations", "")

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
    logging.info(f"Files will be saved to {run_dir}")

    n_ok = 0
    first_item = True
    example_index = 0
    json_file = open(json_path, "w")
    json_file.write("{\n")

    try:
        with h5py.File(h5_path, "a") as out_h5:
            out_examples = out_h5["examples"]
            for example_id, example in load_examples_h5(input_path):
                t0 = time.perf_counter()
                out = process_example_gpu(
                    example_id=example_id,
                    example=example,
                    debug_first_n=args.debug_first_n,
                    example_index=example_index,
                    prompt_on_parse_failure=not args.no_prompt,
                    device=device,
                    dtype=dtype,
                )
                example_index += 1
                if out is None:
                    continue

                n_ok += 1
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

                if (n_ok % 100) == 0:
                    logging.info("Processed %d examples (last %.3fs)", n_ok, time.perf_counter() - t0)

        if n_ok == 0:
            logging.error("No valid examples processed from HDF5 file")
            sys.exit(1)
    except Exception as exc:
        logging.error("Error processing HDF5 file: %s", exc)
        sys.exit(1)
    finally:
        json_file.write("\n}")
        json_file.close()

    samples_txt_path = run_dir / "samples.txt"
    with open(samples_txt_path, "w") as f:
        f.write(f"{n_ok} samples\n")

    logging.info(f"Wrote {h5_path}")
    logging.info(f"Wrote {json_path}")
    logging.info(f"Wrote {samples_txt_path}")


if __name__ == "__main__":
    main()
