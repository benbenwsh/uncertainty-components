"""
Evaluate the trained verbalised-confidence linear probe on user questions (loop).

Runs in a loop: each iteration prompts for a new question, runs the LLM with the
verbalised-confidence prompt, collects embeddings via predict(..., return_latent=True),
runs the saved probe on the chosen layer's embedding, and prints predicted confidence in [0, 1].
Model and probe are loaded once; layer is chosen once. Enter a blank question to quit.

Usage (from repo root):
    python verbalised_confidence_probes/eval_verbalised_confidence_probe.py
    python verbalised_confidence_probes/eval_verbalised_confidence_probe.py --model_name Mistral-7B-Instruct-v0.1
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Allow importing from semantic_uncertainty/uncertainty when run from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEMANTIC_UNCERTAINTY = _REPO_ROOT / "semantic_uncertainty"
if str(_SEMANTIC_UNCERTAINTY) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_UNCERTAINTY))

from uncertainty.utils import utils

# Same as generate_answers_with_confidence.py (just_ask_for_calibration paper)
# TODO: maybe safe the confidence prompt somewhere in the outputs of generate_answers
CONFIDENCE_PROMPT = (
    "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. "
    "Give ONLY the guess and probability, no other words or explanation. For example:\n\n"
    "Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n "
    "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\n"
    "The question is: "
)


def _build_prompt(question: str) -> str:
    return CONFIDENCE_PROMPT + question


def _get_layer_embedding(emb_tok_bef_gen, layer_idx: int):
    """Extract flattened embedding for the given layer from emb_tok_bef_gen."""
    if hasattr(emb_tok_bef_gen, "cpu"):
        emb_array = emb_tok_bef_gen.cpu().numpy()
    elif isinstance(emb_tok_bef_gen, np.ndarray):
        emb_array = emb_tok_bef_gen
    else:
        emb_array = np.array(emb_tok_bef_gen)
    n_layers = emb_array.shape[0]
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(
            f"Layer index {layer_idx} out of range for embeddings with {n_layers} layers (0..{n_layers - 1})."
        )
    layer_emb = emb_array[layer_idx][0].flatten()
    return layer_emb


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate verbalised-confidence linear probe on questions (loop; blank to quit)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Mistral-7B-Instruct-v0.1",
        help="HuggingFace model name (same as used for training).",
    )
    parser.add_argument(
        "--model_max_new_tokens",
        type=int,
        default=50,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        required=True,
        help="Path to verbalised_confidence_probe.pkl",
    )
    args = parser.parse_args()

    args.probe_path = Path(args.probe_path)
    if not args.probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {args.probe_path}")

    # Load model once (confidence=True => HuggingfaceModelAllEmbeddings)
    model_args = argparse.Namespace(
        model_name=args.model_name,
        # TODO: is this correct? did i have this before
        model_max_new_tokens=args.model_max_new_tokens,
        confidence=True,
    )
    print("Loading model...")
    model = utils.init_model(model_args)

    # Load probe once
    with open(args.probe_path, "rb") as f:
        probe_data = pickle.load(f)
    probe_model = probe_data["model"]
    saved_layer_idx = probe_data.get("layer_idx")
    if saved_layer_idx is None:
        raise ValueError(f"Probe file {args.probe_path} has no 'layer_idx'; cannot default layer.")

    # Layer selection (once for all questions)
    layer_prompt = f"Layer index for embeddings (default from probe: {saved_layer_idx}) [{saved_layer_idx}]: "
    layer_str = input(layer_prompt).strip() or str(saved_layer_idx)
    layer_idx = int(layer_str)

    # Loop: prompt for question each time; blank to quit
    while True:
        question = input("\nEnter your question (blank to quit): ").strip()
        if not question:
            print("Quitting.")
            break

        full_prompt = _build_prompt(question)
        print("Running LLM (temperature=0.1, return_latent=True)...")
        answer, log_likelihoods, hidden_states, decoded_tokens = model.predict(
            full_prompt, temperature=0.1, return_latent=True
        )
        emb_sec_last_token, emb_tok_bef_gen, all_embeddings = hidden_states

        print("\n--- LLM response ---")
        print(answer)
        print("--- end response ---")

        layer_emb = _get_layer_embedding(emb_tok_bef_gen, layer_idx)
        raw_pred = probe_model.predict(layer_emb.reshape(1, -1))[0]
        confidence = float(np.clip(raw_pred, 0.0, 1.0))
        print(f"Probe predicted confidence (tok_bef_gen) (layer {layer_idx}): {confidence:.4f}")


if __name__ == "__main__":
    main()
