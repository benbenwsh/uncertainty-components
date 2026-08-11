#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens (see repo sep_enviroment.yaml pip list).
# Install / update env: conda env update -f sep_enviroment.yaml
"""
Greedy decoding on TriviaQA with layerwise zero ablation via TransformerLens.

Does not import semantic_uncertainty.utils or huggingface_models wrappers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import logging
import os
import random
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.weight_processing import ProcessWeights

# ---------------------------------------------------------------------------
# Prompts (match generate_answers_with_confidence.py / utils defaults)
# ---------------------------------------------------------------------------

CONFIDENCE_PROMPT = (
    "Provide your best guess and the probability that it is correct (0.0 to 1.0) "
    "for the following question. Give ONLY the guess and probability, no other words "
    "or explanation. For example:\n\n"
    "Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n "
    "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any "
    "extra commentary whatsoever; just the probability!>\n\n"
    "The question is: "
)

BRIEF_PROMPTS = {
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "<end_of_turn>"]

# ---------------------------------------------------------------------------
# Probability span (match process_generations_verbalised_embeddings_h5.py)
# ---------------------------------------------------------------------------

# Gemma-3 token alternatives (fixed length; each inner list = allowed tokens at that position)
GUESS_PREFIX_TOKENS = [
    ["\n", "\n\n"],
    ["Guess"],
    [":"],
]
PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    ["Probability", " Probability"],
    [":"],
    [" "],
]


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


def parse_guess_and_probability_indices(
    decoded_tokens: list,
) -> tuple[int, int, int] | None:
    """
    Compute token indices for the two embedding subsets (Guess and Probability).

    last_guess_token_index: token after "Guess:" (first token of answer)
    first_prob_token_index: "\n" token before "Probability:"
    end_prob_token_index: token after ``Probability:`` + whitespace (first token of prob value)

    Returns (last_guess_token_index, first_prob_token_index, end_prob_token_index)
    or None on failure.
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
        or end_prob_token_index > len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None

    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def parse_probability_from_response(response_str: str) -> float | None:
    """Extract probability in [0, 1] from a response string (first ``probability:`` match)."""
    if not response_str or not isinstance(response_str, str):
        return None
    matches = list(re.finditer(r"probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?", response_str, re.IGNORECASE))
    if not matches:
        matches = list(re.finditer(r"probability\s*:\s*(\d+(?:[.,]\d+)?)", response_str, re.IGNORECASE))
    if not matches:
        return None
    raw = matches[0].group(1).strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0 or value > 1:
        return None
    return value


# ---------------------------------------------------------------------------
# Dataset + few-shot (match data_utils.py + utils.py defaults)
# ---------------------------------------------------------------------------


def load_eval_dataset(dataset_name: str, seed: int):
    """Load train/validation splits via semantic_uncertainty.data_utils.load_ds."""
    sem_unc_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "semantic_uncertainty")
    if sem_unc_root not in sys.path:
        sys.path.insert(0, sem_unc_root)
    from uncertainty.data.data_utils import load_ds

    train_ds, val_ds = load_ds(dataset_name, seed=seed)
    if train_ds is None or val_ds is None:
        raise ValueError(f"Unsupported or failed dataset load: {dataset_name}")
    return train_ds, val_ds


def split_answerable_indices(dataset: Dataset) -> List[int]:
    def clen(ex):
        return len(ex["answers"]["text"])

    return [i for i, ex in enumerate(dataset) if clen(ex) > 0]


def construct_fewshot_prompt_from_indices(
    dataset: Dataset,
    example_indices: Sequence[int],
    brief: str,
    brief_always: bool,
    use_context: bool,
) -> str:
    """Mirror utils.construct_fewshot_prompt_from_indices + default make_prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ""

    for example_index in example_indices:
        example = dataset[int(example_index)]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        piece = ""
        if brief_always:
            piece += brief
        if use_context and (context is not None):
            piece += f"Context: {context}\n"
        piece += f"Question: {question}\n"
        if answer:
            piece += f"Answer: {answer}\n\n"
        else:
            piece += "Answer:"
        prompt = prompt + piece

    return prompt


def encode_example_id(example_id) -> str:
    return quote(str(example_id), safe="")


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    """Return user-specified output path or create incremental default run path."""
    if cli_output_path:
        return cli_output_path

    base_dir = os.path.join("layerwise_zero_ablation", "results")
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_dir = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "ablation_results.json")


def mini_output_json_path(full_output_path: str) -> str:
    """Return sibling path for compact JSON output."""
    out_dir = os.path.dirname(full_output_path)
    return os.path.join(out_dir, "ablation_results_mini.json")


def config_txt_path(full_output_path: str) -> str:
    """Return sibling path for config output."""
    out_dir = os.path.dirname(full_output_path)
    return os.path.join(out_dir, "config.txt")


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    ablate_layers: Sequence[int],
    prompt_indices: Sequence[int],
    mode_confidence_means: Dict[str, Optional[float]],
    mode_confidence_counts: Dict[str, int],
    finished_at: str,
) -> None:
    lines = [
        "Layerwise Zero Ablation Configuration",
        "====================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        "",
        "[Data]",
        f"dataset={args.dataset}",
        f"random_seed={args.random_seed}",
        f"num_samples={args.num_samples}",
        f"num_few_shot={args.num_few_shot}",
        f"fewshot_prompt_indices={','.join(str(i) for i in prompt_indices)}",
        "",
        "[Prompt/Generation]",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"brief_prompt={args.brief_prompt}",
        f"enable_brief={args.enable_brief}",
        f"brief_always={args.brief_always}",
        f"use_context={args.use_context}",
        "",
        "[Ablation]",
        f"ablation_mode={args.ablation_mode}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        "",
        "[Summary: Mean Parsed Verbalised Confidence]",
    ]
    for out_key in sorted(mode_confidence_means.keys()):
        mode_mean = mode_confidence_means[out_key]
        count_val = int(mode_confidence_counts.get(out_key, 0))
        if mode_mean is None:
            lines.append(f"{out_key}=None ({count_val})")
        else:
            lines.append(f"{out_key}={mode_mean:.6f} ({count_val})")
    lines.extend(
        [
            "",
            "[Run]",
            f"finished_at={finished_at}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@contextmanager
def _without_tl_noop_fp32_upcast():
    """Skip TL's full-state-dict fp32 upcast when no weight processing is requested."""
    original = ProcessWeights.__dict__["process_weights"]
    original_fn = original.__func__ if isinstance(original, staticmethod) else original

    def _process_weights(
        state_dict,
        cfg,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
        adapter=None,
    ):
        if not (
            fold_ln
            or center_writing_weights
            or center_unembed
            or fold_value_biases
            or refactor_factored_attn_matrices
        ):
            return state_dict
        return original_fn(
            state_dict,
            cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            adapter=adapter,
        )

    ProcessWeights.process_weights = staticmethod(_process_weights)
    try:
        yield
    finally:
        ProcessWeights.process_weights = original


def load_hooked_transformer(
    model_name: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> HookedTransformer:
    """Load TransformerLens model with a robust tokenizer fallback."""
    def _from_pretrained(**extra_kwargs):
        with _without_tl_noop_fp32_upcast():
            return HookedTransformer.from_pretrained_no_processing(
                model_name,
                device=device,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )

    try:
        model = _from_pretrained()
    except Exception as exc:
        # Some environments fail to parse certain fast tokenizer JSON files.
        # Fallback to a slow tokenizer and pass it explicitly.
        if "PyPreTokenizerTypeWrapper" not in str(exc):
            raise
        logging.warning(
            "Fast tokenizer load failed (%s). Retrying with use_fast=False.",
            exc,
        )
        hf_token = os.environ.get("HF_TOKEN", None)
        slow_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_bos_token=True,
            trust_remote_code=True,
            use_fast=False,
            token=hf_token,
        )
        model = _from_pretrained(tokenizer=slow_tokenizer)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------
# Layer list + hooks
# ---------------------------------------------------------------------------


def parse_ablate_layers(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        layers = list(range(int(a.strip()), int(b.strip()) + 1))
    else:
        layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for L in layers:
        if L < 0 or L >= n_layers:
            raise ValueError(f"Layer index {L} out of range [0, {n_layers}) for this model.")
    return layers


def _completion_token_index_to_abs_pos(prompt_len: int, completion_index: int) -> int:
    """Map completion-relative token index (0 = first generated token) to full-sequence position."""
    return prompt_len + completion_index - 1


def _absolute_prob_positions(
    prompt_len: int,
    decoded_tokens: List[str],
) -> List[int]:
    """Convert the probability-prefix token span into absolute sequence positions by adding the prompt length."""
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    # If the LLM output does not abide by the format, it returns an empty list.
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    seq_len = prompt_len + len(decoded_tokens)
    out = []
    for k in range(first_prob, end_prob):
        p = _completion_token_index_to_abs_pos(prompt_len, k)
        if 0 <= p < seq_len:
            out.append(p)
    return out


def build_resid_post_hooks(
    model: HookedTransformer,
    layer_indices: Sequence[int],
    ablation_scope: str,
    prompt_len: int,
    seq_len_provider: Callable[[], int], # Function that returns the current seq_len
    decoded_tokens_provider: Callable[[], List[str]], # Function that returns the current decoded tokens
) -> List[Tuple[str, Callable]]:
    """
    ablation_scope: 'all' or 'probability_tokens'
    """

    def positions_to_zero() -> List[int]:
        sl = seq_len_provider() # I think that the seq_len will increase as it generates new tokens (as it appends to the tokens list)
        if ablation_scope == "all":
            return list(range(sl))
        # probability_tokens
        dt = decoded_tokens_provider()
        abs_prob_positions = _absolute_prob_positions(prompt_len, dt)
        prob_tokens = [
            dt[p - prompt_len + 1]
            for p in abs_prob_positions
            if 0 <= p - prompt_len + 1 < len(dt)
        ]
        print(f"Decoded tokens at abs probability positions: {prob_tokens}")

        return abs_prob_positions

    hooks: List[Tuple[str, Callable]] = []

    for L in layer_indices:
        hook_name = f"blocks.{L}.hook_resid_post"

        def _make_hook(layer: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                # Shape of activation: e.g. [1, 133, 4096] ([batch_size, seq_len, hidden_dim])
                del hook
                positions = positions_to_zero()
                print(f"Positions to zero: {positions} for layer {layer}")
                if not positions:
                    return activation
                for p in positions:
                    if 0 <= p < activation.shape[1]:
                        activation[:, p, :] = 0 # All individual elements of this tensor become 0
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(L)))

    return hooks


def _generation_contains_stop(decoded_completion: str) -> bool:
    return any(s in decoded_completion for s in STOP_SEQUENCES)


def _eos_token_ids(tokenizer) -> set[int]:
    """Normalize tokenizer EOS ids (scalar or list) into a set of ints."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return set()
    if isinstance(eos_id, (list, tuple, set)):
        return {int(x) for x in eos_id if x is not None}
    return {int(eos_id)}


def _should_stop_generation(decoded_completion: str, next_id: int, tokenizer) -> bool:
    if _generation_contains_stop(decoded_completion):
        return True
    return next_id in _eos_token_ids(tokenizer)


def _strip_stop_suffixes(text: str) -> str:
    stop_at = len(text)
    for stop in STOP_SEQUENCES:
        if text.endswith(stop):
            stop_at = len(text) - len(stop)
            break
    sliced = text[:stop_at]
    if not all(s not in sliced for s in STOP_SEQUENCES):
        logging.warning("Stop sequences still present inside sliced completion; returning best-effort slice.")
    return sliced.strip()


def _postprocess_response_from_full_decode(
    model: HookedTransformer,
    full_sequence_tokens: torch.Tensor,
    input_text: str,
) -> str:
    """Decode full sequence then remove prompt prefix, mirroring HF behavior."""
    full_answer = model.tokenizer.decode(full_sequence_tokens[0], skip_special_tokens=True)
    if full_answer.startswith(input_text):
        answer = full_answer[len(input_text):]
    else:
        logging.warning("Decoded text does not start with prompt; using full decoded text.")
        answer = full_answer
    return _strip_stop_suffixes(answer)


def greedy_generate_ablated(
    model: HookedTransformer,
    local_prompt: str,
    ablation_scope: str,
    ablate_layers: Sequence[int],
    max_new_tokens: int,
) -> Tuple[str, List[str]]:
    """
    Returns (response_stripped, decoded_tokens for generated span only).
    """
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    generated: List[int] = []
    decoded_tokens: List[str] = []


    def seq_len() -> int:
        return int(tokens.shape[1])

    def dt_provider() -> List[str]:
        return decoded_tokens

    # Hooks that zero-ablate the selected layers at the selected tok positions
    fwd_hooks = build_resid_post_hooks(
        model,
        ablate_layers,
        ablation_scope,
        prompt_len,
        seq_len_provider=seq_len,
        decoded_tokens_provider=dt_provider,
    )

    print("*"*100)
    print("Prompt: ", local_prompt)
    print("Prompt tokens length: ", prompt_len)
    with torch.inference_mode():
        for _step in range(max_new_tokens):
            print("="*100)
            print("Step #", _step)
            out = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=fwd_hooks,
            )
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            next_token_logits = logits[0, -1].float()
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            if "all" in ablation_scope:
                logging.info(
                    "[all_positions step %d] logits[:5]=%s",
                    _step,
                    next_token_logits[:5].detach().cpu().tolist(),
                )
                logging.info(
                    "[all_positions step %d] softmax[:5]=%s",
                    _step,
                    next_token_probs[:5].detach().cpu().tolist(),
                )
            # Greedy decoding (TransformerLens does not have a .generate() method like in transformers lib)
            next_id = int(next_token_logits.argmax(dim=-1).item())
            if "all" in ablation_scope:
                logging.info("[all_positions step %d] token_id=%d", _step, next_id)
            generated.append(next_id)
            piece = model.tokenizer.decode([next_id], skip_special_tokens=False)
            print("Generated token #", _step, ":", piece)
            decoded_tokens.append(piece)

            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)

            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break

    response = _postprocess_response_from_full_decode(model, tokens, local_prompt)
    return response, decoded_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    TRAIN_RATIO = 0.9
    parser = argparse.ArgumentParser(description="Layerwise zero ablation inference (TransformerLens).")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["trivia_qa", "squad", "bioasq", "nq", "svamp", "gsm8k"],
    )
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_max_new_tokens", type=int, default=50)
    parser.add_argument("--brief_prompt", type=str, default="default", choices=["default", "chat"])
    parser.add_argument("--brief_always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_brief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ablate_layers",
        type=str,
        default="12-15",
        help="Inclusive range '12-15' or comma list '12,13,14,15' (Zero-indexing!).",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="both",
        choices=["all", "probability_tokens", "both"],
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "layerwise_zero_ablation/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_eval_dataset(args.dataset, args.random_seed)

    random.seed(args.random_seed)
    answerable_train = split_answerable_indices(train_ds)
    if len(answerable_train) < args.num_few_shot:
        raise ValueError("Not enough answerable training examples for few-shot.")
    prompt_indices = random.sample(answerable_train, args.num_few_shot)

    brief = BRIEF_PROMPTS[args.brief_prompt]
    # Same as generate_answers_with_confidence.py: arg passed to construct_fewshot.
    brief_always_effective = args.brief_always if args.enable_brief else True
    fewshot_prefix = construct_fewshot_prompt_from_indices(
        train_ds,
        prompt_indices,
        brief,
        brief_always=brief_always_effective,
        use_context=args.use_context,
    )

    logging.info("Loading HookedTransformer: %s", args.model_name)
    model = load_hooked_transformer(
        args.model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers)

    results = {"train": {}, "validation": {}}
    mini_results = {"train": {}, "validation": {}}

    modes: List[str]
    if args.ablation_mode == "both":
        modes = ["all", "probability_tokens"]
    else:
        modes = [args.ablation_mode]

    def _json_key_for_ablation_mode(mode: str) -> str:
        return "all_positions" if mode == "all" else "probability_tokens"

    summary_keys = list(dict.fromkeys(_json_key_for_ablation_mode(m) for m in modes))
    mode_confidence_values: Dict[str, List[float]] = {k: [] for k in summary_keys}

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        split_target = (
            round(args.num_samples * TRAIN_RATIO)
            if split_name == "train"
            else round(args.num_samples * (1 - TRAIN_RATIO))
        )
        n = min(split_target, len(eval_ds))
        logging.info("Generating for %d examples (%s split).", n, split_name)

        for i in range(n):
            example = eval_ds[i]
            question = example["question"]
            ex_id = encode_example_id(example["id"])
            local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + question

            entry = {"question": question}
            mini_entry = {"question": question}
            for mode in modes:
                key = "all_positions" if mode == "all" else "probability_tokens"
                if mode == "probability_tokens":
                    logging.debug(
                        "Example %s: probability_tokens ablation uses dynamic span when parseable.",
                        ex_id,
                    )
                response, decoded_tokens = greedy_generate_ablated(
                    model,
                    local_prompt,
                    ablation_scope=mode,
                    ablate_layers=ablate_layers,
                    max_new_tokens=args.model_max_new_tokens,
                )
                entry[key] = {"response": response, "decoded_tokens": decoded_tokens}
                mini_entry[key] = {"response": response}
                parsed_conf = parse_probability_from_response(response)
                if parsed_conf is not None:
                    mode_confidence_values[key].append(float(parsed_conf))
                logging.info("[%s %d/%d] %s %s first line: %r", split_name, i + 1, n, ex_id, key, response[:120])

            results[split_name][ex_id] = entry
            mini_results[split_name][ex_id] = mini_entry

    out_path = resolve_output_json_path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("Wrote %s", out_path)

    mini_out_path = mini_output_json_path(out_path)
    with open(mini_out_path, "w", encoding="utf-8") as f:
        json.dump(mini_results, f, ensure_ascii=False, indent=2)
    logging.info("Wrote %s", mini_out_path)

    mode_confidence_means: Dict[str, Optional[float]] = {}
    mode_confidence_counts: Dict[str, int] = {}
    for summary_key in summary_keys:
        vals = mode_confidence_values[summary_key]
        mode_confidence_means[summary_key] = (sum(vals) / len(vals)) if vals else None
        mode_confidence_counts[summary_key] = len(vals)

    config_out_path = config_txt_path(out_path)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    write_config_txt(
        config_out_path,
        args=args,
        device=device,
        model_n_layers=model.cfg.n_layers,
        ablate_layers=ablate_layers,
        prompt_indices=prompt_indices,
        mode_confidence_means=mode_confidence_means,
        mode_confidence_counts=mode_confidence_counts,
        finished_at=finished_at,
    )
    logging.info("Wrote %s", config_out_path)


if __name__ == "__main__":
    main()
