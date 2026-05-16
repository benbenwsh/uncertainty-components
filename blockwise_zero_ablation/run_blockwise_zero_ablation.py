#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Greedy decoding on TriviaQA with blockwise zero ablation via TransformerLens.

This script mirrors the layerwise mean-ablation evaluation structure, but applies
zero ablation to chosen sub-block outputs (`attn` / `mlp`) at dynamic token spans.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

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

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n"]
GUESS_PREFIX = "\n\nGuess:"
PROBABILITY_MARKER = "\nProbability:"

SUBBLOCK_TO_HOOK = {
    "attn": "hook_attn_out",
    "mlp": "hook_mlp_out",
}


def _token_index_for_char_offset(decoded_tokens: List[str], char_offset: int) -> int:
    cumulative = 0
    for i, tok in enumerate(decoded_tokens):
        cumulative += len(tok)
        if cumulative > char_offset:
            return i
    return max(0, len(decoded_tokens) - 1)


def parse_guess_and_probability_indices(decoded_tokens: List[str]) -> tuple[int, int, int] | None:
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
        or end_prob_token_index > len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None
    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def parse_guess_start_index(decoded_tokens: List[str]) -> Optional[int]:
    full_str = "".join(decoded_tokens)
    if not full_str.startswith(GUESS_PREFIX):
        return None
    last_guess_token_index = _token_index_for_char_offset(decoded_tokens, len(GUESS_PREFIX) - 1) + 1
    if last_guess_token_index <= 0 or last_guess_token_index > len(decoded_tokens):
        return None
    return last_guess_token_index


def load_trivia_qa(seed: int) -> Tuple[Dataset, Dataset]:
    raw = load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")["unmodified"]
    split = raw.train_test_split(test_size=0.2, seed=seed)
    return split["train"], split["test"]


def split_answerable_indices(dataset: Dataset) -> List[int]:
    return [i for i, ex in enumerate(dataset) if len(ex["answers"]["text"]) > 0]


def construct_fewshot_prompt_from_indices(
    dataset: Dataset,
    example_indices: Sequence[int],
    brief: str,
    brief_always: bool,
    use_context: bool,
) -> str:
    prompt = brief if not brief_always else ""
    for example_index in example_indices:
        example = dataset[int(example_index)]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        piece = ""
        if brief_always:
            piece += brief
        if use_context and context is not None:
            piece += f"Context: {context}\n"
        piece += f"Question: {question}\n"
        piece += f"Answer: {answer}\n\n" if answer else "Answer:"
        prompt += piece
    return prompt


def encode_example_id(example_id) -> str:
    return quote(str(example_id), safe="")


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        return cli_output_path
    repo_blockwise_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(repo_blockwise_dir, "results")
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_dir = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "ablation_results.json")


def mini_output_json_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "ablation_results_mini.json")


def config_txt_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "config.txt")


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    ablate_layers: Sequence[int],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    ablate_low_confidence_samples: bool,
    mode_confidence_means: Dict[str, Dict[str, Optional[float]]],
    mode_confidence_counts: Dict[str, Dict[str, int]],
    mode_responses_identical_true: Dict[str, Dict[str, int]],
    finished_at: str,
) -> None:
    non_ablated_group = "high_confidence" if ablate_low_confidence_samples else "low_confidence"
    target_group = "low_confidence" if ablate_low_confidence_samples else "high_confidence"
    lines = [
        "Blockwise Zero Ablation Configuration",
        "====================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        "",
        "[Data]",
        f"input_h5={args.input_h5}",
        f"h5_example_count={h5_example_count}",
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
        f"ablate_subblocks={args.ablate_subblocks}",
        f"ablate_low_confidence_samples={ablate_low_confidence_samples}",
        f"non_ablated_confidence_group={non_ablated_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Mode Confidence Metrics]",
    ]
    for mode_name in args.ablation_mode:
        per_subblock = mode_confidence_means.get(mode_name, {})
        per_subblock_counts = mode_confidence_counts.get(mode_name, {})
        for subblock in args.ablate_subblocks:
            mode_mean = per_subblock.get(subblock)
            valid_count = int(per_subblock_counts.get(subblock, 0))
            metric_key = f"{mode_name}__{subblock}_mean_verbalised_confidence"
            if mode_name == "none":
                if mode_mean is None:
                    lines.append(f"{metric_key}=None ({valid_count})")
                else:
                    lines.append(f"{metric_key}={mode_mean:.6f} ({valid_count})")
            else:
                identical_n = int(
                    mode_responses_identical_true.get(mode_name, {}).get(subblock, 0)
                )
                if mode_mean is None:
                    lines.append(
                        f"{metric_key}=None ({valid_count}) [responses_identical: {identical_n}]"
                    )
                else:
                    lines.append(
                        f"{metric_key}={mode_mean:.6f} ({valid_count}) [responses_identical: {identical_n}]"
                    )
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def mode_to_output_key(mode: str) -> str:
    if mode == "none":
        return "no_replacement"
    if mode in {
        "probability_tokens_mean_replace",
        "probability_last_token_mean_replace",
        "probability_span_except_last_token_mean_replace",
        "all_pre_probability_tokens_mean_replace",
        "guess_tokens_mean_replace",
        "all_pre_guess_tokens_mean_replace",
        "guess_then_guess_probability_zero_ablate",
        "probability_value_autoregressive_zero_ablate",
    }:
        return mode
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def parse_mode_confidence_from_response(response: str) -> Optional[float]:
    parsed = parse_probability_from_response(response)
    return float(parsed) if parsed is not None else None


def parse_probability_from_response(response_str: str) -> float | None:
    if not response_str or not isinstance(response_str, str):
        return None
    matches = list(re.finditer(r"probability\s*:\s*([0-9]+[.,]?[0-9]*)\s*%?", response_str, re.IGNORECASE))
    if not matches:
        matches = list(re.finditer(r"probability\s*:\s*(\d+(?:[.,]\d+)?)", response_str, re.IGNORECASE))
    if not matches:
        return None
    raw = matches[-1].group(1).strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0 or value > 1:
        return None
    return value


def load_hooked_transformer(
    model_name: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> HookedTransformer:
    try:
        return HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            device=device,
            dtype=torch_dtype,
        )
    except Exception as exc:
        if "PyPreTokenizerTypeWrapper" not in str(exc):
            raise
        logging.warning("Fast tokenizer load failed (%s). Retrying with use_fast=False.", exc)
        hf_token = os.environ.get("HF_TOKEN", None)
        slow_tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_bos_token=True, trust_remote_code=True, use_fast=False, token=hf_token
        )
        return HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            device=device,
            dtype=torch_dtype,
            tokenizer=slow_tokenizer,
        )


def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_h5_node(node):
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
    return {k: _read_h5_node(node[k]) for k in node.keys()}


def load_examples_h5(path: Path) -> Dict[str, dict]:
    examples: Dict[str, dict] = {}
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        ex_group = h5_file["examples"]
        for example_id in ex_group.keys():
            obj = _read_h5_node(ex_group[example_id])
            if isinstance(obj, dict):
                examples[str(example_id)] = obj
    return examples


def collect_confidence_group_ids(
    examples_h5: Dict[str, dict],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
) -> Tuple[set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError(f"Example {ex_id} must have exactly one response, got {0 if responses is None else len(responses)}.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 is not a dict.")
        conf = float(resp0.get("verbalised_confidence"))
        if conf <= low_conf_threshold:
            low_ids.add(ex_id)
        if conf >= high_conf_threshold:
            high_ids.add(ex_id)
    return low_ids, high_ids


def parse_ablate_layers(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        layers = list(range(int(a.strip()), int(b.strip()) + 1))
    else:
        layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}) for this model.")
    return layers


def _is_expected_or_plus_one(actual_len: int, expected_len: int) -> bool:
    return actual_len in (expected_len, expected_len + 1)


def _completion_token_index_to_abs_pos(prompt_len: int, completion_index: int) -> int:
    """Map completion-relative token index (0 = first generated token) to full-sequence position.

    First generated token aligns with ``prompt_len - 1`` (same convention as
    ``layerwise_mean_ablation.run_mean_ablation``).
    """
    return prompt_len + completion_index - 1


def _absolute_prob_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
) -> List[int]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    rel_positions = list(range(first_prob, end_prob+1))
    if not _is_expected_or_plus_one(len(rel_positions), expected_probability_tokens):
        return []
    rel_positions = rel_positions[:expected_probability_tokens]
    return [_completion_token_index_to_abs_pos(prompt_len, pos) for pos in rel_positions]


def _absolute_prob_last_token_only_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
) -> List[int]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    full_rel_positions = list(range(first_prob, end_prob+1))
    if not _is_expected_or_plus_one(len(full_rel_positions), expected_probability_tokens):
        return []
    return [_completion_token_index_to_abs_pos(prompt_len, end_prob)]


def _absolute_prob_except_last_token_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
) -> List[int]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, end_prob = parsed
    full_rel_positions = list(range(first_prob, end_prob+1))
    if not _is_expected_or_plus_one(len(full_rel_positions), expected_probability_tokens):
        return []
    return [
        _completion_token_index_to_abs_pos(prompt_len, pos) for pos in range(first_prob, end_prob)
    ]


def _absolute_guess_span_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
) -> List[int]:
    last_guess_token_index = parse_guess_start_index(decoded_tokens)
    if last_guess_token_index is None:
        return []
    guess_positions_rel = list(range(0, last_guess_token_index))
    if not _is_expected_or_plus_one(len(guess_positions_rel), expected_guess_tokens):
        return []
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]
    return [_completion_token_index_to_abs_pos(prompt_len, k) for k in guess_positions_rel]


def _absolute_pre_probability_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Optional[Dict[str, List[int]]]:
    """Absolute positions for pre-probability mean/zero ablation.

    ``prompt`` uses indices ``0 .. prompt_len-2`` (excludes the last prompt position, which
    aligns with the first generated token under the completion-index mapping).
    """
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return None
    last_guess_token_index, first_prob_token_index, end_prob_token_index = parsed

    guess_positions_rel = list(range(0, last_guess_token_index))
    if not _is_expected_or_plus_one(len(guess_positions_rel), expected_guess_tokens):
        return None
    guess_positions_rel = guess_positions_rel[:expected_guess_tokens]

    probability_positions_rel = list(range(first_prob_token_index, end_prob_token_index+1))
    if not _is_expected_or_plus_one(len(probability_positions_rel), expected_probability_tokens):
        return None
    probability_positions_rel = probability_positions_rel[:expected_probability_tokens]

    return {
        "prompt": list(range(0, prompt_len - 1)),
        "guess": [_completion_token_index_to_abs_pos(prompt_len, k) for k in guess_positions_rel],
        "sem_answer": [
            _completion_token_index_to_abs_pos(prompt_len, k)
            for k in range(last_guess_token_index, first_prob_token_index)
        ],
        "probability": [
            _completion_token_index_to_abs_pos(prompt_len, k) for k in probability_positions_rel
        ],
    }


def _absolute_all_pre_guess_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
) -> List[int]:
    """Prompt indices ``0 .. prompt_len-2`` plus guess-span positions (see ``_absolute_guess_span_positions``)."""
    guess_positions = _absolute_guess_span_positions(
        prompt_len,
        decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
    )
    if not guess_positions:
        return []
    return list(range(0, prompt_len - 1)) + guess_positions


def _absolute_guess_then_guess_probability_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> List[int]:
    guess_positions = _absolute_guess_span_positions(
        prompt_len,
        decoded_tokens,
        expected_guess_tokens=expected_guess_tokens,
    )
    if not guess_positions:
        return []
    prob_positions = _absolute_prob_positions(
        prompt_len,
        decoded_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    return guess_positions if not prob_positions else sorted(set(guess_positions + prob_positions))


def _absolute_probability_value_autoregressive_positions(
    prompt_len: int,
    decoded_tokens: List[str],
) -> List[int]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, _, probability_value_start_rel = parsed
    current_rel = len(decoded_tokens)
    if current_rel < probability_value_start_rel:
        return []
    return [
        _completion_token_index_to_abs_pos(prompt_len, k)
        for k in range(probability_value_start_rel, current_rel + 1)
    ]


def _generation_contains_stop(decoded_completion: str) -> bool:
    return any(s in decoded_completion for s in STOP_SEQUENCES)


def _strip_stop_suffixes(text: str) -> str:
    stop_at = len(text)
    for stop in STOP_SEQUENCES:
        if text.endswith(stop):
            stop_at = len(text) - len(stop)
            break
    return text[:stop_at].strip()


def _postprocess_response_from_full_decode(
    model: HookedTransformer,
    full_sequence_tokens: torch.Tensor,
    input_text: str,
) -> str:
    full_answer = model.tokenizer.decode(full_sequence_tokens[0], skip_special_tokens=True)
    if full_answer.startswith(input_text):
        answer = full_answer[len(input_text) :]
    else:
        logging.warning("Decoded text does not start with prompt; using full decoded text.")
        answer = full_answer
    return _strip_stop_suffixes(answer)


def greedy_generate(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
    decoded_tokens_buffer: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    decoded_tokens: List[str] = decoded_tokens_buffer if decoded_tokens_buffer is not None else []
    eos_id = model.tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # if fwd_hooks:
            #     logging.info(f"Decoded tokens in non-none ablation mode: {decoded_tokens}")
            #     logging.info(f"Prompt shape: {tokens.shape}; decoded tokens length: {len(decoded_tokens)}")
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks or [])
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            completion_str = "".join(decoded_tokens)
            if _generation_contains_stop(completion_str):
                break
            if eos_id is not None and next_id == eos_id:
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def build_subblock_zero_hooks(
    layer_indices: Sequence[int],
    *,
    subblock: str,
    positions_provider: Callable[[], List[int]],
) -> List[Tuple[str, Callable]]:
    hook_suffix = SUBBLOCK_TO_HOOK[subblock]
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.{hook_suffix}"

        def _make_hook() -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                # Activation shape: [1, seq_len, hidden_dim]
                del hook
                positions = positions_provider()
                if not positions:
                    return activation
                for abs_pos in positions:
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = 0
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook()))
    return hooks


def greedy_generate_zero_ablated(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    subblock: str,
    positions_provider_builder: Callable[[int, Callable[[], List[str]]], Callable[[], List[int]]],
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    positions_provider = positions_provider_builder(prompt_len, _decoded_tokens_provider)
    fwd_hooks = build_subblock_zero_hooks(
        layer_indices,
        subblock=subblock,
        positions_provider=positions_provider,
    )
    return greedy_generate(
        model,
        local_prompt,
        max_new_tokens,
        fwd_hooks=fwd_hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def _mode_positions_provider_builder(
    mode: str,
    *,
    expected_guess_tokens: int,
    expected_probability_tokens: int,
) -> Callable[[int, Callable[[], List[str]]], Callable[[], List[int]]]:
    def _builder(prompt_len: int, decoded_tokens_provider: Callable[[], List[str]]) -> Callable[[], List[int]]:
        if mode == "probability_tokens_mean_replace":
            return lambda: _absolute_prob_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "probability_last_token_mean_replace":
            return lambda: _absolute_prob_last_token_only_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "probability_span_except_last_token_mean_replace":
            return lambda: _absolute_prob_except_last_token_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "guess_tokens_mean_replace":
            return lambda: _absolute_guess_span_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_guess_tokens=expected_guess_tokens,
            )
        if mode == "all_pre_probability_tokens_mean_replace":
            return lambda: (
                []
                if (
                    pos_map := _absolute_pre_probability_positions(
                        prompt_len,
                        decoded_tokens_provider(),
                        expected_guess_tokens=expected_guess_tokens,
                        expected_probability_tokens=expected_probability_tokens,
                    )
                )
                is None
                else pos_map["prompt"] + pos_map["guess"] + pos_map["sem_answer"] + pos_map["probability"]
            )
        if mode == "all_pre_guess_tokens_mean_replace":
            return lambda: _absolute_all_pre_guess_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_guess_tokens=expected_guess_tokens,
            )
        if mode == "guess_then_guess_probability_zero_ablate":
            return lambda: _absolute_guess_then_guess_probability_positions(
                prompt_len,
                decoded_tokens_provider(),
                expected_guess_tokens=expected_guess_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
        if mode == "probability_value_autoregressive_zero_ablate":
            return lambda: _absolute_probability_value_autoregressive_positions(
                prompt_len,
                decoded_tokens_provider(),
            )
        raise ValueError(f"Unknown ablation mode for provider builder: {mode!r}")

    return _builder


def _normalize_per_layer_mode_means(
    raw_summary: Dict[object, object],
) -> Dict[int, Dict[str, Dict[str, Optional[float]]]]:
    normalized: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
    for raw_layer, raw_modes in raw_summary.items():
        try:
            layer_idx = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid layer key in summary JSON: {raw_layer!r}") from exc
        if not isinstance(raw_modes, dict):
            raise ValueError(f"Expected dict for layer {layer_idx}, got {type(raw_modes).__name__}.")

        normalized[layer_idx] = {}
        for mode_name, raw_subblocks in raw_modes.items():
            if not isinstance(raw_subblocks, dict):
                raise ValueError(
                    f"Expected dict for layer={layer_idx} mode={mode_name!r}, got {type(raw_subblocks).__name__}."
                )
            per_subblock: Dict[str, Optional[float]] = {}
            for subblock_name, raw_value in raw_subblocks.items():
                if raw_value is None:
                    per_subblock[str(subblock_name)] = None
                else:
                    per_subblock[str(subblock_name)] = float(raw_value)
            normalized[layer_idx][str(mode_name)] = per_subblock
    return normalized


def _plot_subblock_mode_confidence(
    *,
    per_layer_mode_means: Dict[int, Dict[str, Dict[str, Optional[float]]]],
    layer_order: Sequence[int],
    subblock: str,
    ablation_modes: Sequence[str],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    modes_non_none = [mode for mode in ablation_modes if mode != "none"]
    for mode in modes_non_none:
        xs: List[int] = []
        ys: List[float] = []
        for layer_idx in layer_order:
            layer_modes = per_layer_mode_means.get(layer_idx, {})
            y_val = layer_modes.get(mode, {}).get(subblock)
            if y_val is not None:
                xs.append(layer_idx)
                ys.append(float(y_val))
        if ys:
            ax.plot(xs, ys, marker="o", label=mode)

    baseline_none_mean: Optional[float] = None
    for layer_idx in layer_order:
        y_val = per_layer_mode_means.get(layer_idx, {}).get("none", {}).get(subblock)
        if y_val is not None:
            baseline_none_mean = float(y_val)
            break
    if baseline_none_mean is not None:
        ax.axhline(y=baseline_none_mean, linestyle="--", label="none (baseline)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Verbalised confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Individual layer verbalised confidence ({subblock})")
    ax.grid(True, alpha=0.3)
    if modes_non_none or baseline_none_mean is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def _plot_combined_subblock_mode_confidence(
    *,
    per_layer_mode_means: Dict[int, Dict[str, Dict[str, Optional[float]]]],
    layer_order: Sequence[int],
    ablation_modes: Sequence[str],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    modes_non_none = [mode for mode in ablation_modes if mode != "none"]
    subblock_colors = {"attn": "red", "mlp": "blue"}
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    mode_markers = {mode: marker_cycle[i % len(marker_cycle)] for i, mode in enumerate(modes_non_none)}
    for subblock in ("attn", "mlp"):
        for mode in modes_non_none:
            xs: List[int] = []
            ys: List[float] = []
            for layer_idx in layer_order:
                layer_modes = per_layer_mode_means.get(layer_idx, {})
                y_val = layer_modes.get(mode, {}).get(subblock)
                if y_val is not None:
                    xs.append(layer_idx)
                    ys.append(float(y_val))
            if ys:
                ax.plot(
                    xs,
                    ys,
                    marker=mode_markers[mode],
                    color=subblock_colors[subblock],
                    label=f"{mode} ({subblock})",
                )

    for subblock in ("attn", "mlp"):
        baseline_none_mean: Optional[float] = None
        for layer_idx in layer_order:
            y_val = per_layer_mode_means.get(layer_idx, {}).get("none", {}).get(subblock)
            if y_val is not None:
                baseline_none_mean = float(y_val)
                break
        if baseline_none_mean is not None:
            ax.axhline(
                y=baseline_none_mean,
                linestyle="--",
                color=subblock_colors[subblock],
                label=f"none (baseline, {subblock})",
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Verbalised confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Individual layer verbalised confidence (attn + mlp)")
    ax.grid(True, alpha=0.3)
    if modes_non_none:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def write_individual_layer_plots(
    *,
    per_layer_mode_means: Dict[int, Dict[str, Dict[str, Optional[float]]]],
    ablation_modes: Sequence[str],
    ablate_subblocks: Sequence[str],
    output_dir: str,
) -> None:
    layer_order = sorted(per_layer_mode_means.keys())
    if not layer_order:
        raise ValueError("No per-layer summary values available to plot.")

    for subblock in ablate_subblocks:
        plot_path = os.path.join(output_dir, f"verbalised_confidence_by_layer_{subblock}.png")
        _plot_subblock_mode_confidence(
            per_layer_mode_means=per_layer_mode_means,
            layer_order=layer_order,
            subblock=subblock,
            ablation_modes=ablation_modes,
            output_path=plot_path,
        )

    if "attn" in ablate_subblocks and "mlp" in ablate_subblocks:
        combined_plot_path = os.path.join(output_dir, "verbalised_confidence_by_layer_attn_mlp.png")
        _plot_combined_subblock_mode_confidence(
            per_layer_mode_means=per_layer_mode_means,
            layer_order=layer_order,
            ablation_modes=ablation_modes,
            output_path=combined_plot_path,
        )


def main() -> None:
    print("helloe")
    TRAIN_RATIO = 0.9
    parser = argparse.ArgumentParser(description="Blockwise zero ablation inference (TransformerLens).")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--input_h5", type=str, default=None, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
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
        nargs="+",
        default=[
            "none",
            "probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_zero_ablate",
            "probability_value_autoregressive_zero_ablate",
        ],
        choices=[
            "none",
            "probability_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_zero_ablate",
            "probability_value_autoregressive_zero_ablate",
        ],
        help=(
            "One or more ablation modes. probability_last_token_mean_replace: ablate only the "
            "last token in the Probability: marker span. "
            "probability_span_except_last_token_mean_replace: ablate all Probability: span tokens "
            "except that last token."
        ),
    )
    parser.add_argument(
        "--ablate_subblocks",
        type=str,
        nargs="+",
        required=True,
        choices=["attn", "mlp"],
        help="One or both subblocks to ablate. If both, experiments run separately per subblock.",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--ablate_low_confidence_samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, target low-confidence examples for ablation. "
            "If false (default), target high-confidence examples."
        ),
    )
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--parse_mode_verbalised_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "blockwise_zero_ablation/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    parser.add_argument(
        "--individual_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, run a separate ablation for each layer under results/individual_layers/<run_id>/<layer_idx>/.",
    )
    parser.add_argument(
        "--plot_from_existing_summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true with --individual_layers, skip inference and build individual-layer plots from an existing "
            "individual_layers/<run_id>/mode_confidence_summary.json."
        ),
    )
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if len(set(args.ablate_subblocks)) != len(args.ablate_subblocks):
        raise ValueError(f"Duplicate subblocks are not allowed: {args.ablate_subblocks}")
    if len(args.ablate_subblocks) < 1:
        raise ValueError("--ablate_subblocks must contain at least one of ['attn', 'mlp'].")
    if args.plot_from_existing_summary and not args.individual_layers:
        raise ValueError("--plot_from_existing_summary requires --individual_layers.")
    if not args.plot_from_existing_summary and not args.input_h5:
        raise ValueError("--input_h5 is required unless --plot_from_existing_summary is enabled.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)
    run_root_norm = run_root.rstrip(os.sep)
    run_id = os.path.basename(run_root_norm)
    results_root = os.path.dirname(run_root_norm)
    individual_layers_root = os.path.join(results_root, "individual_layers")
    individual_root = os.path.join(individual_layers_root, run_id)

    if args.plot_from_existing_summary:
        summary_dir_name = input(
            "Enter the directory name under results/individual_layers containing mode_confidence_summary.json: "
        ).strip()
        if not summary_dir_name:
            raise ValueError("Directory name cannot be empty.")
        if os.path.basename(summary_dir_name) != summary_dir_name:
            raise ValueError(
                "Please provide only the directory name (not a path), e.g. <run_id> under results/individual_layers."
            )

        summary_dir = os.path.join(individual_layers_root, summary_dir_name)
        summary_path = os.path.join(summary_dir, "mode_confidence_summary.json")
        if not os.path.exists(summary_path):
            raise ValueError(
                f"Summary file not found: {summary_path}. Provide a valid directory under results/individual_layers."
            )
        with open(summary_path, "r", encoding="utf-8") as f:
            raw_summary = json.load(f)
        if not isinstance(raw_summary, dict):
            raise ValueError(f"Malformed summary JSON at {summary_path}: expected object at top level.")
        per_layer_mode_means = _normalize_per_layer_mode_means(raw_summary)
        write_individual_layer_plots(
            per_layer_mode_means=per_layer_mode_means,
            ablation_modes=args.ablation_mode,
            ablate_subblocks=args.ablate_subblocks,
            output_dir=summary_dir,
        )
        return

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_trivia_qa(args.random_seed)
    random.seed(args.random_seed)
    answerable_train = split_answerable_indices(train_ds)
    if len(answerable_train) < args.num_few_shot:
        raise ValueError("Not enough answerable training examples for few-shot.")
    prompt_indices = random.sample(answerable_train, args.num_few_shot)

    brief = BRIEF_PROMPTS[args.brief_prompt]
    brief_always_effective = args.brief_always if args.enable_brief else True
    fewshot_prefix = construct_fewshot_prompt_from_indices(
        train_ds,
        prompt_indices,
        brief,
        brief_always=brief_always_effective,
        use_context=args.use_context,
    )

    logging.info("Loading HookedTransformer: %s", args.model_name)
    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers)
    run_layers = list(range(model.cfg.n_layers)) if args.individual_layers else ablate_layers

    examples_h5 = load_examples_h5(Path(args.input_h5))
    low_ids, high_ids = collect_confidence_group_ids(
        examples_h5,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
    )
    if args.ablate_low_confidence_samples:
        if not low_ids:
            raise ValueError(f"No low-confidence examples found at threshold <= {args.low_conf_threshold}.")
        ablation_target_ids = low_ids
        target_group = "low_confidence"
    else:
        if not high_ids:
            raise ValueError(f"No high-confidence examples found at threshold >= {args.high_conf_threshold}.")
        ablation_target_ids = high_ids
        target_group = "high_confidence"
    logging.info("Low-confidence example IDs: %s", low_ids)
    logging.info("High-confidence example IDs: %s", high_ids)
    logging.info(
        "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), layers=%s, ablate_low_confidence_samples=%s, individual_layers=%s",
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        run_layers,
        args.ablate_low_confidence_samples,
        args.individual_layers,
    )
    logging.info(
        "Ablation target group=%s (%d ids).",
        target_group,
        len(ablation_target_ids),
    )

    def run_one_evaluation(
        layer_subset: Sequence[int],
        cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    ) -> Tuple[
        dict,
        dict,
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, Dict[str, object]]],
        Dict[str, Dict[str, int]],
    ]:
        results = {"train": {}, "validation": {}}
        mini_results = {"train": {}, "validation": {}}
        mode_confidence_values: Dict[str, Dict[str, List[float]]] = {
            mode_name: {subblock: [] for subblock in args.ablate_subblocks} for mode_name in args.ablation_mode
        }
        mode_responses_identical_true: Dict[str, Dict[str, int]] = {
            mode_name: {subblock: 0 for subblock in args.ablate_subblocks}
            for mode_name in args.ablation_mode
            if mode_name != "none"
        }
        used_none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}

        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(args.num_samples * (1 - TRAIN_RATIO))
            id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
            split_target_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)
            if split_target > 0 and not split_target_ids:
                logging.warning("No ablation target IDs available for %s split.", split_name)
                continue
            selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
            logging.info("Generating for %d examples (%s split).", len(selected_ids), split_name)

            for i, ex_id in enumerate(selected_ids):
                ds_idx = id_to_index.get(ex_id)
                if ds_idx is None:
                    raise ValueError(f"Example id {ex_id} selected from H5 but not found in {split_name} split.")
                example = eval_ds[int(ds_idx)]
                local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + example["question"]
                entry = {"question": example["question"]}
                mini_entry = {"question": example["question"]}

                baseline_response: Optional[str] = None
                baseline_decoded_tokens: Optional[List[str]] = None
                baseline_mode_confidence: Optional[float] = None
                if "none" in args.ablation_mode:
                    if cached_none is not None and ex_id in cached_none[split_name]:
                        cached = cached_none[split_name][ex_id]
                        baseline_response = str(cached["response"])
                        baseline_decoded_tokens = list(cached["decoded_tokens"])
                        baseline_mode_confidence = cached.get("verbalised_confidence")
                    else:
                        baseline_response, baseline_decoded_tokens = greedy_generate(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            fwd_hooks=None,
                        )
                        baseline_mode_confidence = (
                            parse_mode_confidence_from_response(baseline_response)
                            if args.parse_mode_verbalised_confidence
                            else None
                        )
                        used_none_cache[split_name][ex_id] = {
                            "response": baseline_response,
                            "decoded_tokens": baseline_decoded_tokens,
                            "verbalised_confidence": baseline_mode_confidence,
                        }

                for mode in args.ablation_mode:
                    key = mode_to_output_key(mode)
                    entry[key] = {}
                    mini_entry[key] = {}

                    for subblock in args.ablate_subblocks:
                        if mode == "none":
                            assert baseline_response is not None
                            assert baseline_decoded_tokens is not None
                            response = baseline_response
                            decoded_tokens = baseline_decoded_tokens
                            mode_confidence = baseline_mode_confidence
                        else:
                            provider_builder = _mode_positions_provider_builder(
                                mode,
                                expected_guess_tokens=args.expected_guess_tokens,
                                expected_probability_tokens=args.expected_probability_tokens,
                            )
                            response, decoded_tokens = greedy_generate_zero_ablated(
                                model=model,
                                local_prompt=local_prompt,
                                max_new_tokens=args.model_max_new_tokens,
                                layer_indices=layer_subset,
                                subblock=subblock,
                                positions_provider_builder=provider_builder,
                            )
                            mode_confidence = (
                                parse_mode_confidence_from_response(response)
                                if args.parse_mode_verbalised_confidence
                                else None
                            )

                        entry[key][subblock] = {"response": response, "decoded_tokens": decoded_tokens}
                        mini_entry[key][subblock] = {"response": response}
                        if args.parse_mode_verbalised_confidence:
                            entry[key][subblock]["verbalised_confidence"] = mode_confidence
                            mini_entry[key][subblock]["verbalised_confidence"] = mode_confidence
                            if mode_confidence is not None:
                                mode_confidence_values[mode][subblock].append(float(mode_confidence))

                        if mode != "none" and baseline_response is not None:
                            responses_identical = response == baseline_response
                            entry[key][subblock]["responses_identical"] = responses_identical
                            mini_entry[key][subblock]["responses_identical"] = responses_identical
                            if responses_identical:
                                mode_responses_identical_true[mode][subblock] += 1
                            if args.parse_mode_verbalised_confidence:
                                if mode_confidence is None or baseline_mode_confidence is None:
                                    meets_none_confidence_direction = None
                                elif args.ablate_low_confidence_samples:
                                    meets_none_confidence_direction = mode_confidence < baseline_mode_confidence
                                else:
                                    meets_none_confidence_direction = mode_confidence > baseline_mode_confidence
                                entry[key][subblock]["meets_none_confidence_direction"] = meets_none_confidence_direction
                                mini_entry[key][subblock]["meets_none_confidence_direction"] = meets_none_confidence_direction

                        logging.info(
                            "[%s %d/%d] %s %s/%s first line: %r",
                            split_name,
                            i + 1,
                            len(selected_ids),
                            ex_id,
                            key,
                            subblock,
                            response[:120],
                        )

                results[split_name][ex_id] = entry
                mini_results[split_name][ex_id] = mini_entry

        mode_confidence_means: Dict[str, Dict[str, Optional[float]]] = {}
        mode_confidence_counts: Dict[str, Dict[str, int]] = {}
        for mode_name in args.ablation_mode:
            mode_confidence_means[mode_name] = {}
            mode_confidence_counts[mode_name] = {}
            for subblock in args.ablate_subblocks:
                vals = mode_confidence_values[mode_name][subblock]
                mode_confidence_means[mode_name][subblock] = float(np.mean(vals)) if vals else None
                mode_confidence_counts[mode_name][subblock] = len(vals)
        return (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            used_none_cache,
            mode_responses_identical_true,
        )

    def build_none_cache() -> Dict[str, Dict[str, Dict[str, object]]]:
        none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}
        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(args.num_samples * (1 - TRAIN_RATIO))
            id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
            split_target_ids = sorted(ex_id for ex_id in ablation_target_ids if ex_id in id_to_index)
            selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
            for ex_id in selected_ids:
                ds_idx = id_to_index.get(ex_id)
                if ds_idx is None:
                    continue
                example = eval_ds[int(ds_idx)]
                local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + example["question"]
                response, decoded_tokens = greedy_generate(
                    model=model,
                    local_prompt=local_prompt,
                    max_new_tokens=args.model_max_new_tokens,
                    fwd_hooks=None,
                )
                mode_confidence = (
                    parse_mode_confidence_from_response(response)
                    if args.parse_mode_verbalised_confidence
                    else None
                )
                none_cache[split_name][ex_id] = {
                    "response": response,
                    "decoded_tokens": decoded_tokens,
                    "verbalised_confidence": mode_confidence,
                }
        return none_cache

    if not args.individual_layers:
        (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            _,
            mode_responses_identical_true,
        ) = run_one_evaluation(run_layers, cached_none=None)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        mini_out_path = mini_output_json_path(out_path)
        with open(mini_out_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)

        config_out_path = config_txt_path(out_path)
        write_config_txt(
            config_out_path,
            args=args,
            device=device,
            model_n_layers=model.cfg.n_layers,
            ablate_layers=ablate_layers,
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            ablate_low_confidence_samples=args.ablate_low_confidence_samples,
            mode_confidence_means=mode_confidence_means,
            mode_confidence_counts=mode_confidence_counts,
            mode_responses_identical_true=mode_responses_identical_true,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        logging.info("Wrote %s", out_path)
        logging.info("Wrote %s", mini_out_path)
        logging.info("Wrote %s", config_out_path)
        return

    os.makedirs(individual_root, exist_ok=True)

    cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None
    if "none" in args.ablation_mode:
        logging.info("Computing baseline none-mode once for individual-layer sweep.")
        cached_none = build_none_cache()

    per_layer_mode_means: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
    for layer_idx in run_layers:
        logging.info("Running individual-layer ablation for layer %d", layer_idx)
        layer_dir = os.path.join(individual_root, str(layer_idx))
        os.makedirs(layer_dir, exist_ok=True)
        (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            _,
            mode_responses_identical_true,
        ) = run_one_evaluation([layer_idx], cached_none=cached_none)
        per_layer_mode_means[layer_idx] = mode_confidence_means

        layer_out_path = os.path.join(layer_dir, "ablation_results.json")
        with open(layer_out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        layer_mini_out_path = os.path.join(layer_dir, "ablation_results_mini.json")
        with open(layer_mini_out_path, "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        layer_config_out_path = os.path.join(layer_dir, "config.txt")
        write_config_txt(
            layer_config_out_path,
            args=args,
            device=device,
            model_n_layers=model.cfg.n_layers,
            ablate_layers=[layer_idx],
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            ablate_low_confidence_samples=args.ablate_low_confidence_samples,
            mode_confidence_means=mode_confidence_means,
            mode_confidence_counts=mode_confidence_counts,
            mode_responses_identical_true=mode_responses_identical_true,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        logging.info("Wrote %s", layer_out_path)
        logging.info("Wrote %s", layer_mini_out_path)
        logging.info("Wrote %s", layer_config_out_path)

    summary_path = os.path.join(individual_root, "mode_confidence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(per_layer_mode_means, f, ensure_ascii=False, indent=2)
    logging.info("Wrote %s", summary_path)
    write_individual_layer_plots(
        per_layer_mode_means=per_layer_mode_means,
        ablation_modes=args.ablation_mode,
        ablate_subblocks=args.ablate_subblocks,
        output_dir=individual_root,
    )


if __name__ == "__main__":
    main()
