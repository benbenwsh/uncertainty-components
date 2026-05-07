#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Greedy decoding on TriviaQA with direction perturbations from verbalised-confidence probes.

This script mirrors the mass-mean intervention pipeline, but direction vectors come from
trained linear/ridge verbalised-confidence probes saved per probability token and per layer.

For each probe weight vector theta, the intervention direction is:
  direction = theta / ||theta||^2

Only two ablation modes are supported:
  - none
  - probability_tokens_mean_replace (name kept for compatibility; behavior is
    additive direction perturbation, not direct mean replacement)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import h5py
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
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(repo_dir, "results")
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


def _format_alpha(alpha: float) -> str:
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace(".", "p")


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
    mode_confidence_means: Dict[str, Optional[float]],
    mode_confidence_counts: Dict[str, int],
    finished_at: str,
) -> None:
    lines = [
        "Verbalised Confidence Probe Direction Configuration",
        "===============================================",
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
        f"probe_dir={args.probe_dir}",
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
        f"ablation_targets={args.ablation_targets}",
        f"alpha={args.alpha}",
        "probability_tokens_mean_replace_behavior=additive_direction_perturbation",
        "direction_source=verbalised_confidence_probe_weight",
        "direction_normalization=theta_div_norm_theta_squared",
        "confidence_direction_expectation_for_low_targets=perturbed_confidence_gt_none",
        "confidence_direction_expectation_for_high_targets=perturbed_confidence_lt_none",
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"parse_mode_verbalised_confidence={args.parse_mode_verbalised_confidence}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Summary: Mean Parsed Verbalised Confidence]",
    ]
    for mode_key in sorted(mode_confidence_means.keys()):
        mean_val = mode_confidence_means[mode_key]
        count_val = int(mode_confidence_counts.get(mode_key, 0))
        if mean_val is None:
            lines.append(f"{mode_key}=None ({count_val})")
        else:
            lines.append(f"{mode_key}={mean_val:.6f} ({count_val})")
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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


def _absolute_prob_positions(
    prompt_len: int,
    decoded_tokens: List[str],
    *,
    expected_probability_tokens: int,
) -> List[int]:
    parsed = parse_guess_and_probability_indices(decoded_tokens)
    if parsed is None:
        return []
    _, first_prob, _ = parsed
    end_prob = first_prob + expected_probability_tokens
    if end_prob > len(decoded_tokens):
        return []
    return [prompt_len + pos for pos in range(first_prob, end_prob)]


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
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(max_new_tokens):
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


def build_direction_perturb_hooks(
    layer_to_delta: Dict[int, torch.Tensor],
    *,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    expected_probability_tokens: int,
) -> List[Tuple[str, Callable]]:
    num_prob_tokens = next(iter(layer_to_delta.values())).shape[0]
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_to_delta:
        hook_name = f"blocks.{layer}.hook_resid_post"

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                abs_positions = _absolute_prob_positions(
                    prompt_len,
                    decoded_tokens_provider(),
                    expected_probability_tokens=expected_probability_tokens,
                )
                if len(abs_positions) != num_prob_tokens:
                    return activation
                delta = layer_to_delta[layer_idx].to(activation.dtype)
                for pos_i, abs_pos in enumerate(abs_positions):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = activation[:, abs_pos, :] + delta[pos_i]
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_direction_perturbed(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_to_delta: Dict[int, torch.Tensor],
    expected_probability_tokens: int,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []
    eos_id = model.tokenizer.eos_token_id

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_direction_perturb_hooks(
        layer_to_delta=layer_to_delta,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        expected_probability_tokens=expected_probability_tokens,
    )
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
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


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _collect_low_high_ids(
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
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    return low_ids, high_ids


def _parse_probability_token_dir_name(dirname: str) -> Optional[int]:
    match = re.fullmatch(r"tok_(\d+)_(probability|prob)", dirname)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_probability_token_dirs(probe_dir: Path, expected_probability_tokens: int) -> List[Path]:
    if not probe_dir.exists() or not probe_dir.is_dir():
        raise ValueError(f"Probe directory does not exist or is not a directory: {probe_dir}")
    token_entries: List[Tuple[int, Path]] = []
    for child in probe_dir.iterdir():
        if not child.is_dir():
            continue
        token_idx = _parse_probability_token_dir_name(child.name)
        if token_idx is not None:
            token_entries.append((token_idx, child))
    if len(token_entries) != expected_probability_tokens:
        raise ValueError(
            f"Found {len(token_entries)} probability token directories in {probe_dir}; "
            f"expected exactly {expected_probability_tokens}."
        )
    token_entries.sort(key=lambda x: x[0])
    token_indices = [idx for idx, _ in token_entries]
    zero_based_expected = list(range(expected_probability_tokens))
    one_based_expected = list(range(1, expected_probability_tokens + 1))
    if token_indices not in (zero_based_expected, one_based_expected):
        raise ValueError(
            "Probability token directory indices must be contiguous and either zero-based "
            f"{zero_based_expected} or one-based {one_based_expected}; got {token_indices}."
        )
    return [path for _, path in token_entries]


def _load_probe_direction_for_layer_and_token(
    probe_file: Path,
    *,
    hidden_dim: int,
) -> np.ndarray:
    if not probe_file.exists():
        raise ValueError(f"Missing probe file: {probe_file}")
    with open(probe_file, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Probe payload at {probe_file} must be a dict containing 'model'.")
    model = payload["model"]
    if not hasattr(model, "coef_"):
        raise ValueError(f"Probe model at {probe_file} has no coef_ attribute.")
    theta = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    if theta.shape[0] != hidden_dim:
        raise AssertionError(
            f"Probe weight dim mismatch at {probe_file}: coef dim={theta.shape[0]} vs hidden_dim={hidden_dim}."
        )
    norm_sq = float(np.dot(theta, theta))
    if norm_sq <= 0.0:
        raise ValueError(f"Probe weight at {probe_file} has zero norm; cannot normalise direction.")
    return (theta / norm_sq).astype(np.float32)


def load_probe_directions(
    probe_dir: Path,
    *,
    ablate_layers: Sequence[int],
    expected_probability_tokens: int,
    hidden_dim: int,
) -> Dict[int, np.ndarray]:
    token_dirs = _resolve_probability_token_dirs(probe_dir, expected_probability_tokens)
    layer_to_direction: Dict[int, np.ndarray] = {}
    for layer_idx in ablate_layers:
        per_token: List[np.ndarray] = []
        layer_name = f"layer_{layer_idx + 1}"
        for token_dir in token_dirs:
            probe_path = token_dir / layer_name / "verbalised_confidence_probe.pkl"
            token_direction = _load_probe_direction_for_layer_and_token(probe_path, hidden_dim=hidden_dim)
            per_token.append(token_direction)
        layer_to_direction[layer_idx] = np.stack(per_token, axis=0).astype(np.float32)
    return layer_to_direction


def main() -> None:
    TRAIN_RATIO = 0.9
    parser = argparse.ArgumentParser(
        description="Verbalised-confidence probe direction intervention inference (TransformerLens)."
    )
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="verbalised_confidence_probes/results/mult_toks_all_layers/7_200",
        help="Directory containing tok_n_probability/layer_k/verbalised_confidence_probe.pkl files.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
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
        help="Inclusive range '12-15' or comma list '12,13,14,15' (zero-indexed).",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=["none", "probability_tokens_mean_replace"],
        choices=["none", "probability_tokens_mean_replace"],
        help="Only baseline none and probability_tokens_mean_replace are supported in this script.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        required=True,
        help="One or more alpha values in [0, 1].",
    )
    parser.add_argument(
        "--ablation_targets",
        type=str,
        nargs="+",
        required=True,
        choices=["low", "high"],
        help="One or both targets to perturb: low, high.",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--expected_probability_tokens", type=int, default=6)
    parser.add_argument(
        "--parse_mode_verbalised_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse verbalised confidence from generated responses and report aggregate means.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "linear_probe_direction/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline comparisons.")
    if "probability_tokens_mean_replace" not in args.ablation_mode:
        raise ValueError("Please include 'probability_tokens_mean_replace' in --ablation_mode.")
    for alpha in args.alpha:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}.")
    args.ablation_targets = _dedupe_preserve_order(args.ablation_targets)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    hidden_dim = int(model.cfg.d_model)

    examples_h5 = load_examples_h5(Path(args.input_h5))
    low_ids, high_ids = _collect_low_high_ids(
        examples_h5,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
    )
    probe_directions = load_probe_directions(
        Path(args.probe_dir),
        ablate_layers=ablate_layers,
        expected_probability_tokens=args.expected_probability_tokens,
        hidden_dim=hidden_dim,
    )

    logging.info(
        "Loaded %d H5 examples. low_conf=%d (<=%.3f), high_conf=%d (>=%.3f), layers=%s",
        len(examples_h5),
        len(low_ids),
        args.low_conf_threshold,
        len(high_ids),
        args.high_conf_threshold,
        ablate_layers,
    )

    out_path = resolve_output_json_path(args.output_json)
    results = {"train": {}, "validation": {}}
    mini_results = {"train": {}, "validation": {}}
    mode_confidence_values: Dict[str, List[float]] = {"no_replacement": []}
    for target in args.ablation_targets:
        for alpha in args.alpha:
            key = f"probability_tokens_mean_replace__target_{target}__alpha_{_format_alpha(alpha)}"
            mode_confidence_values[key] = []

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        split_target = round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(
            args.num_samples * (1 - TRAIN_RATIO)
        )
        id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
        target_union_ids: set[str] = set()
        if "low" in args.ablation_targets:
            target_union_ids.update(low_ids)
        if "high" in args.ablation_targets:
            target_union_ids.update(high_ids)
        split_target_ids = sorted(ex_id for ex_id in target_union_ids if ex_id in id_to_index)
        selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
        logging.info("Generating for %d examples (%s split).", len(selected_ids), split_name)

        for i, ex_id in enumerate(selected_ids):
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                continue
            example = eval_ds[int(ds_idx)]
            local_prompt = fewshot_prefix + CONFIDENCE_PROMPT + example["question"]
            entry = {"question": example["question"]}
            mini_entry = {"question": example["question"]}

            baseline_response, baseline_decoded_tokens = greedy_generate(
                model=model,
                local_prompt=local_prompt,
                max_new_tokens=args.model_max_new_tokens,
                fwd_hooks=None,
            )
            baseline_confidence = (
                parse_mode_confidence_from_response(baseline_response) if args.parse_mode_verbalised_confidence else None
            )
            entry["no_replacement"] = {
                "response": baseline_response,
                "decoded_tokens": baseline_decoded_tokens,
            }
            mini_entry["no_replacement"] = {"response": baseline_response}
            if args.parse_mode_verbalised_confidence:
                entry["no_replacement"]["verbalised_confidence"] = baseline_confidence
                mini_entry["no_replacement"]["verbalised_confidence"] = baseline_confidence
                if baseline_confidence is not None:
                    mode_confidence_values["no_replacement"].append(float(baseline_confidence))

            ex_is_low = ex_id in low_ids
            ex_is_high = ex_id in high_ids

            for target in args.ablation_targets:
                if target == "low" and not ex_is_low:
                    continue
                if target == "high" and not ex_is_high:
                    continue
                sign = 1.0 if target == "low" else -1.0
                for alpha in args.alpha:
                    layer_to_delta: Dict[int, torch.Tensor] = {}
                    for layer_idx in ablate_layers:
                        direction = probe_directions[layer_idx]
                        layer_to_delta[layer_idx] = torch.tensor(sign * alpha * direction, device=device, dtype=torch_dtype)
                    response, decoded_tokens = greedy_generate_direction_perturbed(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=args.model_max_new_tokens,
                        layer_to_delta=layer_to_delta,
                        expected_probability_tokens=args.expected_probability_tokens,
                    )
                    mode_confidence = (
                        parse_mode_confidence_from_response(response) if args.parse_mode_verbalised_confidence else None
                    )

                    key = f"probability_tokens_mean_replace__target_{target}__alpha_{_format_alpha(alpha)}"
                    entry[key] = {"response": response, "decoded_tokens": decoded_tokens}
                    mini_entry[key] = {"response": response}
                    responses_identical = response == baseline_response
                    entry[key]["responses_identical"] = responses_identical
                    mini_entry[key]["responses_identical"] = responses_identical
                    if args.parse_mode_verbalised_confidence:
                        entry[key]["verbalised_confidence"] = mode_confidence
                        mini_entry[key]["verbalised_confidence"] = mode_confidence
                        if mode_confidence is not None:
                            mode_confidence_values[key].append(float(mode_confidence))
                        if mode_confidence is None or baseline_confidence is None:
                            meets_none_confidence_direction = None
                        elif target == "low":
                            meets_none_confidence_direction = mode_confidence > baseline_confidence
                        else:
                            meets_none_confidence_direction = mode_confidence < baseline_confidence
                        entry[key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                        mini_entry[key]["meets_none_confidence_direction"] = meets_none_confidence_direction

            results[split_name][ex_id] = entry
            mini_results[split_name][ex_id] = mini_entry
            logging.info("[%s %d/%d] %s first line: %r", split_name, i + 1, len(selected_ids), ex_id, baseline_response[:120])

    mode_confidence_means: Dict[str, Optional[float]] = {}
    mode_confidence_counts: Dict[str, int] = {}
    for mode_key, values in mode_confidence_values.items():
        mode_confidence_means[mode_key] = float(np.mean(values)) if values else None
        mode_confidence_counts[mode_key] = len(values)

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
        mode_confidence_means=mode_confidence_means,
        mode_confidence_counts=mode_confidence_counts,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    logging.info("Wrote %s", out_path)
    logging.info("Wrote %s", mini_out_path)
    logging.info("Wrote %s", config_out_path)


if __name__ == "__main__":
    main()
