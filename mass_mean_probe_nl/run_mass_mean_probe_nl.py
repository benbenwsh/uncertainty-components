#!/usr/bin/env python3
# Dependencies: torch, datasets, transformer_lens, h5py, numpy.
"""
Natural-language mass mean-direction probing via TransformerLens.

This variant removes output-format parsing and instead uses prompt/generation-position
ablation modes driven by prompt and probability-last-token mean directions. With
--probability_last_direction_only, only the probability-last-token direction is used
for estimation and at all ablated positions.
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
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import h5py
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.weight_processing import ProcessWeights

# NATURAL_LANGUAGE_PROMPT = (
#     "Answer the following question.\n"
#     "Question: "
# )
NATURAL_LANGUAGE_PROMPT = (
    "Answer the following question, and express your confidence in your answer.\n"
    "Question: "
)

BRIEF_PROMPTS = {
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "<end_of_turn>"]

ABLATION_MODES = [
    "none",
    "all_tokens_mean_replace",
    "all_prompt_tokens_mean_replace",
    "all_prompt_and_first_generated_mean_replace",
    "first_generated_token_mean_replace",
    "current_generated_token_mean_replace",
    "current_generated_window5_mean_replace",
    "generated_tokens_mean_replace",
]

_NEW_H5_REQUIRED_COMPONENTS = ("res", "attn", "mlp")
_EMBEDDING_FIELD_PROMPT = "embeddings_mean_prompt"
_EMBEDDING_FIELD_PROBABILITY = "embeddings_probability"


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
        out_dir = os.path.dirname(os.path.abspath(cli_output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
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


def mode_to_output_key(mode: str) -> str:
    if mode == "none":
        return "no_replacement"
    if mode in ABLATION_MODES:
        return mode
    raise ValueError(f"Unknown mode: {mode}")


def _format_alpha(alpha: float) -> str:
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace(".", "p")


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
    finished_at: str,
) -> None:
    lines = [
        "Natural Language Mass Mean Probe Configuration",
        "==============================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        "",
        "[Data]",
        f"input_h5={args.input_h5}",
        f"dataset={args.dataset}",
        f"new_h5_format={args.new_h5_format}",
        f"h5_example_count={h5_example_count}",
        f"random_seed={args.random_seed}",
        f"num_samples={args.num_samples}",
        f"num_few_shot={args.num_few_shot}",
        f"fewshot_prompt_indices={','.join(str(i) for i in prompt_indices)}",
        "",
        "[Prompt/Generation]",
        f"natural_language_prompt={NATURAL_LANGUAGE_PROMPT}",
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
        "non_none_mode_behavior=additive_direction_perturbation",
        "direction_definition=high_mean_minus_low_mean",
        f"probability_last_direction_only={args.probability_last_direction_only}",
        (
            "direction_fields=embeddings_probability[-1]"
            if args.probability_last_direction_only
            else "direction_fields=embeddings_mean_prompt,embeddings_probability[-1]"
        ),
        (
            "direction_position_rule=all_positions_use_probability_last"
            if args.probability_last_direction_only
            else "direction_position_rule=positions_lt_prompt_len_minus_1_use_prompt_else_probability_last"
        ),
        f"ablate_layers_spec={args.ablate_layers}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
    ]
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
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
        if "PyPreTokenizerTypeWrapper" not in str(exc):
            raise
        logging.warning("Fast tokenizer load failed (%s). Retrying with use_fast=False.", exc)
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


def _extract_res_field(resp0: dict, ex_id: str, field_name: str, *, new_h5_format: bool):
    value = resp0.get(field_name)
    if not new_h5_format:
        return value
    if not isinstance(value, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be a dict with "
            f"keys {_NEW_H5_REQUIRED_COMPONENTS} when --new_h5_format is set."
        )
    for component in _NEW_H5_REQUIRED_COMPONENTS:
        if component not in value:
            raise ValueError(
                f"Example {ex_id} responses/0/{field_name} missing component '{component}' "
                f"(required when --new_h5_format is set)."
            )
        if value[component] is None:
            raise ValueError(
                f"Example {ex_id} responses/0/{field_name}/{component} is null "
                f"(must be populated when --new_h5_format is set)."
            )
    return value["res"]


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


def _as_layer_hidden(arr_like: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 4:
        return arr[:, 0, -1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected embedding tensor shape: {arr.shape}; expected 4D or 2D.")


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
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks or [])
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def _positions_for_mode(mode: str, *, prompt_len: int, seq_len: int) -> List[int]:
    if mode == "all_tokens_mean_replace":
        return list(range(seq_len))
    if mode == "all_prompt_tokens_mean_replace":
        return list(range(prompt_len-1))
    if mode == "all_prompt_and_first_generated_mean_replace":
        return list(range(prompt_len))
    if mode == "first_generated_token_mean_replace":
        return [prompt_len-1]
    if mode == "current_generated_token_mean_replace":
        return [seq_len - 1]
    if mode == "current_generated_window5_mean_replace":
        positions = list(range(max(0, seq_len - 5), seq_len))
        return positions
    if mode == "generated_tokens_mean_replace":
        return list(range(prompt_len - 1, seq_len))
    raise ValueError(f"Unknown ablation mode for perturb hooks: {mode!r}")


def _direction_mode_activation_applier_builder(
    mode: str,
    *,
    prompt_len: int,
    probability_last_direction_only: bool = False,
) -> Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
    if mode not in ABLATION_MODES or mode == "none":
        raise ValueError(f"Unknown ablation mode for perturb hooks: {mode!r}")

    def _apply_mode(
        activation: torch.Tensor, # (batch_size, seq_len, d_model)
        layer_delta: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        probability_last_delta_vec = layer_delta["probability_last"].to(activation.dtype)
        prompt_delta_vec = (
            probability_last_delta_vec
            if probability_last_direction_only
            else layer_delta["prompt"].to(activation.dtype)
        )
        seq_len = int(activation.shape[1])
        for abs_pos in _positions_for_mode(mode, prompt_len=prompt_len, seq_len=seq_len):
            if 0 <= abs_pos < seq_len:
                if probability_last_direction_only:
                    delta_vec = probability_last_delta_vec
                else:
                    delta_vec = prompt_delta_vec if abs_pos < prompt_len - 1 else probability_last_delta_vec
                activation[:, abs_pos, :] = activation[:, abs_pos, :] + delta_vec
        return activation

    return _apply_mode


def build_direction_perturb_hooks(
    layer_to_delta: Dict[int, Dict[str, torch.Tensor]],
    *,
    mode: str,
    prompt_len: int,
    probability_last_direction_only: bool = False,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    activation_applier = _direction_mode_activation_applier_builder(
        mode,
        prompt_len=prompt_len,
        probability_last_direction_only=probability_last_direction_only,
    )
    for layer in layer_to_delta:
        hook_name = f"blocks.{layer}.hook_resid_post"

        def _make_hook(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                return activation_applier(activation, layer_to_delta[layer_idx])

            return hook_fn

        hooks.append((hook_name, _make_hook(layer)))
    return hooks


def greedy_generate_direction_perturbed(
    model: HookedTransformer,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_to_delta: Dict[int, Dict[str, torch.Tensor]],
    mode: str,
    probability_last_direction_only: bool = False,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []
    hooks = build_direction_perturb_hooks(
        layer_to_delta=layer_to_delta,
        mode=mode,
        prompt_len=prompt_len,
        probability_last_direction_only=probability_last_direction_only,
    )
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            logging.info(f"decoded_tokens: {decoded_tokens}")
            decoded_tokens.append(model.tokenizer.decode([next_id], skip_special_tokens=False))
            next_t = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_t], dim=-1)
            if _should_stop_generation("".join(decoded_tokens), next_id, model.tokenizer):
                break
    return _postprocess_response_from_full_decode(model, tokens, local_prompt), decoded_tokens


def compute_low_high_mixed_means_and_directions(
    examples_h5: Dict[str, dict],
    ablate_layers: Sequence[int],
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
    new_h5_format: bool = False,
    probability_last_direction_only: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], set[str], set[str]]:
    direction_keys = ("probability_last",) if probability_last_direction_only else ("prompt", "probability_last")
    low_vectors: Dict[str, List[np.ndarray]] = {key: [] for key in direction_keys}
    high_vectors: Dict[str, List[np.ndarray]] = {key: [] for key in direction_keys}
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    resid_post_layers = np.asarray(ablate_layers) + 1

    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError(f"Example {ex_id} must have exactly one response, got {0 if responses is None else len(responses)}.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 is not a dict.")

        conf = float(resp0.get("verbalised_confidence"))
        is_low = conf <= low_conf_threshold
        is_high = conf >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)
        if not (is_low or is_high):
            continue

        emb_probability = _extract_res_field(
            resp0, ex_id, _EMBEDDING_FIELD_PROBABILITY, new_h5_format=new_h5_format
        )
        if emb_probability is None:
            raise ValueError(
                f"Example {ex_id} is missing required field {_EMBEDDING_FIELD_PROBABILITY}."
            )
        if not isinstance(emb_probability, (list, tuple)) or len(emb_probability) == 0:
            raise ValueError(f"Example {ex_id} field {_EMBEDDING_FIELD_PROBABILITY} must be a non-empty list.")
        probability_last_selected = _as_layer_hidden(emb_probability[-1])[resid_post_layers, :]
        prompt_selected = None
        if not probability_last_direction_only:
            emb_prompt = _extract_res_field(
                resp0, ex_id, _EMBEDDING_FIELD_PROMPT, new_h5_format=new_h5_format
            )
            if emb_prompt is None:
                raise ValueError(
                    f"Example {ex_id} is missing required field {_EMBEDDING_FIELD_PROMPT}."
                )
            prompt_selected = _as_layer_hidden(emb_prompt)[resid_post_layers, :]
        if is_low:
            if prompt_selected is not None:
                low_vectors["prompt"].append(prompt_selected)
            low_vectors["probability_last"].append(probability_last_selected)
        if is_high:
            if prompt_selected is not None:
                high_vectors["prompt"].append(prompt_selected)
            high_vectors["probability_last"].append(probability_last_selected)

    count_key = "probability_last" if probability_last_direction_only else "prompt"
    if not low_vectors[count_key]:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_vectors[count_key]:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")

    mean_low: Dict[str, np.ndarray] = {}
    mean_high: Dict[str, np.ndarray] = {}
    direction: Dict[str, np.ndarray] = {}
    for key in direction_keys:
        mean_low[key] = np.mean(np.stack(low_vectors[key], axis=0), axis=0).astype(np.float32)
        mean_high[key] = np.mean(np.stack(high_vectors[key], axis=0), axis=0).astype(np.float32)
        direction[key] = (mean_high[key] - mean_low[key]).astype(np.float32)
    return mean_low, mean_high, direction, low_ids, high_ids


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    train_ratio = 0.9
    parser = argparse.ArgumentParser(description="Natural-language mass mean direction probe inference (TransformerLens).")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, expect each embedding field to be a dict with 'res'/'attn'/'mlp' and read from 'res'."
        ),
    )
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
        help="Inclusive range '12-15' or comma list '12,13,14,15' (zero-indexed).",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=ABLATION_MODES,
        choices=ABLATION_MODES,
        help="Ablation mode(s) to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        required=True,
        help="One or more real-valued alpha scale factors (perturbation strength along the direction).",
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
    parser.add_argument(
        "--probability_last_direction_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only probability-last-token mean direction for estimation and at all ablated positions.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output path. If unset, saves to "
            "mass_mean_probe_nl/results/<incrementing_run_id>/ablation_results.json"
        ),
    )
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if "none" not in args.ablation_mode:
        raise ValueError("Please include 'none' in --ablation_mode for baseline comparisons.")
    args.ablation_targets = _dedupe_preserve_order(args.ablation_targets)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    alphas_outside_unit = [a for a in args.alpha if a < 0.0 or a > 1.0]
    if alphas_outside_unit:
        logging.warning(
            "Alpha values outside [0, 1] will be used as-is: %s",
            alphas_outside_unit,
        )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_eval_dataset(args.dataset, args.random_seed)
    random.seed(args.random_seed)
    answerable_train = split_answerable_indices(train_ds)
    if len(answerable_train) < args.num_few_shot:
        raise ValueError("Not enough answerable training examples for few-shot.")
    prompt_indices = random.sample(answerable_train, args.num_few_shot)

    brief = BRIEF_PROMPTS[args.brief_prompt] if args.enable_brief else ""
    fewshot_prefix = construct_fewshot_prompt_from_indices(
        train_ds,
        prompt_indices,
        brief=brief,
        brief_always=args.brief_always,
        use_context=args.use_context,
    )

    model = load_hooked_transformer(args.model_name, device=device, torch_dtype=torch_dtype)
    ablate_layers = parse_ablate_layers(args.ablate_layers, model.cfg.n_layers)
    out_path = resolve_output_json_path(args.output_json)

    examples_h5 = load_examples_h5(Path(args.input_h5))
    _, _, directions, low_ids, high_ids = compute_low_high_mixed_means_and_directions(
        examples_h5=examples_h5,
        ablate_layers=ablate_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        new_h5_format=args.new_h5_format,
        probability_last_direction_only=args.probability_last_direction_only,
    )
    if args.probability_last_direction_only:
        logging.info(
            "Computed probability-last-only directions from %d low and %d high examples.",
            len(low_ids),
            len(high_ids),
        )
    else:
        logging.info(
            "Computed prompt and probability-last directions from %d low and %d high examples.",
            len(low_ids),
            len(high_ids),
        )

    results = {"train": {}, "validation": {}}
    mini_results = {"train": {}, "validation": {}}
    non_none_modes = [m for m in args.ablation_mode if m != "none"]

    for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
        split_target = round(args.num_samples * train_ratio) if split_name == "train" else round(
            args.num_samples * (1 - train_ratio)
        )
        id_to_index = {encode_example_id(ex["id"]): i for i, ex in enumerate(eval_ds)}
        target_union_ids: set[str] = set(low_ids) | set(high_ids)
        split_target_ids = sorted(ex_id for ex_id in target_union_ids if ex_id in id_to_index)
        selected_ids = split_target_ids[: min(split_target, len(split_target_ids))]
        logging.info("Generating for %d examples (%s split).", len(selected_ids), split_name)

        for i, ex_id in enumerate(selected_ids):
            ds_idx = id_to_index.get(ex_id)
            if ds_idx is None:
                continue
            example = eval_ds[int(ds_idx)]
            local_prompt = fewshot_prefix + NATURAL_LANGUAGE_PROMPT + example["question"]
            entry = {"question": example["question"]}
            mini_entry = {"question": example["question"]}

            baseline_response, baseline_decoded_tokens = greedy_generate(
                model=model,
                local_prompt=local_prompt,
                max_new_tokens=args.model_max_new_tokens,
                fwd_hooks=None,
            )
            entry["no_replacement"] = {
                "response": baseline_response,
                "decoded_tokens": baseline_decoded_tokens,
                "responses_identical": True,
            }
            mini_entry["no_replacement"] = {
                "response": baseline_response,
                "responses_identical": True,
            }

            ex_is_low = ex_id in low_ids
            ex_is_high = ex_id in high_ids
            for target in args.ablation_targets:
                if target == "low" and not ex_is_low:
                    continue
                if target == "high" and not ex_is_high:
                    continue
                sign = 1.0 if target == "low" else -1.0
                for mode in non_none_modes:
                    for alpha in args.alpha:
                        layer_to_delta: Dict[int, Dict[str, torch.Tensor]] = {}
                        for layer_i, layer_idx in enumerate(ablate_layers):
                            layer_delta: Dict[str, torch.Tensor] = {
                                "probability_last": torch.tensor(
                                    sign * alpha * directions["probability_last"][layer_i],
                                    device=device,
                                    dtype=torch_dtype,
                                ),
                            }
                            if not args.probability_last_direction_only:
                                layer_delta["prompt"] = torch.tensor(
                                    sign * alpha * directions["prompt"][layer_i],
                                    device=device,
                                    dtype=torch_dtype,
                                )
                            layer_to_delta[layer_idx] = layer_delta
                        response, decoded_tokens = greedy_generate_direction_perturbed(
                            model=model,
                            local_prompt=local_prompt,
                            max_new_tokens=args.model_max_new_tokens,
                            layer_to_delta=layer_to_delta,
                            mode=mode,
                            probability_last_direction_only=args.probability_last_direction_only,
                        )
                        key = f"{mode_to_output_key(mode)}__target_{target}__alpha_{_format_alpha(alpha)}"
                        responses_identical = response == baseline_response
                        entry[key] = {
                            "response": response,
                            "decoded_tokens": decoded_tokens,
                            "responses_identical": responses_identical,
                        }
                        mini_entry[key] = {
                            "response": response,
                            "responses_identical": responses_identical,
                        }

            results[split_name][ex_id] = entry
            mini_results[split_name][ex_id] = mini_entry
            logging.info("[%s %d/%d] %s first line: %r", split_name, i + 1, len(selected_ids), ex_id, baseline_response[:120])

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
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    logging.info("Wrote %s", out_path)
    logging.info("Wrote %s", mini_out_path)
    logging.info("Wrote %s", config_out_path)


if __name__ == "__main__":
    main()
