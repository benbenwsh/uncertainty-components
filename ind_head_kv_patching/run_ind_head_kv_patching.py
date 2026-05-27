#!/usr/bin/env python3
"""Selected-head KV patching ablation runner for structured Guess/Probability decoding."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_zero_ablation.run_blockwise_zero_ablation import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT,
    _absolute_prob_positions,
    collect_confidence_group_ids,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    greedy_generate,
    load_examples_h5,
    load_hooked_transformer,
    load_trivia_qa,
    parse_mode_confidence_from_response,
    split_answerable_indices,
)
from layerwise_mean_ablation.run_mean_ablation import _as_layer_hidden, _is_expected_or_plus_one

TRAIN_RATIO = 0.9
MODE_NONE = "none"
MODE_PROBABILITY_TOKENS = "probability_tokens_mean_replace"
MODE_ANS_ADJ_TOKENS = "ans_adj_tokens_mean_replace"


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path(__file__).resolve().parent / "results"
        base.mkdir(parents=True, exist_ok=True)
        run_id = 1
        while (base / str(run_id)).exists():
            run_id += 1
        run_dir = base / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(run_dir / "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def mini_output_json_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "ablation_results_mini.json")


def config_txt_path(full_output_path: str) -> str:
    return os.path.join(os.path.dirname(full_output_path), "config.txt")


def mode_to_output_key(mode: str) -> str:
    if mode == MODE_NONE:
        return "no_replacement"
    if mode == MODE_PROBABILITY_TOKENS:
        return "probability_tokens_mean_replace"
    if mode == MODE_ANS_ADJ_TOKENS:
        return "ans_adj_tokens_mean_replace"
    raise ValueError(f"Unsupported mode: {mode}")


def resolve_n_key_value_heads(model_cfg, n_heads: int) -> int:
    for attr in (
        "n_key_value_heads",
        "num_key_value_heads",
        "n_kv_heads",
        "num_kv_heads",
        "n_heads_kv",
    ):
        value = getattr(model_cfg, attr, None)
        if value is not None:
            n_kv_heads = int(value)
            if n_kv_heads <= 0:
                raise ValueError(f"Invalid {attr}={n_kv_heads}. Must be > 0.")
            if n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"Invalid GQA head configuration: n_heads={n_heads} is not divisible by {attr}={n_kv_heads}."
                )
            return n_kv_heads
    return int(n_heads)


def parse_layer_head_pairs(spec: str, *, n_layers: int, n_kv_heads: int) -> Dict[int, List[int]]:
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("--ablate_heads_by_layer must be a non-empty comma-separated <layer>.<head> list.")
    out: Dict[int, Set[int]] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if item.count(".") != 1:
            raise ValueError(
                f"Invalid head token {item!r}. Expected format <layer_idx>.<head_idx>, e.g. 12.3."
            )
        layer_str, head_str = item.split(".", 1)
        try:
            layer_idx = int(layer_str)
            head_idx = int(head_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid head token {item!r}. layer/head must be integers in <layer_idx>.<head_idx>."
            ) from exc
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {n_layers}).")
        if head_idx < 0 or head_idx >= n_kv_heads:
            raise ValueError(f"KV head index {head_idx} out of range [0, {n_kv_heads}).")
        out.setdefault(layer_idx, set()).add(head_idx)
    if not out:
        raise ValueError("No valid <layer_idx>.<head_idx> entries found in --ablate_heads_by_layer.")
    return {layer: sorted(heads) for layer, heads in sorted(out.items())}


def format_layer_head_pairs(layer_to_heads: Dict[int, Sequence[int]]) -> str:
    parts: List[str] = []
    for layer_idx in sorted(layer_to_heads):
        for head_idx in sorted(set(int(h) for h in layer_to_heads[layer_idx])):
            parts.append(f"{layer_idx}.{head_idx}")
    return ",".join(parts)


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    device: str,
    model_n_layers: int,
    n_kv_heads: int,
    ablate_layers: Sequence[int],
    selected_layer_heads: Dict[int, Sequence[int]],
    prompt_indices: Sequence[int],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    mode_confidence_means: Dict[str, Optional[float]],
    mode_confidence_counts: Dict[str, int],
    mode_responses_identical_true: Dict[str, int],
    finished_at: str,
) -> None:
    source_group = "low_confidence" if args.mean_from_low_confidence else "high_confidence"
    target_group = "high_confidence" if args.mean_from_low_confidence else "low_confidence"
    lines = [
        "Individual Head KV Patching Configuration",
        "========================================",
        "",
        "[Model]",
        f"model_name={args.model_name}",
        f"model_n_layers={model_n_layers}",
        f"device={device}",
        f"dtype={args.dtype}",
        f"model_n_kv_heads={n_kv_heads}",
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
        f"fixed_patch_strategy=span_key_positions",
        f"ablation_mode={args.ablation_mode}",
        f"ablate_v_only={args.ablate_v_only}",
        f"ablate_heads_by_layer_spec={args.ablate_heads_by_layer}",
        f"ablate_heads_by_layer_resolved={format_layer_head_pairs(selected_layer_heads)}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={source_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_resolved={','.join(str(layer) for layer in ablate_layers)}",
        f"num_ablated_layers={len(ablate_layers)}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_selected_count={low_conf_count}",
        f"high_conf_selected_count={high_conf_count}",
        "",
        "[Mode Confidence Metrics]",
    ]
    for mode_name in args.ablation_mode:
        mode_mean = mode_confidence_means.get(mode_name)
        valid_count = int(mode_confidence_counts.get(mode_name, 0))
        if mode_name == MODE_NONE:
            lines.append(f"{mode_name}={'None' if mode_mean is None else f'{mode_mean:.6f}'} ({valid_count})")
        else:
            identical_n = int(mode_responses_identical_true.get(mode_name, 0))
            lines.append(
                f"{mode_name}={'None' if mode_mean is None else f'{mode_mean:.6f}'} "
                f"({valid_count}) [responses_identical: {identical_n}]"
            )
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _validate_kv_probability_field(resp0: dict, ex_id: str, component: str):
    emb_prob = resp0.get("embeddings_probability")
    if not isinstance(emb_prob, dict):
        raise ValueError(f"Example {ex_id} responses/0/embeddings_probability must be an object.")
    if component not in emb_prob:
        raise ValueError(f"Example {ex_id} responses/0/embeddings_probability missing key '{component}'.")
    value = emb_prob.get(component)
    if value is None:
        raise ValueError(f"Example {ex_id} responses/0/embeddings_probability/{component} is null.")
    if not isinstance(value, list):
        raise ValueError(f"Example {ex_id} responses/0/embeddings_probability/{component} must be a list.")
    return value


def compute_kv_probability_group_means(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_probability_tokens: int,
) -> Tuple[Dict[str, np.ndarray], set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()
    k_vectors: List[np.ndarray] = []
    v_vectors: List[np.ndarray] = []
    layer_idx = np.asarray(ablate_layers)

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

        prob = resp0.get("verbalised_confidence")
        if prob is None:
            raise ValueError(f"Example {ex_id} responses/0/verbalised_confidence is missing.")
        prob = float(prob)
        is_low = prob <= low_conf_threshold
        is_high = prob >= high_conf_threshold
        if is_low:
            low_ids.add(ex_id)
        if is_high:
            high_ids.add(ex_id)

        use_for_mean = is_low if mean_from_low_confidence else is_high
        if not use_for_mean:
            continue

        emb_prob_k = _validate_kv_probability_field(resp0, ex_id, "k")
        emb_prob_v = _validate_kv_probability_field(resp0, ex_id, "v")
        if not _is_expected_or_plus_one(len(emb_prob_k), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/k has length {len(emb_prob_k)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        if not _is_expected_or_plus_one(len(emb_prob_v), expected_probability_tokens):
            raise ValueError(
                f"Example {ex_id} embeddings_probability/v has length {len(emb_prob_v)}; "
                f"expected {expected_probability_tokens} or {expected_probability_tokens + 1}."
            )
        emb_prob_k = emb_prob_k[:expected_probability_tokens]
        emb_prob_v = emb_prob_v[:expected_probability_tokens]

        k_selected: List[np.ndarray] = []
        v_selected: List[np.ndarray] = []
        for tok_arr_k in emb_prob_k:
            k_selected.append(_as_layer_hidden(tok_arr_k)[layer_idx, :])
        for tok_arr_v in emb_prob_v:
            v_selected.append(_as_layer_hidden(tok_arr_v)[layer_idx, :])
        k_vectors.append(np.stack(k_selected, axis=1))
        v_vectors.append(np.stack(v_selected, axis=1))

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not k_vectors or not v_vectors:
        threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
        operator = "<=" if mean_from_low_confidence else ">="
        raise ValueError(
            f"No {source_name} examples with non-null embeddings_probability/k and /v found "
            f"under threshold condition verbalised_confidence {operator} {threshold}."
        )

    return (
        {
            "k_probability": np.mean(np.stack(k_vectors, axis=0), axis=0).astype(np.float32),
            "v_probability": np.mean(np.stack(v_vectors, axis=0), axis=0).astype(np.float32),
        },
        low_ids,
        high_ids,
    )


def _build_layer_kv_means(
    kv_means: Dict[str, np.ndarray],
    *,
    ablate_layers: Sequence[int],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[int, Dict[str, torch.Tensor]]:
    layer_to_means: Dict[int, Dict[str, torch.Tensor]] = {}
    for local_idx, layer_i in enumerate(ablate_layers):
        layer_to_means[int(layer_i)] = {
            "k_probability": torch.tensor(kv_means["k_probability"][local_idx], device=device, dtype=torch_dtype),
            "v_probability": torch.tensor(kv_means["v_probability"][local_idx], device=device, dtype=torch_dtype),
        }
    return layer_to_means


def _reshape_kv_vector_for_activation(
    vector: torch.Tensor,
    activation: torch.Tensor,
    layer: int,
    kv_name: str,
) -> torch.Tensor:
    heads = int(activation.shape[-2])
    d_head = int(activation.shape[-1])
    expected = heads * d_head
    vector = vector.to(device=activation.device, dtype=activation.dtype)
    if int(vector.numel()) == expected:
        return vector.reshape(heads, d_head)
    if int(vector.numel()) == d_head:
        return vector.unsqueeze(0).repeat(heads, 1)
    raise ValueError(
        f"Layer {layer} {kv_name} replacement size mismatch: vector numel={vector.numel()}, "
        f"activation per-position shape=({heads}, {d_head})."
    )


def _build_patch_plan(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_means: Dict[str, torch.Tensor],
    expected_probability_tokens: int,
) -> Optional[List[Tuple[int, torch.Tensor, torch.Tensor]]]:
    abs_positions = _absolute_prob_positions(
        prompt_len,
        decoded_tokens,
        expected_probability_tokens=expected_probability_tokens,
    )
    if not abs_positions:
        return None
    k_probability = layer_means["k_probability"]
    v_probability = layer_means["v_probability"]
    if len(abs_positions) != int(k_probability.shape[0]):
        return None

    if mode == MODE_PROBABILITY_TOKENS:
        selected_positions = abs_positions
    elif mode == MODE_ANS_ADJ_TOKENS:
        selected_positions = abs_positions[:2]
    else:
        raise ValueError(f"Unsupported ablation mode for patch planning: {mode}")
    if not selected_positions:
        return None
    return [(abs_pos, k_probability[j], v_probability[j]) for j, abs_pos in enumerate(selected_positions)]


def _is_patch_active(mode: str, prompt_len: int, decoded_tokens: List[str], expected_probability_tokens: int) -> bool:
    if mode == MODE_NONE:
        return False
    if mode not in {MODE_PROBABILITY_TOKENS, MODE_ANS_ADJ_TOKENS}:
        raise ValueError(f"Unsupported ablation mode for activation: {mode}")
    return bool(
        _absolute_prob_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=expected_probability_tokens,
        )
    )


def build_kv_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    mode: str,
    prompt_len: int,
    expected_probability_tokens: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    ablate_v_only: bool,
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        layer_means = layer_to_means[layer]
        selected_heads = [int(h) for h in selected_heads_by_layer.get(int(layer), [])]
        if not selected_heads:
            raise ValueError(f"No selected KV heads for layer {layer}.")

        hook_k_name = f"blocks.{layer}.attn.hook_k"
        hook_v_name = f"blocks.{layer}.attn.hook_v"

        def _make_k_hook(
            local_layer_means: Dict[str, torch.Tensor], local_layer: int, local_heads: List[int]
        ) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if ablate_v_only:
                    return activation
                decoded_tokens = decoded_tokens_provider()
                if not _is_patch_active(mode, prompt_len, decoded_tokens, expected_probability_tokens):
                    return activation
                patch_plan = _build_patch_plan(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_means=local_layer_means,
                    expected_probability_tokens=expected_probability_tokens,
                )
                if patch_plan is None:
                    return activation
                n_heads = int(activation.shape[2])
                for head_idx in local_heads:
                    if head_idx < 0 or head_idx >= n_heads:
                        raise ValueError(
                            f"KV head index {head_idx} out of range for layer {local_layer} hook_k with {n_heads} heads."
                        )
                for abs_pos, k_vector, _ in patch_plan:
                    if not (0 <= abs_pos < activation.shape[1]):
                        continue
                    replacement = _reshape_kv_vector_for_activation(k_vector, activation, local_layer, "k")
                    for head_idx in local_heads:
                        activation[:, abs_pos, head_idx, :] = replacement[head_idx]
                return activation

            return hook_fn

        def _make_v_hook(
            local_layer_means: Dict[str, torch.Tensor], local_layer: int, local_heads: List[int]
        ) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                decoded_tokens = decoded_tokens_provider()
                if not _is_patch_active(mode, prompt_len, decoded_tokens, expected_probability_tokens):
                    return activation
                patch_plan = _build_patch_plan(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_means=local_layer_means,
                    expected_probability_tokens=expected_probability_tokens,
                )
                if patch_plan is None:
                    return activation
                n_heads = int(activation.shape[2])
                for head_idx in local_heads:
                    if head_idx < 0 or head_idx >= n_heads:
                        raise ValueError(
                            f"KV head index {head_idx} out of range for layer {local_layer} hook_v with {n_heads} heads."
                        )
                for abs_pos, _, v_vector in patch_plan:
                    if not (0 <= abs_pos < activation.shape[1]):
                        continue
                    replacement = _reshape_kv_vector_for_activation(v_vector, activation, local_layer, "v")
                    for head_idx in local_heads:
                        activation[:, abs_pos, head_idx, :] = replacement[head_idx]
                return activation

            return hook_fn

        hooks.append((hook_k_name, _make_k_hook(layer_means, int(layer), selected_heads)))
        hooks.append((hook_v_name, _make_v_hook(layer_means, int(layer), selected_heads)))
    return hooks


def greedy_generate_kv_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    mode: str,
    expected_probability_tokens: int,
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    selected_heads_by_layer: Dict[int, Sequence[int]],
    ablate_v_only: bool,
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_kv_mean_replace_hooks(
        layer_indices=layer_indices,
        mode=mode,
        prompt_len=prompt_len,
        expected_probability_tokens=expected_probability_tokens,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_means=layer_to_means,
        selected_heads_by_layer=selected_heads_by_layer,
        ablate_v_only=ablate_v_only,
    )
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--input_h5",
        type=str,
        default=None,
        help="Path to processed HDF5 containing responses/0/embeddings_probability/{k,v}.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_max_new_tokens", type=int, default=30)
    parser.add_argument("--brief_prompt", type=str, default="default", choices=["default", "chat"])
    parser.add_argument("--brief_always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_brief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablate_heads_by_layer", type=str, required=True)
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=[MODE_NONE, MODE_PROBABILITY_TOKENS],
        choices=[MODE_NONE, MODE_PROBABILITY_TOKENS, MODE_ANS_ADJ_TOKENS],
    )
    parser.add_argument("--ablate_v_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--mean_from_low_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--parse_mode_verbalised_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")

    if not args.input_h5:
        raise ValueError("--input_h5 is required.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)

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
    model_n_layers = int(model.cfg.n_layers)
    model_n_q_heads = int(model.cfg.n_heads)
    model_n_kv_heads = resolve_n_key_value_heads(model.cfg, model_n_q_heads)
    selected_heads_all_layers = parse_layer_head_pairs(
        args.ablate_heads_by_layer, n_layers=model_n_layers, n_kv_heads=model_n_kv_heads
    )

    run_layers = sorted(selected_heads_all_layers.keys())
    if not run_layers:
        raise ValueError("No layers selected via --ablate_heads_by_layer.")

    examples_h5 = load_examples_h5(Path(args.input_h5))
    kv_means, low_ids, high_ids = compute_kv_probability_group_means(
        examples_h5,
        ablate_layers=run_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        mean_from_low_confidence=args.mean_from_low_confidence,
        expected_probability_tokens=args.expected_probability_tokens,
    )

    low_ids_check, high_ids_check = collect_confidence_group_ids(
        examples_h5,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
    )
    if low_ids_check != low_ids or high_ids_check != high_ids:
        raise ValueError("Confidence grouping mismatch between mean builder and collector.")

    if args.mean_from_low_confidence:
        ablation_target_ids = high_ids
        target_group = "high_confidence"
    else:
        ablation_target_ids = low_ids
        target_group = "low_confidence"
    if not ablation_target_ids:
        raise ValueError(f"No examples available in ablation target group: {target_group}.")

    layer_kv_means = _build_layer_kv_means(
        kv_means,
        ablate_layers=run_layers,
        device=device,
        torch_dtype=torch_dtype,
    )
    logging.info(
        "Loaded %d H5 examples. low_conf=%d high_conf=%d target_group=%s target_ids=%d layers=%s ablate_v_only=%s",
        len(examples_h5),
        len(low_ids),
        len(high_ids),
        target_group,
        len(ablation_target_ids),
        run_layers,
        args.ablate_v_only,
    )

    def run_one_evaluation(
        layer_subset: Sequence[int],
        selected_heads_subset: Dict[int, List[int]],
    ) -> Tuple[
        Dict[str, dict],
        Dict[str, dict],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, int],
    ]:
        layer_subset_means = {int(layer_idx): layer_kv_means[int(layer_idx)] for layer_idx in layer_subset}
        results = {"train": {}, "validation": {}}
        mini_results = {"train": {}, "validation": {}}
        mode_confidence_values: Dict[str, List[float]] = {mode_name: [] for mode_name in args.ablation_mode}
        mode_responses_identical_true: Dict[str, int] = {
            mode_name: 0 for mode_name in args.ablation_mode if mode_name != MODE_NONE
        }

        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(
                args.num_samples * (1 - TRAIN_RATIO)
            )
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
                baseline_mode_confidence: Optional[float] = None
                if MODE_NONE in args.ablation_mode:
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
                    baseline_key = mode_to_output_key(MODE_NONE)
                    entry[baseline_key] = {
                        "response": baseline_response,
                        "decoded_tokens": baseline_decoded_tokens,
                    }
                    mini_entry[baseline_key] = {"response": baseline_response}
                    if args.parse_mode_verbalised_confidence:
                        entry[baseline_key]["verbalised_confidence"] = baseline_mode_confidence
                        mini_entry[baseline_key]["verbalised_confidence"] = baseline_mode_confidence
                        if baseline_mode_confidence is not None:
                            mode_confidence_values[MODE_NONE].append(float(baseline_mode_confidence))

                for mode_name in args.ablation_mode:
                    if mode_name == MODE_NONE:
                        continue
                    response, decoded_tokens = greedy_generate_kv_ablated(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=args.model_max_new_tokens,
                        layer_indices=layer_subset,
                        mode=mode_name,
                        expected_probability_tokens=args.expected_probability_tokens,
                        layer_to_means=layer_subset_means,
                        selected_heads_by_layer=selected_heads_subset,
                        ablate_v_only=args.ablate_v_only,
                    )
                    mode_confidence = (
                        parse_mode_confidence_from_response(response)
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    mode_key = mode_to_output_key(mode_name)
                    entry[mode_key] = {"response": response, "decoded_tokens": decoded_tokens}
                    mini_entry[mode_key] = {"response": response}
                    if args.parse_mode_verbalised_confidence:
                        entry[mode_key]["verbalised_confidence"] = mode_confidence
                        mini_entry[mode_key]["verbalised_confidence"] = mode_confidence
                        if mode_confidence is not None:
                            mode_confidence_values[mode_name].append(float(mode_confidence))

                    if baseline_response is not None:
                        responses_identical = response == baseline_response
                        entry[mode_key]["responses_identical"] = responses_identical
                        mini_entry[mode_key]["responses_identical"] = responses_identical
                        if responses_identical:
                            mode_responses_identical_true[mode_name] += 1
                        if args.parse_mode_verbalised_confidence:
                            if mode_confidence is None or baseline_mode_confidence is None:
                                meets_none_confidence_direction = None
                            elif args.mean_from_low_confidence:
                                meets_none_confidence_direction = mode_confidence > baseline_mode_confidence
                            else:
                                meets_none_confidence_direction = mode_confidence < baseline_mode_confidence
                            entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                            mini_entry[mode_key]["meets_none_confidence_direction"] = meets_none_confidence_direction

                    logging.info(
                        "[%s %d/%d] %s %s first line: %r",
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                        mode_key,
                        response[:120],
                    )
                results[split_name][ex_id] = entry
                mini_results[split_name][ex_id] = mini_entry

        mode_confidence_means: Dict[str, Optional[float]] = {}
        mode_confidence_counts: Dict[str, int] = {}
        for mode_name in args.ablation_mode:
            vals = mode_confidence_values[mode_name]
            mode_confidence_means[mode_name] = float(np.mean(vals)) if vals else None
            mode_confidence_counts[mode_name] = len(vals)
        return (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            mode_responses_identical_true,
        )
    selected_subset = {layer: selected_heads_all_layers[layer] for layer in run_layers if layer in selected_heads_all_layers}
    missing_layer_heads = [layer for layer in run_layers if layer not in selected_subset]
    if missing_layer_heads:
        raise ValueError(
            "No selected heads provided for layers in this run: "
            + ",".join(str(layer) for layer in missing_layer_heads)
        )
    (
        results,
        mini_results,
        mode_confidence_means,
        mode_confidence_counts,
        mode_responses_identical_true,
    ) = run_one_evaluation(run_layers, selected_subset)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(mini_output_json_path(out_path), "w", encoding="utf-8") as f:
        json.dump(mini_results, f, ensure_ascii=False, indent=2)
    write_config_txt(
        config_txt_path(out_path),
        args=args,
        device=device,
        model_n_layers=model_n_layers,
        n_kv_heads=model_n_kv_heads,
        ablate_layers=run_layers,
        selected_layer_heads=selected_subset,
        prompt_indices=prompt_indices,
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        mode_confidence_means=mode_confidence_means,
        mode_confidence_counts=mode_confidence_counts,
        mode_responses_identical_true=mode_responses_identical_true,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    logging.info("Saved full outputs to %s", out_path)


if __name__ == "__main__":
    main()

