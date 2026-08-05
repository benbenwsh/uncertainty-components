#!/usr/bin/env python3
"""KV patching ablation runner for structured Guess/Probability decoding."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockwise_zero_ablation.run_blockwise_zero_ablation import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT,
    _absolute_prob_positions,
    _absolute_probability_value_positions,
    collect_confidence_group_ids,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    greedy_generate,
    load_examples_h5,
    load_hooked_transformer,
    load_eval_dataset,
    parse_ablate_layers,
    parse_mode_confidence_from_response,
    split_answerable_indices,
)
from layerwise_mean_ablation.run_mean_ablation import _as_layer_hidden, _is_expected_or_plus_one

TRAIN_RATIO = 0.9
MODE_NONE = "none"
MODE_AFTER_PROB_PREFIX = "current_generated_token_after_prob_prefix_mean_replace"
MODE_PROBABILITY_VALUE_TOKENS = "probability_value_tokens_kv_mean_replace"
SPAN_KV_PATCH_MODES = {MODE_AFTER_PROB_PREFIX, MODE_PROBABILITY_VALUE_TOKENS}


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
    if mode == MODE_AFTER_PROB_PREFIX:
        return "current_generated_token_after_prob_prefix_kv_mean_replace"
    if mode == MODE_PROBABILITY_VALUE_TOKENS:
        return "probability_value_tokens_kv_mean_replace"
    raise ValueError(f"Unsupported mode: {mode}")


def _mode_uses_span_kv_patch(mode: str) -> bool:
    return mode in SPAN_KV_PATCH_MODES


def _mode_is_active_for_step(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    expected_probability_tokens: int,
) -> bool:
    if mode == MODE_AFTER_PROB_PREFIX:
        return bool(
            _absolute_prob_positions(
                prompt_len,
                decoded_tokens,
                expected_probability_tokens=expected_probability_tokens,
            )
        )
    if mode == MODE_PROBABILITY_VALUE_TOKENS:
        return bool(_absolute_probability_value_positions(prompt_len, decoded_tokens))
    return False


def _query_positions_for_kv_patch(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    seq_len: int,
    expected_probability_tokens: int,
) -> List[int]:
    """Query indices to recompute (last_query_span_read) or extend K/V overwrite (span_key_positions)."""
    if mode == MODE_AFTER_PROB_PREFIX:
        return [seq_len - 1]
    if mode == MODE_PROBABILITY_VALUE_TOKENS:
        return _absolute_probability_value_positions(prompt_len, decoded_tokens)
    return []


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
    mode_responses_identical_true: Dict[str, int],
    finished_at: str,
) -> None:
    source_group = "low_confidence" if args.mean_from_low_confidence else "high_confidence"
    target_group = "high_confidence" if args.mean_from_low_confidence else "low_confidence"
    lines = [
        "KV Patching Ablation Configuration",
        "=================================",
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
        f"kv_patch_method={args.kv_patch_method}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={source_group}",
        f"ablation_target_group={target_group}",
        f"ablate_layers_spec={args.ablate_layers}",
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
        if mode_name == "none":
            lines.append(
                f"{mode_name}={'None' if mode_mean is None else f'{mode_mean:.6f}'} ({valid_count})"
            )
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
        raise ValueError(
            f"Example {ex_id} responses/0/embeddings_probability missing key '{component}'."
        )
    value = emb_prob.get(component)
    if value is None:
        raise ValueError(
            f"Example {ex_id} responses/0/embeddings_probability/{component} is null."
        )
    if not isinstance(value, list):
        raise ValueError(
            f"Example {ex_id} responses/0/embeddings_probability/{component} must be a list."
        )
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
            "k_probability": torch.tensor(
                kv_means["k_probability"][local_idx], device=device, dtype=torch_dtype
            ),
            "v_probability": torch.tensor(
                kv_means["v_probability"][local_idx], device=device, dtype=torch_dtype
            ),
        }
    return layer_to_means


def _span_kv_patch_plan(
    *,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_means: Dict[str, torch.Tensor],
    expected_probability_tokens: int,
) -> Optional[List[Tuple[int, torch.Tensor, torch.Tensor]]]:
    """Map each probability-span token to (abs_pos, mean_k_j, mean_v_j)."""
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

    return [
        (abs_pos, k_probability[j], v_probability[j])
        for j, abs_pos in enumerate(abs_positions)
    ]


def _kv_patch_plan_for_mode(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_means: Dict[str, torch.Tensor],
    expected_probability_tokens: int,
) -> Optional[List[Tuple[int, torch.Tensor, torch.Tensor]]]:
    """Prefix-span K/V means; probability-value mode also applies the last prefix K/V at value positions."""
    span_plan = _span_kv_patch_plan(
        prompt_len=prompt_len,
        decoded_tokens=decoded_tokens,
        layer_means=layer_means,
        expected_probability_tokens=expected_probability_tokens,
    )
    if span_plan is None:
        return None
    if mode != MODE_PROBABILITY_VALUE_TOKENS:
        return span_plan

    prob_value_positions = _absolute_probability_value_positions(prompt_len, decoded_tokens)
    if not prob_value_positions:
        return span_plan

    _, k_last, v_last = span_plan[-1]
    extended = list(span_plan)
    for abs_pos in prob_value_positions:
        extended.append((abs_pos, k_last, v_last))
    return extended


def _repeat_kv_heads_for_gqa(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count (GQA)."""
    if n_rep == 1:
        return tensor
    batch, seq_len, n_kv_heads, d_head = tensor.shape
    expanded = tensor[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, n_rep, d_head)
    return expanded.reshape(batch, seq_len, n_kv_heads * n_rep, d_head)


def _recompute_query_attention_with_span_kv(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_span_plan: List[Tuple[int, torch.Tensor, torch.Tensor]],
    query_idx: int,
    layer: int,
) -> torch.Tensor:
    """Recompute hook_z at query_idx using mean K/V at probability-prefix key positions only."""
    k_patched = k.clone()
    v_patched = v.clone()
    for abs_pos, k_vector, v_vector in prefix_span_plan:
        if 0 <= abs_pos < k_patched.shape[1]:
            k_patched[:, abs_pos, :, :] = _reshape_kv_vector_for_activation(
                k_vector, k_patched, layer, "k"
            )
            v_patched[:, abs_pos, :, :] = _reshape_kv_vector_for_activation(
                v_vector, v_patched, layer, "v"
            )

    n_q_heads = int(q.shape[2])
    n_kv_heads = int(k_patched.shape[2])
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"Layer {layer}: query heads ({n_q_heads}) not divisible by kv heads ({n_kv_heads})."
        )
    n_rep = n_q_heads // n_kv_heads
    k_expanded = _repeat_kv_heads_for_gqa(k_patched, n_rep)
    v_expanded = _repeat_kv_heads_for_gqa(v_patched, n_rep)

    d_head = int(q.shape[-1])
    q_at = q[:, query_idx : query_idx + 1, :, :]
    scale = 1.0 / math.sqrt(d_head)
    scores = torch.einsum("b q h d, b s h d -> b h q s", q_at, k_expanded) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("b h q s, b s h d -> b q h d", attn, v_expanded)


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


def build_kv_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    mode: str,
    kv_patch_method: str,
    prompt_len: int,
    expected_probability_tokens: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        layer_means = layer_to_means[layer]

        if kv_patch_method == "span_key_positions":
            hook_k_name = f"blocks.{layer}.attn.hook_k"
            hook_v_name = f"blocks.{layer}.attn.hook_v"

            def _make_span_k_hook(local_layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
                def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                    del hook
                    if not _mode_uses_span_kv_patch(mode):
                        return activation
                    decoded_tokens = decoded_tokens_provider()
                    if not _mode_is_active_for_step(
                        mode=mode,
                        prompt_len=prompt_len,
                        decoded_tokens=decoded_tokens,
                        expected_probability_tokens=expected_probability_tokens,
                    ):
                        return activation
                    patch_plan = _kv_patch_plan_for_mode(
                        mode=mode,
                        prompt_len=prompt_len,
                        decoded_tokens=decoded_tokens,
                        layer_means=local_layer_means,
                        expected_probability_tokens=expected_probability_tokens,
                    )
                    if patch_plan is None:
                        return activation
                    for abs_pos, k_vector, _ in patch_plan:
                        if 0 <= abs_pos < activation.shape[1]:
                            activation[:, abs_pos, :, :] = _reshape_kv_vector_for_activation(
                                k_vector, activation, local_layer, "k"
                            )
                    return activation

                return hook_fn

            def _make_span_v_hook(local_layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
                def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                    del hook
                    if not _mode_uses_span_kv_patch(mode):
                        return activation
                    decoded_tokens = decoded_tokens_provider()
                    if not _mode_is_active_for_step(
                        mode=mode,
                        prompt_len=prompt_len,
                        decoded_tokens=decoded_tokens,
                        expected_probability_tokens=expected_probability_tokens,
                    ):
                        return activation
                    patch_plan = _kv_patch_plan_for_mode(
                        mode=mode,
                        prompt_len=prompt_len,
                        decoded_tokens=decoded_tokens,
                        layer_means=local_layer_means,
                        expected_probability_tokens=expected_probability_tokens,
                    )
                    if patch_plan is None:
                        return activation
                    for abs_pos, _, v_vector in patch_plan:
                        if 0 <= abs_pos < activation.shape[1]:
                            activation[:, abs_pos, :, :] = _reshape_kv_vector_for_activation(
                                v_vector, activation, local_layer, "v"
                            )
                    return activation

                return hook_fn

            hooks.append((hook_k_name, _make_span_k_hook(layer_means, layer)))
            hooks.append((hook_v_name, _make_span_v_hook(layer_means, layer)))
            continue

        if kv_patch_method != "last_query_span_read":
            raise ValueError(f"Unsupported kv_patch_method: {kv_patch_method!r}")

        attn_cache: Dict[str, Optional[torch.Tensor]] = {"q": None, "k": None, "v": None}

        def _make_capture_hook(cache_key: str) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                attn_cache[cache_key] = activation
                return activation

            return hook_fn

        def _make_last_query_z_hook(local_layer_means: Dict[str, torch.Tensor], local_layer: int) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                if not _mode_uses_span_kv_patch(mode):
                    return activation
                decoded_tokens = decoded_tokens_provider()
                if not _mode_is_active_for_step(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    expected_probability_tokens=expected_probability_tokens,
                ):
                    return activation
                q = attn_cache.get("q")
                k = attn_cache.get("k")
                v = attn_cache.get("v")
                if q is None or k is None or v is None:
                    return activation

                prefix_span_plan = _span_kv_patch_plan(
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_means=local_layer_means,
                    expected_probability_tokens=expected_probability_tokens,
                )
                if prefix_span_plan is None:
                    return activation

                seq_len = int(activation.shape[1])
                query_positions = _query_positions_for_kv_patch(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    seq_len=seq_len,
                    expected_probability_tokens=expected_probability_tokens,
                )
                for query_idx in query_positions:
                    if not (0 <= query_idx < seq_len):
                        continue
                    z_q = _recompute_query_attention_with_span_kv(
                        q=q,
                        k=k,
                        v=v,
                        prefix_span_plan=prefix_span_plan,
                        query_idx=query_idx,
                        layer=local_layer,
                    )
                    activation[:, query_idx, :, :] = z_q[:, 0, :, :]
                return activation

            return hook_fn

        hooks.append((f"blocks.{layer}.attn.hook_q", _make_capture_hook("q")))
        hooks.append((f"blocks.{layer}.attn.hook_k", _make_capture_hook("k")))
        hooks.append((f"blocks.{layer}.attn.hook_v", _make_capture_hook("v")))
        hooks.append((f"blocks.{layer}.attn.hook_z", _make_last_query_z_hook(layer_means, layer)))

    return hooks


def greedy_generate_kv_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    mode: str,
    kv_patch_method: str,
    expected_probability_tokens: int,
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_kv_mean_replace_hooks(
        layer_indices=layer_indices,
        mode=mode,
        kv_patch_method=kv_patch_method,
        prompt_len=prompt_len,
        expected_probability_tokens=expected_probability_tokens,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_means=layer_to_means,
    )
    return greedy_generate(
        model=model,
        local_prompt=local_prompt,
        max_new_tokens=max_new_tokens,
        fwd_hooks=hooks,
        decoded_tokens_buffer=decoded_tokens,
    )


def _plot_mode_confidence_by_layer(
    *,
    run_layers: Sequence[int],
    per_layer_mode_means: Dict[int, Dict[str, Optional[float]]],
    mode: str,
    baseline_none_mean: Optional[float],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs: List[int] = []
    ys: List[float] = []
    for layer_idx in run_layers:
        y_val = per_layer_mode_means.get(int(layer_idx), {}).get(mode)
        if y_val is not None:
            xs.append(int(layer_idx))
            ys.append(float(y_val))
    if ys:
        ax.plot(xs, ys, marker="o", label=mode)
    if baseline_none_mean is not None:
        ax.axhline(y=float(baseline_none_mean), linestyle="--", label="none (baseline)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Verbalised confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Individual layer verbalised confidence ({mode})")
    ax.grid(True, alpha=0.3)
    if ys or baseline_none_mean is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


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
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["trivia_qa", "squad", "bioasq", "nq", "svamp", "gsm8k"],
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--model_max_new_tokens", type=int, default=30)
    parser.add_argument("--brief_prompt", type=str, default="default", choices=["default", "chat"])
    parser.add_argument("--brief_always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_brief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablate_layers", type=str, default="12-15")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=[MODE_NONE, MODE_AFTER_PROB_PREFIX],
        choices=[MODE_NONE, MODE_AFTER_PROB_PREFIX, MODE_PROBABILITY_VALUE_TOKENS],
        help=(
            "Ablation modes. current_generated_token_after_prob_prefix_mean_replace: after "
            "Guess/Probability parse succeeds, apply prefix-span K/V mean patching for the "
            "current last query only (see --kv_patch_method). "
            "probability_value_tokens_kv_mean_replace: same prefix-span K/V means, applied at "
            "each probability-value token position (_absolute_probability_value_positions)."
        ),
    )
    parser.add_argument(
        "--kv_patch_method",
        type=str,
        default="last_query_span_read",
        choices=["last_query_span_read", "span_key_positions"],
        help=(
            "How to apply span K/V mean patching. last_query_span_read: recompute attention at "
            "selected query positions using mean K/V at prefix-span key positions only. "
            "span_key_positions: overwrite K/V at prefix-span positions (and probability-value "
            "positions for probability_value_tokens_kv_mean_replace) in hook_k/hook_v."
        ),
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--mean_from_low_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--parse_mode_verbalised_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--individual_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, ignore --ablate_layers and run a separate ablation for each layer. "
            "Outputs are written under results/individual_layers/<run_id>/<layer_idx>/."
        ),
    )
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if not args.input_h5:
        raise ValueError("--input_h5 is required.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_path = resolve_output_json_path(args.output_json)
    run_root = os.path.dirname(out_path)
    run_root_norm = run_root.rstrip(os.sep)
    run_id = os.path.basename(run_root_norm)
    results_root = os.path.dirname(run_root_norm)
    individual_root = os.path.join(results_root, "individual_layers", run_id)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = load_eval_dataset(args.dataset, args.random_seed)
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
        "Loaded %d H5 examples. low_conf=%d high_conf=%d target_group=%s target_ids=%d layers=%s individual_layers=%s",
        len(examples_h5),
        len(low_ids),
        len(high_ids),
        target_group,
        len(ablation_target_ids),
        run_layers,
        args.individual_layers,
    )

    def run_one_evaluation(
        layer_subset: Sequence[int],
        cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    ) -> Tuple[
        Dict[str, dict],
        Dict[str, dict],
        Dict[str, Optional[float]],
        Dict[str, int],
        Dict[str, Dict[str, Dict[str, object]]],
        Dict[str, int],
    ]:
        layer_subset_means = {int(layer_idx): layer_kv_means[int(layer_idx)] for layer_idx in layer_subset}
        results = {"train": {}, "validation": {}}
        mini_results = {"train": {}, "validation": {}}
        mode_confidence_values: Dict[str, List[float]] = {mode_name: [] for mode_name in args.ablation_mode}
        mode_responses_identical_true: Dict[str, int] = {
            mode_name: 0 for mode_name in args.ablation_mode if mode_name != "none"
        }
        used_none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}

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
                    entry[mode_to_output_key(MODE_NONE)] = {
                        "response": baseline_response,
                        "decoded_tokens": baseline_decoded_tokens,
                    }
                    mini_entry[mode_to_output_key(MODE_NONE)] = {"response": baseline_response}
                    if args.parse_mode_verbalised_confidence:
                        entry[mode_to_output_key(MODE_NONE)]["verbalised_confidence"] = baseline_mode_confidence
                        mini_entry[mode_to_output_key(MODE_NONE)]["verbalised_confidence"] = baseline_mode_confidence
                        if baseline_mode_confidence is not None:
                            mode_confidence_values[MODE_NONE].append(float(baseline_mode_confidence))

                for mode in args.ablation_mode:
                    if mode == MODE_NONE:
                        continue
                    response, decoded_tokens = greedy_generate_kv_ablated(
                        model=model,
                        local_prompt=local_prompt,
                        max_new_tokens=args.model_max_new_tokens,
                        layer_indices=layer_subset,
                        mode=mode,
                        kv_patch_method=args.kv_patch_method,
                        expected_probability_tokens=args.expected_probability_tokens,
                        layer_to_means=layer_subset_means,
                    )
                    mode_confidence = (
                        parse_mode_confidence_from_response(response)
                        if args.parse_mode_verbalised_confidence
                        else None
                    )
                    key = mode_to_output_key(mode)
                    entry[key] = {"response": response, "decoded_tokens": decoded_tokens}
                    mini_entry[key] = {"response": response}
                    if args.parse_mode_verbalised_confidence:
                        entry[key]["verbalised_confidence"] = mode_confidence
                        mini_entry[key]["verbalised_confidence"] = mode_confidence
                        if mode_confidence is not None:
                            mode_confidence_values[mode].append(float(mode_confidence))

                    if baseline_response is not None:
                        responses_identical = response == baseline_response
                        entry[key]["responses_identical"] = responses_identical
                        mini_entry[key]["responses_identical"] = responses_identical
                        if responses_identical:
                            mode_responses_identical_true[mode] += 1
                        if args.parse_mode_verbalised_confidence:
                            if mode_confidence is None or baseline_mode_confidence is None:
                                meets_none_confidence_direction = None
                            elif args.mean_from_low_confidence:
                                meets_none_confidence_direction = mode_confidence > baseline_mode_confidence
                            else:
                                meets_none_confidence_direction = mode_confidence < baseline_mode_confidence
                            entry[key]["meets_none_confidence_direction"] = meets_none_confidence_direction
                            mini_entry[key]["meets_none_confidence_direction"] = meets_none_confidence_direction

                    logging.info(
                        "[%s %d/%d] %s %s first line: %r",
                        split_name,
                        i + 1,
                        len(selected_ids),
                        ex_id,
                        key,
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
            used_none_cache,
            mode_responses_identical_true,
        )

    def build_none_cache() -> Dict[str, Dict[str, Dict[str, object]]]:
        none_cache: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "validation": {}}
        for split_name, eval_ds in [("train", train_ds), ("validation", val_ds)]:
            split_target = round(args.num_samples * TRAIN_RATIO) if split_name == "train" else round(
                args.num_samples * (1 - TRAIN_RATIO)
            )
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
        with open(mini_output_json_path(out_path), "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        write_config_txt(
            config_txt_path(out_path),
            args=args,
            device=device,
            model_n_layers=int(model.cfg.n_layers),
            ablate_layers=ablate_layers,
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
        return

    os.makedirs(individual_root, exist_ok=True)
    none_cache = build_none_cache() if MODE_NONE in args.ablation_mode else None
    per_layer_mode_means: Dict[int, Dict[str, Optional[float]]] = {}
    for layer_idx in run_layers:
        layer_dir = os.path.join(individual_root, str(layer_idx))
        os.makedirs(layer_dir, exist_ok=True)
        layer_out_path = os.path.join(layer_dir, "ablation_results.json")
        (
            layer_results,
            layer_mini_results,
            layer_mode_means,
            layer_mode_counts,
            _,
            layer_identical_true,
        ) = run_one_evaluation([layer_idx], cached_none=none_cache)
        per_layer_mode_means[int(layer_idx)] = layer_mode_means
        with open(layer_out_path, "w", encoding="utf-8") as f:
            json.dump(layer_results, f, ensure_ascii=False, indent=2)
        with open(mini_output_json_path(layer_out_path), "w", encoding="utf-8") as f:
            json.dump(layer_mini_results, f, ensure_ascii=False, indent=2)
        write_config_txt(
            config_txt_path(layer_out_path),
            args=args,
            device=device,
            model_n_layers=int(model.cfg.n_layers),
            ablate_layers=[layer_idx],
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            mode_confidence_means=layer_mode_means,
            mode_confidence_counts=layer_mode_counts,
            mode_responses_identical_true=layer_identical_true,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    summary_path = os.path.join(individual_root, "summary.txt")
    modes_non_none = [mode for mode in args.ablation_mode if mode != MODE_NONE]
    baseline_none_mean: Optional[float] = None
    if MODE_NONE in args.ablation_mode and per_layer_mode_means:
        baseline_none_mean = per_layer_mode_means[run_layers[0]].get(MODE_NONE)

    summary_lines = [
        "Individual Layer KV Patching Ablation Summary",
        "============================================",
        "",
        "[Setup]",
        f"model_name={args.model_name}",
        f"input_h5={args.input_h5}",
        f"dataset={args.dataset}",
        f"ablation_mode={args.ablation_mode}",
        f"kv_patch_method={args.kv_patch_method}",
        f"num_layers={model.cfg.n_layers}",
        f"run_layers={','.join(str(layer) for layer in run_layers)}",
        f"num_samples={args.num_samples}",
        f"num_few_shot={args.num_few_shot}",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={'low_confidence' if args.mean_from_low_confidence else 'high_confidence'}",
        f"ablation_target_group={'high_confidence' if args.mean_from_low_confidence else 'low_confidence'}",
        f"parse_mode_verbalised_confidence={args.parse_mode_verbalised_confidence}",
        "",
    ]
    if MODE_NONE in args.ablation_mode:
        if baseline_none_mean is None:
            summary_lines.append("none_mean_verbalised_confidence=None")
        else:
            summary_lines.append(f"none_mean_verbalised_confidence={baseline_none_mean:.6f}")
        summary_lines.append("")

    summary_lines.append("[Per-layer verbalised confidence]")
    if modes_non_none:
        summary_lines.append("layer\t" + "\t".join(modes_non_none))
        for layer_idx in run_layers:
            row_vals: List[str] = []
            for mode in modes_non_none:
                val = per_layer_mode_means[layer_idx].get(mode)
                row_vals.append("None" if val is None else f"{val:.6f}")
            summary_lines.append(f"{layer_idx}\t" + "\t".join(row_vals))
    else:
        summary_lines.append("No non-none modes selected.")
    summary_lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logging.info("Wrote %s", summary_path)

    for mode in modes_non_none:
        plot_path = os.path.join(individual_root, f"verbalised_confidence_by_layer_{mode}.png")
        _plot_mode_confidence_by_layer(
            run_layers=run_layers,
            per_layer_mode_means=per_layer_mode_means,
            mode=mode,
            baseline_none_mean=baseline_none_mean,
            output_path=plot_path,
        )
    logging.info("Saved individual-layer outputs under %s", individual_root)


if __name__ == "__main__":
    main()

