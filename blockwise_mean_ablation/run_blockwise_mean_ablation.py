import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Span position helpers from blockwise_zero use completion index k -> abs pos (prompt_len + k - 1);
# prompt mean targets indices 0 .. prompt_len-2 (exclude last prompt position), matching layerwise_mean.
from blockwise_zero_ablation.run_blockwise_zero_ablation import (
    BRIEF_PROMPTS,
    CONFIDENCE_PROMPT,
    SUBBLOCK_TO_HOOK,
    _absolute_all_pre_guess_positions,
    _absolute_guess_span_positions,
    _absolute_guess_then_guess_probability_positions,
    _absolute_pre_probability_positions,
    _absolute_prob_except_last_token_positions,
    _absolute_prob_last_token_only_positions,
    _absolute_prob_positions,
    _absolute_prob_positions_at_row_indices,
    _normalize_per_layer_mode_means,
    collect_confidence_group_ids,
    construct_fewshot_prompt_from_indices,
    encode_example_id,
    greedy_generate,
    load_examples_h5,
    load_hooked_transformer,
    load_trivia_qa,
    mode_to_output_key,
    parse_ablate_layers,
    parse_mode_confidence_from_response,
    split_answerable_indices,
    write_individual_layer_plots,
)
from layerwise_mean_ablation.run_mean_ablation import (
    PROBABILITY_ROW_INDEX_MODES,
    _absolute_probability_value_start_position,
    _as_layer_hidden,
    _is_expected_or_plus_one,
    _mean_and_count,
    batch_compute_semantic_similarities,
    compute_uncertainty_score,
    compute_verbalised_confidence_effect,
    load_sentence_transformer_for_metrics,
    parse_semantic_answer_from_response,
)
from mass_mean_probe.run_mass_mean_probe import (
    DEFAULT_SEMANTIC_SIMILARITY_MODEL,
    SEMANTIC_SIMILARITY_MODES,
)


TRAIN_RATIO = 0.9
REQUIRED_COMPONENTS = ("res", "attn", "mlp")
TARGET_COMPONENTS = ("attn", "mlp")
REQUIRED_EMBEDDING_FIELDS = (
    "embeddings_mean_prompt",
    "embeddings_guess",
    "embeddings_mean_sem_answer",
    "embeddings_probability",
    "embeddings_mean_prob_val",
)

GENERATED_TOKENS_SOURCE_CHOICES = ("probability_prefix_last_token",)
WHOLE_SEQUENCE_MODES = frozenset({
    "all_tokens_mean_replace",
    "generated_tokens_mean_replace",
})


def resolve_output_json_path(cli_output_path: Optional[str]) -> str:
    if cli_output_path:
        out_path = os.path.abspath(cli_output_path)
    else:
        base = Path("blockwise_mean_ablation") / "results"
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
    mode_confidence_means: Dict[str, Dict[str, Optional[float]]],
    mode_confidence_counts: Dict[str, Dict[str, int]],
    mode_responses_identical_true: Dict[str, Dict[str, int]],
    finished_at: str,
    mode_semantic_similarity_means: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    mode_semantic_similarity_counts: Optional[Dict[str, Dict[str, int]]] = None,
    mode_verbalised_confidence_effect_means: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    mode_verbalised_confidence_effect_counts: Optional[Dict[str, Dict[str, int]]] = None,
    mode_uncertainty_score_means: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    mode_uncertainty_score_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    source_group = "low_confidence" if args.mean_from_low_confidence else "high_confidence"
    target_group = "high_confidence" if args.mean_from_low_confidence else "low_confidence"
    lines = [
        "Blockwise Mean Ablation Configuration",
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
        f"generated_tokens_source={args.generated_tokens_source}",
        f"ablate_subblocks={args.ablate_subblocks}",
        f"mean_from_low_confidence={args.mean_from_low_confidence}",
        f"mean_source_group={source_group}",
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
        "Values below are mean verbalised confidence.",
        "",
    ]
    for mode_name in args.ablation_mode:
        per_subblock = mode_confidence_means.get(mode_name, {})
        per_subblock_counts = mode_confidence_counts.get(mode_name, {})
        for subblock in args.ablate_subblocks:
            mode_mean = per_subblock.get(subblock)
            valid_count = int(per_subblock_counts.get(subblock, 0))
            metric_key = f"{mode_name}__{subblock}"
            if mode_name == "none":
                if mode_mean is None:
                    lines.append(f"{metric_key}=None ({valid_count})")
                else:
                    lines.append(f"{metric_key}={mode_mean:.6f} ({valid_count})")
            else:
                identical_n = int(mode_responses_identical_true.get(mode_name, {}).get(subblock, 0))
                if mode_mean is None:
                    line = f"{metric_key}=None ({valid_count}) [responses_identical: {identical_n}]"
                else:
                    line = (
                        f"{metric_key}={mode_mean:.6f} ({valid_count}) [responses_identical: {identical_n}]"
                    )
                if mode_semantic_similarity_means is not None:
                    sem_mean = mode_semantic_similarity_means.get(mode_name, {}).get(subblock)
                    sem_count = int(
                        mode_semantic_similarity_counts.get(mode_name, {}).get(subblock, 0)
                        if mode_semantic_similarity_counts
                        else 0
                    )
                    if sem_mean is not None and sem_count > 0:
                        line += f" semantic_similarity={sem_mean:.6f} ({sem_count})"
                if mode_verbalised_confidence_effect_means is not None:
                    vce_mean = mode_verbalised_confidence_effect_means.get(mode_name, {}).get(subblock)
                    vce_count = int(
                        mode_verbalised_confidence_effect_counts.get(mode_name, {}).get(subblock, 0)
                        if mode_verbalised_confidence_effect_counts
                        else 0
                    )
                    if vce_mean is not None and vce_count > 0:
                        line += f" verbalised_confidence_effect={vce_mean:.6f} ({vce_count})"
                if mode_uncertainty_score_means is not None:
                    unc_mean = mode_uncertainty_score_means.get(mode_name, {}).get(subblock)
                    unc_count = int(
                        mode_uncertainty_score_counts.get(mode_name, {}).get(subblock, 0)
                        if mode_uncertainty_score_counts
                        else 0
                    )
                    if unc_mean is not None and unc_count > 0:
                        line += f" uncertainty_score={unc_mean:.6f} ({unc_count})"
                lines.append(line)
    lines.extend(["", "[Run]", f"finished_at={finished_at}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _validate_component_field(resp0: dict, ex_id: str, field_name: str, component: str):
    field = resp0.get(field_name)
    if not isinstance(field, dict):
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} must be an object with keys {REQUIRED_COMPONENTS}."
        )
    if component not in field:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name} missing component '{component}'. "
            f"Expected keys include {REQUIRED_COMPONENTS}."
        )
    value = field.get(component)
    if value is None:
        raise ValueError(
            f"Example {ex_id} responses/0/{field_name}/{component} is null. "
            "Input H5 must contain populated res/attn/mlp subfields."
        )
    return value


def compute_pre_probability_group_means_by_component(
    examples_h5: Dict[str, dict],
    *,
    ablate_layers: Sequence[int],
    low_conf_threshold: float,
    high_conf_threshold: float,
    mean_from_low_confidence: bool,
    expected_probability_tokens: int,
    expected_guess_tokens: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], set[str], set[str]]:
    low_ids: set[str] = set()
    high_ids: set[str] = set()

    prompt_vectors: Dict[str, List[np.ndarray]] = {c: [] for c in TARGET_COMPONENTS}
    sem_answer_vectors: Dict[str, List[np.ndarray]] = {c: [] for c in TARGET_COMPONENTS}
    guess_vectors: Dict[str, List[np.ndarray]] = {c: [] for c in TARGET_COMPONENTS}
    probability_vectors: Dict[str, List[np.ndarray]] = {c: [] for c in TARGET_COMPONENTS}
    probability_value_mean_vectors: Dict[str, List[np.ndarray]] = {c: [] for c in TARGET_COMPONENTS}
    layer_idx = np.asarray(ablate_layers)

    for ex_id, ex in examples_h5.items():
        responses = ex.get("responses") or []
        if not responses:
            raise ValueError(f"Example {ex_id} has no responses array.")
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            raise ValueError(f"Example {ex_id} responses/0 must be an object.")

        for field_name in REQUIRED_EMBEDDING_FIELDS:
            for component in REQUIRED_COMPONENTS:
                _validate_component_field(resp0, ex_id, field_name, component)

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

        for component in TARGET_COMPONENTS:
            emb_prompt = _validate_component_field(resp0, ex_id, "embeddings_mean_prompt", component)
            emb_guess = _validate_component_field(resp0, ex_id, "embeddings_guess", component)
            emb_sem_answer = _validate_component_field(
                resp0, ex_id, "embeddings_mean_sem_answer", component
            )
            emb_prob = _validate_component_field(resp0, ex_id, "embeddings_probability", component)
            emb_mean_prob_val = _validate_component_field(
                resp0, ex_id, "embeddings_mean_prob_val", component
            )

            if not isinstance(emb_guess, list):
                raise ValueError(
                    f"Example {ex_id} responses/0/embeddings_guess/{component} must be a list."
                )
            if not _is_expected_or_plus_one(len(emb_guess), expected_guess_tokens):
                raise ValueError(
                    f"Example {ex_id} embeddings_guess/{component} len={len(emb_guess)}; expected "
                    f"{expected_guess_tokens} or {expected_guess_tokens + 1}."
                )
            emb_guess = emb_guess[:expected_guess_tokens]

            if not isinstance(emb_prob, list):
                raise ValueError(
                    f"Example {ex_id} responses/0/embeddings_probability/{component} must be a list."
                )
            if not _is_expected_or_plus_one(len(emb_prob), expected_probability_tokens):
                raise ValueError(
                    f"Example {ex_id} embeddings_probability/{component} len={len(emb_prob)}; expected "
                    f"{expected_probability_tokens} or {expected_probability_tokens + 1}."
                )
            emb_prob = emb_prob[:expected_probability_tokens]

            prompt_layer_hidden = _as_layer_hidden(emb_prompt)[layer_idx, :]
            sem_answer_layer_hidden = _as_layer_hidden(emb_sem_answer)[layer_idx, :]
            mean_prob_val_layer_hidden = _as_layer_hidden(emb_mean_prob_val)[layer_idx, :]

            guess_selected: List[np.ndarray] = []
            for tok_arr in emb_guess:
                guess_selected.append(_as_layer_hidden(tok_arr)[layer_idx, :])
            guess_stacked = np.stack(guess_selected, axis=1)

            prob_selected: List[np.ndarray] = []
            for tok_arr in emb_prob:
                prob_selected.append(_as_layer_hidden(tok_arr)[layer_idx, :])
            prob_stacked = np.stack(prob_selected, axis=1)

            prompt_vectors[component].append(prompt_layer_hidden)
            sem_answer_vectors[component].append(sem_answer_layer_hidden)
            guess_vectors[component].append(guess_stacked)
            probability_vectors[component].append(prob_stacked)
            probability_value_mean_vectors[component].append(mean_prob_val_layer_hidden)

    source_name = "low-confidence" if mean_from_low_confidence else "high-confidence"
    if not low_ids:
        raise ValueError(f"No low-confidence examples found at threshold <= {low_conf_threshold}.")
    if not high_ids:
        raise ValueError(f"No high-confidence examples found at threshold >= {high_conf_threshold}.")
    for component in TARGET_COMPONENTS:
        if not prompt_vectors[component]:
            threshold = low_conf_threshold if mean_from_low_confidence else high_conf_threshold
            operator = "<=" if mean_from_low_confidence else ">="
            raise ValueError(
                f"No {source_name} examples usable for component '{component}' at threshold {operator} {threshold}."
            )

    means_by_component: Dict[str, Dict[str, np.ndarray]] = {}
    for component in TARGET_COMPONENTS:
        means_by_component[component] = {
            "prompt_mean": np.mean(np.stack(prompt_vectors[component], axis=0), axis=0).astype(np.float32),
            "guess": np.mean(np.stack(guess_vectors[component], axis=0), axis=0).astype(np.float32),
            "sem_answer_mean": np.mean(
                np.stack(sem_answer_vectors[component], axis=0), axis=0
            ).astype(np.float32),
            "probability": np.mean(
                np.stack(probability_vectors[component], axis=0), axis=0
            ).astype(np.float32),
            "probability_value_mean": np.mean(
                np.stack(probability_value_mean_vectors[component], axis=0), axis=0
            ).astype(np.float32),
        }
    return means_by_component, low_ids, high_ids


def _build_layer_means(
    means_by_component: Dict[str, Dict[str, np.ndarray]],
    *,
    ablate_layers: Sequence[int],
    device: str,
    torch_dtype: torch.dtype,
) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
    out: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {c: {} for c in TARGET_COMPONENTS}
    for component in TARGET_COMPONENTS:
        comp_means = means_by_component[component]
        for layer_i, layer in enumerate(ablate_layers):
            out[component][int(layer)] = {
                "prompt_mean": torch.tensor(comp_means["prompt_mean"][layer_i], device=device, dtype=torch_dtype),
                "guess": torch.tensor(comp_means["guess"][layer_i], device=device, dtype=torch_dtype),
                "sem_answer_mean": torch.tensor(
                    comp_means["sem_answer_mean"][layer_i], device=device, dtype=torch_dtype
                ),
                "probability": torch.tensor(comp_means["probability"][layer_i], device=device, dtype=torch_dtype),
                "probability_value_mean": torch.tensor(
                    comp_means["probability_value_mean"][layer_i], device=device, dtype=torch_dtype
                ),
            }
    return out


def _positions_for_whole_sequence_mode(mode: str, *, prompt_len: int, seq_len: int) -> List[int]:
    if mode == "all_tokens_mean_replace":
        return list(range(seq_len))
    if mode == "generated_tokens_mean_replace":
        return list(range(prompt_len - 1, seq_len))
    raise ValueError(f"Unknown whole-sequence ablation mode: {mode!r}")


def _generated_token_mean_for_source(
    layer_means: Dict[str, torch.Tensor],
    source: str,
) -> torch.Tensor:
    if source == "probability_prefix_last_token":
        return layer_means["probability"][-1]
    raise ValueError(f"Unknown generated_tokens_source: {source!r}")


def _whole_sequence_positions_and_replacements(
    mode: str,
    *,
    prompt_len: int,
    seq_len: int,
    layer_means: Dict[str, torch.Tensor],
    generated_tokens_source: str,
) -> Tuple[List[int], List[torch.Tensor]]:
    prompt_mean = layer_means["prompt_mean"]
    generated_mean = _generated_token_mean_for_source(layer_means, generated_tokens_source)

    positions = _positions_for_whole_sequence_mode(mode, prompt_len=prompt_len, seq_len=seq_len)

    out_positions: List[int] = []
    out_vectors: List[torch.Tensor] = []
    for abs_pos in positions:
        vector = prompt_mean if abs_pos < prompt_len - 1 else generated_mean
        out_positions.append(abs_pos)
        out_vectors.append(vector)
    return out_positions, out_vectors


def _positions_and_replacements_for_mode(
    *,
    mode: str,
    prompt_len: int,
    decoded_tokens: List[str],
    layer_means: Dict[str, torch.Tensor],
    generated_tokens_source: str = "probability_prefix_last_token",
) -> Tuple[List[int], List[torch.Tensor]]:
    prompt_mean = layer_means["prompt_mean"]
    guess = layer_means["guess"]
    sem_answer_mean = layer_means["sem_answer_mean"]
    probability = layer_means["probability"]
    probability_value_mean = layer_means["probability_value_mean"]
    logging.info("Inside _positions_and_replacements_for_mode")
    logging.info(f"Mode: {mode}, shape of each layer mean: {prompt_mean.shape}, {guess.shape}, {sem_answer_mean.shape}, {probability.shape}")

    if mode == "probability_tokens_mean_replace":
        positions = _absolute_prob_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=int(probability.shape[0]),
        )
        vectors = [probability[i] for i in range(min(len(positions), int(probability.shape[0])))]
        return positions[: len(vectors)], vectors

    if mode in PROBABILITY_ROW_INDEX_MODES:
        row_indices = PROBABILITY_ROW_INDEX_MODES[mode]
        positions = _absolute_prob_positions_at_row_indices(
            prompt_len,
            decoded_tokens,
            row_indices,
            expected_probability_tokens=int(probability.shape[0]),
        )
        if not positions:
            return [], []
        vectors = [probability[i] for i in row_indices]
        return positions, vectors

    if mode == "probability_last_token_mean_replace":
        positions = _absolute_prob_last_token_only_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=int(probability.shape[0]),
        )
        if not positions:
            return [], []
        return [positions[0]], [probability[-1]]

    if mode == "probability_span_except_last_token_mean_replace":
        positions = _absolute_prob_except_last_token_positions(
            prompt_len,
            decoded_tokens,
            expected_probability_tokens=int(probability.shape[0]),
        )
        n = min(len(positions), max(0, int(probability.shape[0]) - 1))
        return positions[:n], [probability[i] for i in range(n)]

    if mode == "guess_tokens_mean_replace":
        positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
        )
        n = min(len(positions), int(guess.shape[0]))
        return positions[:n], [guess[i] for i in range(n)]

    if mode == "all_pre_guess_tokens_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
        )
        prompt_positions = _absolute_all_pre_guess_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
        )
        prompt_count = max(0, len(prompt_positions) - len(guess_positions))
        positions: List[int] = []
        vectors: List[torch.Tensor] = []
        for p in prompt_positions[:prompt_count]:
            positions.append(p)
            vectors.append(prompt_mean)
        n_guess = min(len(guess_positions), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(guess_positions[i])
            vectors.append(guess[i])
        return positions, vectors

    if mode == "all_pre_probability_tokens_mean_replace":
        spans = _absolute_pre_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
            expected_probability_tokens=int(probability.shape[0]),
        )
        if spans is None:
            return [], []
        positions: List[int] = []
        vectors: List[torch.Tensor] = []
        for p in spans["prompt"]:
            positions.append(p)
            vectors.append(prompt_mean)
        n_guess = min(len(spans["guess"]), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(spans["guess"][i])
            vectors.append(guess[i])
        for p in spans["sem_answer"]:
            positions.append(p)
            vectors.append(sem_answer_mean)
        n_prob = min(len(spans["probability"]), int(probability.shape[0]))
        for i in range(n_prob):
            positions.append(spans["probability"][i])
            vectors.append(probability[i])
        return positions, vectors

    if mode == "guess_then_guess_probability_mean_replace":
        guess_positions = _absolute_guess_span_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
        )
        all_positions = _absolute_guess_then_guess_probability_positions(
            prompt_len,
            decoded_tokens,
            expected_guess_tokens=int(guess.shape[0]),
            expected_probability_tokens=int(probability.shape[0]),
        )
        prob_positions = all_positions[len(guess_positions) :]
        positions: List[int] = []
        vectors: List[torch.Tensor] = []
        n_guess = min(len(guess_positions), int(guess.shape[0]))
        for i in range(n_guess):
            positions.append(guess_positions[i])
            vectors.append(guess[i])
        n_prob = min(len(prob_positions), int(probability.shape[0]))
        for i in range(n_prob):
            positions.append(prob_positions[i])
            vectors.append(probability[i])
        return positions, vectors

    if mode == "probability_value_mean_replace":
        start_abs = _absolute_probability_value_start_position(prompt_len, decoded_tokens)
        if start_abs is None:
            return [], []
        seq_len = prompt_len + len(decoded_tokens)
        if start_abs >= seq_len:
            return [], []
        positions = list(range(start_abs, seq_len))
        if not positions:
            return [], []
        vectors = [probability[-1]] + [probability_value_mean] * (len(positions) - 1)
        return positions, vectors

    if mode == "current_generated_token_mean_replace":
        current_abs_pos = prompt_len + len(decoded_tokens) - 1
        if current_abs_pos < 0:
            return [], []
        return [current_abs_pos], [probability[-1]]

    if mode in ("all_tokens_mean_replace", "generated_tokens_mean_replace"):
        seq_len = prompt_len + len(decoded_tokens)
        return _whole_sequence_positions_and_replacements(
            mode,
            prompt_len=prompt_len,
            seq_len=seq_len,
            layer_means=layer_means,
            generated_tokens_source=generated_tokens_source,
        )

    raise ValueError(f"Unsupported mode for mean replacement: {mode!r}")


def build_subblock_mean_replace_hooks(
    layer_indices: Sequence[int],
    *,
    subblock: str,
    mode: str,
    prompt_len: int,
    decoded_tokens_provider: Callable[[], List[str]],
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    generated_tokens_source: str = "probability_prefix_last_token",
) -> List[Tuple[str, Callable]]:
    hook_suffix = SUBBLOCK_TO_HOOK[subblock]
    hooks: List[Tuple[str, Callable]] = []
    for layer in layer_indices:
        hook_name = f"blocks.{layer}.{hook_suffix}"
        layer_means = layer_to_means[layer]

        def _make_hook(local_layer_means: Dict[str, torch.Tensor]) -> Callable:
            def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
                del hook
                decoded_tokens = decoded_tokens_provider()
                positions, vectors = _positions_and_replacements_for_mode(
                    mode=mode,
                    prompt_len=prompt_len,
                    decoded_tokens=decoded_tokens,
                    layer_means=local_layer_means,
                    generated_tokens_source=generated_tokens_source,
                )
                if not positions:
                    return activation
                for abs_pos, vector in zip(positions, vectors):
                    if 0 <= abs_pos < activation.shape[1]:
                        activation[:, abs_pos, :] = vector
                return activation

            return hook_fn

        hooks.append((hook_name, _make_hook(layer_means)))
    return hooks


def greedy_generate_mean_ablated(
    model,
    local_prompt: str,
    max_new_tokens: int,
    *,
    layer_indices: Sequence[int],
    subblock: str,
    mode: str,
    layer_to_means: Dict[int, Dict[str, torch.Tensor]],
    generated_tokens_source: str = "probability_prefix_last_token",
) -> Tuple[str, List[str]]:
    tokens = model.to_tokens(local_prompt)
    prompt_len = int(tokens.shape[1])
    decoded_tokens: List[str] = []

    def _decoded_tokens_provider() -> List[str]:
        return decoded_tokens

    hooks = build_subblock_mean_replace_hooks(
        layer_indices=layer_indices,
        subblock=subblock,
        mode=mode,
        prompt_len=prompt_len,
        decoded_tokens_provider=_decoded_tokens_provider,
        layer_to_means=layer_to_means,
        generated_tokens_source=generated_tokens_source,
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
        help="Path to HDF5 with processed responses containing res/attn/mlp subfields.",
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
    parser.add_argument("--ablate_layers", type=str, default="12-15")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        nargs="+",
        default=[
            "none",
            "probability_tokens_mean_replace",
            "probability_first_token_mean_replace",
            "probability_first_two_tokens_mean_replace",
            "probability_first_two_and_index6_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
            "probability_value_mean_replace",
            "current_generated_token_mean_replace",
            "all_tokens_mean_replace",
            "generated_tokens_mean_replace",
        ],
        choices=[
            "none",
            "probability_tokens_mean_replace",
            "probability_first_token_mean_replace",
            "probability_first_two_tokens_mean_replace",
            "probability_first_two_and_index6_tokens_mean_replace",
            "probability_last_token_mean_replace",
            "probability_span_except_last_token_mean_replace",
            "all_pre_probability_tokens_mean_replace",
            "guess_tokens_mean_replace",
            "all_pre_guess_tokens_mean_replace",
            "guess_then_guess_probability_mean_replace",
            "probability_value_mean_replace",
            "current_generated_token_mean_replace",
            "all_tokens_mean_replace",
            "generated_tokens_mean_replace",
        ],
        help=(
            "One or more ablation modes. probability_first_token_mean_replace: same gating as "
            "probability_tokens_mean_replace but only H5 probability row 0. "
            "probability_first_two_tokens_mean_replace: rows 0 and 1. "
            "probability_first_two_and_index6_tokens_mean_replace: "
            "same gating as probability_tokens_mean_replace but only H5 probability rows 0, 1, and 6 "
            "(fixed index 6, not -1). probability_last_token_mean_replace: mean-replace only the "
            "last token of the H5 probability span (end_prob; first value digit). "
            "probability_span_except_last_token_mean_replace: mean-replace all probability-span tokens "
            "except that last token (all marker-span tokens before end_prob). "
            "probability_value_mean_replace: no hooks until Guess/Probability parse succeeds; then "
            "mean-replace from first probability-value token through current last token, using "
            "probability[-1] for the first position and embeddings_mean_prob_val mean for later positions. "
            "current_generated_token_mean_replace: at each decode step, mean-replace only the current "
            "last sequence token using probability[-1]. "
            "all_tokens_mean_replace: mean-replace every sequence position at each decode step; "
            "prompt_mean for prompt positions, selected --generated_tokens_source for all other "
            "positions (no span parsing). "
            "generated_tokens_mean_replace: same replacement logic, but only generated positions "
            "(prompt_len-1 onward)."
        ),
    )
    parser.add_argument(
        "--generated_tokens_source",
        type=str,
        nargs="+",
        default=["probability_prefix_last_token"],
        choices=list(GENERATED_TOKENS_SOURCE_CHOICES),
        help=(
            "Generated-token mean source for all_tokens_mean_replace and "
            "generated_tokens_mean_replace. Prompt positions always use prompt_mean; every "
            "non-prompt position uses the selected source. Currently only "
            "probability_prefix_last_token (probability[-1]) is supported."
        ),
    )
    parser.add_argument("--ablate_subblocks", type=str, nargs="+", required=True, choices=["attn", "mlp"])
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--mean_from_low_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument("--parse_mode_verbalised_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--individual_layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot_from_existing_summary", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if len(set(args.ablation_mode)) != len(args.ablation_mode):
        raise ValueError(f"Duplicate ablation modes are not allowed: {args.ablation_mode}")
    if any(m in WHOLE_SEQUENCE_MODES for m in args.ablation_mode):
        if len(args.generated_tokens_source) != 1:
            raise ValueError(
                "--generated_tokens_source must specify exactly one value when running "
                f"whole-sequence modes; got {args.generated_tokens_source}"
            )
    generated_tokens_source = args.generated_tokens_source[0]
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
            "Enter directory name under results/individual_layers containing mode_confidence_summary.json: "
        ).strip()
        if not summary_dir_name:
            raise ValueError("Directory name cannot be empty.")
        if os.path.basename(summary_dir_name) != summary_dir_name:
            raise ValueError("Provide only the directory name under results/individual_layers.")
        summary_dir = os.path.join(individual_layers_root, summary_dir_name)
        summary_path = os.path.join(summary_dir, "mode_confidence_summary.json")
        if not os.path.exists(summary_path):
            raise ValueError(f"Summary file not found: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            raw_summary = json.load(f)
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
    means_by_component, low_ids, high_ids = compute_pre_probability_group_means_by_component(
        examples_h5,
        ablate_layers=run_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        mean_from_low_confidence=args.mean_from_low_confidence,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_guess_tokens=args.expected_guess_tokens,
    )

    # Keep original confidence split utility for traceability in logs.
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

    layer_means = _build_layer_means(
        means_by_component,
        ablate_layers=run_layers,
        device=device,
        torch_dtype=torch_dtype,
    )
    logging.info(
        "Loaded %d H5 examples. low_conf=%d high_conf=%d target_group=%s target_ids=%d layers=%s",
        len(examples_h5),
        len(low_ids),
        len(high_ids),
        target_group,
        len(ablation_target_ids),
        run_layers,
    )

    def run_one_evaluation(
        layer_subset: Sequence[int],
        cached_none: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
        sentence_transformer=None,
    ) -> Tuple[
        dict,
        dict,
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, Dict[str, object]]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, Optional[float]]],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, Optional[float]]],
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
        compute_derived_metrics = "none" in args.ablation_mode and len(args.ablation_mode) > 1
        mode_semantic_similarity_values: Dict[str, Dict[str, List[float]]] = {
            mode_name: {subblock: [] for subblock in args.ablate_subblocks}
            for mode_name in args.ablation_mode
            if mode_name != "none"
        }
        mode_verbalised_confidence_effect_values: Dict[str, Dict[str, List[float]]] = {
            mode_name: {subblock: [] for subblock in args.ablate_subblocks}
            for mode_name in args.ablation_mode
            if mode_name != "none"
        }
        mode_uncertainty_score_values: Dict[str, Dict[str, List[float]]] = {
            mode_name: {subblock: [] for subblock in args.ablate_subblocks}
            for mode_name in args.ablation_mode
            if mode_name != "none"
        }
        pending_semantic_similarity: List[
            Tuple[str, str, str, str, str, str]
        ] = []
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
                            response, decoded_tokens = greedy_generate_mean_ablated(
                                model=model,
                                local_prompt=local_prompt,
                                max_new_tokens=args.model_max_new_tokens,
                                layer_indices=layer_subset,
                                subblock=subblock,
                                mode=mode,
                                layer_to_means={layer: layer_means[subblock][layer] for layer in layer_subset},
                                generated_tokens_source=generated_tokens_source,
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
                                elif args.mean_from_low_confidence:
                                    meets_none_confidence_direction = mode_confidence > baseline_mode_confidence
                                else:
                                    meets_none_confidence_direction = mode_confidence < baseline_mode_confidence
                                entry[key][subblock]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )
                                mini_entry[key][subblock]["meets_none_confidence_direction"] = (
                                    meets_none_confidence_direction
                                )

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

                if compute_derived_metrics and baseline_response is not None:
                    baseline_semantic_answer = parse_semantic_answer_from_response(str(baseline_response))
                    if "none" in args.ablation_mode:
                        none_key = mode_to_output_key("none")
                        for subblock in args.ablate_subblocks:
                            if baseline_semantic_answer is not None:
                                entry[none_key][subblock]["semantic_answer"] = baseline_semantic_answer
                                mini_entry[none_key][subblock]["semantic_answer"] = baseline_semantic_answer

                    for mode in args.ablation_mode:
                        if mode == "none":
                            continue
                        mode_key = mode_to_output_key(mode)
                        for subblock in args.ablate_subblocks:
                            subblock_entry = entry[mode_key][subblock]
                            mini_subblock_entry = mini_entry[mode_key][subblock]
                            mode_semantic_answer = parse_semantic_answer_from_response(
                                str(subblock_entry["response"])
                            )
                            if mode_semantic_answer is not None:
                                subblock_entry["semantic_answer"] = mode_semantic_answer
                                mini_subblock_entry["semantic_answer"] = mode_semantic_answer

                            if args.parse_mode_verbalised_confidence and baseline_mode_confidence is not None:
                                mode_confidence = subblock_entry.get("verbalised_confidence")
                                if mode_confidence is not None:
                                    vce = compute_verbalised_confidence_effect(
                                        float(baseline_mode_confidence),
                                        float(mode_confidence),
                                        mean_from_low_confidence=args.mean_from_low_confidence,
                                    )
                                    if vce is not None:
                                        subblock_entry["verbalised_confidence_effect"] = vce
                                        mini_subblock_entry["verbalised_confidence_effect"] = vce
                                        mode_verbalised_confidence_effect_values[mode][subblock].append(
                                            float(vce)
                                        )

                            if (
                                mode in SEMANTIC_SIMILARITY_MODES
                                and sentence_transformer is not None
                                and baseline_semantic_answer is not None
                                and mode_semantic_answer is not None
                            ):
                                pending_semantic_similarity.append(
                                    (
                                        split_name,
                                        ex_id,
                                        mode,
                                        subblock,
                                        str(baseline_semantic_answer),
                                        str(mode_semantic_answer),
                                    )
                                )

                results[split_name][ex_id] = entry
                mini_results[split_name][ex_id] = mini_entry

        if pending_semantic_similarity and sentence_transformer is not None:
            pairs = [(baseline_text, mode_text) for *_rest, baseline_text, mode_text in pending_semantic_similarity]
            similarities = batch_compute_semantic_similarities(sentence_transformer, pairs)
            for task, similarity in zip(pending_semantic_similarity, similarities):
                split_name, ex_id, mode, subblock, _baseline_text, _mode_text = task
                mode_key = mode_to_output_key(mode)
                subblock_entry = results[split_name][ex_id][mode_key][subblock]
                mini_subblock_entry = mini_results[split_name][ex_id][mode_key][subblock]
                subblock_entry["semantic_similarity"] = similarity
                mini_subblock_entry["semantic_similarity"] = similarity
                mode_semantic_similarity_values[mode][subblock].append(similarity)
                vce = subblock_entry.get("verbalised_confidence_effect")
                if vce is not None:
                    uncertainty = compute_uncertainty_score(similarity, float(vce))
                    subblock_entry["uncertainty_score"] = uncertainty
                    mini_subblock_entry["uncertainty_score"] = uncertainty
                    mode_uncertainty_score_values[mode][subblock].append(uncertainty)

        mode_confidence_means: Dict[str, Dict[str, Optional[float]]] = {}
        mode_confidence_counts: Dict[str, Dict[str, int]] = {}
        for mode_name in args.ablation_mode:
            mode_confidence_means[mode_name] = {}
            mode_confidence_counts[mode_name] = {}
            for subblock in args.ablate_subblocks:
                vals = mode_confidence_values[mode_name][subblock]
                mode_confidence_means[mode_name][subblock] = float(np.mean(vals)) if vals else None
                mode_confidence_counts[mode_name][subblock] = len(vals)

        mode_semantic_similarity_means: Dict[str, Dict[str, Optional[float]]] = {}
        mode_semantic_similarity_counts: Dict[str, Dict[str, int]] = {}
        mode_verbalised_confidence_effect_means: Dict[str, Dict[str, Optional[float]]] = {}
        mode_verbalised_confidence_effect_counts: Dict[str, Dict[str, int]] = {}
        mode_uncertainty_score_means: Dict[str, Dict[str, Optional[float]]] = {}
        mode_uncertainty_score_counts: Dict[str, Dict[str, int]] = {}
        for mode_name in args.ablation_mode:
            if mode_name == "none":
                continue
            mode_semantic_similarity_means[mode_name] = {}
            mode_semantic_similarity_counts[mode_name] = {}
            mode_verbalised_confidence_effect_means[mode_name] = {}
            mode_verbalised_confidence_effect_counts[mode_name] = {}
            mode_uncertainty_score_means[mode_name] = {}
            mode_uncertainty_score_counts[mode_name] = {}
            for subblock in args.ablate_subblocks:
                sem_mean, sem_count = _mean_and_count(mode_semantic_similarity_values[mode_name][subblock])
                vce_mean, vce_count = _mean_and_count(
                    mode_verbalised_confidence_effect_values[mode_name][subblock]
                )
                unc_mean, unc_count = _mean_and_count(mode_uncertainty_score_values[mode_name][subblock])
                mode_semantic_similarity_means[mode_name][subblock] = sem_mean
                mode_semantic_similarity_counts[mode_name][subblock] = sem_count
                mode_verbalised_confidence_effect_means[mode_name][subblock] = vce_mean
                mode_verbalised_confidence_effect_counts[mode_name][subblock] = vce_count
                mode_uncertainty_score_means[mode_name][subblock] = unc_mean
                mode_uncertainty_score_counts[mode_name][subblock] = unc_count

        return (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            used_none_cache,
            mode_responses_identical_true,
            mode_semantic_similarity_means,
            mode_semantic_similarity_counts,
            mode_verbalised_confidence_effect_means,
            mode_verbalised_confidence_effect_counts,
            mode_uncertainty_score_means,
            mode_uncertainty_score_counts,
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

    compute_derived_metrics = "none" in args.ablation_mode and len(args.ablation_mode) > 1
    sentence_transformer = None
    if compute_derived_metrics and any(m in SEMANTIC_SIMILARITY_MODES for m in args.ablation_mode):
        logging.info(
            "Loading sentence-transformers model %s for semantic_similarity.",
            DEFAULT_SEMANTIC_SIMILARITY_MODEL,
        )
        sentence_transformer = load_sentence_transformer_for_metrics()

    derived_metric_kwargs: Dict[str, object] = {}
    if compute_derived_metrics:
        derived_metric_kwargs = {
            "mode_semantic_similarity_means": None,
            "mode_semantic_similarity_counts": None,
            "mode_verbalised_confidence_effect_means": None,
            "mode_verbalised_confidence_effect_counts": None,
            "mode_uncertainty_score_means": None,
            "mode_uncertainty_score_counts": None,
        }

    if not args.individual_layers:
        (
            results,
            mini_results,
            mode_confidence_means,
            mode_confidence_counts,
            _,
            mode_responses_identical_true,
            mode_semantic_similarity_means,
            mode_semantic_similarity_counts,
            mode_verbalised_confidence_effect_means,
            mode_verbalised_confidence_effect_counts,
            mode_uncertainty_score_means,
            mode_uncertainty_score_counts,
        ) = run_one_evaluation(run_layers, sentence_transformer=sentence_transformer)
        if compute_derived_metrics:
            derived_metric_kwargs = {
                "mode_semantic_similarity_means": mode_semantic_similarity_means,
                "mode_semantic_similarity_counts": mode_semantic_similarity_counts,
                "mode_verbalised_confidence_effect_means": mode_verbalised_confidence_effect_means,
                "mode_verbalised_confidence_effect_counts": mode_verbalised_confidence_effect_counts,
                "mode_uncertainty_score_means": mode_uncertainty_score_means,
                "mode_uncertainty_score_counts": mode_uncertainty_score_counts,
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(mini_output_json_path(out_path), "w", encoding="utf-8") as f:
            json.dump(mini_results, f, ensure_ascii=False, indent=2)
        write_config_txt(
            config_txt_path(out_path),
            args=args,
            device=device,
            model_n_layers=int(model.cfg.n_layers),
            ablate_layers=run_layers,
            prompt_indices=prompt_indices,
            low_conf_count=len(low_ids),
            high_conf_count=len(high_ids),
            h5_example_count=len(examples_h5),
            mode_confidence_means=mode_confidence_means,
            mode_confidence_counts=mode_confidence_counts,
            mode_responses_identical_true=mode_responses_identical_true,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            **derived_metric_kwargs,
        )
        logging.info("Saved full outputs to %s", out_path)
        return

    os.makedirs(individual_root, exist_ok=True)
    none_cache = build_none_cache() if "none" in args.ablation_mode else None
    per_layer_mode_means: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
    for layer_idx in run_layers:
        layer_dir = os.path.join(individual_root, str(layer_idx))
        os.makedirs(layer_dir, exist_ok=True)
        layer_out_path = os.path.join(layer_dir, "ablation_results.json")
        (
            layer_results,
            layer_mini_results,
            layer_mode_means,
            layer_mode_counts,
            used_none_cache,
            layer_identical_true,
            layer_semantic_similarity_means,
            layer_semantic_similarity_counts,
            layer_verbalised_confidence_effect_means,
            layer_verbalised_confidence_effect_counts,
            layer_uncertainty_score_means,
            layer_uncertainty_score_counts,
        ) = run_one_evaluation([layer_idx], cached_none=none_cache, sentence_transformer=sentence_transformer)
        if compute_derived_metrics:
            derived_metric_kwargs = {
                "mode_semantic_similarity_means": layer_semantic_similarity_means,
                "mode_semantic_similarity_counts": layer_semantic_similarity_counts,
                "mode_verbalised_confidence_effect_means": layer_verbalised_confidence_effect_means,
                "mode_verbalised_confidence_effect_counts": layer_verbalised_confidence_effect_counts,
                "mode_uncertainty_score_means": layer_uncertainty_score_means,
                "mode_uncertainty_score_counts": layer_uncertainty_score_counts,
            }
        if none_cache is None and used_none_cache["train"] and used_none_cache["validation"]:
            none_cache = used_none_cache
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
            **derived_metric_kwargs,
        )

    summary_path = os.path.join(individual_root, "mode_confidence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(per_layer_mode_means, f, ensure_ascii=False, indent=2)
    write_individual_layer_plots(
        per_layer_mode_means=per_layer_mode_means,
        ablation_modes=args.ablation_mode,
        ablate_subblocks=args.ablate_subblocks,
        output_dir=individual_root,
    )
    logging.info("Saved individual-layer outputs under %s", individual_root)


if __name__ == "__main__":
    main()
