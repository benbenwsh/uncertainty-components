#!/usr/bin/env python3
"""Gradient × activation attribution on verbalised-confidence tokens.

Attributes a logit-difference direction to coarse (per-layer attn/MLP) and fine
(per-head + MLP) residual writes. Numeric runs use the pre-period and post-period
predicting positions; ``--linguistic_confidence_prompt`` (Mistral only) uses the
variable-length Confidence: span, matching
direct_logit_attribution/run_direct_logit_attribution.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Callable, Optional, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ans_gen.generate_answers_h5 import (
    CONFIDENCE_PROMPT,
    CONFIDENCE_PROMPT_LINGUISTIC,
    get_transformer_layers,
)
from process_generations.process_generations_more_embs_from_h5 import (
    configure_prefix_tokens_for_model,
    parse_guess_and_probability_indices,
)
from direct_logit_attribution.run_direct_logit_attribution import (
    ANNOTATION_ROUNDING_DP,
    ContrastSpec,
    ExperimentResult,
    SUPPORTED_MODEL_NAMES,
    _decode_h5_string,
    _h5_response0_group,
    _is_verbalised_confidence_one,
    _l_vs_un_token_check,
    _log_l_vs_un_skip,
    _open_h5_readonly,
    _read_decoded_tokens,
    _read_response_string,
    _read_verbalised_confidence_scalar,
    attach_output_log,
    build_adapter_from_config,
    component_labels_coarse,
    component_labels_fine,
    high_low_digit_sets,
    linguistic_first_tokens_for_model,
    parse_individual_ids,
    position_index,
    resolve_all_digit_ids,
    resolve_linguistic_first_token_ids,
    resolve_single_piece_token_id,
    validate_confidence_thresholds,
    write_experiment_outputs,
    write_high_minus_low_from_results,
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Match layerwise_mean_ablation/run_mean_ablation.py greedy stop strings.
STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "<end_of_turn>"]


# ---------------------------------------------------------------------------
# H5 I/O (no cached embeddings)
# ---------------------------------------------------------------------------


@dataclass
class GradExample:
    example_id: str
    question: str
    verbalised_confidence: float
    response: str
    decoded_tokens: list[str]
    first_prob_token_index: int
    generated_token_ids: list[int] | None = None


@dataclass
class GradCohorts:
    all: list[GradExample]
    high: list[GradExample]
    low: list[GradExample]
    all_excl_one: list[GradExample]
    high_excl_one: list[GradExample]
    low_excl_one: list[GradExample]


def _read_question(ex_group: h5py.Group) -> str | None:
    ds = ex_group.get("question")
    if ds is None or not isinstance(ds, h5py.Dataset):
        return None
    try:
        text = _decode_h5_string(ds[()])
    except (TypeError, ValueError, OSError):
        return None
    return text if text else None


def _try_parse_grad_example(
    *,
    ex_id: str,
    ex_group: h5py.Group,
    r0: h5py.Group,
    expected_span: int,
    linguistic_confidence_prompt: bool,
    require_decoded_tokens: bool,
) -> GradExample | None:
    conf = _read_verbalised_confidence_scalar(r0)
    if conf is None:
        return None
    question = _read_question(ex_group)
    if question is None:
        return None
    if not require_decoded_tokens:
        return GradExample(
            example_id=str(ex_id),
            question=question,
            verbalised_confidence=float(conf),
            response=_read_response_string(r0),
            decoded_tokens=[],
            first_prob_token_index=0,
        )
    decoded_tokens = _read_decoded_tokens(r0)
    if not decoded_tokens:
        return None
    parsed = parse_guess_and_probability_indices(
        decoded_tokens, linguistic_confidence_prompt=linguistic_confidence_prompt
    )
    if parsed is None:
        return None
    _last_guess, first_prob, _end_prob = parsed
    if first_prob + expected_span > len(decoded_tokens):
        return None
    return GradExample(
        example_id=str(ex_id),
        question=question,
        verbalised_confidence=float(conf),
        response=_read_response_string(r0),
        decoded_tokens=decoded_tokens,
        first_prob_token_index=int(first_prob),
    )


def load_grad_example_cohorts(
    input_h5: str,
    *,
    expected_span: int,
    max_examples: int,
    high_conf_threshold: float,
    low_conf_threshold: float,
    fill_confidence_splits: bool,
    fill_post_period_excl_one: bool,
    linguistic_confidence_prompt: bool = False,
    require_decoded_tokens: bool = True,
) -> GradCohorts:
    examples_all: list[GradExample] = []
    examples_high: list[GradExample] = []
    examples_low: list[GradExample] = []
    examples_all_excl_one: list[GradExample] = []
    examples_high_excl_one: list[GradExample] = []
    examples_low_excl_one: list[GradExample] = []
    with _open_h5_readonly(input_h5) as f:
        examples = f.get("examples")
        if examples is None or not isinstance(examples, h5py.Group):
            raise ValueError(f"{input_h5} has no 'examples' group.")
        example_ids = sorted(examples.keys(), key=lambda x: (len(x), x))
        for ex_id in tqdm(example_ids, desc="Scanning H5 examples"):
            all_full = len(examples_all) >= max_examples
            high_full = (not fill_confidence_splits) or len(examples_high) >= max_examples
            low_full = (not fill_confidence_splits) or len(examples_low) >= max_examples
            all_excl_full = (
                (not fill_post_period_excl_one)
                or len(examples_all_excl_one) >= max_examples
            )
            high_excl_full = (
                (not fill_post_period_excl_one)
                or (not fill_confidence_splits)
                or len(examples_high_excl_one) >= max_examples
            )
            low_excl_full = (
                (not fill_post_period_excl_one)
                or (not fill_confidence_splits)
                or len(examples_low_excl_one) >= max_examples
            )
            if (
                all_full
                and high_full
                and low_full
                and all_excl_full
                and high_excl_full
                and low_excl_full
            ):
                break

            ex_group = examples[ex_id]
            if not isinstance(ex_group, h5py.Group):
                continue
            r0 = _h5_response0_group(ex_group)
            if r0 is None:
                continue
            conf = _read_verbalised_confidence_scalar(r0)
            if conf is None:
                continue
            conf_f = float(conf)
            is_one = _is_verbalised_confidence_one(conf_f)
            want_all = not all_full
            want_high = (
                fill_confidence_splits
                and (not high_full)
                and conf_f >= high_conf_threshold
            )
            want_low = (
                fill_confidence_splits
                and (not low_full)
                and conf_f <= low_conf_threshold
            )
            want_all_excl = (
                fill_post_period_excl_one and (not all_excl_full) and (not is_one)
            )
            want_high_excl = (
                fill_post_period_excl_one
                and fill_confidence_splits
                and (not high_excl_full)
                and (not is_one)
                and conf_f >= high_conf_threshold
            )
            want_low_excl = (
                fill_post_period_excl_one
                and fill_confidence_splits
                and (not low_excl_full)
                and (not is_one)
                and conf_f <= low_conf_threshold
            )
            if not (
                want_all
                or want_high
                or want_low
                or want_all_excl
                or want_high_excl
                or want_low_excl
            ):
                continue

            parsed = _try_parse_grad_example(
                ex_id=str(ex_id),
                ex_group=ex_group,
                r0=r0,
                expected_span=expected_span,
                linguistic_confidence_prompt=linguistic_confidence_prompt,
                require_decoded_tokens=require_decoded_tokens,
            )
            if parsed is None:
                continue
            if want_all:
                examples_all.append(parsed)
            if want_high:
                examples_high.append(parsed)
            if want_low:
                examples_low.append(parsed)
            if want_all_excl:
                examples_all_excl_one.append(parsed)
            if want_high_excl:
                examples_high_excl_one.append(parsed)
            if want_low_excl:
                examples_low_excl_one.append(parsed)

    if not examples_all:
        if require_decoded_tokens:
            raise ValueError(
                f"No usable examples found in {input_h5} with question, decoded_tokens, "
                "verbalised_confidence, and a parseable probability span."
            )
        raise ValueError(
            f"No usable examples found in {input_h5} with question and "
            "verbalised_confidence."
        )
    return GradCohorts(
        all=examples_all,
        high=examples_high,
        low=examples_low,
        all_excl_one=examples_all_excl_one,
        high_excl_one=examples_high_excl_one,
        low_excl_one=examples_low_excl_one,
    )


def _cohort_example_lists(cohorts: GradCohorts) -> list[list[GradExample]]:
    return [
        cohorts.all,
        cohorts.high,
        cohorts.low,
        cohorts.all_excl_one,
        cohorts.high_excl_one,
        cohorts.low_excl_one,
    ]


def _generation_contains_stop(decoded_completion: str) -> bool:
    return any(s in decoded_completion for s in STOP_SEQUENCES)


def _eos_token_ids(tokenizer) -> set[int]:
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


def greedy_generate_hf(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[str], list[int]]:
    """Greedy last-logit decode.

    Returns ``(response, decoded_pieces, generated_token_ids)``. Token ids are the
    sampled ids; do not re-encode decoded pieces (Mistral ``\\n`` is not 1:1).
    """
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    decoded_tokens: list[str] = []
    generated_ids: list[int] = []
    prev_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config"):
        model.config.use_cache = False
    try:
        with torch.inference_mode():
            for _step in range(int(max_new_tokens)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                next_id = int(outputs.logits[0, -1].argmax(dim=-1).item())
                generated_ids.append(next_id)
                piece = tokenizer.decode([next_id], skip_special_tokens=False)
                decoded_tokens.append(piece)
                next_t = torch.tensor(
                    [[next_id]], device=input_ids.device, dtype=input_ids.dtype
                )
                input_ids = torch.cat([input_ids, next_t], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )
                if _should_stop_generation("".join(decoded_tokens), next_id, tokenizer):
                    break
    finally:
        if prev_use_cache is not None and hasattr(model, "config"):
            model.config.use_cache = prev_use_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _strip_stop_suffixes("".join(decoded_tokens)), decoded_tokens, generated_ids


def fill_cohorts_from_generations(
    cohorts: GradCohorts,
    generate_fn: Callable[
        [str], tuple[str, list[str]] | tuple[str, list[str], list[int]]
    ],
    *,
    expected_span: int,
    linguistic_confidence_prompt: bool,
) -> int:
    """Greedy-generate completions and parse probability spans onto shared GradExamples.

    ``generate_fn(question)`` must return ``(response, decoded_tokens)`` or
    ``(response, decoded_tokens, generated_token_ids)``. Prefer the 3-tuple so
    prefix reconstruction can reuse sampled ids (Mistral ``\\n`` is not 1:1 under
    ``encode``). Examples whose new completion has no parseable span of
    ``expected_span`` are dropped from every cohort list. Returns the number of
    unique example ids dropped.
    """
    unique: dict[str, GradExample] = {}
    for lst in _cohort_example_lists(cohorts):
        for ex in lst:
            unique.setdefault(ex.example_id, ex)

    failed: set[str] = set()
    for ex in tqdm(list(unique.values()), desc="Autoregressive generations"):
        try:
            packed = generate_fn(ex.question)
        except (ValueError, RuntimeError, OSError) as exc:
            logging.warning("Skipping example %s: generation failed (%s)", ex.example_id, exc)
            failed.add(ex.example_id)
            continue
        generated_ids: list[int] | None = None
        if len(packed) == 3:
            response, decoded_tokens, generated_ids = packed
            generated_ids = [int(i) for i in generated_ids]
        elif len(packed) == 2:
            response, decoded_tokens = packed
        else:
            raise ValueError(
                "generate_fn must return (response, decoded_tokens) or "
                f"(response, decoded_tokens, generated_token_ids); got {len(packed)} values."
            )
        parsed = parse_guess_and_probability_indices(
            decoded_tokens, linguistic_confidence_prompt=linguistic_confidence_prompt
        )
        if parsed is None:
            logging.warning(
                "Skipping example %s: generated completion has no parseable probability span. "
                "response=%r",
                ex.example_id,
                response,
            )
            failed.add(ex.example_id)
            continue
        _last_guess, first_prob, _end_prob = parsed
        if first_prob + expected_span > len(decoded_tokens):
            logging.warning(
                "Skipping example %s: generated span too short "
                "(first_prob=%d expected_span=%d n_tokens=%d). response=%r",
                ex.example_id,
                first_prob,
                expected_span,
                len(decoded_tokens),
                response,
            )
            failed.add(ex.example_id)
            continue
        if generated_ids is not None and len(generated_ids) != len(decoded_tokens):
            logging.warning(
                "Skipping example %s: generated_token_ids length %d != decoded_tokens %d.",
                ex.example_id,
                len(generated_ids),
                len(decoded_tokens),
            )
            failed.add(ex.example_id)
            continue
        ex.decoded_tokens = decoded_tokens
        ex.generated_token_ids = generated_ids
        ex.first_prob_token_index = int(first_prob)
        ex.response = response

    if failed:
        for lst in _cohort_example_lists(cohorts):
            lst[:] = [ex for ex in lst if ex.example_id not in failed]
        logging.info(
            "Autoregressive fill dropped %d / %d unique examples with unusable completions.",
            len(failed),
            len(unique),
        )
    return len(failed)


# ---------------------------------------------------------------------------
# Sequence reconstruction
# ---------------------------------------------------------------------------


def _extract_tensor(obj) -> torch.Tensor | None:
    if obj is None:
        return None
    if isinstance(obj, (tuple, list)):
        if not obj:
            return None
        return _extract_tensor(obj[0])
    if torch.is_tensor(obj):
        return obj
    return None


def _wrap_output(original, new_tensor: torch.Tensor):
    if isinstance(original, tuple):
        return (new_tensor,) + original[1:]
    if isinstance(original, list):
        return [new_tensor, *original[1:]]
    return new_tensor


def replace_last_pos(full: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """Replace the last sequence position of ``full`` with ``last`` ([B, H])."""
    if full.ndim != 3:
        raise ValueError(f"Expected rank-3 hidden tensor, got shape {tuple(full.shape)}")
    if last.ndim != 2:
        raise ValueError(f"Expected rank-2 last-pos tensor, got shape {tuple(last.shape)}")
    if full.shape[1] == 1:
        return last.unsqueeze(1)
    return torch.cat([full[:, :-1, :].detach(), last.unsqueeze(1)], dim=1)


def token_id_from_decoded(tokenizer, tok: str) -> int:
    """Invert tokenizer.decode([id]) for a stored decoded token string."""
    candidates = [tok]
    if tok == "":
        candidates.extend([" ", "\n"])
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            decoded = tokenizer.decode([ids[0]], skip_special_tokens=False)
            if decoded == tok or (tok == "" and decoded.strip() == ""):
                return int(ids[0])
            if cand != tok and len(ids) == 1:
                return int(ids[0])
    raise ValueError(
        f"Decoded token {tok!r} did not encode to a single tokenizer id "
        f"(encode={tokenizer.encode(tok, add_special_tokens=False)!r})."
    )


def prefix_generated_token_ids(
    example: GradExample,
    abs_gen_index: int,
    tokenizer,
) -> list[int]:
    """Token ids for the completion prefix ``decoded_tokens[:abs_gen_index]``.

    Uses sampled ids from greedy decode when present. Otherwise inverts stored
    decoded strings (``--no-rerun_autoregressive``).
    """
    ids = example.generated_token_ids
    if ids is not None:
        if abs_gen_index > len(ids):
            raise ValueError(
                f"Example {example.example_id}: abs index {abs_gen_index} out of range "
                f"for generated_token_ids (len={len(ids)})."
            )
        return [int(i) for i in ids[:abs_gen_index]]
    return [
        token_id_from_decoded(tokenizer, tok)
        for tok in example.decoded_tokens[:abs_gen_index]
    ]


def build_prefix_input_ids(
    tokenizer,
    example: GradExample,
    abs_gen_index: int,
    *,
    prompt_prefix: str,
    device: torch.device,
) -> torch.Tensor:
    prompt = prompt_prefix + example.question
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ]
    gen_ids = prefix_generated_token_ids(example, abs_gen_index, tokenizer)
    if gen_ids:
        gen = torch.tensor([gen_ids], dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, gen], dim=1)
    else:
        input_ids = prompt_ids
    return input_ids.to(device)


def build_numeric_contrasts(
    digit_ids: dict[str, tuple[int, str]],
    high_digits: set[str],
    low_digits: set[str],
) -> list[ContrastSpec]:
    dummy = np.zeros(1, dtype=np.float32)
    id0 = digit_ids["0"][0]
    id1 = digit_ids["1"][0]
    all_digit_ids = tuple(digit_ids[d][0] for d in "0123456789")
    high_ids = tuple(digit_ids[d][0] for d in sorted(high_digits))
    low_ids = tuple(digit_ids[d][0] for d in sorted(low_digits))
    return [
        ContrastSpec(
            position="pre_period",
            name="digit_vs_rest",
            u=dummy,
            pos_token_ids=(id0, id1),
            neg_token_ids=None,
        ),
        ContrastSpec(
            position="pre_period",
            name="one_vs_zero",
            u=dummy,
            pos_token_ids=(id1,),
            neg_token_ids=(id0,),
        ),
        ContrastSpec(
            position="post_period",
            name="digit_vs_rest",
            u=dummy,
            pos_token_ids=all_digit_ids,
            neg_token_ids=None,
        ),
        ContrastSpec(
            position="post_period",
            name="high_vs_low",
            u=dummy,
            pos_token_ids=high_ids,
            neg_token_ids=low_ids,
        ),
    ]


def build_linguistic_contrast_specs(
    first_token_ids: Sequence[int],
    *,
    l_token_id: int,
    un_token_id: int,
    first_value_index: int,
    l_vs_un_index: int = 6,
) -> list[ContrastSpec]:
    dummy = np.zeros(1, dtype=np.float32)
    pos_ids = tuple(dict.fromkeys(int(i) for i in first_token_ids))
    if not pos_ids:
        raise ValueError("Linguistic first-token id list is empty after deduplication.")
    if first_value_index < 0:
        raise ValueError(f"first_value_index must be >= 0, got {first_value_index}")
    if l_vs_un_index < 0:
        raise ValueError(f"l_vs_un_index must be >= 0, got {l_vs_un_index}")
    return [
        ContrastSpec(
            position="first_value",
            name="confidence_token_vs_rest",
            u=dummy,
            pos_token_ids=pos_ids,
            neg_token_ids=None,
            span_index=int(first_value_index),
        ),
        ContrastSpec(
            position="mistral_l_vs_un",
            name="L_vs_Un",
            u=dummy,
            pos_token_ids=(int(l_token_id),),
            neg_token_ids=(int(un_token_id),),
            span_index=int(l_vs_un_index),
        ),
    ]


def example_span_len(ex: GradExample) -> int:
    return len(ex.decoded_tokens) - ex.first_prob_token_index


def contrast_span_index(contrast: ContrastSpec, example: GradExample, expected_span: int) -> int:
    if contrast.span_index is not None:
        return position_index(
            contrast.position,
            example_span_len(example),
            span_index=contrast.span_index,
        )
    return position_index(contrast.position, expected_span)


def filter_examples_for_contrast(
    examples: Sequence[GradExample],
    contrast: ContrastSpec,
    *,
    skipped_ids: set[str] | None = None,
) -> list[GradExample]:
    kept: list[GradExample] = []
    for ex in examples:
        if contrast.span_index is not None and example_span_len(ex) <= contrast.span_index:
            continue
        if contrast.name == "L_vs_Un":
            if contrast.span_index is None:
                raise ValueError("L_vs_Un contrast is missing span_index.")
            token, first_prob, reason = _l_vs_un_token_check(ex, contrast.span_index)  # type: ignore[arg-type]
            if reason is not None:
                if skipped_ids is None or ex.example_id not in skipped_ids:
                    if skipped_ids is not None:
                        skipped_ids.add(ex.example_id)
                    _log_l_vs_un_skip(
                        ex,  # type: ignore[arg-type]
                        span_index=contrast.span_index,
                        token=token,
                        first_prob_token_index=first_prob,
                        reason=reason,
                    )
                continue
        kept.append(ex)
    return kept


def logit_diff_from_logits(
    logits: torch.Tensor,
    pos_ids: Sequence[int],
    neg_ids: Sequence[int] | None,
) -> torch.Tensor:
    """pos mean minus neg mean on 1D vocab logits (float32)."""
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")
    pos_idx = torch.as_tensor(list(pos_ids), device=logits.device, dtype=torch.long)
    pos_mean = logits.index_select(0, pos_idx).mean()
    if neg_ids is None:
        n_pos = int(pos_idx.numel())
        vocab = int(logits.numel())
        if vocab <= n_pos:
            raise ValueError("No negative tokens remain for contrast.")
        pos_sum = logits.index_select(0, pos_idx).sum()
        neg_mean = (logits.sum() - pos_sum) / (vocab - n_pos)
    else:
        neg_idx = torch.as_tensor(list(neg_ids), device=logits.device, dtype=torch.long)
        neg_mean = logits.index_select(0, neg_idx).mean()
    return pos_mean - neg_mean


def grad_x_act(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    if t.grad is None:
        return 0.0
    return float((t.detach().float() * t.grad.float()).sum().item())


def gemma_rms_scale(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-token scale so (x * scale) matches Gemma RMSNorm: x / rms * (1+w)."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return inv * (1.0 + weight.float())


def _head_writes_from_concat(
    concat_last: torch.Tensor,
    o_proj,
    n_heads: int,
    head_dim: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    heads: list[torch.Tensor] = []
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        write_h = F.linear(concat_last[..., sl], o_proj.weight[:, sl])
        heads.append(write_h)
    total = heads[0]
    for h_t in heads[1:]:
        total = total + h_t
    if getattr(o_proj, "bias", None) is not None:
        total = total + o_proj.bias
    return heads, total


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class ComponentHooks:
    """Capture last-position residual writes as autograd leaves."""

    def __init__(
        self,
        *,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        use_gemma_post_norm: bool,
        need_fine: bool,
        rms_eps: float,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_gemma_post_norm = use_gemma_post_norm
        self.need_fine = need_fine
        self.rms_eps = rms_eps
        self.handles: list = []
        self.concat_last: list[torch.Tensor | None] = [None] * n_layers
        self.heads: list[list[torch.Tensor] | None] = [None] * n_layers
        self.attn: list[torch.Tensor | None] = [None] * n_layers
        self.mlp: list[torch.Tensor | None] = [None] * n_layers
        self.embed: torch.Tensor | None = None

    def clear_captured(self) -> None:
        self.concat_last = [None] * self.n_layers
        self.heads = [None] * self.n_layers
        self.attn = [None] * self.n_layers
        self.mlp = [None] * self.n_layers
        self.embed = None

    def iter_param_tensors(self):
        if self.embed is not None:
            yield self.embed
        for t in self.attn:
            if t is not None:
                yield t
        for t in self.mlp:
            if t is not None:
                yield t
        for heads in self.heads:
            if heads is None:
                continue
            for h_t in heads:
                yield h_t

    def zero_grads(self) -> None:
        for t in self.iter_param_tensors():
            if t.grad is not None:
                t.grad = None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def register(self, model) -> None:
        layers = get_transformer_layers(model)
        if layers is None:
            raise ValueError("Could not locate transformer layers on the loaded model.")
        if len(layers) != self.n_layers:
            raise ValueError(
                f"Adapter n_layers={self.n_layers} != found layers={len(layers)}"
            )

        embed = model.get_input_embeddings()

        def embed_hook(_mod, _inp, out):
            t = _extract_tensor(out)
            if t is None:
                return out
            last = t[:, -1, :].detach().requires_grad_(True)
            self.embed = last
            return _wrap_output(out, replace_last_pos(t, last))

        self.handles.append(embed.register_forward_hook(embed_hook))

        for idx, layer in enumerate(layers):
            self._register_layer(idx, layer)

    def _register_layer(self, idx: int, layer) -> None:
        self_attn = getattr(layer, "self_attn", None)
        o_proj = getattr(self_attn, "o_proj", None) if self_attn is not None else None
        if o_proj is None:
            raise ValueError(f"Layer {idx} has no self_attn.o_proj.")

        if self.need_fine:
            self._register_fine_attn(idx, layer, o_proj)
        else:
            self._register_coarse_attn(idx, layer, o_proj)

        if self.use_gemma_post_norm and hasattr(layer, "post_feedforward_layernorm"):
            mlp_mod = layer.post_feedforward_layernorm
        else:
            mlp_mod = getattr(layer, "mlp", None)
        if mlp_mod is None:
            raise ValueError(f"Layer {idx} has no MLP residual-write module.")

        def mlp_hook(_mod, _inp, out, layer_idx=idx):
            t = _extract_tensor(out)
            if t is None:
                return out
            last = t[:, -1, :].detach().requires_grad_(True)
            self.mlp[layer_idx] = last
            return _wrap_output(out, replace_last_pos(t, last))

        self.handles.append(mlp_mod.register_forward_hook(mlp_hook))

    def _register_coarse_attn(self, idx: int, layer, o_proj) -> None:
        if self.use_gemma_post_norm and hasattr(layer, "post_attention_layernorm"):
            attn_mod = layer.post_attention_layernorm
        else:
            attn_mod = o_proj

        def attn_hook(_mod, _inp, out, layer_idx=idx):
            t = _extract_tensor(out)
            if t is None:
                return out
            last = t[:, -1, :].detach().requires_grad_(True)
            self.attn[layer_idx] = last
            return _wrap_output(out, replace_last_pos(t, last))

        self.handles.append(attn_mod.register_forward_hook(attn_hook))

    def _register_fine_attn(self, idx: int, layer, o_proj) -> None:
        if self.use_gemma_post_norm and hasattr(layer, "post_attention_layernorm"):

            def o_proj_hook(_mod, inp, out, layer_idx=idx):
                concat = _extract_tensor(inp)
                if concat is not None:
                    self.concat_last[layer_idx] = concat[:, -1, :].detach()
                t = _extract_tensor(out)
                if t is None:
                    return out
                detached = t.detach()
                return _wrap_output(out, replace_last_pos(detached, detached[:, -1, :]))

            def post_attn_hook(mod, _inp, out, layer_idx=idx, proj=o_proj):
                concat_last = self.concat_last[layer_idx]
                if concat_last is None:
                    return out
                concat_leaf = concat_last.detach().requires_grad_(True)
                heads, total_pre = _head_writes_from_concat(
                    concat_leaf, proj, self.n_heads, self.head_dim
                )
                eps = float(getattr(mod, "eps", self.rms_eps))
                scale = gemma_rms_scale(total_pre, mod.weight, eps)
                post_heads = [(h_t.float() * scale).to(dtype=h_t.dtype) for h_t in heads]
                for ph in post_heads:
                    ph.retain_grad()
                total_post = post_heads[0]
                for h_t in post_heads[1:]:
                    total_post = total_post + h_t
                total_post.retain_grad()
                self.heads[layer_idx] = post_heads
                self.attn[layer_idx] = total_post
                t = _extract_tensor(out)
                if t is None:
                    return out
                return _wrap_output(out, replace_last_pos(t.detach(), total_post))

            self.handles.append(o_proj.register_forward_hook(o_proj_hook))
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(post_attn_hook)
            )
            return

        def o_proj_replace_hook(mod, inp, out, layer_idx=idx):
            concat = _extract_tensor(inp)
            t = _extract_tensor(out)
            if concat is None or t is None:
                return out
            concat_leaf = concat[:, -1, :].detach().requires_grad_(True)
            heads, total = _head_writes_from_concat(
                concat_leaf, mod, self.n_heads, self.head_dim
            )
            for h_t in heads:
                h_t.retain_grad()
            total.retain_grad()
            self.heads[layer_idx] = heads
            self.attn[layer_idx] = total
            return _wrap_output(out, replace_last_pos(t.detach(), total))

        self.handles.append(o_proj.register_forward_hook(o_proj_replace_hook))


# ---------------------------------------------------------------------------
# Model load / run
# ---------------------------------------------------------------------------


def load_causal_lm(model_name: str, *, dtype: torch.dtype, device: str):
    hf_token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token
        )
    except Exception as exc:
        logging.warning("Fast tokenizer load failed (%s). Retrying use_fast=False.", exc)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            token=hf_token,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "token": hf_token,
        "torch_dtype": dtype,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return tokenizer, model


def embed_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _attrs_from_capture(
    capture: ComponentHooks,
    *,
    granularity: str,
) -> tuple[list[str], np.ndarray, float]:
    n_layers = capture.n_layers
    n_heads = capture.n_heads
    if granularity == "coarse":
        labels = component_labels_coarse(n_layers)
        row = np.zeros(len(labels), dtype=np.float32)
        for l in range(n_layers):
            row[2 * l] = grad_x_act(capture.attn[l])
            row[2 * l + 1] = grad_x_act(capture.mlp[l])
        return labels, row, grad_x_act(capture.embed)
    labels = component_labels_fine(n_layers, n_heads)
    row = np.zeros(len(labels), dtype=np.float32)
    col = 0
    for l in range(n_layers):
        heads = capture.heads[l]
        if heads is None or len(heads) != n_heads:
            raise RuntimeError(
                f"Fine attribution missing heads at layer {l} "
                f"(got {None if heads is None else len(heads)})."
            )
        for h_t in heads:
            row[col] = grad_x_act(h_t)
            col += 1
        row[col] = grad_x_act(capture.mlp[l])
        col += 1
    return labels, row, grad_x_act(capture.embed)


def run_attribution_for_example(
    *,
    model,
    tokenizer,
    capture: ComponentHooks,
    example: GradExample,
    contrasts: Sequence[ContrastSpec],
    expected_span: int,
    gran_list: Sequence[str],
    prompt_prefix: str,
) -> dict[tuple[str, str, str], tuple[np.ndarray, float, float]]:
    """Return {(position, contrast, gran): (row, embed_attr, logit_diff)} for one example."""
    device = embed_device(model)
    out: dict[tuple[str, str, str], tuple[np.ndarray, float, float]] = {}
    by_position: dict[str, list[ContrastSpec]] = {}
    for c in contrasts:
        by_position.setdefault(c.position, []).append(c)

    for position, pos_contrasts in by_position.items():
        span_idx = contrast_span_index(pos_contrasts[0], example, expected_span)
        for extra in pos_contrasts[1:]:
            other = contrast_span_index(extra, example, expected_span)
            if other != span_idx:
                raise ValueError(
                    f"Contrasts at position {position!r} disagree on span index "
                    f"({span_idx} vs {other})."
                )
        abs_idx = example.first_prob_token_index + span_idx
        if abs_idx < 0 or abs_idx >= len(example.decoded_tokens):
            raise ValueError(
                f"Example {example.example_id}: abs index {abs_idx} out of range "
                f"for {position} (len={len(example.decoded_tokens)})."
            )
        input_ids = build_prefix_input_ids(
            tokenizer,
            example,
            abs_idx,
            prompt_prefix=prompt_prefix,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        capture.clear_captured()
        capture.zero_grads()

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
            pass
        with torch.enable_grad():
            outputs = model(**model_inputs, use_cache=False)
            missing = [
                i
                for i in range(capture.n_layers)
                if capture.attn[i] is None or capture.mlp[i] is None
            ]
            if missing:
                raise RuntimeError(
                    f"Component hooks missed attn/mlp last-pos tensors at layers {missing[:8]}."
                )
            if capture.need_fine:
                missing_heads = [
                    i
                    for i in range(capture.n_layers)
                    if capture.heads[i] is None
                    or len(capture.heads[i]) != capture.n_heads
                ]
                if missing_heads:
                    raise RuntimeError(
                        f"Fine hooks missed head writes at layers {missing_heads[:8]}."
                    )
            logits = outputs.logits[0, -1].float()

            for ci, contrast in enumerate(pos_contrasts):
                retain = ci < len(pos_contrasts) - 1
                ld = logit_diff_from_logits(
                    logits, contrast.pos_token_ids, contrast.neg_token_ids
                )
                ld.backward(retain_graph=retain)
                ld_val = float(ld.detach().item())
                for gran in gran_list:
                    labels, row, embed_attr = _attrs_from_capture(
                        capture, granularity=gran
                    )
                    del labels
                    out[(position, contrast.name, gran)] = (row, embed_attr, ld_val)
                capture.zero_grads()

        del outputs, logits, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def run_cohort_attributions(
    *,
    examples: Sequence[GradExample],
    model,
    tokenizer,
    capture: ComponentHooks,
    contrasts: Sequence[ContrastSpec],
    expected_span: int,
    gran_list: Sequence[str],
    desc: str,
    prompt_prefix: str,
) -> dict[tuple[str, str, str], ExperimentResult]:
    n_ex = len(examples)
    n_layers = capture.n_layers
    n_heads = capture.n_heads
    label_map = {
        "coarse": component_labels_coarse(n_layers),
        "fine": component_labels_fine(n_layers, n_heads),
    }
    keys = [(c.position, c.name, g) for c in contrasts for g in gran_list]
    attrs: dict[tuple[str, str, str], np.ndarray] = {}
    embeds: dict[tuple[str, str, str], np.ndarray] = {}
    scores: dict[tuple[str, str, str], np.ndarray] = {}
    for key in keys:
        n_comp = len(label_map[key[2]])
        attrs[key] = np.zeros((n_ex, n_comp), dtype=np.float32)
        embeds[key] = np.zeros(n_ex, dtype=np.float32)
        scores[key] = np.zeros(n_ex, dtype=np.float32)

    example_ids: list[str] = []
    skipped = 0
    ei = 0
    for ex in tqdm(examples, desc=desc):
        try:
            per = run_attribution_for_example(
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                example=ex,
                contrasts=contrasts,
                expected_span=expected_span,
                gran_list=gran_list,
                prompt_prefix=prompt_prefix,
            )
        except (ValueError, RuntimeError) as exc:
            skipped += 1
            logging.warning("Skipping example %s: %s", ex.example_id, exc)
            continue
        if ei >= n_ex:
            break
        example_ids.append(ex.example_id)
        for key in keys:
            row, embed_attr, ld_val = per[key]
            attrs[key][ei] = row
            embeds[key][ei] = embed_attr
            scores[key][ei] = ld_val
        ei += 1

    if skipped:
        logging.info("%s skipped %d examples.", desc, skipped)
    if ei == 0:
        raise ValueError(f"No examples succeeded for cohort {desc!r}.")

    results: dict[tuple[str, str, str], ExperimentResult] = {}
    for position, name, gran in keys:
        key = (position, name, gran)
        results[key] = ExperimentResult(
            position=position,
            contrast_name=name,
            granularity=gran,
            labels=label_map[gran],
            example_ids=example_ids,
            attributions=attrs[key][:ei],
            embed_attributions=embeds[key][:ei],
            true_scores=scores[key][:ei],
            completeness_residuals=np.zeros(ei, dtype=np.float32),
        )
    return results


# ---------------------------------------------------------------------------
# Run scaffolding
# ---------------------------------------------------------------------------


def resolve_run_root(cli_output_dir: Optional[str]) -> Path:
    if cli_output_dir:
        root = Path(cli_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
    base = SCRIPT_DIR / "results"
    base.mkdir(parents=True, exist_ok=True)
    existing = [d for d in os.listdir(base) if (base / d).is_dir() and d.isdigit()]
    run_idx = max((int(d) for d in existing), default=0) + 1
    root = base / str(run_idx)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_readme(path: Path) -> None:
    path.write_text(
        "# Gradient-based attribution results\n\n"
        "Each component score is gradient × activation of that component's residual "
        "write at the predicting token position: "
        "`⟨∂ logit_diff / ∂ z_c[T], z_c[T]⟩`.\n\n"
        "Unlike Direct Logit Attribution, this includes *indirect* paths through "
        "later layers. Completeness (sum of components = logit_diff) is not expected.\n\n"
        "Numeric Probability: experiments nest under `pre_period/` and `post_period/` "
        "as `{position}/{contrast}__{granularity}`. Post-period experiments drop "
        "samples whose verbalised_confidence is 1.0 and refill `--max_examples_for_mean` "
        "independently.\n\n"
        "Linguistic Confidence (`--linguistic_confidence_prompt`, Mistral only) writes "
        "experiments flat as `{contrast}__{granularity}`. `confidence_token_vs_rest` is "
        "attributed at the first phrase token (`expected_confidence_tokens - 1`). "
        "`L_vs_Un` uses span index 6 and keeps only examples whose decoded token there "
        "is `L` or `Un`. Other models raise.\n\n"
        "When `--confidence_split` is enabled (default), high- and low-confidence "
        "cohorts are written as `...__high_conf/` and `...__low_conf/`. "
        "`...__high_minus_low/` ranks components by signed "
        "(high_conf mean − low_conf mean).\n\n"
        "`--rerun_autoregressive` (default) greedy-decodes a new Guess:/Probability: "
        "completion from each H5 question. Prefix reconstruction uses the sampled "
        "token ids (not re-encoded decoded strings). The H5 is then only a catalog of "
        "example ids, questions, and original verbalised_confidence labels. "
        "`--no-rerun_autoregressive` reconstructs the prefix from stored "
        "`decoded_tokens` instead.\n",
        encoding="utf-8",
    )


def write_config_txt(
    path: Path,
    *,
    args: argparse.Namespace,
    input_h5_resolved: str,
    finished_at: str,
    n_layers: int,
    n_heads: int,
    high_digits: set[str] | None,
    low_digits: set[str] | None,
    first_token_ids: dict[str, tuple[int, str]] | None,
    l_vs_un_token_ids: dict[str, tuple[int, str]] | None,
    l_vs_un_skipped_not_l_or_un: int | None,
    actual_n: dict[str, int],
) -> None:
    lines = [
        "Gradient-based Attribution Config",
        "=================================",
        "",
        f"finished_at={finished_at}",
        f"script={Path(__file__).resolve()}",
        f"input_h5={input_h5_resolved}",
        f"output_dir={args.output_dir}",
        f"device={args.device}",
        f"dtype={args.dtype}",
        f"model_name={args.model_name}",
        f"n_layers={n_layers}",
        f"n_heads={n_heads}",
        "",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_confidence_tokens={args.expected_confidence_tokens}",
        f"linguistic_confidence_prompt={args.linguistic_confidence_prompt}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"confidence_split={args.confidence_split}",
        f"rerun_autoregressive={args.rerun_autoregressive}",
        f"model_max_new_tokens={args.model_max_new_tokens}",
        f"granularity={args.granularity}",
        f"max_examples_for_mean={args.max_examples_for_mean}",
        f"n_individual_examples={args.n_individual_examples}",
        f"bar_chart_top_k={args.bar_chart_top_k}",
        f"high_digits={None if high_digits is None else sorted(high_digits)}",
        f"low_digits={None if low_digits is None else sorted(low_digits)}",
        "",
    ]
    for k, v in actual_n.items():
        lines.append(f"{k}={v}")
    if args.linguistic_confidence_prompt:
        lines += ["", "[LinguisticFirstTokens]"]
        if first_token_ids is None:
            raise ValueError("linguistic run is missing resolved first_token_ids.")
        for piece, (tid, form) in first_token_ids.items():
            lines.append(f"piece={piece!r} token_id={tid} resolved_form={form!r}")
        lines += ["", "[LinguisticLvsUn]"]
        if l_vs_un_token_ids is None:
            raise ValueError("linguistic run is missing resolved L_vs_Un token ids.")
        for piece, (tid, form) in l_vs_un_token_ids.items():
            lines.append(f"piece={piece!r} token_id={tid} resolved_form={form!r}")
        lines.append("span_index=6")
        skipped_n = (
            0 if l_vs_un_skipped_not_l_or_un is None else int(l_vs_un_skipped_not_l_or_un)
        )
        lines.append(f"l_vs_un_skipped_not_l_or_un={skipped_n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Gradient × activation attribution on verbalised-confidence tokens "
            "(numeric pre/post-period digits, or Mistral linguistic L vs Un)."
        )
    )
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODEL_NAMES),
    )
    p.add_argument("--input_h5", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP),
    )
    p.add_argument("--expected_probability_tokens", type=int, default=5)
    p.add_argument(
        "--expected_confidence_tokens",
        type=int,
        default=5,
        help=(
            "When --linguistic_confidence_prompt, unextended Confidence: span length "
            "(prefix plus first phrase token). Stored spans may be longer."
        ),
    )
    p.add_argument(
        "--linguistic_confidence_prompt",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, attribute linguistic Confidence: (Mistral only). "
            "confidence_token_vs_rest at the first phrase token, plus L_vs_Un at "
            "span index 6 (kept only when that decoded token is L or Un)."
        ),
    )
    p.add_argument("--low_conf_threshold", type=float, default=0.1)
    p.add_argument("--high_conf_threshold", type=float, default=0.9)
    p.add_argument(
        "--confidence_split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--rerun_autoregressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), greedy-generate Guess:/Probability: from each H5 "
            "question and attribute that prefix using the sampled token ids. The H5 "
            "supplies example ids, questions, and verbalised_confidence cohort labels "
            "only. If false, reconstruct from stored decoded_tokens."
        ),
    )
    p.add_argument(
        "--model_max_new_tokens",
        type=int,
        default=50,
        help="Max new tokens when --rerun_autoregressive (default: 50).",
    )
    p.add_argument(
        "--granularity",
        nargs="+",
        choices=["coarse", "fine"],
        default=["coarse", "fine"],
    )
    p.add_argument("--n_individual_examples", type=int, default=3)
    p.add_argument("--max_examples_for_mean", type=int, default=50)
    p.add_argument("--individual_example_indices", type=str, default=None)
    p.add_argument("--bar_chart_top_k", type=int, default=25)
    p.add_argument("--output_dir", type=str, default=None)
    return p


def _write_cohort_dirs(
    *,
    run_root: Path,
    results: dict[tuple[str, str, str], ExperimentResult],
    examples_by_id: dict[str, GradExample],
    individual_ids: Sequence[str],
    dirname_suffix: str,
    n_layers: int,
    n_heads: int,
    bar_chart_top_k: int,
    linguistic: bool,
) -> None:
    for (position, name, gran), result in results.items():
        leaf = f"{name}__{gran}{dirname_suffix}"
        if linguistic:
            exp_dir = run_root / leaf
            loc = leaf
        else:
            exp_dir = run_root / position / leaf
            loc = f"{position}/{leaf}"
        logging.info("Writing %s (n=%d)", loc, len(result.example_ids))
        contrast_ids = [eid for eid in individual_ids if eid in set(result.example_ids)]
        if len(contrast_ids) < min(len(individual_ids), len(result.example_ids)):
            for eid in result.example_ids:
                if eid not in contrast_ids:
                    contrast_ids.append(eid)
                if len(contrast_ids) >= len(individual_ids):
                    break
        write_experiment_outputs(
            exp_dir,
            result,
            examples_by_id,  # type: ignore[arg-type]
            contrast_ids[: len(individual_ids)],
            n_layers=n_layers,
            n_heads=n_heads,
            bar_chart_top_k=bar_chart_top_k,
            rounding_dp=ANNOTATION_ROUNDING_DP,
            method_label="grad×act",
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_examples_for_mean <= args.n_individual_examples:
        raise ValueError(
            f"--max_examples_for_mean ({args.max_examples_for_mean}) must be strictly "
            f"greater than --n_individual_examples ({args.n_individual_examples})."
        )
    if args.n_individual_examples < 1:
        raise ValueError("--n_individual_examples must be >= 1")
    if args.bar_chart_top_k < 1:
        raise ValueError("--bar_chart_top_k must be >= 1")
    if args.model_max_new_tokens < 1:
        raise ValueError("--model_max_new_tokens must be >= 1")
    validate_confidence_thresholds(args.high_conf_threshold, args.low_conf_threshold)
    if args.linguistic_confidence_prompt:
        if args.expected_confidence_tokens < 1:
            raise ValueError("--expected_confidence_tokens must be >= 1")
        if args.model_name != "mistralai/Mistral-7B-Instruct-v0.1":
            raise ValueError(
                "--linguistic_confidence_prompt is currently only implemented for "
                "mistralai/Mistral-7B-Instruct-v0.1 "
                f"(got {args.model_name!r})."
            )
        linguistic_first_tokens_for_model(args.model_name)

    gran_list: list[str] = []
    for g in args.granularity:
        if g not in gran_list:
            gran_list.append(g)
    args.granularity = gran_list
    need_fine = "fine" in gran_list

    input_h5 = str(Path(args.input_h5).resolve())
    if not os.path.isfile(input_h5):
        raise FileNotFoundError(f"--input_h5 not found: {input_h5}")

    run_root = resolve_run_root(args.output_dir)
    attach_output_log(run_root)
    logging.info("Run directory: %s", run_root)
    write_readme(run_root / "README.md")

    configure_prefix_tokens_for_model(args.model_name)
    adapter = build_adapter_from_config(args.model_name)
    logging.info(
        "Adapter: n_layers=%d n_heads=%d head_dim=%d hidden=%d",
        adapter.n_layers,
        adapter.n_query_heads,
        adapter.head_dim,
        adapter.hidden_size,
    )

    dtype = DTYPE_MAP[args.dtype]
    logging.info("Loading %s dtype=%s device=%s", args.model_name, args.dtype, args.device)
    tokenizer, model = load_causal_lm(args.model_name, dtype=dtype, device=args.device)

    high_digits: set[str] | None = None
    low_digits: set[str] | None = None
    first_token_ids: dict[str, tuple[int, str]] | None = None
    l_vs_un_token_ids: dict[str, tuple[int, str]] | None = None
    linguistic = bool(args.linguistic_confidence_prompt)
    if linguistic:
        pieces = linguistic_first_tokens_for_model(args.model_name)
        first_token_ids = resolve_linguistic_first_token_ids(tokenizer, pieces)
        for piece, (tid, form) in first_token_ids.items():
            logging.info("Linguistic first token %r -> id=%s form=%r", piece, tid, form)
        l_id, l_form = resolve_single_piece_token_id(tokenizer, "L")
        un_id, un_form = resolve_single_piece_token_id(tokenizer, "Un")
        l_vs_un_token_ids = {"L": (l_id, l_form), "Un": (un_id, un_form)}
        logging.info(
            "L vs Un: L id=%s form=%r; Un id=%s form=%r", l_id, l_form, un_id, un_form
        )
        expected_span = args.expected_confidence_tokens
        prompt_prefix = CONFIDENCE_PROMPT_LINGUISTIC
        contrasts = build_linguistic_contrast_specs(
            [tid for tid, _form in first_token_ids.values()],
            l_token_id=l_id,
            un_token_id=un_id,
            first_value_index=args.expected_confidence_tokens - 1,
        )
        fill_post_period_excl_one = False
    else:
        digit_ids = resolve_all_digit_ids(tokenizer)
        high_digits, low_digits = high_low_digit_sets(
            args.high_conf_threshold, args.low_conf_threshold
        )
        logging.info(
            "High digits %s  Low digits %s", sorted(high_digits), sorted(low_digits)
        )
        expected_span = args.expected_probability_tokens + 2
        prompt_prefix = CONFIDENCE_PROMPT
        contrasts = build_numeric_contrasts(digit_ids, high_digits, low_digits)
        fill_post_period_excl_one = True

    cohorts = load_grad_example_cohorts(
        input_h5,
        expected_span=expected_span,
        max_examples=args.max_examples_for_mean,
        high_conf_threshold=args.high_conf_threshold,
        low_conf_threshold=args.low_conf_threshold,
        fill_confidence_splits=args.confidence_split,
        fill_post_period_excl_one=fill_post_period_excl_one,
        linguistic_confidence_prompt=linguistic,
        require_decoded_tokens=not args.rerun_autoregressive,
    )
    if args.rerun_autoregressive:
        logging.info(
            "Rerunning autoregressive greedy decode (max_new_tokens=%d).",
            args.model_max_new_tokens,
        )
        gen_device = embed_device(model)

        def _generate_one(question: str) -> tuple[str, list[str], list[int]]:
            return greedy_generate_hf(
                model,
                tokenizer,
                prompt_prefix + question,
                max_new_tokens=args.model_max_new_tokens,
                device=gen_device,
            )

        fill_cohorts_from_generations(
            cohorts,
            _generate_one,
            expected_span=expected_span,
            linguistic_confidence_prompt=linguistic,
        )
        if not cohorts.all:
            raise ValueError(
                "No usable examples remain after autoregressive generation "
                "(no parseable probability span of expected length)."
            )
    logging.info("Using %d all-examples (cap=%d)", len(cohorts.all), args.max_examples_for_mean)
    if len(cohorts.all) <= args.n_individual_examples:
        raise ValueError(
            f"All-examples cohort has {len(cohorts.all)} usable examples, which must be "
            f"strictly greater than --n_individual_examples ({args.n_individual_examples})."
        )
    if fill_post_period_excl_one:
        if not cohorts.all_excl_one:
            raise ValueError(
                "No usable post-period examples after excluding verbalised_confidence == 1.0."
            )
        if len(cohorts.all_excl_one) <= args.n_individual_examples:
            raise ValueError(
                f"Post-period all-examples cohort has {len(cohorts.all_excl_one)} examples "
                "after excluding confidence 1.0, which must be strictly greater than "
                f"--n_individual_examples ({args.n_individual_examples})."
            )

    requested_ids = parse_individual_ids(args.individual_example_indices)
    usable_ids_all = [ex.example_id for ex in cohorts.all]
    if requested_ids is not None:
        missing = [i for i in requested_ids if i not in set(usable_ids_all)]
        if missing:
            raise ValueError(
                f"--individual_example_indices not in usable all-examples set: {missing}"
            )
        individual_ids_all = requested_ids
    else:
        individual_ids_all = usable_ids_all[: args.n_individual_examples]

    individual_ids_high: list[str] | None = None
    individual_ids_low: list[str] | None = None
    if args.confidence_split:
        if not cohorts.high or not cohorts.low:
            raise ValueError("confidence_split enabled but high or low cohort is empty.")
        if len(cohorts.high) <= args.n_individual_examples:
            raise ValueError(
                f"High-confidence cohort has {len(cohorts.high)} examples; "
                f"need > {args.n_individual_examples}."
            )
        if len(cohorts.low) <= args.n_individual_examples:
            raise ValueError(
                f"Low-confidence cohort has {len(cohorts.low)} examples; "
                f"need > {args.n_individual_examples}."
            )
        if fill_post_period_excl_one and (
            not cohorts.high_excl_one or not cohorts.low_excl_one
        ):
            raise ValueError(
                "Post-period high or low cohort is empty after excluding confidence 1.0."
            )
        individual_ids_high = [ex.example_id for ex in cohorts.high][
            : args.n_individual_examples
        ]
        individual_ids_low = [ex.example_id for ex in cohorts.low][
            : args.n_individual_examples
        ]
        logging.info(
            "Using %d high-conf / %d low-conf examples",
            len(cohorts.high),
            len(cohorts.low),
        )

    layers = get_transformer_layers(model)
    if layers is None:
        raise ValueError("Could not locate transformer layers.")
    use_gemma_post_norm = any(
        hasattr(layer, "post_feedforward_layernorm") for layer in layers
    )
    logging.info(
        "Attn/MLP hook targets: %s",
        (
            "post_attention_layernorm / post_feedforward_layernorm"
            if use_gemma_post_norm
            else "o_proj / mlp"
        ),
    )

    capture = ComponentHooks(
        n_layers=adapter.n_layers,
        n_heads=adapter.n_query_heads,
        head_dim=adapter.head_dim,
        use_gemma_post_norm=use_gemma_post_norm,
        need_fine=need_fine,
        rms_eps=adapter.rms_norm_eps,
    )
    capture.register(model)

    l_vs_un_skipped_ids: set[str] = set()

    def _run_and_write(
        *,
        cohort_examples: list[GradExample],
        post_period_examples: list[GradExample] | None,
        individual_ids: Sequence[str],
        dirname_suffix: str,
        cohort_label: str,
    ) -> dict[tuple[str, str, str], ExperimentResult]:
        combined: dict[tuple[str, str, str], ExperimentResult] = {}
        examples_by_id: dict[str, GradExample] = {}
        if linguistic:
            for contrast in contrasts:
                filtered = filter_examples_for_contrast(
                    cohort_examples, contrast, skipped_ids=l_vs_un_skipped_ids
                )
                if not filtered:
                    need = (
                        "any usable span"
                        if contrast.span_index is None
                        else f"span_len > {contrast.span_index}"
                    )
                    raise ValueError(
                        f"No {cohort_label} examples with {need} for contrast "
                        f"{contrast.name!r}."
                    )
                if len(filtered) <= args.n_individual_examples:
                    raise ValueError(
                        f"{cohort_label} cohort for contrast {contrast.name!r} has "
                        f"{len(filtered)} examples after span filtering, which must be "
                        f"strictly greater than --n_individual_examples "
                        f"({args.n_individual_examples})."
                    )
                part = run_cohort_attributions(
                    examples=filtered,
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    contrasts=[contrast],
                    expected_span=expected_span,
                    gran_list=gran_list,
                    desc=f"{cohort_label}/{contrast.name}",
                    prompt_prefix=prompt_prefix,
                )
                combined.update(part)
                for ex in filtered:
                    examples_by_id[ex.example_id] = ex
        else:
            if post_period_examples is None:
                raise ValueError("Numeric runs require post-period examples.")
            pre_contrasts = [c for c in contrasts if c.position == "pre_period"]
            post_contrasts = [c for c in contrasts if c.position == "post_period"]
            pre_results = run_cohort_attributions(
                examples=cohort_examples,
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                contrasts=pre_contrasts,
                expected_span=expected_span,
                gran_list=gran_list,
                desc=f"{cohort_label}/pre_period",
                prompt_prefix=prompt_prefix,
            )
            post_results = run_cohort_attributions(
                examples=post_period_examples,
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                contrasts=post_contrasts,
                expected_span=expected_span,
                gran_list=gran_list,
                desc=f"{cohort_label}/post_period",
                prompt_prefix=prompt_prefix,
            )
            combined = {**pre_results, **post_results}
            examples_by_id = {
                ex.example_id: ex
                for ex in list(cohort_examples) + list(post_period_examples)
            }
        _write_cohort_dirs(
            run_root=run_root,
            results=combined,
            examples_by_id=examples_by_id,
            individual_ids=individual_ids,
            dirname_suffix=dirname_suffix,
            n_layers=adapter.n_layers,
            n_heads=adapter.n_query_heads,
            bar_chart_top_k=args.bar_chart_top_k,
            linguistic=linguistic,
        )
        return combined

    all_results = _run_and_write(
        cohort_examples=cohorts.all,
        post_period_examples=(
            cohorts.all_excl_one if fill_post_period_excl_one else None
        ),
        individual_ids=individual_ids_all,
        dirname_suffix="",
        cohort_label="all-examples",
    )
    del all_results

    if args.confidence_split:
        high_results = _run_and_write(
            cohort_examples=cohorts.high,
            post_period_examples=(
                cohorts.high_excl_one if fill_post_period_excl_one else None
            ),
            individual_ids=individual_ids_high or [],
            dirname_suffix="__high_conf",
            cohort_label="high-confidence",
        )
        low_results = _run_and_write(
            cohort_examples=cohorts.low,
            post_period_examples=(
                cohorts.low_excl_one if fill_post_period_excl_one else None
            ),
            individual_ids=individual_ids_low or [],
            dirname_suffix="__low_conf",
            cohort_label="low-confidence",
        )
        for key, high_result in high_results.items():
            low_result = low_results.get(key)
            if low_result is None:
                raise ValueError(f"High-conf result {key} has no matching low-conf run.")
            position, contrast_name, gran = key
            leaf = f"{contrast_name}__{gran}__high_minus_low"
            if linguistic:
                exp_dir = run_root / leaf
                loc = leaf
            else:
                exp_dir = run_root / position / leaf
                loc = f"{position}/{leaf}"
            logging.info("Writing high-minus-low ranking %s", loc)
            write_high_minus_low_from_results(
                exp_dir,
                high_result,
                low_result,
                n_layers=adapter.n_layers,
                n_heads=adapter.n_query_heads,
                bar_chart_top_k=args.bar_chart_top_k,
                rounding_dp=ANNOTATION_ROUNDING_DP,
                method_label="grad×act",
            )

    if linguistic:
        logging.info(
            "L_vs_Un skipped %d examples whose span-index-6 token was not L or Un.",
            len(l_vs_un_skipped_ids),
        )

    capture.remove()
    finished_at = datetime.now().isoformat(timespec="seconds")
    actual_n: dict[str, int] = {
        "actual_n_examples_all": len(cohorts.all),
        "actual_n_examples_high": len(cohorts.high) if args.confidence_split else 0,
        "actual_n_examples_low": len(cohorts.low) if args.confidence_split else 0,
    }
    if fill_post_period_excl_one:
        actual_n.update(
            {
                "actual_n_examples_all_post_period": len(cohorts.all_excl_one),
                "actual_n_examples_high_post_period": (
                    len(cohorts.high_excl_one) if args.confidence_split else 0
                ),
                "actual_n_examples_low_post_period": (
                    len(cohorts.low_excl_one) if args.confidence_split else 0
                ),
            }
        )
    write_config_txt(
        run_root / "config.txt",
        args=args,
        input_h5_resolved=input_h5,
        finished_at=finished_at,
        n_layers=adapter.n_layers,
        n_heads=adapter.n_query_heads,
        high_digits=high_digits,
        low_digits=low_digits,
        first_token_ids=first_token_ids,
        l_vs_un_token_ids=l_vs_un_token_ids,
        l_vs_un_skipped_not_l_or_un=(
            len(l_vs_un_skipped_ids) if linguistic else None
        ),
        actual_n=actual_n,
    )
    logging.info("Finished. Results at %s", run_root)


if __name__ == "__main__":
    main()
