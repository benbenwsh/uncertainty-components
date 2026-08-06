#!/usr/bin/env python3
"""Logit-lens top-k tables for probability-span mass-mean directions.

Recomputes mass-mean directions from an H5 file, projects each (layer, token)
direction through the model's final RMSNorm + lm_head, and writes one unshaded
top-k table PNG per probability-span token position.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layerwise_mean_ablation.run_mean_ablation import load_examples_h5
from mass_mean_probe.run_mass_mean_probe import (
    _as_layer_hidden,
    _expected_probability_span_token_budget,
    _extract_res_field,
    _prefix_tokens_for_linguistic_confidence,
    compute_low_high_span_means_and_directions,
    configure_prefix_tokens_for_model,
)
from logit_lens_improved.run_logit_lens_improved import (
    SUPPORTED_MODEL_NAMES,
    _compute_topk,
    _load_unembedding,
    _resolve_device,
    _save_topk_table_png,
)


MODULE_NAME = "direction_analysis"
RESULTS_STEM = "mass_mean_direction_logit_lens"


def _resolve_run_root(cli_output_dir: Optional[str]) -> str:
    if cli_output_dir:
        os.makedirs(cli_output_dir, exist_ok=True)
        return cli_output_dir

    base_dir = os.path.join(MODULE_NAME, "results", RESULTS_STEM)
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_root = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_root, exist_ok=True)
    return run_root


def _attach_output_log(run_root: str) -> str:
    output_log_path = os.path.join(run_root, "output.log")
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    return output_log_path


def _infer_n_layers(examples_h5: Dict[str, dict], *, new_h5_format: bool) -> int:
    for ex_id, ex_obj in examples_h5.items():
        responses = ex_obj.get("responses")
        if not isinstance(responses, list) or not responses:
            continue
        resp0 = responses[0]
        if not isinstance(resp0, dict):
            continue
        emb_prob = _extract_res_field(
            resp0, ex_id, "embeddings_probability", new_h5_format=new_h5_format
        )
        if emb_prob is None:
            continue
        if isinstance(emb_prob, list):
            if not emb_prob:
                continue
            layer_hidden = _as_layer_hidden(emb_prob[0])
        else:
            layer_hidden = _as_layer_hidden(emb_prob)
        n_stream = int(layer_hidden.shape[0])
        if n_stream < 2:
            raise ValueError(
                f"Example {ex_id} residual stream length {n_stream} is too short "
                "(need embedding + at least one resid_post)."
            )
        return n_stream - 1
    raise ValueError("Could not infer n_layers: no usable embeddings_probability found in H5.")


def _token_labels_from_prefix(
    *,
    span_token_count: int,
    linguistic_confidence_prompt: bool,
) -> List[str]:
    prefix_alts = _prefix_tokens_for_linguistic_confidence(linguistic_confidence_prompt)
    labels: List[str] = []
    for i in range(span_token_count):
        if i < len(prefix_alts) and prefix_alts[i]:
            labels.append(str(prefix_alts[i][0]))
        else:
            labels.append(f"value_{i - len(prefix_alts)}")
    return labels


def write_config_txt(
    path: str,
    *,
    args: argparse.Namespace,
    n_layers: int,
    span_token_count: int,
    direction_probability_shape: Tuple[int, ...],
    low_conf_count: int,
    high_conf_count: int,
    h5_example_count: int,
    vocab_size: int,
    softcap: Optional[float],
    table_paths: Sequence[str],
    finished_at: str,
) -> None:
    lines = [
        "Mass-Mean Direction Logit Lens Config",
        "=====================================",
        "",
        "[Run]",
        f"finished_at={finished_at}",
        f"model_name={args.model_name}",
        f"input_h5={args.input_h5}",
        f"new_h5_format={args.new_h5_format}",
        f"device={args.device if args.device else 'auto'}",
        f"top_k={args.top_k}",
        f"softcap={softcap}",
        f"n_layers={n_layers}",
        f"direction_probability_shape={direction_probability_shape}",
        f"vocab_size={vocab_size}",
        "",
        "[Confidence groups]",
        f"low_conf_threshold={args.low_conf_threshold}",
        f"high_conf_threshold={args.high_conf_threshold}",
        f"low_conf_count={low_conf_count}",
        f"high_conf_count={high_conf_count}",
        f"h5_example_count={h5_example_count}",
        "",
        "[Span]",
        f"linguistic_confidence_prompt={args.linguistic_confidence_prompt}",
        f"expected_probability_tokens={args.expected_probability_tokens}",
        f"expected_confidence_tokens={args.expected_confidence_tokens}",
        f"extend_probability_span={args.extend_probability_span}",
        f"span_token_count={span_token_count}",
        f"expected_guess_tokens={args.expected_guess_tokens}",
        "",
        "[Tables]",
    ]
    for p in table_paths:
        lines.append(f"table={p}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Logit-lens top-k tables for probability-span mass-mean directions "
            "(one table per token position)."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        choices=list(SUPPORTED_MODEL_NAMES),
        help="Supported: Mistral-7B-Instruct-v0.1, gemma-3-12b-it, Qwen2.5-32B-Instruct.",
    )
    parser.add_argument("--input_h5", type=str, required=True, help="Path to *_verbalised_embeddings.h5 file.")
    parser.add_argument(
        "--new_h5_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, read 'res' from the new {res,attn,mlp} H5 response format.",
    )
    parser.add_argument("--low_conf_threshold", type=float, default=0.1)
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--expected_probability_tokens", type=int, default=7)
    parser.add_argument(
        "--expected_confidence_tokens",
        type=int,
        default=5,
        help=(
            "When --linguistic_confidence_prompt, expected Confidence: span token count "
            "(instead of --expected_probability_tokens)."
        ),
    )
    parser.add_argument("--expected_guess_tokens", type=int, default=5)
    parser.add_argument(
        "--extend_probability_span",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true (and not using linguistic confidence), treat probability span length as "
            "expected_probability_tokens + 2."
        ),
    )
    parser.add_argument(
        "--linguistic_confidence_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use Confidence: span budget; if false, use Probability: span budget.",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            f"Optional run directory. If unset, auto-creates under "
            f"{MODULE_NAME}/results/{RESULTS_STEM}/."
        ),
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be >= 1")
    configure_prefix_tokens_for_model(args.model_name)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_root = _resolve_run_root(args.output_dir)
    _attach_output_log(run_root)
    logging.info("Saving outputs to %s", run_root)

    span_token_count = _expected_probability_span_token_budget(
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
        expected_probability_tokens=args.expected_probability_tokens,
        expected_confidence_tokens=args.expected_confidence_tokens,
        extend_probability_span=args.extend_probability_span,
    )
    direction_prob_token_budget = (
        args.expected_probability_tokens
        if args.linguistic_confidence_prompt
        else span_token_count
    )

    logging.info("Loading H5: %s", args.input_h5)
    examples_h5 = load_examples_h5(Path(args.input_h5))
    n_layers = _infer_n_layers(examples_h5, new_h5_format=args.new_h5_format)
    run_layers = list(range(n_layers))
    logging.info("Inferred n_layers=%d from residual stream", n_layers)

    _mean_low, _mean_high, direction, low_ids, high_ids = compute_low_high_span_means_and_directions(
        examples_h5,
        run_layers,
        low_conf_threshold=args.low_conf_threshold,
        high_conf_threshold=args.high_conf_threshold,
        expected_probability_tokens=direction_prob_token_budget,
        expected_guess_tokens=args.expected_guess_tokens,
        new_h5_format=args.new_h5_format,
    )
    direction_probability = np.asarray(direction["probability"], dtype=np.float32)
    if direction_probability.ndim != 3:
        raise ValueError(
            f"Expected direction['probability'] shape (layers, tokens, d_model), "
            f"got {direction_probability.shape}."
        )
    if direction_probability.shape[0] != n_layers:
        raise ValueError(
            f"Direction layer count {direction_probability.shape[0]} != inferred n_layers {n_layers}."
        )
    if direction_probability.shape[1] != direction_prob_token_budget:
        raise ValueError(
            f"Direction token count {direction_probability.shape[1]} != "
            f"budget {direction_prob_token_budget}."
        )

    n_tokens = int(direction_probability.shape[1])
    token_labels = _token_labels_from_prefix(
        span_token_count=n_tokens,
        linguistic_confidence_prompt=args.linguistic_confidence_prompt,
    )

    device = _resolve_device(args.device)
    logging.info("Loading logit lens for %s on %s", args.model_name, device)
    tokenizer, w_u, b_u, norm_weight, norm_eps, softcap = _load_unembedding(
        args.model_name, device
    )
    vocab_size = int(w_u.shape[0])
    if int(w_u.shape[1]) != int(direction_probability.shape[2]):
        raise ValueError(
            f"Hidden dim mismatch: direction d_model={direction_probability.shape[2]} "
            f"vs lm_head in_features={w_u.shape[1]}."
        )

    tables_dir = Path(run_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_paths: List[str] = []
    logging.info("Writing %d unshaded top-k tables (top_k=%d)", n_tokens, args.top_k)
    for tok_idx in range(n_tokens):
        hidden = direction_probability[:, tok_idx, :]
        top_ids, top_vals = _compute_topk(
            hidden,
            top_k=args.top_k,
            device=device,
            w_u=w_u,
            b_u=b_u,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            softcap=softcap,
        )
        out_path = tables_dir / f"probability_token_{tok_idx}.png"
        _save_topk_table_png(
            out_path,
            tokenizer=tokenizer,
            top_ids=top_ids,
            top_vals=top_vals,
            shade_body=False,
        )
        table_paths.append(str(out_path))
        logging.info(
            "Wrote %s (token_label=%r)",
            out_path,
            token_labels[tok_idx] if tok_idx < len(token_labels) else f"pos_{tok_idx}",
        )

    finished_at = datetime.now().isoformat(timespec="seconds")
    write_config_txt(
        os.path.join(run_root, "config.txt"),
        args=args,
        n_layers=n_layers,
        span_token_count=n_tokens,
        direction_probability_shape=tuple(direction_probability.shape),
        low_conf_count=len(low_ids),
        high_conf_count=len(high_ids),
        h5_example_count=len(examples_h5),
        vocab_size=vocab_size,
        softcap=softcap,
        table_paths=table_paths,
        finished_at=finished_at,
    )
    summary = {
        "finished_at": finished_at,
        "input_h5": args.input_h5,
        "model_name": args.model_name,
        "new_h5_format": args.new_h5_format,
        "device": str(device),
        "top_k": args.top_k,
        "softcap": softcap,
        "n_layers": n_layers,
        "span_token_count": n_tokens,
        "direction_probability_shape": list(direction_probability.shape),
        "vocab_size": vocab_size,
        "low_conf_count": len(low_ids),
        "high_conf_count": len(high_ids),
        "h5_example_count": len(examples_h5),
        "token_labels": token_labels,
        "tables": table_paths,
    }
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logging.info("Done. Outputs in %s", run_root)


if __name__ == "__main__":
    main()
