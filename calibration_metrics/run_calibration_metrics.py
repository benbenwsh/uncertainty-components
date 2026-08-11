#!/usr/bin/env python3
"""Compute verbalised-confidence calibration metrics (ECE, MCE, Brier, AUROC).

Generates greedy (temperature=0) Guess/Probability answers on the train split,
parses answer + confidence, scores binary correctness via squad F1>=50, and
writes summary.txt + details.json under calibration_metrics/results/<N>/.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEM_UNC_ROOT = os.path.join(_REPO_ROOT, "semantic_uncertainty")
_PROCESS_GEN_ROOT = os.path.join(_REPO_ROOT, "process_generations")
for _path in (_SEM_UNC_ROOT, _PROCESS_GEN_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate import load  # noqa: E402
from process_generations_tok_bef_gen import parse_probability_from_response  # noqa: E402
from uncertainty.utils.eval_utils import auroc  # noqa: E402

CONFIDENCE_PROMPT = (
    "Provide your best guess and the probability that it is correct (0.0 to 1.0) "
    "for the following question. Give ONLY the guess and probability, no other words "
    "or explanation. For example:\n\n"
    "Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n "
    "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any "
    "extra commentary whatsoever; just the probability!>\n\n"
    "The question is: "
)

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "<end_of_turn>"]
WRITE_LOG_EVERY = 1
TEMPERATURE = 0.0

# # Gemma-3 token alternatives (fixed length; each inner list = allowed tokens at that position)
# GUESS_PREFIX_TOKENS = [
#     ["\n", "\n\n"],
#     ["Guess"],
#     [":"],
# ]
# PROBABILITY_PREFIX_TOKENS = [
#     ["\n"],
#     ["Probability", " Probability"],
#     [":"],
#     [" "],
# ]
# Mistral-7B-Instruct-v0.1 (from ans_gen/generated_answers/1_svamp_mistral)
GUESS_PREFIX_TOKENS = [
    ["\n"],
    ["\n"],
    ["Gu"],
    ["ess"],
    [":"],
]
PROBABILITY_PREFIX_TOKENS = [
    ["\n"],
    ["Pro"],
    ["b"],
    ["ability"],
    [":"],
    [""],  # space before number decodes as empty string
]
# # Qwen-2.5-32B
# GUESS_PREFIX_TOKENS = [
#     [" Guess"],
#     [":"],
# ]
# PROBABILITY_PREFIX_TOKENS = [
#     ["\n"],
#     [" Probability"],
#     [":"],
#     [" "],
# ]


def setup_logger() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_run_output_dir() -> str:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    run_idx = max((int(d) for d in existing), default=0) + 1
    run_dir = os.path.join(base_dir, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def split_dataset(dataset):
    """Return indices of answerable and unanswerable questions."""
    answerable_indices = [
        i for i, ex in enumerate(dataset) if len(ex["answers"]["text"]) > 0
    ]
    unanswerable_indices = [
        i for i, ex in enumerate(dataset) if len(ex["answers"]["text"]) == 0
    ]
    assert set(answerable_indices) | set(unanswerable_indices) == set(range(len(dataset)))
    assert set(answerable_indices) - set(unanswerable_indices) == set(answerable_indices)
    return answerable_indices, unanswerable_indices


def get_reference(example):
    if "answers" not in example:
        example = example["reference"]
    answers = example["answers"]
    answer_starts = answers.get("answer_start", [])
    return {
        "answers": {"answer_start": answer_starts, "text": answers["text"]},
        "id": example["id"],
    }


def get_squad_binary_metric():
    """Same binarised squad logic as utils.get_metric('squad') (F1 >= 50 -> 1.0)."""
    squad_metric = load("squad_v2")

    def metric(response, example, *args, **kwargs):
        del args, kwargs
        if "id" in example:
            exid = example["id"]
        elif "id" in example["reference"]:
            exid = example["reference"]["id"]
        else:
            raise ValueError("Example has no id.")

        prediction = {
            "prediction_text": response,
            "no_answer_probability": 0.0,
            "id": exid,
        }
        results = squad_metric.compute(
            predictions=[prediction],
            references=[get_reference(example)],
        )
        return 1.0 if (results["f1"] >= 50.0) else 0.0

    return metric


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


def parse_guess_and_probability_indices(decoded_tokens: list) -> tuple[int, int, int] | None:
    """
    Returns:
      - last_guess_token_index: first token index of semantic answer
      - first_prob_token_index: token index at "\\n" before "Probability:" (first occurrence)
      - end_prob_token_index: first token index of probability value
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
        or end_prob_token_index >= len(decoded_tokens)
        or last_guess_token_index >= first_prob_token_index
        or first_prob_token_index >= end_prob_token_index
    ):
        return None

    return (last_guess_token_index, first_prob_token_index, end_prob_token_index)


def parse_answer_and_confidence(
    response: str,
    decoded_tokens: list,
) -> tuple[str | None, float | None]:
    """Parse semantic answer (between Guess/Probability prefixes) and verbalised confidence."""
    confidence = parse_probability_from_response(response)
    indices = parse_guess_and_probability_indices(decoded_tokens)
    if indices is None:
        return None, confidence

    last_guess_token_index, first_prob_token_index, _ = indices
    answer = "".join(decoded_tokens[last_guess_token_index:first_prob_token_index]).strip()
    if not answer:
        return None, confidence
    return answer, confidence


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE over [0, 1]."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidences) == 0:
        return float("nan")

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if i == n_bins - 1:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += prop * abs(np.mean(correctness[in_bin]) - np.mean(confidences[in_bin]))
    return float(ece)


def maximum_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width MCE over [0, 1]."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidences) == 0:
        return float("nan")

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if i == n_bins - 1:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(in_bin):
            gap = abs(np.mean(correctness[in_bin]) - np.mean(confidences[in_bin]))
            mce = max(mce, gap)
    return float(mce)


def brier_score(confidences: np.ndarray, correctness: np.ndarray) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidences) == 0:
        return float("nan")
    return float(np.mean((confidences - correctness) ** 2))


def load_causal_lm(model_name: str, *, load_in_8bit: bool, load_in_4bit: bool):
    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot set both --load_in_8bit and --load_in_4bit.")

    hf_token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as exc:
        logging.warning(
            "Fast tokenizer load failed (%s). Retrying with use_fast=False.",
            exc,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            token=hf_token,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "token": hf_token,
    }
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    token_limit = getattr(model.config, "max_position_embeddings", None)
    if token_limit is None:
        text_config = getattr(model.config, "text_config", None)
        token_limit = (
            getattr(text_config, "max_position_embeddings", None) if text_config else None
        )
    token_limit = token_limit or 4096
    return tokenizer, model, token_limit


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text."""

    def __init__(self, stops, tokenizer, initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        generation = self.tokenizer.decode(
            input_ids[0][self.initial_length :],
            skip_special_tokens=False,
        )
        return any(stop in generation for stop in self.stops)


class CausalLMGenerator:
    """Greedy causal LM generation (temperature=0); returns text + decoded tokens."""

    def __init__(self, tokenizer, model, token_limit, max_new_tokens):
        self.tokenizer = tokenizer
        self.model = model
        self.token_limit = token_limit
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = STOP_SEQUENCES + [self.tokenizer.eos_token]
        self.device = next(model.parameters()).device

    def predict(self, input_data: str) -> tuple[str, list[str]]:
        inputs = self.tokenizer(input_data, return_tensors="pt").to(self.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        stopping_criteria = StoppingCriteriaList(
            [
                StoppingCriteriaSub(
                    stops=self.stop_sequences,
                    initial_length=len(inputs["input_ids"][0]),
                    tokenizer=self.tokenizer,
                )
            ]
        )

        # Temperature 0 <=> greedy decoding.
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
            "do_sample": False,
            "num_beams": 1,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                f"Generation exceeding token limit {len(outputs.sequences[0])} > {self.token_limit}"
            )

        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if not full_answer.startswith(input_data):
            raise ValueError("Generated output does not start with the input prompt.")

        input_data_offset = len(input_data)
        answer = full_answer[input_data_offset:]

        stop_at = len(answer)
        sliced_answer = answer
        for stop in self.stop_sequences:
            if answer.endswith(stop):
                stop_at = len(answer) - len(stop)
                sliced_answer = answer[:stop_at]
                break
        sliced_answer = sliced_answer.strip()

        token_stop_index = self.tokenizer(
            full_answer[: input_data_offset + stop_at],
            return_tensors="pt",
        )["input_ids"].shape[1]
        n_input_token = len(inputs["input_ids"][0])
        n_generated = max(token_stop_index - n_input_token, 0)

        decoded_tokens = [
            self.tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in outputs.sequences[0]
        ]
        sliced_decoded_tokens = decoded_tokens[n_input_token : n_input_token + n_generated]
        return sliced_answer, sliced_decoded_tokens


def write_summary_txt(
    path: str,
    args,
    *,
    num_total: int,
    num_parsed: int,
    num_skipped: int,
    ece: float,
    mce: float,
    brier: float,
    auroc_score: float,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Calibration Metrics Summary\n")
        f.write("===========================\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"temperature: {TEMPERATURE}\n")
        f.write(f"model_name: {args.model_name}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write("dataset_split: train\n")
        f.write(f"num_samples: {args.num_samples}\n")
        f.write(f"answerable_only: {args.answerable_only}\n")
        f.write(f"random_seed: {args.random_seed}\n")
        f.write(f"model_max_new_tokens: {args.model_max_new_tokens}\n")
        f.write(f"n_bins: {args.n_bins}\n")
        f.write(f"load_in_8bit: {args.load_in_8bit}\n")
        f.write(f"load_in_4bit: {args.load_in_4bit}\n")
        f.write(f"num_total: {num_total}\n")
        f.write(f"num_parsed: {num_parsed}\n")
        f.write(f"num_skipped: {num_skipped}\n")
        f.write("\nMetrics\n")
        f.write("-------\n")
        f.write(f"ECE: {ece}\n")
        f.write(f"MCE: {mce}\n")
        f.write(f"Brier: {brier}\n")
        f.write(f"AUROC: {auroc_score}\n")
    logging.info("Wrote summary to %s", path)


def main(args):
    from uncertainty.data.data_utils import load_ds

    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name())
    else:
        logging.warning("CUDA is not available; generation will be slow or fail.")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    run_output_dir = make_run_output_dir()
    output_log_path = os.path.join(run_output_dir, "output.log")
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    logging.info("Saving outputs to %s", run_output_dir)

    train_dataset, _validation_dataset = load_ds(args.dataset, seed=args.random_seed)
    answerable_indices, unanswerable_indices = split_dataset(train_dataset)

    if args.answerable_only:
        possible_indices = list(answerable_indices)
    else:
        possible_indices = list(set(answerable_indices) | set(unanswerable_indices))

    indices = possible_indices[: min(args.num_samples, len(possible_indices))]
    logging.info("Train dataset size: %d", len(train_dataset))
    logging.info("Samples to evaluate: %d", len(indices))
    if args.num_samples > len(possible_indices):
        logging.warning(
            "Requested %d samples but only %d available; using all.",
            args.num_samples,
            len(possible_indices),
        )

    tokenizer, hf_model, token_limit = load_causal_lm(
        args.model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    model = CausalLMGenerator(
        tokenizer,
        hf_model,
        token_limit,
        args.model_max_new_tokens,
    )
    binary_metric = get_squad_binary_metric()

    details = []
    confidences = []
    correctness_list = []
    num_skipped = 0

    for it, index in enumerate(tqdm(indices, desc="train"), start=1):
        if it % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        example = train_dataset[index]
        question = example["question"]
        context = example.get("context", "")
        gold_label = list(example["answers"]["text"])
        if args.dataset == "svamp" and context:
            question = context + " " + question
        local_prompt = CONFIDENCE_PROMPT + question

        try:
            predicted_answer, decoded_tokens = model.predict(local_prompt)
        except Exception as exc:
            logging.error("Generation failed for example %s: %s", example.get("id"), exc)
            num_skipped += 1
            details.append(
                {
                    "id": example.get("id"),
                    "question": question,
                    "gold_label": gold_label,
                    "parsed_answer": None,
                    "verbalised_confidence": None,
                    "correctness": None,
                    "skipped": True,
                    "skip_reason": f"generation_error: {exc}",
                }
            )
            continue

        parsed_answer, verbalised_confidence = parse_answer_and_confidence(
            predicted_answer, decoded_tokens
        )

        if (it % WRITE_LOG_EVERY) == 0 or it == 1:
            logging.info("Iteration %d", it)
            logging.info("question: %s", question)
            logging.info("prediction: %s", predicted_answer)
            logging.info("parsed_answer: %s", parsed_answer)
            logging.info("verbalised_confidence: %s", verbalised_confidence)
            logging.info("gold_label: %s", gold_label)
            logging.info("decoded_tokens: %s", decoded_tokens)

        if parsed_answer is None or verbalised_confidence is None:
            reason = []
            if parsed_answer is None:
                reason.append("answer_parse_failed")
            if verbalised_confidence is None:
                reason.append("confidence_parse_failed")
            logging.warning(
                "Skipping example %s: %s. response=%r",
                example.get("id"),
                ",".join(reason),
                predicted_answer,
            )
            num_skipped += 1
            details.append(
                {
                    "id": example.get("id"),
                    "question": question,
                    "gold_label": gold_label,
                    "parsed_answer": parsed_answer,
                    "verbalised_confidence": verbalised_confidence,
                    "correctness": None,
                    "skipped": True,
                    "skip_reason": ",".join(reason),
                }
            )
            continue

        correctness = float(binary_metric(parsed_answer, example))
        if (it % WRITE_LOG_EVERY) == 0 or it == 1:
            logging.info("correctness: %s", correctness)

        confidences.append(float(verbalised_confidence))
        correctness_list.append(correctness)
        details.append(
            {
                "id": example.get("id"),
                "question": question,
                "gold_label": gold_label,
                "parsed_answer": parsed_answer,
                "verbalised_confidence": float(verbalised_confidence),
                "correctness": correctness,
            }
        )

    confidences_arr = np.asarray(confidences, dtype=np.float64)
    correctness_arr = np.asarray(correctness_list, dtype=np.float64)
    num_parsed = len(confidences_arr)
    num_total = len(indices)

    ece = expected_calibration_error(confidences_arr, correctness_arr, n_bins=args.n_bins)
    mce = maximum_calibration_error(confidences_arr, correctness_arr, n_bins=args.n_bins)
    brier = brier_score(confidences_arr, correctness_arr)
    if num_parsed >= 2 and len(np.unique(correctness_arr)) > 1:
        auroc_score = float(auroc(correctness_arr, confidences_arr))
    else:
        auroc_score = float("nan")
        logging.warning(
            "AUROC undefined with num_parsed=%d unique_labels=%s; writing nan.",
            num_parsed,
            np.unique(correctness_arr).tolist() if num_parsed else [],
        )

    logging.info(
        "Metrics — ECE=%.6f MCE=%.6f Brier=%.6f AUROC=%.6f (parsed=%d skipped=%d)",
        ece,
        mce,
        brier,
        auroc_score,
        num_parsed,
        num_skipped,
    )

    summary_path = os.path.join(run_output_dir, "summary.txt")
    write_summary_txt(
        summary_path,
        args,
        num_total=num_total,
        num_parsed=num_parsed,
        num_skipped=num_skipped,
        ece=ece,
        mce=mce,
        brier=brier,
        auroc_score=auroc_score,
    )

    details_path = os.path.join(run_output_dir, "details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logging.info("Wrote details to %s", details_path)
    logging.info("Run complete. Output directory: %s", run_output_dir)

    del model
    del hf_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute verbalised-confidence calibration metrics on the train split."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Full HuggingFace repo ID.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["trivia_qa", "squad", "bioasq", "nq", "svamp", "gsm8k", "math"],
    )
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument(
        "--answerable_only",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--model_max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of equal-width bins for ECE/MCE.",
    )
    parser.add_argument(
        "--load_in_8bit",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--load_in_4bit",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    return parser


if __name__ == "__main__":
    setup_logger()
    parser = build_parser()
    cli_args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown args: {unknown}")
    logging.info("Starting calibration metrics with args: %s", cli_args)
    main(cli_args)
    logging.info("Finished calibration metrics.")
