"""Generate answers and immediately process into verbalised-confidence HDF5.

Combines generation from ``generate_answers_h5.py`` with per-example parsing from
``process_generations_more_embs_from_h5.py``. Successfully parsed examples are
written in the processed schema; failed parses are discarded. Optional
confidence-group filtering keeps only low/high buckets with per-group caps.
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
from collections import defaultdict
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PROCESS_DIR = _REPO_ROOT / "process_generations"
for _path in (_REPO_ROOT, _SCRIPT_DIR, _PROCESS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ans_gen.generate_answers_h5 import (  # noqa: E402
    CONFIDENCE_PROMPT,
    CONFIDENCE_PROMPT_LINGUISTIC,
    TRAIN_RATIO,
    WRITE_LOG_EVERY,
    CausalLMWithEmbeddings,
    encode_example_id,
    get_reference,
    load_causal_lm,
    save_object_h5,
    setup_logger,
    split_dataset,
    write_example_h5,
)
from process_generations_more_embs_from_h5 import (  # noqa: E402
    _confidence_bin,
    configure_prefix_tokens_for_model,
    convert_for_json,
    process_example,
    write_sample_bin_counts_txt,
)
from plotting.confidence_bin_count_plots import (  # noqa: E402
    plot_histograms_from_sample_bin_counts_txt,
)


def make_run_output_dir() -> str:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_answers_with_processing")
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    run_id = max((int(d) for d in existing), default=0) + 1
    run_output_dir = os.path.join(base_dir, str(run_id))
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def write_config_txt(run_output_dir: str, args) -> None:
    config_path = os.path.join(run_output_dir, "config.txt")
    args_dict = vars(args)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Experiment Configuration\n")
        f.write("========================\n")
        f.write(f"script: {Path(__file__).name}\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nAll arguments:\n")
        for key in sorted(args_dict):
            f.write(f"{key}: {args_dict[key]}\n")
    logging.info("Wrote config file to %s", config_path)


def classify_confidence_group(
    conf: float,
    *,
    low_conf_threshold: float,
    high_conf_threshold: float,
) -> str:
    """Return ``low``, ``high``, or ``mid`` for a verbalised confidence value."""
    if conf <= low_conf_threshold:
        return "low"
    if conf >= high_conf_threshold:
        return "high"
    return "mid"


def main(args):
    from uncertainty.data.data_utils import load_ds

    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name())
    else:
        logging.warning("CUDA is not available; generation will be slow or fail.")

    try:
        configure_prefix_tokens_for_model(args.model_name)
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    if args.linguistic_confidence_prompt:
        from process_generations_more_embs_from_h5 import (
            CONFIDENCE_PREFIX_TOKENS,
            LINGUISTIC_PHRASE_TOKENS,
        )

        if args.extend_probability_span and not LINGUISTIC_PHRASE_TOKENS:
            logging.error(
                "--extend_probability_span with --linguistic_confidence_prompt requires "
                "a non-empty linguistic phrase token table for --model_name=%r.",
                args.model_name,
            )
            sys.exit(1)
        if not CONFIDENCE_PREFIX_TOKENS:
            logging.error(
                "--linguistic_confidence_prompt requires a non-empty Confidence: prefix table "
                "for --model_name=%r.",
                args.model_name,
            )
            sys.exit(1)

    if args.max_per_confidence_group is not None and args.max_per_confidence_group < 1:
        logging.error(
            "--max_per_confidence_group must be >= 1 or omitted, got %s",
            args.max_per_confidence_group,
        )
        sys.exit(1)

    if args.low_conf_threshold > args.high_conf_threshold:
        logging.warning(
            "low_conf_threshold (%.4f) > high_conf_threshold (%.4f); "
            "overlapping assignments prefer low.",
            args.low_conf_threshold,
            args.high_conf_threshold,
        )

    if args.max_per_confidence_group is not None and not args.confidence_groups_only:
        logging.info(
            "--max_per_confidence_group=%s is ignored because --confidence_groups_only is off.",
            args.max_per_confidence_group,
        )

    random.seed(args.random_seed)
    experiment_details = {"args": vars(args)}

    run_output_dir = make_run_output_dir()
    output_log_path = os.path.join(run_output_dir, "output.log")
    file_handler = logging.FileHandler(output_log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    logging.info("Saving outputs to %s", run_output_dir)
    write_config_txt(run_output_dir, args)

    train_dataset, validation_dataset = load_ds(args.dataset, seed=args.random_seed)
    answerable_indices, unanswerable_indices = split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, _ = split_dataset(validation_dataset)
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    tokenizer, hf_model, token_limit = load_causal_lm(
        args.model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    model = CausalLMWithEmbeddings(
        args.model_name,
        tokenizer,
        hf_model,
        token_limit,
        args.model_max_new_tokens,
    )

    dataset_splits = ["train"] if args.train_only else ["train", "validation"]

    for dataset_split in dataset_splits:
        logging.info("Starting dataset split: %s", dataset_split)

        if dataset_split == "train":
            dataset = train_dataset
            possible_indices = list(set(answerable_indices) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(len(dataset))

        if args.train_only:
            num_samples = args.num_samples
        else:
            num_samples = (
                round(args.num_samples * TRAIN_RATIO)
                if dataset_split == "train"
                else round(args.num_samples * (1 - TRAIN_RATIO))
            )
        indices = list(possible_indices[: min(num_samples, len(dataset))])
        experiment_details[dataset_split] = {"indices": list(indices)}

        logging.info("Dataset size: %d", len(dataset))
        logging.info("Samples to look at: %d", len(indices))
        if num_samples > len(dataset):
            logging.warning("Not enough samples; using all %d.", len(dataset))

        h5_path = os.path.join(run_output_dir, f"{dataset_split}_verbalised_embeddings.h5")
        json_summary_path = os.path.join(
            run_output_dir, f"{dataset_split}_verbalised_embeddings_summary.json"
        )
        sample_bin_counts_path = Path(run_output_dir) / f"{dataset_split}_sample_bin_counts.txt"

        with h5py.File(h5_path, "w") as h5_file:
            h5_file.attrs["format"] = "native_examples_v1"
            h5_file.attrs["linguistic_confidence_prompt"] = bool(
                args.linguistic_confidence_prompt
            )
            h5_file.require_group("examples")

        n_looked = 0
        n_kept = 0
        n_reject_parse = 0
        n_discard_mid = 0
        n_discard_cap = 0
        n_low = 0
        n_high = 0
        confidence_bin_counts: dict[float, int] = defaultdict(int)
        first_json_item = True
        looked_indices: list = []
        kept_example_ids: list[str] = []

        with (
            h5py.File(h5_path, "a") as out_h5,
            open(json_summary_path, "w", encoding="utf-8") as json_summary_file,
        ):
            out_examples = out_h5["examples"]
            json_summary_file.write("{\n")

            for index in tqdm(indices, desc=dataset_split):
                if (n_looked + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                n_looked += 1
                looked_indices.append(index)

                example = dataset[index]
                question = example["question"]
                context = example.get("context", "")
                correct_answer = example["answers"]["text"]
                if args.dataset == "svamp" and context:
                    question = context + " " + question
                prompt = (
                    CONFIDENCE_PROMPT_LINGUISTIC
                    if args.linguistic_confidence_prompt
                    else CONFIDENCE_PROMPT
                )
                local_prompt = prompt + question

                (
                    predicted_answer,
                    token_log_likelihoods,
                    (
                        emb_sec_last_token,
                        emb_tok_bef_gen,
                        all_embeddings,
                        all_attn_embeddings,
                        all_mlp_embeddings,
                        all_q_embeddings,
                        all_k_embeddings,
                        all_v_embeddings,
                        all_o_embeddings,
                        all_concat_embeddings,
                    ),
                    decoded_tokens,
                ) = model.predict(
                    local_prompt,
                    collect_attn_block_embeddings=args.collect_attn_block_embeddings,
                    collect_mlp_block_embeddings=args.collect_mlp_block_embeddings,
                    collect_qkvo_embeddings=args.collect_qkvo_embeddings,
                    collect_concat_embeddings=args.collect_concat_embeddings,
                )

                if (n_looked % WRITE_LOG_EVERY) == 0 or n_looked == 1:
                    logging.info("Looked-at %d", n_looked)
                    logging.info("question: %s", question)
                    logging.info("prediction:%s", predicted_answer)
                    logging.info("correct answer:%s", correct_answer)
                    logging.info("decoded tokens:%s", decoded_tokens)

                encoded_example_id = encode_example_id(example["id"])
                example_payload = {
                    "question": question,
                    "context": context,
                    "most_likely_answer": {
                        "response": predicted_answer,
                        "token_log_likelihoods": token_log_likelihoods,
                        "correct_answer": correct_answer,
                        "emb_sec_last_token": emb_sec_last_token,
                        "emb_tok_bef_gen": emb_tok_bef_gen,
                        "all_embeddings": all_embeddings,
                        "all_attn_embeddings": all_attn_embeddings,
                        "all_mlp_embeddings": all_mlp_embeddings,
                        "all_q_embeddings": all_q_embeddings,
                        "all_k_embeddings": all_k_embeddings,
                        "all_v_embeddings": all_v_embeddings,
                        "all_o_embeddings": all_o_embeddings,
                        "all_concat_embeddings": all_concat_embeddings,
                        "decoded_tokens": decoded_tokens,
                    },
                    "reference": get_reference(example),
                    "responses": [],
                }

                processed = process_example(
                    encoded_example_id,
                    example_payload,
                    expected_guess_tokens=args.expected_guess_tokens,
                    expected_probability_tokens=args.expected_probability_tokens,
                    expected_confidence_tokens=args.expected_confidence_tokens,
                    collect_attn_block_embeddings=args.collect_attn_block_embeddings,
                    collect_mlp_block_embeddings=args.collect_mlp_block_embeddings,
                    collect_qkvo_embeddings=args.collect_qkvo_embeddings,
                    collect_concat_embeddings=args.collect_concat_embeddings,
                    attention_score_tokenwise_k_mode=args.attention_score_tokenwise_k_mode,
                    extend_probability_span=args.extend_probability_span,
                    linguistic_confidence_prompt=args.linguistic_confidence_prompt,
                )
                # Drop raw payload ASAP (processed holds sliced embeddings only).
                del example_payload

                if processed is None:
                    n_reject_parse += 1
                    continue

                try:
                    conf = float(processed["responses"][0]["verbalised_confidence"])
                except (KeyError, IndexError, TypeError, ValueError):
                    logging.warning(
                        "Example %s missing verbalised_confidence after process; discarding.",
                        encoded_example_id,
                    )
                    n_reject_parse += 1
                    continue

                # Include all successfully parsed confidences in bin stats, even if
                # later discarded by confidence-group / per-group-cap filters.
                confidence_bin_counts[_confidence_bin(conf)] += 1

                group = None
                if args.confidence_groups_only:
                    group = classify_confidence_group(
                        conf,
                        low_conf_threshold=args.low_conf_threshold,
                        high_conf_threshold=args.high_conf_threshold,
                    )
                    if group == "mid":
                        n_discard_mid += 1
                        continue
                    if args.max_per_confidence_group is not None:
                        if group == "low" and n_low >= args.max_per_confidence_group:
                            n_discard_cap += 1
                            continue
                        if group == "high" and n_high >= args.max_per_confidence_group:
                            n_discard_cap += 1
                            continue

                t_write_0 = time.perf_counter()
                if encoded_example_id in out_examples:
                    del out_examples[encoded_example_id]
                write_example_h5(out_examples, encoded_example_id, processed)

                if not first_json_item:
                    json_summary_file.write(",\n")
                json_summary_file.write(f'  "{encoded_example_id}": ')
                json_summary_str = json.dumps(
                    convert_for_json(processed, shape_only=True), indent=2
                )
                json_summary_indented = "\n".join(
                    "    " + line if line.strip() else line
                    for line in json_summary_str.split("\n")
                )
                json_summary_file.write(json_summary_indented)
                first_json_item = False

                n_kept += 1
                kept_example_ids.append(encoded_example_id)
                if group == "low":
                    n_low += 1
                elif group == "high":
                    n_high += 1

                if (n_kept % WRITE_LOG_EVERY) == 0:
                    logging.info(
                        "Kept %d / looked-at %d examples (last write %.3fs)",
                        n_kept,
                        n_looked,
                        time.perf_counter() - t_write_0,
                    )

                if (
                    args.confidence_groups_only
                    and args.max_per_confidence_group is not None
                    and n_low >= args.max_per_confidence_group
                    and n_high >= args.max_per_confidence_group
                ):
                    logging.info(
                        "Both confidence groups reached max_per_confidence_group=%d "
                        "(low=%d, high=%d); stopping %s early after %d looked-at.",
                        args.max_per_confidence_group,
                        n_low,
                        n_high,
                        dataset_split,
                        n_looked,
                    )
                    break

            json_summary_file.write("\n}")

        experiment_details[dataset_split] = {
            "indices": looked_indices,
            "kept_example_ids": kept_example_ids,
            "n_looked": n_looked,
            "n_kept": n_kept,
            "n_binned": sum(confidence_bin_counts.values()),
            "n_reject_parse": n_reject_parse,
            "n_discard_mid": n_discard_mid,
            "n_discard_cap": n_discard_cap,
            "n_low": n_low,
            "n_high": n_high,
        }

        n_binned = sum(confidence_bin_counts.values())
        write_sample_bin_counts_txt(
            sample_bin_counts_path,
            total_samples=n_binned,
            bin_counts=confidence_bin_counts,
        )
        hist_path, hist_paper_path, hist_paper_log_path = (
            plot_histograms_from_sample_bin_counts_txt(sample_bin_counts_path)
        )
        logging.info(
            "Split %s done: looked-at=%d kept=%d binned=%d reject_parse=%d "
            "discard_mid=%d discard_cap=%d low=%d high=%d",
            dataset_split,
            n_looked,
            n_kept,
            n_binned,
            n_reject_parse,
            n_discard_mid,
            n_discard_cap,
            n_low,
            n_high,
        )
        logging.info("Wrote %s", h5_path)
        logging.info("Wrote %s", json_summary_path)
        logging.info("Wrote %s", sample_bin_counts_path)
        logging.info("Wrote %s", hist_path)
        logging.info("Wrote %s", hist_paper_path)
        logging.info("Wrote %s", hist_paper_log_path)

    experiment_details_path = os.path.join(run_output_dir, "experiment_details.h5")
    save_object_h5(experiment_details_path, "experiment_details", experiment_details)
    logging.info("Run complete. Output directory: %s", run_output_dir)
    del model


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate verbalised-confidence answers and immediately process them "
            "into processed verbalised-embedding HDF5."
        )
    )
    # --- generate flags ---
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of dataset examples to look at (not number kept).",
    )
    parser.add_argument(
        "--answerable_only",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--model_max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--collect_attn_block_embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--collect_mlp_block_embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--collect_qkvo_embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--collect_concat_embeddings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Collect concatenated attention-head activations immediately before o_proj/W_O "
            "as all_concat_embeddings."
        ),
    )
    parser.add_argument(
        "--linguistic_confidence_prompt",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, use the natural-language Confidence: prompt; "
            "if false (default), use the numeric Probability: prompt."
        ),
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
    # --- process flags ---
    parser.add_argument(
        "--expected_guess_tokens",
        type=int,
        default=3,
        help="Expected number of stored guess embeddings (list length).",
    )
    parser.add_argument(
        "--expected_probability_tokens",
        type=int,
        default=5,
        help="Expected number of stored probability embeddings (list length).",
    )
    parser.add_argument(
        "--expected_confidence_tokens",
        type=int,
        default=5,
        help=(
            "When --linguistic_confidence_prompt, expected number of stored embeddings "
            "in the unextended Confidence: span (prefix plus first phrase token)."
        ),
    )
    parser.add_argument(
        "--extend_probability_span",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Extend embeddings_probability past the first probability-value / first "
            "confidence-phrase token."
        ),
    )
    parser.add_argument(
        "--attention_score_tokenwise_k_mode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Also store tokenwise K prompt/semantic-answer fields for downstream attention "
            "score analysis."
        ),
    )
    # --- combined-script flags ---
    parser.add_argument(
        "--train_only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true (default), only use train examples. "
            "If false, use the usual 0.9/0.1 train/validation split of num_samples."
        ),
    )
    parser.add_argument(
        "--confidence_groups_only",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, keep only low/high confidence examples and discard the middle band. "
            "Enables --max_per_confidence_group."
        ),
    )
    parser.add_argument(
        "--low_conf_threshold",
        type=float,
        default=0.3,
        help="Examples with verbalised_confidence <= this are low confidence.",
    )
    parser.add_argument(
        "--high_conf_threshold",
        type=float,
        default=0.7,
        help="Examples with verbalised_confidence >= this are high confidence.",
    )
    parser.add_argument(
        "--max_per_confidence_group",
        type=int,
        default=None,
        help=(
            "Max kept examples per low/high confidence group. Only used when "
            "--confidence_groups_only is set. Omit or leave unset for no per-group cap. "
            "When both groups are full, generation for that split stops early."
        ),
    )
    return parser


if __name__ == "__main__":
    setup_logger()
    parser = build_parser()
    cli_args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown args: {unknown}")
    logging.info("Starting ans_gen with processing with args: %s", cli_args)
    main(cli_args)
    logging.info("Finished ans_gen with processing.")
