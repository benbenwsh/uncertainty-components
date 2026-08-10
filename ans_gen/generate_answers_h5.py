"""Generate verbalised-confidence answers with embeddings, streamed to HDF5."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import time
from urllib.parse import quote

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

_SEM_UNC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "semantic_uncertainty")
if _SEM_UNC_ROOT not in sys.path:
    sys.path.insert(0, _SEM_UNC_ROOT)

CONFIDENCE_PROMPT = "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: "

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n"]
WRITE_LOG_EVERY = 1
TRAIN_RATIO = 0.9


def setup_logger() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def _tensor_to_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if torch.is_tensor(obj):
        tensor = obj.detach().cpu()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        return tensor.numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _write_ndarray_dataset(group: h5py.Group, name: str, arr: np.ndarray) -> None:
    if arr.ndim == 0:
        group.create_dataset(name, data=arr)
        return

    init_shape = (0,) + tuple(arr.shape[1:])
    max_shape = (None,) + tuple(arr.shape[1:])
    ds = group.create_dataset(
        name,
        shape=init_shape,
        maxshape=max_shape,
        dtype=arr.dtype,
        chunks=True,
    )
    ds.resize(arr.shape)
    ds[...] = arr


def _write_h5_node(group: h5py.Group, name: str, obj) -> None:
    if obj is None:
        none_group = group.create_group(name)
        none_group.attrs["__type__"] = "none"
        return

    if isinstance(obj, dict):
        sub = group.create_group(name)
        sub.attrs["__type__"] = "dict"
        for key, value in obj.items():
            _write_h5_node(sub, str(key), value)
        return

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            sub = group.create_group(name)
            sub.attrs["__type__"] = "tuple" if isinstance(obj, tuple) else "list"
            sub.attrs["__len__"] = 0
            return
        if all(isinstance(x, str) for x in obj):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.asarray(list(obj), dtype=dt))
            return
        sub = group.create_group(name)
        sub.attrs["__type__"] = "tuple" if isinstance(obj, tuple) else "list"
        sub.attrs["__len__"] = len(obj)
        for i, item in enumerate(obj):
            _write_h5_node(sub, str(i), item)
        return

    if isinstance(obj, str):
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=np.asarray(obj, dtype=dt))
        return

    if isinstance(obj, (bool, int, float, np.bool_, np.integer, np.floating)):
        group.create_dataset(name, data=obj)
        return

    arr = _tensor_to_numpy(obj)
    if arr.dtype == np.dtype("O"):
        _write_h5_node(group, name, arr.tolist())
        return
    _write_ndarray_dataset(group, name, arr)


def write_example_h5(examples_group: h5py.Group, example_id: str, example_payload: dict) -> None:
    logging.info("Writing example %s", example_id)
    if example_id in examples_group:
        del examples_group[example_id]
    _write_h5_node(examples_group, example_id, example_payload)


def encode_example_id(example_id) -> str:
    return quote(str(example_id), safe="")


def save_object_h5(path, key, obj):
    with h5py.File(path, "w") as h5_file:
        _write_h5_node(h5_file, key, obj)


def write_config_txt(run_output_dir: str, args) -> None:
    config_path = os.path.join(run_output_dir, "config.txt")
    args_dict = vars(args)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Experiment Configuration\n")
        f.write("========================\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nAll arguments:\n")
        for key in sorted(args_dict):
            f.write(f"{key}: {args_dict[key]}\n")
    logging.info("Wrote config file to %s", config_path)


def ensure_config_vocab_size(model, tokenizer=None) -> None:
    """Ensure ``model.config.vocab_size`` exists (needed by ``compute_transition_scores``).

    Multimodal configs such as ``Gemma3Config`` nest vocab size under ``text_config``.
    """
    config = model.config
    if getattr(config, "vocab_size", None) is not None:
        return

    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "vocab_size", None) is not None:
        config.vocab_size = int(text_config.vocab_size)
        logging.info(
            "Set model.config.vocab_size=%s from text_config (required for transition scores).",
            config.vocab_size,
        )
        return

    if tokenizer is not None and getattr(tokenizer, "vocab_size", None) is not None:
        config.vocab_size = int(tokenizer.vocab_size)
        logging.info(
            "Set model.config.vocab_size=%s from tokenizer (required for transition scores).",
            config.vocab_size,
        )
        return

    raise AttributeError(
        f"{type(config).__name__} has no vocab_size and none could be inferred "
        "from text_config or tokenizer."
    )


def get_transformer_layers(model):
    """Locate transformer block list across common causal-LM architectures."""
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),  # Gemma3ForConditionalGeneration
        ("language_model", "layers"),
        ("model", "h"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return obj
    return None


def load_causal_lm(
    model_name: str,
    *,
    load_in_8bit: bool,
    load_in_4bit: bool,
):
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
    ensure_config_vocab_size(model, tokenizer)

    token_limit = getattr(model.config, "max_position_embeddings", None)
    if token_limit is None:
        text_config = getattr(model.config, "text_config", None)
        token_limit = getattr(text_config, "max_position_embeddings", None) if text_config else None
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


class CausalLMWithEmbeddings:
    """Greedy causal LM generation with optional per-layer embedding hooks."""

    def __init__(self, model_name, tokenizer, model, token_limit, max_new_tokens):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.token_limit = token_limit
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = STOP_SEQUENCES + [self.tokenizer.eos_token]
        self.device = next(model.parameters()).device

    def predict(
        self,
        input_data,
        *,
        collect_attn_block_embeddings=False,
        collect_mlp_block_embeddings=False,
        collect_qkvo_embeddings=False,
        collect_concat_embeddings=True,
    ):
        if isinstance(input_data, tuple):
            input_data = input_data[0]

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

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_hidden_states": True,
            "do_sample": False,
            "num_beams": 1,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": pad_token_id,
        }

        attn_layer_outputs = []
        mlp_layer_outputs = []
        q_layer_outputs = []
        k_layer_outputs = []
        v_layer_outputs = []
        o_layer_outputs = []
        concat_layer_inputs = []
        hook_handles = []

        collect_attn = collect_attn_block_embeddings
        collect_mlp = collect_mlp_block_embeddings
        collect_qkvo = collect_qkvo_embeddings
        collect_concat = collect_concat_embeddings

        if collect_attn or collect_mlp or collect_qkvo or collect_concat:
            layers = get_transformer_layers(self.model)
            if layers is None:
                logging.warning(
                    "Could not locate transformer layers; skipping attn/mlp/qkvo/concat collection."
                )
                collect_attn = False
                collect_mlp = False
                collect_qkvo = False
                collect_concat = False
            else:
                if collect_attn:
                    attn_layer_outputs = [[] for _ in range(len(layers))]
                if collect_mlp:
                    mlp_layer_outputs = [[] for _ in range(len(layers))]
                if collect_qkvo:
                    q_layer_outputs = [[] for _ in range(len(layers))]
                    k_layer_outputs = [[] for _ in range(len(layers))]
                    v_layer_outputs = [[] for _ in range(len(layers))]
                    o_layer_outputs = [[] for _ in range(len(layers))]
                if collect_concat:
                    concat_layer_inputs = [[] for _ in range(len(layers))]

                def _extract_hidden_tensor(module_output):
                    if isinstance(module_output, (tuple, list)):
                        if len(module_output) == 0:
                            return None
                        return module_output[0]
                    return module_output

                def _make_attn_hook(layer_idx):
                    def _hook(_, __, module_output):
                        tensor = _extract_hidden_tensor(module_output)
                        if tensor is not None:
                            attn_layer_outputs[layer_idx].append(tensor.detach())

                    return _hook

                def _make_mlp_hook(layer_idx):
                    def _hook(_, __, module_output):
                        tensor = _extract_hidden_tensor(module_output)
                        if tensor is not None:
                            mlp_layer_outputs[layer_idx].append(tensor.detach())

                    return _hook

                def _make_qkvo_hook(layer_outputs, layer_idx):
                    def _hook(_, __, module_output):
                        tensor = _extract_hidden_tensor(module_output)
                        if tensor is not None:
                            layer_outputs[layer_idx].append(tensor.detach())

                    return _hook

                def _extract_pre_hook_input(module_input):
                    if not isinstance(module_input, (tuple, list)) or len(module_input) == 0:
                        return None
                    tensor = module_input[0]
                    if isinstance(tensor, (tuple, list)):
                        if len(tensor) == 0:
                            return None
                        tensor = tensor[0]
                    return tensor

                def _make_concat_pre_hook(layer_idx):
                    def _hook(_, module_input):
                        tensor = _extract_pre_hook_input(module_input)
                        if tensor is not None:
                            concat_layer_inputs[layer_idx].append(tensor.detach())

                    return _hook

                for layer_idx, layer in enumerate(layers):
                    if collect_attn and hasattr(layer, "self_attn"):
                        hook_handles.append(
                            layer.self_attn.register_forward_hook(_make_attn_hook(layer_idx))
                        )
                    if collect_mlp and hasattr(layer, "mlp"):
                        hook_handles.append(
                            layer.mlp.register_forward_hook(_make_mlp_hook(layer_idx))
                        )
                    if collect_qkvo and hasattr(layer, "self_attn"):
                        self_attn = layer.self_attn
                        if hasattr(self_attn, "q_proj"):
                            hook_handles.append(
                                self_attn.q_proj.register_forward_hook(
                                    _make_qkvo_hook(q_layer_outputs, layer_idx)
                                )
                            )
                        if hasattr(self_attn, "k_proj"):
                            hook_handles.append(
                                self_attn.k_proj.register_forward_hook(
                                    _make_qkvo_hook(k_layer_outputs, layer_idx)
                                )
                            )
                        if hasattr(self_attn, "v_proj"):
                            hook_handles.append(
                                self_attn.v_proj.register_forward_hook(
                                    _make_qkvo_hook(v_layer_outputs, layer_idx)
                                )
                            )
                        if hasattr(self_attn, "o_proj"):
                            hook_handles.append(
                                self_attn.o_proj.register_forward_hook(
                                    _make_qkvo_hook(o_layer_outputs, layer_idx)
                                )
                            )
                    if collect_concat and hasattr(layer, "self_attn"):
                        self_attn = layer.self_attn
                        if hasattr(self_attn, "o_proj"):
                            hook_handles.append(
                                self_attn.o_proj.register_forward_pre_hook(
                                    _make_concat_pre_hook(layer_idx)
                                )
                            )

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
        finally:
            for handle in hook_handles:
                handle.remove()

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
        n_generated = token_stop_index - n_input_token

        decoded_tokens = [
            self.tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in outputs.sequences[0]
        ]
        logging.info("decoded tokens before slicing:%s", decoded_tokens)
        sliced_decoded_tokens = decoded_tokens[n_input_token:]

        if n_generated == 0:
            logging.warning(
                "Only stop words were generated; using one token for likelihoods/embeddings."
            )
            n_generated = 1

        hidden = outputs.decoder_hidden_states if "decoder_hidden_states" in outputs else outputs.hidden_states

        if len(hidden) == 1:
            sec_last_input = hidden[0]
        elif (n_generated - 2) >= len(hidden):
            sec_last_input = hidden[-2]
        else:
            sec_last_input = hidden[n_generated - 1]

        all_embeddings = []
        for h in hidden:
            if isinstance(h, (list, tuple)):
                stacked = torch.stack([layer for layer in h])
            else:
                stacked = h
            all_embeddings.append(stacked.cpu())

        emb_sec_last_token = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
        last_tok_bef_gen_input = hidden[0]
        emb_tok_bef_gen = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        def _pack_layerwise_outputs(layer_outputs, name):
            if not layer_outputs:
                return None
            if any(len(per_layer) == 0 for per_layer in layer_outputs):
                logging.warning(
                    "No %s outputs captured for at least one layer; skipping %s collection.",
                    name,
                    name,
                )
                return None

            min_calls = min(len(per_layer) for per_layer in layer_outputs)
            expected_calls = len(hidden)
            if min_calls < expected_calls:
                logging.warning(
                    "Captured %d %s calls but hidden has %d steps. Using %d captured steps.",
                    min_calls,
                    name,
                    expected_calls,
                    min_calls,
                )
                use_calls = min_calls
                start_idx = 0
            else:
                use_calls = expected_calls
                start_idx = min_calls - expected_calls

            packed = []
            for call_idx in range(start_idx, start_idx + use_calls):
                packed.append(
                    torch.stack([per_layer[call_idx].cpu() for per_layer in layer_outputs])
                )
            return packed

        all_attn_embeddings = (
            _pack_layerwise_outputs(attn_layer_outputs, "attention") if collect_attn else None
        )
        all_mlp_embeddings = (
            _pack_layerwise_outputs(mlp_layer_outputs, "mlp") if collect_mlp else None
        )
        all_q_embeddings = (
            _pack_layerwise_outputs(q_layer_outputs, "q_proj") if collect_qkvo else None
        )
        all_k_embeddings = (
            _pack_layerwise_outputs(k_layer_outputs, "k_proj") if collect_qkvo else None
        )
        all_v_embeddings = (
            _pack_layerwise_outputs(v_layer_outputs, "v_proj") if collect_qkvo else None
        )
        all_o_embeddings = (
            _pack_layerwise_outputs(o_layer_outputs, "o_proj") if collect_qkvo else None
        )
        all_concat_embeddings = (
            _pack_layerwise_outputs(concat_layer_inputs, "concat") if collect_concat else None
        )

        # Gemma3Config (and similar multimodal configs) may lack top-level vocab_size.
        ensure_config_vocab_size(self.model, self.tokenizer)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]
        log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == 0:
            raise ValueError("No log-likelihoods computed for generation.")

        hidden_states = (
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
        )
        return sliced_answer, log_likelihoods, hidden_states, sliced_decoded_tokens


def make_run_output_dir() -> str:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_answers")
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


def main(args):
    from uncertainty.data.data_utils import load_ds

    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name())
    else:
        logging.warning("CUDA is not available; generation will be slow or fail.")

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

    for dataset_split in ["train", "validation"]:
        logging.info("Starting dataset split: %s", dataset_split)

        if dataset_split == "train":
            dataset = train_dataset
            possible_indices = list(set(answerable_indices) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(len(dataset))

        num_samples = (
            round(args.num_samples * TRAIN_RATIO)
            if dataset_split == "train"
            else round(args.num_samples * (1 - TRAIN_RATIO))
        )
        indices = possible_indices[: min(num_samples, len(dataset))]
        experiment_details[dataset_split] = {"indices": list(indices)}

        logging.info("Dataset size: %d", len(dataset))
        logging.info("Samples to generate: %d", len(indices))
        if num_samples > len(dataset):
            logging.warning("Not enough samples; using all %d.", len(dataset))

        generations_filepath = os.path.join(run_output_dir, f"{dataset_split}_generations.h5")
        it = 0

        with h5py.File(generations_filepath, "w") as h5_file:
            h5_file.attrs["format"] = "native_examples_v1"
            examples_group = h5_file.require_group("examples")

            for index in tqdm(indices, desc=dataset_split):
                if (it + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                it += 1

                example = dataset[index]
                question = example["question"]
                context = example.get("context", "")
                correct_answer = example["answers"]["text"]
                if args.dataset == "svamp" and context:
                    question = context + " " + question
                local_prompt = CONFIDENCE_PROMPT + question

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

                if (it % WRITE_LOG_EVERY) == 0 or it == 1:
                    logging.info("Iteration %d", it)
                    logging.info("question: %s", question)
                    logging.info("prediction:%s", predicted_answer)
                    logging.info("correct answer:%s", correct_answer)
                    logging.info("decoded tokens:%s", decoded_tokens)

                most_likely_answer_dict = {
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
                }

                example_payload = {
                    "question": question,
                    "context": context,
                    "most_likely_answer": most_likely_answer_dict,
                    "reference": get_reference(example),
                    "responses": [],
                }

                t_write_0 = time.perf_counter()
                encoded_example_id = encode_example_id(example["id"])
                write_example_h5(examples_group, encoded_example_id, example_payload)
                if (it % WRITE_LOG_EVERY) == 0:
                    logging.info(
                        "Wrote %d examples to %s (last write %.3fs)",
                        it,
                        os.path.basename(generations_filepath),
                        time.perf_counter() - t_write_0,
                    )

    experiment_details_path = os.path.join(run_output_dir, "experiment_details.h5")
    save_object_h5(experiment_details_path, "experiment_details", experiment_details)
    logging.info("Run complete. Output directory: %s", run_output_dir)
    del model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate verbalised-confidence answers with embeddings to HDF5."
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
    logging.info("Starting ans_gen with args: %s", cli_args)
    main(cli_args)
    logging.info("Finished ans_gen.")
