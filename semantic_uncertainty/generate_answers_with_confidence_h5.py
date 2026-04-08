"""Predict with LLM on task, appending a confidence request to the final question."""

import gc
import os
import logging
import random
import time
from urllib.parse import quote
from tqdm import tqdm

import h5py
import numpy as np
import torch
import openai
import wandb

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute


utils.setup_logger()
openai.api_key = os.getenv("a")  # Set up OpenAI API credentials.

# Prompt template is from just_ask_for_calibration paper
CONFIDENCE_PROMPT = "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: "

# The bigger this number is, the fewer per-example write logs.
WRITE_LOG_EVERY = 10


def _tensor_to_numpy(obj):
    """Convert tensor/array-like to numpy."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _write_ndarray_dataset(group: h5py.Group, name: str, arr: np.ndarray) -> None:
    """Write ndarray as chunked dataset; axis 0 is extendable for ndim>=1."""
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
    """Recursively write Python object to HDF5 without pickle serialization."""
    if obj is None:
        none_group = group.create_group(name)
        none_group.attrs["__type__"] = "none"
        return

    if isinstance(obj, dict):
        sub = group.create_group(name)
        sub.attrs["__type__"] = "dict"
        for k, v in obj.items():
            _write_h5_node(sub, str(k), v)
        return

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            sub = group.create_group(name)
            sub.attrs["__type__"] = "tuple" if isinstance(obj, tuple) else "list"
            sub.attrs["__len__"] = 0
            return
        # Store simple string lists compactly as one dataset.
        if all(isinstance(x, str) for x in obj):
            dt = h5py.string_dtype(encoding="utf-8")
            data = np.asarray(list(obj), dtype=dt)
            group.create_dataset(name, data=data)
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
        # Fallback for object arrays; convert to recursive list storage.
        _write_h5_node(group, name, arr.tolist())
        return
    _write_ndarray_dataset(group, name, arr)


def write_example_h5(examples_group: h5py.Group, example_id: str, example_payload: dict) -> None:
    """Write one example under examples/<example_id>."""
    logging.info("Writing example %s", example_id)
    if example_id in examples_group:
        del examples_group[example_id]
    _write_h5_node(examples_group, example_id, example_payload)


def encode_example_id(example_id) -> str:
    """URL-encode example IDs so HDF5 keys stay single-level."""
    return quote(str(example_id), safe="")


def save_object_h5(path, key, obj):
    """Save a single Python object into native HDF5."""
    with h5py.File(path, "w") as h5_file:
        _write_h5_node(h5_file, key, obj)


def main(args):
    logging.info('GPU: %s', torch.cuda.get_device_name())
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': vars(args)}
    random.seed(args.random_seed)

    # Choose output directory based on save_to_wandb
    if args.save_to_wandb:
        user = os.environ['USER']
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        output_dir = f"{scratch_dir}/{user}/uncertainty"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        entity = os.environ['WANDB_ENT']
        slurm_jobid = os.getenv('SLURM_JOB_ID', None)
        wandb.init(
            entity=entity,
            project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
            dir=output_dir,
            config=args,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
        )
        run_output_dir = wandb.run.dir
        logging.info('Finished wandb init.')
    else:
        # Save to generated_answers/n with n incrementing (1, 2, 3, ...)
        base_dir = "generated_answers"
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
        n = max((int(d) for d in existing), default=0) + 1
        run_output_dir = os.path.join(base_dir, str(n))
        os.makedirs(run_output_dir, exist_ok=True)
        # Mirror console logs to output.log in this run folder (similar to wandb's output.log)
        output_log_path = os.path.join(run_output_dir, 'output.log')
        file_handler = logging.FileHandler(output_log_path, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logging.getLogger().addHandler(file_handler)
        logging.info(f'Wandb disabled. Files will be saved to {run_output_dir}')

    metric = utils.get_metric(args.metric)

    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80 * '#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        if args.save_to_wandb:
            wandb.config.update(
                {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
            wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80 * '#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80 * '#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store aggregate metrics only; examples are streamed directly to HDF5.
        accuracies, results_dict, p_trues = [], {}, []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # 9:1 split of total num_samples between train (90%) and validation (10%).
        TRAIN_RATIO = 0.9
        num_samples = round(args.num_samples * TRAIN_RATIO) if dataset_split == 'train' else round(args.num_samples * (1 - TRAIN_RATIO))
        # I changed it to become not random
        indices = possible_indices[:min(num_samples, len(dataset))]

        logging.info('Size of dataset: %d', len(dataset))
        logging.info('Number of samples to generate: %d', len(indices))

        experiment_details[dataset_split] = {'indices': list(indices)}

        if num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        generations_filename = f'{dataset_split}_generations.h5'
        generations_filepath = os.path.join(run_output_dir, generations_filename)

        # Initialize/clear the HDF5 file and stream each example directly.
        with h5py.File(generations_filepath, "w") as h5_file:
            h5_file.attrs["format"] = "native_examples_v1"
            examples_group = h5_file.require_group("examples")

            for index in tqdm(indices):
                if ((it + 1) % 10) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                it += 1

                # Grab example at index.
                example = dataset[index]
                question, context = example["question"], example['context']
                correct_answer = example['answers']['text']

                current_input = CONFIDENCE_PROMPT + question
                local_prompt = prompt + current_input

                logging.info('Full prompt:\n' + local_prompt)

                full_responses = []

                # We sample 1 low temperature answer on which we will compute the
                # accuracy and args.num_generation high temperature answers which will
                # be used to estimate the entropy.
                if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                    num_generations = 1
                else:
                    num_generations = args.num_generations + 1

                for i in range(num_generations):
                    # Temperature for first generation is always `0.1`.
                    temperature = 0.1 if i == 0 else args.temperature

                    predicted_answer, token_log_likelihoods, (emb_sec_last_token, emb_tok_bef_gen, all_embeddings), decoded_tokens = model.predict(local_prompt, temperature, return_latent=True)

                    emb_sec_last_token = emb_sec_last_token.cpu() if emb_sec_last_token is not None else None
                    emb_tok_bef_gen = emb_tok_bef_gen.cpu() if emb_tok_bef_gen is not None else None

                    compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                    if correct_answer and compute_acc:
                        acc = metric(predicted_answer, example, model)
                    else:
                        acc = 0.0  # pylint: disable=invalid-name

                    if i == 0:
                        # Logging.
                        logging.info('Iteration ' + str(it) + ':  ' + 80 * '#')
                        if args.use_context:
                            logging.info('context: '.ljust(15) + str(context))
                        logging.info('question: '.ljust(15) + question)
                        logging.info('low-t prediction: '.ljust(15) + '\n' + predicted_answer)
                        logging.info('correct answer: '.ljust(15) + str(correct_answer))
                        logging.info('decoded tokens: '.ljust(15) + '\n' + str(decoded_tokens))

                        logging.info('emb_sec_last_token layer 0: '.ljust(15) + '\n' + str(emb_sec_last_token[0][0][:5]))
                        logging.info('emb_sec_last_token layer 32: '.ljust(15) + '\n' + str(emb_sec_last_token[-1][0][:5]))

                        accuracies.append(acc)
                        most_likely_answer_dict = {
                            'response': predicted_answer,
                            'token_log_likelihoods': token_log_likelihoods,
                            'accuracy': acc,
                            'correct_answer': correct_answer,
                            'emb_sec_last_token': emb_sec_last_token,
                            'emb_tok_bef_gen': emb_tok_bef_gen,
                            'all_embeddings': all_embeddings,
                            'decoded_tokens': decoded_tokens,
                        }
                    else:
                        logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                        # Aggregate predictions over num_generations.
                        full_responses.append(
                            (predicted_answer, token_log_likelihoods, all_embeddings, acc))

                example_payload = {
                    'question': question,
                    'context': context,
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': utils.get_reference(example),
                    'responses': full_responses,
                }

                t_write_0 = time.perf_counter()
                encoded_example_id = encode_example_id(example["id"])
                write_example_h5(examples_group, encoded_example_id, example_payload)
                write_elapsed = time.perf_counter() - t_write_0
                if (it % WRITE_LOG_EVERY) == 0:
                    logging.info(
                        "Wrote %d examples to %s (last write %.3fs)",
                        it,
                        generations_filename,
                        write_elapsed,
                    )

                if args.compute_p_true and dataset_split == 'validation':
                    # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                    p_true = p_true_utils.calculate_p_true(
                        model, question, most_likely_answer_dict['response'],
                        [r[0] for r in full_responses], p_true_few_shot_prompt,
                        hint=args.p_true_hint)
                    p_trues.append(p_true)
                    logging.info('p_true: %s', p_true)

        # Save locally (always done above) and upload to wandb if enabled
        if args.save_to_wandb:
            wandb.save(generations_filepath)

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        if args.save_to_wandb:
            wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false': [1 - p for p in p_trues],
                    'p_false_fixed': [1 - np.exp(p) for p in p_trues],
                }
            uncertainty_measures_path = os.path.join(run_output_dir, 'uncertainty_measures.h5')
            save_object_h5(uncertainty_measures_path, "uncertainty_measures", results_dict)
            if args.save_to_wandb:
                wandb.save(uncertainty_measures_path)

    experiment_details_path = os.path.join(run_output_dir, 'experiment_details.h5')
    save_object_h5(experiment_details_path, "experiment_details", experiment_details)
    if args.save_to_wandb:
        wandb.save(experiment_details_path)
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
