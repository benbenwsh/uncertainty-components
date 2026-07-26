"""Data Loading Utilities."""
import logging
import os
import json
import hashlib
import datasets


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')

        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        _validation_dataset = [reformat(d) for d in validation_dataset]
        # For semantic entropy generation: merge training with test set for more samples.
        validation_dataset = _validation_dataset + train_dataset

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "gsm8k":
        # openai/gsm8k parquet currently lists duplicate shards (2x expected size);
        # skip size verification and keep unique questions.
        dataset = datasets.load_dataset(
            "openai/gsm8k",
            "main",
            verification_mode="no_checks",
        )
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))

        def _gsm8k_final_answer(answer_text):
            answer_text = str(answer_text)
            if "####" in answer_text:
                return answer_text.rsplit("####", 1)[-1].strip()
            return answer_text.strip()

        def _dedupe_by_question(split):
            seen = set()
            out = []
            for ex in split:
                question = ex["question"]
                if question in seen:
                    continue
                seen.add(question)
                out.append(ex)
            return out

        train_dataset = _dedupe_by_question(train_dataset)
        validation_dataset = _dedupe_by_question(validation_dataset)

        reformat = lambda x: {
            "question": x["question"],
            "context": "",
            "id": md5hash(str(x["question"])),
            "answers": {"text": [_gsm8k_final_answer(x["answer"])]},
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "math":
        # Hendrycks MATH (competition math). Parquet metadata can disagree with
        # shard sizes; skip size verification like gsm8k.
        dataset = datasets.load_dataset(
            "DigitalLearningGmbH/MATH-lighteval",
            verification_mode="no_checks",
        )
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))

        def _math_boxed_answer(solution_text):
            """Extract the final answer from the last \\boxed{...} in the solution."""
            solution_text = str(solution_text)
            idx = solution_text.rfind(r"\boxed")
            if idx < 0:
                return None
            brace_start = solution_text.find("{", idx)
            if brace_start < 0:
                return None
            depth = 0
            for j in range(brace_start, len(solution_text)):
                if solution_text[j] == "{":
                    depth += 1
                elif solution_text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return solution_text[brace_start + 1:j].strip()
            return None

        def _reformat_math(example):
            answer = _math_boxed_answer(example["solution"])
            if answer is None:
                return None
            return {
                "question": example["problem"],
                "context": "",
                "id": md5hash(str(example["problem"])),
                "level": example.get("level"),
                "type": example.get("type"),
                "answers": {"text": [answer]},
            }

        train_dataset = [r for d in train_dataset if (r := _reformat_math(d)) is not None]
        validation_dataset = [
            r for d in validation_dataset if (r := _reformat_math(d)) is not None
        ]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
       
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
    
    elif dataset_name == "med_qa":
        dataset = datasets.load_dataset("bigbio/med_qa")
        logging.info('Dataset: %s', dataset)
        for key in 'train', 'validation':
            ids = ['train' + str(i) for i in range(len(dataset[key]))]
            dataset[key] = dataset[key].add_column("id", ids)

            new_column = [None] * len(dataset[key])
            dataset[key] = dataset[key].add_column("context", new_column)

            answers = [
                {'text': [answer], 'answer_start': [0]}
                for answer in dataset[key][:]['answer']
            ]
            dataset[key] = dataset[key].add_column("answers", answers)

            if add_options:
                options = dataset[key][:]['options']
                options_string = [
                    [option['value'] + '\n' for option in option_list]
                    for option_list in options
                ]
                questions = dataset[key][:]['question']
                # zip questions and options
                questions_options = [
                    question + '\n' + ''.join(option_list)
                    for question, option_list in zip(questions, options_string)
                ]

                dataset[key] = dataset[key].remove_columns(['question'])
                dataset[key] = dataset[key].add_column(
                    "question", questions_options)

        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        # scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = "~/uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # split into training and validation set
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "record":
        # Load the JSON file
        for split in ["train", "dev"]:
            dataset_dictionary = {
                "id": [], "question": [], "context": [], "answers": []}
            path = f"~/uncertainty/data/record/{split}.json"
            with open(path, "rb") as file:
                data = json.load(file)

            # Extract the relevant information and create a dictionary
            for item in data["data"]:
                for qa in item["qas"]:  # pylint: disable=invalid-name
                    dataset_dictionary["id"].append(qa["id"])
                    dataset_dictionary["question"].append(qa["query"])
                    dataset_dictionary["context"].append(
                        item["passage"]["text"])
                    list_of_answer_strings = []
                    list_of_answer_starts = []
                    for answer in qa["answers"]:
                        list_of_answer_strings.append(answer["text"])
                        list_of_answer_starts.append(answer["start"])
                    dataset_dictionary["answers"].append({
                        "text": list_of_answer_strings,
                        "answer_start": list_of_answer_starts})

            # Create the Hugging Face dataset
            if split == "train":
                train_dataset = datasets.Dataset.from_dict(dataset_dictionary)
                logging.info('train_dataset[0]: %s', train_dataset[0])
            else:
                validation_dataset = datasets.Dataset.from_dict(dataset_dictionary)
                logging.info('validation_dataset[0]: %s', validation_dataset[0])

    return train_dataset, validation_dataset
