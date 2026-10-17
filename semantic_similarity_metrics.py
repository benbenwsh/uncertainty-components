"""Semantic-similarity scorers used after ablation generation.

``sentence_transformer`` is all-MiniLM-L6-v2 cosine similarity between the
no-intervention semantic answer and the intervention semantic answer.
``xverify`` is IAAR-Shanghai/xVerify-3B-Ia, which judges whether the
intervention answer matches that no-intervention answer for the input-H5
question.
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch

SEMANTIC_SIMILARITY_METHODS = ("sentence_transformer", "xverify")
DEFAULT_SEMANTIC_SIMILARITY_METHOD = "sentence_transformer"
SENTENCE_TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"
XVERIFY_MODEL_NAME = "xVerify-3B-Ia"
XVERIFY_MODEL_ID = "IAAR-Shanghai/xVerify-3B-Ia"
XVERIFY_MAX_NEW_TOKENS = 16
XVERIFY_BATCH_SIZE = 8
XVERIFY_MAX_PROMPT_TOKENS = 4096

XVERIFY_JUDGE_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''

# Official Llama-3 wrapper for xVerify-3B-Ia (src/xVerify/prompts/base_template.py).
XVERIFY_3B_IA_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful AI assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

_JUDGEMENT_RE = re.compile(r"\[\s*(incorrect|correct)\s*\]", flags=re.IGNORECASE)


def add_semantic_similarity_method_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--semantic_similarity_method",
        type=str,
        default=DEFAULT_SEMANTIC_SIMILARITY_METHOD,
        choices=list(SEMANTIC_SIMILARITY_METHODS),
        help=(
            "How semantic_similarity is computed after generation. "
            "sentence_transformer: all-MiniLM-L6-v2 cosine between the no-intervention "
            "semantic answer and the intervention semantic answer. "
            "xverify: IAAR-Shanghai/xVerify-3B-Ia judges the intervention answer against "
            "the no-intervention answer for the input_h5 question "
            "([Correct]=1, [Incorrect]=0)."
        ),
    )


def semantic_similarity_model_name(method: str) -> str:
    if method == "sentence_transformer":
        return SENTENCE_TRANSFORMER_MODEL_NAME
    if method == "xverify":
        return XVERIFY_MODEL_ID
    raise ValueError(
        f"Unknown semantic_similarity_method {method!r}. "
        f"Expected one of {list(SEMANTIC_SIMILARITY_METHODS)}."
    )


def semantic_similarity_config_lines(method: str) -> List[str]:
    return [
        f"semantic_similarity_method={method}",
        f"semantic_similarity_model={semantic_similarity_model_name(method)}",
    ]


def release_cuda_cache() -> None:
    """Free cached CUDA blocks after the caller has deleted the language model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_sentence_transformer_for_metrics(
    model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME,
    device: Optional[str] = None,
):
    from sentence_transformers import SentenceTransformer

    if device is None:
        return SentenceTransformer(model_name)
    return SentenceTransformer(model_name, device=device)


def batch_compute_semantic_similarities(
    sentence_model,
    pairs: Sequence[tuple],
) -> List[float]:
    if not pairs:
        return []
    from sentence_transformers import util

    texts_a = [text_a for text_a, _text_b in pairs]
    texts_b = [_text_b for _text_a, _text_b in pairs]
    emb_a = sentence_model.encode(texts_a, convert_to_tensor=True)
    emb_b = sentence_model.encode(texts_b, convert_to_tensor=True)
    sims = util.cos_sim(emb_a, emb_b).diag()
    return [max(0.0, float(sim.item())) for sim in sims]


def parse_xverify_judgement(text: str) -> Optional[float]:
    """Map an xVerify completion to 1.0 (Correct) or 0.0 (Incorrect)."""
    if not text or not str(text).strip():
        return None
    matches: List[tuple] = []
    for match in _JUDGEMENT_RE.finditer(str(text)):
        label = match.group(1).lower()
        matches.append((match.start(), 0.0 if label == "incorrect" else 1.0))
    if matches:
        matches.sort(key=lambda item: item[0])
        return float(matches[-1][1])
    stripped = str(text).strip().lower()
    if stripped in {"correct", "[correct]"}:
        return 1.0
    if stripped in {"incorrect", "[incorrect]"}:
        return 0.0
    return None


def _fill_placeholders(template: str, replacements: Mapping[str, str]) -> str:
    """Replace ``{key}`` tokens once, without scanning text that was just inserted."""
    if not replacements:
        return template
    pattern = re.compile(
        "|".join(re.escape("{" + key + "}") for key in replacements)
    )

    def _replace(match: re.Match) -> str:
        key = match.group(0)[1:-1]
        return replacements[key]

    return pattern.sub(_replace, template)


def build_xverify_prompt(question: str, llm_output: str, correct_answer: str) -> str:
    judge = _fill_placeholders(
        XVERIFY_JUDGE_PROMPT,
        {
            "question": question,
            "output": llm_output,
            "answer": correct_answer,
        },
    )
    return _fill_placeholders(XVERIFY_3B_IA_CHAT_TEMPLATE, {"query": judge})


def load_questions_from_h5(path: str | Path) -> Dict[str, str]:
    """Read examples/<id>/question without loading embedding datasets."""
    questions: Dict[str, str] = {}
    with h5py.File(path, "r") as h5_file:
        if "examples" not in h5_file:
            raise ValueError(f"HDF5 file has no 'examples' group: {path}")
        ex_group = h5_file["examples"]
        for example_id in ex_group.keys():
            node = ex_group[example_id]
            if "question" not in node:
                continue
            value = node["question"][()]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str) and value.strip():
                questions[str(example_id)] = value
    return questions


class SemanticSimilarityScorer:
    """Lazy scorer. Construct it only after the language model has been released."""

    def __init__(self, method: str, device: Optional[str] = None):
        if method not in SEMANTIC_SIMILARITY_METHODS:
            raise ValueError(
                f"Unknown semantic_similarity_method {method!r}. "
                f"Expected one of {list(SEMANTIC_SIMILARITY_METHODS)}."
            )
        self.method = method
        self.device = device
        self._sentence_model = None
        self._xverify_tokenizer = None
        self._xverify_model = None

    def close(self) -> None:
        self._sentence_model = None
        self._xverify_tokenizer = None
        self._xverify_model = None
        release_cuda_cache()

    def score_answers(
        self,
        *,
        example_ids: Sequence[str],
        correct_answers: Sequence[str],
        llm_outputs: Sequence[str],
        questions_by_id: Optional[Mapping[str, str]] = None,
    ) -> List[float]:
        count = len(example_ids)
        if count != len(correct_answers) or count != len(llm_outputs):
            raise ValueError(
                "example_ids, correct_answers, and llm_outputs must have the same length; "
                f"got {count}, {len(correct_answers)}, {len(llm_outputs)}."
            )
        if count == 0:
            return []
        if self.method == "sentence_transformer":
            pairs = list(zip(correct_answers, llm_outputs))
            return batch_compute_semantic_similarities(self._sentence_transformer(), pairs)
        if questions_by_id is None:
            raise ValueError("xVerify scoring requires questions loaded from input_h5.")
        questions: List[str] = []
        for ex_id in example_ids:
            question = questions_by_id.get(str(ex_id))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Example {ex_id} is missing a question string in input_h5.")
            questions.append(question)
        return self._score_xverify(questions, list(llm_outputs), list(correct_answers))

    def _sentence_transformer(self):
        if self._sentence_model is None:
            logging.info(
                "Loading sentence-transformers model %s for semantic_similarity on device=%s.",
                SENTENCE_TRANSFORMER_MODEL_NAME,
                self.device,
            )
            self._sentence_model = load_sentence_transformer_for_metrics(
                SENTENCE_TRANSFORMER_MODEL_NAME,
                device=self.device,
            )
        return self._sentence_model

    def _xverify(self):
        if self._xverify_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            logging.info(
                "Loading %s for semantic_similarity on device=%s dtype=%s.",
                XVERIFY_MODEL_ID,
                device,
                dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(XVERIFY_MODEL_ID, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                XVERIFY_MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model.to(device)
            model.eval()
            self._xverify_tokenizer = tokenizer
            self._xverify_model = model
        return self._xverify_tokenizer, self._xverify_model

    def _score_xverify(
        self,
        questions: Sequence[str],
        llm_outputs: Sequence[str],
        correct_answers: Sequence[str],
    ) -> List[float]:
        tokenizer, model = self._xverify()
        prompts = [
            build_xverify_prompt(question, llm_output, correct_answer)
            for question, llm_output, correct_answer in zip(questions, llm_outputs, correct_answers)
        ]
        scores: List[float] = []
        for start in range(0, len(prompts), XVERIFY_BATCH_SIZE):
            batch = prompts[start : start + XVERIFY_BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=XVERIFY_MAX_PROMPT_TOKENS,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=XVERIFY_MAX_NEW_TOKENS,
                    do_sample=False,
                )
            prompt_width = int(inputs["input_ids"].shape[1])
            for row in output_ids:
                text = tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
                parsed = parse_xverify_judgement(text)
                if parsed is None:
                    logging.warning("Unparseable xVerify judgement %r; scoring 0.", text)
                    parsed = 0.0
                scores.append(float(parsed))
        return scores


def open_semantic_similarity_scorer(
    method: str,
    device: Optional[str],
    input_h5: str,
) -> tuple:
    """Return (scorer, questions_by_id). Questions are loaded only for xverify."""
    scorer = SemanticSimilarityScorer(method, device=device)
    questions_by_id = load_questions_from_h5(input_h5) if method == "xverify" else None
    return scorer, questions_by_id
