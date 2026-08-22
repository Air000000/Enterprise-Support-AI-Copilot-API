from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from experiments.evals.adapters.techqa import (
    TechQAGenerationCase,
    build_techqa_generation_cases,
)
from experiments.evals.build_techqa_index import search_techqa_index
from experiments.evals.judges.deepeval_dashscope import DashScopeDeepEvalModel
from rag_runtime.query_rag_chroma import generate_answer

DEFAULT_TECHQA_MANIFEST_PATH = Path(
    "experiments/evals/datasets/techqa/manifest.json"
)
DEFAULT_GENERATION_TOP_K = 3
DEFAULT_REFUSAL_MAX_DISTANCE = 0.9
DEFAULT_REFUSAL_ANSWER = "我在已提供资料中没有找到足够依据。"

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]
Searcher = Callable[..., list[Any]]
Generator = Callable[[str, str], str]
Judge = Callable[..., "GenerationJudgeResult"]
Clock = Callable[[], float]
EvalSplit = Literal["train", "dev"]


@dataclass(frozen=True)
class GenerationJudgeResult:
    correctness_score: float
    correctness_reason: str
    faithfulness_score: float
    faithfulness_reason: str


@dataclass(frozen=True)
class TechQAGenerationEvalResult:
    question_id: str
    question: str
    gold_answer: str
    answerable: bool
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    retrieval_context: tuple[str, ...]
    generated_answer: str
    retrieval_status: str
    top_distance: float | None
    abstained: bool
    hallucinated: bool
    correctness_score: float | None
    correctness_reason: str | None
    faithfulness_score: float | None
    faithfulness_reason: str | None
    e2e_latency_ms: float


@dataclass(frozen=True)
class TechQAGenerationEvalSummary:
    split: EvalSplit
    query_count: int
    answerable_count: int
    impossible_count: int
    correctness_mean: float | None
    faithfulness_mean: float | None
    abstention_accuracy: float
    hallucination_rate: float
    e2e_latency_p50_ms: float
    e2e_latency_p95_ms: float
    results: tuple[TechQAGenerationEvalResult, ...]


def _load_manifest(
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_frozen_techqa_generation_cases(
    *,
    dataset_loader: DatasetLoader | None = None,
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
    expected_record_count: int | None = 910,
) -> list[TechQAGenerationCase]:
    """Load NVIDIA TechQA generation metadata from the frozen revision."""
    manifest = _load_manifest(manifest_path)
    generation = manifest["generation_dataset"]

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    rows = dataset_loader(
        generation["repo"],
        split="train",
        revision=generation["revision"],
    )
    cases = build_techqa_generation_cases(rows)

    if expected_record_count is not None and len(cases) != expected_record_count:
        raise RuntimeError(
            "Frozen TechQA generation record count mismatch: "
            f"expected={expected_record_count}, actual={len(cases)}"
        )

    return cases


def build_techqa_retrieved_context(results: Iterable[Any]) -> str:
    """Format only the chunks returned by the isolated TechQA retriever."""
    context_parts: list[str] = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            (
                f"[Source {index}]\n"
                f"document_id: {result.document_id}\n"
                f"chunk_id: {result.chunk_id}\n"
                f"distance: {float(result.distance):.4f}\n\n"
                f"content:\n{result.content}"
            )
        )

    return "\n\n---\n\n".join(context_parts)


@lru_cache(maxsize=1)
def _default_judge_model() -> DashScopeDeepEvalModel:
    return DashScopeDeepEvalModel(model_name="qwen3.5-plus")


def judge_techqa_generation(
    *,
    question: str,
    generated_answer: str,
    gold_answer: str,
    retrieval_context: list[str],
) -> GenerationJudgeResult:
    """Score one answerable TechQA output with DeepEval using the DashScope judge."""
    model = _default_judge_model()
    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=gold_answer,
        retrieval_context=retrieval_context,
    )

    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually correct and answers "
            "the question based on the expected output."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        async_mode=False,
    )
    faithfulness = FaithfulnessMetric(
        model=model,
        include_reason=True,
        async_mode=False,
    )

    correctness.measure(test_case)
    faithfulness.measure(test_case)

    return GenerationJudgeResult(
        correctness_score=float(correctness.score or 0.0),
        correctness_reason=str(correctness.reason or ""),
        faithfulness_score=float(faithfulness.score or 0.0),
        faithfulness_reason=str(faithfulness.reason or ""),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def evaluate_techqa_generation_cases(
    cases: Iterable[TechQAGenerationCase],
    *,
    searcher: Searcher = search_techqa_index,
    generator: Generator = generate_answer,
    judge: Judge = judge_techqa_generation,
    split: EvalSplit = "train",
    top_k: int = DEFAULT_GENERATION_TOP_K,
    refusal_max_distance: float = DEFAULT_REFUSAL_MAX_DISTANCE,
    clock: Clock = time.perf_counter,
) -> TechQAGenerationEvalSummary:
    """Evaluate the production-equivalent RAG answer path on one TechQA split."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    selected_cases = sorted(
        (case for case in cases if case.split == split),
        key=lambda case: case.question_id,
    )
    if not selected_cases:
        raise ValueError(f"No TechQA generation cases found for split={split}")

    results: list[TechQAGenerationEvalResult] = []
    correctness_scores: list[float] = []
    faithfulness_scores: list[float] = []
    impossible_abstentions: list[bool] = []
    impossible_hallucinations: list[bool] = []
    latencies_ms: list[float] = []

    for case in selected_cases:
        started = clock()
        retrieved = searcher(case.question, top_k=top_k)
        retrieval_context = [str(result.content) for result in retrieved]
        context = build_techqa_retrieved_context(retrieved)
        top_distance = float(retrieved[0].distance) if retrieved else None

        if not retrieved:
            retrieval_status = "no_context"
            generated_answer = DEFAULT_REFUSAL_ANSWER
        elif top_distance is not None and top_distance > refusal_max_distance:
            retrieval_status = "refused_low_relevance"
            generated_answer = DEFAULT_REFUSAL_ANSWER
        else:
            retrieval_status = "ok"
            generated_answer = generator(case.question, context)

        e2e_latency_ms = (clock() - started) * 1000.0
        latencies_ms.append(e2e_latency_ms)

        abstained = generated_answer.strip() == DEFAULT_REFUSAL_ANSWER
        hallucinated = (not case.answerable) and not abstained

        correctness_score: float | None = None
        correctness_reason: str | None = None
        faithfulness_score: float | None = None
        faithfulness_reason: str | None = None

        if case.answerable:
            judge_result = judge(
                question=case.question,
                generated_answer=generated_answer,
                gold_answer=case.gold_answer,
                retrieval_context=retrieval_context,
            )
            correctness_score = judge_result.correctness_score
            correctness_reason = judge_result.correctness_reason
            faithfulness_score = judge_result.faithfulness_score
            faithfulness_reason = judge_result.faithfulness_reason
            correctness_scores.append(correctness_score)
            faithfulness_scores.append(faithfulness_score)
        else:
            impossible_abstentions.append(abstained)
            impossible_hallucinations.append(hallucinated)

        results.append(
            TechQAGenerationEvalResult(
                question_id=case.question_id,
                question=case.question,
                gold_answer=case.gold_answer,
                answerable=case.answerable,
                retrieved_chunk_ids=tuple(str(result.chunk_id) for result in retrieved),
                retrieved_document_ids=tuple(
                    str(result.document_id) for result in retrieved
                ),
                retrieval_context=tuple(retrieval_context),
                generated_answer=generated_answer,
                retrieval_status=retrieval_status,
                top_distance=top_distance,
                abstained=abstained,
                hallucinated=hallucinated,
                correctness_score=correctness_score,
                correctness_reason=correctness_reason,
                faithfulness_score=faithfulness_score,
                faithfulness_reason=faithfulness_reason,
                e2e_latency_ms=e2e_latency_ms,
            )
        )

    answerable_count = len(correctness_scores)
    impossible_count = len(impossible_abstentions)

    return TechQAGenerationEvalSummary(
        split=split,
        query_count=len(results),
        answerable_count=answerable_count,
        impossible_count=impossible_count,
        correctness_mean=(
            sum(correctness_scores) / answerable_count if answerable_count else None
        ),
        faithfulness_mean=(
            sum(faithfulness_scores) / answerable_count if answerable_count else None
        ),
        abstention_accuracy=(
            sum(impossible_abstentions) / impossible_count if impossible_count else 0.0
        ),
        hallucination_rate=(
            sum(impossible_hallucinations) / impossible_count
            if impossible_count
            else 0.0
        ),
        e2e_latency_p50_ms=_percentile(latencies_ms, 0.50),
        e2e_latency_p95_ms=_percentile(latencies_ms, 0.95),
        results=tuple(results),
    )
