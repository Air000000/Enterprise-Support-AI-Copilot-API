from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
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
DEFAULT_GENERATION_REPORT_DIR = Path("experiments/evals/reports/e0_generation")
DEFAULT_GENERATION_CHECKPOINT_PATH = (
    DEFAULT_GENERATION_REPORT_DIR / "train_checkpoint.jsonl"
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
CaseEvaluator = Callable[[TechQAGenerationCase], "TechQAGenerationEvalResult"]


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


def _summarize_generation_results(
    results: Iterable[TechQAGenerationEvalResult],
    *,
    split: EvalSplit,
) -> TechQAGenerationEvalSummary:
    ordered_results = tuple(sorted(results, key=lambda result: result.question_id))
    correctness_scores = [
        result.correctness_score
        for result in ordered_results
        if result.answerable and result.correctness_score is not None
    ]
    faithfulness_scores = [
        result.faithfulness_score
        for result in ordered_results
        if result.answerable and result.faithfulness_score is not None
    ]
    impossible_results = [result for result in ordered_results if not result.answerable]
    latencies_ms = [result.e2e_latency_ms for result in ordered_results]

    answerable_count = sum(result.answerable for result in ordered_results)
    impossible_count = len(impossible_results)

    return TechQAGenerationEvalSummary(
        split=split,
        query_count=len(ordered_results),
        answerable_count=answerable_count,
        impossible_count=impossible_count,
        correctness_mean=(
            sum(correctness_scores) / len(correctness_scores)
            if correctness_scores
            else None
        ),
        faithfulness_mean=(
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else None
        ),
        abstention_accuracy=(
            sum(result.abstained for result in impossible_results) / impossible_count
            if impossible_count
            else 0.0
        ),
        hallucination_rate=(
            sum(result.hallucinated for result in impossible_results) / impossible_count
            if impossible_count
            else 0.0
        ),
        e2e_latency_p50_ms=_percentile(latencies_ms, 0.50),
        e2e_latency_p95_ms=_percentile(latencies_ms, 0.95),
        results=ordered_results,
    )


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

    return _summarize_generation_results(results, split=split)


def _generation_result_from_payload(
    payload: Mapping[str, Any],
) -> TechQAGenerationEvalResult:
    normalized = dict(payload)
    normalized["retrieved_chunk_ids"] = tuple(normalized["retrieved_chunk_ids"])
    normalized["retrieved_document_ids"] = tuple(normalized["retrieved_document_ids"])
    normalized["retrieval_context"] = tuple(normalized["retrieval_context"])
    return TechQAGenerationEvalResult(**normalized)


def load_generation_checkpoint(
    checkpoint_path: str | Path = DEFAULT_GENERATION_CHECKPOINT_PATH,
) -> list[TechQAGenerationEvalResult]:
    """Load completed generation-eval results from an append-only JSONL checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        return []

    results: list[TechQAGenerationEvalResult] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            results.append(_generation_result_from_payload(json.loads(line)))
    return results


def _append_generation_checkpoint(
    result: TechQAGenerationEvalResult,
    checkpoint_path: str | Path,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _evaluate_single_generation_case(
    case: TechQAGenerationCase,
) -> TechQAGenerationEvalResult:
    summary = evaluate_techqa_generation_cases([case], split=case.split)
    return summary.results[0]


def run_resumable_generation_eval(
    cases: Iterable[TechQAGenerationCase],
    *,
    evaluator: CaseEvaluator = _evaluate_single_generation_case,
    checkpoint_path: str | Path = DEFAULT_GENERATION_CHECKPOINT_PATH,
    split: EvalSplit = "train",
) -> TechQAGenerationEvalSummary:
    """Evaluate only missing cases and persist every successful result immediately."""
    selected_cases = sorted(
        (case for case in cases if case.split == split),
        key=lambda case: case.question_id,
    )
    if not selected_cases:
        raise ValueError(f"No TechQA generation cases found for split={split}")

    target_ids = {case.question_id for case in selected_cases}
    checkpoint_results = load_generation_checkpoint(checkpoint_path)
    results_by_id = {
        result.question_id: result
        for result in checkpoint_results
        if result.question_id in target_ids
    }

    completed = len(results_by_id)
    total = len(selected_cases)
    if completed:
        print(f"Resuming generation eval from {completed}/{total} completed cases...")

    for case in selected_cases:
        if case.question_id in results_by_id:
            continue

        result = evaluator(case)
        if result.question_id != case.question_id:
            raise RuntimeError(
                "Generation evaluator returned mismatched question_id: "
                f"expected={case.question_id}, actual={result.question_id}"
            )

        _append_generation_checkpoint(result, checkpoint_path)
        results_by_id[case.question_id] = result
        completed += 1
        print(f"Completed generation eval: {completed}/{total} ({case.question_id})")

    ordered_results = [results_by_id[case.question_id] for case in selected_cases]
    return _summarize_generation_results(ordered_results, split=split)


def write_generation_reports(
    summary: TechQAGenerationEvalSummary,
    *,
    report_dir: str | Path = DEFAULT_GENERATION_REPORT_DIR,
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
) -> None:
    """Write reproducible TechQA generation results and aggregate metrics."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    generation = manifest["generation_dataset"]

    run_manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "e0_generation",
        "split": summary.split,
        "query_count": summary.query_count,
        "answerable_count": summary.answerable_count,
        "impossible_count": summary.impossible_count,
        "generation_top_k": DEFAULT_GENERATION_TOP_K,
        "refusal_max_distance": DEFAULT_REFUSAL_MAX_DISTANCE,
        "context_source": "actual retrieved chunks from techqa_e0_dense",
        "generation_dataset": {
            "repo": generation["repo"],
            "revision": generation["revision"],
            "metadata_sha256": generation["metadata_sha256"],
        },
        "generator": "rag_runtime.query_rag_chroma.generate_answer",
        "judge": {
            "framework": "deepeval",
            "model": "qwen3.5-plus",
            "correctness_metric": "GEval",
            "faithfulness_metric": "FaithfulnessMetric",
        },
    }
    (output_dir / f"{summary.split}_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / f"{summary.split}_results.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for result in summary.results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    metrics_payload = {
        "query_count": summary.query_count,
        "answerable_count": summary.answerable_count,
        "impossible_count": summary.impossible_count,
        "correctness_mean": summary.correctness_mean,
        "faithfulness_mean": summary.faithfulness_mean,
        "abstention_accuracy": summary.abstention_accuracy,
        "hallucination_rate": summary.hallucination_rate,
        "e2e_latency_p50_ms": summary.e2e_latency_p50_ms,
        "e2e_latency_p95_ms": summary.e2e_latency_p95_ms,
    }
    (output_dir / f"{summary.split}_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("Loading frozen TechQA generation cases...")
    cases = load_frozen_techqa_generation_cases()
    train_count = sum(case.split == "train" for case in cases)
    print(f"Loaded cases: {len(cases)} (TRAIN={train_count})")

    print("Running resumable E0 generation evaluation on TRAIN only...")
    summary = run_resumable_generation_eval(
        cases,
        checkpoint_path=DEFAULT_GENERATION_CHECKPOINT_PATH,
        split="train",
    )
    write_generation_reports(summary)

    print("TechQA E0 TRAIN generation evaluation completed.")
    print(f"Queries:              {summary.query_count}")
    print(f"Answerable:           {summary.answerable_count}")
    print(f"Impossible:           {summary.impossible_count}")
    print(f"Correctness mean:     {summary.correctness_mean}")
    print(f"Faithfulness mean:    {summary.faithfulness_mean}")
    print(f"Abstention accuracy:  {summary.abstention_accuracy:.6f}")
    print(f"Hallucination rate:   {summary.hallucination_rate:.6f}")
    print(f"E2E p50 latency:      {summary.e2e_latency_p50_ms:.3f} ms")
    print(f"E2E p95 latency:      {summary.e2e_latency_p95_ms:.3f} ms")
    print(f"Reports:              {DEFAULT_GENERATION_REPORT_DIR}")


if __name__ == "__main__":
    main()
