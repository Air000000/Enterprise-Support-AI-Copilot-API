from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from experiments.evals.eval_techqa_generation import (
    TechQAGenerationEvalResult,
    load_generation_checkpoint,
)
from experiments.evals.eval_techqa_retrieval import (
    DEFAULT_REPORT_DIR,
    TechQARetrievalResult,
)

FailureBucket = Literal[
    "retrieval_miss_top20",
    "retrieval_rank_6_20",
    "retrieval_rank_4_5",
    "top3_chunk_admission_gap",
    "gold_in_context_threshold_refusal",
    "gold_in_context_model_abstain",
    "gold_in_context_low_correctness",
    "gold_in_context_mid_correctness",
    "gold_in_context_high_correctness",
]

DEFAULT_RETRIEVAL_RESULTS_PATH = DEFAULT_REPORT_DIR / "train_results.jsonl"
DEFAULT_GENERATION_RESULTS_PATH = DEFAULT_REPORT_DIR / "train_generation_results.jsonl"
DEFAULT_FAILURE_ANALYSIS_DETAIL_PATH = DEFAULT_REPORT_DIR / "train_failure_analysis.jsonl"
DEFAULT_FAILURE_ANALYSIS_SUMMARY_PATH = (
    DEFAULT_REPORT_DIR / "train_failure_analysis_summary.json"
)


@dataclass(frozen=True)
class GenerationFailureAnalysisRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    retrieval_document_ranking: tuple[str, ...]
    generation_retrieved_document_ids: tuple[str, ...]
    gold_document_rank: int | None
    gold_in_generation_context: bool
    retrieval_status: str
    abstained: bool
    correctness_score: float | None
    faithfulness_score: float | None
    failure_bucket: FailureBucket


def _index_unique_by_question_id(
    results: Iterable[TechQARetrievalResult | TechQAGenerationEvalResult],
    *,
    kind: str,
) -> dict[str, TechQARetrievalResult | TechQAGenerationEvalResult]:
    indexed: dict[str, TechQARetrievalResult | TechQAGenerationEvalResult] = {}
    for result in results:
        if result.question_id in indexed:
            raise ValueError(f"duplicate {kind} question_id: {result.question_id}")
        indexed[result.question_id] = result
    return indexed


def _gold_document_rank(retrieval: TechQARetrievalResult) -> int | None:
    relevant = set(retrieval.relevant_document_ids)
    for rank, document_id in enumerate(retrieval.document_ranking, start=1):
        if document_id in relevant:
            return rank
    return None


def _classify_bucket(
    *,
    retrieval: TechQARetrievalResult,
    generation: TechQAGenerationEvalResult,
    gold_document_rank: int | None,
    gold_in_generation_context: bool,
) -> FailureBucket:
    if gold_document_rank is None:
        return "retrieval_miss_top20"
    if gold_document_rank >= 6:
        return "retrieval_rank_6_20"
    if gold_document_rank >= 4:
        return "retrieval_rank_4_5"
    if not gold_in_generation_context:
        return "top3_chunk_admission_gap"
    if generation.retrieval_status == "refused_low_relevance":
        return "gold_in_context_threshold_refusal"
    if generation.abstained:
        return "gold_in_context_model_abstain"

    score = generation.correctness_score
    if score is None:
        raise ValueError(
            f"correctness_score is required for {generation.question_id} when gold is in context"
        )
    if score <= 0.4:
        return "gold_in_context_low_correctness"
    if score < 0.8:
        return "gold_in_context_mid_correctness"
    return "gold_in_context_high_correctness"


def build_generation_failure_analysis(
    retrieval_results: Iterable[TechQARetrievalResult],
    generation_results: Iterable[TechQAGenerationEvalResult],
) -> list[GenerationFailureAnalysisRecord]:
    """Join frozen E0 retrieval and answerable generation results into observable buckets."""
    retrieval_by_id = _index_unique_by_question_id(
        retrieval_results,
        kind="retrieval",
    )

    answerable_generation = [result for result in generation_results if result.answerable]
    generation_by_id = _index_unique_by_question_id(
        answerable_generation,
        kind="generation",
    )

    retrieval_ids = set(retrieval_by_id)
    generation_ids = set(generation_by_id)
    if retrieval_ids != generation_ids:
        missing_retrieval = sorted(generation_ids - retrieval_ids)
        missing_generation = sorted(retrieval_ids - generation_ids)
        raise ValueError(
            "ID mismatch between retrieval and answerable generation results: "
            f"missing_retrieval={missing_retrieval}, "
            f"missing_generation={missing_generation}"
        )

    records: list[GenerationFailureAnalysisRecord] = []
    for question_id in sorted(generation_ids):
        retrieval = retrieval_by_id[question_id]
        generation = generation_by_id[question_id]
        assert isinstance(retrieval, TechQARetrievalResult)
        assert isinstance(generation, TechQAGenerationEvalResult)

        if retrieval.question.rstrip() != generation.question.rstrip():
            raise ValueError(
                f"question mismatch for {question_id}: "
                f"retrieval={retrieval.question!r}, generation={generation.question!r}"
            )

        rank = _gold_document_rank(retrieval)
        relevant_documents = set(retrieval.relevant_document_ids)
        gold_in_context = any(
            document_id in relevant_documents
            for document_id in generation.retrieved_document_ids
        )
        bucket = _classify_bucket(
            retrieval=retrieval,
            generation=generation,
            gold_document_rank=rank,
            gold_in_generation_context=gold_in_context,
        )

        records.append(
            GenerationFailureAnalysisRecord(
                question_id=question_id,
                question=generation.question,
                relevant_document_ids=retrieval.relevant_document_ids,
                retrieval_document_ranking=retrieval.document_ranking,
                generation_retrieved_document_ids=generation.retrieved_document_ids,
                gold_document_rank=rank,
                gold_in_generation_context=gold_in_context,
                retrieval_status=generation.retrieval_status,
                abstained=generation.abstained,
                correctness_score=generation.correctness_score,
                faithfulness_score=generation.faithfulness_score,
                failure_bucket=bucket,
            )
        )

    return records


def _retrieval_result_from_payload(payload: Mapping[str, Any]) -> TechQARetrievalResult:
    normalized = dict(payload)
    for field in (
        "relevant_document_ids",
        "raw_chunk_ids",
        "raw_document_ids",
        "document_ranking",
    ):
        normalized[field] = tuple(normalized[field])
    return TechQARetrievalResult(**normalized)


def load_retrieval_results(
    results_path: str | Path = DEFAULT_RETRIEVAL_RESULTS_PATH,
) -> list[TechQARetrievalResult]:
    path = Path(results_path)
    results: list[TechQARetrievalResult] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            results.append(_retrieval_result_from_payload(json.loads(line)))
    return results


def summarize_generation_failure_analysis(
    records: Iterable[GenerationFailureAnalysisRecord],
) -> dict[str, Any]:
    ordered_records = list(records)
    answerable_count = len(ordered_records)
    bucket_counts = Counter(record.failure_bucket for record in ordered_records)
    gold_in_context_count = sum(
        record.gold_in_generation_context for record in ordered_records
    )

    return {
        "answerable_count": answerable_count,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_rates": {
            bucket: count / answerable_count if answerable_count else 0.0
            for bucket, count in sorted(bucket_counts.items())
        },
        "gold_in_generation_context_count": gold_in_context_count,
        "gold_not_in_generation_context_count": answerable_count - gold_in_context_count,
    }


def materialize_generation_failure_analysis(
    *,
    retrieval_results_path: str | Path = DEFAULT_RETRIEVAL_RESULTS_PATH,
    generation_results_path: str | Path = DEFAULT_GENERATION_RESULTS_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    """Materialize deterministic E0 failure-analysis detail and summary artifacts once."""
    output_dir = Path(report_dir)
    detail_path = output_dir / "train_failure_analysis.jsonl"
    summary_path = output_dir / "train_failure_analysis_summary.json"

    if detail_path.exists():
        raise FileExistsError(detail_path)
    if summary_path.exists():
        raise FileExistsError(summary_path)

    retrieval_results = load_retrieval_results(retrieval_results_path)
    generation_results = load_generation_checkpoint(generation_results_path)
    records = build_generation_failure_analysis(retrieval_results, generation_results)
    summary = summarize_generation_failure_analysis(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    with detail_path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with summary_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary
