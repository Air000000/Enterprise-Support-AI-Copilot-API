from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from experiments.evals.eval_techqa_generation import (
    DEFAULT_GENERATION_REPORT_DIR,
    DEFAULT_REFUSAL_ANSWER,
    TechQAGenerationEvalResult,
)

DEFAULT_ABSTENTION_AUDIT_PATH = (
    DEFAULT_GENERATION_REPORT_DIR / "abstention_audit.jsonl"
)

MANUAL_ABSTENTION_AUDIT_CATEGORIES = (
    "corpus_supported_impossible",
    "semantic_abstention",
    "true_unsafe_answer",
    "correct_abstain",
)

AutomaticSignal = Literal[
    "exact_refusal",
    "semantic_refusal_candidate",
    "non_refusal_candidate",
]


@dataclass(frozen=True)
class AbstentionAuditRecord:
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
    automatic_signal: AutomaticSignal
    manual_category: str
    manual_notes: str


def _automatic_signal(result: TechQAGenerationEvalResult) -> AutomaticSignal:
    answer = result.generated_answer.strip()
    if answer == DEFAULT_REFUSAL_ANSWER:
        return "exact_refusal"
    if DEFAULT_REFUSAL_ANSWER in answer:
        return "semantic_refusal_candidate"
    return "non_refusal_candidate"


def build_abstention_audit_records(
    results: Iterable[TechQAGenerationEvalResult],
) -> list[AbstentionAuditRecord]:
    """Build deterministic offline review records for impossible TechQA cases."""
    impossible_results = sorted(
        (result for result in results if not result.answerable),
        key=lambda result: result.question_id,
    )

    return [
        AbstentionAuditRecord(
            question_id=result.question_id,
            question=result.question,
            gold_answer=result.gold_answer,
            answerable=result.answerable,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
            retrieved_document_ids=result.retrieved_document_ids,
            retrieval_context=result.retrieval_context,
            generated_answer=result.generated_answer,
            retrieval_status=result.retrieval_status,
            top_distance=result.top_distance,
            abstained=result.abstained,
            hallucinated=result.hallucinated,
            automatic_signal=_automatic_signal(result),
            manual_category="",
            manual_notes="",
        )
        for result in impossible_results
    ]


def write_abstention_audit_template(
    records: Iterable[AbstentionAuditRecord],
    *,
    output_path: str | Path = DEFAULT_ABSTENTION_AUDIT_PATH,
) -> None:
    """Write the offline abstention audit template exactly once."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
