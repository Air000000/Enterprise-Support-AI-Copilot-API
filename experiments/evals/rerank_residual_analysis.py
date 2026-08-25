from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

ResidualBucket = Literal[
    "resolved_top20",
    "rerank_residual_top20",
    "dense_candidate_miss_top100",
]


@dataclass(frozen=True)
class RerankResidualRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    dense_candidate_document_ids: tuple[str, ...]
    e1_document_ranking: tuple[str, ...]
    residual_bucket: ResidualBucket


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in indexed:
            raise RuntimeError(f"duplicate {label} question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def _collapse_document_ids(document_ids: Sequence[Any]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in document_ids:
        document_id = str(value)
        if document_id in seen:
            continue
        seen.add(document_id)
        ordered.append(document_id)
    return tuple(ordered)


def _contains_relevant(
    ranking: Sequence[Any],
    relevant_document_ids: Sequence[str],
) -> bool:
    relevant = set(relevant_document_ids)
    return any(str(document_id) in relevant for document_id in ranking)


def build_residual_records(
    e0_rows: Sequence[Mapping[str, Any]],
    e1_rows: Sequence[Mapping[str, Any]],
) -> list[RerankResidualRecord]:
    e0_by_id = _index_unique(e0_rows, label="E0")
    e1_by_id = _index_unique(e1_rows, label="E1")

    if set(e0_by_id) != set(e1_by_id):
        raise RuntimeError("E0/E1 question_id mismatch")

    records: list[RerankResidualRecord] = []
    for question_id in sorted(e0_by_id):
        e0 = e0_by_id[question_id]
        e1 = e1_by_id[question_id]

        dense_chunk_ids = tuple(str(value) for value in e0["raw_chunk_ids"])
        rerank_dense_chunk_ids = tuple(str(value) for value in e1["dense_chunk_ids"])
        if rerank_dense_chunk_ids != dense_chunk_ids:
            raise RuntimeError(
                f"dense candidate chunk identity mismatch for {question_id}"
            )

        relevant_document_ids = tuple(
            str(value) for value in e0["relevant_document_ids"]
        )
        dense_candidate_document_ids = _collapse_document_ids(
            e0["raw_document_ids"]
        )
        e1_document_ranking = tuple(
            str(value) for value in e1["document_ranking"]
        )

        if _contains_relevant(e1_document_ranking[:20], relevant_document_ids):
            residual_bucket: ResidualBucket = "resolved_top20"
        elif _contains_relevant(
            dense_candidate_document_ids,
            relevant_document_ids,
        ):
            residual_bucket = "rerank_residual_top20"
        else:
            residual_bucket = "dense_candidate_miss_top100"

        records.append(
            RerankResidualRecord(
                question_id=question_id,
                question=str(e0["question"]),
                relevant_document_ids=relevant_document_ids,
                dense_candidate_document_ids=dense_candidate_document_ids,
                e1_document_ranking=e1_document_ranking,
                residual_bucket=residual_bucket,
            )
        )

    return records
