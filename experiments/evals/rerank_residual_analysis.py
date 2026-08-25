from __future__ import annotations

import hashlib
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


def select_candidate_miss_review_sample(
    records: Sequence[RerankResidualRecord],
    *,
    sample_size: int = 30,
) -> list[RerankResidualRecord]:
    if sample_size <= 0:
        return []

    misses = [
        record
        for record in records
        if record.residual_bucket == "dense_candidate_miss_top100"
    ]
    return sorted(
        misses,
        key=lambda record: hashlib.sha256(
            record.question_id.encode("utf-8")
        ).hexdigest(),
    )[: min(sample_size, len(misses))]


def summarize_residual_records(
    records: Sequence[RerankResidualRecord],
) -> dict[str, Any]:
    ordered_records = list(records)
    query_count = len(ordered_records)
    bucket_counts = {
        bucket: sum(record.residual_bucket == bucket for record in ordered_records)
        for bucket in (
            "dense_candidate_miss_top100",
            "rerank_residual_top20",
            "resolved_top20",
        )
    }
    bucket_rates = {
        bucket: count / query_count if query_count else 0.0
        for bucket, count in bucket_counts.items()
    }
    return {
        "query_count": query_count,
        "bucket_counts": bucket_counts,
        "bucket_rates": bucket_rates,
        "dense_candidate_miss_count": bucket_counts[
            "dense_candidate_miss_top100"
        ],
    }


def _document_excerpt(
    document_id: str,
    *,
    documents_by_id: Mapping[str, str],
    excerpt_chars: int,
) -> dict[str, str]:
    return {
        "document_id": document_id,
        "text_excerpt": documents_by_id[document_id][:excerpt_chars],
    }


def build_candidate_miss_review_rows(
    records: Sequence[RerankResidualRecord],
    *,
    documents_by_id: Mapping[str, str],
    excerpt_chars: int = 600,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "question_id": record.question_id,
                "question": record.question,
                "gold_documents": [
                    _document_excerpt(
                        document_id,
                        documents_by_id=documents_by_id,
                        excerpt_chars=excerpt_chars,
                    )
                    for document_id in record.relevant_document_ids
                ],
                "dense_top5": [
                    _document_excerpt(
                        document_id,
                        documents_by_id=documents_by_id,
                        excerpt_chars=excerpt_chars,
                    )
                    for document_id in record.dense_candidate_document_ids[:5]
                ],
                "e1_top5": [
                    _document_excerpt(
                        document_id,
                        documents_by_id=documents_by_id,
                        excerpt_chars=excerpt_chars,
                    )
                    for document_id in record.e1_document_ranking[:5]
                ],
                "manual_label": "",
                "notes": "",
            }
        )
    return rows
