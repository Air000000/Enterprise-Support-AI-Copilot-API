from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.evals.adapters.techqa import (
    TechQADocument,
    TechQAGenerationCase,
    build_techqa_generation_cases,
)
from experiments.evals.retrievers.bm25_techqa_chunks import (
    TechQAChunk,
    TechQAChunkBM25Retriever,
    build_techqa_chunks,
)

DEFAULT_EVIDENCE_SAMPLE_SEED = "techqa-evidence-audit-v1"
DEFAULT_ANNOTATION_BM25_TOP_K = 3


@dataclass(frozen=True)
class EvidenceAnnotationPackCase:
    question_id: str
    question: str
    gold_answer: str
    gold_document_id: str
    candidate_chunks: tuple[TechQAChunk, ...]


def _stable_sample_key(
    question_id: str,
    *,
    seed: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{question_id}".encode("utf-8")
    ).hexdigest()
    return digest, question_id


def select_representative_train_cases(
    cases: Sequence[TechQAGenerationCase],
    *,
    sample_size: int = 60,
    seed: str = DEFAULT_EVIDENCE_SAMPLE_SEED,
) -> list[TechQAGenerationCase]:
    eligible = [
        case
        for case in cases
        if case.split == "train" and case.answerable
    ]

    ordered = sorted(
        eligible,
        key=lambda case: _stable_sample_key(
            case.question_id,
            seed=seed,
        ),
    )

    return ordered[:sample_size]


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _local_bm25_anchor_indexes(
    chunks: Sequence[TechQAChunk],
    *,
    gold_answer: str,
) -> list[int]:
    if not chunks:
        return []

    retriever = TechQAChunkBM25Retriever(chunks)
    anchors = retriever.search(
        gold_answer,
        top_k=min(DEFAULT_ANNOTATION_BM25_TOP_K, len(chunks)),
    )

    position_by_chunk_id = {
        chunk.chunk_id: index
        for index, chunk in enumerate(chunks)
    }

    return [
        position_by_chunk_id[chunk.chunk_id]
        for chunk in anchors
    ]


def build_annotation_candidates(
    chunks: Sequence[TechQAChunk],
    *,
    gold_answer: str,
) -> list[TechQAChunk]:
    normalized_answer = _normalize_whitespace(gold_answer)

    anchor_indexes = [
        index
        for index, chunk in enumerate(chunks)
        if normalized_answer
        in _normalize_whitespace(chunk.content)
    ]

    if not anchor_indexes:
        anchor_indexes = _local_bm25_anchor_indexes(
            chunks,
            gold_answer=gold_answer,
        )

    candidate_indexes: set[int] = set()

    for anchor_index in anchor_indexes:
        for index in (
            anchor_index - 1,
            anchor_index,
            anchor_index + 1,
        ):
            if 0 <= index < len(chunks):
                candidate_indexes.add(index)

    return [
        chunks[index]
        for index in sorted(candidate_indexes)
    ]


def build_annotation_pack_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = 60,
    seed: str = DEFAULT_EVIDENCE_SAMPLE_SEED,
) -> list[EvidenceAnnotationPackCase]:
    rows_by_id = {
        str(row["id"]): row
        for row in rows
    }

    generation_cases = build_techqa_generation_cases(rows)
    selected_cases = select_representative_train_cases(
        generation_cases,
        sample_size=sample_size,
        seed=seed,
    )

    pack_cases: list[EvidenceAnnotationPackCase] = []

    for case in selected_cases:
        row = rows_by_id[case.question_id]
        context = row["contexts"][0]

        gold_document_id = str(context["filename"])
        gold_document = TechQADocument(
            document_id=gold_document_id,
            text=str(context["text"]),
        )

        chunks = build_techqa_chunks([gold_document])
        candidates = build_annotation_candidates(
            chunks,
            gold_answer=case.gold_answer,
        )

        pack_cases.append(
            EvidenceAnnotationPackCase(
                question_id=case.question_id,
                question=case.question,
                gold_answer=case.gold_answer,
                gold_document_id=gold_document_id,
                candidate_chunks=tuple(candidates),
            )
        )

    return pack_cases


def build_annotation_pack_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = 60,
    seed: str = DEFAULT_EVIDENCE_SAMPLE_SEED,
) -> str:
    rows_by_id = {
        str(row["id"]): row
        for row in rows
    }

    pack_cases = build_annotation_pack_cases(
        rows,
        sample_size=sample_size,
        seed=seed,
    )

    sections: list[str] = []

    for case_number, case in enumerate(pack_cases, start=1):
        row = rows_by_id[case.question_id]
        context = row["contexts"][0]
        gold_document_text = str(context["text"])

        all_chunks = build_techqa_chunks(
            [
                TechQADocument(
                    document_id=case.gold_document_id,
                    text=gold_document_text,
                )
            ]
        )

        normalized_answer = _normalize_whitespace(case.gold_answer)
        localization_mode = (
            "exact_anchor"
            if any(
                normalized_answer
                in _normalize_whitespace(chunk.content)
                for chunk in all_chunks
            )
            else "local_bm25"
        )

        lines = [
            f"## {case_number:02d} {case.question_id}",
            "",
            "Question:",
            case.question,
            "",
            "Gold answer:",
            case.gold_answer,
            "",
            "Gold document:",
            case.gold_document_id,
            "",
            "Localization:",
            localization_mode,
            "",
        ]

        for candidate_number, chunk in enumerate(
            case.candidate_chunks,
            start=1,
        ):
            lines.extend(
                [
                    f"### Candidate {candidate_number}",
                    "",
                    f"chunk_id: {chunk.chunk_id}",
                    "",
                    chunk.content,
                    "",
                    "evidence_label: ___",
                    "",
                ]
            )

        lines.extend(
            [
                "<details>",
                "<summary>Full gold document</summary>",
                "",
                gold_document_text,
                "",
                "</details>",
                "",
                "questionable_gold: ___",
                "",
                "notes:",
                "",
                "---",
                "",
            ]
        )

        sections.append("\n".join(lines))

    return "\n".join(sections)


def answer_evidence_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    evidence_labels: Mapping[str, int],
    *,
    k: int,
) -> int:
    return int(
        any(
            evidence_labels.get(chunk_id) == 2
            for chunk_id in retrieved_chunk_ids[:k]
        )
    )


def answer_evidence_reciprocal_rank_at_k(
    retrieved_chunk_ids: Sequence[str],
    evidence_labels: Mapping[str, int],
    *,
    k: int,
) -> float:
    for rank, chunk_id in enumerate(
        retrieved_chunk_ids[:k],
        start=1,
    ):
        if evidence_labels.get(chunk_id) == 2:
            return 1.0 / rank

    return 0.0


def useful_evidence_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    evidence_labels: Mapping[str, int],
    *,
    k: int,
) -> int:
    return int(
        any(
            evidence_labels.get(chunk_id) in (1, 2)
            for chunk_id in retrieved_chunk_ids[:k]
        )
    )


def gold_doc_hit_but_evidence_miss_at_k(
    retrieved_chunk_ids: Sequence[str],
    evidence_labels: Mapping[str, int],
    *,
    gold_document_id: str,
    k: int,
) -> int:
    top_k_chunk_ids = retrieved_chunk_ids[:k]

    gold_doc_hit = any(
        chunk_id.startswith(f"{gold_document_id}_chunk_")
        for chunk_id in top_k_chunk_ids
    )

    answer_evidence_hit = any(
        evidence_labels.get(chunk_id) == 2
        for chunk_id in top_k_chunk_ids
    )

    return int(gold_doc_hit and not answer_evidence_hit)


def build_evidence_summary(
    retrieval_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    hit_k: int,
    mrr_k: int,
) -> dict[str, int | float]:
    retrieval_by_question_id = {
        str(row["question_id"]): row
        for row in retrieval_rows
    }

    questionable_gold_count = sum(
        bool(row["questionable_gold"])
        for row in label_rows
    )

    evaluated_rows = [
        row
        for row in label_rows
        if not bool(row["questionable_gold"])
    ]

    answer_hits: list[int] = []
    answer_rrs: list[float] = []
    useful_hits: list[int] = []
    gold_doc_hits: list[int] = []
    gold_doc_evidence_misses: list[int] = []

    for label_row in evaluated_rows:
        question_id = str(label_row["question_id"])
        retrieval_row = retrieval_by_question_id[question_id]

        retrieved_chunk_ids = list(retrieval_row["raw_chunk_ids"])
        evidence_labels = {
            str(candidate["chunk_id"]): int(candidate["evidence_label"])
            for candidate in label_row["candidate_labels"]
        }
        gold_document_id = str(
            retrieval_row["relevant_document_ids"][0]
        )

        answer_hits.append(
            answer_evidence_hit_at_k(
                retrieved_chunk_ids,
                evidence_labels,
                k=hit_k,
            )
        )
        answer_rrs.append(
            answer_evidence_reciprocal_rank_at_k(
                retrieved_chunk_ids,
                evidence_labels,
                k=mrr_k,
            )
        )
        useful_hits.append(
            useful_evidence_hit_at_k(
                retrieved_chunk_ids,
                evidence_labels,
                k=hit_k,
            )
        )
        gold_doc_hits.append(
            int(
                any(
                    chunk_id.startswith(
                        f"{gold_document_id}_chunk_"
                    )
                    for chunk_id
                    in retrieved_chunk_ids[:hit_k]
                )
            )
        )
        gold_doc_evidence_misses.append(
            gold_doc_hit_but_evidence_miss_at_k(
                retrieved_chunk_ids,
                evidence_labels,
                gold_document_id=gold_document_id,
                k=hit_k,
            )
        )

    evaluated_query_count = len(evaluated_rows)

    return {
        "labeled_query_count": len(label_rows),
        "evaluated_query_count": evaluated_query_count,
        "questionable_gold_count": questionable_gold_count,
        "answer_evidence_hit_rate": (
            sum(answer_hits) / evaluated_query_count
        ),
        "answer_evidence_mrr": (
            sum(answer_rrs) / evaluated_query_count
        ),
        "useful_evidence_hit_rate": (
            sum(useful_hits) / evaluated_query_count
        ),
        "gold_doc_hit_but_evidence_miss_rate": (
            sum(gold_doc_evidence_misses)
            / sum(gold_doc_hits)
            if sum(gold_doc_hits)
            else 0.0
        ),
    }
