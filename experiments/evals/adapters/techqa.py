from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

TechQASplit = Literal["train", "dev"]


@dataclass(frozen=True)
class TechQADocument:
    document_id: str
    text: str


@dataclass(frozen=True)
class TechQARetrievalCase:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    split: TechQASplit


@dataclass(frozen=True)
class TechQAGenerationCase:
    question_id: str
    question: str
    gold_answer: str
    answerable: bool
    split: TechQASplit


def recover_techqa_split(question_id: str) -> TechQASplit:
    if question_id.startswith("TRAIN_"):
        return "train"
    if question_id.startswith("DEV_"):
        return "dev"
    raise ValueError(f"Unsupported TechQA question id: {question_id}")


def build_techqa_documents(rows: Iterable[Mapping[str, Any]]) -> list[TechQADocument]:
    documents_by_id: dict[str, str] = {}

    for row in rows:
        document_id = str(row["_id"])
        text = str(row["text"])

        existing_text = documents_by_id.get(document_id)
        if existing_text is not None and existing_text != text:
            raise ValueError(f"Conflicting text for TechQA document: {document_id}")

        documents_by_id[document_id] = text

    return [
        TechQADocument(document_id=document_id, text=documents_by_id[document_id])
        for document_id in sorted(documents_by_id)
    ]


def build_qrels_by_query(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    qrels_by_query: dict[str, dict[str, int]] = {}

    for row in rows:
        question_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        relevance = int(row["score"])
        qrels_by_query.setdefault(question_id, {})[document_id] = relevance

    return {
        question_id: {
            document_id: qrels_by_query[question_id][document_id]
            for document_id in sorted(qrels_by_query[question_id])
        }
        for question_id in sorted(qrels_by_query)
    }


def build_techqa_retrieval_cases(
    query_rows: Iterable[Mapping[str, Any]],
    qrels_by_query: Mapping[str, Mapping[str, int]],
) -> list[TechQARetrievalCase]:
    cases: list[TechQARetrievalCase] = []

    for row in query_rows:
        question_id = str(row["_id"])
        relevant_documents = qrels_by_query[question_id]

        cases.append(
            TechQARetrievalCase(
                question_id=question_id,
                question=str(row["text"]),
                relevant_document_ids=tuple(sorted(relevant_documents)),
                split=recover_techqa_split(question_id),
            )
        )

    return sorted(cases, key=lambda case: case.question_id)


def build_techqa_generation_cases(
    rows: Iterable[Mapping[str, Any]],
) -> list[TechQAGenerationCase]:
    cases = [
        TechQAGenerationCase(
            question_id=str(row["id"]),
            question=str(row["question"]),
            gold_answer=str(row["answer"]),
            answerable=not bool(row["is_impossible"]),
            split=recover_techqa_split(str(row["id"])),
        )
        for row in rows
    ]

    return sorted(cases, key=lambda case: case.question_id)
