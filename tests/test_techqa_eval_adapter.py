import pytest

from experiments.evals.adapters.techqa import (
    TechQADocument,
    TechQAGenerationCase,
    TechQARetrievalCase,
    build_qrels_by_query,
    build_techqa_documents,
    build_techqa_generation_cases,
    build_techqa_retrieval_cases,
    recover_techqa_split,
)


def test_build_techqa_documents_deduplicates_same_document():
    documents = build_techqa_documents(
        [
            {"_id": "doc_b", "text": "Document B"},
            {"_id": "doc_a", "text": "Document A"},
            {"_id": "doc_b", "text": "Document B"},
        ]
    )

    assert documents == [
        TechQADocument(document_id="doc_a", text="Document A"),
        TechQADocument(document_id="doc_b", text="Document B"),
    ]


def test_build_techqa_documents_rejects_conflicting_document_text():
    with pytest.raises(ValueError, match="doc_a"):
        build_techqa_documents(
            [
                {"_id": "doc_a", "text": "first"},
                {"_id": "doc_a", "text": "different"},
            ]
        )


def test_build_qrels_and_retrieval_cases_preserve_gold_documents():
    qrels = build_qrels_by_query(
        [
            {"query-id": "TRAIN_Q001", "corpus-id": "doc_b", "score": 1},
            {"query-id": "TRAIN_Q001", "corpus-id": "doc_a", "score": 1},
            {"query-id": "DEV_Q002", "corpus-id": "doc_c", "score": 2},
        ]
    )

    assert qrels == {
        "DEV_Q002": {"doc_c": 2},
        "TRAIN_Q001": {"doc_a": 1, "doc_b": 1},
    }

    cases = build_techqa_retrieval_cases(
        [
            {"_id": "TRAIN_Q001", "text": "train question"},
            {"_id": "DEV_Q002", "text": "dev question"},
        ],
        qrels,
    )

    assert cases == [
        TechQARetrievalCase(
            question_id="DEV_Q002",
            question="dev question",
            relevant_document_ids=("doc_c",),
            split="dev",
        ),
        TechQARetrievalCase(
            question_id="TRAIN_Q001",
            question="train question",
            relevant_document_ids=("doc_a", "doc_b"),
            split="train",
        ),
    ]


def test_recover_techqa_split_uses_original_question_id_prefix():
    assert recover_techqa_split("TRAIN_Q001") == "train"
    assert recover_techqa_split("DEV_Q002") == "dev"

    with pytest.raises(ValueError, match="UNKNOWN_Q003"):
        recover_techqa_split("UNKNOWN_Q003")


def test_build_generation_cases_preserves_impossible_and_gold_answer():
    cases = build_techqa_generation_cases(
        [
            {
                "id": "TRAIN_Q001",
                "question": "answerable question",
                "answer": "gold answer",
                "is_impossible": False,
                "contexts": [{"filename": "gold.txt", "text": "gold context"}],
            },
            {
                "id": "DEV_Q002",
                "question": "impossible question",
                "answer": "",
                "is_impossible": True,
                "contexts": [{"filename": "must-not-be-corpus.txt", "text": "ignored"}],
            },
        ]
    )

    assert cases == [
        TechQAGenerationCase(
            question_id="DEV_Q002",
            question="impossible question",
            gold_answer="",
            answerable=False,
            split="dev",
        ),
        TechQAGenerationCase(
            question_id="TRAIN_Q001",
            question="answerable question",
            gold_answer="gold answer",
            answerable=True,
            split="train",
        ),
    ]
