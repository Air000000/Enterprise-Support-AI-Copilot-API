from types import SimpleNamespace

from ranx import Qrels, Run, evaluate

from experiments.evals.ir.ranx_adapter import (
    build_ranx_qrels,
    collapse_chunk_results_to_document_ranking,
)


def test_collapse_keeps_first_document_occurrence():
    results = [
        SimpleNamespace(document_id="a"),
        SimpleNamespace(document_id="a"),
        SimpleNamespace(document_id="a"),
        SimpleNamespace(document_id="b"),
        SimpleNamespace(document_id="c"),
        SimpleNamespace(document_id="b"),
    ]

    assert collapse_chunk_results_to_document_ranking(results) == [
        "a",
        "b",
        "c",
    ]


def test_build_ranx_qrels_from_rows():
    qrels = build_ranx_qrels(
        [
            {"query-id": "q1", "corpus-id": "d1", "score": 1},
            {"query-id": "q2", "corpus-id": "d2", "score": 1},
        ]
    )

    assert isinstance(qrels, Qrels)

    perfect_run = Run(
        {
            "q1": {"d1": 1.0},
            "q2": {"d2": 1.0},
        }
    )
    assert evaluate(qrels, perfect_run, "mrr") == 1.0


def test_build_ranx_qrels_preserves_multiple_relevant_documents():
    qrels = build_ranx_qrels(
        [
            {"query-id": "q1", "corpus-id": "d1", "score": 1},
            {"query-id": "q1", "corpus-id": "d2", "score": 1},
        ]
    )

    partial_run = Run({"q1": {"d1": 1.0}})
    assert evaluate(qrels, partial_run, "recall@2") == 0.5
