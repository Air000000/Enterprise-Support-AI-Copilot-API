from types import SimpleNamespace

from ranx import Qrels, Run, evaluate

from experiments.evals.ir.ranx_adapter import (
    build_ranx_qrels,
    build_ranx_run,
    collapse_chunk_results_to_document_ranking,
    evaluate_ir_run,
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


def test_build_ranx_run_preserves_document_ranking():
    run = build_ranx_run(
        {
            "q1": ["d2", "d1", "d3"],
        }
    )
    qrels = Qrels({"q1": {"d1": 1}})

    assert isinstance(run, Run)
    assert evaluate(qrels, run, "mrr@10") == 0.5


def test_evaluate_ir_run_uses_frozen_primary_metrics():
    qrels = Qrels(
        {
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        }
    )
    run = build_ranx_run(
        {
            "q1": ["d1", "other"],
            "q2": ["other", "d2"],
        }
    )

    assert evaluate_ir_run(qrels, run) == {
        "recall@5": 1.0,
        "recall@20": 1.0,
        "mrr@10": 0.75,
    }
