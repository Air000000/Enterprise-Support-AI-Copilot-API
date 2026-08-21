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


def test_build_ranx_qrels_preserves_relevance_labels():
    qrels = build_ranx_qrels(
        {
            "q1": {"d1": 1},
            "q2": {"d2": 2},
        }
    )

    assert isinstance(qrels, Qrels)
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_build_ranx_qrels_preserves_multiple_relevant_documents():
    qrels = build_ranx_qrels(
        {
            "q1": {
                "d1": 1,
                "d2": 1,
            }
        }
    )

    partial_run = Run({"q1": {"d1": 1.0}})
    assert evaluate(qrels, partial_run, "recall@2") == 0.5


def test_build_ranx_run_preserves_document_ranking():
    run = build_ranx_run("q1", ["d2", "d1", "d3"])
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
    run = Run(
        {
            "q1": {"d1": 2.0, "other": 1.0},
            "q2": {"other": 2.0, "d2": 1.0},
        }
    )

    assert evaluate_ir_run(qrels, run) == {
        "recall@5": 1.0,
        "recall@20": 1.0,
        "mrr@10": 0.75,
    }
