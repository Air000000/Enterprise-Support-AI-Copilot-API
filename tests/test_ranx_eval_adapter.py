from types import SimpleNamespace

from experiments.evals.ir.ranx_adapter import (
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
