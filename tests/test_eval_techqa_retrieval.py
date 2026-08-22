from types import SimpleNamespace

import pytest

from experiments.evals.eval_techqa_retrieval import (
    DEFAULT_CANDIDATE_CHUNK_K,
    DEFAULT_DOCUMENT_TOP_K,
    evaluate_techqa_retrieval_cases,
    load_frozen_techqa_retrieval_cases,
)


def test_load_frozen_retrieval_cases_uses_manifest_identity_and_preserves_split():
    observed: list[tuple[str, str, str, str]] = []

    query_rows = [
        {
            "_id": f"TRAIN_Q{index:03d}",
            "text": f"train question {index}",
        }
        for index in range(450)
    ] + [
        {
            "_id": f"DEV_Q{index:03d}",
            "text": f"dev question {index}",
        }
        for index in range(160)
    ]
    qrel_rows = [
        {
            "query-id": row["_id"],
            "corpus-id": f"doc_{index:03d}",
            "score": 1,
        }
        for index, row in enumerate(query_rows)
    ]

    def fake_load_dataset(path, name, *, split, revision):
        observed.append((path, name, split, revision))
        if name == "queries":
            return query_rows
        if name == "default":
            return qrel_rows
        raise AssertionError(f"unexpected config: {name}")

    cases = load_frozen_techqa_retrieval_cases(dataset_loader=fake_load_dataset)

    assert observed == [
        (
            "bowang0911/TechQA-RAG-Eval",
            "queries",
            "train",
            "68323f8f191fd5df93e2b2673d79a5da3a805638",
        ),
        (
            "bowang0911/TechQA-RAG-Eval",
            "default",
            "train",
            "68323f8f191fd5df93e2b2673d79a5da3a805638",
        ),
    ]
    assert len(cases) == 610
    assert sum(case.split == "train" for case in cases) == 450
    assert sum(case.split == "dev" for case in cases) == 160


def test_e0_train_eval_uses_100_chunks_then_collapses_to_document_top_20():
    cases = load_frozen_techqa_retrieval_cases(
        dataset_loader=lambda path, name, *, split, revision: (
            [
                {"_id": "TRAIN_Q001", "text": "alpha question"},
                {"_id": "DEV_Q001", "text": "dev question"},
            ]
            if name == "queries"
            else [
                {"query-id": "TRAIN_Q001", "corpus-id": "doc_gold", "score": 1},
                {"query-id": "DEV_Q001", "corpus-id": "doc_dev", "score": 1},
            ]
        ),
        expected_query_count=None,
    )

    calls: list[tuple[str, int]] = []

    def fake_searcher(query: str, *, top_k: int):
        calls.append((query, top_k))
        return [
            SimpleNamespace(chunk_id="a0", document_id="doc_a", distance=0.01),
            SimpleNamespace(chunk_id="a1", document_id="doc_a", distance=0.02),
            SimpleNamespace(chunk_id="g0", document_id="doc_gold", distance=0.03),
        ] + [
            SimpleNamespace(
                chunk_id=f"x{index}",
                document_id=f"doc_{index:02d}",
                distance=0.10 + index / 1000,
            )
            for index in range(25)
        ]

    summary = evaluate_techqa_retrieval_cases(
        cases,
        searcher=fake_searcher,
        split="train",
        clock=iter([0.000, 0.010]).__next__,
    )

    assert DEFAULT_CANDIDATE_CHUNK_K == 100
    assert DEFAULT_DOCUMENT_TOP_K == 20
    assert calls == [("alpha question", 100)]
    assert summary.query_count == 1

    result = summary.results[0]
    assert result.question_id == "TRAIN_Q001"
    assert result.raw_chunk_ids[:3] == ("a0", "a1", "g0")
    assert result.raw_document_ids[:3] == ("doc_a", "doc_a", "doc_gold")
    assert result.document_ranking[:3] == ("doc_a", "doc_gold", "doc_00")
    assert len(result.document_ranking) == 20
    assert result.latency_ms == pytest.approx(10.0)


def test_e0_train_eval_reports_ranx_metrics_and_latency_percentiles():
    from experiments.evals.adapters.techqa import TechQARetrievalCase

    cases = [
        TechQARetrievalCase(
            question_id="TRAIN_Q001",
            question="question one",
            relevant_document_ids=("doc_gold_1",),
            split="train",
        ),
        TechQARetrievalCase(
            question_id="TRAIN_Q002",
            question="question two",
            relevant_document_ids=("doc_gold_2",),
            split="train",
        ),
        TechQARetrievalCase(
            question_id="DEV_Q001",
            question="held out",
            relevant_document_ids=("doc_dev",),
            split="dev",
        ),
    ]

    rankings = {
        "question one": [
            SimpleNamespace(chunk_id="a0", document_id="doc_other", distance=0.1),
            SimpleNamespace(chunk_id="g1", document_id="doc_gold_1", distance=0.2),
        ],
        "question two": [
            SimpleNamespace(chunk_id="g2", document_id="doc_gold_2", distance=0.1),
            SimpleNamespace(chunk_id="b0", document_id="doc_other", distance=0.2),
        ],
    }
    searched: list[str] = []

    def fake_searcher(query: str, *, top_k: int):
        searched.append(query)
        return rankings[query]

    clock = iter([0.000, 0.010, 0.010, 0.040]).__next__
    summary = evaluate_techqa_retrieval_cases(
        cases,
        searcher=fake_searcher,
        split="train",
        clock=clock,
    )

    assert searched == ["question one", "question two"]
    assert summary.query_count == 2
    assert summary.metrics == {
        "recall@5": 1.0,
        "recall@20": 1.0,
        "mrr@10": 0.75,
    }
    assert summary.latency_p50_ms == pytest.approx(20.0)
    assert summary.latency_p95_ms == pytest.approx(29.0)
