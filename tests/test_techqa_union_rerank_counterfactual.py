from importlib import import_module

import pytest


def _load_module():
    try:
        return import_module(
            "experiments.evals.eval_techqa_union_rerank_counterfactual"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "R2 union-rerank counterfactual evaluator is not implemented yet"
        )


def test_union_candidate_ids_preserve_dense_order_then_append_unseen_bm25():
    module = _load_module()

    dense_ids = (
        "d0",
        "shared0",
        "d1",
        "shared1",
    )
    bm25_ids = (
        "b0",
        "shared1",
        "b1",
        "shared0",
    )

    result = module.build_union_candidate_ids(
        dense_ids,
        bm25_ids,
    )

    assert result == (
        "d0",
        "shared0",
        "d1",
        "shared1",
        "b0",
        "b1",
    )


def test_build_union_snapshot_record_materializes_full_union_candidates():
    module = _load_module()

    dense_ids = tuple(
        ["shared", "d1", "d2"]
        + [f"d{i}" for i in range(3, 100)]
    )
    bm25_ids = tuple(
        ["shared", "b1", "b2"]
        + [f"b{i}" for i in range(3, 100)]
    )

    all_ids = set(dense_ids) | set(bm25_ids)

    chunks_by_id = {
        chunk_id: module.TechQAChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            chunk_index=0,
            content=f"content for {chunk_id}",
        )
        for chunk_id in all_ids
    }

    frozen_record = module.HybridSnapshotRecord(
        question_id="TRAIN_Q1",
        question="technical support query",
        relevant_document_ids=("doc-d1",),
        dense_chunk_ids=dense_ids,
        bm25_chunk_ids=bm25_ids,
        fused_candidates=(),
    )

    result = module.build_union_snapshot_record(
        frozen_record,
        chunks_by_id=chunks_by_id,
    )

    assert result.question_id == "TRAIN_Q1"
    assert result.question == "technical support query"
    assert result.relevant_document_ids == ("doc-d1",)

    assert result.dense_chunk_ids == dense_ids
    assert result.bm25_chunk_ids == bm25_ids

    union_ids = tuple(
        candidate.chunk_id
        for candidate in result.union_candidates
    )

    assert len(union_ids) == 199
    assert union_ids[:3] == (
        "shared",
        "d1",
        "d2",
    )
    assert union_ids[100:103] == (
        "b1",
        "b2",
        "b3",
    )

    assert result.union_candidates[0].content == "content for shared"



def test_build_union_snapshot_record_rejects_non_train_query():
    module = _load_module()

    chunk = module.TechQAChunk(
        chunk_id="c0",
        document_id="doc-c0",
        chunk_index=0,
        content="content",
    )

    frozen_record = module.HybridSnapshotRecord(
        question_id="DEV_Q1",
        question="technical support query",
        relevant_document_ids=("doc-c0",),
        dense_chunk_ids=("c0",),
        bm25_chunk_ids=("c0",),
        fused_candidates=(),
    )

    with pytest.raises(
        RuntimeError,
        match="R2 requires TRAIN-only input",
    ):
        module.build_union_snapshot_record(
            frozen_record,
            chunks_by_id={"c0": chunk},
        )


@pytest.mark.parametrize(
    ("dense_ids", "bm25_ids", "expected_message"),
    [
        (
            tuple(f"d{i}" for i in range(99)),
            tuple(f"b{i}" for i in range(100)),
            "Dense candidate ranking must contain exactly 100 unique chunks",
        ),
        (
            tuple(["d0"] * 2 + [f"d{i}" for i in range(1, 99)]),
            tuple(f"b{i}" for i in range(100)),
            "Dense candidate ranking must contain exactly 100 unique chunks",
        ),
        (
            tuple(f"d{i}" for i in range(100)),
            tuple(f"b{i}" for i in range(99)),
            "BM25 candidate ranking must contain exactly 100 unique chunks",
        ),
        (
            tuple(f"d{i}" for i in range(100)),
            tuple(["b0"] * 2 + [f"b{i}" for i in range(1, 99)]),
            "BM25 candidate ranking must contain exactly 100 unique chunks",
        ),
    ],
)
def test_build_union_snapshot_record_requires_exact_unique_source_top100(
    dense_ids,
    bm25_ids,
    expected_message,
):
    module = _load_module()

    all_ids = set(dense_ids) | set(bm25_ids)

    chunks_by_id = {
        chunk_id: module.TechQAChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            chunk_index=0,
            content=f"content for {chunk_id}",
        )
        for chunk_id in all_ids
    }

    frozen_record = module.HybridSnapshotRecord(
        question_id="TRAIN_Q1",
        question="technical support query",
        relevant_document_ids=("doc-d0",),
        dense_chunk_ids=dense_ids,
        bm25_chunk_ids=bm25_ids,
        fused_candidates=(),
    )

    with pytest.raises(
        RuntimeError,
        match=expected_message,
    ):
        module.build_union_snapshot_record(
            frozen_record,
            chunks_by_id=chunks_by_id,
        )


def test_rerank_union_snapshot_record_sends_complete_union_to_same_reranker():
    from experiments.evals.rerankers.qwen3_reranker import (
        RerankedCandidate,
    )

    module = _load_module()

    candidates = tuple(
        module.HybridCandidate(
            chunk_id=f"c{i}",
            document_id=(
                "shared-doc"
                if i < 2
                else f"doc-{i}"
            ),
            content=f"content {i}",
        )
        for i in range(199)
    )

    record = module.UnionSnapshotRecord(
        question_id="TRAIN_Q1",
        question="technical support query   ",
        relevant_document_ids=("shared-doc",),
        dense_chunk_ids=tuple(
            f"d{i}"
            for i in range(100)
        ),
        bm25_chunk_ids=tuple(
            f"b{i}"
            for i in range(100)
        ),
        union_candidates=candidates,
    )

    seen = {}

    class FakeResult:
        def __init__(self):
            self.results = tuple(
                RerankedCandidate(
                    chunk_id=candidates[index].chunk_id,
                    document_id=candidates[index].document_id,
                    content=candidates[index].content,
                    original_index=index,
                    relevance_score=float(199 - index),
                )
                for index in reversed(range(199))
            )
            self.request_id = "req-r2"
            self.total_tokens = 45678

    def fake_reranker(query, rerank_candidates):
        seen["query"] = query
        seen["chunk_ids"] = tuple(
            candidate.chunk_id
            for candidate in rerank_candidates
        )
        return FakeResult()

    clock = iter(
        [10.0, 10.125]
    ).__next__

    result = module.rerank_union_snapshot_record(
        record,
        reranker=fake_reranker,
        clock=clock,
    )

    assert seen["query"] == "technical support query"

    assert seen["chunk_ids"] == tuple(
        f"c{i}"
        for i in range(199)
    )

    assert result.union_chunk_ids == tuple(
        f"c{i}"
        for i in range(199)
    )

    assert result.reranked_chunk_ids[0] == "c198"
    assert len(result.reranked_chunk_ids) == 199

    assert result.document_ranking[0] == "doc-198"
    assert result.request_id == "req-r2"
    assert result.total_tokens == 45678
    assert result.rerank_latency_ms == 125.0


def test_frozen_r2_cohort_contract_is_exactly_23_unique_train_queries():
    module = _load_module()

    contract = module.build_frozen_r2_cohort_contract()

    expected_fusion_loss_ids = (
        "TRAIN_Q005",
        "TRAIN_Q011",
        "TRAIN_Q098",
        "TRAIN_Q122",
        "TRAIN_Q130",
        "TRAIN_Q135",
        "TRAIN_Q137",
        "TRAIN_Q141",
        "TRAIN_Q207",
        "TRAIN_Q261",
        "TRAIN_Q318",
        "TRAIN_Q319",
        "TRAIN_Q328",
        "TRAIN_Q343",
        "TRAIN_Q372",
        "TRAIN_Q460",
        "TRAIN_Q497",
        "TRAIN_Q543",
        "TRAIN_Q565",
    )

    expected_evidence_loss_ids = (
        "TRAIN_Q045",
        "TRAIN_Q122",
        "TRAIN_Q152",
        "TRAIN_Q524",
        "TRAIN_Q579",
    )

    expected_query_ids = (
        *expected_fusion_loss_ids,
        "TRAIN_Q045",
        "TRAIN_Q152",
        "TRAIN_Q524",
        "TRAIN_Q579",
    )

    assert contract["fusion_loss_ids"] == expected_fusion_loss_ids
    assert contract["evidence_loss_ids"] == expected_evidence_loss_ids

    assert contract["overlap_ids"] == (
        "TRAIN_Q122",
    )

    assert contract["query_ids"] == expected_query_ids
    assert len(contract["query_ids"]) == 23
    assert len(set(contract["query_ids"])) == 23

    assert all(
        question_id.startswith("TRAIN_")
        for question_id in contract["query_ids"]
    )

    assert contract["max_provider_calls"] == 23
    assert contract["provider_calls"] == 0
    assert contract["dev_artifact_opened"] is False

    assert contract["selection_policy"] == (
        "frozen_two_cohort_no_posthoc_selection"
    )





def test_select_frozen_r2_paid_records_admits_only_the_23_frozen_queries():
    module = _load_module()

    contract = module.build_frozen_r2_cohort_contract()
    frozen_ids = contract["query_ids"]

    def make_record(question_id):
        return module.UnionSnapshotRecord(
            question_id=question_id,
            question=f"question for {question_id}",
            relevant_document_ids=(),
            dense_chunk_ids=(),
            bm25_chunk_ids=(),
            union_candidates=(),
        )

    source_records = [
        make_record("TRAIN_Q999"),
        *(
            make_record(question_id)
            for question_id in reversed(frozen_ids)
        ),
        make_record("TRAIN_Q888"),
    ]

    selected = module.select_frozen_r2_paid_records(
        source_records
    )

    selected_ids = tuple(
        record.question_id
        for record in selected
    )

    assert selected_ids == frozen_ids
    assert len(selected_ids) == 23
    assert len(set(selected_ids)) == 23

    assert "TRAIN_Q999" not in selected_ids
    assert "TRAIN_Q888" not in selected_ids

    assert len(selected) <= contract["max_provider_calls"]



def test_run_resumable_r2_paid_eval_calls_only_frozen_queries_once(
    tmp_path,
):
    module = _load_module()

    contract = module.build_frozen_r2_cohort_contract()
    frozen_ids = contract["query_ids"]

    def make_record(question_id):
        return module.UnionSnapshotRecord(
            question_id=question_id,
            question=f"question for {question_id}",
            relevant_document_ids=(
                f"gold-{question_id}",
            ),
            dense_chunk_ids=(),
            bm25_chunk_ids=(),
            union_candidates=(),
        )

    records = [
        make_record("TRAIN_Q999"),
        *(
            make_record(question_id)
            for question_id in reversed(frozen_ids)
        ),
        make_record("TRAIN_Q888"),
    ]

    provider_calls = []

    def fake_evaluator(record):
        provider_calls.append(
            record.question_id
        )

        return module.UnionRerankResult(
            question_id=record.question_id,
            relevant_document_ids=(
                record.relevant_document_ids
            ),
            union_chunk_ids=(),
            reranked_chunk_ids=(),
            reranked_document_ids=(),
            document_ranking=(),
            rerank_latency_ms=1.0,
            request_id=(
                f"req-{record.question_id}"
            ),
            total_tokens=100,
        )

    checkpoint_path = (
        tmp_path / "paid_checkpoint.jsonl"
    )

    inflight_path = (
        tmp_path / "paid_inflight.json"
    )

    summary = module.run_resumable_r2_paid_eval(
        records,
        evaluator=fake_evaluator,
        checkpoint_path=checkpoint_path,
        inflight_path=inflight_path,
        max_provider_calls=23,
        token_stop_threshold=2_760_000,
    )

    assert tuple(provider_calls) == frozen_ids

    assert len(provider_calls) == 23
    assert len(set(provider_calls)) == 23

    assert summary.completed_query_count == 23
    assert summary.provider_calls == 23
    assert summary.provider_total_tokens == 2300
    assert summary.stopped_reason is None

    checkpoint_lines = [
        line
        for line in checkpoint_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(checkpoint_lines) == 23

    assert not inflight_path.exists()
