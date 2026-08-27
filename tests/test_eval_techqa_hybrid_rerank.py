from importlib import import_module

import pytest

from experiments.evals.rerankers.qwen3_reranker import RerankedCandidate


def _load_module():
    try:
        return import_module("experiments.evals.eval_techqa_hybrid_rerank")
    except ModuleNotFoundError:
        pytest.fail("R4 C1 hybrid rerank evaluator is not implemented yet")


def test_candidate_snapshot_contract_preserves_cross_source_rrf_credit():
    module = _load_module()

    dense_ids = [f"d{i}" for i in range(99)]
    bm25_ids = [f"b{i}" for i in range(99)]

    # The shared chunk is rank 50 in both sources.
    # Its two RRF contributions must beat a rank-1 chunk
    # that appears in only one source.
    dense_ids.insert(49, "shared")
    bm25_ids.insert(49, "shared")

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

    record = {
        "question_id": "TRAIN_Q1",
        "question": "shared technical query",
        "relevant_document_ids": ["doc-shared"],
        "raw_chunk_ids": dense_ids,
    }

    bm25_chunks = [
        chunks_by_id[chunk_id]
        for chunk_id in bm25_ids
    ]

    result = module.build_hybrid_snapshot_record(
        record,
        chunks_by_id=chunks_by_id,
        bm25_searcher=lambda query, top_k: bm25_chunks[:top_k],
    )

    assert len(result.dense_chunk_ids) == 100
    assert len(set(result.dense_chunk_ids)) == 100

    assert len(result.bm25_chunk_ids) == 100
    assert len(set(result.bm25_chunk_ids)) == 100

    fused_ids = [
        candidate.chunk_id
        for candidate in result.fused_candidates
    ]

    assert len(fused_ids) == 100
    assert len(set(fused_ids)) == 100

    # This is the key C1 contract:
    # cross-source overlap must retain both RRF rank contributions.
    assert fused_ids[0] == "shared"


def test_one_query_orchestration_preserves_frozen_candidates_and_provider_identity():
    module = _load_module()

    candidates = tuple(
        module.HybridCandidate(
            chunk_id=f"c{i}",
            document_id=(
                "shared-doc"
                if i < 2
                else f"doc{i}"
            ),
            content=f"content {i}",
        )
        for i in range(100)
    )

    record = module.HybridSnapshotRecord(
        question_id="TRAIN_Q1",
        question="technical query",
        relevant_document_ids=("shared-doc",),
        dense_chunk_ids=tuple(
            f"d{i}"
            for i in range(100)
        ),
        bm25_chunk_ids=tuple(
            f"b{i}"
            for i in range(100)
        ),
        fused_candidates=candidates,
    )

    class FakeResult:
        def __init__(self):
            self.results = tuple(
                RerankedCandidate(
                    chunk_id=candidates[index].chunk_id,
                    document_id=candidates[index].document_id,
                    content=candidates[index].content,
                    original_index=index,
                    relevance_score=float(100 - index),
                )
                for index in reversed(range(100))
            )
            self.request_id = "req-c1"
            self.total_tokens = 12345

    seen = {}

    def fake_reranker(query, rerank_candidates):
        seen["query"] = query
        seen["chunk_ids"] = [
            item.chunk_id
            for item in rerank_candidates
        ]
        return FakeResult()

    clock = iter([1.0, 1.25]).__next__

    result = module.rerank_snapshot_record(
        record,
        reranker=fake_reranker,
        clock=clock,
    )

    assert seen["query"] == "technical query"

    assert seen["chunk_ids"] == [
        f"c{i}"
        for i in range(100)
    ]

    assert len(result.reranked_chunk_ids) == 100

    assert result.document_ranking[0] == "doc99"

    # c0/c1 both map to shared-doc, so first-occurrence
    # document collapse yields 99 unique documents.
    assert len(result.document_ranking) == 99

    assert result.request_id == "req-c1"
    assert result.total_tokens == 12345
    assert result.rerank_latency_ms == 250.0


def test_paid_runner_safety_contract(tmp_path, monkeypatch):
    import hashlib
    import json
    from dataclasses import asdict

    module = _load_module()

    def make_record(
        question_id="TRAIN_Q1",
        *,
        candidate_count=100,
    ):
        candidates = tuple(
            module.HybridCandidate(
                chunk_id=f"{question_id}-c{i}",
                document_id=f"{question_id}-doc{i}",
                content=f"content {i}",
            )
            for i in range(candidate_count)
        )

        return module.HybridSnapshotRecord(
            question_id=question_id,
            question="technical query",
            relevant_document_ids=(
                f"{question_id}-doc0",
            ),
            dense_chunk_ids=tuple(
                f"{question_id}-d{i}"
                for i in range(100)
            ),
            bm25_chunk_ids=tuple(
                f"{question_id}-b{i}"
                for i in range(100)
            ),
            fused_candidates=candidates,
        )

    def write_snapshot(records, path):
        path.write_text(
            "".join(
                json.dumps(
                    asdict(record),
                    separators=(",", ":"),
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

        return hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    def write_manifest(path, snapshot_sha):
        path.write_text(
            json.dumps(
                {
                    "run": "r4_c1_hybrid_rerank",
                    "split": "train",
                    "query_count": 1,
                    "snapshot": {
                        "sha256": snapshot_sha,
                    },
                }
            ),
            encoding="utf-8",
        )

    def completed_result(
        question_id,
        *,
        total_tokens,
    ):
        return module.HybridRerankResult(
            question_id=question_id,
            relevant_document_ids=(
                f"{question_id}-doc0",
            ),
            fused_chunk_ids=tuple(
                f"{question_id}-c{i}"
                for i in range(100)
            ),
            reranked_chunk_ids=tuple(
                f"{question_id}-c{i}"
                for i in range(100)
            ),
            reranked_document_ids=tuple(
                f"{question_id}-doc{i}"
                for i in range(100)
            ),
            document_ranking=tuple(
                f"{question_id}-doc{i}"
                for i in range(100)
            ),
            rerank_latency_ms=100.0,
            request_id=f"req-{question_id}",
            total_tokens=total_tokens,
        )

    # --------------------------------------------------------
    # 1. Valid frozen snapshot loads.
    # --------------------------------------------------------

    snapshot_path = tmp_path / "snapshot.jsonl"
    manifest_path = tmp_path / "manifest.json"

    record = make_record()
    snapshot_sha = write_snapshot(
        [record],
        snapshot_path,
    )
    write_manifest(
        manifest_path,
        snapshot_sha,
    )

    loaded = module.load_validated_paid_snapshot(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        expected_count=1,
    )

    assert len(loaded) == 1
    assert loaded[0].question_id == "TRAIN_Q1"

    # --------------------------------------------------------
    # 2. Snapshot SHA mismatch must fail before paid work.
    # --------------------------------------------------------

    bad_manifest_path = tmp_path / "bad-manifest.json"
    write_manifest(
        bad_manifest_path,
        "0" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match="snapshot SHA256 mismatch",
    ):
        module.load_validated_paid_snapshot(
            snapshot_path=snapshot_path,
            manifest_path=bad_manifest_path,
            expected_count=1,
        )

    # --------------------------------------------------------
    # 3. Candidate count != 100 must fail before paid work.
    # --------------------------------------------------------

    bad_snapshot_path = tmp_path / "bad-snapshot.jsonl"
    bad_record = make_record(
        candidate_count=99,
    )
    bad_snapshot_sha = write_snapshot(
        [bad_record],
        bad_snapshot_path,
    )

    bad_candidate_manifest = (
        tmp_path / "bad-candidate-manifest.json"
    )
    write_manifest(
        bad_candidate_manifest,
        bad_snapshot_sha,
    )

    with pytest.raises(
        RuntimeError,
        match="exactly 100 fused candidates",
    ):
        module.load_validated_paid_snapshot(
            snapshot_path=bad_snapshot_path,
            manifest_path=bad_candidate_manifest,
            expected_count=1,
        )

    # --------------------------------------------------------
    # 4. DEV is forbidden.
    # --------------------------------------------------------

    dev_snapshot_path = tmp_path / "dev-snapshot.jsonl"
    dev_record = make_record(
        question_id="DEV_Q1",
    )
    dev_snapshot_sha = write_snapshot(
        [dev_record],
        dev_snapshot_path,
    )

    dev_manifest_path = tmp_path / "dev-manifest.json"
    write_manifest(
        dev_manifest_path,
        dev_snapshot_sha,
    )

    with pytest.raises(
        RuntimeError,
        match="TRAIN-only",
    ):
        module.load_validated_paid_snapshot(
            snapshot_path=dev_snapshot_path,
            manifest_path=dev_manifest_path,
            expected_count=1,
        )

    # --------------------------------------------------------
    # 5. Completed checkpoint query must never be replayed.
    # --------------------------------------------------------

    checkpoint_path = tmp_path / "checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            asdict(
                completed_result(
                    "TRAIN_Q1",
                    total_tokens=100,
                )
            )
        )
        + "\n",
        encoding="utf-8",
    )

    inflight_path = tmp_path / "inflight.json"

    evaluator_calls = []

    def must_not_call(record):
        evaluator_calls.append(record.question_id)
        raise AssertionError(
            "completed query was replayed"
        )

    summary = module.run_resumable_paid_eval(
        [record],
        evaluator=must_not_call,
        checkpoint_path=checkpoint_path,
        inflight_path=inflight_path,
    )

    assert evaluator_calls == []
    assert summary.completed_query_count == 1

    # --------------------------------------------------------
    # 6. Token threshold prevents the next provider request.
    # --------------------------------------------------------

    second_record = make_record(
        question_id="TRAIN_Q2",
    )

    threshold_checkpoint = (
        tmp_path / "threshold-checkpoint.jsonl"
    )
    threshold_checkpoint.write_text(
        json.dumps(
            asdict(
                completed_result(
                    "TRAIN_Q1",
                    total_tokens=13_500_000,
                )
            )
        )
        + "\n",
        encoding="utf-8",
    )

    threshold_calls = []

    def threshold_evaluator(record):
        threshold_calls.append(record.question_id)
        raise AssertionError(
            "provider called after token threshold"
        )

    summary = module.run_resumable_paid_eval(
        [record, second_record],
        evaluator=threshold_evaluator,
        checkpoint_path=threshold_checkpoint,
        inflight_path=(
            tmp_path / "threshold-inflight.json"
        ),
    )

    assert threshold_calls == []
    assert summary.provider_total_tokens == 13_500_000
    assert summary.stopped_reason == "token_stop_threshold"

    # --------------------------------------------------------
    # 7. Provider failure leaves inflight state durable.
    # --------------------------------------------------------

    error_checkpoint = (
        tmp_path / "error-checkpoint.jsonl"
    )
    error_inflight = (
        tmp_path / "error-inflight.json"
    )

    def failing_evaluator(record):
        raise RuntimeError("provider boom")

    with pytest.raises(
        RuntimeError,
        match="provider boom",
    ):
        module.run_resumable_paid_eval(
            [record],
            evaluator=failing_evaluator,
            checkpoint_path=error_checkpoint,
            inflight_path=error_inflight,
        )

    assert json.loads(
        error_inflight.read_text(
            encoding="utf-8",
        )
    )["question_id"] == "TRAIN_Q1"

    # --------------------------------------------------------
    # 8. Uncertain inflight query cannot auto-replay.
    # --------------------------------------------------------

    uncertain_checkpoint = (
        tmp_path / "uncertain-checkpoint.jsonl"
    )
    uncertain_inflight = (
        tmp_path / "uncertain-inflight.json"
    )

    uncertain_inflight.write_text(
        json.dumps(
            {
                "question_id": "TRAIN_Q1",
            }
        ),
        encoding="utf-8",
    )

    uncertain_calls = []

    def uncertain_evaluator(record):
        uncertain_calls.append(
            record.question_id
        )
        raise AssertionError(
            "uncertain request was replayed"
        )

    with pytest.raises(
        RuntimeError,
        match="inflight/uncertain",
    ):
        module.run_resumable_paid_eval(
            [record],
            evaluator=uncertain_evaluator,
            checkpoint_path=uncertain_checkpoint,
            inflight_path=uncertain_inflight,
        )

    assert uncertain_calls == []

    # --------------------------------------------------------
    # 9. Missing provider token usage must fail closed.
    # --------------------------------------------------------

    missing_token_checkpoint = (
        tmp_path / "missing-token-checkpoint.jsonl"
    )
    missing_token_inflight = (
        tmp_path / "missing-token-inflight.json"
    )

    def missing_token_evaluator(record):
        return completed_result(
            record.question_id,
            total_tokens=None,
        )

    with pytest.raises(
        RuntimeError,
        match="total_tokens",
    ):
        module.run_resumable_paid_eval(
            [record],
            evaluator=missing_token_evaluator,
            checkpoint_path=missing_token_checkpoint,
            inflight_path=missing_token_inflight,
        )

    assert json.loads(
        missing_token_inflight.read_text(
            encoding="utf-8",
        )
    )["question_id"] == "TRAIN_Q1"

    # --------------------------------------------------------
    # 10. qwen3-rerank SDK automatic retries must be disabled.
    # --------------------------------------------------------

    from experiments.evals.rerankers import (
        qwen3_reranker,
    )

    captured_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        qwen3_reranker,
        "OpenAI",
        FakeOpenAI,
    )
    monkeypatch.setenv(
        "DASHSCOPE_RERANK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "DASHSCOPE_RERANK_BASE_URL",
        "https://example.invalid/v1",
    )

    qwen3_reranker.get_rerank_client()

    assert captured_kwargs["max_retries"] == 0
