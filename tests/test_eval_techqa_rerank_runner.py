import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import pytest

import experiments.evals.eval_techqa_rerank as rerank_eval


def _require(name: str):
    value = getattr(rerank_eval, name, None)
    if value is None:
        pytest.fail(f"experiments.evals.eval_techqa_rerank.{name} is not implemented yet")
    return value


def _result(
    question_id: str,
    *,
    relevant_document_id: str,
    document_ranking: tuple[str, ...],
    latency_ms: float = 10.0,
    total_tokens: int = 100,
):
    return rerank_eval.TechQARerankResult(
        question_id=question_id,
        relevant_document_ids=(relevant_document_id,),
        dense_chunk_ids=(f"{question_id}_dense_0",),
        reranked_chunk_ids=(f"{question_id}_rerank_0",),
        reranked_document_ids=document_ranking,
        document_ranking=document_ranking,
        rerank_latency_ms=latency_ms,
        request_id=f"req-{question_id}",
        total_tokens=total_tokens,
    )


def test_resumable_rerank_skips_checkpointed_records_and_appends_only_missing(tmp_path):
    runner = _require("run_resumable_rerank_eval")
    loader = _require("load_rerank_checkpoint")
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    existing = _result(
        "TRAIN_Q001",
        relevant_document_id="doc_gold_1",
        document_ranking=("doc_gold_1", "doc_other"),
    )
    checkpoint_path.write_text(
        json.dumps(asdict(existing), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records = [
        {"question_id": "TRAIN_Q001"},
        {"question_id": "TRAIN_Q002"},
    ]
    evaluated: list[str] = []

    def fake_evaluator(record):
        question_id = record["question_id"]
        evaluated.append(question_id)
        return _result(
            question_id,
            relevant_document_id="doc_gold_2",
            document_ranking=("doc_other", "doc_gold_2"),
        )

    summary = runner(
        records,
        evaluator=fake_evaluator,
        checkpoint_path=checkpoint_path,
    )

    assert evaluated == ["TRAIN_Q002"]
    assert summary.query_count == 2
    assert [result.question_id for result in summary.results] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]
    assert [result.question_id for result in loader(checkpoint_path)] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]


def test_resumable_rerank_persists_each_success_before_later_provider_failure(tmp_path):
    runner = _require("run_resumable_rerank_eval")
    loader = _require("load_rerank_checkpoint")
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    calls: list[str] = []

    def flaky_evaluator(record):
        question_id = record["question_id"]
        calls.append(question_id)
        if question_id == "TRAIN_Q002":
            raise RuntimeError("simulated rerank provider failure")
        return _result(
            question_id,
            relevant_document_id="doc_gold_1",
            document_ranking=("doc_gold_1",),
        )

    with pytest.raises(RuntimeError, match="simulated rerank provider failure"):
        runner(
            [
                {"question_id": "TRAIN_Q001"},
                {"question_id": "TRAIN_Q002"},
            ],
            evaluator=flaky_evaluator,
            checkpoint_path=checkpoint_path,
        )

    assert calls == ["TRAIN_Q001", "TRAIN_Q002"]
    assert [result.question_id for result in loader(checkpoint_path)] == ["TRAIN_Q001"]


def test_completed_rerank_checkpoint_rebuilds_metrics_and_reports_without_provider_calls(
    tmp_path,
):
    runner = _require("run_resumable_rerank_eval")
    writer = _require("write_rerank_reports")
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    results = [
        _result(
            "TRAIN_Q001",
            relevant_document_id="doc_gold_1",
            document_ranking=("doc_other", "doc_gold_1"),
            latency_ms=10.0,
            total_tokens=100,
        ),
        _result(
            "TRAIN_Q002",
            relevant_document_id="doc_gold_2",
            document_ranking=("doc_gold_2", "doc_other"),
            latency_ms=30.0,
            total_tokens=300,
        ),
    ]
    checkpoint_path.write_text(
        "".join(json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )

    def forbidden_evaluator(record):
        raise AssertionError("completed checkpoint must not call the rerank provider")

    summary = runner(
        [
            {"question_id": "TRAIN_Q001"},
            {"question_id": "TRAIN_Q002"},
        ],
        evaluator=forbidden_evaluator,
        checkpoint_path=checkpoint_path,
    )

    assert summary.metrics == {
        "recall@5": 1.0,
        "recall@20": 1.0,
        "mrr@10": 0.75,
    }
    assert summary.rerank_latency_p50_ms == pytest.approx(20.0)
    assert summary.rerank_latency_p95_ms == pytest.approx(29.0)
    assert summary.provider_total_tokens == 400

    report_dir = tmp_path / "e1_rerank"
    writer(summary, report_dir=report_dir)

    metrics = json.loads((report_dir / "train_metrics.json").read_text(encoding="utf-8"))
    result_lines = (report_dir / "train_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert metrics["query_count"] == 2
    assert metrics["metrics"] == summary.metrics
    assert metrics["rerank_latency_p50_ms"] == pytest.approx(20.0)
    assert metrics["rerank_latency_p95_ms"] == pytest.approx(29.0)
    assert metrics["provider_total_tokens"] == 400
    assert len(result_lines) == 2


def test_rerank_manifest_locks_frozen_e0_source_and_rejects_identity_drift(tmp_path):
    builder = _require("build_rerank_run_manifest")
    ensure = _require("ensure_rerank_run_manifest")

    e0_results_path = tmp_path / "train_results.jsonl"
    e0_results_path.write_text('{"question_id":"TRAIN_Q001"}\n', encoding="utf-8")
    e0_manifest_path = tmp_path / "train_manifest.json"
    e0_manifest_path.write_text(
        json.dumps({"run": "e0_dense", "candidate_chunk_k": 100}),
        encoding="utf-8",
    )

    expected = builder(
        e0_results_path=e0_results_path,
        e0_manifest_path=e0_manifest_path,
        project_sha="project-sha",
        created_at="2026-08-24T04:00:00+08:00",
    )

    assert expected["identity"]["run"] == "e1_rerank"
    assert expected["identity"]["split"] == "train"
    assert expected["identity"]["source_e0"] == {
        "results_sha256": hashlib.sha256(e0_results_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(e0_manifest_path.read_bytes()).hexdigest(),
    }
    assert expected["identity"]["reranker"] == {
        "model": "qwen3-rerank",
        "instruction": rerank_eval.DEFAULT_RERANK_INSTRUCTION,
        "candidate_chunk_k": 100,
        "query_normalization": "rstrip",
    }
    assert expected["identity"]["document_ranking"] == {
        "top_k": 20,
        "rule": "first occurrence of document_id after chunk rerank",
    }
    assert expected["provenance"]["project_sha"] == "project-sha"

    run_manifest_path = tmp_path / "train_manifest_e1.json"
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    ensure(
        expected,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
    )
    persisted = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert persisted["identity"] == expected["identity"]

    changed = deepcopy(expected)
    changed["identity"]["reranker"]["model"] = "different-model"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ensure(
            changed,
            run_manifest_path=run_manifest_path,
            checkpoint_path=checkpoint_path,
        )
