import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import experiments.evals.eval_techqa_retrieval as retrieval_eval
from experiments.evals.adapters.techqa import TechQARetrievalCase


def _result(question_id: str) -> retrieval_eval.TechQARetrievalResult:
    return retrieval_eval.TechQARetrievalResult(
        question_id=question_id,
        question=f"question {question_id}",
        relevant_document_ids=(f"doc_{question_id}",),
        raw_chunk_ids=(f"chunk_{question_id}",),
        raw_document_ids=(f"doc_{question_id}",),
        document_ranking=(f"doc_{question_id}",),
        latency_ms=10.0,
    )


def test_resumable_dev_retrieval_skips_checkpointed_queries_and_appends_missing(tmp_path):
    runner = getattr(retrieval_eval, "run_resumable_retrieval_eval", None)
    assert runner is not None, "run_resumable_retrieval_eval is not implemented yet"

    cases = [
        TechQARetrievalCase(
            question_id=f"DEV_Q00{index}",
            question=f"question {index}",
            relevant_document_ids=(f"doc_DEV_Q00{index}",),
            split="dev",
        )
        for index in range(1, 4)
    ]
    checkpoint_path = tmp_path / "dev_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(asdict(_result("DEV_Q001"))) + "\n",
        encoding="utf-8",
    )

    evaluated = []

    def fake_evaluator(case):
        evaluated.append(case.question_id)
        return _result(case.question_id)

    summary = runner(
        cases,
        evaluator=fake_evaluator,
        checkpoint_path=checkpoint_path,
        split="dev",
    )

    assert evaluated == ["DEV_Q002", "DEV_Q003"]
    assert summary.split == "dev"
    assert summary.query_count == 3
    assert summary.metrics == {
        "recall@5": 1.0,
        "recall@20": 1.0,
        "mrr@10": 1.0,
    }

    rows = [
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["question_id"] for row in rows] == [
        "DEV_Q001",
        "DEV_Q002",
        "DEV_Q003",
    ]


def _write_dataset_manifest(tmp_path):
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "corpus-sha",
                    "queries_sha256": "queries-sha",
                    "qrels_sha256": "qrels-sha",
                },
                "baseline_rag": {
                    "retriever": "chroma_dense",
                    "chunk_strategy": "paragraph_aware_character",
                    "chunk_size_chars": 800,
                    "chunk_overlap_chars": 120,
                    "min_chunk_size_chars": 150,
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_manifest


def test_dev_manifest_identity_freezes_retrieval_contract(monkeypatch, tmp_path):
    builder = getattr(retrieval_eval, "build_dev_retrieval_run_manifest", None)
    assert builder is not None, "build_dev_retrieval_run_manifest is not implemented yet"

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(
        retrieval_eval,
        "load_dotenv",
        lambda: False,
        raising=False,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path)

    manifest = builder(
        dataset_manifest_path=dataset_manifest,
        query_count=160,
        project_sha="abc123",
        created_at="2026-08-24T00:00:00+00:00",
    )

    identity = manifest["identity"]
    assert identity["run"] == "e0_dense"
    assert identity["split"] == "dev"
    assert identity["query_count"] == 160
    assert identity["candidate_chunk_k"] == 100
    assert identity["document_top_k"] == 20
    assert identity["query_normalization"] == "rstrip"
    assert identity["embedding_model"] == "text-embedding-v4"
    assert identity["document_ranking_rule"] == (
        "collapse chunk results by document_id, retaining the first occurrence"
    )
    assert identity["retrieval_dataset"] == {
        "repo": "bowang0911/TechQA-RAG-Eval",
        "revision": "frozen-revision",
        "corpus_sha256": "corpus-sha",
        "queries_sha256": "queries-sha",
        "qrels_sha256": "qrels-sha",
    }
    assert manifest["provenance"] == {
        "project_sha": "abc123",
        "created_at": "2026-08-24T00:00:00+00:00",
    }


def test_dev_manifest_loads_dotenv_before_resolving_embedding_model(
    monkeypatch,
    tmp_path,
):
    builder = retrieval_eval.build_dev_retrieval_run_manifest
    dataset_manifest = _write_dataset_manifest(tmp_path)
    observed = []

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    def fake_load_dotenv():
        observed.append("load_dotenv")
        monkeypatch.setenv("EMBEDDING_MODEL", "dotenv-embedding-model")
        return True

    monkeypatch.setattr(
        retrieval_eval,
        "load_dotenv",
        fake_load_dotenv,
        raising=False,
    )

    manifest = builder(
        dataset_manifest_path=dataset_manifest,
        query_count=160,
        project_sha="abc123",
        created_at="2026-08-24T00:00:00+00:00",
    )

    assert observed == ["load_dotenv"]
    assert manifest["identity"]["embedding_model"] == "dotenv-embedding-model"


def test_dev_manifest_rejects_embedding_model_drift(monkeypatch, tmp_path):
    builder = retrieval_eval.build_dev_retrieval_run_manifest
    ensure = retrieval_eval.ensure_dev_retrieval_run_manifest
    dataset_manifest = _write_dataset_manifest(tmp_path)
    run_manifest_path = tmp_path / "dev_manifest.json"
    checkpoint_path = tmp_path / "dev_checkpoint.jsonl"

    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    original = builder(
        dataset_manifest_path=dataset_manifest,
        query_count=160,
        project_sha="abc123",
        created_at="2026-08-24T00:00:00+00:00",
    )
    ensure(
        original,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
    )

    monkeypatch.setenv("EMBEDDING_MODEL", "different-embedding-model")
    changed = builder(
        dataset_manifest_path=dataset_manifest,
        query_count=160,
        project_sha="abc123",
        created_at="2026-08-24T00:01:00+00:00",
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        ensure(
            changed,
            run_manifest_path=run_manifest_path,
            checkpoint_path=checkpoint_path,
        )


def test_dev_main_locks_manifest_before_first_search(monkeypatch):
    runner = getattr(retrieval_eval, "run_resumable_retrieval_eval", None)
    assert runner is not None, "run_resumable_retrieval_eval is not implemented yet"

    cases = [
        TechQARetrievalCase(
            question_id=f"DEV_Q{index:03d}",
            question=f"dev question {index}",
            relevant_document_ids=(f"doc_{index:03d}",),
            split="dev",
        )
        for index in range(160)
    ]
    expected_manifest = {
        "identity": {"run": "e0_dense", "split": "dev"},
        "provenance": {},
    }
    order = []
    summary = SimpleNamespace(
        split="dev",
        query_count=160,
        metrics={"recall@5": 0.5, "recall@20": 0.7, "mrr@10": 0.4},
        latency_p50_ms=1000.0,
        latency_p95_ms=1500.0,
        results=(),
    )

    def fake_loader():
        order.append("load")
        return cases

    def fake_manifest_builder(*args, **kwargs):
        order.append("build_manifest")
        return expected_manifest

    def fake_manifest_lock(manifest, **kwargs):
        assert manifest is expected_manifest
        order.append("lock_manifest")

    def fake_single_case_evaluator(case):
        order.append("search")
        return _result(case.question_id)

    def fake_runner(input_cases, *, evaluator, checkpoint_path, split):
        assert input_cases is cases
        assert split == "dev"
        order.append("runner")
        evaluator(input_cases[0])
        return summary

    def fake_writer(input_summary):
        assert input_summary is summary
        order.append("writer")

    def fail_legacy_batch(*args, **kwargs):
        raise AssertionError("DEV main must use the resumable path")

    monkeypatch.setattr(
        retrieval_eval,
        "load_frozen_techqa_retrieval_cases",
        fake_loader,
    )
    monkeypatch.setattr(
        retrieval_eval,
        "build_dev_retrieval_run_manifest",
        fake_manifest_builder,
        raising=False,
    )
    monkeypatch.setattr(
        retrieval_eval,
        "ensure_dev_retrieval_run_manifest",
        fake_manifest_lock,
        raising=False,
    )
    monkeypatch.setattr(
        retrieval_eval,
        "evaluate_techqa_retrieval_case",
        fake_single_case_evaluator,
        raising=False,
    )
    monkeypatch.setattr(
        retrieval_eval,
        "run_resumable_retrieval_eval",
        fake_runner,
        raising=False,
    )
    monkeypatch.setattr(
        retrieval_eval,
        "evaluate_techqa_retrieval_cases",
        fail_legacy_batch,
    )
    monkeypatch.setattr(retrieval_eval, "write_e0_reports", fake_writer)

    retrieval_eval.main(["--split", "dev"])

    assert order == [
        "load",
        "build_manifest",
        "lock_manifest",
        "runner",
        "search",
        "writer",
    ]