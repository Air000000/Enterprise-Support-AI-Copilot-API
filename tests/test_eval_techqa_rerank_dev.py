import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import experiments.evals.eval_techqa_rerank as rerank_eval


def _summary():
    return SimpleNamespace(
        query_count=160,
        metrics={
            "recall@5": 0.7,
            "recall@20": 0.8,
            "mrr@10": 0.6,
        },
        rerank_latency_p50_ms=1000.0,
        rerank_latency_p95_ms=2000.0,
        provider_total_tokens=123456,
        results=(),
    )


def test_dev_manifest_locks_split_count_and_frozen_e0_source(tmp_path):
    e0_results_path = tmp_path / "dev_results.jsonl"
    e0_results_path.write_text('{"question_id":"DEV_Q001"}\n', encoding="utf-8")
    e0_manifest_path = tmp_path / "dev_manifest.json"
    e0_manifest_path.write_text(
        json.dumps({"run": "e0_dense", "candidate_chunk_k": 100}),
        encoding="utf-8",
    )

    manifest = rerank_eval.build_rerank_run_manifest(
        e0_results_path=e0_results_path,
        e0_manifest_path=e0_manifest_path,
        project_sha="dev-project-sha",
        created_at="2026-08-24T09:00:00+08:00",
        split="dev",
        query_count=160,
    )

    identity = manifest["identity"]
    assert identity["run"] == "e1_rerank"
    assert identity["split"] == "dev"
    assert identity["query_count"] == 160
    assert identity["source_e0"] == {
        "results_sha256": hashlib.sha256(e0_results_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(e0_manifest_path.read_bytes()).hexdigest(),
    }
    assert identity["reranker"] == {
        "model": "qwen3-rerank",
        "instruction": rerank_eval.DEFAULT_RERANK_INSTRUCTION,
        "candidate_chunk_k": 100,
        "query_normalization": "rstrip",
    }


def test_write_rerank_reports_uses_dev_artifact_names(tmp_path):
    rerank_eval.write_rerank_reports(
        _summary(),
        report_dir=tmp_path,
        split="dev",
    )

    assert (tmp_path / "dev_results.jsonl").exists()
    assert (tmp_path / "dev_metrics.json").exists()
    assert not (tmp_path / "train_results.jsonl").exists()
    assert not (tmp_path / "train_metrics.json").exists()


def test_main_dev_routes_only_dev_inputs_and_outputs(monkeypatch, tmp_path):
    dev_e0_results = tmp_path / "e0_dense" / "dev_results.jsonl"
    dev_e0_manifest = tmp_path / "e0_dense" / "dev_manifest.json"
    dev_checkpoint = tmp_path / "e1_rerank" / "dev_checkpoint.jsonl"
    dev_manifest = tmp_path / "e1_rerank" / "dev_manifest.json"
    report_dir = tmp_path / "e1_rerank"

    monkeypatch.setattr(
        rerank_eval,
        "DEFAULT_E0_DEV_RERANK_RESULTS_PATH",
        dev_e0_results,
        raising=False,
    )
    monkeypatch.setattr(
        rerank_eval,
        "DEFAULT_E0_DEV_RERANK_MANIFEST_PATH",
        dev_e0_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        rerank_eval,
        "DEFAULT_RERANK_DEV_CHECKPOINT_PATH",
        dev_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(
        rerank_eval,
        "DEFAULT_RERANK_DEV_RUN_MANIFEST_PATH",
        dev_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        rerank_eval,
        "DEFAULT_RERANK_REPORT_DIR",
        report_dir,
    )
    monkeypatch.setattr(
        rerank_eval,
        "DEFAULT_RERANK_DEV_COUNT",
        160,
        raising=False,
    )

    records = [
        {"question_id": f"DEV_Q{index:03d}"}
        for index in range(160)
    ]
    summary = _summary()
    collection = object()
    expected_manifest = {
        "identity": {"run": "e1_rerank", "split": "dev"},
        "provenance": {},
    }
    order = []

    def fake_loader(path, *, expected_count):
        assert Path(path) == dev_e0_results
        assert expected_count == 160
        order.append("load_dev_e0")
        return records

    def fake_manifest_builder(**kwargs):
        assert Path(kwargs["e0_results_path"]) == dev_e0_results
        assert Path(kwargs["e0_manifest_path"]) == dev_e0_manifest
        assert kwargs["split"] == "dev"
        assert kwargs["query_count"] == 160
        order.append("build_dev_manifest")
        return expected_manifest

    def fake_manifest_lock(manifest, *, run_manifest_path, checkpoint_path):
        assert manifest is expected_manifest
        assert Path(run_manifest_path) == dev_manifest
        assert Path(checkpoint_path) == dev_checkpoint
        order.append("lock_dev_manifest")

    def fake_collection_loader():
        order.append("open_collection")
        return collection

    def fake_runner(input_records, *, evaluator, checkpoint_path):
        assert input_records is records
        assert Path(checkpoint_path) == dev_checkpoint
        order.append("runner")
        return summary

    def fake_writer(input_summary, *, report_dir, split):
        assert input_summary is summary
        assert Path(report_dir) == Path(report_dir_for_assertion)
        assert split == "dev"
        order.append("write_dev")

    report_dir_for_assertion = report_dir
    monkeypatch.setattr(rerank_eval, "load_frozen_e0_rerank_records", fake_loader)
    monkeypatch.setattr(rerank_eval, "build_rerank_run_manifest", fake_manifest_builder)
    monkeypatch.setattr(rerank_eval, "ensure_rerank_run_manifest", fake_manifest_lock)
    monkeypatch.setattr(rerank_eval, "open_frozen_techqa_collection", fake_collection_loader)
    monkeypatch.setattr(rerank_eval, "run_resumable_rerank_eval", fake_runner)
    monkeypatch.setattr(rerank_eval, "write_rerank_reports", fake_writer)
    monkeypatch.setattr(rerank_eval, "_resolve_project_sha", lambda: "dev-project-sha")

    rerank_eval.main(["--split", "dev"])

    assert order == [
        "load_dev_e0",
        "build_dev_manifest",
        "lock_dev_manifest",
        "open_collection",
        "runner",
        "write_dev",
    ]
