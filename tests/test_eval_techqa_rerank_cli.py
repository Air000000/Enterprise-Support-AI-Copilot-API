import json
from types import SimpleNamespace

import pytest

import experiments.evals.eval_techqa_rerank as rerank_eval


def _require(name: str):
    value = getattr(rerank_eval, name, None)
    if value is None:
        pytest.fail(f"experiments.evals.eval_techqa_rerank.{name} is not implemented yet")
    return value


def test_load_frozen_e0_rerank_records_requires_exact_train_count(tmp_path):
    loader = _require("load_frozen_e0_rerank_records")
    path = tmp_path / "train_results.jsonl"
    path.write_text(
        json.dumps({"question_id": "TRAIN_Q001"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected=450.*actual=1"):
        loader(path)


def test_main_locks_manifest_before_rerank_then_writes_reports(monkeypatch):
    main = _require("main")
    records = [
        {"question_id": f"TRAIN_Q{index:03d}"}
        for index in range(1, 451)
    ]
    collection = object()
    summary = SimpleNamespace(
        query_count=450,
        metrics={
            "recall@5": 0.7,
            "recall@20": 0.8,
            "mrr@10": 0.6,
        },
        rerank_latency_p50_ms=1000.0,
        rerank_latency_p95_ms=2000.0,
        provider_total_tokens=123456,
    )
    expected_manifest = {
        "identity": {"run": "e1_rerank"},
        "provenance": {},
    }
    order = []

    def fake_loader(*args, **kwargs):
        order.append("load_e0")
        return records

    def fake_manifest_builder(*args, **kwargs):
        order.append("build_manifest")
        return expected_manifest

    def fake_manifest_lock(manifest, **kwargs):
        assert manifest is expected_manifest
        order.append("lock_manifest")

    def fake_collection_loader(*args, **kwargs):
        order.append("open_collection")
        return collection

    def fake_evaluate(record, *, collection):
        assert collection is collection_for_assertion
        order.append("provider")
        return SimpleNamespace(question_id=record["question_id"])

    def fake_runner(input_records, *, evaluator, checkpoint_path):
        assert input_records is records
        order.append("runner")
        evaluator(input_records[0])
        return summary

    def fake_writer(input_summary, *, report_dir):
        assert input_summary is summary
        order.append("writer")

    collection_for_assertion = collection
    monkeypatch.setattr(
        rerank_eval,
        "load_frozen_e0_rerank_records",
        fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        rerank_eval,
        "build_rerank_run_manifest",
        fake_manifest_builder,
    )
    monkeypatch.setattr(
        rerank_eval,
        "ensure_rerank_run_manifest",
        fake_manifest_lock,
    )
    monkeypatch.setattr(
        rerank_eval,
        "open_frozen_techqa_collection",
        fake_collection_loader,
        raising=False,
    )
    monkeypatch.setattr(
        rerank_eval,
        "rerank_frozen_e0_record",
        fake_evaluate,
    )
    monkeypatch.setattr(
        rerank_eval,
        "run_resumable_rerank_eval",
        fake_runner,
    )
    monkeypatch.setattr(
        rerank_eval,
        "write_rerank_reports",
        fake_writer,
    )

    main()

    assert order == [
        "load_e0",
        "build_manifest",
        "lock_manifest",
        "open_collection",
        "runner",
        "provider",
        "writer",
    ]
