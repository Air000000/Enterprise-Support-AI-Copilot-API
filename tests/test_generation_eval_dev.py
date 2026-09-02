import json
from pathlib import Path
from types import SimpleNamespace

import experiments.evals.eval_techqa_generation as generation_eval


def _frozen_manifest(tmp_path: Path) -> Path:
    manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "retrieval_dataset": {
            "corpus_sha256": "corpus-sha",
            "queries_sha256": "queries-sha",
            "qrels_sha256": "qrels-sha",
        },
        "generation_dataset": {
            "repo": "nvidia/TechQA-RAG-Eval",
            "revision": "generation-revision",
            "metadata_sha256": "generation-sha",
        },
        "baseline_rag": {
            "retriever": "chroma_dense",
            "chunk_strategy": "paragraph_aware_character",
            "chunk_size_chars": 800,
            "chunk_overlap_chars": 120,
            "min_chunk_size_chars": 150,
        },
    }
    path = tmp_path / "techqa_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_dev_generation_manifest_locks_split_and_query_count(tmp_path):
    manifest = generation_eval.build_generation_run_manifest(
        manifest_path=_frozen_manifest(tmp_path),
        project_sha="dev-project-sha",
        generation_model="qwen3.5-plus",
        embedding_model="text-embedding-v4",
        python_version="3.11.14",
        dependency_versions={
            "deepeval": "4.1.5",
            "openai": "2.38.0",
            "chromadb": "1.5.9",
        },
        prompt_sha256="prompt-sha",
        eval_source_sha256="eval-source-sha",
        created_at="2026-08-24T10:00:00+08:00",
        split="dev",
        query_count=310,
    )

    identity = manifest["identity"]
    assert identity["run"] == "g0_generation"
    assert identity["split"] == "dev"
    assert identity["query_count"] == 310
    assert identity["retrieval"]["retriever"] == "chroma_dense"
    assert identity["retrieval"]["candidate_k"] == 100
    assert identity["retrieval"]["reranker_model"] == "qwen3-rerank"
    assert identity["retrieval"]["context_policy"] == (
        "document_aware_forward_expansion_v1"
    )
    assert identity["retrieval"]["rerank_anchor_top_k"] == 3
    assert identity["retrieval"]["dense_top1_rescue"] is True
    assert identity["retrieval"]["forward_sibling_chunks"] == 3
    assert identity["retrieval"]["max_context_chunks"] == 16
    assert "context_top_k" not in identity["retrieval"]
    assert identity["retrieval"]["refusal_signal"] == "dense_top1_distance"
    assert identity["generation"]["model"] == "qwen3.5-plus"
    assert identity["judge"]["framework"] == "deepeval"


def test_main_dev_routes_to_independent_generation_artifacts(monkeypatch, tmp_path):
    dev_checkpoint = tmp_path / "e0_dense" / "dev_generation_checkpoint.jsonl"
    dev_manifest = tmp_path / "e0_dense" / "dev_generation_manifest.json"
    report_dir = tmp_path / "e0_dense"

    monkeypatch.setattr(
        generation_eval,
        "DEFAULT_DEV_GENERATION_CHECKPOINT_PATH",
        dev_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(
        generation_eval,
        "DEFAULT_DEV_GENERATION_RUN_MANIFEST_PATH",
        dev_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        generation_eval,
        "DEFAULT_GENERATION_REPORT_DIR",
        report_dir,
    )

    cases = [
        SimpleNamespace(question_id=f"DEV_Q{index:03d}", split="dev")
        for index in range(160)
    ] + [
        SimpleNamespace(question_id=f"DEV_I{index:03d}", split="dev")
        for index in range(150)
    ]
    summary = SimpleNamespace(
        split="dev",
        query_count=310,
        answerable_count=160,
        impossible_count=150,
        correctness_mean=0.5,
        faithfulness_mean=0.8,
        abstention_accuracy=0.4,
        hallucination_rate=0.6,
        e2e_latency_p50_ms=1000.0,
        e2e_latency_p95_ms=2000.0,
    )
    expected_manifest = {
        "identity": {"run": "g0_generation", "split": "dev", "query_count": 310},
        "provenance": {},
    }
    order = []

    def fake_builder(**kwargs):
        assert kwargs["split"] == "dev"
        assert kwargs["query_count"] == 310
        order.append("build_dev_manifest")
        return expected_manifest

    def fake_ensure(manifest, *, run_manifest_path, checkpoint_path):
        assert manifest is expected_manifest
        assert Path(run_manifest_path) == dev_manifest
        assert Path(checkpoint_path) == dev_checkpoint
        order.append("lock_dev_manifest")

    def fake_loader():
        order.append("load_cases")
        return cases

    def fake_runner(input_cases, *, checkpoint_path, split):
        assert input_cases is cases
        assert Path(checkpoint_path) == dev_checkpoint
        assert split == "dev"
        order.append("runner")
        return summary

    def fake_writer(input_summary, *, report_dir):
        assert input_summary is summary
        assert Path(report_dir) == report_dir_for_assertion
        order.append("writer")

    report_dir_for_assertion = report_dir
    monkeypatch.setattr(generation_eval, "build_generation_run_manifest", fake_builder)
    monkeypatch.setattr(generation_eval, "ensure_generation_run_manifest", fake_ensure)
    monkeypatch.setattr(
        generation_eval,
        "load_frozen_techqa_generation_cases",
        fake_loader,
    )
    monkeypatch.setattr(generation_eval, "run_resumable_generation_eval", fake_runner)
    monkeypatch.setattr(generation_eval, "write_generation_reports", fake_writer)

    generation_eval.main(["--split", "dev"])

    assert order == [
        "build_dev_manifest",
        "lock_dev_manifest",
        "load_cases",
        "runner",
        "writer",
    ]
