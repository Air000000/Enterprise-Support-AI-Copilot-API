import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _run_manifest(tmp_path: Path) -> dict:
    return generation_eval.build_generation_run_manifest(
        manifest_path=_frozen_manifest(tmp_path),
        project_sha="project-sha",
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
        created_at="2026-08-22T09:00:00+00:00",
    )


def test_build_generation_run_manifest_freezes_experiment_identity(tmp_path):
    run_manifest = _run_manifest(tmp_path)

    identity = run_manifest["identity"]
    assert identity["benchmark"] == "TechQA-RAG-Eval"
    assert identity["run"] == "g0_generation"
    assert identity["split"] == "train"
    assert identity["data"] == {
        "corpus_sha256": "corpus-sha",
        "queries_sha256": "queries-sha",
        "qrels_sha256": "qrels-sha",
        "generation_sha256": "generation-sha",
        "generation_revision": "generation-revision",
    }
    assert identity["retrieval"] == {
        "retriever": "chroma_dense",
        "collection": "techqa_e0_dense",
        "embedding_model": "text-embedding-v4",
        "chunk_strategy": "paragraph_aware_character",
        "chunk_size_chars": 800,
        "chunk_overlap_chars": 120,
        "min_chunk_size_chars": 150,
        "candidate_k": 100,
        "reranker_model": "qwen3-rerank",
        "reranker_instruction": (
            "Rank the candidate passages by relevance to resolving "
            "the technical support query."
        ),
        "context_top_k": 3,
        "refusal_signal": "dense_top1_distance",
        "refusal_max_distance": 0.9,
    }
    assert identity["generation"]["model"] == "qwen3.5-plus"
    assert identity["generation"]["thinking_mode"] == "provider_default"
    assert identity["generation"]["prompt_sha256"] == "prompt-sha"
    assert identity["generation"]["source_sha256"] == "eval-source-sha"
    assert identity["judge"]["framework"] == "deepeval"
    assert identity["judge"]["framework_version"] == "4.1.5"
    assert identity["judge"]["model"] == "qwen3.5-plus"
    assert identity["judge"]["correctness_evaluation_steps"] == (
        generation_eval.CORRECTNESS_EVALUATION_STEPS
    )
    assert identity["latency"] == {
        "e2e_definition": "dense retrieval + rerank + generation",
        "judge_included": False,
    }
    assert run_manifest["provenance"]["project_sha"] == "project-sha"
    assert run_manifest["provenance"]["python"] == "3.11.14"
    assert run_manifest["provenance"]["created_at"] == "2026-08-22T09:00:00+00:00"


def test_manifest_lock_requires_explicit_adoption_then_rejects_identity_drift(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    checkpoint_path.write_text('{"question_id":"TRAIN_Q000"}\n', encoding="utf-8")
    run_manifest_path = tmp_path / "train_run_manifest.json"
    expected = _run_manifest(tmp_path)

    with pytest.raises(RuntimeError, match="existing checkpoint"):
        generation_eval.ensure_generation_run_manifest(
            expected,
            run_manifest_path=run_manifest_path,
            checkpoint_path=checkpoint_path,
        )

    generation_eval.ensure_generation_run_manifest(
        expected,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
        adopt_existing_checkpoint=True,
    )
    persisted = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert persisted["identity"] == expected["identity"]
    assert persisted["provenance"]["adopted_existing_checkpoint_cases"] == 1

    generation_eval.ensure_generation_run_manifest(
        expected,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
    )

    changed = deepcopy(expected)
    changed["identity"]["generation"]["model"] = "different-model"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        generation_eval.ensure_generation_run_manifest(
            changed,
            run_manifest_path=run_manifest_path,
            checkpoint_path=checkpoint_path,
        )


def test_main_checks_run_manifest_before_resuming_generation(monkeypatch):
    cases = [SimpleNamespace(question_id="TRAIN_Q000", split="train")]
    summary = SimpleNamespace(
        query_count=600,
        answerable_count=450,
        impossible_count=150,
        correctness_mean=0.8,
        faithfulness_mean=0.9,
        abstention_accuracy=0.7,
        hallucination_rate=0.3,
        e2e_latency_p50_ms=1200.0,
        e2e_latency_p95_ms=2400.0,
    )
    order = []
    expected_manifest = {"identity": {"run": "e0_generation"}, "provenance": {}}

    monkeypatch.setattr(
        generation_eval,
        "load_frozen_techqa_generation_cases",
        lambda: cases,
    )
    monkeypatch.setattr(
        generation_eval,
        "build_generation_run_manifest",
        lambda: expected_manifest,
    )

    def fake_ensure(manifest, **kwargs):
        order.append(("lock", manifest))

    def fake_runner(input_cases, *, checkpoint_path, split):
        order.append(("runner", input_cases, checkpoint_path, split))
        return summary

    def fake_writer(input_summary):
        order.append(("writer", input_summary))

    monkeypatch.setattr(generation_eval, "ensure_generation_run_manifest", fake_ensure)
    monkeypatch.setattr(generation_eval, "run_resumable_generation_eval", fake_runner)
    monkeypatch.setattr(generation_eval, "write_generation_reports", fake_writer)

    generation_eval.main()

    assert order[0] == ("lock", expected_manifest)
    assert order[1][0] == "runner"
    assert order[2] == ("writer", summary)
