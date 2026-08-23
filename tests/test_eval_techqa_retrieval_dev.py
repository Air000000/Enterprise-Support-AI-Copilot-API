import json
from types import SimpleNamespace

import experiments.evals.eval_techqa_retrieval as retrieval_eval
from experiments.evals.adapters.techqa import TechQARetrievalCase


def test_dev_retrieval_normalizes_trailing_whitespace_before_dense_search():
    case = TechQARetrievalCase(
        question_id="DEV_Q001",
        question="held out technical question  \n",
        relevant_document_ids=("doc_gold",),
        split="dev",
    )
    observed = []

    def fake_searcher(query: str, *, top_k: int):
        observed.append((query, top_k))
        return [
            SimpleNamespace(
                chunk_id="gold_chunk",
                document_id="doc_gold",
                distance=0.1,
            )
        ]

    summary = retrieval_eval.evaluate_techqa_retrieval_cases(
        [case],
        searcher=fake_searcher,
        split="dev",
        clock=iter([0.0, 0.01]).__next__,
    )

    assert observed == [("held out technical question", 100)]
    assert summary.split == "dev"
    assert summary.query_count == 1


def test_write_e0_reports_uses_split_specific_dev_artifact_names(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
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
    summary = retrieval_eval.TechQARetrievalEvalSummary(
        split="dev",
        query_count=1,
        candidate_chunk_k=100,
        document_top_k=20,
        metrics={"recall@5": 1.0, "recall@20": 1.0, "mrr@10": 1.0},
        latency_p50_ms=10.0,
        latency_p95_ms=10.0,
        results=(),
    )

    retrieval_eval.write_e0_reports(
        summary,
        report_dir=tmp_path,
        manifest_path=manifest_path,
    )

    assert (tmp_path / "dev_manifest.json").exists()
    assert (tmp_path / "dev_results.jsonl").exists()
    assert (tmp_path / "dev_metrics.json").exists()
    assert not (tmp_path / "train_manifest.json").exists()
    assert not (tmp_path / "train_results.jsonl").exists()
    assert not (tmp_path / "train_metrics.json").exists()

    persisted_manifest = json.loads(
        (tmp_path / "dev_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_manifest["split"] == "dev"
    assert persisted_manifest["query_count"] == 1


def test_cli_can_run_frozen_dev_split_without_running_train(monkeypatch):
    dev_cases = [
        TechQARetrievalCase(
            question_id=f"DEV_Q{index:03d}",
            question=f"dev question {index}",
            relevant_document_ids=(f"doc_{index:03d}",),
            split="dev",
        )
        for index in range(160)
    ]
    observed = []
    summary = SimpleNamespace(
        split="dev",
        query_count=160,
        metrics={"recall@5": 0.5, "recall@20": 0.7, "mrr@10": 0.4},
        latency_p50_ms=1000.0,
        latency_p95_ms=1500.0,
    )

    def fake_loader():
        observed.append("load")
        return dev_cases

    def fake_evaluator(cases, *, split):
        assert cases is dev_cases
        observed.append(("evaluate", split))
        return summary

    def fake_writer(input_summary):
        assert input_summary is summary
        observed.append(("write", input_summary.split))

    monkeypatch.setattr(
        retrieval_eval,
        "load_frozen_techqa_retrieval_cases",
        fake_loader,
    )
    monkeypatch.setattr(
        retrieval_eval,
        "evaluate_techqa_retrieval_cases",
        fake_evaluator,
    )
    monkeypatch.setattr(retrieval_eval, "write_e0_reports", fake_writer)

    retrieval_eval.main(["--split", "dev"])

    assert observed == [
        "load",
        ("evaluate", "dev"),
        ("write", "dev"),
    ]
