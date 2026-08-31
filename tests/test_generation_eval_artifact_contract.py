from pathlib import Path

from experiments.evals.eval_techqa_generation import (
    DEFAULT_DEV_GENERATION_CHECKPOINT_PATH,
    DEFAULT_DEV_GENERATION_RUN_MANIFEST_PATH,
    DEFAULT_GENERATION_CHECKPOINT_PATH,
    DEFAULT_GENERATION_REPORT_DIR,
    DEFAULT_GENERATION_RUN_MANIFEST_PATH,
    DEFAULT_JUDGE_CALIBRATION_PATH,
    TechQAGenerationEvalResult,
    TechQAGenerationEvalSummary,
    write_generation_reports,
)


def _summary() -> TechQAGenerationEvalSummary:
    result = TechQAGenerationEvalResult(
        question_id="TRAIN_Q001",
        question="question",
        gold_answer="gold answer",
        answerable=True,
        retrieved_chunk_ids=("doc_chunk_0",),
        retrieved_document_ids=("doc",),
        retrieval_context=("retrieved evidence",),
        generated_answer="generated answer",
        retrieval_status="ok",
        top_distance=0.2,
        abstained=False,
        hallucinated=False,
        correctness_score=0.8,
        correctness_reason="correct",
        faithfulness_score=0.9,
        faithfulness_reason="grounded",
        e2e_latency_ms=100.0,
    )
    return TechQAGenerationEvalSummary(
        split="train",
        query_count=1,
        answerable_count=1,
        impossible_count=0,
        correctness_mean=0.8,
        faithfulness_mean=0.9,
        abstention_accuracy=0.0,
        hallucination_rate=0.0,
        e2e_latency_p50_ms=100.0,
        e2e_latency_p95_ms=100.0,
        results=(result,),
    )


def test_default_generation_artifacts_are_isolated_in_g0_generation():
    expected_dir = Path("experiments/evals/reports/g0_generation")

    assert DEFAULT_GENERATION_REPORT_DIR == expected_dir
    assert DEFAULT_GENERATION_CHECKPOINT_PATH == (
        expected_dir / "train_generation_checkpoint.jsonl"
    )
    assert DEFAULT_GENERATION_RUN_MANIFEST_PATH == (
        expected_dir / "train_generation_manifest.json"
    )
    assert DEFAULT_DEV_GENERATION_CHECKPOINT_PATH == (
        expected_dir / "dev_generation_checkpoint.jsonl"
    )
    assert DEFAULT_DEV_GENERATION_RUN_MANIFEST_PATH == (
        expected_dir / "dev_generation_manifest.json"
    )
    assert DEFAULT_JUDGE_CALIBRATION_PATH == (
        expected_dir / "judge_calibration.jsonl"
    )


def test_generation_report_writer_preserves_retrieval_artifacts(tmp_path):
    retrieval_results = tmp_path / "train_results.jsonl"
    retrieval_metrics = tmp_path / "train_metrics.json"
    retrieval_manifest = tmp_path / "train_manifest.json"
    retrieval_results.write_text("retrieval-results\n", encoding="utf-8")
    retrieval_metrics.write_text("retrieval-metrics\n", encoding="utf-8")
    retrieval_manifest.write_text("retrieval-manifest\n", encoding="utf-8")

    write_generation_reports(_summary(), report_dir=tmp_path)

    assert retrieval_results.read_text(encoding="utf-8") == "retrieval-results\n"
    assert retrieval_metrics.read_text(encoding="utf-8") == "retrieval-metrics\n"
    assert retrieval_manifest.read_text(encoding="utf-8") == "retrieval-manifest\n"

    assert (tmp_path / "train_generation_results.jsonl").exists()
    assert (tmp_path / "train_generation_metrics.json").exists()
