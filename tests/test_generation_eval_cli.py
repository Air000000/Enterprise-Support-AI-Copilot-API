from types import SimpleNamespace

import experiments.evals.eval_techqa_generation as generation_eval


def test_main_runs_train_generation_eval_with_resume_and_writes_reports(monkeypatch):
    cases = [SimpleNamespace(question_id="TRAIN_Q001", split="train")]
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
    observed = {}
    run_manifest = {"identity": {"run": "e0_generation"}, "provenance": {}}

    def fake_loader():
        observed["loaded"] = True
        return cases

    def fake_runner(input_cases, *, checkpoint_path, split):
        observed["runner_cases"] = input_cases
        observed["checkpoint_path"] = checkpoint_path
        observed["split"] = split
        return summary

    def fake_writer(input_summary):
        observed["written_summary"] = input_summary

    monkeypatch.setattr(
        generation_eval,
        "build_generation_run_manifest",
        lambda: run_manifest,
    )
    monkeypatch.setattr(
        generation_eval,
        "ensure_generation_run_manifest",
        lambda manifest: observed.setdefault("locked_manifest", manifest),
    )
    monkeypatch.setattr(
        generation_eval,
        "load_frozen_techqa_generation_cases",
        fake_loader,
    )
    monkeypatch.setattr(
        generation_eval,
        "run_resumable_generation_eval",
        fake_runner,
    )
    monkeypatch.setattr(generation_eval, "write_generation_reports", fake_writer)

    generation_eval.main()

    assert observed["locked_manifest"] is run_manifest
    assert observed["loaded"] is True
    assert observed["runner_cases"] is cases
    assert observed["checkpoint_path"] == generation_eval.DEFAULT_GENERATION_CHECKPOINT_PATH
    assert observed["split"] == "train"
    assert observed["written_summary"] is summary
