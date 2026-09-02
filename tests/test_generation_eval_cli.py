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


def test_main_pilot_routes_frozen_12_train_cases_to_isolated_artifacts(
    monkeypatch,
    tmp_path,
):
    all_cases = [
        SimpleNamespace(
            question_id=f"TRAIN_Q{index:03d}",
            split="train",
        )
        for index in range(24)
    ]
    pilot_cases = all_cases[:12]

    summary = SimpleNamespace(
        query_count=12,
        answerable_count=6,
        impossible_count=6,
        correctness_mean=0.8,
        faithfulness_mean=0.9,
        abstention_accuracy=0.7,
        hallucination_rate=0.3,
        e2e_latency_p50_ms=1200.0,
        e2e_latency_p95_ms=2400.0,
    )

    pilot_dir = tmp_path / "g0_generation" / "pilot"
    pilot_checkpoint = (
        pilot_dir / "train_generation_checkpoint.jsonl"
    )
    pilot_manifest_path = (
        pilot_dir / "train_generation_manifest.json"
    )

    expected_manifest = {
        "identity": {
            "run": "g0_generation",
            "split": "train",
            "query_count": 12,
        },
        "provenance": {},
    }

    observed = {}

    monkeypatch.setattr(
        generation_eval,
        "DEFAULT_G0_PILOT_REPORT_DIR",
        pilot_dir,
        raising=False,
    )
    monkeypatch.setattr(
        generation_eval,
        "DEFAULT_G0_PILOT_CHECKPOINT_PATH",
        pilot_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(
        generation_eval,
        "DEFAULT_G0_PILOT_RUN_MANIFEST_PATH",
        pilot_manifest_path,
        raising=False,
    )

    def fake_loader():
        observed["loaded"] = True
        return all_cases

    def fake_selector(input_cases):
        observed["selector_cases"] = input_cases
        return pilot_cases

    def fake_builder(**kwargs):
        observed["builder_kwargs"] = kwargs
        return expected_manifest

    def fake_ensure(
        manifest,
        *,
        run_manifest_path,
        checkpoint_path,
    ):
        observed["locked_manifest"] = manifest
        observed["manifest_path"] = run_manifest_path
        observed["checkpoint_path"] = checkpoint_path

    def fake_runner(
        input_cases,
        *,
        checkpoint_path,
        split,
    ):
        observed["runner_cases"] = input_cases
        observed["runner_checkpoint"] = checkpoint_path
        observed["runner_split"] = split
        return summary

    def fake_writer(
        input_summary,
        *,
        report_dir,
    ):
        observed["written_summary"] = input_summary
        observed["report_dir"] = report_dir

    monkeypatch.setattr(
        generation_eval,
        "load_frozen_techqa_generation_cases",
        fake_loader,
    )
    monkeypatch.setattr(
        generation_eval,
        "select_g0_train_pilot_cases",
        fake_selector,
    )
    monkeypatch.setattr(
        generation_eval,
        "build_generation_run_manifest",
        fake_builder,
    )
    monkeypatch.setattr(
        generation_eval,
        "ensure_generation_run_manifest",
        fake_ensure,
    )
    monkeypatch.setattr(
        generation_eval,
        "run_resumable_generation_eval",
        fake_runner,
    )
    monkeypatch.setattr(
        generation_eval,
        "write_generation_reports",
        fake_writer,
    )

    generation_eval.main(
        [
            "--split",
            "train",
            "--pilot",
        ]
    )

    assert observed["loaded"] is True
    assert observed["selector_cases"] is all_cases

    assert observed["builder_kwargs"] == {
        "split": "train",
        "query_count": 12,
    }

    assert observed["locked_manifest"] is expected_manifest
    assert observed["manifest_path"] == pilot_manifest_path
    assert observed["checkpoint_path"] == pilot_checkpoint

    assert observed["runner_cases"] is pilot_cases
    assert observed["runner_checkpoint"] == pilot_checkpoint
    assert observed["runner_split"] == "train"

    assert observed["written_summary"] is summary
    assert observed["report_dir"] == pilot_dir


def test_main_rejects_dev_pilot_before_building_manifest(
    monkeypatch,
):
    import pytest

    def forbidden_builder(**kwargs):
        raise AssertionError(
            "DEV pilot must be rejected before manifest build"
        )

    monkeypatch.setattr(
        generation_eval,
        "build_generation_run_manifest",
        forbidden_builder,
    )

    with pytest.raises(SystemExit) as exc_info:
        generation_eval.main(
            [
                "--split",
                "dev",
                "--pilot",
            ]
        )

    assert exc_info.value.code == 2
