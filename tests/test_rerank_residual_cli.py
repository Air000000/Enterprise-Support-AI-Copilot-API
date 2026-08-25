from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


def _load_module():
    return import_module("experiments.evals.rerank_residual_analysis")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_summarize_cli_writes_review_summary(tmp_path: Path, capsys) -> None:
    module = _load_module()
    report_dir = tmp_path / "e1_rerank"
    _write_jsonl(
        report_dir / "train_residual_review.jsonl",
        [
            {"question_id": "TRAIN_Q1", "manual_label": "lexical_candidate", "notes": "a"},
            {
                "question_id": "TRAIN_Q2",
                "manual_label": "semantic_or_indirect_miss",
                "notes": "b",
            },
            {
                "question_id": "TRAIN_Q3",
                "manual_label": "qrel_or_query_ambiguity",
                "notes": "c",
            },
        ],
    )

    module.main(["summarize", "--report-dir", str(report_dir)])

    summary = json.loads(
        (report_dir / "train_residual_review_summary.json").read_text(encoding="utf-8")
    )
    assert summary == {
        "reviewed_count": 3,
        "label_counts": {
            "lexical_candidate": 1,
            "semantic_or_indirect_miss": 1,
            "qrel_or_query_ambiguity": 1,
        },
        "label_rates": {
            "lexical_candidate": 1 / 3,
            "semantic_or_indirect_miss": 1 / 3,
            "qrel_or_query_ambiguity": 1 / 3,
        },
        "population_rate_claim_allowed": False,
    }
    stdout = capsys.readouterr().out
    assert "reviewed_count = 3" in stdout
    assert "provider_calls = 0" in stdout


def test_freeze_gate_cli_uses_actual_residual_miss_count(tmp_path: Path, capsys) -> None:
    module = _load_module()
    report_dir = tmp_path / "e1_rerank"
    _write_json(
        report_dir / "train_residual_summary.json",
        {
            "query_count": 450,
            "bucket_counts": {
                "dense_candidate_miss_top100": 63,
                "rerank_residual_top20": 20,
                "resolved_top20": 367,
            },
            "bucket_rates": {
                "dense_candidate_miss_top100": 0.14,
                "rerank_residual_top20": 20 / 450,
                "resolved_top20": 367 / 450,
            },
            "dense_candidate_miss_count": 63,
        },
    )

    module.main(["freeze-gate", "--report-dir", str(report_dir)])

    gate = json.loads((report_dir / "r3_gate.json").read_text(encoding="utf-8"))
    assert gate["dense_candidate_miss_count"] == 63
    assert gate["required_recovered_dense_misses"] == 10
    assert gate["required_net_gain_cases"] == 7
    assert gate["required_net_gain_pp"] == 7 / 450 * 100.0
    assert gate["admission_logic"] == (
        "recovered_dense_misses >= required_recovered_dense_misses AND "
        "hybrid_hit100 - dense_hit100 >= required_net_gain_cases"
    )
    stdout = capsys.readouterr().out
    assert "dense_candidate_miss_count = 63" in stdout
    assert "required_recovered_dense_misses = 10" in stdout
    assert "required_net_gain_cases = 7" in stdout
    assert "provider_calls = 0" in stdout
