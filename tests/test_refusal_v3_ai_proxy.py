import hashlib
import json
import sys
from pathlib import Path

import pytest

from experiments.evals import refusal_v3_ai_proxy as v3


def test_v3_freeze_cli_and_guards(tmp_path, monkeypatch) -> None:
    packets = []
    targets = []
    for index in range(80):
        question_id = f"TRAIN_Q{index:03d}"
        packets.append(
            {
                "question_id": question_id,
                "question": f"Question {index}",
                "sources": [
                    {"source_id": f"Source {rank}", "content": f"Content {rank}"}
                    for rank in range(1, 15)
                ],
            }
        )
        targets.append(
            {
                "question_id": question_id,
                "target_class": (
                    "SUFFICIENT"
                    if index < 44
                    else "INSUFFICIENT"
                    if index < 71
                    else "QUESTIONABLE"
                ),
            }
        )
    packet_path = tmp_path / "packets.jsonl"
    source_path = tmp_path / "source.jsonl"
    v3.v2._write_jsonl(packet_path, packets)
    v3.v2._write_jsonl(source_path, targets)
    monkeypatch.setattr(v3, "EXPECTED_PACKET_SHA256", v3.v2._sha256(packet_path))
    monkeypatch.setattr(
        v3, "EXPECTED_TARGETS_SOURCE_SHA256", v3.v2._sha256(source_path)
    )
    inputs_path = tmp_path / "inputs.jsonl"
    targets_path = tmp_path / "targets.jsonl"
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refusal_v3_ai_proxy",
            "--packet-path",
            str(packet_path),
            "--targets-source-path",
            str(source_path),
            "--inputs-path",
            str(inputs_path),
            "--targets-path",
            str(targets_path),
            "--manifest-path",
            str(manifest_path),
        ],
    )
    v3.main()
    inputs = v3.v2._read_jsonl(inputs_path)
    selected_targets = v3.v2._read_jsonl(targets_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(inputs) == len(selected_targets) == 50
    assert sum(row["target_class"] == "SUFFICIENT" for row in selected_targets) == 25
    assert sum(row["target_class"] == "INSUFFICIENT" for row in selected_targets) == 25
    assert manifest["selection"]["questionable_excluded"] == 9
    assert manifest["selection"]["selection_seed"] == v3.SELECTION_SEED
    assert manifest["controls"]["provider_calls"] == 0
    frozen_hashes = [
        v3.v2._sha256(path) for path in (inputs_path, targets_path, manifest_path)
    ]
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        v3.main()
    assert frozen_hashes == [
        v3.v2._sha256(path) for path in (inputs_path, targets_path, manifest_path)
    ]
    monkeypatch.setattr(v3, "EXPECTED_PACKET_SHA256", "tampered")
    with pytest.raises(RuntimeError, match="blind packet SHA mismatch"):
        v3.main()
    monkeypatch.setattr(v3, "EXPECTED_PACKET_SHA256", v3.v2._sha256(packet_path))
    monkeypatch.setattr(v3, "EXPECTED_TARGETS_SOURCE_SHA256", "tampered")
    with pytest.raises(RuntimeError, match="AI-draft targets SHA mismatch"):
        v3.main()


def test_v3_frozen_contract_matches_selector_and_manifest() -> None:
    report_dir = Path("experiments/evals/reports/refusal_evidence_sufficiency")
    manifest = json.loads(
        (report_dir / "v3_ai_proxy_holdout_manifest.json").read_text()
    )
    contract = json.loads((report_dir / "v3_ai_proxy_run_contract.json").read_text())
    prompt_sha = hashlib.sha256(v3.v2.CLASSIFIER_SYSTEM_PROMPT_V2.encode()).hexdigest()
    artifacts = manifest["artifacts"]
    assert artifacts["blind_packet_sha256"] == v3.EXPECTED_PACKET_SHA256
    assert artifacts["source_targets_sha256"] == v3.EXPECTED_TARGETS_SOURCE_SHA256
    assert (
        contract["frozen_inputs"]["source_targets_sha256"]
        == v3.EXPECTED_TARGETS_SOURCE_SHA256
    )
    assert (
        contract["frozen_inputs"]["classifier_inputs_sha256"]
        == artifacts["holdout_inputs_sha256"]
        == v3.EXPECTED_HOLDOUT_INPUTS_SHA256
    )
    assert (
        contract["frozen_inputs"]["targets_sha256"]
        == artifacts["holdout_targets_sha256"]
    )
    assert (
        contract["classifier"]["prompt_sha256"]
        == artifacts["prompt_sha256"]
        == prompt_sha
    )
    assert (
        contract["population"]["selection_seed"]
        == manifest["selection"]["selection_seed"]
        == v3.SELECTION_SEED
    )
    assert (
        contract["population"]["sufficient"]
        == contract["population"]["insufficient"]
        == v3.PER_CLASS
    )
    assert (
        contract["classifier"]["required_predictions"]
        == manifest["selection"]["selected_total"]
        == 50
    )
    assert (
        contract["classifier"]["max_calls"]
        == 50 + contract["controls"]["failed_attempt_retry_budget"]
        == 55
    )
    assert contract["classifier"]["max_output_tokens"] == 512
    assert contract["gate"]["insufficient_recall_min"] == 0.70
    assert contract["gate"]["sufficient_to_insufficient_max"] == 3


def _review_inputs(tmp_path, monkeypatch):
    rows = [
        {
            "question_id": f"TRAIN_Q{index:03d}",
            "question": f"Exact question {index}\n第二行",
            "sources": [
                {
                    "source_id": f"Source {rank}",
                    "content": f"Exact evidence {index}:{rank}\n原文",
                    "doc_id": "must not leak",
                }
                for rank in range(1, 15)
            ],
            "target_class": "must not leak",
            "prediction": "must not leak",
            "gold_answer": "must not leak",
            "annotation": {"notes": "must not leak"},
        }
        for index in range(50)
    ]
    path = tmp_path / "inputs.jsonl"
    v3.v2._write_jsonl(path, rows)
    monkeypatch.setattr(v3, "EXPECTED_HOLDOUT_INPUTS_SHA256", v3.v2._sha256(path))
    return path, rows


def test_blind_review_cli_preserves_all_inputs_and_excludes_leaks(tmp_path, monkeypatch):
    inputs_path, rows = _review_inputs(tmp_path, monkeypatch)
    original = inputs_path.read_bytes()
    packet_path = tmp_path / "review.jsonl"
    manifest_path = tmp_path / "manifest.json"
    read_paths = []
    read_jsonl = v3.v2._read_jsonl

    def record_read(path):
        read_paths.append(Path(path))
        return read_jsonl(path)

    monkeypatch.setattr(v3.v2, "_read_jsonl", record_read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refusal_v3_ai_proxy",
            "--prepare-blind-review",
            "--inputs-path", str(inputs_path),
            "--review-packet-path", str(packet_path),
            "--review-manifest-path", str(manifest_path),
        ],
    )
    v3.main()
    assert read_paths == [inputs_path]
    packets = read_jsonl(packet_path)
    assert len(packets) == 50
    assert {row["question_id"] for row in packets} == {
        row["question_id"] for row in rows
    }
    by_id = {row["question_id"]: row for row in rows}
    for row in packets:
        original_row = by_id[row["question_id"]]
        assert set(row) == {"question_id", "question", "sources", "annotation"}
        assert row["question"] == original_row["question"]
        assert row["sources"] == [
            {key: source[key] for key in ("source_id", "content")}
            for source in original_row["sources"]
        ]
        assert row["annotation"] == {
            "target_class": None,
            "supporting_source_ids": [],
            "notes": "",
            "bounded_guidance": {
                "supported": None,
                "supporting_source_ids": [],
                "conditions_and_limits": "",
            },
        }
    assert "must not leak" not in packet_path.read_text(encoding="utf-8")
    assert inputs_path.read_bytes() == original
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifacts"]["packet_sha256"] == v3.v2._sha256(packet_path)
    assert manifest["purpose"] == "HISTORICAL_DIAGNOSTIC_NOT_FRESH_CONFIRMATION"
    assert all(value is False or value == 0 for value in manifest["controls"].values())
    frozen_outputs = (packet_path.read_bytes(), manifest_path.read_bytes())
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        v3.main()
    assert frozen_outputs == (packet_path.read_bytes(), manifest_path.read_bytes())
    v3.prepare_blind_review(inputs_path, tmp_path / "second.jsonl", tmp_path / "m2.json")
    assert (tmp_path / "second.jsonl").read_bytes() == packet_path.read_bytes()
    blocked_packet = tmp_path / "blocked.jsonl"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        v3.prepare_blind_review(inputs_path, blocked_packet, manifest_path)
    assert not blocked_packet.exists()
    monkeypatch.setattr(v3, "EXPECTED_HOLDOUT_INPUTS_SHA256", "tampered")
    with pytest.raises(RuntimeError, match="inputs SHA mismatch"):
        v3.prepare_blind_review(inputs_path, blocked_packet, tmp_path / "m3.json")
    assert not blocked_packet.exists()


@pytest.mark.parametrize("invalid", ["count", "duplicate", "rank", "empty_content"])
def test_blind_review_rejects_invalid_inputs_before_writing(tmp_path, monkeypatch, invalid):
    path, rows = _review_inputs(tmp_path, monkeypatch)
    if invalid == "count":
        rows.pop()
    elif invalid == "duplicate":
        rows[1]["question_id"] = rows[0]["question_id"]
    elif invalid == "rank":
        rows[0]["sources"][0]["source_id"] = "Source 14"
    else:
        rows[0]["sources"][0]["content"] = ""
    # A different path avoids overwriting even synthetic inputs.
    bad_path = tmp_path / "invalid.jsonl"
    v3.v2._write_jsonl(bad_path, rows)
    monkeypatch.setattr(v3, "EXPECTED_HOLDOUT_INPUTS_SHA256", v3.v2._sha256(bad_path))
    with pytest.raises(RuntimeError, match="unique historical|invalid ordered"):
        v3.prepare_blind_review(bad_path, tmp_path / "packet.jsonl", tmp_path / "m.json")
    assert not (tmp_path / "packet.jsonl").exists()
    assert not (tmp_path / "m.json").exists()
    assert path.exists()
