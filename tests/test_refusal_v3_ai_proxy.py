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
