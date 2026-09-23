import hashlib
import json
from pathlib import Path

from experiments.evals.refusal_v2_ai_proxy import (
    CLASSIFIER_SYSTEM_PROMPT_V2,
    build_ai_proxy_holdout,
)


def test_build_ai_proxy_holdout_is_deterministic_and_clean() -> None:
    packets = []
    targets = []
    for index in range(13):
        question_id = f"TRAIN_Q{index:03d}"
        packets.append(
            {
                "question_id": question_id,
                "question": f"Question {index}",
                "gold_answer": "must not leak",
                "selection_stratum": "must not leak",
                "sources": [
                    {
                        "source_id": f"Source {rank}",
                        "document_id": "must not leak",
                        "content": f"Content {rank}",
                    }
                    for rank in range(1, 15)
                ],
            }
        )
        targets.append(
            {
                "question_id": question_id,
                "target_class": (
                    "SUFFICIENT"
                    if index < 6
                    else "INSUFFICIENT"
                    if index < 12
                    else "QUESTIONABLE"
                ),
            }
        )

    first = build_ai_proxy_holdout(
        packets,
        targets,
        per_class=2,
        seed="test-seed",
        excluded_question_ids={"TRAIN_Q000", "TRAIN_Q006"},
    )
    second = build_ai_proxy_holdout(
        packets,
        targets,
        per_class=2,
        seed="test-seed",
        excluded_question_ids={"TRAIN_Q000", "TRAIN_Q006"},
    )

    assert first == second
    inputs, holdout_targets, manifest = first
    assert len(inputs) == len(holdout_targets) == 4
    assert manifest["questionable_excluded"] == 1
    assert not {"TRAIN_Q000", "TRAIN_Q006"} & {
        row["question_id"] for row in inputs
    }
    assert set(inputs[0]) == {"question_id", "question", "sources"}
    assert set(inputs[0]["sources"][0]) == {"source_id", "content"}
    assert {row["target_class"] for row in holdout_targets} == {
        "SUFFICIENT",
        "INSUFFICIENT",
    }


def test_frozen_contract_matches_manifest_and_prompt() -> None:
    report_dir = Path(
        "experiments/evals/reports/refusal_evidence_sufficiency"
    )
    manifest = json.loads(
        (report_dir / "v2_ai_proxy_holdout_manifest.json").read_text()
    )
    contract = json.loads(
        (report_dir / "v2_ai_proxy_run_contract.json").read_text()
    )
    prompt_sha256 = hashlib.sha256(
        CLASSIFIER_SYSTEM_PROMPT_V2.encode()
    ).hexdigest()

    assert contract["frozen_inputs"]["classifier_inputs_sha256"] == (
        manifest["artifacts"]["holdout_inputs_sha256"]
    )
    assert contract["frozen_inputs"]["targets_sha256"] == (
        manifest["artifacts"]["holdout_targets_sha256"]
    )
    assert contract["classifier"]["prompt_sha256"] == prompt_sha256
    assert manifest["artifacts"]["prompt_sha256"] == prompt_sha256
