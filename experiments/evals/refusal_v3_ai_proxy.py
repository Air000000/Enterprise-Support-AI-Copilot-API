from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_v2_ai_proxy as v2

DEFAULT_PACKET_PATH = Path("data/refusal_v3_confirmation_set/annotation_packets.jsonl")
DEFAULT_TARGETS_SOURCE_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v3_confirmation_targets.jsonl"
)
DEFAULT_INPUTS_PATH = Path("data/refusal_v3_ai_proxy/holdout_inputs.jsonl")
DEFAULT_TARGETS_PATH = Path("data/refusal_v3_ai_proxy/holdout_targets.jsonl")
DEFAULT_MANIFEST_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v3_ai_proxy_holdout_manifest.json"
)

EXPECTED_PACKET_SHA256 = (
    "5afa8c766bccfb2f8a1e2703d6d7777ec37548accdb40c5d279541cb72db1ac1"
)
EXPECTED_TARGETS_SOURCE_SHA256 = (
    "5b0d8b5b167e8ae742210c2deb8ff91c711143bf8da10451bc15f835d9ca8d68"
)
SELECTION_SEED = "refusal-v3-ai-proxy-holdout-v1"
PER_CLASS = 25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the refusal v3 AI-proxy holdout inputs."
    )
    parser.add_argument("--packet-path", default=str(DEFAULT_PACKET_PATH))
    parser.add_argument(
        "--targets-source-path",
        default=str(DEFAULT_TARGETS_SOURCE_PATH),
    )
    parser.add_argument("--inputs-path", default=str(DEFAULT_INPUTS_PATH))
    parser.add_argument("--targets-path", default=str(DEFAULT_TARGETS_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    packet_path = Path(args.packet_path)
    targets_source_path = Path(args.targets_source_path)
    if v2._sha256(packet_path) != EXPECTED_PACKET_SHA256:
        raise RuntimeError("frozen v3 blind packet SHA mismatch")
    if v2._sha256(targets_source_path) != EXPECTED_TARGETS_SOURCE_SHA256:
        raise RuntimeError("frozen v3 AI-draft targets SHA mismatch")

    inputs, targets, selection = v2.build_ai_proxy_holdout(
        v2._read_jsonl(packet_path),
        v2._read_jsonl(targets_source_path),
        per_class=PER_CLASS,
        seed=SELECTION_SEED,
        excluded_question_ids=set(),
    )
    inputs_path = Path(args.inputs_path)
    targets_path = Path(args.targets_path)
    manifest_path = Path(args.manifest_path)
    existing = [
        str(path)
        for path in (inputs_path, targets_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise RuntimeError("refusing to overwrite: " + ", ".join(existing))

    v2._write_jsonl(inputs_path, inputs)
    v2._write_jsonl(targets_path, targets)
    manifest = {
        "schema_version": 1,
        "run": "refusal_v3_ai_proxy_holdout_v1",
        "status": "PREREGISTERED_INPUTS_FROZEN",
        "label_provenance": "AI_DRAFT_DEVELOPMENT_ONLY",
        "selection": selection,
        "artifacts": {
            "blind_packet_sha256": EXPECTED_PACKET_SHA256,
            "source_targets_sha256": EXPECTED_TARGETS_SOURCE_SHA256,
            "holdout_inputs_path": str(inputs_path).replace("\\", "/"),
            "holdout_inputs_sha256": v2._sha256(inputs_path),
            "holdout_targets_path": str(targets_path).replace("\\", "/"),
            "holdout_targets_sha256": v2._sha256(targets_path),
            "prompt_sha256": hashlib.sha256(
                v2.CLASSIFIER_SYSTEM_PROMPT_V2.encode()
            ).hexdigest(),
        },
        "controls": {
            "provider_calls": 0,
            "dev_artifact_opened": False,
            "classifier_predictions_created": False,
            "gold_answer_opened": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print("REFUSAL_V3_AI_PROXY_PREFLIGHT=PASS")
    print(f"HOLDOUT_CASES={len(inputs)}")
    print(f"INPUTS_SHA256={v2._sha256(inputs_path)}")
    print(f"TARGETS_SHA256={v2._sha256(targets_path)}")
    print(f"PROMPT_SHA256={manifest['artifacts']['prompt_sha256']}")
    print("PROVIDER_CALLS=0")
    print("DEV_ARTIFACT_OPENED=NO")


if __name__ == "__main__":
    main()
