from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
EXPECTED_HOLDOUT_INPUTS_SHA256 = (
    "87f91ba55b78bfc16db2c018b092212e39cf9d171b52e62853a5e536b6fd344d"
)
DEFAULT_REVIEW_PACKET_PATH = Path("data/refusal_v3_blind_review/annotation_packets.jsonl")
DEFAULT_REVIEW_MANIFEST_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/v3_blind_review_manifest.json"
)
REVIEW_ORDER_SEED = "refusal-v3-blind-review-order-v1"
EXPECTED_REVIEW_PACKET_SHA256 = (
    "54ce28f647df741128614f3b604265ac430abf2d2033ab04c051592462a2e310"
)
DEFAULT_AI_REVIEWS_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v3_codex_review_annotations.jsonl"
)
DEFAULT_ANNOTATED_REVIEW_PATH = Path(
    "data/refusal_v3_blind_review/annotation_packets_annotated_ai_draft.jsonl"
)


def materialize_ai_review(packet_path, reviews_path, output_path) -> dict:
    """Apply development-only annotations; never import or rescore old targets."""
    packet_path, reviews_path, output_path = map(
        Path, (packet_path, reviews_path, output_path)
    )
    if v2._sha256(packet_path) != EXPECTED_REVIEW_PACKET_SHA256:
        raise RuntimeError("frozen review packet SHA mismatch")
    packets = v2._read_jsonl(packet_path)
    reviews = v2._read_jsonl(reviews_path)
    ids = [row["question_id"] for row in packets]
    by_id = {row["question_id"]: row["annotation"] for row in reviews}
    if (
        len(packets) != 50 or len(set(ids)) != 50
        or len(reviews) != 50 or len(by_id) != 50 or set(by_id) != set(ids)
        or any(set(row) != {"question_id", "annotation"} for row in reviews)
    ):
        raise RuntimeError("expected matching 50 unique review IDs")
    for row in packets:
        annotation = by_id[row["question_id"]]
        blank = row["annotation"]
        if not isinstance(annotation, dict) or set(annotation) != set(blank):
            raise RuntimeError("invalid annotation schema")
        label, notes = annotation["target_class"], annotation["notes"]
        guidance = annotation["bounded_guidance"]
        if (
            not isinstance(label, str)
            or label not in {"SUFFICIENT", "INSUFFICIENT", "QUESTIONABLE"}
            or not isinstance(notes, str) or not notes.strip()
            or not isinstance(guidance, dict)
            or set(guidance) != set(blank["bounded_guidance"])
        ):
            raise RuntimeError("invalid annotation class, notes or guidance")
        supported, limits = guidance["supported"], guidance["conditions_and_limits"]
        if (
            supported is not None and type(supported) is not bool
            or not isinstance(limits, str)
            or supported is True and not limits.strip()
            or supported is False and limits != ""
            or supported is None and label != "QUESTIONABLE"
        ):
            raise RuntimeError("invalid bounded guidance conditions")
        allowed = {source["source_id"] for source in row["sources"]}
        for citations, required in (
            (annotation["supporting_source_ids"], label == "SUFFICIENT"),
            (guidance["supporting_source_ids"], supported is True),
        ):
            if (
                not isinstance(citations, list)
                or any(not isinstance(cite, str) or cite not in allowed for cite in citations)
                or len(set(citations)) != len(citations)
                or required and not citations
            ):
                raise RuntimeError("invalid Source N citations")
        if supported is not True and guidance["supporting_source_ids"]:
            raise RuntimeError("unsupported guidance cannot cite sources")
        row["annotation"] = annotation
    # ponytail: one merge into the frozen form, not a new labeling/runner framework.
    if output_path.exists():
        raise RuntimeError("refusing to overwrite annotated review")
    v2._write_jsonl(output_path, packets)
    return {
        "cases": len(packets),
        "classes": dict(Counter(row["annotation"]["target_class"] for row in packets)),
        "bounded_guidance": dict(Counter(
            str(row["annotation"]["bounded_guidance"]["supported"]).lower()
            for row in packets
        )),
        "packet_sha256": v2._sha256(packet_path),
        "annotations_sha256": v2._sha256(reviews_path),
        "annotated_packet_sha256": v2._sha256(output_path),
    }


def prepare_blind_review(inputs_path, packet_path, manifest_path) -> dict:
    """Package all historical inputs, without reading labels or predictions."""
    inputs_path, packet_path, manifest_path = map(
        Path, (inputs_path, packet_path, manifest_path)
    )
    if v2._sha256(inputs_path) != EXPECTED_HOLDOUT_INPUTS_SHA256:
        raise RuntimeError("frozen v3 holdout inputs SHA mismatch")
    rows = v2._read_jsonl(inputs_path)
    ids = [row["question_id"] for row in rows]
    if len(rows) != 50 or len(set(ids)) != 50:
        raise RuntimeError("expected 50 unique historical cases")
    packets = []
    for row in rows:
        if (
            not isinstance(row["question_id"], str)
            or not row["question_id"].startswith("TRAIN_Q")
            or not isinstance(row["question"], str)
            or not row["question"].strip()
            or len(row["sources"]) != 14
        ):
            raise RuntimeError("invalid TRAIN question or Top14")
        sources = []
        for rank, source in enumerate(row["sources"], 1):
            if (
                source["source_id"] != f"Source {rank}"
                or not isinstance(source["content"], str)
                or not source["content"].strip()
            ):
                raise RuntimeError("invalid ordered Source N content")
            sources.append(
                {"source_id": source["source_id"], "content": source["content"]}
            )
        packets.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "sources": sources,
                "annotation": {
                    "target_class": None,
                    "supporting_source_ids": [],
                    "notes": "",
                    "bounded_guidance": {
                        "supported": None,
                        "supporting_source_ids": [],
                        "conditions_and_limits": "",
                    },
                },
            }
        )
    # ponytail: shuffle presentation only; no sampling or new confirmation set.
    packets.sort(
        key=lambda row: hashlib.sha256(
            f"{REVIEW_ORDER_SEED}:{row['question_id']}".encode()
        ).hexdigest()
    )
    if packet_path.resolve() == manifest_path.resolve() or any(
        path.exists() for path in (packet_path, manifest_path)
    ):
        raise RuntimeError("refusing to overwrite review output")
    v2._write_jsonl(packet_path, packets)
    manifest = {
        "schema_version": 1,
        "status": "BLIND_REVIEW_PACKET_READY_NOT_ANNOTATED",
        "purpose": "HISTORICAL_DIAGNOSTIC_NOT_FRESH_CONFIRMATION",
        "cases": len(packets),
        "review_order_seed": REVIEW_ORDER_SEED,
        "artifacts": {
            "input_sha256": EXPECTED_HOLDOUT_INPUTS_SHA256,
            "packet_path": packet_path.as_posix(),
            "packet_sha256": v2._sha256(packet_path),
        },
        "controls": {
            "provider_calls": 0,
            "cost_cny": 0,
            "targets_opened": False,
            "predictions_opened": False,
            "audit_opened": False,
            "gold_answer_opened": False,
            "dev_opened": False,
            "historical_result_changed": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


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
    parser.add_argument("--prepare-blind-review", action="store_true")
    parser.add_argument("--materialize-ai-review", action="store_true")
    parser.add_argument("--ai-reviews-path", default=str(DEFAULT_AI_REVIEWS_PATH))
    parser.add_argument("--annotated-review-path", default=str(DEFAULT_ANNOTATED_REVIEW_PATH))
    parser.add_argument("--review-packet-path", default=str(DEFAULT_REVIEW_PACKET_PATH))
    parser.add_argument("--review-manifest-path", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    args = parser.parse_args()

    if args.materialize_ai_review:
        if args.prepare_blind_review:
            parser.error("choose one review operation")
        result = materialize_ai_review(
            args.review_packet_path, args.ai_reviews_path, args.annotated_review_path
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.prepare_blind_review:
        manifest = prepare_blind_review(
            args.inputs_path, args.review_packet_path, args.review_manifest_path
        )
        print(f"BLIND_REVIEW_CASES={manifest['cases']}")
        print(f"PACKET_SHA256={manifest['artifacts']['packet_sha256']}")
        print("PROVIDER_CALLS=0")
        print("DEV_OPENED=NO")
        return

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
