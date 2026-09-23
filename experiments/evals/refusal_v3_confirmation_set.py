from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.evals import refusal_v2_confirmation_set as v2

DEFAULT_V2_SELECTION_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v2_confirmation_selection.json"
)
DEFAULT_PACKET_PATH = Path(
    "data/refusal_v3_confirmation_set/annotation_packets.jsonl"
)
DEFAULT_SELECTION_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v3_confirmation_selection.json"
)
DEFAULT_TARGETS_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v3_confirmation_targets.jsonl"
)
DEFAULT_ANNOTATION_FREEZE_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v3_confirmation_annotation_freeze.json"
)

EXPECTED_V2_SELECTION_SHA256 = (
    "721779d07ade0197c026febb735070c2bd1b4838e3de90ebb977eed16fa4b8da"
)
SELECTION_SEED = "refusal-v3-confirmation-set-v1"
PER_STRATUM = 40
CONTEXT_TOP_K = 14
RUN_ID = "refusal_v3_confirmation_annotation_v1"


def _load_v2_selected_ids(path: str | Path) -> set[str]:
    source = Path(path)
    if v2._sha256(source) != EXPECTED_V2_SELECTION_SHA256:
        raise RuntimeError("v2 selection SHA mismatch")
    selection = json.loads(source.read_text(encoding="utf-8"))
    if selection.get("run") != "refusal_v2_confirmation_set_v1":
        raise RuntimeError("unexpected v2 selection run")
    selected = selection.get("selected_cases")
    if not isinstance(selected, list) or len(selected) != 80:
        raise RuntimeError("v2 selection must contain 80 cases")
    question_ids = {str(row["question_id"]) for row in selected}
    if len(question_ids) != len(selected):
        raise RuntimeError("duplicate v2 selected question_id")
    return question_ids


def build_blind_confirmation_pool(
    metadata_rows: Sequence[Mapping[str, Any]],
    snapshot_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    excluded_question_ids: set[str],
    per_stratum: int = PER_STRATUM,
    top_k: int = CONTEXT_TOP_K,
    seed: str = SELECTION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rich_packets, compact_cases, population = v2.build_confirmation_pool(
        metadata_rows,
        snapshot_rows,
        result_rows,
        excluded_question_ids=excluded_question_ids,
        per_stratum=per_stratum,
        top_k=top_k,
        seed=seed,
    )
    packets = [
        {
            "question_id": str(packet["question_id"]),
            "question": str(packet["question"]),
            "sources": [
                {
                    "source_id": str(source["source_id"]),
                    "content": str(source["content"]),
                }
                for source in packet["sources"]
            ],
            "annotation": {
                "target_class": None,
                "supporting_source_ids": [],
                "notes": "",
            },
        }
        for packet in rich_packets
    ]
    return packets, compact_cases, population


def _validate_frozen_selection(
    selection_path: str | Path,
    packets: Sequence[Mapping[str, Any]],
    compact_cases: Sequence[Mapping[str, Any]],
) -> None:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if (
        selection.get("run") != "refusal_v3_confirmation_set_v1"
        or selection.get("status") != "SELECTED_NOT_ANNOTATED"
        or selection.get("selected_cases") != list(compact_cases)
        or selection.get("annotation_packet", {}).get("sha256")
        != v2._jsonl_sha256(packets)
    ):
        raise RuntimeError("v3 frozen selection changed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the blind TRAIN-only refusal v3 confirmation pool."
    )
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--snapshot-path", default=str(v2.DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--results-path", default=str(v2.DEFAULT_RESULTS_PATH))
    parser.add_argument(
        "--prior-labels-path",
        default=str(v2.DEFAULT_PRIOR_LABELS_PATH),
    )
    parser.add_argument(
        "--v2-selection-path",
        default=str(DEFAULT_V2_SELECTION_PATH),
    )
    parser.add_argument("--packet-path", default=str(DEFAULT_PACKET_PATH))
    parser.add_argument("--selection-path", default=str(DEFAULT_SELECTION_PATH))
    parser.add_argument(
        "--freeze-annotated-packet",
        help="Validate a completed blind packet and freeze compact targets.",
    )
    parser.add_argument(
        "--annotation-source",
        choices=("human", "ai-draft"),
        help="Required provenance for a completed annotation packet.",
    )
    parser.add_argument("--targets-path", default=str(DEFAULT_TARGETS_PATH))
    parser.add_argument(
        "--annotation-freeze-path",
        default=str(DEFAULT_ANNOTATION_FREEZE_PATH),
    )
    args = parser.parse_args()

    expected_hashes = {
        Path(args.metadata_path): v2.EXPECTED_METADATA_SHA256,
        Path(args.snapshot_path): v2.EXPECTED_SNAPSHOT_SHA256,
        Path(args.results_path): v2.EXPECTED_RESULTS_SHA256,
        Path(args.prior_labels_path): v2.EXPECTED_PRIOR_LABELS_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = v2._sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen artifact SHA mismatch: {path}={actual}")

    metadata_rows = json.loads(
        Path(args.metadata_path).read_text(encoding="utf-8")
    )
    snapshot_rows = v2._read_jsonl(args.snapshot_path)
    result_rows = v2._read_jsonl(args.results_path)
    prior_label_rows = v2._read_jsonl(args.prior_labels_path)
    if not isinstance(metadata_rows, list) or len(metadata_rows) != 910:
        raise RuntimeError("frozen TechQA generation metadata changed")
    if len(snapshot_rows) != 450 or len(result_rows) != 450:
        raise RuntimeError("frozen TRAIN retrieval artifacts changed")
    if len(prior_label_rows) != 60:
        raise RuntimeError("prior evidence-audit population changed")

    v2_selected_ids = _load_v2_selected_ids(args.v2_selection_path)
    excluded_question_ids = v2_selected_ids | {
        str(row["question_id"]) for row in prior_label_rows
    }
    packets, compact_cases, population = build_blind_confirmation_pool(
        metadata_rows,
        snapshot_rows,
        result_rows,
        excluded_question_ids=excluded_question_ids,
    )

    if args.freeze_annotated_packet:
        if args.annotation_source is None:
            parser.error("--annotation-source is required when freezing")
        _validate_frozen_selection(args.selection_path, packets, compact_cases)
        report = v2.validate_and_freeze_annotations(
            packets,
            v2._read_jsonl(args.freeze_annotated_packet),
            annotation_source=args.annotation_source,
            targets_path=Path(args.targets_path),
            freeze_path=Path(args.annotation_freeze_path),
            run=RUN_ID,
        )
        print(f"REFUSAL_V3_ANNOTATION_FREEZE={report['status']}")
        print(f"ANNOTATION_SOURCE={report['annotation_source']}")
        print(f"ANNOTATED_CASES={report['rows']}")
        print(f"SUFFICIENT_CASES={report['counts']['SUFFICIENT']}")
        print(f"INSUFFICIENT_CASES={report['counts']['INSUFFICIENT']}")
        print(f"QUESTIONABLE_CASES={report['counts']['QUESTIONABLE']}")
        print(f"TARGETS_SHA256={report['targets']['sha256']}")
        print("PROVIDER_CALLS=0")
        print("DEV_ARTIFACT_OPENED=NO")
        return

    packet_path = Path(args.packet_path)
    selection_path = Path(args.selection_path)
    existing = [
        str(path) for path in (packet_path, selection_path) if path.exists()
    ]
    if existing:
        raise RuntimeError("refusing to overwrite: " + ", ".join(existing))

    v2.write_jsonl(packet_path, packets)
    selection = {
        "schema_version": 1,
        "run": "refusal_v3_confirmation_set_v1",
        "status": "SELECTED_NOT_ANNOTATED",
        "selection": {
            "seed": SELECTION_SEED,
            "algorithm": "lowest_sha256(seed + ':' + question_id)",
            "per_stratum": PER_STRATUM,
            "context_top_k": CONTEXT_TOP_K,
            "excluded_prior_audit_cases": 60,
            "excluded_v2_confirmation_cases": 80,
        },
        "source_sha256": {
            "metadata": v2.EXPECTED_METADATA_SHA256,
            "fused_snapshot": v2.EXPECTED_SNAPSHOT_SHA256,
            "rerank_results": v2.EXPECTED_RESULTS_SHA256,
            "prior_evidence_labels": v2.EXPECTED_PRIOR_LABELS_SHA256,
            "v2_selection": EXPECTED_V2_SELECTION_SHA256,
        },
        "population": population,
        "selected_cases": compact_cases,
        "annotation_contract": {
            "target": "answerability_from_visible_question_and_top14",
            "gold_answer_visible": False,
            "gold_document_ids_visible": False,
            "selection_stratum_visible": False,
            "source_chunk_and_document_ids_visible": False,
        },
        "annotation_packet": {
            "path": str(packet_path).replace("\\", "/"),
            "rows": len(packets),
            "sha256": v2._sha256(packet_path),
            "classifier_eligible": False,
        },
        "controls": {
            "provider_calls": 0,
            "dev_artifact_opened": False,
            "classifier_predictions_created": False,
        },
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("REFUSAL_V3_CONFIRMATION_SELECTION=COMPLETE")
    print(f"SELECTED_CASES={len(packets)}")
    print(f"ANNOTATION_PACKET_SHA256={v2._sha256(packet_path)}")
    print("GOLD_ANSWER_VISIBLE=NO")
    print("PROVIDER_CALLS=0")
    print("DEV_ARTIFACT_OPENED=NO")
    print("NEXT_ACTION=BLIND_ANNOTATION_AND_FREEZE_TARGETS")


if __name__ == "__main__":
    main()
