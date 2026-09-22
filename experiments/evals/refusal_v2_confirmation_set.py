from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_SNAPSHOT_PATH = Path(
    "experiments/evals/reports/r4_c1_hybrid_rerank/"
    "train_fused_snapshot.jsonl"
)
DEFAULT_RESULTS_PATH = Path(
    "experiments/evals/reports/r4_c1_hybrid_rerank/"
    "train_results.jsonl"
)
DEFAULT_PRIOR_LABELS_PATH = Path(
    "experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl"
)
DEFAULT_PACKET_PATH = Path(
    "data/refusal_v2_confirmation_set/annotation_packets.jsonl"
)
DEFAULT_SELECTION_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v2_confirmation_selection.json"
)

EXPECTED_METADATA_SHA256 = (
    "69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "12db56e50efaf11dab4a28ff3c1b4df223e2ad985a8b48021f4c7dd9fdd889d2"
)
EXPECTED_RESULTS_SHA256 = (
    "12b312a8403cda2c5fe8afe53aa18853891a7e87feaf863e2d438ca15367ee5b"
)
EXPECTED_PRIOR_LABELS_SHA256 = (
    "d522eff8daba8435d34ee0e51ad8c56fbbbb759b3f71dbf3a253cd9a1b506013"
)
SELECTION_SEED = "refusal-v2-confirmation-set-v1"
PER_STRATUM = 40
CONTEXT_TOP_K = 14


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in indexed:
            raise RuntimeError(f"duplicate question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def _selection_hash(question_id: str, *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def build_confirmation_pool(
    metadata_rows: Sequence[Mapping[str, Any]],
    snapshot_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    excluded_question_ids: set[str],
    per_stratum: int = PER_STRATUM,
    top_k: int = CONTEXT_TOP_K,
    seed: str = SELECTION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    metadata = {
        str(row["id"]): row
        for row in metadata_rows
        if str(row["id"]).startswith("TRAIN_")
        and not bool(row["is_impossible"])
    }
    snapshots = _index(snapshot_rows)
    results = _index(result_rows)
    strata: dict[str, list[tuple[str, str]]] = {
        "gold_document_in_top14": [],
        "gold_document_outside_top14": [],
    }

    for question_id, result in results.items():
        if question_id in excluded_question_ids:
            continue
        if question_id not in metadata or question_id not in snapshots:
            raise RuntimeError(f"missing frozen TRAIN data for {question_id}")

        relevant_document_ids = tuple(
            str(value) for value in result["relevant_document_ids"]
        )
        if len(relevant_document_ids) != 1:
            raise RuntimeError(f"expected one gold document for {question_id}")
        reranked_document_ids = [
            str(value) for value in result["reranked_document_ids"]
        ]
        if len(reranked_document_ids) < top_k:
            raise RuntimeError(f"rerank result shorter than TopK: {question_id}")

        stratum = (
            "gold_document_in_top14"
            if relevant_document_ids[0] in reranked_document_ids[:top_k]
            else "gold_document_outside_top14"
        )
        strata[stratum].append(
            (_selection_hash(question_id, seed=seed), question_id)
        )

    for values in strata.values():
        values.sort()
        if len(values) < per_stratum:
            raise RuntimeError("not enough fresh TRAIN cases for each stratum")

    selected = sorted(
        (
            (selection_hash, question_id, stratum)
            for stratum, values in strata.items()
            for selection_hash, question_id in values[:per_stratum]
        ),
        key=lambda item: item[1],
    )

    packets: list[dict[str, Any]] = []
    compact_cases: list[dict[str, Any]] = []
    for selection_hash, question_id, stratum in selected:
        metadata_row = metadata[question_id]
        snapshot = snapshots[question_id]
        result = results[question_id]
        if _normalize_text(str(metadata_row["question"])) != _normalize_text(
            str(snapshot["question"])
        ):
            raise RuntimeError(f"question text mismatch: {question_id}")

        candidates = {
            str(candidate["chunk_id"]): candidate
            for candidate in snapshot["fused_candidates"]
        }
        sources: list[dict[str, str]] = []
        for rank, chunk_id_value in enumerate(
            result["reranked_chunk_ids"][:top_k],
            start=1,
        ):
            chunk_id = str(chunk_id_value)
            candidate = candidates.get(chunk_id)
            if candidate is None:
                raise RuntimeError(
                    f"reranked chunk missing from snapshot: {chunk_id}"
                )
            sources.append(
                {
                    "source_id": f"Source {rank}",
                    "chunk_id": chunk_id,
                    "document_id": str(candidate["document_id"]),
                    "content": str(candidate["content"]),
                }
            )

        packets.append(
            {
                "question_id": question_id,
                "selection_stratum": stratum,
                "question": str(snapshot["question"]),
                "gold_answer": str(metadata_row["answer"]),
                "relevant_document_ids": [
                    str(value) for value in result["relevant_document_ids"]
                ],
                "sources": sources,
                "annotation": {
                    "target_class": None,
                    "supporting_source_ids": [],
                    "notes": "",
                },
            }
        )
        compact_cases.append(
            {
                "question_id": question_id,
                "selection_stratum": stratum,
                "selection_sha256": selection_hash,
            }
        )

    population = {
        "fresh_answerable_train": sum(len(values) for values in strata.values()),
        "gold_document_in_top14": len(strata["gold_document_in_top14"]),
        "gold_document_outside_top14": len(
            strata["gold_document_outside_top14"]
        ),
        "selected_total": len(selected),
    }
    return packets, compact_cases, population


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fresh TRAIN-only refusal v2 confirmation pool."
    )
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument(
        "--prior-labels-path",
        default=str(DEFAULT_PRIOR_LABELS_PATH),
    )
    parser.add_argument("--packet-path", default=str(DEFAULT_PACKET_PATH))
    parser.add_argument("--selection-path", default=str(DEFAULT_SELECTION_PATH))
    args = parser.parse_args()

    expected_hashes = {
        Path(args.metadata_path): EXPECTED_METADATA_SHA256,
        Path(args.snapshot_path): EXPECTED_SNAPSHOT_SHA256,
        Path(args.results_path): EXPECTED_RESULTS_SHA256,
        Path(args.prior_labels_path): EXPECTED_PRIOR_LABELS_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen artifact SHA mismatch: {path}={actual}")

    metadata_rows = json.loads(Path(args.metadata_path).read_text(encoding="utf-8"))
    if not isinstance(metadata_rows, list) or len(metadata_rows) != 910:
        raise RuntimeError("frozen TechQA generation metadata changed")
    snapshot_rows = _read_jsonl(args.snapshot_path)
    result_rows = _read_jsonl(args.results_path)
    prior_label_rows = _read_jsonl(args.prior_labels_path)
    if len(snapshot_rows) != 450 or len(result_rows) != 450:
        raise RuntimeError("frozen TRAIN retrieval artifacts changed")
    if len(prior_label_rows) != 60:
        raise RuntimeError("prior evidence-audit population changed")

    packets, compact_cases, population = build_confirmation_pool(
        metadata_rows,
        snapshot_rows,
        result_rows,
        excluded_question_ids={
            str(row["question_id"]) for row in prior_label_rows
        },
    )
    packet_path = Path(args.packet_path)
    write_jsonl(packet_path, packets)

    selection = {
        "schema_version": 1,
        "run": "refusal_v2_confirmation_set_v1",
        "status": "SELECTED_NOT_ANNOTATED",
        "selection": {
            "seed": SELECTION_SEED,
            "algorithm": "lowest_sha256(seed + ':' + question_id)",
            "per_stratum": PER_STRATUM,
            "context_top_k": CONTEXT_TOP_K,
            "excluded_prior_audit_cases": 60,
        },
        "source_sha256": {
            "metadata": EXPECTED_METADATA_SHA256,
            "fused_snapshot": EXPECTED_SNAPSHOT_SHA256,
            "rerank_results": EXPECTED_RESULTS_SHA256,
            "prior_evidence_labels": EXPECTED_PRIOR_LABELS_SHA256,
        },
        "population": population,
        "selected_cases": compact_cases,
        "annotation_packet": {
            "path": str(packet_path).replace("\\", "/"),
            "rows": len(packets),
            "sha256": _sha256(packet_path),
            "contains_gold_answer_for_annotation_only": True,
            "classifier_eligible": False,
        },
        "controls": {
            "provider_calls": 0,
            "dev_artifact_opened": False,
            "classifier_predictions_created": False,
        },
    }
    selection_path = Path(args.selection_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("REFUSAL_V2_CONFIRMATION_SELECTION=COMPLETE")
    print(f"SELECTED_CASES={len(packets)}")
    print(f"ANNOTATION_PACKET_SHA256={_sha256(packet_path)}")
    print("PROVIDER_CALLS=0")
    print("DEV_ARTIFACT_OPENED=NO")
    print("NEXT_ACTION=ANNOTATE_AND_FREEZE_TARGETS")


if __name__ == "__main__":
    main()
