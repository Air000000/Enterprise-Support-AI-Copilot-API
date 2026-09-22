import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from experiments.evals.refusal_v2_confirmation_set import (
    build_confirmation_pool,
    validate_and_freeze_annotations,
)


def _case(question_id: str, *, gold_in_top_k: bool) -> tuple[dict, dict, dict]:
    gold_document = f"gold-{question_id}"
    distractor_document = f"other-{question_id}"
    ranked_documents = (
        [gold_document, distractor_document]
        if gold_in_top_k
        else [distractor_document, f"other-2-{question_id}"]
    )
    ranked_chunks = [
        f"{document}-chunk" for document in ranked_documents
    ]
    metadata = {
        "id": question_id,
        "is_impossible": False,
        "question": f" Question {question_id}? ",
        "answer": f"Answer {question_id}",
    }
    snapshot = {
        "question_id": question_id,
        "question": f"Question   {question_id}?",
        "fused_candidates": [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content": f"Content {rank}",
            }
            for rank, (chunk_id, document_id) in enumerate(
                zip(ranked_chunks, ranked_documents, strict=True),
                start=1,
            )
        ],
    }
    result = {
        "question_id": question_id,
        "relevant_document_ids": [gold_document],
        "reranked_chunk_ids": ranked_chunks,
        "reranked_document_ids": ranked_documents,
    }
    return metadata, snapshot, result


def test_build_confirmation_pool_is_deterministic_and_fresh() -> None:
    rows = [
        _case(f"TRAIN_Q{index:03d}", gold_in_top_k=index < 3)
        for index in range(6)
    ]
    metadata, snapshots, results = map(list, zip(*rows, strict=True))

    first = build_confirmation_pool(
        metadata,
        snapshots,
        results,
        excluded_question_ids={"TRAIN_Q000"},
        per_stratum=2,
        top_k=2,
        seed="test-seed",
    )
    second = build_confirmation_pool(
        metadata,
        snapshots,
        results,
        excluded_question_ids={"TRAIN_Q000"},
        per_stratum=2,
        top_k=2,
        seed="test-seed",
    )

    assert first == second
    packets, compact_cases, population = first
    assert len(packets) == len(compact_cases) == 4
    assert "TRAIN_Q000" not in {packet["question_id"] for packet in packets}
    assert population == {
        "fresh_answerable_train": 5,
        "gold_document_in_top14": 2,
        "gold_document_outside_top14": 3,
        "selected_total": 4,
    }
    assert {
        packet["selection_stratum"] for packet in packets
    } == {"gold_document_in_top14", "gold_document_outside_top14"}
    assert all(packet["annotation"]["target_class"] is None for packet in packets)
    assert all(
        [source["source_id"] for source in packet["sources"]]
        == ["Source 1", "Source 2"]
        for packet in packets
    )


def test_validate_and_freeze_annotations_rejects_input_changes() -> None:
    rows = [
        _case(f"TRAIN_Q{index:03d}", gold_in_top_k=index < 2)
        for index in range(4)
    ]
    metadata, snapshots, results = map(list, zip(*rows, strict=True))
    packets, _, _ = build_confirmation_pool(
        metadata,
        snapshots,
        results,
        excluded_question_ids=set(),
        per_stratum=2,
        top_k=2,
    )
    annotated = json.loads(json.dumps(packets))
    for index, packet in enumerate(annotated):
        sufficient = index < 2
        packet["annotation"] = {
            "target_class": "SUFFICIENT" if sufficient else "INSUFFICIENT",
            "supporting_source_ids": ["Source 1"] if sufficient else [],
            "notes": "Direct support." if sufficient else "Material support missing.",
        }

    with TemporaryDirectory(prefix=".test-refusal-v2-", dir=".") as directory:
        output_dir = Path(directory)
        report = validate_and_freeze_annotations(
            packets,
            annotated,
            annotation_source="human",
            targets_path=output_dir / "targets.jsonl",
            freeze_path=output_dir / "freeze.json",
            minimum_per_class=2,
        )
        assert report["status"] == "TARGETS_FROZEN"
        assert report["human_confirmation_population_eligible"] is True
        assert report["counts"] == {
            "INSUFFICIENT": 2,
            "QUESTIONABLE": 0,
            "SUFFICIENT": 2,
        }
        targets = [
            json.loads(line)
            for line in (output_dir / "targets.jsonl").read_text().splitlines()
        ]
        assert set(targets[0]) == {
            "question_id",
            "target_class",
            "supporting_source_ids",
            "notes",
        }

        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            validate_and_freeze_annotations(
                packets,
                annotated,
                annotation_source="human",
                targets_path=output_dir / "targets.jsonl",
                freeze_path=output_dir / "freeze.json",
                minimum_per_class=2,
            )

        ai_report = validate_and_freeze_annotations(
            packets,
            annotated,
            annotation_source="ai-draft",
            targets_path=output_dir / "ai-targets.jsonl",
            freeze_path=output_dir / "ai-freeze.json",
            minimum_per_class=2,
        )
        assert ai_report["status"] == (
            "AI_DRAFT_TARGETS_FROZEN_FOR_DEVELOPMENT"
        )
        assert ai_report["human_confirmation_population_eligible"] is False

        annotated[0]["question"] = "Changed question"
        with pytest.raises(RuntimeError, match="frozen packet content changed"):
            validate_and_freeze_annotations(
                packets,
                annotated,
                annotation_source="human",
                targets_path=output_dir / "changed-targets.jsonl",
                freeze_path=output_dir / "changed-freeze.json",
                minimum_per_class=2,
            )
