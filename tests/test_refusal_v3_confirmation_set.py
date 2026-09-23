import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.evals import refusal_v2_confirmation_set as v2
from experiments.evals.refusal_v3_confirmation_set import (
    DEFAULT_V2_SELECTION_PATH,
    RUN_ID,
    _load_v2_selected_ids,
    build_blind_confirmation_pool,
)


def _case(question_id: str, *, gold_in_top_k: bool) -> tuple[dict, dict, dict]:
    gold = f"gold-{question_id}"
    other = f"other-{question_id}"
    documents = [gold, other] if gold_in_top_k else [other, f"other-2-{question_id}"]
    chunks = [f"{document}-chunk" for document in documents]
    return (
        {
            "id": question_id,
            "is_impossible": False,
            "question": f"Question {question_id}?",
            "answer": f"Hidden answer {question_id}",
        },
        {
            "question_id": question_id,
            "question": f"Question {question_id}?",
            "fused_candidates": [
                {
                    "chunk_id": chunk,
                    "document_id": document,
                    "content": f"Visible content {rank}",
                }
                for rank, (chunk, document) in enumerate(
                    zip(chunks, documents, strict=True),
                    start=1,
                )
            ],
        },
        {
            "question_id": question_id,
            "relevant_document_ids": [gold],
            "reranked_chunk_ids": chunks,
            "reranked_document_ids": documents,
        },
    )


def test_v3_excludes_the_frozen_v2_selection() -> None:
    assert len(_load_v2_selected_ids(DEFAULT_V2_SELECTION_PATH)) == 80


def test_v3_packet_is_fresh_and_physically_blind() -> None:
    rows = [
        _case(f"TRAIN_Q{index:03d}", gold_in_top_k=index < 4)
        for index in range(8)
    ]
    metadata, snapshots, results = map(list, zip(*rows, strict=True))
    packets, _, _ = build_blind_confirmation_pool(
        metadata,
        snapshots,
        results,
        excluded_question_ids={"TRAIN_Q000", "TRAIN_Q004"},
        per_stratum=2,
        top_k=2,
        seed="test-v3",
    )

    assert len(packets) == 4
    assert not {"TRAIN_Q000", "TRAIN_Q004"} & {
        packet["question_id"] for packet in packets
    }
    assert all(
        set(packet) == {"question_id", "question", "sources", "annotation"}
        for packet in packets
    )
    assert all(
        set(source) == {"source_id", "content"}
        for packet in packets
        for source in packet["sources"]
    )
    assert "Hidden answer" not in json.dumps(packets)


def test_v3_freeze_records_correct_run_and_rejects_payload_changes() -> None:
    rows = [
        _case(f"TRAIN_Q{index:03d}", gold_in_top_k=index < 2)
        for index in range(4)
    ]
    metadata, snapshots, results = map(list, zip(*rows, strict=True))
    packets, _, _ = build_blind_confirmation_pool(
        metadata,
        snapshots,
        results,
        excluded_question_ids=set(),
        per_stratum=2,
        top_k=2,
    )
    annotated = json.loads(json.dumps(packets))
    for index, packet in enumerate(annotated):
        packet["annotation"] = {
            "target_class": "SUFFICIENT" if index < 2 else "INSUFFICIENT",
            "supporting_source_ids": ["Source 1"] if index < 2 else [],
            "notes": "Visible evidence supports this label.",
        }

    with TemporaryDirectory(prefix=".test-refusal-v3-", dir=".") as directory:
        output = Path(directory)
        report = v2.validate_and_freeze_annotations(
            packets,
            annotated,
            annotation_source="human",
            targets_path=output / "targets.jsonl",
            freeze_path=output / "freeze.json",
            minimum_per_class=2,
            run=RUN_ID,
        )

        assert report["run"] == RUN_ID
        assert report["status"] == "TARGETS_FROZEN"
