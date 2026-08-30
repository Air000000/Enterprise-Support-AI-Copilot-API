from importlib import import_module

import pytest


def _load_module():
    try:
        return import_module(
            "experiments.evals.r4_c1_postmortem"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "R4 C1 postmortem is not implemented yet"
        )


def test_postmortem_separates_candidate_and_ranking_failures(tmp_path):
    module = _load_module()

    chunk_document_ids = {
        # Q1: Dense 和 BM25 都包含 gold1。
        "q1-d-gold": "gold1",
        "q1-b-gold": "gold1",
        "q1-shared": "other1",
        "q1-d-only": "other2",
        "q1-b-only": "other3",

        # Q2: Dense miss，BM25 把 gold2 补回来。
        "q2-d1": "other4",
        "q2-d2": "other5",
        "q2-shared": "other6",
        "q2-b-gold": "gold2",
        "q2-b-only": "other7",

        # Q3: Dense 原本有 gold3，但 fused Top-K 把它丢掉。
        "q3-d-gold": "gold3",
        "q3-d-only": "other8",
        "q3-shared": "other9",
        "q3-b-only": "other10",
        "q3-fused-extra": "other11",
    }

    snapshot_rows = [
        {
            "question_id": "TRAIN_Q1",
            "question": "q1",
            "relevant_document_ids": ["gold1"],
            "dense_chunk_ids": [
                "q1-d-gold",
                "q1-shared",
                "q1-d-only",
            ],
            "bm25_chunk_ids": [
                "q1-b-gold",
                "q1-shared",
                "q1-b-only",
            ],
            "fused_candidates": [
                {
                    "chunk_id": "q1-shared",
                    "document_id": "other1",
                    "content": "shared",
                },
                {
                    "chunk_id": "q1-d-gold",
                    "document_id": "gold1",
                    "content": "gold dense",
                },
                {
                    "chunk_id": "q1-b-gold",
                    "document_id": "gold1",
                    "content": "gold bm25",
                },
                {
                    "chunk_id": "q1-d-only",
                    "document_id": "other2",
                    "content": "dense only",
                },
            ],
        },
        {
            "question_id": "TRAIN_Q2",
            "question": "q2",
            "relevant_document_ids": ["gold2"],
            "dense_chunk_ids": [
                "q2-d1",
                "q2-d2",
                "q2-shared",
            ],
            "bm25_chunk_ids": [
                "q2-b-gold",
                "q2-shared",
                "q2-b-only",
            ],
            "fused_candidates": [
                {
                    "chunk_id": "q2-b-gold",
                    "document_id": "gold2",
                    "content": "rescued gold",
                },
                {
                    "chunk_id": "q2-shared",
                    "document_id": "other6",
                    "content": "shared",
                },
                {
                    "chunk_id": "q2-d1",
                    "document_id": "other4",
                    "content": "dense",
                },
            ],
        },
        {
            "question_id": "TRAIN_Q3",
            "question": "q3",
            "relevant_document_ids": ["gold3"],
            "dense_chunk_ids": [
                "q3-d-gold",
                "q3-d-only",
                "q3-shared",
            ],
            "bm25_chunk_ids": [
                "q3-shared",
                "q3-b-only",
                "q3-fused-extra",
            ],
            "fused_candidates": [
                {
                    "chunk_id": "q3-shared",
                    "document_id": "other9",
                    "content": "shared",
                },
                {
                    "chunk_id": "q3-b-only",
                    "document_id": "other10",
                    "content": "bm25 only",
                },
                {
                    "chunk_id": "q3-fused-extra",
                    "document_id": "other11",
                    "content": "other",
                },
            ],
        },
    ]

    result_rows = [
        {
            "question_id": "TRAIN_Q1",
            "relevant_document_ids": ["gold1"],
            "document_ranking": [
                "other1",
                "gold1",
                "other2",
            ],
        },
        {
            "question_id": "TRAIN_Q2",
            "relevant_document_ids": ["gold2"],
            "document_ranking": [
                *[f"q2-rank-{i}" for i in range(20)],
                "gold2",
            ],
        },
        {
            "question_id": "TRAIN_Q3",
            "relevant_document_ids": ["gold3"],
            "document_ranking": [
                "other9",
                "other10",
                "other11",
            ],
        },
    ]

    records = module.build_postmortem_records(
        snapshot_rows,
        result_rows,
        chunk_document_ids=chunk_document_ids,
    )

    by_id = {
        record.question_id: record
        for record in records
    }

    q1 = by_id["TRAIN_Q1"]
    assert q1.dense_gold_hit is True
    assert q1.bm25_gold_hit is True
    assert q1.fused_gold_hit is True
    assert q1.residual_bucket == "resolved_top20"
    assert q1.final_gold_rank == 2

    # 4 fused chunks 映射到 3 个 unique docs，
    # gold1 由两个 chunks 占据。
    assert q1.fused_unique_document_count == 3
    assert q1.fused_duplicate_ratio == pytest.approx(0.25)
    assert q1.max_chunks_per_document == 2
    assert q1.dense_only_chunk_count == 2
    assert q1.bm25_only_chunk_count == 1
    assert q1.shared_chunk_count == 1

    q2 = by_id["TRAIN_Q2"]
    assert q2.dense_gold_hit is False
    assert q2.bm25_gold_hit is True
    assert q2.fused_gold_hit is True
    assert q2.hybrid_rescued_dense_miss is True
    assert q2.residual_bucket == "ranking_miss_top20"
    assert q2.final_gold_rank == 21

    q3 = by_id["TRAIN_Q3"]
    assert q3.dense_gold_hit is True
    assert q3.bm25_gold_hit is False
    assert q3.fused_gold_hit is False
    assert q3.hybrid_lost_dense_hit is True
    assert q3.residual_bucket == "candidate_miss_top100"
    assert q3.final_gold_rank is None

    summary = module.summarize_postmortem(records)

    assert summary["query_count"] == 3
    assert summary["dense_gold_hit_count"] == 2
    assert summary["bm25_gold_hit_count"] == 2
    assert summary["fused_gold_hit_count"] == 2
    assert summary["hybrid_rescued_dense_misses"] == 1
    assert summary["hybrid_lost_dense_hits"] == 1
    assert summary["net_candidate_gain"] == 0

    assert summary["resolved_top20_count"] == 1
    assert summary["ranking_miss_top20_count"] == 1
    assert summary["candidate_miss_top100_count"] == 1


    # --------------------------------------------------------
    # Persisted zero-cost postmortem evidence
    # --------------------------------------------------------

    import json

    # Real TechQA chunk IDs are:
    # <document_id>_chunk_<chunk_index>.
    # rsplit must therefore preserve document IDs that themselves
    # contain "_chunk_".
    assert (
        module.document_id_from_chunk_id(
            "doc-with_chunk_name_chunk_7"
        )
        == "doc-with_chunk_name"
    )

    snapshot_path = tmp_path / "snapshot.jsonl"
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "postmortem"

    snapshot_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in snapshot_rows
        ),
        encoding="utf-8",
    )

    results_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in result_rows
        ),
        encoding="utf-8",
    )

    persisted = module.run_postmortem(
        snapshot_path=snapshot_path,
        results_path=results_path,
        output_dir=output_dir,
        chunk_document_ids=chunk_document_ids,
        expected_count=3,
    )

    # The analysis itself is explicitly zero-provider / TRAIN-only.
    assert persisted["query_count"] == 3
    assert persisted["provider_calls"] == 0
    assert persisted["dev_artifact_opened"] is False

    assert persisted["candidate_complementarity"] == {
        "dense_only_gold_hit_count": 1,
        "bm25_only_gold_hit_count": 1,
        "both_gold_hit_count": 1,
        "neither_gold_hit_count": 0,
        "hybrid_rescued_dense_misses": 1,
        "hybrid_lost_dense_hits": 1,
        "net_candidate_gain": 0,
    }

    assert persisted["residual_attribution"] == {
        "resolved_top20_count": 1,
        "ranking_miss_top20_count": 1,
        "candidate_miss_top100_count": 1,
    }

    assert persisted["source_composition"][
        "dense_only_chunk_count_total"
    ] == 3
    assert persisted["source_composition"][
        "bm25_only_chunk_count_total"
    ] == 4
    assert persisted["source_composition"][
        "shared_chunk_count_total"
    ] == 3

    # q1 has duplicate-document crowding; q2/q3 do not.
    assert persisted["crowding"][
        "fused_duplicate_ratio_p50"
    ] == pytest.approx(0.0)
    assert persisted["crowding"][
        "fused_duplicate_ratio_max"
    ] == pytest.approx(0.25)

    summary_path = output_dir / "postmortem_summary.json"
    cases_path = output_dir / "postmortem_cases.jsonl"
    report_path = output_dir / "postmortem.md"

    assert summary_path.exists()
    assert cases_path.exists()
    assert report_path.exists()

    cases = [
        json.loads(line)
        for line in cases_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(cases) == 3
    assert {
        row["question_id"]: row["residual_bucket"]
        for row in cases
    } == {
        "TRAIN_Q1": "resolved_top20",
        "TRAIN_Q2": "ranking_miss_top20",
        "TRAIN_Q3": "candidate_miss_top100",
    }

    report = report_path.read_text(encoding="utf-8")

    assert "## Candidate complementarity" in report
    assert "## Candidate miss vs ranking miss" in report
    assert "## Chunk crowding" in report
    assert "## Source composition" in report
    assert "## Interpretation boundary" in report
