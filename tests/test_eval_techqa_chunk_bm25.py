from importlib import import_module

import pytest
from experiments.evals.adapters.techqa import TechQARetrievalCase

def _load_module():
    try:
        return import_module("experiments.evals.eval_techqa_chunk_bm25")
    except ModuleNotFoundError:
        pytest.fail("chunk BM25 audit evaluator is not implemented yet")


def test_cutoff_observation_measures_duplicate_slot_pressure() -> None:
    module = _load_module()

    observation = module.build_cutoff_observation(
        raw_document_ids=[
            "a",
            "a",
            "a",
            "b",
            "c",
            "gold",
        ],
        relevant_document_ids=["gold"],
        cutoff=5,
    )

    assert observation.returned_chunk_count == 5
    assert observation.unique_document_count == 3
    assert observation.duplicate_slot_count == 2
    assert observation.duplicate_ratio == pytest.approx(0.4)
    assert observation.gold_document_hit_within_chunk_k is False

def test_paired_ranks_identify_crowding_rescue() -> None:
    module = _load_module()

    raw_document_ids = [
        "a",
        "a",
        "a",
        "b",
        "a",
        "c",
        "d",
        "e",
        "f",
        "gold",
    ]

    collapsed = module.collapse_document_ids(raw_document_ids)

    assert module.first_relevant_rank(
        raw_document_ids,
        ["gold"],
    ) == 10

    assert module.first_relevant_rank(
        collapsed,
        ["gold"],
    ) == 7

    observation = module.build_cutoff_observation(
        raw_document_ids=raw_document_ids,
        relevant_document_ids=["gold"],
        cutoff=8,
    )

    assert observation.gold_document_hit_within_chunk_k is False
    assert observation.crowding_rescue is True

def test_adaptive_depth_expands_without_qrels_until_100_unique_documents() -> None:
    module = _load_module()
    calls: list[int] = []

    def searcher(
        query: str,
        top_k: int,
    ):
        calls.append(top_k)

        if top_k == 500:
            return [
                module.TechQAChunkRef(
                    f"a_chunk_{index}",
                    "a",
                    index,
                )
                for index in range(500)
            ]

        return [
            module.TechQAChunkRef(
                f"doc{index}_chunk_0",
                f"doc{index}",
                0,
            )
            for index in range(100)
        ]

    clock_values = iter(
        [
            1.000,
            1.010,
            2.000,
            2.030,
        ]
    )

    result = module.retrieve_with_unique_document_depth(
        "query",
        searcher=searcher,
        clock=lambda: next(clock_values),
        max_depth=2000,
    )

    assert calls == [500, 1000]
    assert result.final_search_depth == 1000
    assert len(
        module.collapse_document_ids(
            [
                item.document_id
                for item in result.candidates
            ]
        )
    ) == 100
    assert result.latency_ms == pytest.approx(40.0)

def test_evaluate_audit_case_keeps_raw_chunk_and_unique_document_budgets_separate() -> None:
    module = _load_module()

    case = TechQARetrievalCase(
        question_id="TRAIN_Q1",
        question="Error 0x80070005   \n",
        relevant_document_ids=("gold",),
        split="train",
    )

    candidates = [
        module.TechQAChunkRef("a0", "a", 0),
        module.TechQAChunkRef("a1", "a", 1),
        module.TechQAChunkRef("b0", "b", 0),
        module.TechQAChunkRef("gold0", "gold", 0),
    ] + [
        module.TechQAChunkRef(
            f"d{index}_chunk_0",
            f"d{index}",
            0,
        )
        for index in range(100)
    ]

    def searcher(
        query: str,
        top_k: int,
    ):
        assert query == "Error 0x80070005"
        return candidates[:top_k]

    clock_values = iter([1.000, 1.010])

    result = module.evaluate_audit_case(
        case,
        searcher=searcher,
        clock=lambda: next(clock_values),
        max_depth=len(candidates),
    )

    assert result.raw_top100_document_ids[:4] == (
        "a",
        "a",
        "b",
        "gold",
    )

    assert result.document_top100[:3] == (
        "a",
        "b",
        "gold",
    )

    assert result.first_gold_chunk_rank == 4
    assert result.first_gold_document_rank == 3
    assert result.crowding_gap == 1

    assert result.cutoff_observations[0].cutoff == 20

def test_build_audit_summary_aggregates_diversity_coverage_and_document_recall() -> None:
    module = _load_module()

    def observation(
        cutoff: int,
        *,
        unique_document_count: int,
        duplicate_ratio: float,
        gold_hit: bool,
        crowding_rescue: bool,
    ):
        returned_chunk_count = cutoff
        duplicate_slot_count = round(
            returned_chunk_count * duplicate_ratio
        )

        return module.ChunkCutoffObservation(
            cutoff=cutoff,
            returned_chunk_count=returned_chunk_count,
            unique_document_count=unique_document_count,
            duplicate_slot_count=duplicate_slot_count,
            duplicate_ratio=duplicate_ratio,
            gold_document_hit_within_chunk_k=gold_hit,
            crowding_rescue=crowding_rescue,
        )

    results = [
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q1",
            question="q1",
            relevant_document_ids=("gold1",),
            audit_search_depth=500,
            latency_ms=10.0,
            raw_top100_chunk_ids=("c1",),
            raw_top100_document_ids=("gold1",),
            document_top100=("gold1", "x"),
            first_gold_chunk_rank=1,
            first_gold_document_rank=1,
            crowding_gap=0,
            cutoff_observations=(
                observation(
                    20,
                    unique_document_count=18,
                    duplicate_ratio=0.10,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
                observation(
                    50,
                    unique_document_count=40,
                    duplicate_ratio=0.20,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
                observation(
                    100,
                    unique_document_count=75,
                    duplicate_ratio=0.25,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
            ),
        ),
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q2",
            question="q2",
            relevant_document_ids=("gold2",),
            audit_search_depth=1000,
            latency_ms=30.0,
            raw_top100_chunk_ids=("c2",),
            raw_top100_document_ids=("x",),
            document_top100=("x", "gold2"),
            first_gold_chunk_rank=25,
            first_gold_document_rank=2,
            crowding_gap=23,
            cutoff_observations=(
                observation(
                    20,
                    unique_document_count=10,
                    duplicate_ratio=0.50,
                    gold_hit=False,
                    crowding_rescue=True,
                ),
                observation(
                    50,
                    unique_document_count=30,
                    duplicate_ratio=0.40,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
                observation(
                    100,
                    unique_document_count=60,
                    duplicate_ratio=0.40,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
            ),
        ),
    ]

    summary = module.build_audit_summary(results)

    assert summary["query_count"] == 2

    cutoff20 = summary["cutoffs"]["20"]

    assert (
        cutoff20["unique_document_count_p05"]
        <= cutoff20["unique_document_count_p50"]
    )
    assert (
        cutoff20["duplicate_ratio_p95"]
        >= cutoff20["duplicate_ratio_p50"]
    )

    assert cutoff20["gold_document_hit_count"] == 1
    assert cutoff20["gold_document_hit_rate"] == pytest.approx(0.5)

    assert cutoff20["crowding_rescue_count"] == 1
    assert cutoff20["crowding_rescue_rate"] == pytest.approx(0.5)

    assert summary["collapsed_document_recall"] == {
        "recall@20": pytest.approx(1.0),
        "recall@50": pytest.approx(1.0),
        "recall@100": pytest.approx(1.0),
    }

    assert summary["crowding_gap"]["observed_count"] == 2
    assert summary["latency_ms"]["p50"] == pytest.approx(20.0)
    assert summary["latency_ms"]["p95"] > summary["latency_ms"]["p50"]

def test_build_diagnostic_cases_orders_high_duplication_and_crowding_rescue_deterministically() -> None:
    module = _load_module()

    def make_observation(
        cutoff: int,
        *,
        duplicate_ratio: float,
        crowding_rescue: bool,
    ):
        returned_chunk_count = cutoff
        unique_document_count = round(
            returned_chunk_count * (1.0 - duplicate_ratio)
        )

        return module.ChunkCutoffObservation(
            cutoff=cutoff,
            returned_chunk_count=returned_chunk_count,
            unique_document_count=unique_document_count,
            duplicate_slot_count=(
                returned_chunk_count - unique_document_count
            ),
            duplicate_ratio=duplicate_ratio,
            gold_document_hit_within_chunk_k=(
                not crowding_rescue
            ),
            crowding_rescue=crowding_rescue,
        )

    results = [
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q1",
            question="q1",
            relevant_document_ids=("gold1",),
            audit_search_depth=500,
            latency_ms=10.0,
            raw_top100_chunk_ids=("q1_c1",),
            raw_top100_document_ids=("a",),
            document_top100=("a", "gold1"),
            first_gold_chunk_rank=80,
            first_gold_document_rank=10,
            crowding_gap=70,
            cutoff_observations=(
                make_observation(
                    20,
                    duplicate_ratio=0.20,
                    crowding_rescue=True,
                ),
                make_observation(
                    50,
                    duplicate_ratio=0.30,
                    crowding_rescue=False,
                ),
                make_observation(
                    100,
                    duplicate_ratio=0.40,
                    crowding_rescue=False,
                ),
            ),
        ),
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q2",
            question="q2",
            relevant_document_ids=("gold2",),
            audit_search_depth=500,
            latency_ms=20.0,
            raw_top100_chunk_ids=("q2_c1",),
            raw_top100_document_ids=("b",),
            document_top100=("b", "gold2"),
            first_gold_chunk_rank=95,
            first_gold_document_rank=15,
            crowding_gap=80,
            cutoff_observations=(
                make_observation(
                    20,
                    duplicate_ratio=0.50,
                    crowding_rescue=True,
                ),
                make_observation(
                    50,
                    duplicate_ratio=0.40,
                    crowding_rescue=False,
                ),
                make_observation(
                    100,
                    duplicate_ratio=0.60,
                    crowding_rescue=False,
                ),
            ),
        ),
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q0",
            question="q0",
            relevant_document_ids=("gold0",),
            audit_search_depth=500,
            latency_ms=30.0,
            raw_top100_chunk_ids=("q0_c1",),
            raw_top100_document_ids=("c",),
            document_top100=("c", "gold0"),
            first_gold_chunk_rank=90,
            first_gold_document_rank=10,
            crowding_gap=80,
            cutoff_observations=(
                make_observation(
                    20,
                    duplicate_ratio=0.10,
                    crowding_rescue=True,
                ),
                make_observation(
                    50,
                    duplicate_ratio=0.20,
                    crowding_rescue=False,
                ),
                make_observation(
                    100,
                    duplicate_ratio=0.60,
                    crowding_rescue=False,
                ),
            ),
        ),
    ]

    diagnostics = module.build_diagnostic_cases(
        results,
        limit_per_group=10,
    )

    assert list(diagnostics) == [
        "high_duplication_cases",
        "crowding_rescue_cases",
    ]

    assert [
        row["question_id"]
        for row in diagnostics["high_duplication_cases"]
    ] == [
        "TRAIN_Q0",
        "TRAIN_Q2",
        "TRAIN_Q1",
    ]

    assert [
        row["question_id"]
        for row in diagnostics["crowding_rescue_cases"]
    ] == [
        "TRAIN_Q0",
        "TRAIN_Q2",
        "TRAIN_Q1",
    ]

    assert all(
        row["crowding_rescue"] is True
        for row in diagnostics["crowding_rescue_cases"]
    )

    assert diagnostics["crowding_rescue_cases"][0]["cutoff"] == 20
    assert diagnostics["crowding_rescue_cases"][0]["crowding_gap"] == 80