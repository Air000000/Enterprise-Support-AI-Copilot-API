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