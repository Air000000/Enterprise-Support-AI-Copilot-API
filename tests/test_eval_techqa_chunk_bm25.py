from importlib import import_module

import pytest


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