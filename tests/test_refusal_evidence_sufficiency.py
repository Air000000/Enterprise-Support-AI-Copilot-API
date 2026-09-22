import copy

import pytest

from experiments.evals import refusal_evidence_sufficiency as refusal


def _freeze_contract() -> dict:
    return {
        "status": "FROZEN_FOR_PORTFOLIO_V1",
        "retrieval": {
            "dense_k": 100,
            "bm25_k": 100,
            "fusion": "equal_weight_chunk_rrf",
            "rrf_k": 60,
            "fused_k": 100,
            "reranker_model": "qwen3-rerank",
            "parameter_research_closed": True,
        },
        "context": {
            "policy": "flat_rerank_top14_v1",
            "top_k": 14,
            "dense_top1_rescue": False,
            "forward_siblings": 0,
            "locality_second_rerank": False,
            "whole_document_expansion": False,
            "section_expansion": False,
            "parent_child": False,
            "research_closed": True,
        },
        "audited_evidence": {
            "usable_cases": 54,
            "flat_top14": {
                "answer_hits": 35,
                "useful_hits": 43,
            },
        },
    }


def _snapshot(
    question_id: str,
    *,
    question: str = "How do I fix this?",
) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "fused_candidates": [
            {
                "chunk_id": f"doc_chunk_{index}",
                "document_id": "doc",
                "content": f"content {index}",
            }
            for index in range(4)
        ],
    }


def _result(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "reranked_chunk_ids": [
            "doc_chunk_2",
            "doc_chunk_0",
            "doc_chunk_3",
            "doc_chunk_1",
        ],
    }


def test_validate_freeze_contract_accepts_exact_frozen_contract():
    refusal.validate_freeze_contract(_freeze_contract())


def test_validate_freeze_contract_rejects_changed_rrf_k():
    payload = copy.deepcopy(_freeze_contract())
    payload["retrieval"]["rrf_k"] = 61

    with pytest.raises(RuntimeError, match="rrf_k"):
        refusal.validate_freeze_contract(payload)


def test_phase_b_cases_use_topk_answer_evidence_and_exclude_questionable():
    cases = refusal.build_phase_b_cases(
        [
            _snapshot("TRAIN_Q001"),
            _snapshot("TRAIN_Q002"),
        ],
        [
            _result("TRAIN_Q001"),
            _result("TRAIN_Q002"),
        ],
        [
            {
                "question_id": "TRAIN_Q001",
                "candidate_labels": [
                    {
                        "chunk_id": "doc_chunk_0",
                        "evidence_label": 2,
                    },
                ],
                "questionable_gold": False,
            },
            {
                "question_id": "TRAIN_Q002",
                "candidate_labels": [
                    {
                        "chunk_id": "doc_chunk_1",
                        "evidence_label": 2,
                    },
                ],
                "questionable_gold": True,
            },
        ],
        top_k=2,
    )

    assert len(cases) == 1
    assert cases[0].question_id == "TRAIN_Q001"
    assert cases[0].evidence_sufficient is True
    assert [
        source.chunk_id
        for source in cases[0].classifier_input.sources
    ] == [
        "doc_chunk_2",
        "doc_chunk_0",
    ]


def test_phase_b_cases_mark_missing_answer_chunk_insufficient():
    cases = refusal.build_phase_b_cases(
        [_snapshot("TRAIN_Q001")],
        [_result("TRAIN_Q001")],
        [
            {
                "question_id": "TRAIN_Q001",
                "candidate_labels": [
                    {
                        "chunk_id": "doc_chunk_1",
                        "evidence_label": 2,
                    },
                ],
                "questionable_gold": False,
            },
        ],
        top_k=2,
    )

    assert cases[0].evidence_sufficient is False


def test_classifier_payload_does_not_expose_labels_ids_or_gold_metadata():
    case = refusal.PhaseBCase(
        question_id="TRAIN_Q001",
        classifier_input=refusal.ClassifierInput(
            question="What is supported?",
            sources=(
                refusal.ContextSource(
                    source_id="Source 1",
                    chunk_id="secret_chunk",
                    document_id="secret_doc",
                    content="Relevant support text.",
                ),
            ),
        ),
        evidence_sufficient=True,
    )

    payload = refusal.classifier_payload(case.classifier_input)

    assert payload == {
        "question": "What is supported?",
        "sources": [
            {
                "source_id": "Source 1",
                "content": "Relevant support text.",
            },
        ],
    }

    prompt = "\n".join(
        message["content"]
        for message in refusal.build_classifier_messages(
            case.classifier_input
        )
    )

    assert "TRAIN_Q001" not in prompt
    assert "secret_chunk" not in prompt
    assert "secret_doc" not in prompt
    assert "evidence_sufficient" not in prompt

    refusal.assert_classifier_input_is_clean(case)


def test_phase_b_cases_reject_non_train_case():
    with pytest.raises(RuntimeError, match="non-TRAIN"):
        refusal.build_phase_b_cases(
            [_snapshot("DEV_Q001")],
            [_result("DEV_Q001")],
            [
                {
                    "question_id": "DEV_Q001",
                    "candidate_labels": [],
                    "questionable_gold": False,
                },
            ],
            top_k=2,
        )


def test_phase_b_cases_reject_reranked_chunk_outside_frozen_snapshot():
    result = _result("TRAIN_Q001")
    result["reranked_chunk_ids"][0] = "missing_chunk"

    with pytest.raises(
        RuntimeError,
        match="missing from frozen fused snapshot",
    ):
        refusal.build_phase_b_cases(
            [_snapshot("TRAIN_Q001")],
            [result],
            [
                {
                    "question_id": "TRAIN_Q001",
                    "candidate_labels": [],
                    "questionable_gold": False,
                },
            ],
            top_k=2,
        )
