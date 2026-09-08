from __future__ import annotations

import importlib
import importlib.util
import hashlib

import pytest

from experiments.evals.adapters.techqa import TechQAGenerationCase
from experiments.evals.eval_techqa_generation import G0RetrievedChunk
from experiments.evals.rerankers.qwen3_reranker import (
    RerankCandidate,
    RerankResult,
    RerankedCandidate,
)


MODULE_NAME = "experiments.evals.eval_techqa_g2_admission"


def _g2_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, "eval_techqa_g2_admission is not implemented"
    return importlib.import_module(MODULE_NAME)


def _g2_runner():
    runner = getattr(_g2_module(), "run_g2_admission_diagnostic", None)
    assert runner is not None, "run_g2_admission_diagnostic is not implemented"
    return runner


def _comparison_runner():
    runner = getattr(_g2_module(), "run_g1_g2_admission_comparison", None)
    assert runner is not None, "run_g1_g2_admission_comparison is not implemented"
    return runner


def _shared_rerank(*document_ids: str) -> RerankResult:
    return RerankResult(
        results=tuple(
            RerankedCandidate(
                chunk_id=f"chunk-{index}",
                document_id=document_id,
                content=f"content-{index}",
                original_index=index,
                relevance_score=1.0 - index / 100.0,
            )
            for index, document_id in enumerate(document_ids)
        ),
        request_id="shared-global-rerank",
        total_tokens=123,
    )


def _chunk(
    chunk_id: str,
    document_id: str,
    *,
    chunk_index: int = 0,
    distance: float = 0.1,
) -> G0RetrievedChunk:
    return G0RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=f"content {chunk_id}",
        distance=distance,
    )


def _generation_case(
    question_id: str,
    *,
    answerable: bool = True,
    split: str = "train",
) -> TechQAGenerationCase:
    return TechQAGenerationCase(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer",
        answerable=answerable,
        split=split,  # type: ignore[arg-type]
    )


def _g2_preregistered_case_selector():
    selector = getattr(_g2_module(), "select_g2_preregistered_cases", None)
    assert selector is not None, "select_g2_preregistered_cases is not implemented"
    return selector


def test_g2_preregistered_selection_uses_only_eligible_train_cases_before_hashing():
    selector = _g2_preregistered_case_selector()
    cases = (
        _generation_case("TRAIN_ELIGIBLE_DEEP"),
        _generation_case("TRAIN_IMPOSSIBLE", answerable=False),
        _generation_case("TRAIN_NON_TRAIN_SPLIT", split="dev"),
        _generation_case("TRAIN_MISSING_TRACE"),
        _generation_case("TRAIN_GOLD_ABSENT"),
        _generation_case("TRAIN_EXCLUDED"),
    )
    traces = {
        "TRAIN_ELIGIBLE_DEEP": (
            _chunk("first", "other-document-1"),
            _chunk("second", "other-document-2"),
            _chunk("third", "other-document-3"),
            _chunk("fourth", "other-document-4"),
            _chunk("fifth", "other-document-5"),
            _chunk("sixth", "gold-document"),
        ),
        "TRAIN_IMPOSSIBLE": (_chunk("impossible", "gold-document"),),
        "TRAIN_NON_TRAIN_SPLIT": (
            _chunk("non-train", "gold-document"),
        ),
        "TRAIN_GOLD_ABSENT": (_chunk("only", "other-document"),),
        "TRAIN_EXCLUDED": (_chunk("excluded", "gold-document"),),
    }
    relevant_documents = {
        "TRAIN_ELIGIBLE_DEEP": "gold-document",
        "TRAIN_IMPOSSIBLE": "gold-document",
        "TRAIN_NON_TRAIN_SPLIT": "gold-document",
        "TRAIN_GOLD_ABSENT": "gold-document",
        "TRAIN_EXCLUDED": "gold-document",
    }

    assert selector(
        cases,
        e0_trace_by_question_id=traces,
        relevant_document_by_question_id=relevant_documents,
        excluded_question_ids={"TRAIN_EXCLUDED"},
        sample_size=1,
    ) == ("TRAIN_ELIGIBLE_DEEP",)


def test_g2_preregistered_selection_hashes_eligible_ids_and_is_repeatable():
    selector = _g2_preregistered_case_selector()
    eligible_ids = tuple(f"TRAIN_Q{index:03d}" for index in range(31))
    cases = tuple(_generation_case(question_id) for question_id in eligible_ids)
    traces = {
        question_id: (_chunk(f"{question_id}-chunk", "gold-document"),)
        for question_id in eligible_ids
    }
    relevant_documents = {question_id: "gold-document" for question_id in eligible_ids}
    excluded_question_ids = {"TRAIN_Q000"}
    seed = "selection-seed"
    expected = tuple(
        sorted(
            (question_id for question_id in eligible_ids if question_id not in excluded_question_ids),
            key=lambda question_id: (
                hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest(),
                question_id,
            ),
        )[:30]
    )

    first = selector(
        cases,
        e0_trace_by_question_id=traces,
        relevant_document_by_question_id=relevant_documents,
        excluded_question_ids=excluded_question_ids,
        seed=seed,
    )
    second = selector(
        cases,
        e0_trace_by_question_id=traces,
        relevant_document_by_question_id=relevant_documents,
        excluded_question_ids=excluded_question_ids,
        seed=seed,
    )

    assert first == second == expected
    assert "TRAIN_Q000" not in first


def test_g2_preregistered_selection_rejects_insufficient_eligible_population():
    selector = _g2_preregistered_case_selector()
    case = _generation_case("TRAIN_ONLY_CASE")

    with pytest.raises(ValueError, match="Insufficient eligible"):
        selector(
            (case,),
            e0_trace_by_question_id={
                case.question_id: (_chunk("gold", "gold-document"),),
            },
            relevant_document_by_question_id={case.question_id: "gold-document"},
            excluded_question_ids=(),
            sample_size=2,
        )


def test_rerank_informed_admission_uses_first_unique_docs_in_rerank_order():
    module = _g2_module()
    selector = getattr(module, "select_rerank_informed_document_ids", None)
    assert selector is not None, "select_rerank_informed_document_ids is not implemented"

    result = _shared_rerank("D3", "D3", "D1", "D5", "D2", "D4", "D6")

    assert selector(result, limit=5) == ("D3", "D1", "D5", "D2", "D4")


def test_rerank_informed_admission_returns_all_unique_docs_when_under_limit():
    module = _g2_module()
    selector = getattr(module, "select_rerank_informed_document_ids", None)
    assert selector is not None, "select_rerank_informed_document_ids is not implemented"

    result = _shared_rerank("D2", "D2", "D1")

    assert selector(result, limit=5) == ("D2", "D1")


@pytest.mark.parametrize("limit", [0, -1])
def test_rerank_informed_admission_requires_positive_limit(limit: int):
    module = _g2_module()
    selector = getattr(module, "select_rerank_informed_document_ids", None)
    assert selector is not None, "select_rerank_informed_document_ids is not implemented"

    with pytest.raises(ValueError, match="limit must be positive"):
        selector(_shared_rerank("D1"), limit=limit)


def test_g2_diagnostic_admits_global_rerank_docs_and_calls_merged_reranker_once():
    runner = _g2_runner()
    dense_ranked = tuple(
        _chunk(f"dense-{index}", document_id, distance=index / 100.0)
        for index, document_id in enumerate(("D1", "D2", "D3", "D4", "D5", "D9"))
    )
    shared_global_rerank = _shared_rerank("D9", "D8", "D7", "D6", "D5", "D4")
    loader_calls: list[str] = []
    chunks_by_document = {
        document_id: (_chunk(f"{document_id}-chunk-0", document_id),)
        for document_id in ("D9", "D8", "D7", "D6", "D5")
    }
    merged_calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    def load_document_chunks(document_id: str):
        loader_calls.append(document_id)
        return chunks_by_document[document_id]

    def merged_reranker(question: str, candidates):
        candidate_tuple = tuple(candidates)
        merged_calls.append((question, candidate_tuple))
        return RerankResult(
            results=tuple(
                RerankedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    content=candidate.content,
                    original_index=index,
                    relevance_score=1.0 - index / 100.0,
                )
                for index, candidate in reversed(tuple(enumerate(candidate_tuple)))
            ),
            request_id="merged-rerank",
            total_tokens=456,
        )

    diagnostic = runner(
        "TRAIN_SYNTHETIC",
        "Which evidence resolves the issue?",
        dense_ranked,
        shared_global_rerank=shared_global_rerank,
        candidate_document_limit=5,
        load_document_chunks=load_document_chunks,
        candidate_pool_max_chunks=500,
        merged_reranker=merged_reranker,
        evidence_limit=16,
        max_context_chunks=16,
    )

    assert diagnostic.candidate_document_ids == ("D9", "D8", "D7", "D6", "D5")
    assert diagnostic.candidate_document_ids != ("D1", "D2", "D3", "D4", "D5")
    assert loader_calls == ["D9", "D8", "D7", "D6", "D5"]
    assert len(merged_calls) == 1
    assert diagnostic.merged_rerank_invocations == 1
    assert tuple(chunk.document_id for chunk in diagnostic.final_context) == (
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
    )


def test_g2_diagnostic_reuses_existing_g1_downstream_helpers(monkeypatch):
    module = _g2_module()
    runner = _g2_runner()
    calls: list[tuple[str, object]] = []
    pool = (_chunk("pool-1", "D9"),)
    selected = (_chunk("selected-1", "D9"),)
    final = (_chunk("final-1", "D9"),)

    def fake_pool_builder(candidate_document_ids, *, load_document_chunks, max_chunks):
        calls.append(("pool", (tuple(candidate_document_ids), max_chunks)))
        return pool

    def fake_evidence_selector(question, candidate_pool, *, reranker, limit):
        calls.append(("evidence", (question, tuple(candidate_pool), limit)))
        return selected

    def fake_assembler(selected_evidence, *, max_context_chunks):
        calls.append(("assembly", (tuple(selected_evidence), max_context_chunks)))
        return final

    monkeypatch.setattr(module, "build_document_local_candidate_pool", fake_pool_builder, raising=False)
    monkeypatch.setattr(module, "select_document_local_evidence", fake_evidence_selector, raising=False)
    monkeypatch.setattr(module, "assemble_selected_evidence_context", fake_assembler, raising=False)

    diagnostic = runner(
        "TRAIN_HELPERS",
        "question",
        (_chunk("dense-1", "D1"),),
        shared_global_rerank=_shared_rerank("D9"),
        candidate_document_limit=5,
        load_document_chunks=lambda _: (),
        candidate_pool_max_chunks=500,
        merged_reranker=lambda *_: None,
        evidence_limit=16,
        max_context_chunks=16,
    )

    assert calls == [
        ("pool", (("D9",), 500)),
        ("evidence", ("question", pool, 16)),
        ("assembly", (selected, 16)),
    ]
    assert diagnostic.candidate_pool == pool
    assert diagnostic.selected_evidence == selected
    assert diagnostic.final_context == final


@pytest.mark.parametrize(
    "invalid_budget",
    (
        "candidate_document_limit",
        "candidate_pool_max_chunks",
        "evidence_limit",
        "max_context_chunks",
    ),
)
def test_g2_diagnostic_validates_all_budgets_before_side_effects(invalid_budget: str):
    runner = _g2_runner()
    side_effects: list[str] = []
    budgets = {
        "candidate_document_limit": 5,
        "candidate_pool_max_chunks": 500,
        "evidence_limit": 16,
        "max_context_chunks": 16,
    }
    budgets[invalid_budget] = 0

    def forbidden_loader(_: str):
        side_effects.append("loader")
        raise AssertionError("invalid budgets must fail before document loading")

    def forbidden_reranker(*_):
        side_effects.append("reranker")
        raise AssertionError("invalid budgets must fail before reranking")

    with pytest.raises(ValueError, match="positive"):
        runner(
            "TRAIN_INVALID",
            "question",
            (_chunk("dense-1", "D1"),),
            shared_global_rerank=_shared_rerank("D9"),
            load_document_chunks=forbidden_loader,
            merged_reranker=forbidden_reranker,
            **budgets,
        )

    assert side_effects == []


def test_g1_g2_comparison_changes_only_document_admission_order():
    runner = _comparison_runner()
    dense_ranked = tuple(
        _chunk(f"dense-{index}", document_id, distance=index / 100.0)
        for index, document_id in enumerate(("D1", "D2", "D3", "D4", "D5", "D9"))
    )
    shared_global_rerank = _shared_rerank("D9", "D8", "D7", "D6", "D5", "D4")
    chunks_by_document = {
        document_id: (_chunk(f"{document_id}-chunk-0", document_id),)
        for document_id in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")
    }

    def load_document_chunks(document_id: str):
        return chunks_by_document[document_id]

    def merged_reranker(question: str, candidates):
        candidate_tuple = tuple(candidates)
        return RerankResult(
            results=tuple(
                RerankedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    content=candidate.content,
                    original_index=index,
                    relevance_score=1.0 - index / 100.0,
                )
                for index, candidate in enumerate(candidate_tuple)
            ),
            request_id="merged-rerank",
            total_tokens=456,
        )

    comparison = runner(
        "TRAIN_CAUSAL",
        "question",
        dense_ranked,
        shared_global_rerank=shared_global_rerank,
        load_document_chunks=load_document_chunks,
        merged_reranker=merged_reranker,
    )

    assert comparison.question_id == "TRAIN_CAUSAL"
    assert comparison.dense_candidate_chunk_ids == tuple(
        chunk.chunk_id for chunk in dense_ranked
    )
    assert comparison.g1_candidate_document_ids == ("D1", "D2", "D3", "D4", "D5")
    assert comparison.g2_candidate_document_ids == ("D9", "D8", "D7", "D6", "D5")
    assert tuple(chunk.document_id for chunk in comparison.g1_context) == (
        "D1", "D2", "D3", "D4", "D5"
    )
    assert tuple(chunk.document_id for chunk in comparison.g2_context) == (
        "D9", "D8", "D7", "D6", "D5"
    )


def test_g1_g2_comparison_reuses_identical_downstream_contract(monkeypatch):
    module = _g2_module()
    runner = _comparison_runner()
    dense_ranked = (
        _chunk("dense-1", "D1"),
        _chunk("dense-2", "D2"),
    )
    shared_global_rerank = _shared_rerank("D9", "D8")
    calls: list[tuple[str, object]] = []

    def load_document_chunks(document_id: str):
        return (_chunk(f"{document_id}-loaded", document_id),)

    def merged_reranker(question: str, candidates):
        candidate_tuple = tuple(candidates)
        return RerankResult(
            results=tuple(
                RerankedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    content=candidate.content,
                    original_index=index,
                    relevance_score=1.0 - index / 100.0,
                )
                for index, candidate in enumerate(candidate_tuple)
            ),
            request_id="merged-rerank",
            total_tokens=456,
        )

    def fake_pool_builder(candidate_document_ids, *, load_document_chunks, max_chunks):
        document_ids = tuple(candidate_document_ids)
        calls.append(("pool", (document_ids, load_document_chunks, max_chunks)))
        return tuple(_chunk(f"{document_id}-pool", document_id) for document_id in document_ids)

    def fake_evidence_selector(question, candidate_pool, *, reranker, limit):
        pool = tuple(candidate_pool)
        calls.append(("evidence", (question, pool, reranker, limit)))
        return pool

    def fake_assembler(selected_evidence, *, max_context_chunks):
        selected = tuple(selected_evidence)
        calls.append(("assembly", (selected, max_context_chunks)))
        return selected

    monkeypatch.setattr(module, "build_document_local_candidate_pool", fake_pool_builder, raising=False)
    monkeypatch.setattr(module, "select_document_local_evidence", fake_evidence_selector, raising=False)
    monkeypatch.setattr(module, "assemble_selected_evidence_context", fake_assembler, raising=False)

    comparison = runner(
        "TRAIN_CONTRACT",
        "question",
        dense_ranked,
        shared_global_rerank=shared_global_rerank,
        load_document_chunks=load_document_chunks,
        merged_reranker=merged_reranker,
    )

    assert comparison.g1_candidate_document_ids == ("D1", "D2")
    assert comparison.g2_candidate_document_ids == ("D9", "D8")
    assert len(calls) == 6

    g1_pool_call, g1_evidence_call, g1_assembly_call = calls[:3]
    g2_pool_call, g2_evidence_call, g2_assembly_call = calls[3:]

    assert g1_pool_call[0] == g2_pool_call[0] == "pool"
    assert g1_pool_call[1][1] is g2_pool_call[1][1] is load_document_chunks
    assert g1_pool_call[1][2] == g2_pool_call[1][2] == 500

    assert g1_evidence_call[0] == g2_evidence_call[0] == "evidence"
    assert g1_evidence_call[1][0] == g2_evidence_call[1][0] == "question"
    assert g1_evidence_call[1][2] is g2_evidence_call[1][2] is merged_reranker
    assert g1_evidence_call[1][3] == g2_evidence_call[1][3] == 16

    assert g1_assembly_call[0] == g2_assembly_call[0] == "assembly"
    assert g1_assembly_call[1][1] == g2_assembly_call[1][1] == 16
