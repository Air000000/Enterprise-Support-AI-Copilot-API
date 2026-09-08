from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.evals.eval_techqa_generation import (
    G0RetrievedChunk,
    assemble_selected_evidence_context,
    build_document_local_candidate_pool,
    select_candidate_document_ids,
    select_document_local_evidence,
)
from experiments.evals.rerankers.qwen3_reranker import RerankResult


@dataclass(frozen=True)
class G2AdmissionDiagnostic:
    question_id: str
    question: str
    dense_ranked_chunks: tuple[G0RetrievedChunk, ...]
    candidate_document_ids: tuple[str, ...]
    candidate_pool: tuple[G0RetrievedChunk, ...]
    selected_evidence: tuple[G0RetrievedChunk, ...]
    final_context: tuple[G0RetrievedChunk, ...]
    merged_rerank_invocations: int


@dataclass(frozen=True)
class G1G2AdmissionComparison:
    question_id: str
    dense_candidate_chunk_ids: tuple[str, ...]
    g1_candidate_document_ids: tuple[str, ...]
    g2_candidate_document_ids: tuple[str, ...]
    g1_context: tuple[G0RetrievedChunk, ...]
    g2_context: tuple[G0RetrievedChunk, ...]


def select_rerank_informed_document_ids(
    rerank_result: RerankResult,
    *,
    limit: int,
) -> tuple[str, ...]:
    """Select the first unique document IDs in shared global rerank order."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    selected: list[str] = []
    seen: set[str] = set()

    for candidate in rerank_result.results:
        document_id = candidate.document_id
        if document_id in seen:
            continue

        seen.add(document_id)
        selected.append(document_id)
        if len(selected) >= limit:
            break

    return tuple(selected)


def run_g1_g2_admission_comparison(
    question_id: str,
    question: str,
    dense_ranked_chunks: Sequence[G0RetrievedChunk],
    *,
    shared_global_rerank: RerankResult,
    candidate_document_limit: int,
    load_document_chunks: Callable[[str], Sequence[G0RetrievedChunk]],
    candidate_pool_max_chunks: int,
    merged_reranker: Callable[..., Any],
    evidence_limit: int,
    max_context_chunks: int,
) -> G1G2AdmissionComparison:
    """Compare G1 and G2 document admission with identical downstream steps."""
    dense_ranked = tuple(dense_ranked_chunks)
    g1_candidate_document_ids = select_candidate_document_ids(
        dense_ranked,
        limit=candidate_document_limit,
    )
    g2_candidate_document_ids = select_rerank_informed_document_ids(
        shared_global_rerank,
        limit=candidate_document_limit,
    )

    g1_candidate_pool = build_document_local_candidate_pool(
        g1_candidate_document_ids,
        load_document_chunks=load_document_chunks,
        max_chunks=candidate_pool_max_chunks,
    )
    g1_selected_evidence = select_document_local_evidence(
        question,
        g1_candidate_pool,
        reranker=merged_reranker,
        limit=evidence_limit,
    )
    g1_context = assemble_selected_evidence_context(
        g1_selected_evidence,
        max_context_chunks=max_context_chunks,
    )

    g2_candidate_pool = build_document_local_candidate_pool(
        g2_candidate_document_ids,
        load_document_chunks=load_document_chunks,
        max_chunks=candidate_pool_max_chunks,
    )
    g2_selected_evidence = select_document_local_evidence(
        question,
        g2_candidate_pool,
        reranker=merged_reranker,
        limit=evidence_limit,
    )
    g2_context = assemble_selected_evidence_context(
        g2_selected_evidence,
        max_context_chunks=max_context_chunks,
    )

    return G1G2AdmissionComparison(
        question_id=question_id,
        dense_candidate_chunk_ids=tuple(chunk.chunk_id for chunk in dense_ranked),
        g1_candidate_document_ids=g1_candidate_document_ids,
        g2_candidate_document_ids=g2_candidate_document_ids,
        g1_context=g1_context,
        g2_context=g2_context,
    )


def run_g2_admission_diagnostic(
    question_id: str,
    question: str,
    dense_ranked_chunks: Sequence[G0RetrievedChunk],
    *,
    shared_global_rerank: RerankResult,
    candidate_document_limit: int,
    load_document_chunks: Callable[[str], Sequence[G0RetrievedChunk]],
    candidate_pool_max_chunks: int,
    merged_reranker: Callable[..., Any],
    evidence_limit: int,
    max_context_chunks: int,
) -> G2AdmissionDiagnostic:
    """Run the offline G2 path using an already-computed global rerank result."""
    if any(
        limit <= 0
        for limit in (
            candidate_document_limit,
            candidate_pool_max_chunks,
            evidence_limit,
            max_context_chunks,
        )
    ):
        raise ValueError("all diagnostic limits must be positive")

    dense_ranked = tuple(dense_ranked_chunks)
    candidate_document_ids = select_rerank_informed_document_ids(
        shared_global_rerank,
        limit=candidate_document_limit,
    )
    candidate_pool = build_document_local_candidate_pool(
        candidate_document_ids,
        load_document_chunks=load_document_chunks,
        max_chunks=candidate_pool_max_chunks,
    )

    merged_rerank_invocations = 0

    def tracked_merged_reranker(*args: Any) -> Any:
        nonlocal merged_rerank_invocations
        merged_rerank_invocations += 1
        return merged_reranker(*args)

    selected_evidence = select_document_local_evidence(
        question,
        candidate_pool,
        reranker=tracked_merged_reranker,
        limit=evidence_limit,
    )
    final_context = assemble_selected_evidence_context(
        selected_evidence,
        max_context_chunks=max_context_chunks,
    )

    return G2AdmissionDiagnostic(
        question_id=question_id,
        question=question,
        dense_ranked_chunks=dense_ranked,
        candidate_document_ids=candidate_document_ids,
        candidate_pool=candidate_pool,
        selected_evidence=selected_evidence,
        final_context=final_context,
        merged_rerank_invocations=merged_rerank_invocations,
    )
