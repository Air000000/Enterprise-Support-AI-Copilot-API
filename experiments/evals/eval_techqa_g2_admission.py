from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.evals.eval_techqa_generation import (
    G0RetrievedChunk,
    assemble_selected_evidence_context,
    build_document_local_candidate_pool,
    select_candidate_document_ids,
    select_document_local_evidence,
)
from experiments.evals.adapters.techqa import TechQAGenerationCase
from experiments.evals.rerankers.qwen3_reranker import RerankResult


G2_SAMPLE_SEED = "techqa-g2-rerank-informed-admission-v1"
G2_SAMPLE_SIZE = 30


def run_offline_admission_preflight(
    *,
    e0_trace_by_question_id: Mapping[str, Sequence[G0RetrievedChunk]],
    load_document_chunks: Callable[[str], Sequence[G0RetrievedChunk]],
    shared_rerank_by_question_id: Mapping[str, RerankResult],
    merged_reranks_by_question_id: Mapping[str, Mapping[str, RerankResult]],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Check frozen inputs using data-only rerank replays; no provider defaults.

    The caller supplies a frozen corpus loader and the preregistered E0 subset.
    Fake replay scores exercise mechanics only, never evidence quality. Output
    contains canonical identities and descriptive capacities, not tuning advice.
    """
    if not e0_trace_by_question_id:
        raise ValueError("preflight requires at least one frozen case")
    cases: list[dict[str, Any]] = []
    capacities: dict[str, Any] = {}

    def checked_loader(document_id: str) -> tuple[G0RetrievedChunk, ...]:
        chunks = tuple(load_document_chunks(document_id))
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError(f"loaded chunk outside requested document {document_id!r}")
        return chunks

    for question_id in sorted(e0_trace_by_question_id):
        dense = tuple(e0_trace_by_question_id[question_id])
        dense_ids = [chunk.chunk_id for chunk in dense]
        if len(dense) != 100 or len(set(dense_ids)) != 100:
            raise ValueError(f"{question_id}: Dense Top100 must contain 100 unique chunks")
        dense_documents = {chunk.chunk_id: chunk.document_id for chunk in dense}
        shared = shared_rerank_by_question_id[question_id]
        shared_ids = [item.chunk_id for item in shared.results]
        if (
            len(shared_ids) != 100
            or set(shared_ids) != set(dense_ids)
            or any(dense_documents.get(item.chunk_id) != item.document_id for item in shared.results)
        ):
            raise ValueError(f"{question_id}: shared replay must preserve Dense Top100 identity")

        # Reorder existing replay rows in memory to express G1's Dense admission.
        # This is not another rerank invocation; both arms use the same diagnostic.
        shared_by_id = {item.chunk_id: item for item in shared.results}
        dense_admission = RerankResult(
            results=tuple(shared_by_id[chunk_id] for chunk_id in dense_ids),
            request_id="offline-dense-admission",
            total_tokens=0,
        )
        case: dict[str, Any] = {"question_id": question_id, "dense_chunk_ids": dense_ids}
        capacities[question_id] = {}
        for arm, admission in (("g1", dense_admission), ("g2", shared)):
            merged = merged_reranks_by_question_id[question_id][arm]

            def replay_merged(_question: str, candidates: Sequence[Any]) -> RerankResult:
                candidate_documents = {item.chunk_id: item.document_id for item in candidates}
                replay_ids = [item.chunk_id for item in merged.results]
                if (
                    len(replay_ids) != len(set(replay_ids))
                    or any(
                        candidate_documents.get(item.chunk_id) != item.document_id
                        for item in merged.results
                    )
                ):
                    raise ValueError(f"{question_id}/{arm}: invalid merged replay identity")
                return merged

            diagnostic = run_g2_admission_diagnostic(
                question_id,
                "",  # Replay is data-only and does not inspect query text.
                dense,
                shared_global_rerank=admission,
                candidate_document_limit=5,
                load_document_chunks=checked_loader,
                candidate_pool_max_chunks=500,
                merged_reranker=replay_merged,
                evidence_limit=16,
                max_context_chunks=16,
            )
            expected_docs = tuple(dict.fromkeys(item.document_id for item in admission.results))[:5]
            if diagnostic.dense_ranked_chunks != dense or diagnostic.candidate_document_ids != expected_docs:
                raise ValueError(f"{question_id}/{arm}: frozen admission identity changed")
            for label, chunks, limit in (
                ("candidate", diagnostic.candidate_pool, 500),
                ("context", diagnostic.final_context, 16),
            ):
                ids = [chunk.chunk_id for chunk in chunks]
                if (
                    len(ids) > limit
                    or len(ids) != len(set(ids))
                    or any(chunk.document_id not in expected_docs for chunk in chunks)
                ):
                    raise ValueError(f"{question_id}/{arm}: invalid {label} capacity or identity")
            if diagnostic.merged_rerank_invocations != bool(diagnostic.candidate_pool):
                raise ValueError(f"{question_id}/{arm}: unexpected merged replay invocation count")
            case[f"{arm}_admitted_doc_ids"] = list(diagnostic.candidate_document_ids)
            case[f"{arm}_candidate_chunk_ids"] = [chunk.chunk_id for chunk in diagnostic.candidate_pool]
            capacities[question_id][arm] = {
                "candidate_chunks": len(diagnostic.candidate_pool),
                "context_chunks": len(diagnostic.final_context),
            }
        cases.append(case)

    canonical = json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    summary = {
        "mode": "offline_replay_preflight",
        "cases": cases,
        "capacity_by_question_id": capacities,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


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


def select_g2_preregistered_cases(
    cases: Sequence[TechQAGenerationCase],
    *,
    e0_trace_by_question_id: Mapping[str, Sequence[G0RetrievedChunk]],
    relevant_document_by_question_id: Mapping[str, str],
    excluded_question_ids: Collection[str],
    seed: str = G2_SAMPLE_SEED,
    sample_size: int = G2_SAMPLE_SIZE,
) -> tuple[str, ...]:
    """Select fresh, eligible G2 cases with the frozen deterministic order."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    excluded = set(excluded_question_ids)
    eligible_question_ids: list[str] = []
    for case in cases:
        question_id = case.question_id
        trace = e0_trace_by_question_id.get(question_id)
        relevant_document_id = relevant_document_by_question_id.get(question_id)
        if (
            question_id in excluded
            or case.split != "train"
            or not question_id.startswith("TRAIN_")
            or not case.answerable
            or not trace
            or relevant_document_id is None
            or not any(
                chunk.document_id == relevant_document_id for chunk in trace
            )
        ):
            continue
        eligible_question_ids.append(question_id)

    ordered_question_ids = sorted(
        eligible_question_ids,
        key=lambda question_id: (
            hashlib.sha256(
                f"{seed}:{question_id}".encode("utf-8")
            ).hexdigest(),
            question_id,
        ),
    )
    if len(ordered_question_ids) < sample_size:
        raise ValueError(
            "Insufficient eligible G2 preregistration cases: "
            f"required={sample_size}, actual={len(ordered_question_ids)}"
        )
    return tuple(ordered_question_ids[:sample_size])


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
    load_document_chunks: Callable[[str], Sequence[G0RetrievedChunk]],
    merged_reranker: Callable[..., Any],
) -> G1G2AdmissionComparison:
    """Compare G1 and G2 document admission with identical downstream steps."""
    dense_ranked = tuple(dense_ranked_chunks)
    g1_candidate_document_ids = select_candidate_document_ids(
        dense_ranked,
        limit=5,
    )
    g2_candidate_document_ids = select_rerank_informed_document_ids(
        shared_global_rerank,
        limit=5,
    )

    g1_candidate_pool = build_document_local_candidate_pool(
        g1_candidate_document_ids,
        load_document_chunks=load_document_chunks,
        max_chunks=500,
    )
    g1_selected_evidence = select_document_local_evidence(
        question,
        g1_candidate_pool,
        reranker=merged_reranker,
        limit=16,
    )
    g1_context = assemble_selected_evidence_context(
        g1_selected_evidence,
        max_context_chunks=16,
    )

    g2_candidate_pool = build_document_local_candidate_pool(
        g2_candidate_document_ids,
        load_document_chunks=load_document_chunks,
        max_chunks=500,
    )
    g2_selected_evidence = select_document_local_evidence(
        question,
        g2_candidate_pool,
        reranker=merged_reranker,
        limit=16,
    )
    g2_context = assemble_selected_evidence_context(
        g2_selected_evidence,
        max_context_chunks=16,
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
