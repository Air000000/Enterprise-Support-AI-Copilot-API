from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from experiments.evals.adapters.techqa import TechQARetrievalCase

@dataclass(frozen=True)
class ChunkCutoffObservation:
    cutoff: int
    returned_chunk_count: int
    unique_document_count: int
    duplicate_slot_count: int
    duplicate_ratio: float
    gold_document_hit_within_chunk_k: bool
    crowding_rescue: bool


def collapse_document_ids(
    document_ids: Sequence[str],
) -> list[str]:
    collapsed: list[str] = []
    seen: set[str] = set()

    for document_id in document_ids:
        value = str(document_id)

        if value in seen:
            continue

        seen.add(value)
        collapsed.append(value)

    return collapsed


def first_relevant_rank(
    document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
) -> int | None:
    relevant = {
        str(document_id)
        for document_id in relevant_document_ids
    }

    for rank, document_id in enumerate(
        document_ids,
        start=1,
    ):
        if str(document_id) in relevant:
            return rank

    return None


@dataclass(frozen=True)
class TechQAChunkRef:
    chunk_id: str
    document_id: str
    chunk_index: int


@dataclass(frozen=True)
class AdaptiveSearchResult:
    candidates: tuple[object, ...]
    final_search_depth: int
    latency_ms: float

RAW_CUTOFFS = (20, 50, 100)


@dataclass(frozen=True)
class TechQAChunkBM25AuditResult:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    audit_search_depth: int
    latency_ms: float
    raw_top100_chunk_ids: tuple[str, ...]
    raw_top100_document_ids: tuple[str, ...]
    document_top100: tuple[str, ...]
    first_gold_chunk_rank: int | None
    first_gold_document_rank: int | None
    crowding_gap: int | None
    cutoff_observations: tuple[ChunkCutoffObservation, ...]

def retrieve_with_unique_document_depth(
    query: str,
    *,
    searcher: Callable[..., list[object]],
    clock: Callable[[], float] = time.perf_counter,
    initial_depth: int = 500,
    required_unique_documents: int = 100,
    max_depth: int = 172614,
) -> AdaptiveSearchResult:
    if (
        initial_depth <= 0
        or required_unique_documents <= 0
        or max_depth <= 0
    ):
        raise ValueError(
            "audit depth values must be greater than 0"
        )

    depth = min(initial_depth, max_depth)
    total_latency_ms = 0.0
    latest: list[object] = []

    while True:
        started = clock()

        latest = searcher(
            query,
            top_k=depth,
        )

        total_latency_ms += (
            clock() - started
        ) * 1000.0

        unique_document_count = len(
            collapse_document_ids(
                [
                    str(candidate.document_id)
                    for candidate in latest
                ]
            )
        )

        if unique_document_count >= required_unique_documents:
            break

        if depth >= max_depth or len(latest) < depth:
            break

        depth = min(
            depth * 2,
            max_depth,
        )

    return AdaptiveSearchResult(
        candidates=tuple(latest),
        final_search_depth=depth,
        latency_ms=total_latency_ms,
    )

def evaluate_audit_case(
    case: TechQARetrievalCase,
    *,
    searcher: Callable[..., list[object]],
    clock: Callable[[], float] = time.perf_counter,
    max_depth: int = 172614,
) -> TechQAChunkBM25AuditResult:
    search_result = retrieve_with_unique_document_depth(
        case.question.rstrip(),
        searcher=searcher,
        clock=clock,
        max_depth=max_depth,
    )

    raw_chunk_ids = tuple(
        str(candidate.chunk_id)
        for candidate in search_result.candidates
    )
    raw_document_ids = tuple(
        str(candidate.document_id)
        for candidate in search_result.candidates
    )

    collapsed_document_ids = tuple(
        collapse_document_ids(raw_document_ids)
    )

    first_gold_chunk_rank = first_relevant_rank(
        raw_document_ids,
        case.relevant_document_ids,
    )
    first_gold_document_rank = first_relevant_rank(
        collapsed_document_ids,
        case.relevant_document_ids,
    )

    crowding_gap = (
        first_gold_chunk_rank - first_gold_document_rank
        if (
            first_gold_chunk_rank is not None
            and first_gold_document_rank is not None
        )
        else None
    )

    cutoff_observations = tuple(
        build_cutoff_observation(
            raw_document_ids=raw_document_ids,
            relevant_document_ids=case.relevant_document_ids,
            cutoff=cutoff,
        )
        for cutoff in RAW_CUTOFFS
    )

    return TechQAChunkBM25AuditResult(
        question_id=case.question_id,
        question=case.question,
        relevant_document_ids=case.relevant_document_ids,
        audit_search_depth=search_result.final_search_depth,
        latency_ms=search_result.latency_ms,
        raw_top100_chunk_ids=raw_chunk_ids[:100],
        raw_top100_document_ids=raw_document_ids[:100],
        document_top100=collapsed_document_ids[:100],
        first_gold_chunk_rank=first_gold_chunk_rank,
        first_gold_document_rank=first_gold_document_rank,
        crowding_gap=crowding_gap,
        cutoff_observations=cutoff_observations,
    )

def build_cutoff_observation(
    raw_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
    cutoff: int,
) -> ChunkCutoffObservation:
    raw_prefix = [
        str(document_id)
        for document_id in raw_document_ids[:cutoff]
    ]

    returned_chunk_count = len(raw_prefix)
    unique_document_count = len(set(raw_prefix))
    duplicate_slot_count = (
        returned_chunk_count - unique_document_count
    )
    duplicate_ratio = (
        duplicate_slot_count / returned_chunk_count
        if returned_chunk_count
        else 0.0
    )

    first_gold_chunk_rank = first_relevant_rank(
        raw_document_ids,
        relevant_document_ids,
    )

    collapsed_document_ids = collapse_document_ids(
        raw_document_ids
    )
    first_gold_document_rank = first_relevant_rank(
        collapsed_document_ids,
        relevant_document_ids,
    )

    gold_document_hit_within_chunk_k = (
        first_gold_chunk_rank is not None
        and first_gold_chunk_rank <= cutoff
    )

    crowding_rescue = (
        first_gold_chunk_rank is not None
        and first_gold_document_rank is not None
        and first_gold_chunk_rank > cutoff
        and first_gold_document_rank <= cutoff
    )

    return ChunkCutoffObservation(
        cutoff=cutoff,
        returned_chunk_count=returned_chunk_count,
        unique_document_count=unique_document_count,
        duplicate_slot_count=duplicate_slot_count,
        duplicate_ratio=duplicate_ratio,
        gold_document_hit_within_chunk_k=(
            gold_document_hit_within_chunk_k
        ),
        crowding_rescue=crowding_rescue,
    )