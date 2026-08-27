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

def percentile(
    values: Sequence[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )
    fraction = position - lower_index

    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def build_audit_summary(
    results: Sequence[TechQAChunkBM25AuditResult],
) -> dict[str, object]:
    query_count = len(results)

    cutoffs: dict[str, dict[str, float | int]] = {}

    for cutoff in RAW_CUTOFFS:
        observations = [
            next(
                observation
                for observation in result.cutoff_observations
                if observation.cutoff == cutoff
            )
            for result in results
        ]

        unique_document_counts = [
            observation.unique_document_count
            for observation in observations
        ]
        duplicate_ratios = [
            observation.duplicate_ratio
            for observation in observations
        ]

        gold_hit_count = sum(
            observation.gold_document_hit_within_chunk_k
            for observation in observations
        )
        crowding_rescue_count = sum(
            observation.crowding_rescue
            for observation in observations
        )

        cutoffs[str(cutoff)] = {
            "unique_document_count_p05": percentile(
                unique_document_counts,
                0.05,
            ),
            "unique_document_count_p50": percentile(
                unique_document_counts,
                0.50,
            ),
            "duplicate_ratio_p50": percentile(
                duplicate_ratios,
                0.50,
            ),
            "duplicate_ratio_p95": percentile(
                duplicate_ratios,
                0.95,
            ),
            "gold_document_hit_count": gold_hit_count,
            "gold_document_hit_rate": (
                gold_hit_count / query_count
                if query_count
                else 0.0
            ),
            "crowding_rescue_count": crowding_rescue_count,
            "crowding_rescue_rate": (
                crowding_rescue_count / query_count
                if query_count
                else 0.0
            ),
        }

    collapsed_document_recall: dict[str, float] = {}

    for cutoff in RAW_CUTOFFS:
        hit_count = sum(
            first_relevant_rank(
                result.document_top100[:cutoff],
                result.relevant_document_ids,
            )
            is not None
            for result in results
        )

        collapsed_document_recall[f"recall@{cutoff}"] = (
            hit_count / query_count
            if query_count
            else 0.0
        )

    crowding_gaps = [
        result.crowding_gap
        for result in results
        if result.crowding_gap is not None
    ]

    latencies_ms = [
        result.latency_ms
        for result in results
    ]

    return {
        "query_count": query_count,
        "cutoffs": cutoffs,
        "collapsed_document_recall": (
            collapsed_document_recall
        ),
        "crowding_gap": {
            "observed_count": len(crowding_gaps),
            "p50": percentile(crowding_gaps, 0.50),
            "p95": percentile(crowding_gaps, 0.95),
        },
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
        },
    }

def build_diagnostic_cases(
    results: Sequence[TechQAChunkBM25AuditResult],
    *,
    limit_per_group: int = 10,
) -> dict[str, list[dict[str, object]]]:
    def build_record(
        result: TechQAChunkBM25AuditResult,
        observation: ChunkCutoffObservation,
    ) -> dict[str, object]:
        return {
            "question_id": result.question_id,
            "question": result.question,
            "cutoff": observation.cutoff,
            "returned_chunk_count": (
                observation.returned_chunk_count
            ),
            "unique_document_count": (
                observation.unique_document_count
            ),
            "duplicate_ratio": observation.duplicate_ratio,
            "relevant_document_ids": (
                result.relevant_document_ids
            ),
            "first_gold_chunk_rank": (
                result.first_gold_chunk_rank
            ),
            "first_gold_document_rank": (
                result.first_gold_document_rank
            ),
            "crowding_gap": result.crowding_gap,
            "crowding_rescue": observation.crowding_rescue,
            "raw_top100_chunk_ids": (
                result.raw_top100_chunk_ids
            ),
            "raw_top100_document_ids": (
                result.raw_top100_document_ids
            ),
        }

    high_duplication_cases: list[dict[str, object]] = []
    crowding_rescue_cases: list[dict[str, object]] = []

    for result in results:
        top100_observation = next(
            observation
            for observation in result.cutoff_observations
            if observation.cutoff == 100
        )

        high_duplication_cases.append(
            build_record(
                result,
                top100_observation,
            )
        )

        for observation in result.cutoff_observations:
            if observation.crowding_rescue:
                crowding_rescue_cases.append(
                    build_record(
                        result,
                        observation,
                    )
                )

    high_duplication_cases.sort(
        key=lambda row: (
            -float(row["duplicate_ratio"]),
            str(row["question_id"]),
        )
    )

    def crowding_sort_key(
        row: dict[str, object],
    ) -> tuple[int, int, str]:
        crowding_gap = row["crowding_gap"]
        gap = (
            int(crowding_gap)
            if crowding_gap is not None
            else -1
        )

        return (
            int(row["cutoff"]),
            -gap,
            str(row["question_id"]),
        )

    crowding_rescue_cases.sort(
        key=crowding_sort_key
    )

    return {
        "high_duplication_cases": (
            high_duplication_cases[:limit_per_group]
        ),
        "crowding_rescue_cases": (
            crowding_rescue_cases[:limit_per_group]
        ),
    }