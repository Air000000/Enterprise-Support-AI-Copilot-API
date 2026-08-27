from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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