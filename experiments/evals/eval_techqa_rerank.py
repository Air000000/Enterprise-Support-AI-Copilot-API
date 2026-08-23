from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from experiments.evals.ir.ranx_adapter import (
    collapse_chunk_results_to_document_ranking,
)
from experiments.evals.rerankers.qwen3_reranker import (
    RerankCandidate,
    rerank_candidates,
)


class FrozenCandidatePoolError(RuntimeError):
    """Raised when the frozen E0 candidate pool cannot be reconstructed exactly."""


@dataclass(frozen=True)
class TechQARerankResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    dense_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    reranked_document_ids: tuple[str, ...]
    document_ranking: tuple[str, ...]
    rerank_latency_ms: float
    request_id: str | None
    total_tokens: int | None


Reranker = Callable[[str, list[RerankCandidate]], Any]
Clock = Callable[[], float]


def _rehydrate_candidates(
    record: Mapping[str, Any],
    *,
    collection: Any,
) -> list[RerankCandidate]:
    dense_chunk_ids = [str(chunk_id) for chunk_id in record["raw_chunk_ids"]]
    response = collection.get(
        ids=dense_chunk_ids,
        include=["documents", "metadatas"],
    )

    by_chunk_id: dict[str, RerankCandidate] = {}
    for chunk_id, content, metadata in zip(
        response["ids"],
        response["documents"],
        response["metadatas"],
    ):
        chunk_key = str(chunk_id)
        by_chunk_id[chunk_key] = RerankCandidate(
            chunk_id=chunk_key,
            document_id=str(metadata["document_id"]),
            content=str(content),
        )

    missing = [chunk_id for chunk_id in dense_chunk_ids if chunk_id not in by_chunk_id]
    if missing:
        raise FrozenCandidatePoolError(
            "missing frozen Chroma candidate(s): " + ", ".join(missing)
        )

    return [by_chunk_id[chunk_id] for chunk_id in dense_chunk_ids]


def rerank_frozen_e0_record(
    record: Mapping[str, Any],
    *,
    collection: Any,
    reranker: Reranker = rerank_candidates,
    clock: Clock = time.perf_counter,
) -> TechQARerankResult:
    """Rerank one frozen E0 candidate pool without rerunning dense retrieval."""
    candidates = _rehydrate_candidates(record, collection=collection)

    started = clock()
    rerank_result = reranker(str(record["question"]), candidates)
    rerank_latency_ms = (clock() - started) * 1000.0

    reranked_chunk_ids = tuple(
        str(candidate.chunk_id) for candidate in rerank_result.results
    )
    reranked_document_ids = tuple(
        str(candidate.document_id) for candidate in rerank_result.results
    )
    document_ranking = tuple(
        collapse_chunk_results_to_document_ranking(rerank_result.results)
    )

    return TechQARerankResult(
        question_id=str(record["question_id"]),
        relevant_document_ids=tuple(
            str(document_id) for document_id in record["relevant_document_ids"]
        ),
        dense_chunk_ids=tuple(candidate.chunk_id for candidate in candidates),
        reranked_chunk_ids=reranked_chunk_ids,
        reranked_document_ids=reranked_document_ids,
        document_ranking=document_ranking,
        rerank_latency_ms=rerank_latency_ms,
        request_id=rerank_result.request_id,
        total_tokens=rerank_result.total_tokens,
    )
