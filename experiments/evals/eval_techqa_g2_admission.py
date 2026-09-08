from __future__ import annotations

from experiments.evals.rerankers.qwen3_reranker import RerankResult


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
