from __future__ import annotations

from collections.abc import Sequence


def fuse_rrf(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = 60,
    top_k: int = 100,
) -> list[str]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            document_id = str(document_id)

            scores[document_id] = (
                scores.get(document_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

            best_rank[document_id] = min(
                best_rank.get(document_id, rank),
                rank,
            )

    ordered = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            best_rank[document_id],
            document_id,
        ),
    )

    return ordered[:top_k]