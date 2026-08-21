from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def collapse_chunk_results_to_document_ranking(
    results: Iterable[Any],
) -> list[str]:
    """
    Convert chunk-level retrieval ranking into document-level ranking.

    Keep the first occurrence of each document because the input order already
    represents the retriever ranking.
    """
    document_ranking: list[str] = []
    seen: set[str] = set()

    for result in results:
        document_id = result.document_id

        if document_id in seen:
            continue

        seen.add(document_id)
        document_ranking.append(document_id)

    return document_ranking
