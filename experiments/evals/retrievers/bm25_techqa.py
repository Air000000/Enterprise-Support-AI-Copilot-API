from __future__ import annotations

import re
from collections.abc import Sequence

import bm25s

from experiments.evals.adapters.techqa import TechQADocument

TECHNICAL_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
)


def tokenize_technical(text: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in TECHNICAL_TOKEN_RE.finditer(text)
    ]


class TechQABM25Retriever:
    def __init__(self, documents: Sequence[TechQADocument]) -> None:
        ordered = sorted(documents, key=lambda document: document.document_id)
        self.document_ids = [document.document_id for document in ordered]
        corpus_tokens = [tokenize_technical(document.text) for document in ordered]
        self._retriever = bm25s.BM25(
            k1=1.5,
            b=0.75,
            method="lucene",
            corpus=self.document_ids,
            backend="numpy",
        )
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 100) -> list[str]:
        if top_k <= 0 or not self.document_ids:
            return []

        query_tokens = tokenize_technical(query)
        if not query_tokens:
            return []

        limit = min(top_k, len(self.document_ids))
        documents = self._retriever.retrieve(
            [query_tokens],
            k=limit,
            return_as="documents",
            show_progress=False,
            backend_selection="numpy",
        )
        return [str(document_id) for document_id in documents[0]]
