from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import bm25s

from experiments.evals.adapters.techqa import TechQADocument
from experiments.evals.retrievers.bm25_techqa import tokenize_technical
from rag_runtime.text_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    build_chunk_id,
    split_text,
)


@dataclass(frozen=True)
class TechQAChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str


def build_techqa_chunks(
    documents: Sequence[TechQADocument],
) -> list[TechQAChunk]:
    chunks: list[TechQAChunk] = []

    for document in sorted(
        documents,
        key=lambda item: item.document_id,
    ):
        contents = split_text(
            document.text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
            min_chunk_size=MIN_CHUNK_SIZE,
        )

        for chunk_index, content in enumerate(contents):
            chunks.append(
                TechQAChunk(
                    chunk_id=build_chunk_id(
                        document.document_id,
                        chunk_index,
                    ),
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content=content,
                )
            )

    return chunks


class TechQAChunkBM25Retriever:
    def __init__(
        self,
        chunks: Sequence[TechQAChunk],
    ) -> None:
        ordered = sorted(
            chunks,
            key=lambda item: (
                item.document_id,
                item.chunk_index,
                item.chunk_id,
            ),
        )

        self._chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in ordered
        }
        self._chunk_ids = [
            chunk.chunk_id
            for chunk in ordered
        ]

        corpus_tokens = [
            tokenize_technical(chunk.content)
            for chunk in ordered
        ]

        self._retriever = bm25s.BM25(
            k1=1.5,
            b=0.75,
            method="lucene",
            corpus=self._chunk_ids,
            backend="numpy",
        )
        self._retriever.index(
            corpus_tokens,
            show_progress=False,
        )

    def search(
    self,
    query: str,
    top_k: int = 100,
) -> list[TechQAChunk]:
        if top_k <= 0 or not self._chunk_ids:
            return []

        query_tokens = tokenize_technical(query)

        if not query_tokens:
            return []

        chunk_ids = self._retriever.retrieve(
            [query_tokens],
            k=min(top_k, len(self._chunk_ids)),
            return_as="documents",
            show_progress=False,
            backend_selection="numpy",
        )[0]

        return [
            self._chunks_by_id[str(chunk_id)]
            for chunk_id in chunk_ids
        ]