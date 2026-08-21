from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.evals.adapters.techqa import TechQADocument
from rag_runtime.build_chroma_index import get_chroma_client
from rag_runtime.build_rag_index import embed_texts
from rag_runtime.text_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    build_chunk_id,
    split_text,
)

DEFAULT_TECHQA_CHROMA_DIR = Path("data/eval_chroma/techqa_e0")
DEFAULT_TECHQA_COLLECTION_NAME = "techqa_e0_dense"
DEFAULT_ADD_BATCH_SIZE = 128
CHUNK_STRATEGY = "paragraph_aware_character"

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class TechQAIndexBuildSummary:
    document_count: int
    chunk_count: int
    indexing_seconds: float


@dataclass(frozen=True)
class TechQAIndexSearchResult:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    distance: float


def _reset_techqa_collection(
    chroma_dir: str | Path,
    collection_name: str,
    corpus_sha256: str,
):
    client = get_chroma_client(chroma_dir)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    return client.create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={
            "benchmark": "techqa",
            "corpus_sha256": corpus_sha256,
            "chunk_strategy": CHUNK_STRATEGY,
        },
    )


def _add_chunk_batch(
    collection,
    batch: list[dict[str, Any]],
    embedder: Embedder,
) -> None:
    if not batch:
        return

    documents = [item["content"] for item in batch]
    embeddings = embedder(documents)

    if len(embeddings) != len(batch):
        raise RuntimeError(
            "Embedding count mismatch: "
            f"chunks={len(batch)}, embeddings={len(embeddings)}"
        )

    collection.add(
        ids=[item["chunk_id"] for item in batch],
        documents=documents,
        embeddings=embeddings,
        metadatas=[item["metadata"] for item in batch],
    )


def build_techqa_index(
    documents: Iterable[TechQADocument],
    *,
    chroma_dir: str | Path = DEFAULT_TECHQA_CHROMA_DIR,
    collection_name: str = DEFAULT_TECHQA_COLLECTION_NAME,
    corpus_sha256: str,
    embedder: Embedder = embed_texts,
    add_batch_size: int = DEFAULT_ADD_BATCH_SIZE,
) -> TechQAIndexBuildSummary:
    """Build an isolated dense Chroma index from the full TechQA corpus."""
    if add_batch_size <= 0:
        raise ValueError("add_batch_size must be greater than 0")

    started = time.perf_counter()
    ordered_documents = sorted(documents, key=lambda item: item.document_id)
    collection = _reset_techqa_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256=corpus_sha256,
    )

    chunk_count = 0
    batch: list[dict[str, Any]] = []

    for document in ordered_documents:
        chunk_texts = split_text(
            document.text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
            min_chunk_size=MIN_CHUNK_SIZE,
        )

        for chunk_index, content in enumerate(chunk_texts):
            chunk_id = build_chunk_id(document.document_id, chunk_index)
            batch.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": {
                        "benchmark": "techqa",
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "chunk_strategy": CHUNK_STRATEGY,
                        "corpus_sha256": corpus_sha256,
                        "document_id": document.document_id,
                    },
                }
            )
            chunk_count += 1

            if len(batch) >= add_batch_size:
                _add_chunk_batch(collection, batch, embedder)
                batch = []

    _add_chunk_batch(collection, batch, embedder)

    return TechQAIndexBuildSummary(
        document_count=len(ordered_documents),
        chunk_count=chunk_count,
        indexing_seconds=time.perf_counter() - started,
    )


def search_techqa_index(
    query: str,
    *,
    chroma_dir: str | Path = DEFAULT_TECHQA_CHROMA_DIR,
    collection_name: str = DEFAULT_TECHQA_COLLECTION_NAME,
    top_k: int = 20,
    embedder: Embedder = embed_texts,
) -> list[TechQAIndexSearchResult]:
    """Query the isolated TechQA dense index and return chunk-level results."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    query_embeddings = embedder([query])
    if len(query_embeddings) != 1:
        raise RuntimeError(
            "Query embedding count mismatch: "
            f"expected=1, embeddings={len(query_embeddings)}"
        )

    raw_results: dict[str, Any] = collection.query(
        query_embeddings=[query_embeddings[0]],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw_results["ids"][0]
    stored_documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    return [
        TechQAIndexSearchResult(
            chunk_id=str(chunk_id),
            document_id=str(metadata["document_id"]),
            chunk_index=int(metadata["chunk_index"]),
            content=str(content),
            distance=float(distance),
        )
        for chunk_id, content, metadata, distance in zip(
            ids,
            stored_documents,
            metadatas,
            distances,
        )
    ]
