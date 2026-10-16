"""Opt-in managed-document search; no answer generation or refusal-score reuse."""

import math
from dataclasses import dataclass, replace
from pathlib import Path

from sqlmodel import Session, select

from database import engine
from experiments.evals.ir.rrf import fuse_rrf
from experiments.evals.rerankers.qwen3_reranker import (
    RerankCandidate,
    get_rerank_client,
    rerank_candidates,
)
from experiments.evals.retrievers.bm25_techqa import tokenize_technical
from experiments.evals.retrievers.bm25_techqa_chunks import TechQAChunk, TechQAChunkBM25Retriever
from models.document import Document, DocumentChunk
from rag_runtime.query_chroma import ChromaSearchResult, build_where_filter, embed_texts, get_collection

CANDIDATE_K, RRF_K, CONTEXT_K = 100, 60, 14
MAX_VISIBLE_CHUNKS = 5000


@dataclass
class HybridSearchResult(ChromaSearchResult):
    distance: float | None
    rerank_score: float | None = None
    fusion_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def preview(self):
        return self.content[:160].replace("\n", " ")


def load_visible_chunks(tenant_id, category, *, embedding_ids=None):
    statement = select(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id).where(
        Document.tenant_id == tenant_id, Document.status == "indexed",
        DocumentChunk.tenant_id == tenant_id, DocumentChunk.category == Document.category)
    if category is not None:
        statement = statement.where(Document.category == category)
    if embedding_ids is not None:
        statement = statement.where(DocumentChunk.embedding_id.in_(embedding_ids))
    statement = statement.order_by(Document.id, DocumentChunk.chunk_index, DocumentChunk.embedding_id).limit(MAX_VISIBLE_CHUNKS + 1)
    with Session(engine) as session:
        rows = session.exec(statement).all()
    if len(rows) > MAX_VISIBLE_CHUNKS:
        raise RuntimeError("Hybrid demo corpus exceeds 5000 visible chunks")
    corpus = {}
    for chunk, document in rows:
        expected = f"doc:{document.id}:v{document.version}:chunk:{chunk.chunk_index}"
        if chunk.embedding_id != expected or not chunk.content.strip():
            continue  # Stale versions never enter either channel or the external reranker.
        if expected in corpus:
            raise RuntimeError("Duplicate managed chunk identity")
        corpus[expected] = HybridSearchResult(
            chunk_id=expected, document_id=document.id, chunk_index=chunk.chunk_index,
            content=chunk.content, tenant_id=document.tenant_id, category=document.category,
            title=Path(document.filename).stem, source_path=document.source_path, distance=None)
    return corpus


def search_hybrid(query: str, top_k: int, tenant_id: str, category: str | None = None):
    if not isinstance(tenant_id, str) or not tenant_id.strip() or not query.strip():
        raise ValueError("Hybrid requires a tenant and nonempty query")
    if type(top_k) is not int or not 1 <= top_k <= CONTEXT_K:
        raise ValueError("Hybrid top_k must be between 1 and 14")
    query = query.rstrip()
    corpus = load_visible_chunks(tenant_id, category)
    if not corpus:
        return []  # No embedding or rerank request for an empty authorized corpus.

    raw = get_collection().query(
        ids=list(corpus), query_embeddings=embed_texts([query]),
        n_results=min(CANDIDATE_K, len(corpus)), where=build_where_filter(tenant_id, category),
        include=["distances"])
    dense, distances = raw["ids"][0], raw["distances"][0]
    if (len(dense) != len(distances) or len(set(dense)) != len(dense)
            or any(cid not in corpus for cid in dense)
            or any(type(d) not in (int, float) or not math.isfinite(d) for d in distances)):
        raise RuntimeError("Invalid or out-of-scope dense ranking")
    distance_by_id = dict(zip(dense, distances, strict=True))

    # ponytail: rebuild a tenant-scoped BM25 index per request (<=5000 chunks); add
    # versioned per-tenant caching only if measured latency justifies invalidation complexity.
    bm25 = []
    if tokenize_technical(query) and any(tokenize_technical(c.content) for c in corpus.values()):
        chunks = [TechQAChunk(c.chunk_id, c.document_id, c.chunk_index, c.content) for c in corpus.values()]
        bm25 = [c.chunk_id for c in TechQAChunkBM25Retriever(chunks).search(query, top_k=CANDIDATE_K)]
    fused = fuse_rrf([dense, bm25], rrf_k=RRF_K, top_k=CANDIDATE_K)
    if not fused:
        return []
    expected = {cid: corpus[cid] for cid in fused}

    def require_current_visibility():
        if load_visible_chunks(tenant_id, category, embedding_ids=fused) != expected:
            raise RuntimeError("Managed documents changed during Hybrid search")

    require_current_visibility()  # Recheck before sending candidate bodies to the provider.
    candidates = [RerankCandidate(cid, corpus[cid].document_id, corpus[cid].content) for cid in fused]
    with get_rerank_client().with_options(timeout=15) as client:
        reranked = rerank_candidates(query, candidates, client=client)
    if (len(reranked.results) != len(fused) or {r.chunk_id for r in reranked.results} != set(fused)
            or any(not math.isfinite(r.relevance_score) for r in reranked.results)):
        raise RuntimeError("Invalid rerank permutation or score")
    require_current_visibility()  # Do not publish stale/deleted results after the external call.
    dense_ranks = {cid: rank for rank, cid in enumerate(dense, 1)}
    bm25_ranks = {cid: rank for rank, cid in enumerate(bm25, 1)}
    return [replace(corpus[row.chunk_id], distance=distance_by_id.get(row.chunk_id),
                    rerank_score=row.relevance_score, dense_rank=dense_ranks.get(row.chunk_id),
                    bm25_rank=bm25_ranks.get(row.chunk_id), fusion_score=sum(
                        1 / (RRF_K + ranks[row.chunk_id]) for ranks in (dense_ranks, bm25_ranks) if row.chunk_id in ranks))
            for row in reranked.results[:CONTEXT_K][:top_k]]
