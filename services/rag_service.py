import os

from dotenv import load_dotenv

from rag_runtime.query_chroma import search_chroma
from rag_runtime.query_rag_chroma import ask_rag


def search_documents(
    query: str, 
    top_k: int,
    tenant_id: str = "tenant_demo",
    category: str | None = None,
    retrieval_mode: str = "dense",
):   
    '''
    搜索与查询语句相关的文档片段
    '''
    if retrieval_mode == "hybrid":
        load_dotenv()
        if os.getenv("RAG_HYBRID_SEARCH_ENABLED", "false").strip().lower() != "true":
            raise RuntimeError("Hybrid search is disabled")
        from rag_runtime.hybrid_search import search_hybrid
        return search_hybrid(query, top_k, tenant_id, category)
    if retrieval_mode != "dense":
        raise ValueError("Unknown retrieval mode")
    return search_chroma(
        query=query, 
        top_k=top_k, 
        tenant_id=tenant_id,
        category=category)


def answer_question(
    question: str, 
    top_k: int, 
    max_distance: float | None,
    tenant_id: str | None = "tenant_demo",
    category: str | None = None,
    retrieval_mode: str = "dense",
    refusal_policy: str | None = None,
):    
    '''
    回答用户的问题，并返回答案和相关的引用来源
    '''
    if retrieval_mode == "hybrid":
        load_dotenv()
        if any(os.getenv(flag, "false").strip().lower() != "true" for flag in (
            "RAG_HYBRID_SEARCH_ENABLED", "RAG_HYBRID_ASK_ENABLED"
        )):
            raise RuntimeError("Experimental Hybrid ask is disabled")
        if max_distance is not None or refusal_policy != "v5_3_experimental":
            raise ValueError("Hybrid ask requires explicit experimental policy and no distance threshold")
        from rag_runtime.hybrid_ask import ask_hybrid
        return ask_hybrid(question, top_k, tenant_id, category)
    if retrieval_mode != "dense" or refusal_policy is not None:
        raise ValueError("Unknown retrieval mode or unsupported Dense refusal policy")
    return ask_rag(
        question=question,
        top_k=top_k,
        max_distance=max_distance,
        tenant_id=tenant_id,
        category=category,
    )
