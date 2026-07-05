import json
import time
from fastapi import APIRouter, Depends, HTTPException



from schemas.rag import (
    RagSearchRequest,
    RagSearchResponse,
    RagSearchResultResponse,
    RagAskRequest,
    RagAskResponse,
    RagSourceResponse,
)
from schemas.agent_ops import RetrievalLogCreate
from services.agent_ops_service import create_retrieval_log as create_retrieval_log_service
from services.rag_service import answer_question, search_documents
from auth import CurrentUser, get_current_user


# 创建一个 APIRouter 实例，设置前缀为 "/rag"，
# 所有在这个 router 中定义的路由都会自动加上这个前缀。
# 同时给这个 router 打上 "RAG" 标签，方便在 API 文档中分类显示。
router = APIRouter(prefix="/rag", tags=["RAG"]) 


def make_preview(text: str, max_length: int = 200) -> str:
    """对文本进行清洗和截断，生成预览文本"""
    normalized = " ".join(text.split()) # 按任意空白切开，包括换行、多个空格

    if len(normalized) <= max_length:   # 用单个空格拼回去
        return normalized

    return normalized[:max_length] + "..."  # 截断  


def safe_create_retrieval_log(
    retrieval_log_create: RetrievalLogCreate,
) -> None:
    try:
        create_retrieval_log_service(retrieval_log_create)
    except Exception:
        # Retrieval logging is best-effort.
        # It should not break the user-facing RAG API.
        pass


@router.post("/search", response_model=RagSearchResponse)
def rag_search(
    request: RagSearchRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """根据用户的查询语句，搜索相关的文档片段"""
    started_at = time.perf_counter()

    try:
        results = search_documents(
            query=request.query,
            top_k=request.top_k,
            tenant_id=user.tenant_id,
            category=request.category,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        safe_create_retrieval_log(
            RetrievalLogCreate(
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                endpoint="search",
                query_text=request.query,
                top_k=request.top_k,
                category=request.category,
                retrieval_status="failed",
                total_hits=0,
                latency_ms=latency_ms,
                error_message=str(exc),
            )
        )

        raise HTTPException(
            status_code=500,
            detail=f"RAG search failed: {exc}",
        ) from exc

    response_results = []

    for index, item in enumerate(results, start=1):
        response_results.append(
            RagSearchResultResponse(
                rank=index,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                title=item.title,
                source_path=item.source_path,
                chunk_index=item.chunk_index,
                distance=round(item.distance, 4),
                tenant_id=getattr(item, "tenant_id", None),
                category=getattr(item, "category", None),
                preview=make_preview(item.content),
            )
        )

    source_documents = [
        {
            "rank": result.rank,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "title": result.title,
            "source_path": result.source_path,
            "chunk_index": result.chunk_index,
            "distance": result.distance,
            "tenant_id": result.tenant_id,
            "category": result.category,
        }
        for result in response_results
    ]

    scores = [
        result.distance
        for result in response_results
    ]

    latency_ms = int((time.perf_counter() - started_at) * 1000)

    safe_create_retrieval_log(
        RetrievalLogCreate(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            endpoint="search",
            query_text=request.query,
            top_k=request.top_k,
            category=request.category,
            retrieval_status=(
                "ok"
                if response_results
                else "no_context"
            ),
            total_hits=len(response_results),
            top_distance=(
                response_results[0].distance
                if response_results
                else None
            ),
            source_documents_json=json.dumps(
                source_documents,
                ensure_ascii=False,
            ),
            scores_json=json.dumps(
                scores,
                ensure_ascii=False,
            ),
            latency_ms=latency_ms,
        )
    )

    return RagSearchResponse(
        query=request.query,
        top_k=request.top_k,
        total_hits=len(response_results),
        results=response_results,
    )


@router.post("/ask", response_model=RagAskResponse)
def rag_ask(
    request: RagAskRequest,
    user: CurrentUser = Depends(get_current_user),
):
    started_at = time.perf_counter()

    try:
        rag_result = answer_question(
            question=request.question,
            top_k=request.top_k,
            max_distance=request.max_distance,
            tenant_id=user.tenant_id,
            category=request.category,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        safe_create_retrieval_log(
            RetrievalLogCreate(
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                endpoint="ask",
                query_text=request.question,
                top_k=request.top_k,
                category=request.category,
                retrieval_status="failed",
                total_hits=0,
                latency_ms=latency_ms,
                error_message=str(exc),
            )
        )

        raise HTTPException(
            status_code=500,
            detail=f"RAG ask failed: {exc}",
        ) from exc

    source_responses = []

    for index, source in enumerate(rag_result.sources, start=1):
        source_responses.append(
            RagSourceResponse(
                rank=index,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                title=source.title,
                source_path=source.source_path,
                chunk_index=source.chunk_index,
                distance=round(source.distance, 4),
                tenant_id=getattr(source, "tenant_id", None),
                category=getattr(source, "category", None),
                preview=make_preview(source.preview),
            )
        )

    source_documents = [
        {
            "rank": source.rank,
            "document_id": source.document_id,
            "chunk_id": source.chunk_id,
            "title": source.title,
            "source_path": source.source_path,
            "chunk_index": source.chunk_index,
            "distance": source.distance,
            "tenant_id": source.tenant_id,
            "category": source.category,
        }
        for source in source_responses
    ]

    scores = [
        source.distance
        for source in source_responses
    ]

    latency_ms = int((time.perf_counter() - started_at) * 1000)

    safe_create_retrieval_log(
        RetrievalLogCreate(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            endpoint="ask",
            query_text=request.question,
            top_k=request.top_k,
            category=request.category,
            retrieval_status=rag_result.retrieval_status,
            total_hits=len(source_responses),
            top_distance=(
                round(rag_result.top_distance, 4)
                if rag_result.top_distance is not None
                else None
            ),
            source_documents_json=json.dumps(
                source_documents,
                ensure_ascii=False,
            ),
            scores_json=json.dumps(
                scores,
                ensure_ascii=False,
            ),
            latency_ms=latency_ms,
        )
    )

    return RagAskResponse(
        question=request.question,
        answer=rag_result.answer,
        retrieval_status=rag_result.retrieval_status,
        top_distance=(
            round(rag_result.top_distance, 4)
            if rag_result.top_distance is not None
            else None
        ),
        sources=source_responses,
    )
