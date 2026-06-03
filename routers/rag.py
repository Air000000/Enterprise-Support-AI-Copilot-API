from fastapi import APIRouter, HTTPException

from schemas.rag import (
    RagSearchRequest,
    RagSearchResponse,
    RagSearchResultResponse,
    RagAskRequest,
    RagAskResponse,
    RagSourceResponse,
)
from services.rag_service import answer_question, search_documents


# 创建一个 APIRouter 实例，设置前缀为 "/rag"，
# 所有在这个 router 中定义的路由都会自动加上这个前缀。
# 同时给这个 router 打上 "RAG" 标签，方便在 API 文档中分类显示。
router = APIRouter(prefix="/rag", tags=["RAG"]) 

MOCK_TENANT_ID = "tenant_demo"

def make_preview(text: str, max_length: int = 200) -> str:
    """对文本进行清洗和截断，生成预览文本"""
    normalized = " ".join(text.split()) # 按任意空白切开，包括换行、多个空格

    if len(normalized) <= max_length:   # 用单个空格拼回去
        return normalized

    return normalized[:max_length] + "..."  # 截断  


@router.post("/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest):
    """根据用户的查询语句，搜索相关的文档片段"""
    try:
        results = search_documents(
            query=request.query,
            top_k=request.top_k,
            tenant_id=MOCK_TENANT_ID,
            category=request.category,
        )
    except Exception as exc:
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

    return RagSearchResponse(
        query=request.query,
        top_k=request.top_k,
        total_hits=len(response_results),
        results=response_results,
    )


@router.post("/ask", response_model=RagAskResponse)
def rag_ask(request: RagAskRequest):
    try:
        rag_result = answer_question(
            question=request.question,
            top_k=request.top_k,
            max_distance=request.max_distance,
            tenant_id=MOCK_TENANT_ID,
            category=request.category,
        )   
    except Exception as exc:
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