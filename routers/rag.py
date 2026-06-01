from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.rag_service import answer_question, search_documents

from typing import Optional

router = APIRouter(prefix="/rag", tags=["RAG"]) # 创建一个 APIRouter 实例，设置前缀为 "/rag"，所有在这个 router 中定义的路由都会自动加上这个前缀。同时给这个 router 打上 "RAG" 标签，方便在 API 文档中分类显示。

class RagSearchRequest(BaseModel):
    '''
    用户发给 API 的请求
    '''
    query: str = Field(..., min_length=1)   
    top_k: int = Field(default=3, ge=1, le=10)  # 大于等于1，小于等于10

class RagSearchResultResponse(BaseModel):
    '''
    一条检索结果
    '''
    rank: int
    document_id: str
    chunk_id: str
    title: str
    source_path: str
    chunk_index: int
    distance: float
    preview: str

class RagSearchResponse(BaseModel):
    '''
    整个接口返回结果
    '''
    query: str
    top_k: int
    total_hits: int
    results: list[RagSearchResultResponse]

class RagAskRequest(BaseModel): # 用户问问题时传进来的请求体
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    max_distance: float = Field(default=0.9, gt=0)

class RagSourceResponse(BaseModel): # sources 里的单条引用来源
    rank: int
    document_id: str
    chunk_id: str
    title: str
    source_path: str
    chunk_index: int
    distance: float
    preview: str


class RagAskResponse(BaseModel):    # 整个问答接口返回体
    question: str
    answer: str
    retrieval_status: str
    top_distance: Optional[float] = None
    sources: list[RagSourceResponse]

def make_preview(text: str, max_length: int = 200) -> str:
    '''
    对文本进行清洗和截断，生成预览文本
    '''
    normalized = " ".join(text.split()) # 按任意空白切开，包括换行、多个空格

    if len(normalized) <= max_length:   # 用单个空格拼回去
        return normalized   

    return normalized[:max_length] + "..."  # 截断  


@router.post("/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest):
    try:
        results = search_documents(
            query=request.query,
            top_k=request.top_k,
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