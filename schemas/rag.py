'''
schema 是 API 契约，主要服务 router；
service 不应该依赖 router，也尽量不要依赖 HTTP request schema。
'''

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional


class RagSearchRequest(BaseModel):
    '''
    用户发给 API 的请求
    '''
    query: str = Field(..., min_length=1)   
    top_k: int = Field(default=3, ge=1, le=14, description="Dense: 1-10/default3; Hybrid: 1-14/default14 when omitted")
    category: Optional[str] = None
    retrieval_mode: Literal["dense", "hybrid"] = "dense"

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value):
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def mode_limit(self):
        if self.retrieval_mode == "dense" and self.top_k > 10:
            raise ValueError("Dense top_k must not exceed 10")
        if self.retrieval_mode == "hybrid" and "top_k" not in self.model_fields_set:
            self.top_k = 14
        return self

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
    distance: float | None
    rerank_score: float | None = None
    fusion_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    tenant_id: Optional[str] = None
    category: Optional[str] = None
    preview: str


class RagSearchResponse(BaseModel):
    '''
    整个接口返回结果
    '''
    query: str
    top_k: int
    total_hits: int
    results: list[RagSearchResultResponse]
    retrieval_mode: Literal["dense", "hybrid"] = "dense"


class RagAskRequest(BaseModel): # 用户问问题时传进来的请求体
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    max_distance: float = Field(default=0.9, gt=0)
    # tenant_id: Optional[str] = None
    category: Optional[str] = None

class RagSourceResponse(BaseModel): # sources 里的单条引用来源
    rank: int
    document_id: str
    chunk_id: str
    title: str
    source_path: str
    chunk_index: int
    distance: float
    tenant_id: Optional[str] = None
    category: Optional[str] = None
    preview: str


class RagAskResponse(BaseModel):    # 整个问答接口返回体
    question: str
    answer: str
    retrieval_status: str
    top_distance: Optional[float] = None
    sources: list[RagSourceResponse]
