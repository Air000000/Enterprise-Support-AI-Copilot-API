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
    top_k: int = Field(default=14, ge=1, le=14, description="Hybrid default 14/max 14; explicit Dense default 3/max 10")
    category: Optional[str] = None
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value):
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def mode_limit(self):
        if self.retrieval_mode == "dense" and "top_k" not in self.model_fields_set:
            self.top_k = 3
        if self.retrieval_mode == "dense" and self.top_k > 10:
            raise ValueError("Dense top_k must not exceed 10")
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
    top_k: int = Field(default=14, ge=1, le=14, description="Hybrid default 14/max 14; explicit Dense default 3/max 10")
    max_distance: float | None = Field(default=None, gt=0, description="Dense only; defaults to 0.9 when Dense is explicitly selected")
    # tenant_id: Optional[str] = None
    category: Optional[str] = None
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    refusal_policy: Literal["v5_3_experimental", "v5_3_1_experimental", "v5_3_1_contract_experimental", "source_scope_experimental"] | None = "source_scope_experimental"

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def mode_limit(self):
        if self.retrieval_mode == "dense":
            if "top_k" not in self.model_fields_set:
                self.top_k = 3
            if "max_distance" not in self.model_fields_set:
                self.max_distance = 0.9
            if "refusal_policy" not in self.model_fields_set:
                self.refusal_policy = None
            if self.top_k > 10 or self.refusal_policy is not None or self.max_distance is None:
                raise ValueError("Dense top_k must not exceed 10 and uses the existing distance policy")
        else:
            if self.max_distance is not None or self.refusal_policy is None:
                raise ValueError("Hybrid requires an evidence policy and no distance threshold")
        return self

class RagSourceResponse(RagSearchResultResponse):
    pass


class RagAskResponse(BaseModel):    # 整个问答接口返回体
    question: str
    answer: str
    retrieval_status: str
    top_distance: Optional[float] = None
    sources: list[RagSourceResponse]
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    refusal_policy: Literal["dense_distance", "v5_3_experimental", "v5_3_1_experimental", "v5_3_1_contract_experimental", "source_scope_experimental"] = "dense_distance"
    evidence_gate: dict | None = None
