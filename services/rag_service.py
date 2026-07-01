from rag_runtime.query_chroma import search_chroma
from rag_runtime.query_rag_chroma import ask_rag


def search_documents(
    query: str, 
    top_k: int,
    tenant_id: str = "tenant_demo",
    category: str | None = None,
):   
    '''
    搜索与查询语句相关的文档片段
    '''
    return search_chroma(
        query=query, 
        top_k=top_k, 
        tenant_id=tenant_id,
        category=category)


def answer_question(
    question: str, 
    top_k: int, 
    max_distance: float, 
    tenant_id: str | None = "tenant_demo",
    category: str | None = None,
):    
    '''
    回答用户的问题，并返回答案和相关的引用来源
    '''
    return ask_rag(
        question=question,
        top_k=top_k,
        max_distance=max_distance,
        tenant_id=tenant_id,
        category=category,
    )