from __future__ import annotations


from experiments.rag_local.query_chroma import ChromaSearchResult, search_chroma
from experiments.evals import eval_core

def result_to_preview(result: ChromaSearchResult) -> dict:
    '''
    将 SearchResult 转换为包含预览信息的字典，方便 eval 报告展示。
    '''
    return {
        "chunk_id": result.chunk_id,    
        "document_id": result.document_id,
        "title": result.title,
        "source_path": result.source_path,
        "chunk_index": result.chunk_index,
        "distance": result.distance,
        "preview": result.content[:120].replace("\n", " "),
    }


def evaluate(top_k: int = 3) -> dict:
    return eval_core.evaluate_retriever(
        retriever=search_chroma,
        result_to_preview=result_to_preview,
        top_k=top_k,
        eval_file=eval_core.ENTERPRISE_EVAL_FILE,
    )


def main() -> None:
    top_k = 3
    report = evaluate(top_k=top_k)
    eval_core.print_retrieval_report(report, title="Retrieval Eval Report - Chroma", top_k=top_k, metric_key="distance")


if __name__ == "__main__":
    main()