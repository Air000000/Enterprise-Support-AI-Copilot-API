from __future__ import annotations

import json
from pathlib import Path

from experiments.rag_local.query_chroma import ChromaSearchResult, search_chroma


EVAL_FILE = Path("experiments/evals/eval_questions.jsonl") # 评测问题列表，每行一个 JSON 对象，包含 question 和 expected_document_id


def load_eval_cases(path: Path = EVAL_FILE) -> list[dict]:
    '''
    从 JSONL 文件加载评测用例。
    '''
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    cases: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1): 
            line = line.strip()

            if not line:
                continue

            case = json.loads(line)

            if "question" not in case:  
                raise ValueError(f"Line {line_no}: missing field 'question'")

            if "expected_document_id" not in case:
                raise ValueError(f"Line {line_no}: missing field 'expected_document_id'")

            cases.append(case)

    if not cases:   
        raise ValueError(f"No eval cases found in: {path}")

    return cases


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
    '''
    评测函数。
    '''
    cases = load_eval_cases()   

    hit_at_1 = 0
    hit_at_k = 0
    top1_miss_cases = []
    failed_cases: list[dict] = []
    

    for case in cases:
        question = case["question"]
        expected_document_id = case["expected_document_id"]

        results = search_chroma(query=question, top_k=top_k)

        retrieved_document_ids = [result.document_id for result in results]

        is_hit_1 = (
            len(retrieved_document_ids) > 0 # 确保至少有一个结果
            and retrieved_document_ids[0] == expected_document_id
        )   
        is_hit_k = expected_document_id in retrieved_document_ids

        if is_hit_1:
            hit_at_1 += 1

        if is_hit_k:
            hit_at_k += 1
            
        if not is_hit_1 and is_hit_k:
            top1_miss_cases.append({
                    "question": question,
                    "expected_document_id": expected_document_id,
                    "retrieved_document_ids": retrieved_document_ids,
                    "top_results": [result_to_preview(result) for result in results],
                }
            )
        elif not is_hit_k:
            failed_cases.append(
                {
                    "question": question,
                    "expected_document_id": expected_document_id,
                    "retrieved_document_ids": retrieved_document_ids,
                    "top_results": [result_to_preview(result) for result in results],
                }
            )   

    total = len(cases)  

    return {
        "total": total,
        "hit@1": hit_at_1 / total,
        f"hit@{top_k}": hit_at_k / total,
        "top1_miss_cases": top1_miss_cases,
        "failed_cases": failed_cases,
    }


def print_report(report: dict, top_k: int = 3) -> None:
    print("=" * 80)
    print("Retrieval Eval Report - Chroma")
    print("=" * 80)
    print(f"Total: {report['total']}")
    print(f"hit@1: {report['hit@1']:.2f}")
    print(f"hit@{top_k}: {report[f'hit@{top_k}']:.2f}")

    print()
    print(f"top1_miss_cases: {len(report['top1_miss_cases'])}")
    for case in report["top1_miss_cases"]:
        print("-" * 80)
        print(f"question: {case['question']}")
        print(f"expected_document_id: {case['expected_document_id']}")
        print(f"retrieved_document_ids: {case['retrieved_document_ids']}")
        print("top_results:")

        for item in case["top_results"]:
            print(
                f"  - document_id={item['document_id']} | "
                f"chunk_id={item['chunk_id']} | "
                f"distance={item['distance']:.4f}"
            )
            print(f"    preview={item['preview']}")
            
    print()
    print("Failed cases:")
    failed_cases = report["failed_cases"]

    if not failed_cases:
        print("No failed cases.")
        return

    for case in failed_cases:
        print("-" * 80)
        print(f"question: {case['question']}")
        print(f"expected_document_id: {case['expected_document_id']}")
        print(f"retrieved_document_ids: {case['retrieved_document_ids']}")
        print("top_results:")

        for item in case["top_results"]:
            print(
                f"  - document_id={item['document_id']} | "
                f"chunk_id={item['chunk_id']} | "
                f"distance={item['distance']:.4f}"
            )
            print(f"    preview={item['preview']}")


def main() -> None:
    top_k = 3
    report = evaluate(top_k=top_k)
    print_report(report, top_k=top_k)


if __name__ == "__main__":
    main()