from pathlib import Path
import json
import time


LEARNING_EVAL_FILE = Path("experiments/evals/eval_learning_questions.jsonl")
ENTERPRISE_EVAL_FILE = Path("experiments/evals/eval_enterprise_questions.jsonl")

EVAL_FILE = LEARNING_EVAL_FILE


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

def update_category_stats(
    category_stats: dict,
    category: str,
    is_hit_1: bool,
    is_hit_k: bool,
    reciprocal_rank: float,
    latency_ms: float,
) -> None:
    """
    更新指定类别的统计数据。
    """
    if category not in category_stats:
        category_stats[category] = {
            "total": 0,
            "hit_at_1": 0,
            "hit_at_k": 0,
            "reciprocal_ranks": [],
            "latencies_ms": [],
        }

    stats = category_stats[category]
    stats["total"] += 1

    if is_hit_1:
        stats["hit_at_1"] += 1

    if is_hit_k:
        stats["hit_at_k"] += 1

    stats["reciprocal_ranks"].append(reciprocal_rank)
    stats["latencies_ms"].append(latency_ms)

def build_category_metrics(category_stats: dict, top_k: int) -> dict:
    """
    根据统计数据计算每个类别的评测指标。
    """
    category_metrics = {}

    for category, stats in sorted(category_stats.items()):
        total = stats["total"]

        category_metrics[category] = {
            "total": total,
            "hit@1": stats["hit_at_1"] / total,
            f"hit@{top_k}": stats["hit_at_k"] / total,
            f"mrr@{top_k}": sum(stats["reciprocal_ranks"]) / total,
            "avg_latency_ms": sum(stats["latencies_ms"]) / total,
        }

    return category_metrics

def evaluate_retriever(
    retriever,
    result_to_preview,
    top_k: int = 3,
    eval_file: Path = EVAL_FILE,
) -> dict:
    category_stats = {}
    cases = load_eval_cases(eval_file)

    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    top1_miss_cases = []
    failed_cases: list[dict] = []

    for case in cases:
        question = case["question"]
        category = case.get("category", "uncategorized")
        expected_document_id = case["expected_document_id"]

        start_time = time.perf_counter()
        results = retriever(query=question, top_k=top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies_ms.append(latency_ms)

        retrieved_document_ids = [result.document_id for result in results]

        is_hit_1 = (
            len(retrieved_document_ids) > 0
            and retrieved_document_ids[0] == expected_document_id
        )
        is_hit_k = expected_document_id in retrieved_document_ids

        if is_hit_1:
            hit_at_1 += 1

        if is_hit_k:
            hit_at_k += 1
            rank = retrieved_document_ids.index(expected_document_id) + 1
            reciprocal_rank = 1 / rank
        else:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

        update_category_stats(
            category_stats=category_stats,
            category=category,
            is_hit_1=is_hit_1,
            is_hit_k=is_hit_k,
            reciprocal_rank=reciprocal_rank,
            latency_ms=latency_ms,
        )

        if not is_hit_1 and is_hit_k:
            top1_miss_cases.append(
                {
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
        f"mrr@{top_k}": sum(reciprocal_ranks) / total,
        "avg_latency_ms": sum(latencies_ms) / total,
        "top1_miss_cases": top1_miss_cases,
        "failed_cases": failed_cases,
        "category_metrics": build_category_metrics(category_stats, top_k),
    }


def print_retrieval_report(
    report: dict,
    title: str,
    top_k: int = 3,
    metric_key: str = "score",
) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total: {report['total']}")
    print(f"hit@1: {report['hit@1']:.2f}")
    print(f"hit@{top_k}: {report[f'hit@{top_k}']:.2f}")
    print(f"mrr@{top_k}: {report[f'mrr@{top_k}']:.2f}")
    print(f"avg_latency_ms: {report['avg_latency_ms']:.2f}")

    category_metrics = report.get("category_metrics", {})

    if category_metrics:
        print()
        print("Category breakdown:")
        for category, metrics in category_metrics.items():
            print(
                f"- {category}: "
                f"total={metrics['total']}, "
                f"hit@1={metrics['hit@1']:.2f}, "
                f"hit@{top_k}={metrics[f'hit@{top_k}']:.2f}, "
                f"mrr@{top_k}={metrics[f'mrr@{top_k}']:.2f}, "
                f"avg_latency_ms={metrics['avg_latency_ms']:.2f}"
            )

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
                f"{metric_key}={item[metric_key]:.4f}"
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
                f"{metric_key}={item[metric_key]:.4f}"
            )
            print(f"    preview={item['preview']}")