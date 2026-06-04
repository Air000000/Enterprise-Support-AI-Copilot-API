from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from experiments.rag_local.query_chroma import search_chroma


DEFAULT_CASES_PATH = Path("experiments/evals/enterprise_rag_cases.jsonl")
DEFAULT_TOP_K = 3

Category = Literal["it", "hr", "finance", "admin", "security"]


class EnterpriseEvalCase(BaseModel):
    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    expected_document_id: str = Field(..., min_length=1)
    category: Category
    tenant_id: str = Field(..., min_length=1)


@dataclass
class EnterpriseEvalResult:
    case_id: str
    question: str
    expected_document_id: str
    category: str
    tenant_id: str
    retrieved_document_ids: list[str]
    retrieved_chunk_ids: list[str]
    hit_at_1: bool
    hit_at_k: bool


def load_cases(path: Path) -> list[EnterpriseEvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Eval cases file not found: {path}")

    cases: list[EnterpriseEvalCase] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            try:
                case = EnterpriseEvalCase.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid eval case at line {line_number}: {exc}"
                ) from exc

            cases.append(case)

    return cases


def evaluate_case(
    case: EnterpriseEvalCase,
    top_k: int,
) -> EnterpriseEvalResult:
    search_results = search_chroma(
        query=case.question,
        top_k=top_k,
        tenant_id=case.tenant_id,
        category=case.category,
    )

    retrieved_document_ids = [
        result.document_id
        for result in search_results
    ]

    retrieved_chunk_ids = [
        result.chunk_id
        for result in search_results
    ]

    hit_at_1 = (
        len(retrieved_document_ids) > 0
        and retrieved_document_ids[0] == case.expected_document_id
    )

    hit_at_k = case.expected_document_id in retrieved_document_ids

    return EnterpriseEvalResult(
        case_id=case.id,
        question=case.question,
        expected_document_id=case.expected_document_id,
        category=case.category,
        tenant_id=case.tenant_id,
        retrieved_document_ids=retrieved_document_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        hit_at_1=hit_at_1,
        hit_at_k=hit_at_k,
    )


def evaluate_cases(
    cases: list[EnterpriseEvalCase],
    top_k: int,
) -> list[EnterpriseEvalResult]:
    results: list[EnterpriseEvalResult] = []

    for case in cases:
        result = evaluate_case(
            case=case,
            top_k=top_k,
        )
        results.append(result)

    return results


def print_category_report(
    results: list[EnterpriseEvalResult],
    top_k: int,
) -> None:
    categories = sorted({result.category for result in results})

    print("\nCategory breakdown:")
    for category in categories:
        category_results = [
            result for result in results
            if result.category == category
        ]

        total = len(category_results)
        hit_at_1_count = sum(result.hit_at_1 for result in category_results)
        hit_at_k_count = sum(result.hit_at_k for result in category_results)

        print("-" * 100)
        print(f"category: {category}")
        print(f"cases:    {total}")
        print(f"hit@1:    {hit_at_1_count / total:.2f} ({hit_at_1_count}/{total})")
        print(f"hit@{top_k}:    {hit_at_k_count / total:.2f} ({hit_at_k_count}/{total})")


def print_case_details(
    title: str,
    results: list[EnterpriseEvalResult],
) -> None:
    print(f"\n{title}:")

    if not results:
        print("No cases.")
        return

    for result in results:
        print("-" * 100)
        print(f"id:         {result.case_id}")
        print(f"category:   {result.category}")
        print(f"question:   {result.question}")
        print(f"expected:   {result.expected_document_id}")
        print(f"documents:  {result.retrieved_document_ids}")
        print(f"chunks:     {result.retrieved_chunk_ids}")


def print_report(
    results: list[EnterpriseEvalResult],
    top_k: int,
) -> None:
    total = len(results)

    if total == 0:
        print("No eval cases found.")
        return

    hit_at_1_count = sum(result.hit_at_1 for result in results)
    hit_at_k_count = sum(result.hit_at_k for result in results)

    top1_miss_cases = [
        result for result in results
        if not result.hit_at_1 and result.hit_at_k
    ]

    failed_cases = [
        result for result in results
        if not result.hit_at_k
    ]

    print("=" * 100)
    print("Enterprise Chroma Retrieval Eval")
    print("-" * 100)
    print(f"Total cases:      {total}")
    print(f"hit@1:            {hit_at_1_count / total:.2f} ({hit_at_1_count}/{total})")
    print(f"hit@{top_k}:            {hit_at_k_count / total:.2f} ({hit_at_k_count}/{total})")
    print(f"top1_miss_cases:  {len(top1_miss_cases)}")
    print(f"failed_cases:     {len(failed_cases)}")

    print_category_report(
        results=results,
        top_k=top_k,
    )

    print_case_details(
        title="Top1 miss cases",
        results=top1_miss_cases,
    )

    print_case_details(
        title="Failed cases",
        results=failed_cases,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate enterprise RAG retrieval with Chroma."
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to enterprise RAG eval cases JSONL file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieval results to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    cases = load_cases(args.cases_path)
    results = evaluate_cases(
        cases=cases,
        top_k=args.top_k,
    )

    print_report(
        results=results,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()