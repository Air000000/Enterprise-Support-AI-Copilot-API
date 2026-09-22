from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

DEFAULT_FREEZE_PATH = Path(
    "experiments/evals/reports/portfolio_v1_rag_freeze/freeze.json"
)
DEFAULT_ARTIFACT_HASHES_PATH = Path(
    "experiments/evals/reports/r4_c1_hybrid_rerank/artifact_hashes.json"
)
DEFAULT_SNAPSHOT_PATH = Path(
    "experiments/evals/reports/r4_c1_hybrid_rerank/train_fused_snapshot.jsonl"
)
DEFAULT_RESULTS_PATH = Path(
    "experiments/evals/reports/r4_c1_hybrid_rerank/train_results.jsonl"
)
DEFAULT_EVIDENCE_LABELS_PATH = Path(
    "experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl"
)
DEFAULT_EVIDENCE_METRICS_PATH = Path(
    "experiments/evals/reports/r1_evidence_audit/evidence_metrics.json"
)
DEFAULT_PHASE_A_OUTPUT_DIR = Path("data/refusal_evidence_sufficiency_phase_a")

EXPECTED_SNAPSHOT_SHA256 = (
    "12db56e50efaf11dab4a28ff3c1b4df223e2ad985a8b48021f4c7dd9fdd889d2"
)
EXPECTED_RESULTS_SHA256 = (
    "12b312a8403cda2c5fe8afe53aa18853891a7e87feaf863e2d438ca15367ee5b"
)
EXPECTED_EVIDENCE_LABELS_SHA256 = (
    "d522eff8daba8435d34ee0e51ad8c56fbbbb759b3f71dbf3a253cd9a1b506013"
)
EXPECTED_LABELED_CASES = 60
EXPECTED_USABLE_CASES = 54
EXPECTED_QUESTIONABLE_CASES = 6
EXPECTED_SUFFICIENT_CASES = 35
EXPECTED_INSUFFICIENT_CASES = 16
EXPECTED_AMBIGUOUS_CASES = 3
FROZEN_CONTEXT_TOP_K = 14

PhaseBTarget = Literal[
    "SUFFICIENT_PROXY",
    "INSUFFICIENT_PROXY",
    "AMBIGUOUS_MULTI_CHUNK",
]

CLASSIFIER_SYSTEM_PROMPT = """
You are an evidence-sufficiency gate for a technical-support RAG system.

Decide whether the supplied Context is sufficient to answer the Question
without relying on outside knowledge.

SUFFICIENT requires direct support for the material claims needed by the
answer. Topic relevance, keyword overlap, or a related product/version alone
is not enough.

If the evidence is partial, ambiguous, conflicting, or would require a
material unsupported inference, return INSUFFICIENT.

Do not answer the user's question.
Return only the required structured decision.
""".strip()

FORBIDDEN_CLASSIFIER_FIELDS = {
    "gold_answer",
    "answerable",
    "is_impossible",
    "relevant_document_ids",
    "evidence_label",
    "evidence_labels",
    "questionable_gold",
    "qrels",
}


@dataclass(frozen=True)
class ContextSource:
    source_id: str
    chunk_id: str
    document_id: str
    content: str


@dataclass(frozen=True)
class ClassifierInput:
    question: str
    sources: tuple[ContextSource, ...]


@dataclass(frozen=True)
class PhaseBCase:
    question_id: str
    classifier_input: ClassifierInput
    target_class: PhaseBTarget

    @property
    def evidence_sufficient(self) -> bool:
        return self.target_class == "SUFFICIENT_PROXY"

    @property
    def binary_gate_eligible(self) -> bool:
        return self.target_class != "AMBIGUOUS_MULTI_CHUNK"


@dataclass(frozen=True)
class PhaseAPreflightSummary:
    freeze_status: str
    snapshot_sha256: str
    results_sha256: str
    evidence_labels_sha256: str
    labeled_case_count: int
    questionable_case_count: int
    usable_case_count: int
    sufficient_case_count: int
    insufficient_case_count: int
    ambiguous_case_count: int
    gated_case_count: int
    context_top_k: int
    provider_calls: int
    dev_artifact_opened: bool
    classifier_inputs_path: str
    targets_path: str
    classifier_inputs_sha256: str
    targets_sha256: str


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"JSONL row must be an object: {path}")
            rows.append(payload)
    return rows


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_freeze_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "FROZEN_FOR_PORTFOLIO_V1":
        raise RuntimeError("portfolio-v1 freeze status is not frozen")

    retrieval = payload.get("retrieval")
    context = payload.get("context")
    if not isinstance(retrieval, Mapping) or not isinstance(context, Mapping):
        raise RuntimeError("portfolio-v1 freeze contract is malformed")

    expected_retrieval = {
        "dense_k": 100,
        "bm25_k": 100,
        "fusion": "equal_weight_chunk_rrf",
        "rrf_k": 60,
        "fused_k": 100,
        "reranker_model": "qwen3-rerank",
        "parameter_research_closed": True,
    }
    for key, expected in expected_retrieval.items():
        if retrieval.get(key) != expected:
            raise RuntimeError(
                f"frozen retrieval contract changed: {key}="
                f"{retrieval.get(key)!r}, expected={expected!r}"
            )

    expected_context = {
        "policy": "flat_rerank_top14_v1",
        "top_k": FROZEN_CONTEXT_TOP_K,
        "dense_top1_rescue": False,
        "forward_siblings": 0,
        "locality_second_rerank": False,
        "whole_document_expansion": False,
        "section_expansion": False,
        "parent_child": False,
        "research_closed": True,
    }
    for key, expected in expected_context.items():
        if context.get(key) != expected:
            raise RuntimeError(
                f"frozen context contract changed: {key}="
                f"{context.get(key)!r}, expected={expected!r}"
            )

    audited = payload.get("audited_evidence")
    if not isinstance(audited, Mapping):
        raise RuntimeError("audited evidence block missing from freeze contract")
    flat_top14 = audited.get("flat_top14")
    if not isinstance(flat_top14, Mapping):
        raise RuntimeError("flat_top14 evidence block missing from freeze contract")
    if audited.get("usable_cases") != EXPECTED_USABLE_CASES:
        raise RuntimeError("freeze contract usable-case count changed")
    if flat_top14.get("answer_hits") != EXPECTED_SUFFICIENT_CASES:
        raise RuntimeError("freeze contract Top14 answer-hit count changed")


def _validate_artifact_hash_contract(payload: Mapping[str, Any]) -> None:
    large = payload.get("large_local_artifacts")
    if not isinstance(large, Mapping):
        raise RuntimeError("R4 C1 artifact hash contract is malformed")

    expected = {
        "train_fused_snapshot.jsonl": EXPECTED_SNAPSHOT_SHA256,
        "train_results.jsonl": EXPECTED_RESULTS_SHA256,
    }
    for filename, expected_sha in expected.items():
        item = large.get(filename)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"missing artifact hash contract for {filename}")
        if item.get("sha256") != expected_sha:
            raise RuntimeError(f"artifact hash contract changed for {filename}")


def _validate_evidence_metrics(
    payload: Mapping[str, Any],
    *,
    actual_labels_sha256: str,
) -> None:
    audit_set = payload.get("audit_set")
    if not isinstance(audit_set, Mapping):
        raise RuntimeError("evidence audit metadata is malformed")

    expected = {
        "labeled_query_count": EXPECTED_LABELED_CASES,
        "evaluated_query_count": EXPECTED_USABLE_CASES,
        "questionable_gold_count": EXPECTED_QUESTIONABLE_CASES,
        "labels_sha256": EXPECTED_EVIDENCE_LABELS_SHA256,
    }
    for key, value in expected.items():
        if audit_set.get(key) != value:
            raise RuntimeError(
                f"evidence audit contract changed: {key}="
                f"{audit_set.get(key)!r}, expected={value!r}"
            )

    if actual_labels_sha256 != EXPECTED_EVIDENCE_LABELS_SHA256:
        raise RuntimeError(
            "evidence label file SHA256 mismatch: "
            f"actual={actual_labels_sha256}"
        )


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise RuntimeError(f"duplicate {label}: {value}")
        indexed[value] = row
    return indexed


def _build_sources(
    snapshot: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    top_k: int,
) -> tuple[ContextSource, ...]:
    candidates = snapshot.get("fused_candidates")
    reranked = result.get("reranked_chunk_ids")
    if not isinstance(candidates, list) or not isinstance(reranked, list):
        raise RuntimeError("frozen C1 record is missing candidate/rerank data")
    if len(reranked) < top_k:
        raise RuntimeError("frozen C1 rerank result is shorter than Top14")

    by_chunk_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise RuntimeError("invalid fused candidate payload")
        chunk_id = str(candidate["chunk_id"])
        if chunk_id in by_chunk_id:
            raise RuntimeError(f"duplicate fused candidate chunk_id: {chunk_id}")
        by_chunk_id[chunk_id] = candidate

    sources: list[ContextSource] = []
    for rank, chunk_id_value in enumerate(reranked[:top_k], start=1):
        chunk_id = str(chunk_id_value)
        candidate = by_chunk_id.get(chunk_id)
        if candidate is None:
            raise RuntimeError(
                "reranked chunk missing from frozen fused snapshot: " + chunk_id
            )
        sources.append(
            ContextSource(
                source_id=f"Source {rank}",
                chunk_id=chunk_id,
                document_id=str(candidate["document_id"]),
                content=str(candidate["content"]),
            )
        )

    return tuple(sources)


def build_phase_b_cases(
    snapshot_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    evidence_label_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = FROZEN_CONTEXT_TOP_K,
) -> tuple[PhaseBCase, ...]:
    snapshots = _index_unique(
        snapshot_rows,
        key="question_id",
        label="snapshot question",
    )
    results = _index_unique(
        result_rows,
        key="question_id",
        label="result question",
    )

    usable_labels = [
        row
        for row in evidence_label_rows
        if not bool(row.get("questionable_gold"))
    ]
    usable_labels.sort(key=lambda row: str(row["question_id"]))

    cases: list[PhaseBCase] = []
    for label_row in usable_labels:
        question_id = str(label_row["question_id"])
        if not question_id.startswith("TRAIN_"):
            raise RuntimeError(f"Phase B contains non-TRAIN case: {question_id}")

        snapshot = snapshots.get(question_id)
        result = results.get(question_id)
        if snapshot is None or result is None:
            raise RuntimeError(f"missing frozen C1 artifacts for {question_id}")

        sources = _build_sources(snapshot, result, top_k=top_k)
        top_chunk_ids = {source.chunk_id for source in sources}

        candidate_labels = label_row.get("candidate_labels")
        if not isinstance(candidate_labels, list):
            raise RuntimeError(f"invalid evidence labels for {question_id}")

        answer_bearing_ids = {
            str(item["chunk_id"])
            for item in candidate_labels
            if isinstance(item, Mapping) and int(item["evidence_label"]) == 2
        }

        if top_chunk_ids & answer_bearing_ids:
            target_class: PhaseBTarget = "SUFFICIENT_PROXY"
        elif answer_bearing_ids:
            target_class = "INSUFFICIENT_PROXY"
        else:
            target_class = "AMBIGUOUS_MULTI_CHUNK"

        cases.append(
            PhaseBCase(
                question_id=question_id,
                classifier_input=ClassifierInput(
                    question=str(snapshot["question"]),
                    sources=sources,
                ),
                target_class=target_class,
            )
        )

    return tuple(cases)


def classifier_payload(classifier_input: ClassifierInput) -> dict[str, Any]:
    return {
        "question": classifier_input.question,
        "sources": [
            {
                "source_id": source.source_id,
                "content": source.content,
            }
            for source in classifier_input.sources
        ],
    }


def render_classifier_user_prompt(classifier_input: ClassifierInput) -> str:
    context = "\n\n---\n\n".join(
        f"[{source.source_id}]\n{source.content}"
        for source in classifier_input.sources
    )
    return (
        "Question:\n"
        f"{classifier_input.question}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Return a JSON object with exactly these fields:\n"
        '- decision: "SUFFICIENT" or "INSUFFICIENT"\n'
        "- reason: a short evidence-based explanation\n"
        "- supporting_source_ids: a JSON array using Source N identifiers only"
    )


def build_classifier_messages(
    classifier_input: ClassifierInput,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_classifier_user_prompt(classifier_input),
        },
    ]


def assert_classifier_input_is_clean(case: PhaseBCase) -> None:
    payload = classifier_payload(case.classifier_input)
    if set(payload) != {"question", "sources"}:
        raise RuntimeError(
            "classifier payload contains unexpected top-level fields"
        )

    sources = payload["sources"]
    if not isinstance(sources, list):
        raise RuntimeError("classifier payload sources must be a list")
    for source in sources:
        if not isinstance(source, Mapping):
            raise RuntimeError("classifier source must be an object")
        if set(source) != {"source_id", "content"}:
            raise RuntimeError("classifier source contains unexpected fields")
        if set(source) & FORBIDDEN_CLASSIFIER_FIELDS:
            raise RuntimeError("classifier source contains forbidden fields")

    if set(payload) & FORBIDDEN_CLASSIFIER_FIELDS:
        raise RuntimeError("classifier payload contains forbidden fields")

    prompt = "\n".join(
        message["content"]
        for message in build_classifier_messages(case.classifier_input)
    )
    if case.question_id in prompt:
        raise RuntimeError("question_id leaked into classifier prompt")


def _classifier_input_row(case: PhaseBCase) -> dict[str, Any]:
    return {
        "question_id": case.question_id,
        **classifier_payload(case.classifier_input),
    }


def _target_row(case: PhaseBCase) -> dict[str, Any]:
    return {
        "question_id": case.question_id,
        "target_class": case.target_class,
        "binary_gate_eligible": case.binary_gate_eligible,
    }


def run_phase_a_preflight(
    *,
    freeze_path: str | Path = DEFAULT_FREEZE_PATH,
    artifact_hashes_path: str | Path = DEFAULT_ARTIFACT_HASHES_PATH,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    evidence_labels_path: str | Path = DEFAULT_EVIDENCE_LABELS_PATH,
    evidence_metrics_path: str | Path = DEFAULT_EVIDENCE_METRICS_PATH,
    output_dir: str | Path = DEFAULT_PHASE_A_OUTPUT_DIR,
) -> PhaseAPreflightSummary:
    freeze = _read_json(freeze_path)
    validate_freeze_contract(freeze)

    artifact_hashes = _read_json(artifact_hashes_path)
    _validate_artifact_hash_contract(artifact_hashes)

    snapshot = Path(snapshot_path)
    results = Path(results_path)
    if not snapshot.exists() or not results.exists():
        missing = [
            str(path)
            for path in (snapshot, results)
            if not path.exists()
        ]
        raise RuntimeError(
            "Phase A requires the frozen local R4 C1 artifacts: "
            + ", ".join(missing)
        )

    snapshot_sha = _sha256(snapshot)
    results_sha = _sha256(results)
    if snapshot_sha != EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError(f"frozen snapshot SHA256 mismatch: {snapshot_sha}")
    if results_sha != EXPECTED_RESULTS_SHA256:
        raise RuntimeError(f"frozen results SHA256 mismatch: {results_sha}")

    evidence_labels_sha = _sha256(evidence_labels_path)
    evidence_metrics = _read_json(evidence_metrics_path)
    _validate_evidence_metrics(
        evidence_metrics,
        actual_labels_sha256=evidence_labels_sha,
    )

    snapshot_rows = _read_jsonl(snapshot)
    result_rows = _read_jsonl(results)
    evidence_label_rows = _read_jsonl(evidence_labels_path)

    if len(snapshot_rows) != 450 or len(result_rows) != 450:
        raise RuntimeError(
            "frozen C1 artifacts must each contain 450 TRAIN rows"
        )
    if len(evidence_label_rows) != EXPECTED_LABELED_CASES:
        raise RuntimeError("evidence label row count changed")

    cases = build_phase_b_cases(
        snapshot_rows,
        result_rows,
        evidence_label_rows,
    )
    if len(cases) != EXPECTED_USABLE_CASES:
        raise RuntimeError(
            f"usable Phase B case count mismatch: {len(cases)}"
        )

    sufficient_count = sum(
        case.target_class == "SUFFICIENT_PROXY"
        for case in cases
    )
    insufficient_count = sum(
        case.target_class == "INSUFFICIENT_PROXY"
        for case in cases
    )
    ambiguous_count = sum(
        case.target_class == "AMBIGUOUS_MULTI_CHUNK"
        for case in cases
    )
    gated_count = sufficient_count + insufficient_count

    if sufficient_count != EXPECTED_SUFFICIENT_CASES:
        raise RuntimeError(
            "Flat Top14 answer-bearing count mismatch: "
            f"actual={sufficient_count}, "
            f"expected={EXPECTED_SUFFICIENT_CASES}"
        )
    if insufficient_count != EXPECTED_INSUFFICIENT_CASES:
        raise RuntimeError(
            "Flat Top14 defensible insufficient-proxy count mismatch: "
            f"actual={insufficient_count}, "
            f"expected={EXPECTED_INSUFFICIENT_CASES}"
        )
    if ambiguous_count != EXPECTED_AMBIGUOUS_CASES:
        raise RuntimeError(
            "multi-chunk ambiguous-case count mismatch: "
            f"actual={ambiguous_count}, "
            f"expected={EXPECTED_AMBIGUOUS_CASES}"
        )
    if gated_count != 51:
        raise RuntimeError(
            f"binary Phase B gate must contain 51 cases: {gated_count}"
        )

    for case in cases:
        assert_classifier_input_is_clean(case)
        if len(case.classifier_input.sources) != FROZEN_CONTEXT_TOP_K:
            raise RuntimeError(
                "classifier input did not preserve exactly Top14"
            )

    output = Path(output_dir)
    inputs_path = output / "phase_b_classifier_inputs.jsonl"
    targets_path = output / "phase_b_targets.jsonl"
    preflight_path = output / "preflight.json"

    _write_jsonl(
        inputs_path,
        (_classifier_input_row(case) for case in cases),
    )
    _write_jsonl(
        targets_path,
        (_target_row(case) for case in cases),
    )

    summary = PhaseAPreflightSummary(
        freeze_status=str(freeze["status"]),
        snapshot_sha256=snapshot_sha,
        results_sha256=results_sha,
        evidence_labels_sha256=evidence_labels_sha,
        labeled_case_count=len(evidence_label_rows),
        questionable_case_count=sum(
            bool(row.get("questionable_gold"))
            for row in evidence_label_rows
        ),
        usable_case_count=len(cases),
        sufficient_case_count=sufficient_count,
        insufficient_case_count=insufficient_count,
        ambiguous_case_count=ambiguous_count,
        gated_case_count=gated_count,
        context_top_k=FROZEN_CONTEXT_TOP_K,
        provider_calls=0,
        dev_artifact_opened=False,
        classifier_inputs_path=str(inputs_path),
        targets_path=str(targets_path),
        classifier_inputs_sha256=_sha256(inputs_path),
        targets_sha256=_sha256(targets_path),
    )
    _write_json(preflight_path, asdict(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-provider Phase A preflight for refusal "
            "evidence sufficiency."
        )
    )
    parser.add_argument(
        "--freeze-path",
        default=str(DEFAULT_FREEZE_PATH),
    )
    parser.add_argument(
        "--artifact-hashes-path",
        default=str(DEFAULT_ARTIFACT_HASHES_PATH),
    )
    parser.add_argument(
        "--snapshot-path",
        default=str(DEFAULT_SNAPSHOT_PATH),
    )
    parser.add_argument(
        "--results-path",
        default=str(DEFAULT_RESULTS_PATH),
    )
    parser.add_argument(
        "--evidence-labels-path",
        default=str(DEFAULT_EVIDENCE_LABELS_PATH),
    )
    parser.add_argument(
        "--evidence-metrics-path",
        default=str(DEFAULT_EVIDENCE_METRICS_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PHASE_A_OUTPUT_DIR),
    )
    args = parser.parse_args()

    summary = run_phase_a_preflight(
        freeze_path=args.freeze_path,
        artifact_hashes_path=args.artifact_hashes_path,
        snapshot_path=args.snapshot_path,
        results_path=args.results_path,
        evidence_labels_path=args.evidence_labels_path,
        evidence_metrics_path=args.evidence_metrics_path,
        output_dir=args.output_dir,
    )

    print("REFUSAL_PREFLIGHT=PASS")
    print(f"USABLE_CASES={summary.usable_case_count}")
    print(f"SUFFICIENT_PROXY_CASES={summary.sufficient_case_count}")
    print(f"INSUFFICIENT_PROXY_CASES={summary.insufficient_case_count}")
    print(
        "AMBIGUOUS_MULTI_CHUNK_CASES="
        f"{summary.ambiguous_case_count}"
    )
    print(f"GATED_CASES={summary.gated_case_count}")
    print(f"CONTEXT_TOP_K={summary.context_top_k}")
    print(f"PROVIDER_CALLS={summary.provider_calls}")
    print("DEV_ARTIFACT_OPENED=NO")
    print(
        "CLASSIFIER_INPUTS_SHA256="
        f"{summary.classifier_inputs_sha256}"
    )
    print(f"TARGETS_SHA256={summary.targets_sha256}")
    print("NEXT_ACTION=STOP_FOR_REFUSAL_PREFLIGHT_REVIEW")


if __name__ == "__main__":
    main()
