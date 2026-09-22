from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI

from experiments.evals.refusal_evidence_sufficiency import (
    ClassifierInput,
    ContextSource,
    build_classifier_messages,
)

DEFAULT_CONTRACT_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "phase_b_run_contract.json"
)
DEFAULT_INPUTS_PATH = Path(
    "data/refusal_evidence_sufficiency_phase_a/"
    "phase_b_classifier_inputs.jsonl"
)
DEFAULT_TARGETS_PATH = Path(
    "data/refusal_evidence_sufficiency_phase_a/"
    "phase_b_targets.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/refusal_evidence_sufficiency_phase_b"
)
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_OUTPUT_DIR / "predictions.jsonl"
)
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "summary.json"
)

EXPECTED_INPUTS_SHA256 = (
    "bf0164d096758bf0c4cdda3ea0e106a3914e6e43aa1b627f13e8d6cd7efb9fa9"
)
EXPECTED_TARGETS_SHA256 = (
    "7b1535d033da3b8bf6b6d94c9a6f93105c76e593828dd53e1b0873d35b89e655"
)
EXPECTED_CASES = 54
EXPECTED_GATED_CASES = 51
EXPECTED_SUFFICIENT_PROXY = 35
EXPECTED_INSUFFICIENT_PROXY = 16
EXPECTED_AMBIGUOUS = 3

Decision = Literal["SUFFICIENT", "INSUFFICIENT"]
TargetClass = Literal[
    "SUFFICIENT_PROXY",
    "INSUFFICIENT_PROXY",
    "AMBIGUOUS_MULTI_CHUNK",
]


@dataclass(frozen=True)
class PhaseBPrediction:
    question_id: str
    decision: Decision
    reason: str
    supporting_source_ids: tuple[str, ...]
    model: str
    request_id: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_cny: float


@dataclass(frozen=True)
class PhaseBEvaluation:
    total_predictions: int
    gated_predictions: int
    ambiguous_predictions: int
    accuracy: float
    balanced_accuracy: float
    sufficient_recall: float
    insufficient_recall: float
    sufficient_to_insufficient: int
    insufficient_to_sufficient: int
    sufficient_true_positive: int
    sufficient_false_negative: int
    insufficient_true_negative: int
    insufficient_false_positive: int
    ambiguous_sufficient: int
    ambiguous_insufficient: int
    provider_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_cny: float
    latency_p50_ms: float
    latency_p95_ms: float
    gate_balanced_accuracy_pass: bool
    gate_sufficient_recall_pass: bool
    gate_insufficient_recall_pass: bool
    gate_over_refusal_pass: bool
    decision: str


def _sha256(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"JSONL row must be an object: {path}")
        rows.append(payload)
    return rows


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("run") != "refusal_evidence_sufficiency_phase_b_v1":
        raise RuntimeError("unexpected Phase B run identifier")
    if contract.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("Phase B contract status changed")

    population = contract.get("population")
    frozen_inputs = contract.get("frozen_inputs")
    classifier = contract.get("classifier")
    gate = contract.get("gate")
    controls = contract.get("controls")
    pricing = contract.get("pricing_snapshot")

    if not all(
        isinstance(value, Mapping)
        for value in (
            population,
            frozen_inputs,
            classifier,
            gate,
            controls,
            pricing,
        )
    ):
        raise RuntimeError("Phase B contract is malformed")

    expected_population = {
        "total_cases": 54,
        "gated_cases": 51,
        "sufficient_proxy": 35,
        "insufficient_proxy": 16,
        "ambiguous_multi_chunk": 3,
    }
    for key, expected in expected_population.items():
        if population.get(key) != expected:
            raise RuntimeError(
                f"Phase B population changed: {key}="
                f"{population.get(key)!r}, expected={expected!r}"
            )

    if frozen_inputs.get("classifier_inputs_sha256") != EXPECTED_INPUTS_SHA256:
        raise RuntimeError("classifier input SHA contract changed")
    if frozen_inputs.get("targets_sha256") != EXPECTED_TARGETS_SHA256:
        raise RuntimeError("target SHA contract changed")
    if frozen_inputs.get("context_policy") != "flat_rerank_top14_v1":
        raise RuntimeError("context policy changed")
    if frozen_inputs.get("top_k") != 14:
        raise RuntimeError("context TopK changed")

    expected_classifier = {
        "model": "qwen3.5-plus-2026-04-20",
        "temperature": 0.0,
        "enable_thinking": False,
        "response_format": "json_object",
        "max_output_tokens": 256,
        "max_calls": 54,
    }
    for key, expected in expected_classifier.items():
        if classifier.get(key) != expected:
            raise RuntimeError(
                f"classifier contract changed: {key}="
                f"{classifier.get(key)!r}, expected={expected!r}"
            )

    expected_gate = {
        "balanced_accuracy_min": 0.80,
        "sufficient_recall_min": 0.85,
        "insufficient_recall_min": 0.70,
        "sufficient_to_insufficient_max": 5,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise RuntimeError(
                f"gate contract changed: {key}="
                f"{gate.get(key)!r}, expected={expected!r}"
            )

    if pricing.get("hard_cost_cap_cny") != 3.0:
        raise RuntimeError("Phase B hard cost cap changed")
    if controls.get("dev_artifact_opened") is not False:
        raise RuntimeError("DEV must remain closed")
    if controls.get("retrieval_calls") != 0:
        raise RuntimeError("Phase B must not rerun retrieval")
    if controls.get("rerank_calls") != 0:
        raise RuntimeError("Phase B must not rerun rerank")
    if controls.get("generation_calls") != 0:
        raise RuntimeError("Phase B must not run generation")
    if controls.get("judge_calls") != 0:
        raise RuntimeError("Phase B must not run judge")


def load_classifier_inputs(
    path: str | Path = DEFAULT_INPUTS_PATH,
) -> list[dict[str, Any]]:
    source = Path(path)
    actual_sha = _sha256(source)
    if actual_sha != EXPECTED_INPUTS_SHA256:
        raise RuntimeError(
            "classifier input SHA256 mismatch: "
            f"actual={actual_sha}"
        )

    rows = _read_jsonl(source)
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(
            f"Phase B requires exactly {EXPECTED_CASES} classifier inputs"
        )

    question_ids: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id", ""))
        if not question_id.startswith("TRAIN_"):
            raise RuntimeError("Phase B input contains non-TRAIN case")
        if question_id in question_ids:
            raise RuntimeError(f"duplicate question_id: {question_id}")
        question_ids.add(question_id)

        if set(row) != {"question_id", "question", "sources"}:
            raise RuntimeError(
                f"classifier input has unexpected fields: {question_id}"
            )
        sources = row["sources"]
        if not isinstance(sources, list) or len(sources) != 14:
            raise RuntimeError(
                f"classifier input must contain exactly 14 sources: {question_id}"
            )
        for index, source_item in enumerate(sources, start=1):
            if not isinstance(source_item, Mapping):
                raise RuntimeError("classifier source must be an object")
            if set(source_item) != {"source_id", "content"}:
                raise RuntimeError("classifier source contains extra metadata")
            if source_item["source_id"] != f"Source {index}":
                raise RuntimeError("classifier source order changed")

    return rows


def _classifier_input_from_row(row: Mapping[str, Any]) -> ClassifierInput:
    return ClassifierInput(
        question=str(row["question"]),
        sources=tuple(
            ContextSource(
                source_id=str(source["source_id"]),
                chunk_id="",
                document_id="",
                content=str(source["content"]),
            )
            for source in row["sources"]
        ),
    )


def _parse_classifier_json(
    raw_text: str,
    *,
    allowed_source_ids: set[str],
) -> tuple[Decision, str, tuple[str, ...]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("classifier returned invalid JSON") from error

    if not isinstance(payload, dict):
        raise RuntimeError("classifier response must be a JSON object")
    if set(payload) != {
        "decision",
        "reason",
        "supporting_source_ids",
    }:
        raise RuntimeError("classifier response schema mismatch")

    decision = str(payload["decision"])
    if decision not in {"SUFFICIENT", "INSUFFICIENT"}:
        raise RuntimeError("classifier decision is invalid")

    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("classifier reason must be non-empty")

    source_ids = payload["supporting_source_ids"]
    if not isinstance(source_ids, list):
        raise RuntimeError("supporting_source_ids must be a list")

    normalized: list[str] = []
    for source_id in source_ids:
        value = str(source_id)
        if value not in allowed_source_ids:
            raise RuntimeError(
                "classifier cited a source outside the frozen context: "
                + value
            )
        if value not in normalized:
            normalized.append(value)

    if decision == "SUFFICIENT" and not normalized:
        raise RuntimeError(
            "SUFFICIENT decision must cite at least one source"
        )

    return decision, reason.strip(), tuple(normalized)


def resolve_classifier_auth() -> tuple[str, str, str]:
    """Resolve a Singapore-scoped key/base-URL pair."""
    load_dotenv()

    phase_b_key = os.getenv("DASHSCOPE_PHASE_B_API_KEY")
    phase_b_base = os.getenv("DASHSCOPE_PHASE_B_BASE_URL")

    if phase_b_key or phase_b_base:
        if not phase_b_key or not phase_b_base:
            raise RuntimeError(
                "DASHSCOPE_PHASE_B_API_KEY and "
                "DASHSCOPE_PHASE_B_BASE_URL must be configured together."
            )
        api_key = phase_b_key
        base_url = phase_b_base
        key_source = "DASHSCOPE_PHASE_B_API_KEY"
    else:
        rerank_key = os.getenv("DASHSCOPE_RERANK_API_KEY")
        rerank_base = os.getenv("DASHSCOPE_RERANK_BASE_URL")
        if rerank_key and rerank_base:
            api_key = rerank_key
            base_url = rerank_base
            key_source = "DASHSCOPE_RERANK_API_KEY"
        else:
            generic_key = os.getenv("DASHSCOPE_API_KEY")
            if generic_key:
                raise RuntimeError(
                    "Phase B is frozen to Singapore/International, but only "
                    "the generic DASHSCOPE_API_KEY is configured. The project "
                    "pairs that key with the Beijing chat endpoint, so it must "
                    "not be reused against a Singapore endpoint. Configure "
                    "DASHSCOPE_PHASE_B_API_KEY + DASHSCOPE_PHASE_B_BASE_URL, "
                    "or keep the existing Singapore rerank key/base-url pair."
                )
            raise RuntimeError(
                "Missing Singapore credentials for Phase B. Configure "
                "DASHSCOPE_PHASE_B_API_KEY + DASHSCOPE_PHASE_B_BASE_URL, "
                "or DASHSCOPE_RERANK_API_KEY + DASHSCOPE_RERANK_BASE_URL."
            )

    base_url = base_url.rstrip("/")
    shared_singapore = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    workspace_host = ".ap-southeast-1.maas.aliyuncs.com/"
    workspace_suffixes = (
        "compatible-api/v1",
        "compatible-mode/v1",
    )
    if not (
        base_url == shared_singapore
        or (
            base_url.startswith("https://")
            and workspace_host in base_url
            and base_url.endswith(workspace_suffixes)
        )
    ):
        raise RuntimeError(
            "Phase B base URL must be a Singapore/International "
            "OpenAI-compatible endpoint. Got: "
            + base_url
        )

    return api_key, base_url, key_source


def get_classifier_client() -> OpenAI:
    api_key, base_url, _ = resolve_classifier_auth()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )


def classify_one(
    row: Mapping[str, Any],
    *,
    client: Any,
    model: str,
    input_rate_cny_per_1m: float,
    output_rate_cny_per_1m: float,
    clock: Callable[[], float] = time.perf_counter,
) -> PhaseBPrediction:
    classifier_input = _classifier_input_from_row(row)
    messages = build_classifier_messages(classifier_input)
    question_id = str(row["question_id"])

    prompt_text = "\n".join(
        str(message["content"])
        for message in messages
    )
    if question_id in prompt_text:
        raise RuntimeError("question_id leaked into classifier prompt")

    started = clock()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    latency_ms = (clock() - started) * 1000.0

    raw_text = response.choices[0].message.content or ""
    allowed_source_ids = {
        str(source["source_id"])
        for source in row["sources"]
    }
    decision, reason, supporting_source_ids = _parse_classifier_json(
        raw_text,
        allowed_source_ids=allowed_source_ids,
    )

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(
        getattr(usage, "completion_tokens", 0) or 0
    )
    total_tokens = int(
        getattr(usage, "total_tokens", 0)
        or prompt_tokens + completion_tokens
    )
    cost_cny = (
        prompt_tokens * input_rate_cny_per_1m
        + completion_tokens * output_rate_cny_per_1m
    ) / 1_000_000.0

    request_id = getattr(response, "id", None)
    if request_id is not None:
        request_id = str(request_id)

    return PhaseBPrediction(
        question_id=question_id,
        decision=decision,
        reason=reason,
        supporting_source_ids=supporting_source_ids,
        model=model,
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        estimated_cost_cny=cost_cny,
    )


def load_predictions(
    path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> list[PhaseBPrediction]:
    source = Path(path)
    if not source.exists():
        return []

    predictions: list[PhaseBPrediction] = []
    seen: set[str] = set()
    for payload in _read_jsonl(source):
        payload = dict(payload)
        payload["supporting_source_ids"] = tuple(
            str(value)
            for value in payload["supporting_source_ids"]
        )
        prediction = PhaseBPrediction(**payload)
        if prediction.question_id in seen:
            raise RuntimeError(
                f"duplicate checkpoint question_id: {prediction.question_id}"
            )
        seen.add(prediction.question_id)
        predictions.append(prediction)
    return predictions


def run_paid_phase_b(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    inputs_path: str | Path = DEFAULT_INPUTS_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    client: Any | None = None,
) -> tuple[PhaseBPrediction, ...]:
    contract = _read_json(contract_path)
    validate_contract(contract)

    rows = load_classifier_inputs(inputs_path)
    classifier = contract["classifier"]
    pricing = contract["pricing_snapshot"]

    model = str(classifier["model"])
    max_calls = int(classifier["max_calls"])
    hard_cost_cap_cny = float(pricing["hard_cost_cap_cny"])
    input_rate = float(pricing["input_cny_per_1m_tokens"])
    output_rate = float(pricing["output_cny_per_1m_tokens"])

    checkpoint = Path(checkpoint_path)
    existing = load_predictions(checkpoint)
    by_question_id = {
        prediction.question_id: prediction
        for prediction in existing
    }

    input_ids = {str(row["question_id"]) for row in rows}
    if not set(by_question_id).issubset(input_ids):
        raise RuntimeError(
            "checkpoint contains question IDs outside frozen inputs"
        )

    spent_cny = sum(
        prediction.estimated_cost_cny
        for prediction in existing
    )
    if spent_cny >= hard_cost_cap_cny and len(existing) < len(rows):
        raise RuntimeError("hard cost cap already reached")

    provider = client or get_classifier_client()

    for row in rows:
        question_id = str(row["question_id"])
        if question_id in by_question_id:
            continue
        if len(by_question_id) >= max_calls:
            raise RuntimeError("maximum provider-call count reached")

        prediction = classify_one(
            row,
            client=provider,
            model=model,
            input_rate_cny_per_1m=input_rate,
            output_rate_cny_per_1m=output_rate,
        )
        projected_spend = spent_cny + prediction.estimated_cost_cny
        if projected_spend > hard_cost_cap_cny:
            raise RuntimeError(
                "hard cost cap exceeded after provider response; "
                "prediction was not checkpointed"
            )

        _append_jsonl(checkpoint, asdict(prediction))
        by_question_id[question_id] = prediction
        spent_cny = projected_spend

    if len(by_question_id) != EXPECTED_CASES:
        raise RuntimeError(
            "Phase B did not complete all 54 classifier calls"
        )

    return tuple(
        by_question_id[str(row["question_id"])]
        for row in rows
    )


def load_targets(
    path: str | Path = DEFAULT_TARGETS_PATH,
) -> list[dict[str, Any]]:
    source = Path(path)
    actual_sha = _sha256(source)
    if actual_sha != EXPECTED_TARGETS_SHA256:
        raise RuntimeError(
            "Phase B target SHA256 mismatch: "
            f"actual={actual_sha}"
        )
    rows = _read_jsonl(source)
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError("Phase B target count changed")
    return rows


def evaluate_predictions(
    predictions: Sequence[PhaseBPrediction],
    targets: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> PhaseBEvaluation:
    validate_contract(contract)

    pred_by_id = {
        prediction.question_id: prediction
        for prediction in predictions
    }
    if len(pred_by_id) != len(predictions):
        raise RuntimeError("duplicate prediction question_id")

    target_by_id: dict[str, Mapping[str, Any]] = {}
    for target in targets:
        question_id = str(target["question_id"])
        if question_id in target_by_id:
            raise RuntimeError("duplicate target question_id")
        target_by_id[question_id] = target

    if set(pred_by_id) != set(target_by_id):
        raise RuntimeError("prediction and target question IDs differ")

    tp = fn = tn = fp = 0
    ambiguous_sufficient = 0
    ambiguous_insufficient = 0

    for question_id, target in target_by_id.items():
        prediction = pred_by_id[question_id]
        target_class = str(target["target_class"])
        gate_eligible = bool(target["binary_gate_eligible"])

        if not gate_eligible:
            if target_class != "AMBIGUOUS_MULTI_CHUNK":
                raise RuntimeError("non-gated target must be ambiguous")
            if prediction.decision == "SUFFICIENT":
                ambiguous_sufficient += 1
            else:
                ambiguous_insufficient += 1
            continue

        if target_class == "SUFFICIENT_PROXY":
            if prediction.decision == "SUFFICIENT":
                tp += 1
            else:
                fn += 1
        elif target_class == "INSUFFICIENT_PROXY":
            if prediction.decision == "INSUFFICIENT":
                tn += 1
            else:
                fp += 1
        else:
            raise RuntimeError("unexpected gated target class")

    gated_count = tp + fn + tn + fp
    if gated_count != EXPECTED_GATED_CASES:
        raise RuntimeError(
            f"expected {EXPECTED_GATED_CASES} gated predictions"
        )

    sufficient_recall = tp / (tp + fn)
    insufficient_recall = tn / (tn + fp)
    balanced_accuracy = (
        sufficient_recall + insufficient_recall
    ) / 2.0
    accuracy = (tp + tn) / gated_count

    gate = contract["gate"]
    gate_balanced = (
        balanced_accuracy
        >= float(gate["balanced_accuracy_min"])
    )
    gate_sufficient = (
        sufficient_recall
        >= float(gate["sufficient_recall_min"])
    )
    gate_insufficient = (
        insufficient_recall
        >= float(gate["insufficient_recall_min"])
    )
    gate_over_refusal = (
        fn
        <= int(gate["sufficient_to_insufficient_max"])
    )

    decision = (
        "ADMIT_TRAIN_ABSTENTION_PROBE"
        if all(
            (
                gate_balanced,
                gate_sufficient,
                gate_insufficient,
                gate_over_refusal,
            )
        )
        else "REJECT_EVIDENCE_SUFFICIENCY_V1"
    )

    latencies = [
        prediction.latency_ms
        for prediction in predictions
    ]

    return PhaseBEvaluation(
        total_predictions=len(predictions),
        gated_predictions=gated_count,
        ambiguous_predictions=(
            ambiguous_sufficient + ambiguous_insufficient
        ),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        sufficient_recall=sufficient_recall,
        insufficient_recall=insufficient_recall,
        sufficient_to_insufficient=fn,
        insufficient_to_sufficient=fp,
        sufficient_true_positive=tp,
        sufficient_false_negative=fn,
        insufficient_true_negative=tn,
        insufficient_false_positive=fp,
        ambiguous_sufficient=ambiguous_sufficient,
        ambiguous_insufficient=ambiguous_insufficient,
        provider_calls=len(predictions),
        prompt_tokens=sum(
            prediction.prompt_tokens
            for prediction in predictions
        ),
        completion_tokens=sum(
            prediction.completion_tokens
            for prediction in predictions
        ),
        total_tokens=sum(
            prediction.total_tokens
            for prediction in predictions
        ),
        estimated_cost_cny=sum(
            prediction.estimated_cost_cny
            for prediction in predictions
        ),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        gate_balanced_accuracy_pass=gate_balanced,
        gate_sufficient_recall_pass=gate_sufficient,
        gate_insufficient_recall_pass=gate_insufficient,
        gate_over_refusal_pass=gate_over_refusal,
        decision=decision,
    )


def evaluate_checkpoint(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    targets_path: str | Path = DEFAULT_TARGETS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> PhaseBEvaluation:
    contract = _read_json(contract_path)
    predictions = load_predictions(checkpoint_path)
    if len(predictions) != EXPECTED_CASES:
        raise RuntimeError(
            "evaluation requires all 54 classifier predictions"
        )
    targets = load_targets(targets_path)

    summary = evaluate_predictions(
        predictions,
        targets,
        contract=contract,
    )
    _write_json(Path(summary_path), asdict(summary))
    return summary


def print_preflight(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    inputs_path: str | Path = DEFAULT_INPUTS_PATH,
) -> None:
    contract = _read_json(contract_path)
    validate_contract(contract)
    rows = load_classifier_inputs(inputs_path)

    char_count = sum(
        len(str(row["question"]))
        + sum(
            len(str(source["content"]))
            for source in row["sources"]
        )
        for row in rows
    )
    _, base_url, key_source = resolve_classifier_auth()

    print("PHASE_B_RUNNER_PREFLIGHT=PASS")
    print(f"INPUT_CASES={len(rows)}")
    print(f"INPUT_SHA256={_sha256(inputs_path)}")
    print(f"TOTAL_QUESTION_CONTEXT_CHARS={char_count}")
    print(f"MODEL={contract['classifier']['model']}")
    print(f"AUTH_KEY_SOURCE={key_source}")
    print(f"BASE_URL={base_url}")
    print("ENABLE_THINKING=NO")
    print("TEMPERATURE=0.0")
    print(
        "HARD_COST_CAP_CNY="
        f"{contract['pricing_snapshot']['hard_cost_cap_cny']}"
    )
    print("PROVIDER_CALLS=0")
    print("DEV_ARTIFACT_OPENED=NO")
    print("NEXT_ACTION=EXPLICITLY_AUTHORIZE_PHASE_B_PAID_RUN")


def _print_evaluation(summary: PhaseBEvaluation) -> None:
    print("REFUSAL_PHASE_B=COMPLETE")
    print(f"TOTAL_PREDICTIONS={summary.total_predictions}")
    print(f"GATED_PREDICTIONS={summary.gated_predictions}")
    print(f"AMBIGUOUS_PREDICTIONS={summary.ambiguous_predictions}")
    print(f"ACCURACY={summary.accuracy:.6f}")
    print(
        f"BALANCED_ACCURACY={summary.balanced_accuracy:.6f}"
    )
    print(
        f"SUFFICIENT_RECALL={summary.sufficient_recall:.6f}"
    )
    print(
        f"INSUFFICIENT_RECALL={summary.insufficient_recall:.6f}"
    )
    print(
        "SUFFICIENT_TO_INSUFFICIENT="
        f"{summary.sufficient_to_insufficient}"
    )
    print(
        "INSUFFICIENT_TO_SUFFICIENT="
        f"{summary.insufficient_to_sufficient}"
    )
    print(
        f"AMBIGUOUS_SUFFICIENT={summary.ambiguous_sufficient}"
    )
    print(
        f"AMBIGUOUS_INSUFFICIENT={summary.ambiguous_insufficient}"
    )
    print(f"PROVIDER_CALLS={summary.provider_calls}")
    print(f"TOTAL_TOKENS={summary.total_tokens}")
    print(
        f"ESTIMATED_COST_CNY={summary.estimated_cost_cny:.6f}"
    )
    print(
        f"LATENCY_P50_MS={summary.latency_p50_ms:.3f}"
    )
    print(
        f"LATENCY_P95_MS={summary.latency_p95_ms:.3f}"
    )
    print(
        "GATE_BALANCED_ACCURACY="
        f"{'PASS' if summary.gate_balanced_accuracy_pass else 'FAIL'}"
    )
    print(
        "GATE_SUFFICIENT_RECALL="
        f"{'PASS' if summary.gate_sufficient_recall_pass else 'FAIL'}"
    )
    print(
        "GATE_INSUFFICIENT_RECALL="
        f"{'PASS' if summary.gate_insufficient_recall_pass else 'FAIL'}"
    )
    print(
        "GATE_OVER_REFUSAL="
        f"{'PASS' if summary.gate_over_refusal_pass else 'FAIL'}"
    )
    print(f"DECISION={summary.decision}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or evaluate refusal evidence-sufficiency Phase B."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-paid",
        action="store_true",
        help="Execute the 54-call paid classifier run.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the completed checkpoint against frozen targets.",
    )
    parser.add_argument(
        "--contract-path",
        default=str(DEFAULT_CONTRACT_PATH),
    )
    parser.add_argument(
        "--inputs-path",
        default=str(DEFAULT_INPUTS_PATH),
    )
    parser.add_argument(
        "--targets-path",
        default=str(DEFAULT_TARGETS_PATH),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT_PATH),
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
    )
    args = parser.parse_args()

    if args.run_paid:
        run_paid_phase_b(
            contract_path=args.contract_path,
            inputs_path=args.inputs_path,
            checkpoint_path=args.checkpoint_path,
        )
        print("PHASE_B_PAID_RUN=COMPLETE")
        print("TARGETS_OPENED=NO")
        print("NEXT_ACTION=RUN_PHASE_B_EVALUATION")
        return

    if args.evaluate:
        summary = evaluate_checkpoint(
            contract_path=args.contract_path,
            checkpoint_path=args.checkpoint_path,
            targets_path=args.targets_path,
            summary_path=args.summary_path,
        )
        _print_evaluation(summary)
        return

    print_preflight(
        contract_path=args.contract_path,
        inputs_path=args.inputs_path,
    )


if __name__ == "__main__":
    main()
