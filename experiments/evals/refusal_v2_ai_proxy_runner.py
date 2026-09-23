from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from experiments.evals import refusal_phase_b_runner as shared
from experiments.evals.refusal_v2_ai_proxy import (
    CLASSIFIER_SYSTEM_PROMPT_V2,
    build_classifier_messages_v2,
)

DEFAULT_CONTRACT_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v2_ai_proxy_v2_1_run_contract.json"
)
DEFAULT_INPUTS_PATH = Path(
    "data/refusal_v2_ai_proxy/holdout_inputs.jsonl"
)
DEFAULT_TARGETS_PATH = Path(
    "data/refusal_v2_ai_proxy/holdout_targets.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/refusal_v2_ai_proxy/run_v2_1")
DEFAULT_CHECKPOINT_PATH = DEFAULT_OUTPUT_DIR / "predictions.jsonl"
DEFAULT_FAILED_ATTEMPTS_PATH = DEFAULT_OUTPUT_DIR / "failed_attempts.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "summary.json"

EXPECTED_INPUTS_SHA256 = (
    "0b5f9cd0c797930c7d7efefe5f9721435cb94de3bb98721d2a703d50b4cb269d"
)
EXPECTED_TARGETS_SHA256 = (
    "b4828f0b9870c53774c5b78722ddf2f3dc0d23a85aeea71281a4cda6f1603fd7"
)
EXPECTED_PROMPT_SHA256 = (
    "eff5768921067cc04a0686fff0ff50599f4bf295f3ad263e63a143f110fe3c20"
)
EXPECTED_CASES = 40
EXPECTED_PRIOR_CHECKPOINT_SHA256 = (
    "bbfd3ff87e332b516fae0e78afba588453cde113d9caefbd7bc6c17e26f1c6ca"
)
EXPECTED_FAILURES_SHA256 = (
    "d5ece92619caab15a1129b5c37144021d5166bc94f923aedf84cbd20f7c5dc1a"
)


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("run") != "refusal_v2_ai_proxy_holdout_v2_1":
        raise RuntimeError("unexpected v2 run identifier")
    if contract.get("status") != "PREREGISTERED_RESUME_NOT_RUN":
        raise RuntimeError("v2 contract status changed")

    population = contract.get("population")
    frozen = contract.get("frozen_inputs")
    classifier = contract.get("classifier")
    pricing = contract.get("pricing_snapshot")
    gate = contract.get("gate")
    controls = contract.get("controls")
    amendment = contract.get("amendment")
    if not all(
        isinstance(value, Mapping)
        for value in (
            population,
            frozen,
            classifier,
            pricing,
            gate,
            controls,
            amendment,
        )
    ):
        raise RuntimeError("v2 contract is malformed")

    expected_population = {
        "total_cases": 40,
        "sufficient": 20,
        "insufficient": 20,
        "questionable_excluded": 6,
        "selection_seed": "refusal-v2-ai-proxy-holdout-v1",
    }
    if any(
        population.get(key) != value
        for key, value in expected_population.items()
    ):
        raise RuntimeError("v2 population contract changed")

    if frozen.get("classifier_inputs_sha256") != EXPECTED_INPUTS_SHA256:
        raise RuntimeError("v2 input SHA contract changed")
    if frozen.get("targets_sha256") != EXPECTED_TARGETS_SHA256:
        raise RuntimeError("v2 target SHA contract changed")
    if frozen.get("context_policy") != "flat_rerank_top14_v1":
        raise RuntimeError("v2 context policy changed")
    if frozen.get("top_k") != 14:
        raise RuntimeError("v2 context TopK changed")

    expected_classifier = {
        "model": "qwen3.5-plus-2026-04-20",
        "temperature": 0,
        "enable_thinking": False,
        "response_format": "json_object",
        "max_output_tokens": 512,
        "max_calls": 41,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
    }
    if any(
        classifier.get(key) != value
        for key, value in expected_classifier.items()
    ):
        raise RuntimeError("v2 classifier contract changed")
    actual_prompt_sha = hashlib.sha256(
        CLASSIFIER_SYSTEM_PROMPT_V2.encode()
    ).hexdigest()
    if actual_prompt_sha != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("v2 prompt implementation changed")

    expected_pricing = {
        "input_cny_per_1m_tokens": 2.936,
        "output_cny_per_1m_tokens": 17.614,
        "hard_cost_cap_cny": 1.5,
    }
    if any(
        pricing.get(key) != value
        for key, value in expected_pricing.items()
    ):
        raise RuntimeError("v2 pricing contract changed")

    expected_gate = {
        "balanced_accuracy_min": 0.8,
        "sufficient_recall_min": 0.85,
        "insufficient_recall_min": 0.7,
        "sufficient_to_insufficient_max": 3,
        "pass_decision": "ADMIT_TO_INDEPENDENT_HUMAN_CONFIRMATION",
        "fail_decision": "REJECT_EVIDENCE_SUFFICIENCY_V2_AI_PROXY",
    }
    if any(
        gate.get(key) != value for key, value in expected_gate.items()
    ):
        raise RuntimeError("v2 gate contract changed")

    if controls.get("dev_artifact_opened") is not False:
        raise RuntimeError("DEV must remain closed")
    for key in (
        "retrieval_calls",
        "rerank_calls",
        "generation_calls",
        "judge_calls",
    ):
        if controls.get(key) != 0:
            raise RuntimeError(f"v2 forbids {key}")
    if controls.get("target_file_must_not_be_loaded_by_paid_classifier_loop") is not True:
        raise RuntimeError("v2 target isolation changed")
    if controls.get("all_provider_attempts_count_toward_limits") is not True:
        raise RuntimeError("v2 provider-attempt accounting changed")
    if controls.get("no_prompt_tuning_after_unblinding") is not True:
        raise RuntimeError("v2 unblinding control changed")
    if controls.get("pass_does_not_admit_phase_c") is not True:
        raise RuntimeError("v2 must not admit Phase C")
    expected_amendment = {
        "supersedes_run": "refusal_v2_ai_proxy_holdout_v1",
        "reason": "provider JSON truncated at the 256-token output limit",
        "prior_successful_predictions": 38,
        "prior_failed_attempts": 1,
        "prior_provider_attempts": 39,
        "additional_calls_max": 2,
        "carry_forward_successful_predictions": True,
        "prompt_changed": False,
        "model_changed": False,
        "gate_changed": False,
    }
    if any(
        amendment.get(key) != value
        for key, value in expected_amendment.items()
    ):
        raise RuntimeError("v2.1 amendment contract changed")


def validate_resume_state(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    failed_attempts_path: str | Path = DEFAULT_FAILED_ATTEMPTS_PATH,
) -> tuple[list[shared.PhaseBPrediction], list[dict[str, Any]]]:
    checkpoint = Path(checkpoint_path)
    failures_path = Path(failed_attempts_path)
    predictions = shared.load_predictions(checkpoint)
    failures = shared.load_failed_attempts(failures_path)
    if shared._sha256(failures_path) != EXPECTED_FAILURES_SHA256:
        raise RuntimeError("v2.1 prior failure artifact changed")
    if len(failures) != 1:
        raise RuntimeError("v2.1 requires exactly one prior failed attempt")
    failure = failures[0]
    if not (
        failure.get("question_id") == "TRAIN_Q584"
        and failure.get("stage") == "schema"
        and failure.get("error_message") == "classifier returned invalid JSON"
        and int(failure.get("completion_tokens", 0)) == 256
    ):
        raise RuntimeError("v2.1 prior failure identity changed")
    if len(predictions) == 38:
        if shared._sha256(checkpoint) != EXPECTED_PRIOR_CHECKPOINT_SHA256:
            raise RuntimeError("v2.1 prior checkpoint artifact changed")
    elif len(predictions) == EXPECTED_CASES:
        with checkpoint.open("rb") as handle:
            prefix = b"".join(handle.readline() for _ in range(38))
        if hashlib.sha256(prefix).hexdigest() != EXPECTED_PRIOR_CHECKPOINT_SHA256:
            raise RuntimeError("v2.1 carried-forward checkpoint rows changed")
    else:
        raise RuntimeError("v2.1 checkpoint must contain 38 or 40 predictions")
    return predictions, failures


def load_classifier_inputs(
    path: str | Path = DEFAULT_INPUTS_PATH,
) -> list[dict[str, Any]]:
    source = Path(path)
    if shared._sha256(source) != EXPECTED_INPUTS_SHA256:
        raise RuntimeError("v2 classifier input SHA mismatch")
    rows = shared._read_jsonl(source)
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError("v2 requires exactly 40 classifier inputs")

    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id", ""))
        if not question_id.startswith("TRAIN_") or question_id in seen:
            raise RuntimeError("invalid or duplicate v2 question_id")
        seen.add(question_id)
        if set(row) != {"question_id", "question", "sources"}:
            raise RuntimeError("v2 classifier input contains extra fields")
        sources = row["sources"]
        if not isinstance(sources, list) or len(sources) != 14:
            raise RuntimeError("v2 classifier input must contain Top14")
        for rank, source_item in enumerate(sources, start=1):
            if not isinstance(source_item, Mapping):
                raise RuntimeError("v2 source must be an object")
            if set(source_item) != {"source_id", "content"}:
                raise RuntimeError("v2 source contains extra metadata")
            if source_item["source_id"] != f"Source {rank}":
                raise RuntimeError("v2 source order changed")
    return rows


def load_targets(
    path: str | Path = DEFAULT_TARGETS_PATH,
) -> list[dict[str, Any]]:
    source = Path(path)
    if shared._sha256(source) != EXPECTED_TARGETS_SHA256:
        raise RuntimeError("v2 target SHA mismatch")
    rows = shared._read_jsonl(source)
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError("v2 target count changed")
    if any(set(row) != {"question_id", "target_class"} for row in rows):
        raise RuntimeError("v2 target schema changed")
    counts = {
        target_class: sum(
            row["target_class"] == target_class for row in rows
        )
        for target_class in ("SUFFICIENT", "INSUFFICIENT")
    }
    if counts != {"SUFFICIENT": 20, "INSUFFICIENT": 20}:
        raise RuntimeError("v2 target balance changed")
    return rows


def run_paid(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    inputs_path: str | Path = DEFAULT_INPUTS_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    failed_attempts_path: str | Path = DEFAULT_FAILED_ATTEMPTS_PATH,
    client: Any | None = None,
) -> tuple[shared.PhaseBPrediction, ...]:
    validate_resume_state(checkpoint_path, failed_attempts_path)
    return shared.run_paid_phase_b(
        contract_path=contract_path,
        inputs_path=inputs_path,
        checkpoint_path=checkpoint_path,
        failed_attempts_path=failed_attempts_path,
        client=client,
        contract_validator=validate_contract,
        inputs_loader=load_classifier_inputs,
        messages_builder=build_classifier_messages_v2,
        expected_cases=EXPECTED_CASES,
    )


def evaluate_predictions(
    predictions: Sequence[shared.PhaseBPrediction],
    targets: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> shared.PhaseBEvaluation:
    validate_contract(contract)
    pred_by_id = {prediction.question_id: prediction for prediction in predictions}
    target_by_id = {str(target["question_id"]): target for target in targets}
    if len(pred_by_id) != len(predictions) or len(target_by_id) != len(targets):
        raise RuntimeError("duplicate v2 prediction or target ID")
    if set(pred_by_id) != set(target_by_id):
        raise RuntimeError("v2 prediction and target IDs differ")

    tp = fn = tn = fp = 0
    for question_id, target in target_by_id.items():
        decision = pred_by_id[question_id].decision
        target_class = str(target["target_class"])
        if target_class == "SUFFICIENT":
            if decision == "SUFFICIENT":
                tp += 1
            else:
                fn += 1
        elif target_class == "INSUFFICIENT":
            if decision == "INSUFFICIENT":
                tn += 1
            else:
                fp += 1
        else:
            raise RuntimeError("invalid v2 target class")

    sufficient_recall = tp / (tp + fn)
    insufficient_recall = tn / (tn + fp)
    balanced_accuracy = (sufficient_recall + insufficient_recall) / 2.0
    accuracy = (tp + tn) / EXPECTED_CASES
    gate = contract["gate"]
    gate_balanced = balanced_accuracy >= float(gate["balanced_accuracy_min"])
    gate_sufficient = sufficient_recall >= float(gate["sufficient_recall_min"])
    gate_insufficient = insufficient_recall >= float(
        gate["insufficient_recall_min"]
    )
    gate_over_refusal = fn <= int(gate["sufficient_to_insufficient_max"])
    passed = all(
        (gate_balanced, gate_sufficient, gate_insufficient, gate_over_refusal)
    )
    latencies = [prediction.latency_ms for prediction in predictions]

    return shared.PhaseBEvaluation(
        total_predictions=len(predictions),
        gated_predictions=EXPECTED_CASES,
        ambiguous_predictions=0,
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
        ambiguous_sufficient=0,
        ambiguous_insufficient=0,
        provider_calls=len(predictions),
        prompt_tokens=sum(prediction.prompt_tokens for prediction in predictions),
        completion_tokens=sum(
            prediction.completion_tokens for prediction in predictions
        ),
        total_tokens=sum(prediction.total_tokens for prediction in predictions),
        estimated_cost_cny=sum(
            prediction.estimated_cost_cny for prediction in predictions
        ),
        latency_p50_ms=shared._percentile(latencies, 0.50),
        latency_p95_ms=shared._percentile(latencies, 0.95),
        gate_balanced_accuracy_pass=gate_balanced,
        gate_sufficient_recall_pass=gate_sufficient,
        gate_insufficient_recall_pass=gate_insufficient,
        gate_over_refusal_pass=gate_over_refusal,
        decision=str(
            gate["pass_decision"] if passed else gate["fail_decision"]
        ),
    )


def evaluate_checkpoint(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    targets_path: str | Path = DEFAULT_TARGETS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    failed_attempts_path: str | Path = DEFAULT_FAILED_ATTEMPTS_PATH,
) -> shared.PhaseBEvaluation:
    contract = shared._read_json(contract_path)
    predictions, failures = validate_resume_state(
        checkpoint_path,
        failed_attempts_path,
    )
    if len(predictions) != EXPECTED_CASES:
        raise RuntimeError("v2 evaluation requires all 40 predictions")
    summary = evaluate_predictions(
        predictions,
        load_targets(targets_path),
        contract=contract,
    )
    failed = failures[0]
    summary = replace(
        summary,
        provider_calls=len(predictions) + len(failures),
        prompt_tokens=summary.prompt_tokens + int(failed["prompt_tokens"]),
        completion_tokens=(
            summary.completion_tokens + int(failed["completion_tokens"])
        ),
        total_tokens=summary.total_tokens + int(failed["total_tokens"]),
        estimated_cost_cny=(
            summary.estimated_cost_cny
            + float(failed["estimated_cost_cny"])
        ),
    )
    shared._write_json(Path(summary_path), asdict(summary))
    return summary


def print_preflight(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    inputs_path: str | Path = DEFAULT_INPUTS_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    failed_attempts_path: str | Path = DEFAULT_FAILED_ATTEMPTS_PATH,
) -> None:
    contract = shared._read_json(contract_path)
    validate_contract(contract)
    rows = load_classifier_inputs(inputs_path)
    _, base_url, key_source = shared.resolve_classifier_auth()
    predictions, failures = validate_resume_state(
        checkpoint_path,
        failed_attempts_path,
    )
    char_count = sum(
        len(str(row["question"]))
        + sum(len(str(source["content"])) for source in row["sources"])
        for row in rows
    )

    print("REFUSAL_V2_1_AI_PROXY_RUNNER_PREFLIGHT=PASS")
    print(f"INPUT_CASES={len(rows)}")
    print(f"INPUT_SHA256={shared._sha256(inputs_path)}")
    print(f"TOTAL_QUESTION_CONTEXT_CHARS={char_count}")
    print(f"MODEL={contract['classifier']['model']}")
    print(f"MAX_OUTPUT_TOKENS={contract['classifier']['max_output_tokens']}")
    print(f"MAX_PROVIDER_CALLS={contract['classifier']['max_calls']}")
    print(f"PROMPT_SHA256={EXPECTED_PROMPT_SHA256}")
    print(f"AUTH_KEY_SOURCE={key_source}")
    print(f"BASE_URL={base_url}")
    print("ENABLE_THINKING=NO")
    print("TEMPERATURE=0.0")
    print(f"HARD_COST_CAP_CNY={contract['pricing_snapshot']['hard_cost_cap_cny']}")
    print(f"CHECKPOINTED_PREDICTIONS={len(predictions)}")
    print(f"FAILED_ATTEMPTS={len(failures)}")
    print(f"PROVIDER_CALLS={len(predictions) + len(failures)}")
    print("DEV_ARTIFACT_OPENED=NO")
    print("NEXT_ACTION=RUN_FROZEN_V2_1_TWO_CALL_RESUME")


def _print_evaluation(summary: shared.PhaseBEvaluation) -> None:
    print("REFUSAL_V2_1_AI_PROXY=COMPLETE")
    print(f"TOTAL_PREDICTIONS={summary.total_predictions}")
    print(f"ACCURACY={summary.accuracy:.6f}")
    print(f"BALANCED_ACCURACY={summary.balanced_accuracy:.6f}")
    print(f"SUFFICIENT_RECALL={summary.sufficient_recall:.6f}")
    print(f"INSUFFICIENT_RECALL={summary.insufficient_recall:.6f}")
    print(f"SUFFICIENT_TO_INSUFFICIENT={summary.sufficient_to_insufficient}")
    print(f"INSUFFICIENT_TO_SUFFICIENT={summary.insufficient_to_sufficient}")
    print(f"PROVIDER_CALLS={summary.provider_calls}")
    print(f"TOTAL_TOKENS={summary.total_tokens}")
    print(f"ESTIMATED_COST_CNY={summary.estimated_cost_cny:.6f}")
    print(f"LATENCY_P50_MS={summary.latency_p50_ms:.3f}")
    print(f"LATENCY_P95_MS={summary.latency_p95_ms:.3f}")
    print(f"DECISION={summary.decision}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or evaluate the refusal v2 AI-proxy holdout."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.run_paid:
        run_paid()
        print("REFUSAL_V2_1_AI_PROXY_PAID_RUN=COMPLETE")
        print("TARGETS_OPENED=NO")
        print("NEXT_ACTION=RUN_SEPARATE_V2_PROXY_EVALUATION")
    elif args.evaluate:
        _print_evaluation(evaluate_checkpoint())
    else:
        print_preflight()


if __name__ == "__main__":
    main()
