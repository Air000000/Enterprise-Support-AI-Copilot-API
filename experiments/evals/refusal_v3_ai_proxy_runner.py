from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from experiments.evals import refusal_phase_b_runner as shared
from experiments.evals import refusal_v2_ai_proxy_runner as v2
from experiments.evals.refusal_v2_ai_proxy import (
    CLASSIFIER_SYSTEM_PROMPT_V2,
    build_classifier_messages_v2,
)
from experiments.evals.refusal_v3_ai_proxy import (
    DEFAULT_INPUTS_PATH,
    DEFAULT_TARGETS_PATH,
)

DEFAULT_CONTRACT_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/v3_ai_proxy_run_contract.json"
)
DEFAULT_OUTPUT_DIR = Path("data/refusal_v3_ai_proxy/run_v1")
EXPECTED_CONTRACT_SHA256 = (
    "eaecb0641a381db9ab7c789a0af5d3f9b1ff8679c1cfb63ebc1d918c2e3dcf94"
)
EXPECTED_INPUTS_SHA256 = (
    "87f91ba55b78bfc16db2c018b092212e39cf9d171b52e62853a5e536b6fd344d"
)
EXPECTED_TARGETS_SHA256 = (
    "a1965b36786b96d956a63b2af6533d14c1182fb3b42155ab8cd15879b7823c83"
)
EXPECTED_CASES = 50


def validate_contract(contract) -> None:
    # Canonical JSON ignores platform line endings, not experimental settings.
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen v3 contract changed")
    prompt_sha = hashlib.sha256(CLASSIFIER_SYSTEM_PROMPT_V2.encode()).hexdigest()
    if prompt_sha != contract["classifier"]["prompt_sha256"]:
        raise RuntimeError("frozen v3 prompt implementation changed")


def load_classifier_inputs(path=DEFAULT_INPUTS_PATH):
    return v2.load_classifier_inputs(
        path, expected_sha256=EXPECTED_INPUTS_SHA256, expected_cases=EXPECTED_CASES
    )


def load_state(output_dir, rows, contract):
    output_dir = Path(output_dir)
    predictions = shared.load_predictions(output_dir / "predictions.jsonl")
    failures = shared.load_failed_attempts(output_dir / "failed_attempts.jsonl")
    ids = {row["question_id"] for row in rows}
    pricing = contract["pricing_snapshot"]
    for record in [asdict(prediction) for prediction in predictions] + failures:
        if (
            record["question_id"] not in ids
            or record["model"] != contract["classifier"]["model"]
        ):
            raise RuntimeError("checkpoint identity differs from frozen run")
        if record.get("usage_available", True) is not True:
            raise RuntimeError(
                "unknown provider usage requires accounting review before resume"
            )
        prompt = int(record["prompt_tokens"])
        completion = int(record["completion_tokens"])
        cost = float(record["estimated_cost_cny"])
        expected_cost = (
            prompt * pricing["input_cny_per_1m_tokens"]
            + completion * pricing["output_cny_per_1m_tokens"]
        ) / 1_000_000
        if (
            not 0 < prompt <= 256_000
            or not 0 <= completion <= contract["classifier"]["max_output_tokens"]
            or record["total_tokens"] != prompt + completion
        ):
            raise RuntimeError("checkpoint token accounting is invalid")
        if not math.isfinite(cost) or not math.isclose(
            cost, expected_cost, abs_tol=1e-12
        ):
            raise RuntimeError("checkpoint cost differs from frozen accounting rates")
    for prediction in predictions:
        shared._parse_classifier_json(
            json.dumps(
                {
                    "decision": prediction.decision,
                    "reason": prediction.reason,
                    "supporting_source_ids": prediction.supporting_source_ids,
                }
            ),
            allowed_source_ids={f"Source {rank}" for rank in range(1, 15)},
        )
    if len(failures) > contract["controls"]["failed_attempt_retry_budget"]:
        raise RuntimeError("failed-attempt retry budget exhausted")
    if len(predictions) + len(failures) > contract["classifier"]["max_calls"]:
        raise RuntimeError("maximum provider-call count exceeded")
    spent = sum(prediction.estimated_cost_cny for prediction in predictions) + sum(
        record["estimated_cost_cny"] for record in failures
    )
    if spent > pricing["hard_cost_cap_cny"]:
        raise RuntimeError("hard accounting cap exceeded")
    return predictions, failures, spent


def run_paid(
    *, inputs_path=DEFAULT_INPUTS_PATH, output_dir=DEFAULT_OUTPUT_DIR, client=None
):
    contract = shared._read_json(DEFAULT_CONTRACT_PATH)
    validate_contract(contract)
    rows = load_classifier_inputs(inputs_path)
    return run_checkpointed_probe(
        rows=rows, contract=contract, output_dir=output_dir, client=client,
        messages_builder=build_classifier_messages_v2,
    )


def run_checkpointed_probe(*, rows, contract, output_dir, client, messages_builder):
    """Reuse accounting/checkpoint mechanics; callers validate their own contract."""
    predictions, failures, spent = load_state(output_dir, rows, contract)
    completed = {prediction.question_id for prediction in predictions}
    calls = len(predictions) + len(failures)
    classifier = contract["classifier"]
    pricing = contract["pricing_snapshot"]
    provider = client
    for row in rows:
        if row["question_id"] in completed:
            continue
        if calls >= classifier["max_calls"]:
            raise RuntimeError("maximum provider-call count reached")
        messages = messages_builder(shared._classifier_input_from_row(row))
        # ponytail: UTF-8 byte bound plus framing margin; no tokenizer dependency.
        input_bound = (
            sum(len(message["content"].encode("utf-8")) for message in messages) + 512
        )
        if input_bound > 256_000:
            raise RuntimeError("request exceeds frozen pricing context band")
        reservation = (
            input_bound * pricing["input_cny_per_1m_tokens"]
            + classifier["max_output_tokens"] * pricing["output_cny_per_1m_tokens"]
        ) / 1_000_000
        if spent + reservation > pricing["hard_cost_cap_cny"]:
            raise RuntimeError("insufficient accounting budget for next request")
        if provider is None:
            provider = shared.get_classifier_client()
        prediction = shared.classify_one(
            row,
            client=provider,
            model=classifier["model"],
            input_rate_cny_per_1m=pricing["input_cny_per_1m_tokens"],
            output_rate_cny_per_1m=pricing["output_cny_per_1m_tokens"],
            max_output_tokens=classifier["max_output_tokens"],
            messages_builder=messages_builder,
            failed_attempt_recorder=lambda record: shared._append_jsonl(
                Path(output_dir) / "failed_attempts.jsonl", record
            ),
        )
        shared._append_jsonl(Path(output_dir) / "predictions.jsonl", asdict(prediction))
        predictions.append(prediction)
        completed.add(prediction.question_id)
        calls += 1
        spent += prediction.estimated_cost_cny
        # ponytail: reread at most 50 rows; validate every persisted response.
        load_state(output_dir, rows, contract)
        print(
            f"CHECKPOINTED={len(predictions)}/{len(rows)} PROVIDER_CALLS={calls} COST_CNY={spent:.6f}",
            flush=True,
        )
    return tuple(predictions)


def evaluate_checkpoint(
    *,
    inputs_path=DEFAULT_INPUTS_PATH,
    targets_path=DEFAULT_TARGETS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    contract = shared._read_json(DEFAULT_CONTRACT_PATH)
    validate_contract(contract)
    rows = load_classifier_inputs(inputs_path)
    predictions, failures, spent = load_state(output_dir, rows, contract)
    if len(predictions) != EXPECTED_CASES:
        raise RuntimeError(
            "v3 evaluation requires all 50 predictions before targets are opened"
        )
    if shared._sha256(targets_path) != EXPECTED_TARGETS_SHA256:
        raise RuntimeError("v3 target SHA mismatch")
    targets = shared._read_jsonl(targets_path)
    if len(targets) != EXPECTED_CASES or any(
        set(row) != {"question_id", "target_class"} for row in targets
    ):
        raise RuntimeError("v3 target count or schema changed")
    if any(
        sum(row["target_class"] == label for row in targets) != 25
        for label in ("SUFFICIENT", "INSUFFICIENT")
    ):
        raise RuntimeError("v3 target balance changed")
    summary = v2.evaluate_predictions(
        predictions,
        targets,
        contract=contract,
        contract_validator=validate_contract,
        expected_cases=EXPECTED_CASES,
    )
    summary = replace(
        summary,
        provider_calls=len(predictions) + len(failures),
        estimated_cost_cny=spent,
        **{
            key: getattr(summary, key) + sum(record[key] for record in failures)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    )
    shared._write_json(Path(output_dir) / "summary.json", asdict(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen v3 blind AI-proxy classifier."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--inputs-path", default=str(DEFAULT_INPUTS_PATH))
    parser.add_argument("--targets-path", default=str(DEFAULT_TARGETS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.run_paid:
        run_paid(inputs_path=args.inputs_path, output_dir=args.output_dir)
        print("REFUSAL_V3_PAID_RUN=COMPLETE\nTARGETS_OPENED=NO")
    elif args.evaluate:
        print(
            json.dumps(
                asdict(
                    evaluate_checkpoint(
                        inputs_path=args.inputs_path,
                        targets_path=args.targets_path,
                        output_dir=args.output_dir,
                    )
                ),
                indent=2,
            )
        )
    else:
        contract = shared._read_json(DEFAULT_CONTRACT_PATH)
        validate_contract(contract)
        rows = load_classifier_inputs(args.inputs_path)
        predictions, failures, spent = load_state(args.output_dir, rows, contract)
        _, base_url, key_source = shared.resolve_classifier_auth()
        print("REFUSAL_V3_RUNNER_PREFLIGHT=PASS")
        print(
            f"INPUT_CASES={len(rows)} INPUT_SHA256={shared._sha256(args.inputs_path)}"
        )
        print(f"AUTH_KEY_SOURCE={key_source} BASE_URL={base_url}")
        print(
            f"MODEL={contract['classifier']['model']} MAX_CALLS=55 MAX_OUTPUT_TOKENS=512"
        )
        print(
            f"CHECKPOINTED={len(predictions)} PROVIDER_CALLS={len(predictions) + len(failures)} COST_CNY={spent:.6f}"
        )
        print("TARGETS_OPENED=NO\nDEV_ARTIFACT_OPENED=NO")


if __name__ == "__main__":
    main()
