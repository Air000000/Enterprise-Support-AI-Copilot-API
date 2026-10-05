"""One post-hoc TRAIN probe, reusing the frozen runner's checkpoint mechanics."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_v3_ai_proxy_runner as reuse
from experiments.evals.refusal_development_policy import (
    CLASSIFIER_SYSTEM_PROMPT_V4_DEV,
    build_classifier_messages_v4_dev,
)

CONTRACT_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/v4_development_run_contract.json"
)
CONTRACT_SHA256 = "d7bd916c48fdafb05f48d4684666b574e43183cdc2a5c68df9f201fbfecc8e38"
OUTPUT_DIR = Path("data/refusal_v4_development/run_v1")


def load_probe(inputs_path=None):
    contract = reuse.shared._read_json(CONTRACT_PATH)
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != CONTRACT_SHA256:
        raise RuntimeError("frozen development contract changed")
    if hashlib.sha256(CLASSIFIER_SYSTEM_PROMPT_V4_DEV.encode()).hexdigest() != (
        contract["classifier"]["prompt_sha256"]
    ):
        raise RuntimeError("frozen development prompt changed")
    frozen = contract["frozen_inputs"]
    rows = reuse.v2.load_classifier_inputs(
        inputs_path or frozen["classifier_inputs_path"],
        expected_sha256=frozen["classifier_inputs_sha256"], expected_cases=50,
    )
    return contract, rows


def probe_state(output_dir, rows, contract, *, create=False):
    output_dir = Path(output_dir)
    identity_path = output_dir / "run_identity.json"
    identity = {"run": contract["run"], "contract_sha256": CONTRACT_SHA256}
    if identity_path.exists():
        if reuse.shared._read_json(identity_path) != identity:
            raise RuntimeError("checkpoint belongs to a different development run")
    elif any((output_dir / name).exists() for name in (
        "predictions.jsonl", "failed_attempts.jsonl", "summary.json"
    )):
        raise RuntimeError("unexpected checkpoint without run identity")
    state = reuse.load_state(output_dir, rows, contract)
    if create and not identity_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        with identity_path.open("x", encoding="utf-8") as file:
            file.write(json.dumps(identity, indent=2) + "\n")
    return state


def run_paid(*, inputs_path=None, output_dir=OUTPUT_DIR, client=None):
    contract, rows = load_probe(inputs_path)
    probe_state(output_dir, rows, contract, create=True)
    return reuse.run_checkpointed_probe(
        rows=rows, contract=contract, output_dir=output_dir, client=client,
        messages_builder=build_classifier_messages_v4_dev,
    )


def evaluate(*, inputs_path=None, output_dir=OUTPUT_DIR):
    contract, rows = load_probe(inputs_path)
    predictions, failures, spent = probe_state(output_dir, rows, contract)
    if len(predictions) != 50:
        raise RuntimeError("all 50 predictions required before opening AI review")
    frozen = contract["frozen_inputs"]
    if reuse.shared._sha256(frozen["review_path"]) != frozen["review_sha256"]:
        raise RuntimeError("frozen AI review SHA mismatch")
    reviews = reuse.shared._read_jsonl(frozen["review_path"])
    labels = {row["question_id"]: row["annotation"]["target_class"] for row in reviews}
    if len(reviews) != 50 or set(labels) != {row["question_id"] for row in rows}:
        raise RuntimeError("AI review identities changed")
    if (
        sum(label == "SUFFICIENT" for label in labels.values()) != 17
        or sum(label == "INSUFFICIENT" for label in labels.values()) != 31
        or {qid for qid, label in labels.items() if label == "QUESTIONABLE"}
        != set(frozen["questionable_descriptive_only"])
    ):
        raise RuntimeError("AI review population changed")
    matrix = {"sufficient_as_sufficient": 0, "sufficient_as_insufficient": 0,
              "insufficient_as_sufficient": 0, "insufficient_as_insufficient": 0}
    disagreements, questionable = [], []
    for prediction in predictions:
        label = labels[prediction.question_id]
        record = {"question_id": prediction.question_id, "ai_review": label,
                  "decision": prediction.decision, "reason": prediction.reason}
        if label == "QUESTIONABLE":
            questionable.append(record)
        else:
            matrix[f"{label.lower()}_as_{prediction.decision.lower()}"] += 1
            if label != prediction.decision:
                disagreements.append(record)
    sufficient_recall = matrix["sufficient_as_sufficient"] / 17
    insufficient_recall = matrix["insufficient_as_insufficient"] / 31
    latencies = [prediction.latency_ms for prediction in predictions]
    summary = {
        "status": "COMPLETE_POST_HOC_DEVELOPMENT_PROBE_NO_PROMOTION",
        "contract_sha256": CONTRACT_SHA256,
        "label_provenance": frozen["label_provenance"],
        "predictions": 50, "binary_cases": 48,
        "confusion_against_ai_review": matrix,
        "agreement_with_ai_review": (
            matrix["sufficient_as_sufficient"] + matrix["insufficient_as_insufficient"]
        ) / 48,
        "balanced_agreement_with_ai_review": (sufficient_recall + insufficient_recall) / 2,
        "sufficient_recall_against_ai_review": sufficient_recall,
        "insufficient_recall_against_ai_review": insufficient_recall,
        "disagreements": disagreements, "questionable_descriptive_only": questionable,
        "provider_calls": len(predictions) + len(failures),
        "failed_attempts": len(failures), "estimated_cost_cny": spent,
        "latency_p50_ms": reuse.shared._percentile(latencies, 0.5),
        "latency_p95_ms": reuse.shared._percentile(latencies, 0.95),
        "dev_opened": False, "promotion_allowed": False,
        "historical_fail_changed": False,
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        summary[key] = sum(getattr(p, key) for p in predictions) + sum(f[key] for f in failures)
    # ponytail: one descriptive summary, no acceptance gate or second judge framework.
    with (Path(output_dir) / "summary.json").open("x", encoding="utf-8") as file:
        file.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--inputs-path")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--env-file", help="Explicit local dotenv source; never copied to artifacts")
    args = parser.parse_args()
    if args.env_file:
        if not Path(args.env_file).is_file():
            parser.error("explicit env file does not exist")
        reuse.shared.load_dotenv(args.env_file, override=True)
    if args.run_paid:
        run_paid(inputs_path=args.inputs_path, output_dir=args.output_dir)
        print("V4_DEVELOPMENT_PROBE=COMPLETE TARGETS_OPENED=NO")
    elif args.evaluate:
        print(json.dumps(evaluate(inputs_path=args.inputs_path, output_dir=args.output_dir), indent=2))
    else:
        contract, rows = load_probe(args.inputs_path)
        predictions, failures, spent = probe_state(args.output_dir, rows, contract)
        _, base_url, key_source = reuse.shared.resolve_classifier_auth()
        print("V4_DEVELOPMENT_PREFLIGHT=PASS")
        print(f"CASES={len(rows)} CONTRACT_SHA256={CONTRACT_SHA256}")
        print(f"AUTH_KEY_SOURCE={key_source} BASE_URL={base_url}")
        print(f"CHECKPOINTED={len(predictions)} PRIOR_CALLS={len(predictions) + len(failures)} COST_CNY={spent:.6f}")
        print("PROVIDER_CALLS_THIS_COMMAND=0 REVIEW_OPENED=NO DEV_OPENED=NO")


if __name__ == "__main__":
    main()
