"""Frozen TRAIN-only v4/v5 diagnostic; no serving integration or automatic retries."""

import argparse
import hashlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path

from experiments.evals import refusal_phase_b_runner as shared
from experiments.evals import refusal_structured_evidence as structured
from experiments.evals import refusal_v2_ai_proxy_runner as v2
from experiments.evals.refusal_development_policy import (
    CLASSIFIER_SYSTEM_PROMPT_V4_DEV,
    build_classifier_messages_v4_dev,
)
from experiments.evals.refusal_v5_preparation import canonical_sha256

REPORT_DIR = Path("experiments/evals/reports/refusal_evidence_sufficiency")
CONTRACT_PATH = REPORT_DIR / "v5_train_run_contract_v1_1.json"
CONTRACT_SHA256 = "ed66bfbbb841a199670e6922f29cdd30ed1e17ab5d9f3cfec6f633f504064032"
OUTPUT_DIR = Path("data/refusal_v5_train/paired_run_v1")
BUILDERS = {"v4": build_classifier_messages_v4_dev,
            "v5": structured.build_classifier_messages_v5_dev}


def load_probe(inputs_path=None, *, contract_path=None, contract_sha256=None,
               builders=None, prompts=None, validators=None):
    builders = BUILDERS if builders is None else builders
    contract = shared._read_json(contract_path or CONTRACT_PATH)
    if canonical_sha256(contract) != (contract_sha256 or CONTRACT_SHA256):
        raise RuntimeError("frozen paired contract changed")
    prompts = prompts or {"v4": CLASSIFIER_SYSTEM_PROMPT_V4_DEV,
                          "v5": structured.CLASSIFIER_SYSTEM_PROMPT_V5_DEV}
    if set(builders) != set(prompts) or set(builders) != set(contract["arms"]) or len(builders) != 2:
        raise RuntimeError("paired arm bindings mismatch")
    for arm, prompt in prompts.items():
        if hashlib.sha256(prompt.encode()).hexdigest() != contract["arms"][arm]["prompt_sha256"]:
            raise RuntimeError("frozen paired prompt changed")
    validators = validators or {"v5_validator_normalized_source_sha256": structured}
    for key, module in validators.items():
        source = Path(module.__file__).read_text(encoding="utf-8")
        if hashlib.sha256(source.encode()).hexdigest() != contract[key]:
            raise RuntimeError("frozen validator changed")
    frozen = contract["frozen_inputs"]
    selection = shared._read_json(REPORT_DIR / "v5_train_selection.json")
    if canonical_sha256(selection) != frozen["selection_canonical_sha256"]:
        raise RuntimeError("frozen TRAIN selection changed")
    rows = v2.load_classifier_inputs(
        inputs_path or frozen["inputs_path"], expected_sha256=frozen["inputs_sha256"],
        expected_cases=contract["cases"],
    )
    if [row["question_id"] for row in rows] != selection["selected_question_ids"]:
        raise RuntimeError("TRAIN question order changed")
    for index, row in enumerate(rows):
        for arm, builder in builders.items():
            messages = builder(shared._classifier_input_from_row(row))
            bound = sum(len(m["content"].encode()) for m in messages) + 512
            if bound != contract["arms"][arm]["request_input_byte_bounds"][index] or bound > 256_000:
                raise RuntimeError("frozen request byte bound changed")
            if row["question_id"] in "\n".join(m["content"] for m in messages):
                raise RuntimeError("question ID leaked into request")
    # Labels/annotation freeze are deliberately not read here or in the paid loop.
    return contract, rows


def schedule(rows, arms=("v4", "v5")):
    return [(row, arm) for index, row in enumerate(rows)
            for arm in (arms if index % 2 == 0 else tuple(reversed(arms)))]


def parse_result(arm, raw, row, finish_reason):
    if finish_reason != "stop":
        raise ValueError("response did not finish normally")
    classifier_input = shared._classifier_input_from_row(row)
    if arm == "v5":
        return structured.validate_evidence_response(raw, classifier_input)
    decision, reason, ids = shared._parse_classifier_json(
        raw, allowed_source_ids={s.source_id for s in classifier_input.sources},
    )
    return {"decision": decision, "reason": reason, "supporting_source_ids": list(ids)}


def raw_admission(arm, raw):
    """Descriptive status-only admission BEFORE quote checks; not a valid prediction."""
    try:
        value = json.loads(raw, object_pairs_hook=structured._unique_object)
        if arm == "v4":
            return value["decision"] if value["decision"] in {"SUFFICIENT", "INSUFFICIENT"} else None
        items, ambiguities = value["requirements"], value["unresolved_ambiguities"]
        if (not isinstance(items, list) or not items or not isinstance(ambiguities, list)
                or not all(isinstance(a, str) and a.strip() for a in ambiguities)
                or not all(isinstance(i, dict) and isinstance(i.get("status"), str)
                           and i["status"] in {"SUPPORTED", "MISSING", "CONFLICTING", "CONDITIONAL"}
                           for i in items)):
            return None
        return "SUFFICIENT" if not ambiguities and all(i["status"] == "SUPPORTED" for i in items) else "INSUFFICIENT"
    except (ValueError, TypeError, KeyError):
        return None


def cost(record, contract):
    pricing = contract["pricing_snapshot"]
    return (record["prompt_tokens"] * pricing["input_cny_per_1m_tokens"]
            + record["completion_tokens"] * pricing["output_cny_per_1m_tokens"]) / 1_000_000


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_state(output_dir, rows, contract, *, create=False, builders=None,
               result_parser=None, contract_sha256=None):
    builders = BUILDERS if builders is None else builders
    result_parser = result_parser or parse_result
    root = Path(output_dir)
    identity = {"run": contract["run"], "contract_sha256": contract_sha256 or CONTRACT_SHA256}
    identity_path = root / "run_identity.json"
    if identity_path.exists():
        if shared._read_json(identity_path) != identity:
            raise RuntimeError("checkpoint identity mismatch")
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError("unexpected checkpoint without identity")
    elif create:
        root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)
    events = []
    allowed = {"run_identity.json", "pending_request.json", "runner.lock", "summary.json", *builders}
    if root.exists() and any(p.name not in allowed for p in root.iterdir()):
        raise RuntimeError("unexpected paired checkpoint file")
    for arm in builders:
        directory = root / arm
        if directory.exists() and any(p.name != "attempts.jsonl" for p in directory.iterdir()):
            raise RuntimeError("unexpected arm checkpoint file")
        path = root / arm / "attempts.jsonl"
        arm_events = shared._read_jsonl(path) if path.exists() else []
        if any(event.get("arm") != arm for event in arm_events):
            raise RuntimeError("checkpoint arm mismatch")
        events.extend(arm_events)
    if any(type(e.get("call_index")) is not int for e in events):
        raise RuntimeError("invalid checkpoint call index")
    events.sort(key=lambda e: e["call_index"])
    ordered = schedule(rows, tuple(builders))
    next_pair, spent = 0, 0.0
    counts = {arm: {"calls": 0, "failures": 0} for arm in builders}
    for index, event in enumerate(events):
        if next_pair >= len(ordered):
            raise RuntimeError("unexpected call after completion")
        row, arm = ordered[next_pair]
        if (event["call_index"] != index or event.get("question_id") != row["question_id"]
                or event.get("arm") != arm or event.get("model") != contract["classifier"]["model"]):
            raise RuntimeError("checkpoint schedule/model mismatch")
        if event.get("usage_available") is not True:
            raise RuntimeError("unknown usage: stop and reconcile accounting; no resume")
        tokens = [event.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        bound = contract["arms"][arm]["request_input_byte_bounds"][next_pair // 2]
        if (not all(type(t) is int for t in tokens) or not 0 < tokens[0] <= bound
                or not 0 <= tokens[1] <= contract["arms"][arm]["max_output_tokens"]
                or tokens[2] != tokens[0] + tokens[1]):
            raise RuntimeError("invalid checkpoint usage")
        charged = event.get("estimated_cost_cny")
        latency = event.get("latency_ms")
        if (not isinstance(charged, (int, float)) or not math.isfinite(charged)
                or not math.isclose(charged, cost(event, contract), abs_tol=1e-12)
                or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            raise RuntimeError("invalid checkpoint cost/latency")
        raw = event.get("raw_response")
        if not isinstance(raw, str):
            raise RuntimeError("missing raw response")
        try:
            result = result_parser(arm, raw, row, event.get("finish_reason"))
        except (ValueError, RuntimeError, TypeError, KeyError):
            result = None
        if event.get("result") != result or event.get("raw_model_admission") != raw_admission(arm, raw):
            raise RuntimeError("checkpoint raw/derived result mismatch")
        counts[arm]["calls"] += 1
        if result is None:
            counts[arm]["failures"] += 1
            if contract["controls"].get("failed_pair_is_terminal") and event.get("error_stage") != "schema":
                raise RuntimeError("non-schema failure: audit required; no resume")
        if result is not None or contract["controls"].get("failed_pair_is_terminal"):
            next_pair += 1
        if (counts[arm]["calls"] > contract["arms"][arm]["max_calls"]
                or counts[arm]["failures"] > contract["controls"]["max_failed_attempts_per_arm"]):
            raise RuntimeError("checkpoint arm budget exceeded")
        spent += charged
    if len(events) > contract["max_calls_total"] or spent > contract["pricing_snapshot"]["hard_cost_cap_cny"]:
        raise RuntimeError("checkpoint total budget exceeded")
    pending_path = root / "pending_request.json"
    if pending_path.exists():
        pending = shared._read_json(pending_path)
        matching = [e for e in events if e["call_index"] == pending.get("call_index")]
        if len(matching) != 1 or pending != {k: matching[0][k] for k in ("call_index", "arm", "question_id")}:
            raise RuntimeError("uncheckpointed request: unknown charge; audit required")
    return events, next_pair, counts, spent


def attempt(row, arm, index, contract, client, *, builders=None, result_parser=None):
    builders = BUILDERS if builders is None else builders
    result_parser = result_parser or parse_result
    record = {"call_index": index, "question_id": row["question_id"], "arm": arm,
              "model": contract["classifier"]["model"], "request_id": None,
              "raw_response": None, "finish_reason": None, "result": None,
              "raw_model_admission": None, "usage_available": False,
              "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
              "estimated_cost_cny": None, "error_type": None, "error_stage": "provider"}
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=record["model"], temperature=0, response_format={"type": "json_object"},
            extra_body={"enable_thinking": False}, max_tokens=contract["arms"][arm]["max_output_tokens"],
            messages=builders[arm](shared._classifier_input_from_row(row)),
        )
        record["error_stage"] = "schema"
        record["request_id"] = getattr(response, "id", None)
        usage = getattr(response, "usage", None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            record[key] = getattr(usage, key, None)
        tokens = [record[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        record["usage_available"] = (all(type(t) is int and t >= 0 for t in tokens)
                                     and tokens[0] > 0 and tokens[2] == tokens[0] + tokens[1])
        if record["usage_available"]:
            record["estimated_cost_cny"] = cost(record, contract)
        record["raw_response"] = ""  # A malformed choice is a charged schema failure, not unknown usage.
        record["raw_response"] = response.choices[0].message.content or ""
        record["finish_reason"] = response.choices[0].finish_reason
        record["raw_model_admission"] = raw_admission(arm, record["raw_response"])
        record["result"] = result_parser(arm, record["raw_response"], row, record["finish_reason"])
        record["error_stage"] = None
    except Exception as error:
        # Never persist exception text: SDK/provider errors can contain credentials.
        record["error_type"] = type(error).__name__
    if record["error_stage"] != "provider" and not record["usage_available"]:
        record["error_stage"] = "usage"
    record["latency_ms"] = (time.perf_counter() - start) * 1000
    return record


def run_paid(*, inputs_path=None, output_dir=OUTPUT_DIR, client=None, resume_after_audit=False,
             probe_loader=None, builders=None, result_parser=None, contract_sha256=None):
    builders = BUILDERS if builders is None else builders
    contract, rows = (probe_loader or load_probe)(inputs_path)
    read_state = partial(load_state, builders=builders, result_parser=result_parser,
                         contract_sha256=contract_sha256)
    root = Path(output_dir)
    read_state(root, rows, contract, create=True)
    # ponytail: one local exclusive lock, not a multi-host experiment scheduler.
    lock = root / "runner.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        events, next_pair, counts, spent = read_state(root, rows, contract)
        if events and events[-1]["result"] is None and not resume_after_audit:
            raise RuntimeError("prior failure: explicit resume-after-audit required")
        pending = root / "pending_request.json"
        if pending.exists():
            pending.unlink()  # load_state proved this attempt is already durably recorded.
        provider = client
        ordered = schedule(rows, tuple(builders))
        while next_pair < len(ordered):
            row, arm = ordered[next_pair]
            limits = contract["arms"][arm]
            if (len(events) >= contract["max_calls_total"] or counts[arm]["calls"] >= limits["max_calls"]
                    or counts[arm]["failures"] >= contract["controls"]["max_failed_attempts_per_arm"]):
                raise RuntimeError("attempt/failure budget exhausted")
            reservation = cost({"prompt_tokens": limits["request_input_byte_bounds"][next_pair // 2],
                                "completion_tokens": limits["max_output_tokens"]}, contract)
            if spent + reservation > contract["pricing_snapshot"]["hard_cost_cap_cny"]:
                raise RuntimeError("insufficient budget for next request")
            if provider is None:
                provider = shared.get_classifier_client()  # Existing Singapore auth; SDK retries disabled.
            write_json(pending, {"call_index": len(events), "arm": arm, "question_id": row["question_id"]})
            record = attempt(row, arm, len(events), contract, provider,
                             builders=builders, result_parser=result_parser)
            path = root / arm / "attempts.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            pending.unlink()
            events, next_pair, counts, spent = read_state(root, rows, contract)
            print(f"CHECKPOINTED={next_pair}/{len(ordered)} CALLS={len(events)} COST_CNY={spent:.6f}", flush=True)
            if record["result"] is None:
                raise RuntimeError("response failure checkpointed; stop and audit")
        return events
    finally:
        lock.unlink()


def evaluate(*, inputs_path=None, output_dir=OUTPUT_DIR, probe_loader=None,
             builders=None, result_parser=None, contract_sha256=None):
    builders = BUILDERS if builders is None else builders
    candidate = tuple(builders)[1]
    contract, rows = (probe_loader or load_probe)(inputs_path)
    events, completed, _, spent = load_state(
        output_dir, rows, contract, builders=builders, result_parser=result_parser,
        contract_sha256=contract_sha256)
    if (Path(output_dir) / "runner.lock").exists():
        raise RuntimeError("active/stale runner lock: audit before evaluation")
    if completed != contract.get("required_outcomes_total", contract.get("required_predictions_total")):
        if contract["controls"].get("failed_pair_is_terminal"):
            raise RuntimeError("all 40 attempted outcomes required before opening targets")
        raise RuntimeError("all 40 valid predictions required before opening targets")
    frozen = contract["frozen_inputs"]
    freeze = shared._read_json(REPORT_DIR / "v5_train_annotation_freeze.json")
    if canonical_sha256(freeze) != frozen["annotation_freeze_canonical_sha256"]:
        raise RuntimeError("annotation freeze changed")
    if shared._sha256(frozen["targets_path"]) != frozen["targets_sha256"]:
        raise RuntimeError("targets SHA changed")
    targets = shared._read_jsonl(frozen["targets_path"])
    labels = {t["question_id"]: t["target_class"] for t in targets}
    if (len(targets) != len(rows) or set(labels) != {r["question_id"] for r in rows}
            or list(labels.values()).count("SUFFICIENT") != 14
            or list(labels.values()).count("INSUFFICIENT") != 6):
        raise RuntimeError("frozen target population changed")
    successes = {(e["question_id"], e["arm"]): e for e in events if e["result"] is not None}
    cases, arms = [], {}
    for row in rows:
        qid = row["question_id"]
        predictions = {arm: successes.get((qid, arm), {}).get("result") for arm in builders}
        correct = {arm: predictions[arm]["decision"] == labels[qid] for arm in builders if predictions[arm]}
        transition = { (True, True): "both_agree", (False, True): f"{candidate}_gains_agreement",
                       (True, False): f"{candidate}_loses_agreement", (False, False): "both_disagree"}[
                           correct["v4"], correct[candidate]] if len(correct) == 2 else "missing_valid_prediction"
        cases.append({"question_id": qid, "ai_target": labels[qid], "predictions": predictions,
                      "transition_against_ai_proxy": transition})
    common = [c for c in cases if all(c["predictions"].values())]
    denominators = {label: sum(c["ai_target"] == label for c in common)
                    for label in ("SUFFICIENT", "INSUFFICIENT")}
    for arm in builders:
        attempts = [e for e in events if e["arm"] == arm]
        failures = [e for e in attempts if e["result"] is None]
        matrix = {f"{label.lower()}_as_{decision.lower()}": sum(
            c["ai_target"] == label and c["predictions"][arm]["decision"] == decision for c in common)
                  for label in ("SUFFICIENT", "INSUFFICIENT") for decision in ("SUFFICIENT", "INSUFFICIENT")}
        agreements = sum(c["predictions"][arm] is not None
                         and c["predictions"][arm]["decision"] == c["ai_target"] for c in cases)
        arms[arm] = {"case_denominator": len(rows), "valid_predictions": sum(e["result"] is not None for e in attempts),
                     "failure_case_count": len({e["question_id"] for e in failures}),
                     "failed_attempts": len(failures), "provider_calls": len(attempts),
                     "format_failed_attempts": sum(e["error_stage"] == "schema" for e in attempts),
                     "provider_failed_attempts": sum(e["error_stage"] == "provider" for e in attempts),
                     "confusion_on_common_valid_pairs": matrix,
                     "common_valid_class_denominators": denominators,
                     "sufficient_recall_against_ai_proxy": matrix["sufficient_as_sufficient"] / denominators["SUFFICIENT"] if denominators["SUFFICIENT"] else None,
                     "insufficient_recall_against_ai_proxy": matrix["insufficient_as_insufficient"] / denominators["INSUFFICIENT"] if denominators["INSUFFICIENT"] else None,
                     "agreement_with_ai_proxy": (matrix["sufficient_as_sufficient"] + matrix["insufficient_as_insufficient"]) / len(common) if common else None,
                     "valid_and_agree_all_case_count": agreements,
                     "valid_and_agree_all_case_rate": agreements / len(rows),
                     "estimated_cost_cny": sum(e["estimated_cost_cny"] for e in attempts),
                     "latency_p50_ms": shared._percentile([e["latency_ms"] for e in attempts], 0.5),
                     "latency_p95_ms": shared._percentile([e["latency_ms"] for e in attempts], 0.95)}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            arms[arm][key] = sum(e[key] for e in attempts)
    summary = {"status": "COMPLETE_PAIRED_AI_PROXY_DIAGNOSTIC_NO_PROMOTION",
               "contract_sha256": contract_sha256 or CONTRACT_SHA256, "label_provenance": contract["label_provenance"],
               "case_denominator": len(rows), "common_valid_pairs": len(common), "provider_calls": len(events),
               "quality_metrics_population": "COMMON_VALID_PAIRS_CONDITIONAL_NOT_ALL_CASE_ACCURACY",
               "estimated_cost_cny": spent, "arms": arms, "cases": cases,
               "paired_transitions_against_ai_proxy": {
                   t: sum(c["transition_against_ai_proxy"] == t for c in cases)
                   for t in ("both_agree", "both_disagree", f"{candidate}_gains_agreement", f"{candidate}_loses_agreement", "missing_valid_prediction")},
               f"{candidate}_admission_attempts": [{"question_id": e["question_id"], "call_index": e["call_index"],
                                          "raw_status_decision": e["raw_model_admission"],
                                          "final_valid_decision": e["result"]["decision"] if e["result"] else None,
                                          "validation_failed": e["result"] is None}
                                         for e in events if e["arm"] == candidate],
               "correlated_families": [[c for c in cases if c["question_id"] in family]
                                       for family in (("TRAIN_Q029", "TRAIN_Q490"), ("TRAIN_Q348", "TRAIN_Q394"))],
               "dev_opened": False, "promotion_allowed": False, "historical_fail_changed": False}
    write_json(Path(output_dir) / "summary.json", summary)
    return summary


def main(*, probe_loader=None, builders=None, result_parser=None, contract_sha256=None,
         output_dir=OUTPUT_DIR):
    builders = BUILDERS if builders is None else builders
    options = {"probe_loader": probe_loader, "builders": builders,
               "result_parser": result_parser, "contract_sha256": contract_sha256}
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--resume-after-audit", action="store_true")
    parser.add_argument("--inputs-path")
    parser.add_argument("--output-dir", default=str(output_dir))
    parser.add_argument("--env-file", help="Local auth only; never copied into artifacts")
    args = parser.parse_args()
    if args.resume_after_audit and not args.run_paid:
        parser.error("resume-after-audit requires run-paid and separate authorization")
    if args.env_file:
        if not Path(args.env_file).is_file():
            parser.error("explicit env file does not exist")
        shared.load_dotenv(args.env_file, override=True)
    if args.run_paid:
        run_paid(inputs_path=args.inputs_path, output_dir=args.output_dir,
                 resume_after_audit=args.resume_after_audit, **options)
        print("PAIRED_TRAIN_COMPLETE TARGETS_OPENED=NO DEV_OPENED=NO")
    elif args.evaluate:
        print(json.dumps(evaluate(inputs_path=args.inputs_path, output_dir=args.output_dir, **options), indent=2))
    else:
        contract, rows = (probe_loader or load_probe)(args.inputs_path)
        events, completed, _, spent = load_state(
            args.output_dir, rows, contract, builders=builders, result_parser=result_parser,
            contract_sha256=contract_sha256)
        if (Path(args.output_dir) / "runner.lock").exists():
            raise RuntimeError("active/stale runner lock: audit before execution")
        print(f"PAIRED_PREFLIGHT=PASS CASES={len(rows)} CONTRACT_SHA256={contract_sha256 or CONTRACT_SHA256}")
        print(f"CHECKPOINTED={completed}/40 PRIOR_CALLS={len(events)} COST_CNY={spent:.6f}")
        print("PROVIDER_CALLS_THIS_COMMAND=0 TARGETS_OPENED=NO DEV_OPENED=NO AUTHORIZATION_REQUIRED=YES")


if __name__ == "__main__":
    main()
