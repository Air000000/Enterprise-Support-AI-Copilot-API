"""Bounded paired execution of fictional controls; no natural TRAIN/DEV or serving."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from experiments.evals import refusal_citation_selection as citation
from experiments.evals import refusal_requirement_coverage as coverage
from experiments.evals import refusal_v5_2_controls as controls
from experiments.evals import refusal_v5_paired_runner as paired

REPORT = paired.REPORT_DIR
DATA = Path("data/refusal_v5_2_controls")
CONTRACT_PATH = REPORT / "v5_2_control_run_contract.json"
INPUTS = DATA / "inputs.jsonl"
ROOT = DATA / "run_v1"
BUILDERS = {"v5_1": citation.build_classifier_messages_v5_1_dev,
            "v5_2": coverage.build_classifier_messages_v5_2_dev}
SOURCE_FILES = ("refusal_v5_2_control_runner.py", "refusal_v5_paired_runner.py",
                "refusal_phase_b_runner.py", "refusal_citation_selection.py",
                "refusal_requirement_coverage.py", "refusal_structured_evidence.py",
                "refusal_development_policy.py", "refusal_evidence_sufficiency.py",
                "refusal_v5_2_controls.py", "refusal_v5_2_preparation.py", "refusal_v5_preparation.py")


def source_hashes():
    return {name: hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in SOURCE_FILES}


def prepare_contract():
    cases = controls.load_cases()
    manifest = paired.shared._read_json(controls.MANIFEST_PATH)
    if controls.prep.prior.canonical_sha256(cases) != manifest["cases_canonical_sha256"]:
        raise RuntimeError("frozen controls changed")
    rows = controls.classifier_rows(cases)
    preflight = controls.prep.paired_preflight(rows)
    if preflight != manifest["paired_preflight"]:
        raise RuntimeError("frozen protocol/preflight changed")
    if INPUTS.exists() or CONTRACT_PATH.exists():
        raise RuntimeError("refusing to overwrite execution inputs/contract")
    controls.prep.prior._write_jsonl(INPUTS, rows)
    contract = {
        "run": "refusal_v5_2_controls_paired_v1", "cases": 14, "max_calls_total": 28,
        "required_outcomes_total": 28, "inputs_sha256": paired.shared._sha256(INPUTS),
        "case_order": manifest["case_order"], "manifest_sha256": controls.prep.prior.canonical_sha256(manifest),
        "source_hashes": source_hashes(),
        "classifier": {"model": preflight["model"], "region": "Singapore", "temperature": 0,
                       "enable_thinking": False, "response_format": "json_object"},
        "arms": {arm: {"max_calls": 14, "max_output_tokens": 2048,
                       "prompt_sha256": data["prompt_sha256"], "request_sha256": data["request_sha256"],
                       "request_input_byte_bounds": data["input_byte_plus_framing_bounds"]}
                 for arm, data in preflight["arms"].items()},
        "pricing_snapshot": {"input_cny_per_1m_tokens": 3, "output_cny_per_1m_tokens": 18,
                             "hard_cost_cap_cny": 3, "accounting_cny_per_usd": 7.5,
                             "planning_upper_cny": preflight["planning_cost_usd"] * 7.5,
                             **preflight["pricing"]},
        "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 14,
                     "targets_outside_paid_loop": True, "raw_before_parse": True, "sdk_retries": 0,
                     "unknown_charge_blocks_resume": True, "successful_outcomes_never_retried": True,
                     "dev_opened": False, "promotion_allowed": False},
        "authorization": "User merged PR69 and confirmed continue in response to 14 fictional controls / <=28 Singapore qwen3.5-plus-2026-04-20 calls / CNY3 proposal; no natural TRAIN, recharge or extra calls",
        "caveat": "Fixed FX and UTF-8 byte/framing proxy, not certified tokens or invoice guarantee",
    }
    controls.prep.prior._write_json(CONTRACT_PATH, contract)
    return contract


def load_probe():
    contract = paired.shared._read_json(CONTRACT_PATH)
    manifest = paired.shared._read_json(controls.MANIFEST_PATH)  # Counts/hashes, no per-case targets.
    if (contract["source_hashes"] != source_hashes() or contract["cases"] != 14
            or contract["max_calls_total"] != 28 or contract["pricing_snapshot"]["hard_cost_cap_cny"] != 3
            or contract["classifier"]["model"] != "qwen3.5-plus-2026-04-20"
            or contract["manifest_sha256"] != controls.prep.prior.canonical_sha256(manifest)
            or contract["inputs_sha256"] != paired.shared._sha256(INPUTS)):
        raise RuntimeError("frozen execution binding changed")
    rows = paired.shared._read_jsonl(INPUTS)
    if len(rows) != 14 or [r["question_id"] for r in rows] != contract["case_order"]:
        raise RuntimeError("control population/order changed")
    for index, row in enumerate(rows):
        if set(row) != {"question_id", "question", "sources"}:
            raise RuntimeError("label/metadata leaked into classifier input")
        for arm, builder in BUILDERS.items():
            messages = builder(paired.shared._classifier_input_from_row(row))
            if controls.prep.prior.canonical_sha256(messages) != contract["arms"][arm]["request_sha256"][index]:
                raise RuntimeError("request fingerprint changed")
    return contract, rows


def parse_result(arm, raw, row, finish_reason):
    if finish_reason != "stop" or arm not in BUILDERS:
        raise ValueError("invalid arm or incomplete response")
    validator = citation.validate_citation_selection_response if arm == "v5_1" else coverage.validate_requirement_coverage_response
    return validator(raw, paired.shared._classifier_input_from_row(row))


def finish_record(record, row):
    record = dict(record)
    record["raw_model_admission"] = paired.raw_admission(record["arm"], record["raw_response"])
    try:
        record["result"] = parse_result(record["arm"], record["raw_response"], row, record["finish_reason"])
        record["error_stage"] = None if record["usage_available"] else "usage"
    except (ValueError, RuntimeError, TypeError, KeyError) as error:
        record["result"] = None
        record["error_type"] = record["error_type"] or type(error).__name__
    return record


def request_record(row, arm, index, contract, client):
    record = {"call_index": index, "question_id": row["question_id"], "arm": arm,
              "model": contract["classifier"]["model"], "request_id": None,
              "raw_response": None, "finish_reason": None, "usage_available": False,
              "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
              "estimated_cost_cny": None, "error_type": None, "error_stage": "provider"}
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(model=record["model"], temperature=0,
            extra_body={"enable_thinking": False}, response_format={"type": "json_object"},
            max_tokens=2048, messages=BUILDERS[arm](paired.shared._classifier_input_from_row(row)))
        record["error_stage"] = "schema"
        record["request_id"] = getattr(response, "id", None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            record[key] = getattr(getattr(response, "usage", None), key, None)
        tokens = [record[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        record["usage_available"] = all(type(t) is int and t >= 0 for t in tokens) and tokens[0] > 0 and sum(tokens[:2]) == tokens[2]
        if record["usage_available"]:
            record["estimated_cost_cny"] = paired.cost(record, contract)
        record["raw_response"] = ""
        record["raw_response"] = response.choices[0].message.content or ""
        record["finish_reason"] = response.choices[0].finish_reason
    except Exception as error:
        record["error_type"] = type(error).__name__  # Never save SDK exception messages/body.
    if record["error_stage"] != "provider" and not record["usage_available"]:
        record["error_stage"] = "usage"
    record["latency_ms"] = (time.perf_counter() - start) * 1000
    return record


def run_paid(*, root=ROOT, client=None, max_new_calls=28, probe_loader=load_probe):
    contract, rows = probe_loader()
    digest = controls.prep.prior.canonical_sha256(contract)
    root = Path(root)
    stage = root.parent / f"{root.name}_raw_response.json"
    identity = {"run": contract["run"], "contract_sha256": digest}
    identity_path = root / "run_identity.json"
    if identity_path.exists():
        if paired.shared._read_json(identity_path) != identity:
            raise RuntimeError("checkpoint identity mismatch")
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError("checkpoint without identity")
    else:
        root.mkdir(parents=True, exist_ok=True)
        paired.write_json(identity_path, identity)
    lock = root / "runner.lock"
    with lock.open("x", encoding="utf-8") as file:
        file.write(str(os.getpid()))
        file.flush()
        os.fsync(file.fileno())
    try:
        pending = root / "pending_request.json"
        ordered = paired.schedule(rows, tuple(BUILDERS))
        if stage.exists():
            raw = paired.shared._read_json(stage)
            index = raw["call_index"]
            if type(index) is not int or not 0 <= index < len(ordered):
                raise RuntimeError("invalid staged response index")
            row, arm = ordered[index]
            marker = {k: raw[k] for k in ("call_index", "arm", "question_id")}
            if (not pending.exists() or paired.shared._read_json(pending) != marker
                    or arm != raw["arm"] or row["question_id"] != raw["question_id"]
                    or raw["model"] != contract["classifier"]["model"]):
                raise RuntimeError("orphan/mismatched raw response: audit required")
            path = root / arm / "attempts.jsonl"
            prior = paired.shared._read_jsonl(path) if path.exists() else []
            record = finish_record(raw, row)
            matching = [e for e in prior if e["call_index"] == index]
            if not matching:
                paired.shared._append_jsonl(path, record)
            elif matching != [record]:
                raise RuntimeError("raw response/journal differ")
            stage.unlink()  # Raw response is now durably retained in the final journal.
        def read_state():
            return paired.load_state(root, rows, contract, builders=BUILDERS,
                                     result_parser=parse_result, contract_sha256=digest)
        events, completed, counts, spent = read_state()
        if pending.exists():
            pending.unlink()  # load_state proves the pending attempt is already journaled.
        new_calls = 0
        while completed < len(ordered) and new_calls < max_new_calls:
            row, arm = ordered[completed]
            limits = contract["arms"][arm]
            if len(events) >= contract["max_calls_total"] or counts[arm]["calls"] >= limits["max_calls"]:
                raise RuntimeError("call cap reached")
            reservation = paired.cost({"prompt_tokens": limits["request_input_byte_bounds"][completed // 2],
                                       "completion_tokens": limits["max_output_tokens"]}, contract)
            if spent + reservation > contract["pricing_snapshot"]["hard_cost_cap_cny"]:
                raise RuntimeError("spend cap reached")
            if client is None:
                client = paired.shared.get_classifier_client().with_options(timeout=60)
            paired.write_json(pending, {"call_index": len(events), "arm": arm, "question_id": row["question_id"]})
            raw = request_record(row, arm, len(events), contract, client)
            paired.write_json(stage, raw)  # Durable raw response/usage BEFORE any parsing.
            record = finish_record(raw, row)
            paired.shared._append_jsonl(root / arm / "attempts.jsonl", record)
            stage.unlink()
            pending.unlink()
            events, completed, counts, spent = read_state()
            new_calls += 1
            print(f"OUTCOMES={completed}/{len(ordered)} CALLS={len(events)} COST_CNY={spent:.6f}", flush=True)
        return events
    finally:
        lock.unlink()


def evaluate(*, root=ROOT):
    contract, rows = load_probe()
    events, completed, _, spent = paired.load_state(root, rows, contract, builders=BUILDERS,
        result_parser=parse_result, contract_sha256=controls.prep.prior.canonical_sha256(contract))
    if (Path(root) / "runner.lock").exists() or completed != 28:
        raise RuntimeError("all 28 terminal outcomes and stopped runner required before evaluation")
    cases = controls.load_cases()  # Targets opened ONLY after every paid outcome.
    if controls.prep.prior.canonical_sha256(cases) != paired.shared._read_json(controls.MANIFEST_PATH)["cases_canonical_sha256"]:
        raise RuntimeError("frozen targets changed")
    outcomes = {(e["question_id"], e["arm"]): e for e in events}
    arms, families = {}, {}
    for arm in BUILDERS:
        attempts = [e for e in events if e["arm"] == arm]
        matrix = {f"{target.lower()}_as_{decision.lower()}": sum(
            c["expected_class"] == target and outcomes[c["question_id"], arm]["result"] is not None
            and outcomes[c["question_id"], arm]["result"]["decision"] == decision for c in cases)
            for target in ("SUFFICIENT", "INSUFFICIENT") for decision in ("SUFFICIENT", "INSUFFICIENT")}
        arms[arm] = {"case_denominator": 14, "confusion": matrix,
                     "failures": sum(e["result"] is None for e in attempts),
                     "valid_correct": matrix["sufficient_as_sufficient"] + matrix["insufficient_as_insufficient"],
                     "estimated_cost_cny": sum(e["estimated_cost_cny"] for e in attempts),
                     "latency_p50_ms": paired.shared._percentile([e["latency_ms"] for e in attempts], .5),
                     "prompt_tokens": sum(e["prompt_tokens"] for e in attempts),
                     "completion_tokens": sum(e["completion_tokens"] for e in attempts)}
    for family in sorted(controls.FAMILIES):
        group = [c for c in cases if c["family"] == family]
        families[family] = {arm: [{"question_id": c["question_id"], "expected": c["expected_class"],
            "decision": outcomes[c["question_id"], arm]["result"]["decision"] if outcomes[c["question_id"], arm]["result"] else None}
            for c in group] for arm in BUILDERS}
    summary = {"status": "CONTROLLED_MODEL_DIAGNOSTIC_COMPLETE_NO_PROMOTION", "cases": 14,
               "provider_calls": len(events), "estimated_cost_cny": spent, "arms": arms, "families": families,
               "contract_sha256": controls.prep.prior.canonical_sha256(contract),
               "journal_sha256": {arm: paired.shared._sha256(Path(root) / arm / "attempts.jsonl") for arm in BUILDERS},
               "provenance": "Fictional paired fixtures; construction targets, not real-data accuracy or independent gold",
               "natural_train_pooled": False, "dev_opened": False, "promotion_allowed": False}
    controls.prep.prior._write_json(REPORT / "v5_2_control_result.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-contract", action="store_true")
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--env-file")
    parser.add_argument("--max-new-calls", type=int, default=28)
    args = parser.parse_args()
    if args.prepare_contract:
        value = prepare_contract()
        print(json.dumps({"max_calls": 28, "planning_upper_cny": value["pricing_snapshot"]["planning_upper_cny"]}))
    elif args.run_paid:
        if args.env_file:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        events = run_paid(max_new_calls=args.max_new_calls)
        print(json.dumps({"terminal_outcomes": len(events)}))
    else:
        value = evaluate()
        print(json.dumps({k: value[k] for k in ("status", "provider_calls", "estimated_cost_cny", "arms")}, indent=2))
