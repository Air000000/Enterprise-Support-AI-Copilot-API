"""Authorized 28-call fictional v5.2/v5.2.1 regression; no TRAIN/DEV or serving."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from experiments.evals import refusal_v5_2_1_preparation as prep

prior, paired = prep.prior, prep.prior.paired
canonical = prior.controls.prep.prior.canonical_sha256
PREPARATION_SHA256 = "11d47a3eaeb2ecba03b7752fa678d6377f9a90e97964b5eba389fa8914fff11e"
CONTRACT = prior.REPORT / "v5_2_1_regression_run_contract.json"
RESULT = prior.REPORT / "v5_2_1_regression_result.json"
ROOT = Path("data/refusal_v5_2_1_controls/run_v1")
INPUTS = ROOT.parent / "inputs.jsonl"


def parse_result(arm, raw, row, finish_reason):
    if arm not in prep.BUILDERS or finish_reason != "stop":
        raise ValueError("invalid arm or incomplete response")
    return prior.coverage.validate_requirement_coverage_response(raw, paired.shared._classifier_input_from_row(row))


# ponytail: load the frozen runner in an isolated namespace, not a copied engine or shared monkeypatch.
_spec = importlib.util.spec_from_file_location("_v5_2_1_checkpoint_engine", prior.__file__)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
_engine.BUILDERS, _engine.parse_result = prep.BUILDERS, parse_result


def execution_contract(manifest, inputs_sha256):
    return {**manifest, "status": "AUTHORIZED_NOT_RUN", "inputs_sha256": inputs_sha256,
            "preparation_canonical_sha256": canonical(manifest),
            "runner_normalized_source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
            "controls": {**manifest["controls"], "executable_run_contract": True, "paid_runner_ready": True,
                         "failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 14, "sdk_retries": 0},
            "authorization": "User confirmed PR72 merged and explicitly authorized 14 fictional questions/sources to Singapore qwen3.5-plus-2026-04-20, <=28 calls/CNY3; no labels, DEV, retries or recharge"}


def prepare_contract():
    manifest = paired.shared._read_json(prep.REPORT)
    if canonical(manifest) != PREPARATION_SHA256 or manifest != prep.build_preparation():
        raise RuntimeError("frozen preparation changed")
    if INPUTS.exists() or CONTRACT.exists():
        raise RuntimeError("refusing to overwrite execution inputs/contract")
    rows = prior.controls.classifier_rows(prior.controls.load_cases())
    prior.controls.prep.prior._write_jsonl(INPUTS, rows)
    value = execution_contract(manifest, paired.shared._sha256(INPUTS))
    prior.controls.prep.prior._write_json(CONTRACT, value)
    return value


def load_probe():
    manifest = paired.shared._read_json(prep.REPORT)  # Metadata only, no target rows.
    contract = paired.shared._read_json(CONTRACT)
    if canonical(manifest) != PREPARATION_SHA256 or contract != execution_contract(manifest, paired.shared._sha256(INPUTS)):
        raise RuntimeError("frozen execution binding changed")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() != expected:
            raise RuntimeError("frozen dependency changed")
    rows = paired.shared._read_jsonl(INPUTS)
    if (canonical(rows) != manifest["inputs_canonical_sha256"] or len(rows) != 14
            or [r["question_id"] for r in rows] != manifest["case_order"]):
        raise RuntimeError("frozen classifier inputs changed")
    for index, row in enumerate(rows):
        if set(row) != {"question_id", "question", "sources"}:
            raise RuntimeError("labels/metadata leaked into classifier input")
        for arm, builder in prep.BUILDERS.items():
            messages = builder(paired.shared._classifier_input_from_row(row))
            bound = sum(len(m["content"].encode("utf-8")) for m in messages) + 512
            if (canonical(messages) != manifest["arms"][arm]["request_sha256"][index]
                    or bound != manifest["arms"][arm]["request_input_byte_bounds"][index]):
                raise RuntimeError("frozen request changed")
    return contract, rows


def run_paid(*, root=ROOT, client=None, max_new_calls=28, probe_loader=load_probe):
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 28:
        raise ValueError("max_new_calls must be between 0 and 28")
    return _engine.run_paid(root=root, client=client, max_new_calls=max_new_calls, probe_loader=probe_loader)


def evaluate(*, root=ROOT):
    contract, rows = load_probe()
    events, completed, _, spent = paired.load_state(root, rows, contract, builders=prep.BUILDERS,
        result_parser=parse_result, contract_sha256=canonical(contract))
    if completed != 28 or (Path(root) / "runner.lock").exists():
        raise RuntimeError("all 28 terminal outcomes and stopped runner required")
    cases = prior.controls.load_cases()  # Targets opened only AFTER the entire paid loop.
    if canonical(cases) != contract["cases_canonical_sha256"]:
        raise RuntimeError("frozen targets changed")
    outcomes = {(e["question_id"], e["arm"]): e for e in events}
    transitions, arms = [], {}
    for case in cases:
        decisions = {arm: outcomes[case["question_id"], arm]["result"] for arm in prep.BUILDERS}
        transitions.append({"question_id": case["question_id"], "family": case["family"],
            "expected": case["expected_class"],
            **{arm: value["decision"] if value else None for arm, value in decisions.items()}})
    for arm in prep.BUILDERS:
        attempts = [e for e in events if e["arm"] == arm]
        confusion = {f"{target.lower()}_as_{decision.lower()}": sum(
            t["expected"] == target and t[arm] == decision for t in transitions)
            for target in ("SUFFICIENT", "INSUFFICIENT") for decision in ("SUFFICIENT", "INSUFFICIENT")}
        arms[arm] = {"case_denominator": 14, "confusion": confusion,
            "valid_correct": confusion["sufficient_as_sufficient"] + confusion["insufficient_as_insufficient"],
            "failures": sum(e["result"] is None for e in attempts),
            "estimated_cost_cny": sum(e["estimated_cost_cny"] for e in attempts),
            "latency_p50_ms": paired.shared._percentile([e["latency_ms"] for e in attempts], .5),
            "prompt_tokens": sum(e["prompt_tokens"] for e in attempts),
            "completion_tokens": sum(e["completion_tokens"] for e in attempts)}
    new_errors = [t["question_id"] for t in transitions if t["v5_2_1"] is not None
                  and t["v5_2_1"] != t["expected"] and t["v5_2"] in (None, t["expected"])]
    common = [t for t in transitions if t["v5_2"] is not None and t["v5_2_1"] is not None]
    signal = arms["v5_2_1"]["failures"] < arms["v5_2"]["failures"] and not new_errors
    summary = {"status": "FORMAT_FIX_DEVELOPMENT_SIGNAL" if signal else "NO_DEMONSTRATED_FORMAT_FIX_GAIN",
        "cases": 14, "provider_calls": len(events), "estimated_cost_cny": spent, "arms": arms,
        "transitions": transitions, "new_valid_semantic_errors": new_errors,
        "common_valid": {"cases": len(common), **{arm: sum(t[arm] == t["expected"] for t in common) for arm in prep.BUILDERS}},
        "contract_canonical_sha256": canonical(contract),
        "journal_sha256": {arm: paired.shared._sha256(Path(root) / arm / "attempts.jsonl") for arm in prep.BUILDERS},
        "provenance": contract["provenance"], "natural_train_pooled": False, "dev_opened": False, "promotion_allowed": False}
    prior.controls.prep.prior._write_json(RESULT, summary)
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
        print(json.dumps({"max_calls": value["max_calls_total"], "planning_cost_cny": value["planning_cost_cny"]}))
    elif args.run_paid:
        if args.env_file:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        print(json.dumps({"terminal_outcomes": len(run_paid(max_new_calls=args.max_new_calls))}))
    else:
        value = evaluate()
        print(json.dumps({k: value[k] for k in ("status", "provider_calls", "estimated_cost_cny", "arms")}, indent=2))
