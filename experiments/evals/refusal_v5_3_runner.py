"""Authorized observed-TRAIN regression; no independent acceptance or serving."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.evals import refusal_natural_diagnostic_runner as previous
from experiments.evals import refusal_v5_3_preparation as prep

canonical, paired = prep.canonical, prep.prior.paired
CONTRACT = prep.prior.prior.REPORT / "v5_3_regression_run_contract.json"
RESULT = prep.prior.prior.REPORT / "v5_3_regression_result.json"
ROOT = Path("data/refusal_natural_v5_3/run_v1")
PREPARATION_SHA256 = "403bedd5717d31d1a7561051da2841403ab220503532ec20052a5727b7c2dd7a"

# ponytail: isolate the existing runner; adapt its preparation names, not its durable engine.
_spec = importlib.util.spec_from_file_location("_v5_3_natural_regression", previous.__file__)
_reuse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reuse)
_reuse.prep = SimpleNamespace(**vars(prep), paired=paired, natural=prep.prior.natural,
                             INPUTS=prep.prior.INPUTS, TARGETS=prep.prior.TARGETS)
_reuse.PREPARATION_SHA256, _reuse.CONTRACT = PREPARATION_SHA256, CONTRACT


def execution_contract(manifest):
    budget = manifest["preflight"]
    return {**manifest, "status": "AUTHORIZED_NOT_RUN", "preparation_canonical_sha256": canonical(manifest),
        "runner_normalized_source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "reused_runner_normalized_sha256": hashlib.sha256(Path(previous.__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "max_calls_total": 60, "classifier": budget["classifier"], "pricing_snapshot": budget["pricing_snapshot"],
        "arms": {arm: {"max_calls": 30, "max_output_tokens": value["max_output_tokens"],
                       "request_input_byte_bounds": value["input_byte_plus_framing_bounds"]}
                 for arm, value in budget["arms"].items()},
        "controls": {**manifest["controls"], "executable_run_contract": True, "paid_runner_ready": True,
                     "failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 30, "sdk_retries": 0},
        "authorization": "User explicitly confirmed PR77 merged and authorized execution under PR77: 30 questions/full Top14, <=60 Singapore qwen3.5-plus-2026-04-20 calls/CNY8; no labels, DEV, retries, recharge or promotion"}


_reuse.execution_contract = execution_contract
prepare_contract = _reuse.prepare_contract
population_metrics = _reuse.population_metrics


def load_probe():
    manifest = paired.shared._read_json(prep.REPORT)
    if hashlib.sha256(prep.PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest() != manifest["protocol_normalized_sha256"]:
        raise RuntimeError("frozen protocol changed")
    return _reuse.load_probe()  # Input/source/request/contract bindings; never target rows.


def parse_result(arm, raw, row, finish_reason):
    if arm not in prep.BUILDERS or finish_reason != "stop":
        raise ValueError("invalid arm or incomplete response")
    return prep.candidate.validate_requirement_coverage_response(raw, paired.shared._classifier_input_from_row(row))


_reuse._engine.BUILDERS, _reuse._engine.parse_result = prep.BUILDERS, parse_result


def run_paid(*, root=ROOT, client=None, max_new_calls=60, probe_loader=load_probe):
    return _reuse.run_paid(root=root, client=client, max_new_calls=max_new_calls, probe_loader=probe_loader)


def evaluate(*, root=ROOT):
    contract, rows = load_probe()
    events, completed, _, spent = paired.load_state(root, rows, contract, builders=prep.BUILDERS,
        result_parser=parse_result, contract_sha256=canonical(contract))
    if completed != 60 or (Path(root) / "runner.lock").exists():
        raise RuntimeError("all 60 terminal outcomes and stopped runner required")
    if paired.shared._sha256(prep.prior.TARGETS) != contract["targets_sha256"]:
        raise RuntimeError("frozen target hash changed")
    labels = paired.shared._read_jsonl(prep.prior.TARGETS)  # After all paid outcomes, never in classifier requests.
    if [t["question_id"] for t in labels] != contract["case_order"]:
        raise RuntimeError("target order changed")
    outcomes = {(e["question_id"], e["arm"]): e["result"] for e in events}
    transitions = [{"question_id": t["question_id"], "target_class": t["target_class"],
        **{arm: outcomes[t["question_id"], arm]["decision"] if outcomes[t["question_id"], arm] else None
           for arm in prep.BUILDERS}} for t in labels]
    primary = population_metrics(transitions)
    if primary["target_counts"] != contract["target_counts"]:
        raise RuntimeError("frozen target counts changed")
    common = [t for t in transitions if t["target_class"] != "AMBIGUOUS" and all(t[a] is not None for a in prep.BUILDERS)]
    summary = {"status": "OBSERVED_DEVELOPMENT_REGRESSION_COMPLETE_NO_ACCEPTANCE", "provider_calls": len(events),
        "estimated_cost_cny": spent, "primary": primary, "common_valid_binary": population_metrics(common),
        "all_outcome_transitions": {f"{a or 'FAILURE'}_to_{b or 'FAILURE'}": sum(
            t["v5_2_1"] == a and t["v5_3"] == b for t in transitions)
            for a in ("SUFFICIENT", "INSUFFICIENT", None) for b in ("SUFFICIENT", "INSUFFICIENT", None)},
        "paired_binary_agreement_transitions": {name: sum((t["v5_2_1"] == t["target_class"], t["v5_3"] == t["target_class"]) == state for t in common)
            for name, state in (("both_agree", (True, True)), ("baseline_only", (True, False)),
                                ("candidate_only", (False, True)), ("neither_agrees", (False, False)))},
        "family_sensitivity": {f"drop_{qid}": population_metrics([t for t in transitions if t["question_id"] != qid])
                               for qid in contract["known_correlated_pair"]},
        "usage": {arm: {"calls": sum(e["arm"] == arm for e in events),
                       "prompt_tokens": sum(e["prompt_tokens"] for e in events if e["arm"] == arm),
                       "completion_tokens": sum(e["completion_tokens"] for e in events if e["arm"] == arm),
                       "estimated_cost_cny": sum(e["estimated_cost_cny"] for e in events if e["arm"] == arm),
                       "latency_p50_ms": paired.shared._percentile([e["latency_ms"] for e in events if e["arm"] == arm], .5)}
                  for arm in prep.BUILDERS},
        "contract_canonical_sha256": canonical(contract), "targets_sha256": contract["targets_sha256"],
        "journal_sha256": {arm: paired.shared._sha256(Path(root) / arm / "attempts.jsonl") for arm in prep.BUILDERS},
        "provenance": contract["provenance"], "original_class_support_rule_met": False,
        "original_formal_rule_changed": False, "formal_pass_gate": None, "dev_opened": False, "promotion_allowed": False}
    if RESULT.exists() or (Path(root) / "summary.json").exists():
        raise RuntimeError("refusing to overwrite frozen evaluation")
    paired.write_json(Path(root) / "summary.json", {"aggregate": summary, "private_transitions": transitions})
    prep.prior.natural.prior._write_json(RESULT, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-contract", action="store_true")
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--env-file")
    parser.add_argument("--max-new-calls", type=int, default=60)
    args = parser.parse_args()
    if args.prepare_contract:
        value = prepare_contract()
        print(json.dumps({"max_calls": 60, "planning_cost_cny": value["preflight"]["planning_cost_cny"]}))
    elif args.run_paid:
        if args.env_file:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        print(json.dumps({"terminal_outcomes": len(run_paid(max_new_calls=args.max_new_calls))}))
    else:
        value = evaluate()
        print(json.dumps({k: value[k] for k in ("status", "provider_calls", "estimated_cost_cny", "primary")}, indent=2))
