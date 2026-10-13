"""Bounded natural TRAIN diagnosis; private labels/results, no acceptance or serving."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from experiments.evals import refusal_natural_diagnostic_preparation as prep

prior, paired = prep.prior, prep.paired
canonical = prep.natural.prior.canonical_sha256
PREPARATION_SHA256 = "57e8de2e6594acadb36201e6017befbec9306600a5661926d50a1a78c3566bf9"
CONTRACT = prior.REPORT / "natural_v5_2_1_run_contract.json"
RESULT = prior.REPORT / "natural_v5_2_1_result.json"
ROOT = Path("data/refusal_natural_v5_2_1/run_v1")


def parse_result(arm, raw, row, finish_reason):
    if arm not in prep.BUILDERS or finish_reason != "stop":
        raise ValueError("invalid arm or incomplete response")
    validator = prior.citation.validate_citation_selection_response if arm == "v5_1" else prior.coverage.validate_requirement_coverage_response
    return validator(raw, paired.shared._classifier_input_from_row(row))


# ponytail: the same frozen durable engine, isolated arm/parser bindings, no new client.
_spec = importlib.util.spec_from_file_location("_natural_diagnostic_engine", prior.__file__)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
_engine.BUILDERS, _engine.parse_result = prep.BUILDERS, parse_result


def execution_contract(manifest):
    budget = manifest["preflight"]
    return {**manifest, "status": "AUTHORIZED_NOT_RUN", "preparation_canonical_sha256": canonical(manifest),
        "runner_normalized_source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "max_calls_total": 60, "classifier": budget["classifier"], "pricing_snapshot": budget["pricing_snapshot"],
        "arms": {arm: {"max_calls": 30, "max_output_tokens": value["max_output_tokens"],
                       "request_input_byte_bounds": value["input_byte_plus_framing_bounds"]}
                 for arm, value in budget["arms"].items()},
        "controls": {**manifest["controls"], "executable_run_contract": True, "paid_runner_ready": True,
                     "failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 30, "sdk_retries": 0},
        "authorization": "User confirmed PR74 merged and replied continue to explicit 30 questions/full Top14, <=60 Singapore qwen3.5-plus-2026-04-20 calls/CNY8 proposal; no labels, DEV, retries, recharge or promotion"}


def prepare_contract():
    manifest = paired.shared._read_json(prep.REPORT)
    if canonical(manifest) != PREPARATION_SHA256 or manifest != prep.build_preparation():
        raise RuntimeError("frozen preparation changed")
    value = execution_contract(manifest)
    prep.natural.prior._write_json(CONTRACT, value)
    return value


def load_probe():
    manifest = paired.shared._read_json(prep.REPORT)  # Counts/hashes only; no target rows.
    contract = paired.shared._read_json(CONTRACT)
    if canonical(manifest) != PREPARATION_SHA256 or contract != execution_contract(manifest):
        raise RuntimeError("frozen execution binding changed")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() != expected:
            raise RuntimeError("frozen dependency changed")
    rows = paired.v2.load_classifier_inputs(prep.INPUTS, expected_sha256=manifest["inputs_sha256"], expected_cases=30)
    if [r["question_id"] for r in rows] != manifest["case_order"]:
        raise RuntimeError("frozen input order changed")
    for index, row in enumerate(rows):
        for arm, builder in prep.BUILDERS.items():
            messages = builder(paired.shared._classifier_input_from_row(row))
            bound = sum(len(m["content"].encode("utf-8")) for m in messages) + 512
            if (canonical(messages) != manifest["preflight"]["arms"][arm]["request_sha256"][index]
                    or bound != contract["arms"][arm]["request_input_byte_bounds"][index]):
                raise RuntimeError("frozen request changed")
    return contract, rows


def run_paid(*, root=ROOT, client=None, max_new_calls=60, probe_loader=load_probe):
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 60:
        raise ValueError("max_new_calls must be between 0 and 60")
    return _engine.run_paid(root=root, client=client, max_new_calls=max_new_calls, probe_loader=probe_loader)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def population_metrics(items):
    binary = [t for t in items if t["target_class"] != "AMBIGUOUS"]
    ambiguous = [t for t in items if t["target_class"] == "AMBIGUOUS"]
    counts = {target: sum(t["target_class"] == target for t in items)
              for target in ("SUFFICIENT", "INSUFFICIENT", "AMBIGUOUS")}
    arms = {}
    for arm in prep.BUILDERS:
        matrix = {f"{target.lower()}_as_{decision.lower()}": sum(t["target_class"] == target and t[arm] == decision for t in binary)
                  for target in ("SUFFICIENT", "INSUFFICIENT") for decision in ("SUFFICIENT", "INSUFFICIENT")}
        correct = matrix["sufficient_as_sufficient"] + matrix["insufficient_as_insufficient"]
        failed = sum(t[arm] is None for t in binary)
        admitted = sum(t[arm] == "SUFFICIENT" for t in items)
        binary_admitted = sum(t[arm] == "SUFFICIENT" for t in binary)
        arms[arm] = {"confusion": matrix, "valid_correct_binary": correct,
            "all_case_failures": sum(t[arm] is None for t in items), "binary_failures": failed,
            "class_failures": {target: sum(t[arm] is None and t["target_class"] == target for t in binary)
                               for target in ("SUFFICIENT", "INSUFFICIENT")},
            "binary_agreement_including_failures": ratio(correct, len(binary)),
            "binary_agreement_possible_bounds": [ratio(correct, len(binary)), ratio(correct + failed, len(binary))],
            "sufficient_proxy_recall": ratio(matrix["sufficient_as_sufficient"], counts["SUFFICIENT"]),
            "insufficient_proxy_recall": ratio(matrix["insufficient_as_insufficient"], counts["INSUFFICIENT"]),
            "admitted_all": admitted, "admission_coverage": ratio(admitted, len(items)),
            "admitted_binary": binary_admitted,
            "admitted_binary_proxy_agreement": ratio(matrix["sufficient_as_sufficient"], binary_admitted),
            "ambiguous_outcomes": {decision or "FAILURE": sum(t[arm] == decision for t in ambiguous)
                                   for decision in ("SUFFICIENT", "INSUFFICIENT", None)}}
    return {"cases": len(items), "binary_cases": len(binary), "ambiguous_cases": len(ambiguous),
            "target_counts": counts, "arms": arms}


def evaluate(*, root=ROOT):
    contract, rows = load_probe()
    events, completed, _, spent = paired.load_state(root, rows, contract, builders=prep.BUILDERS,
        result_parser=parse_result, contract_sha256=canonical(contract))
    if completed != 60 or (Path(root) / "runner.lock").exists():
        raise RuntimeError("all 60 terminal outcomes and stopped runner required")
    if paired.shared._sha256(prep.TARGETS) != contract["targets_sha256"]:
        raise RuntimeError("frozen target hash changed")
    labels = paired.shared._read_jsonl(prep.TARGETS)  # First target read, after all paid outcomes.
    if [t["question_id"] for t in labels] != contract["case_order"]:
        raise RuntimeError("target order changed")
    outcomes = {(e["question_id"], e["arm"]): e["result"] for e in events}
    transitions = [{"question_id": t["question_id"], "target_class": t["target_class"],
        **{arm: outcomes[t["question_id"], arm]["decision"] if outcomes[t["question_id"], arm] else None
           for arm in prep.BUILDERS}} for t in labels]
    primary = population_metrics(transitions)
    if primary["target_counts"] != contract["target_counts"]:
        raise RuntimeError("frozen target counts changed")
    common = [t for t in transitions if t["target_class"] != "AMBIGUOUS" and all(t[arm] is not None for arm in prep.BUILDERS)]
    summary = {"status": "EXPLORATORY_DIAGNOSTIC_COMPLETE_NO_ACCEPTANCE", "provider_calls": len(events),
        "estimated_cost_cny": spent, "primary": primary, "common_valid_binary": population_metrics(common),
        "all_outcome_transitions": {f"{a or 'FAILURE'}_to_{b or 'FAILURE'}": sum(
            t["v5_1"] == a and t["v5_2_1"] == b for t in transitions)
            for a in ("SUFFICIENT", "INSUFFICIENT", None) for b in ("SUFFICIENT", "INSUFFICIENT", None)},
        "paired_binary_agreement_transitions": {name: sum((t["v5_1"] == t["target_class"], t["v5_2_1"] == t["target_class"]) == state for t in common)
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
    prep.natural.prior._write_json(RESULT, summary)  # No target rows, real bodies or raw responses in public output.
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
