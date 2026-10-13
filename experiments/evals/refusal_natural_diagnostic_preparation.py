"""Inert natural TRAIN development diagnostic; no independent acceptance claim."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.evals import refusal_v5_2_1_preparation as reuse

prior, paired = reuse.prior, reuse.prior.paired
natural = prior.controls.prep
REPORT = prior.REPORT / "natural_v5_2_1_diagnostic_preparation.json"
INPUTS, TARGETS = natural.DATA_DIR / "inputs.jsonl", natural.DATA_DIR / "targets.jsonl"
BUILDERS = {"v5_1": prior.citation.build_classifier_messages_v5_1_dev,
            "v5_2_1": reuse.candidate.build_classifier_messages_v5_2_1_dev}
SOURCE_FILES = (*reuse.SOURCE_FILES, "refusal_v2_ai_proxy_runner.py", "refusal_natural_diagnostic_preparation.py")


def paired_preflight(rows):
    if len(rows) != 30:
        raise RuntimeError("complete frozen 30-case diagnostic required")
    # ponytail: retain the existing v5.1 preflight and compute only the new arm.
    old = natural.paired_preflight(rows)
    messages = [BUILDERS["v5_2_1"](paired.shared._classifier_input_from_row(row)) for row in rows]
    bounds = [sum(len(m["content"].encode("utf-8")) for m in value) + 512 for value in messages]
    if max(bounds) + 2048 > 256_000:
        raise RuntimeError("pricing context band exceeded")
    arms = {"v5_1": old["arms"]["v5_1"], "v5_2_1": {
        "prompt_sha256": hashlib.sha256(messages[0][0]["content"].encode()).hexdigest(),
        "request_sha256": [natural.prior.canonical_sha256(value) for value in messages],
        "input_byte_plus_framing_bounds": bounds, "max_output_tokens": 2048, "planned_calls": 30,
        "planning_cost_usd": (sum(bounds) * 0.4 + 30 * 2048 * 2.4) / 1_000_000,
    }}
    total = sum(arm["planning_cost_usd"] for arm in arms.values()) * 7.5
    if total > 8:
        raise RuntimeError("proposed CNY8 accounting budget exceeded")
    return {"arms": arms, "planned_calls": 60, "planning_cost_cny": total,
            "classifier": {"model": old["model"], "region": old["region"], "temperature": 0,
                           "enable_thinking": False, "response_format": "json_object"},
            "pricing_snapshot": {**reuse.PRICING, "hard_cost_cap_cny": 8}}


def build_preparation():
    freeze = paired.shared._read_json(prior.REPORT / "v5_2_annotation_freeze.json")
    selection = paired.shared._read_json(natural.REPORT)
    if selection["inputs_sha256"] != freeze["inputs_sha256"] or paired.shared._sha256(TARGETS) != freeze["targets_sha256"]:
        raise RuntimeError("frozen input/target bindings changed")
    rows = paired.v2.load_classifier_inputs(INPUTS, expected_sha256=freeze["inputs_sha256"], expected_cases=30)
    labels = paired.shared._read_jsonl(TARGETS)
    order = selection["selected_question_ids"]
    counts = dict(Counter(row["target_class"] for row in labels))
    if ([r["question_id"] for r in rows] != order or [r["question_id"] for r in labels] != order
            or counts != freeze["counts"] or counts != {"SUFFICIENT": 21, "INSUFFICIENT": 4, "AMBIGUOUS": 5}):
        raise RuntimeError("frozen population/order/classes changed")
    return {
        "run": "natural_v5_1_v5_2_1_exploratory_diagnostic_v1", "status": "PREPARED_NOT_RUN_NOT_AUTHORIZED",
        "cases": 30, "case_order": order, "target_counts": counts, "binary_cases": 25, "ambiguous_cases": 5,
        "inputs_path": INPUTS.as_posix(), "targets_path": TARGETS.as_posix(),
        "inputs_sha256": freeze["inputs_sha256"], "targets_sha256": freeze["targets_sha256"],
        "annotation_freeze_canonical_sha256": natural.prior.canonical_sha256(freeze),
        "selection_canonical_sha256": natural.prior.canonical_sha256(selection),
        "original_protocol_normalized_sha256": hashlib.sha256((prior.REPORT / "v5_2_train_protocol.md").read_text(encoding="utf-8").encode()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest()
                          for name in SOURCE_FILES},
        "preflight": paired_preflight(rows), "checkpoint_root": "data/refusal_natural_v5_2_1/run_v1",
        "order": "frozen question-ID order; even index v5_1 then v5_2_1, odd index reversed",
        "known_correlated_pair": freeze["observed_correlated_selected_family"],
        "family_sensitivity": "Primary includes all30; additionally recompute removing each known correlated member separately, without new calls or relabeling; not independent-family validation",
        "category_counts": freeze["category_counts"], "provider_calls": 0,
        "metrics": {"all_case_outcomes_and_failures_denominator": 30,
                    "binary_target_agreement_denominator_including_failures": 25,
                    "ambiguous_decisions_descriptive_only_denominator": 5,
                    "class_denominators_including_failures": {"SUFFICIENT": 21, "INSUFFICIENT": 4},
                    "common_valid_binary_pairs": "Report actual denominator/classes, confusion and transitions separately",
                    "admission_coverage": "All30; binary admitted proxy agreement separately; ambiguous admissions do not count as correct",
                    "failure_sensitivity": "Keep null outcomes; report possible binary target-agreement lower/upper bounds, not errors as refusals",
                    "cost_tokens_latency": "Per arm; include every charged outcome"},
        "controls": {"authorization_required": True, "paid_runner_ready": False, "executable_run_contract": False,
                     "max_calls_total": 60, "schema_failure_terminal_no_retry": True,
                     "raw_before_parse_required": True, "unknown_charge_blocks_resume": True,
                     "targets_outside_paid_loop_required": True, "successful_outcomes_never_repeated": True,
                     "dev_opened": False, "promotion_allowed": False, "formal_pass_gate": None,
                     "original_class_support_rule_met": freeze["class_support_at_least_five_each"],
                     "original_formal_rule_changed": False, "independent_or_representative_acceptance_claim": False,
                     "new_retrieval_or_annotations": False},
        "decision_rule": "Descriptive development diagnosis only, regardless of results; no formal pass, significance, independent validation or production promotion",
        "provenance": freeze["provenance"] + "; all TRAIN already used in project retrieval/automated analysis; not representative enterprise traffic; conflict category absent",
    }


if __name__ == "__main__":
    value = build_preparation()
    natural.prior._write_json(REPORT, value)
    print(json.dumps({"status": value["status"], "cases": value["cases"], "calls": 60,
                      "planning_cost_cny": value["preflight"]["planning_cost_cny"], "provider_calls": 0}))
