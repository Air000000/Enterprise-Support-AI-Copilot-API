"""Zero-call preparation on the frozen prospective50; no paid execution entrypoint."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_prospective_annotation as readiness
from experiments.evals import refusal_version_audit_clarification as candidate

prior = readiness.prep
canonical, shared = prior.canonical, prior.shared
REPORT = prior.selection.prior.REPORT_DIR / "v5_3_1_prospective_preparation.json"
PROTOCOL = REPORT.with_name("v5_3_1_prospective_protocol.md")
DATA = Path("data/refusal_v5_3_1_prospective_train")
BUILDERS = {"v5_3": candidate.prior.build_classifier_messages_v5_3_dev,
            "v5_3_1": candidate.build_classifier_messages_v5_3_1_dev}
PRIOR_STATE_PATHS = (readiness.CONTRACT, readiness.FREEZE, readiness.ROOT,
    prior.DATA / "blind_draft_annotations.jsonl", prior.DATA / "reviewed_annotations.jsonl",
    prior.DATA / "targets.jsonl", prior.DATA / "paired_run_v1", DATA)
SOURCE_FILES = (*prior.SOURCE_FILES, "refusal_prospective_annotation.py",
                "refusal_version_audit_clarification.py", "refusal_v5_3_1_preparation.py")


def preflight(rows):
    # ponytail: reuse the frozen annotation/baseline calculation; replace only the comparison arm.
    previous = prior.preflight(rows)
    messages = [BUILDERS["v5_3_1"](shared._classifier_input_from_row(row)) for row in rows]
    bounds = [sum(len(m["content"].encode("utf-8")) for m in value) + 512 for value in messages]
    if max(bounds) + 2048 > 256_000:
        raise RuntimeError("planning context band exceeded")
    arms = {arm: previous["arms"][arm] for arm in ("annotation", "v5_3")}
    arms["v5_3_1"] = {"prompt_sha256": hashlib.sha256(messages[0][0]["content"].encode()).hexdigest(),
        "request_sha256": [canonical(value) for value in messages],
        "input_byte_plus_framing_bounds": bounds, "max_output_tokens": 2048, "max_calls": 50,
        "planning_cost_cny": (sum(bounds) * 3 + 50 * 2048 * 18) / 1_000_000}
    costs = {"annotation": arms["annotation"]["planning_cost_cny"],
             "paired_classification": sum(arms[a]["planning_cost_cny"] for a in BUILDERS)}
    if costs["annotation"] > 10 or costs["paired_classification"] > 15:
        raise RuntimeError("proposed stage budgets exceeded")
    return {**previous, "arms": arms, "stage_planning_cost_cny": costs,
        "planning_cost_cny": sum(costs.values()),
        "pricing": {**previous["pricing"], "checked_date": "2026-10-20"}}


def build_preparation():
    # This existing read-only loader validates the frozen selection, inputs, order and sources.
    frozen, rows = readiness.load_inputs()
    if any(path.exists() for path in PRIOR_STATE_PATHS):
        raise RuntimeError("existing annotation/target/checkpoint/contract requires audit; no fresh-state assumption")
    value = {"run": "prospective_train_v5_3_v5_3_1_development_comparison_v1",
        "status": "PREPARED_NOT_ANNOTATED_NOT_AUTHORIZED", "cases": 50,
        "case_order": frozen["selected_question_ids"],
        "inputs_path": (prior.DATA / "inputs.jsonl").as_posix(), "inputs_sha256": frozen["inputs_sha256"],
        "selection_canonical_sha256": canonical(frozen),
        "selection_protocol_normalized_sha256": frozen["protocol_normalized_sha256"],
        "protocol_normalized_sha256": hashlib.sha256(PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest()
                          for name in SOURCE_FILES},
        "preflight": preflight(rows), "checkpoint_root": DATA.as_posix(),
        "order": "frozen question-ID order; even index v5_3 then v5_3_1, odd index reversed",
        "prior_state_checked_paths": [path.as_posix() for path in PRIOR_STATE_PATHS],
        "prior_state_observation": "No artifacts at these paths in this checkout; not a search of other checkouts or machines",
        "provider_calls": 0, "labels_frozen": False, "targets_sha256": None,
        "metrics": {"all_case_denominator": 50,
            "binary_agreement": "All frozen binary targets, including failed predictions; report per-class confusion and recall",
            "paired_transitions": "Common-valid binary fixes/regressions, with actual denominator and class counts",
            "ambiguous": "Descriptive only; never counted as correct or silently excluded from all50 outcomes",
            "failures": "Null outcomes; report format failures and possible agreement lower/upper bounds separately",
            "admission": "All50 coverage; admitted-binary proxy agreement and ambiguous admissions separately",
            "version_audit": "Descriptive reasons/evidence and contradictions; no new facet accuracy without pre-frozen facet targets",
            "cost_latency": "Every charged outcome per stage/arm; tokens, accounting cost and provider latency"},
        "controls": {**frozen["controls"], "executable_run_contract": False, "runner_ready": False,
            "max_calls_total": 150, "max_annotation_calls": 50, "max_classifier_calls_per_arm": 50,
            "sdk_max_retries": 0, "network_provider_or_unknown_charge_stops": True,
            "classifier_schema_failure_terminal_null": True, "new_checkpoint_required": True,
            "annotation_review_and_freeze_before_predictions": True,
            "old_execution_permissions_reused": False, "independent_acceptance_claim": False,
            "generation_allowed": False, "retrieval_allowed": False},
        "provenance": frozen["provenance"],
        "decision_rule": "Descriptive fresh-to-refusal TRAIN comparison, not independent acceptance, PASS or default promotion"}
    value["plan_canonical_sha256"] = canonical(value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Save inert metadata exclusively; never call a model")
    args = parser.parse_args()
    value = build_preparation()
    if args.write_manifest:
        prior.selection.prior._write_json(REPORT, value)  # Exclusive creation; never overwrite a freeze.
    print(json.dumps({key: value[key] for key in ("status", "cases", "provider_calls", "plan_canonical_sha256")} | {
        "planned_calls": 150, "planning_cost_cny": value["preflight"]["planning_cost_cny"],
        "proposed_accounting_cap_cny": 25}, indent=2))
