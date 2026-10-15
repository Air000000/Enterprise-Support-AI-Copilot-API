"""Inert observed-TRAIN v5.2.1/v5.3 regression; no paid entrypoint."""

import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_natural_diagnostic_preparation as prior
from experiments.evals import refusal_question_scoped_coverage as candidate

REPORT = prior.prior.REPORT / "v5_3_regression_preparation.json"
PROTOCOL = prior.prior.REPORT / "v5_3_regression_protocol.md"
BUILDERS = {"v5_2_1": prior.BUILDERS["v5_2_1"], "v5_3": candidate.build_classifier_messages_v5_3_dev}
SOURCE_FILES = (*prior.SOURCE_FILES, "refusal_question_scoped_coverage.py", "refusal_v5_3_preparation.py")
canonical = prior.natural.prior.canonical_sha256


def paired_preflight(rows):
    # ponytail: reuse the frozen baseline calculation, compute only the new arm.
    old = prior.paired_preflight(rows)
    messages = [BUILDERS["v5_3"](prior.paired.shared._classifier_input_from_row(row)) for row in rows]
    bounds = [sum(len(m["content"].encode("utf-8")) for m in value) + 512 for value in messages]
    if max(bounds) + 2048 > 256_000:
        raise RuntimeError("pricing context band exceeded")
    arms = {"v5_2_1": old["arms"]["v5_2_1"], "v5_3": {
        "prompt_sha256": hashlib.sha256(messages[0][0]["content"].encode()).hexdigest(),
        "request_sha256": [canonical(value) for value in messages],
        "input_byte_plus_framing_bounds": bounds, "max_output_tokens": 2048, "planned_calls": 30,
        "planning_cost_usd": (sum(bounds) * 0.4 + 30 * 2048 * 2.4) / 1_000_000,
    }}
    total = sum(arm["planning_cost_usd"] for arm in arms.values()) * 7.5
    if total > 8:
        raise RuntimeError("proposed CNY8 accounting budget exceeded")
    return {**old, "arms": arms, "planning_cost_cny": total,
            "pricing_snapshot": {**old["pricing_snapshot"], "checked_date": "2026-10-15"}}


def build_preparation():
    previous = prior.paired.shared._read_json(prior.REPORT)
    if previous != prior.build_preparation():
        raise RuntimeError("previous frozen preparation changed")
    rows = prior.paired.v2.load_classifier_inputs(prior.INPUTS, expected_sha256=previous["inputs_sha256"], expected_cases=30)
    return {
        **previous, "run": "observed_train_v5_2_1_v5_3_development_regression_v1",
        "preflight": paired_preflight(rows), "checkpoint_root": "data/refusal_natural_v5_3/run_v1",
        "order": "frozen question-ID order; even index v5_2_1 then v5_3, odd index reversed",
        "source_hashes": {name: hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest()
                          for name in SOURCE_FILES},
        "protocol_normalized_sha256": hashlib.sha256(PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest(),
        "previous_preparation_canonical_sha256": canonical(previous),
        "observed_prior_result_canonical_sha256": canonical(prior.paired.shared._read_json(prior.prior.REPORT / "natural_v5_2_1_result.json")),
        "baseline_policy": "Fresh concurrent v5_2_1 calls; historical results are provenance only, never a replacement arm",
        "controls": {**previous["controls"], "exclusive_local_lock_required": True,
                     "new_checkpoint_required": True, "sdk_max_retries": 0, "max_calls_per_arm": 30},
        "hypothesis": "Question-scoped requirements may reduce unnecessary refusals; incident applicability may reduce false admissions. Count fixes, regressions and failures across all30, not just observed problem cases",
        "provenance": previous["provenance"] + "; v5.3 designed after observing this population's predictions and targeted evidence; explicitly not unseen validation",
    }


if __name__ == "__main__":
    value = build_preparation()
    prior.natural.prior._write_json(REPORT, value)
    print(json.dumps({"status": value["status"], "cases": value["cases"], "calls": 60,
                      "planning_cost_cny": value["preflight"]["planning_cost_cny"], "provider_calls": 0}))
