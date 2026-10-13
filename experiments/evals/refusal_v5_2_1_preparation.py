"""Inert v5.2/v5.2.1 development regression preflight; no paid entrypoint."""

import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_coverage_citation_example as candidate
from experiments.evals import refusal_v5_2_control_runner as prior

REPORT = prior.REPORT / "v5_2_1_regression_preparation.json"
BUILDERS = {"v5_2": prior.coverage.build_classifier_messages_v5_2_dev,
            "v5_2_1": candidate.build_classifier_messages_v5_2_1_dev}
SOURCE_FILES = (*prior.SOURCE_FILES, "refusal_coverage_citation_example.py", "refusal_v5_2_1_preparation.py")
PRICING = {"input_cny_per_1m_tokens": 3, "output_cny_per_1m_tokens": 18,
           "input_usd_per_1m_tokens": 0.4, "output_usd_per_1m_tokens": 2.4,
           "accounting_cny_per_usd": 7.5, "hard_cost_cap_cny": 3,
           "checked_date": "2026-10-13", "context_band": "0<Token<=256K",
           "source": "https://www.alibabacloud.com/help/en/model-studio/model-pricing",
           "caveat": "UTF-8 bytes + 512 framing is a planning proxy, not certified tokens or an invoice cap; fixed FX, no discounts"}


def build_preparation():
    shared, controls = prior.paired.shared, prior.controls
    canonical = controls.prep.prior.canonical_sha256
    cases = controls.load_cases()
    manifest = shared._read_json(controls.MANIFEST_PATH)
    if canonical(cases) != manifest["cases_canonical_sha256"]:
        raise RuntimeError("frozen development controls changed")
    rows = controls.classifier_rows(cases)
    arms = {}
    for arm, builder in BUILDERS.items():
        messages = [builder(shared._classifier_input_from_row(row)) for row in rows]
        bounds = [sum(len(m["content"].encode("utf-8")) for m in value) + 512 for value in messages]
        if max(bounds) + 2048 > 256_000:
            raise RuntimeError("pricing context band exceeded")
        arms[arm] = {
            "prompt_sha256": hashlib.sha256(messages[0][0]["content"].encode()).hexdigest(),
            "request_sha256": [canonical(value) for value in messages],
            "request_input_byte_bounds": bounds, "max_output_tokens": 2048, "max_calls": 14,
            "planning_cost_cny": prior.paired.cost({"prompt_tokens": sum(bounds),
                "completion_tokens": 14 * 2048}, {"pricing_snapshot": PRICING}),
        }
    total = sum(a["planning_cost_cny"] for a in arms.values())
    if total > PRICING["hard_cost_cap_cny"]:
        raise RuntimeError("proposed accounting budget exceeded")
    return {
        "run": "refusal_v5_2_1_controlled_regression_v1", "status": "PREPARED_NOT_RUN_NOT_AUTHORIZED",
        "cases": 14, "pairs": 7, "max_calls_total": 28, "provider_calls": 0,
        "case_order": [r["question_id"] for r in rows], "inputs_canonical_sha256": canonical(rows),
        "cases_canonical_sha256": canonical(cases), "arms": arms,
        "source_hashes": {name: hashlib.sha256((Path("experiments/evals") / name).read_text(
            encoding="utf-8").encode()).hexdigest() for name in SOURCE_FILES},
        "prior_result_canonical_sha256": canonical(shared._read_json(prior.REPORT / "v5_2_control_result.json")),
        "natural_freeze_canonical_sha256": canonical(shared._read_json(prior.REPORT / "v5_2_annotation_freeze.json")),
        "classifier": {"model": "qwen3.5-plus-2026-04-20", "region": "Singapore", "temperature": 0,
                       "enable_thinking": False, "response_format": "json_object"},
        "pricing_snapshot": PRICING, "planning_cost_cny": total,
        "order": "frozen case order; even index v5_2 then v5_2_1, odd index reversed",
        "checkpoint_root": "data/refusal_v5_2_1_controls/run_v1",
        "controls": {"authorization_required": True, "executable_run_contract": False,
                     "paid_runner_ready": False, "schema_failure_terminal_no_retry": True,
                     "unknown_charge_blocks_resume": True, "raw_before_parse_required": True,
                     "exclusive_local_lock_required": True, "successful_outcomes_never_repeated": True,
                     "targets_excluded_from_requests": True, "targets_outside_paid_loop_required": True,
                     "new_checkpoint_required": True, "dev_opened": False, "promotion_allowed": False},
        "metrics": ["all-14 format/provider failures per arm", "common-valid paired target agreement",
                    "new valid false admissions/refusals", "per-case transitions including null failures",
                    "tokens/cost/latency per arm"],
        "decision_rule": "Format-fix signal only if v5_2_1 has fewer all-case failures than concurrent v5_2 and no new valid false admissions/refusals; equality is inconclusive, never production approval",
        "provenance": "Observed fictional development regressions: seven correlated pairs, construction targets; not unseen validation, real-data accuracy, independent gold or natural TRAIN augmentation",
    }


if __name__ == "__main__":
    # ponytail: freeze an inert specification; reuse checkpoint machinery only after authorization.
    value = build_preparation()
    prior.controls.prep.prior._write_json(REPORT, value)
    print(json.dumps({k: value[k] for k in ("status", "max_calls_total", "planning_cost_cny", "provider_calls")}))
