"""Offline controlled-case preparation, not a model score or TRAIN augmentation."""

import json
import re
from collections import Counter

from experiments.evals import refusal_v5_2_preparation as prep

CASE_PATH = prep.REPORT.parent / "v5_2_control_cases.jsonl"
MANIFEST_PATH = prep.REPORT.parent / "v5_2_control_preparation.json"
FAMILIES = {"steps", "applicability", "version", "time", "condition", "conflict", "negative_answer"}


def validate_cases(rows):
    if len(rows) != 14:
        raise ValueError("exactly 14 fixed controls required")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "question_id", "family", "expected_class", "rationale", "question", "sources"
        }:
            raise ValueError("control schema mismatch")
        if (not isinstance(row["question_id"], str) or not re.fullmatch(r"CONTROL_[A-Z_]+", row["question_id"])
                or not isinstance(row["family"], str) or row["family"] not in FAMILIES
                or not isinstance(row["expected_class"], str) or row["expected_class"] not in {"SUFFICIENT", "INSUFFICIENT"}
                or any(not isinstance(row[k], str) or not row[k].strip() for k in ("question", "rationale"))):
            raise ValueError("invalid control identity/class/text")
        sources = row["sources"]
        if (not isinstance(sources, list) or len(sources) != 2 or any(
            not isinstance(s, dict) or set(s) != {"source_id", "content"}
            or s["source_id"] != f"Source {i}" or not isinstance(s["content"], str) or not s["content"].strip()
            for i, s in enumerate(sources, 1)
        )):
            raise ValueError("two ordered text-only sources required")
        ids.append(row["question_id"])
    if len(set(ids)) != 14 or Counter(r["family"] for r in rows) != Counter({f: 2 for f in FAMILIES}):
        raise ValueError("duplicate ID or incomplete family pairs")
    for family in FAMILIES:
        a, b = [r for r in rows if r["family"] == family]
        if (a["question"] != b["question"] or {a["expected_class"], b["expected_class"]} != {"SUFFICIENT", "INSUFFICIENT"}
                or sum(x != y for x, y in zip(a["sources"], b["sources"], strict=True)) != 1):
            raise ValueError("each family needs identical question, one changed source and opposite targets")
    return rows


def load_cases():
    return validate_cases(prep.prior.reuse._read_jsonl(CASE_PATH))


def classifier_rows(rows):
    validate_cases(rows)
    # ponytail: explicit projection, no new labeling framework or mining more TRAIN.
    return [{key: row[key] for key in ("question_id", "question", "sources")} for row in rows]


def prepare():
    rows = load_cases()
    preflight = prep.paired_preflight(classifier_rows(rows))
    natural = json.loads((prep.REPORT.parent / "v5_2_annotation_freeze.json").read_text(encoding="utf-8"))
    result = {
        "status": "CONTROLLED_CASES_FROZEN_NOT_RUN_NOT_AUTHORIZED", "cases": 14, "pairs": 7,
        "case_order": [r["question_id"] for r in rows],
        "cases_canonical_sha256": prep.prior.canonical_sha256(rows),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "expected_class_counts": dict(Counter(r["expected_class"] for r in rows)),
        "provenance": "Fictional human-readable fixtures authored by current assistant; expected targets follow construction, not blind independent gold",
        "relation_to_natural_train": "Separate controlled stress diagnostic; never added to natural class counts or quality denominators",
        "natural_freeze_canonical_sha256": prep.prior.canonical_sha256(natural),
        "natural_insufficient_cases_unchanged": natural["counts"]["INSUFFICIENT"],
        "natural_class_support_met_unchanged": natural["class_support_at_least_five_each"],
        "no_retrieval_top14_claim": True, "provider_calls": 0, "dev_opened": False,
        "promotion_allowed": False, "paid_runner_ready": False, "paired_preflight": preflight,
    }
    prep.prior._write_json(MANIFEST_PATH, result)
    return {"status": result["status"], "cases": 14, "planned_calls": preflight["planned_calls"],
            "planning_cost_usd": preflight["planning_cost_usd"], "provider_calls": 0}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
