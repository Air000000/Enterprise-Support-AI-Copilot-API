"""PR80 annotation readiness and preprediction AI review; explicit approval required."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from experiments.evals import refusal_prospective_preparation as prep

shared, canonical = prep.shared, prep.canonical
DATA, ROOT = prep.DATA, prep.DATA / "blind_annotation"
CONTRACT = prep.selection.prior.REPORT_DIR / "prospective_annotation_contract.json"
FREEZE = prep.selection.prior.REPORT_DIR / "prospective_annotation_freeze.json"
PREPARATION_SHA256 = "e093d72579836ecc13f10d64e63a2e62e2a76dbd3bd4b193696427de61edb8b8"
AUTHORIZATION = "Requires explicit user approval of PR80: 50 full-Top14 Singapore qwen3.6-plus-2026-04-02 annotation + 100 qwen3.5-plus-2026-04-20 paired classification calls / CNY25; annotation cap50/CNY10, classifier cap100/CNY15, no paid retries, DEV, recharge or promotion. A merge-only/continue reply does not establish this approval."

# ponytail: use the frozen durable annotator in an isolated namespace, no new client.
_spec = importlib.util.spec_from_file_location("_prospective_annotation_engine", prep.annotation.__file__)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)


def load_inputs():
    manifest = shared._read_json(prep.REPORT)
    if canonical(manifest) != PREPARATION_SHA256:
        raise RuntimeError("frozen preparation changed")
    for name, digest in manifest["source_hashes"].items():
        if hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() != digest:
            raise RuntimeError("frozen source changed")
    if hashlib.sha256(prep.PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest() != manifest["protocol_normalized_sha256"]:
        raise RuntimeError("frozen protocol changed")
    rows = prep.paired.prior.paired.v2.load_classifier_inputs(DATA / "inputs.jsonl", expected_sha256=manifest["inputs_sha256"], expected_cases=50)
    if [row["question_id"] for row in rows] != manifest["selected_question_ids"]:
        raise RuntimeError("frozen order changed")
    arm = manifest["preflight"]["arms"]["annotation"]
    for index, row in enumerate(rows):
        messages = prep.annotation.messages(row)
        if (canonical(messages) != arm["request_sha256"][index]
                or sum(len(m["content"].encode()) for m in messages) + 512 != arm["input_byte_plus_framing_bounds"][index]):
            raise RuntimeError("annotation request changed")
    return manifest, rows


def execution_contract(manifest):
    arm = manifest["preflight"]["arms"]["annotation"]
    return {"run": "prospective_refusal_annotation_v1", "preparation_canonical_sha256": canonical(manifest),
        "model": prep.annotation.MODEL, "region": "Singapore", "cases": 50, "max_calls": 50,
        "max_output_tokens": 4096, "hard_accounting_cap_cny": 10,
        "input_cny_per_million": 3.75, "output_cny_per_million": 22.5,
        "inputs_sha256": manifest["inputs_sha256"], "prompt_sha256": arm["prompt_sha256"],
        "request_sha256": arm["request_sha256"], "input_byte_bounds": arm["input_byte_plus_framing_bounds"],
        "runner_normalized_source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "pricing": manifest["preflight"]["pricing"], "planning_cost_cny": arm["planning_cost_cny"],
        "authorization": AUTHORIZATION, "controls": {"sdk_retries": 0, "paid_retries": False,
            "raw_before_parse": True, "unknown_charge_blocks_resume": True, "dev_opened": False,
            "promotion_allowed": False, "review_and_target_freeze_before_classifier": True}}


def prepare_contract(*, authorization_confirmed=False):
    if authorization_confirmed is not True:
        raise RuntimeError("explicit PR80 data/model/budget authorization required")
    manifest, _ = load_inputs()
    value = execution_contract(manifest)
    prep.selection.prior._write_json(CONTRACT, value)
    return value


def load_probe():
    manifest, rows = load_inputs()
    contract = shared._read_json(CONTRACT)
    if contract != execution_contract(manifest):
        raise RuntimeError("frozen annotation contract changed")
    return contract, rows


def run_annotation(*, root=ROOT, client=None, max_new_calls=50, probe_loader=load_probe):
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 50:
        raise ValueError("max_new_calls must be between 0 and 50")
    contract, rows = probe_loader()
    # No resume-after-audit switch: only a validated saved-response recovery can advance a failed row.
    return _engine.run(contract, rows, root, client=client, max_new_calls=max_new_calls)


def accounted_draft(*, root=ROOT):
    state = run_annotation(root=root, max_new_calls=0)
    if state["completed"] != 50 or state["calls"] != 50 or (Path(root) / "runner.lock").exists():
        raise RuntimeError("all50 stopped, accounted annotation outcomes required")
    _, rows = load_probe()
    events = shared._read_jsonl(Path(root) / "attempts.jsonl")
    path = Path(root) / "citation_recoveries.jsonl"
    recoveries = {r["call_index"]: r["annotation"] for r in shared._read_jsonl(path)} if path.exists() else {}
    draft = [{"question_id": row["question_id"], "annotation": recoveries[index] if index in recoveries else event["result"]["annotation"]}
             for index, (row, event) in enumerate(zip(rows, events, strict=True))]
    return state, events, rows, draft


def export_draft():
    _, _, _, draft = accounted_draft()
    prep.selection.prior._write_jsonl(DATA / "blind_draft_annotations.jsonl", draft)
    return len(draft)


def freeze_review(review_path, *, root=ROOT):
    if (DATA / "paired_run_v1").exists():
        raise RuntimeError("classifier checkpoint exists before review freeze")
    state, events, rows, draft = accounted_draft(root=root)
    if shared._read_jsonl(DATA / "blind_draft_annotations.jsonl") != draft:
        raise RuntimeError("frozen draft differs from accounted raw responses")
    review = shared._read_jsonl(review_path)
    changed = prep.selection.validate_review(rows, draft, review)
    targets = [{"question_id": r["question_id"], "target_class": r["annotation"]["target_class"]} for r in review]
    path = DATA / "targets.jsonl"
    if FREEZE.exists() or path.exists() or (DATA / "reviewed_annotations.jsonl").exists():
        raise RuntimeError("refusing to overwrite target/review freeze")
    prep.selection.prior._write_jsonl(DATA / "reviewed_annotations.jsonl", review)
    prep.selection.prior._write_jsonl(path, targets)
    counts = Counter(t["target_class"] for t in targets)
    recovery_path = Path(root) / "citation_recoveries.jsonl"
    value = {"status": "AI_PROXY_TARGETS_FROZEN_BEFORE_CLASSIFIER", "cases": 50,
        "counts": {c: counts[c] for c in ("SUFFICIENT", "INSUFFICIENT", "AMBIGUOUS")},
        "class_changes_from_draft": changed, "provider_calls": len(events),
        "estimated_annotation_cost_cny": state["estimated_cost_cny"],
        "prompt_tokens": sum(e["prompt_tokens"] for e in events), "completion_tokens": sum(e["completion_tokens"] for e in events),
        "original_schema_failures": sum(e["result"] is None for e in events),
        "contract_canonical_sha256": canonical(shared._read_json(CONTRACT)),
        "inputs_sha256": shared._sha256(DATA / "inputs.jsonl"), "targets_sha256": shared._sha256(path),
        "review_sha256": shared._sha256(DATA / "reviewed_annotations.jsonl"),
        "draft_sha256": shared._sha256(DATA / "blind_draft_annotations.jsonl"),
        "journal_sha256": shared._sha256(Path(root) / "attempts.jsonl"),
        "citation_recoveries_sha256": shared._sha256(recovery_path) if recovery_path.exists() else None,
        "review_depth": "Current assistant reviewed every question/draft and cited evidence with targeted fuller context; not an independent full-source expert adjudication",
        "class_support_at_least_five_each": all(counts[c] >= 5 for c in ("SUFFICIENT", "INSUFFICIENT")),
        "classifier_predictions_created": False, "dev_opened": False, "promotion_allowed": False,
        "provenance": "Same-family Qwen draft plus current-assistant preprediction review: AI proxy, not independent human/expert gold; all TRAIN previously used in retrieval/analysis"}
    prep.selection.prior._write_json(FREEZE, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-contract", action="store_true")
    mode.add_argument("--run-paid", action="store_true")
    mode.add_argument("--export-draft", action="store_true")
    mode.add_argument("--freeze-review")
    parser.add_argument("--env-file")
    parser.add_argument("--max-new-calls", type=int, default=50)
    parser.add_argument("--confirm-pr80-authorization", action="store_true",
                        help="Record an already-obtained explicit user approval; this flag is not permission itself")
    args = parser.parse_args()
    if args.prepare_contract:
        print(json.dumps({"max_calls": prepare_contract(authorization_confirmed=args.confirm_pr80_authorization)["max_calls"]}))
    elif args.run_paid:
        if args.env_file:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        print(json.dumps(run_annotation(max_new_calls=args.max_new_calls)))
    elif args.export_draft:
        print(json.dumps({"exported": export_draft()}))
    else:
        print(json.dumps(freeze_review(args.freeze_review)))
