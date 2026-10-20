"""Authorized additive draft amendment; original charged journals stay immutable."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_v5_3_1_runner as run

ROOT = run.DATA / "blind_drafts_v1_1"
CONTRACT = run.REPORT / "v5_3_1_draft_amendment_contract.json"
PROTOCOL = run.REPORT / "v5_3_1_draft_amendment.md"
STOP = run.REPORT / "v5_3_1_prospective_annotation_stop.json"
engine = run.isolate(run.prep.prior.annotation, "draft_outcomes")
strict_parse = engine.parse
write = run.prep.prior.selection.prior


def draft_outcome(raw, row, finish):
    # ponytail: only the draft envelope changes; retain the durable request engine.
    if not isinstance(raw, str) or finish not in {"stop", "length"}:
        raise ValueError("provider/incomplete response requires audit")
    try:
        parsed = strict_parse(raw, row, finish)
    except (ValueError, TypeError, KeyError):
        parsed = None
    return {"question_id": row["question_id"], "schema_valid": parsed is not None,
        "strict_annotation": parsed, "reviewed_label": False}


engine.parse = draft_outcome


def amendment_contract():
    old, rows = run.load_probe("annotation")
    stop = run.shared._read_json(STOP)
    original = run.ANNOTATION_ROOT
    if (run.canonical(old) != stop["annotation_contract_sha256"]
            or run.shared._sha256(original / "attempts.jsonl") != stop["journal_sha256"]
            or run.shared._sha256(original / "citation_recoveries.jsonl") != stop["citation_recoveries_sha256"]
            or run.shared._read_json(original / "identity.json") != {"contract_sha256": run.canonical(old)}
            or any((original / name).exists() for name in ("pending.json", "response.json", "runner.lock"))):
        raise RuntimeError("original charged stop changed or unresolved")
    amended = old | {"run": "prospective_v5_3_1_annotation_drafts_v1_1",
        "draft_amendment": {"authorization": "User explicitly permitted the proposed draft/review separation after PR100 merge; no additional calls or budget",
            "old_contract_sha256": run.canonical(old), "original_journal_sha256": stop["journal_sha256"],
            "original_recoveries_sha256": stop["citation_recoveries_sha256"], "stop_report_sha256": run.canonical(stop),
            "inherited_calls": 3, "inherited_cost_cny": .0748575,
            "source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest()}}
    return amended, rows


def prepare(*, authorization_confirmed=False):
    if authorization_confirmed is not True:
        raise RuntimeError("explicit amendment approval required")
    contract, rows = amendment_contract()
    events = run.shared._read_jsonl(run.ANNOTATION_ROOT / "attempts.jsonl")
    if len(events) != 3 or any(e["row_index"] != i or e["call_index"] != i or e["error_type"] is not None
            or e["usage_available"] is not True or e["finish_reason"] != "stop"
            or e["question_id"] != rows[i]["question_id"]
            or e["result"] != draft_outcome(e["raw_response"], rows[i], e["finish_reason"])["strict_annotation"]
            for i, e in enumerate(events)):
        raise RuntimeError("migration must preserve exactly three accounted charged drafts")
    if ROOT.exists() or CONTRACT.exists() or run.ROOT.exists() or run.FREEZE.exists() or run.CONTRACT.exists():
        raise RuntimeError("amendment state already exists or predictions/labels created")
    write._write_json(CONTRACT, contract)
    ROOT.mkdir()
    engine.write_json(ROOT / "identity.json", {"contract_sha256": run.canonical(contract)})
    write._write_jsonl(ROOT / "attempts.jsonl", [e | {
        "result": draft_outcome(e["raw_response"], rows[i], e["finish_reason"])} for i, e in enumerate(events)])
    return run_drafts(max_new_calls=0)


def load_probe():
    expected, rows = amendment_contract()
    contract = run.shared._read_json(CONTRACT)
    if contract != expected:
        raise RuntimeError("amendment contract changed")
    original = run.shared._read_jsonl(run.ANNOTATION_ROOT / "attempts.jsonl")
    events = run.shared._read_jsonl(ROOT / "attempts.jsonl")
    derived = [e | {"result": draft_outcome(e["raw_response"], rows[i], e["finish_reason"])} for i, e in enumerate(original)]
    if events[:3] != derived or (ROOT / "citation_recoveries.jsonl").exists():
        raise RuntimeError("charged prefix migration changed")
    if any(e["error_type"] is not None for e in events):
        raise RuntimeError("provider exception saved; audit before continuation")
    return contract, rows


def run_drafts(*, client=None, max_new_calls=47):
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 47:
        raise ValueError("remaining annotation call cap is 0..47")
    if max_new_calls and client is None:
        raise RuntimeError("explicit bounded client required")
    contract, rows = load_probe()
    return engine.run(contract, rows, ROOT, client=client, max_new_calls=max_new_calls)


def freeze_review(path):
    if run.ROOT.exists() or run.CONTRACT.exists():
        raise RuntimeError("classifier exists before label freeze")
    state = run_drafts(max_new_calls=0)
    if state["completed"] != 50 or state["calls"] != 50:
        raise RuntimeError("all50 accounted drafts required")
    contract, rows = load_probe()
    events = run.shared._read_jsonl(ROOT / "attempts.jsonl")
    review = run.shared._read_jsonl(path)
    if [r["question_id"] for r in review] != [r["question_id"] for r in rows]:
        raise RuntimeError("complete ordered evidence review required")
    changes = Counter()
    for row, event, reviewed in zip(rows, events, review, strict=True):
        if set(reviewed) != {"question_id", "annotation"} or strict_parse(
                json.dumps({"annotation": reviewed["annotation"]}), row, "stop") != reviewed:
            raise RuntimeError("invalid reviewed annotation")
        try:
            raw_annotation = json.loads(event["raw_response"])["annotation"]
        except (ValueError, KeyError, TypeError):
            raw_annotation = {}
        for field in ("target_class", "supporting_source_ids", "notes"):
            changes[field] += raw_annotation.get(field) != reviewed["annotation"][field]
    destinations = (run.DATA / "reviewed_annotations.jsonl", run.DATA / "targets.jsonl", run.FREEZE)
    if any(p.exists() for p in destinations):
        raise RuntimeError("refusing to overwrite label freeze")
    targets = [{"question_id": r["question_id"], "target_class": r["annotation"]["target_class"]} for r in review]
    write._write_jsonl(destinations[0], review)
    write._write_jsonl(destinations[1], targets)
    counts = Counter(t["target_class"] for t in targets)
    value = {"status": "AI_PROXY_TARGETS_FROZEN_BEFORE_CLASSIFIER", "cases": 50,
        "counts": {c: counts[c] for c in ("SUFFICIENT", "INSUFFICIENT", "AMBIGUOUS")},
        "changes_from_raw_draft": dict(changes), "provider_calls": 50,
        "estimated_annotation_cost_cny": state["estimated_cost_cny"],
        "usage": {k: sum(e[k] for e in events) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "original_schema_failures": sum(not e["result"]["schema_valid"] for e in events),
        "inputs_sha256": contract["inputs_sha256"], "targets_sha256": run.shared._sha256(destinations[1]),
        "review_sha256": run.shared._sha256(destinations[0]), "journal_sha256": run.shared._sha256(ROOT / "attempts.jsonl"),
        "amendment_contract_sha256": run.canonical(contract),
        "original_journal_sha256": contract["draft_amendment"]["original_journal_sha256"],
        "review_depth": "Current assistant reviewed every question/raw draft and original evidence before predictions; not independent expert adjudication",
        "class_support_at_least_five_each": all(counts[c] >= 5 for c in ("SUFFICIENT", "INSUFFICIENT")),
        "classifier_predictions_created": False, "dev_opened": False, "promotion_allowed": False,
        "provenance": run.read_plan()["provenance"]}
    write._write_json(run.FREEZE, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "annotate", "freeze-review"))
    parser.add_argument("--confirm-amendment", action="store_true")
    parser.add_argument("--env-file")
    parser.add_argument("--review-file")
    parser.add_argument("--max-new-calls", type=int, default=47)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            result = prepare(authorization_confirmed=args.confirm_amendment)
        elif args.mode == "freeze-review":
            if not args.review_file:
                raise RuntimeError("explicit review file required")
            result = freeze_review(args.review_file)
        else:
            if not args.env_file:
                raise RuntimeError("explicit credential file required")
            contract, _ = load_probe()
            with run.bounded_client("annotation", contract, args.env_file) as client:
                result = run_drafts(client=client, max_new_calls=args.max_new_calls)
        print(json.dumps(result))
    except Exception as error:
        print(f"Stopped ({type(error).__name__}); audit saved state; no automatic retry.")
        raise SystemExit(1) from None
