"""Select new refusal TRAIN diagnostics from frozen artifacts; no model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_v2_confirmation_set as reuse
from experiments.evals.refusal_v2_ai_proxy import _write_jsonl

REPORT_DIR = Path("experiments/evals/reports/refusal_evidence_sufficiency")
DATA_DIR = Path("data/refusal_v5_train")
SEED = "refusal-v5-new-train-diagnostic-v1"
CASES = 20
SELECTION_HASHES = {
    "v2_confirmation_selection.json": "721779d07ade0197c026febb735070c2bd1b4838e3de90ebb977eed16fa4b8da",
    "v3_confirmation_selection.json": "dfbddbe256d58fdad00d7ba0f13f719c1d829d75742ba5c8d54a85b075205118",
}


def select_packets(snapshots, results, excluded_ids, *, cases=CASES):
    snapshots, results = reuse._index(snapshots), reuse._index(results)
    if set(snapshots) != set(results) or any(not qid.startswith("TRAIN_Q") for qid in results):
        raise RuntimeError("matching TRAIN-only frozen artifacts required")
    eligible = set(results) - excluded_ids
    if len(eligible) < cases:
        raise RuntimeError("not enough unselected TRAIN cases")
    selected = sorted(sorted(eligible, key=lambda qid: reuse._selection_hash(qid, seed=SEED))[:cases])
    packets = []
    for qid in selected:
        snapshot, result = snapshots[qid], results[qid]
        question = snapshot["question"]
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError("empty question")
        candidates = {c["chunk_id"]: c for c in snapshot["fused_candidates"]}
        chunk_ids, doc_ids = result["reranked_chunk_ids"][:14], result["reranked_document_ids"][:14]
        if len(chunk_ids) != 14 or len(doc_ids) != 14 or len(set(chunk_ids)) != 14:
            raise RuntimeError("expected unique frozen Top14")
        sources = []
        for rank, (cid, did) in enumerate(zip(chunk_ids, doc_ids, strict=True), 1):
            candidate = candidates.get(cid)
            if candidate is None or candidate["document_id"] != did:
                raise RuntimeError("reranked identity differs from snapshot")
            content = candidate["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("empty source text")
            sources.append({"source_id": f"Source {rank}", "content": content})
        packets.append({"question_id": qid, "question": question, "sources": sources,
                        "annotation": {"target_class": None, "supporting_source_ids": [], "notes": ""}})
    return packets, len(eligible)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(snapshot_path, results_path):
    sources = {Path(snapshot_path): reuse.EXPECTED_SNAPSHOT_SHA256,
               Path(results_path): reuse.EXPECTED_RESULTS_SHA256,
               reuse.DEFAULT_PRIOR_LABELS_PATH: reuse.EXPECTED_PRIOR_LABELS_SHA256,
               **{REPORT_DIR / name: digest for name, digest in SELECTION_HASHES.items()}}
    for path, expected in sources.items():
        if reuse._sha256(path) != expected:
            raise RuntimeError(f"frozen source SHA mismatch: {path}")
    # ponytail: exclude all earlier packets, not just the 50 paid cases; no gold metadata read.
    excluded = {r["question_id"] for r in reuse._read_jsonl(reuse.DEFAULT_PRIOR_LABELS_PATH)}
    for name in SELECTION_HASHES:
        selection = json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))
        ids = [r["question_id"] for r in selection["selected_cases"]]
        if len(ids) != 80 or len(set(ids)) != 80:
            raise RuntimeError("historical selection population changed")
        excluded.update(ids)
    snapshots, results = reuse._read_jsonl(snapshot_path), reuse._read_jsonl(results_path)
    if len(snapshots) != 450 or len(results) != 450:
        raise RuntimeError("frozen TRAIN population changed")
    packets, population = select_packets(snapshots, results, excluded)
    packet_path, inputs_path = DATA_DIR / "annotation_packets.jsonl", DATA_DIR / "inputs.jsonl"
    selection_path = REPORT_DIR / "v5_train_selection.json"
    if any(path.exists() for path in (packet_path, inputs_path, selection_path)):
        raise RuntimeError("refusing to overwrite selection outputs")
    _write_jsonl(packet_path, packets)
    _write_jsonl(inputs_path, [{k: v for k, v in p.items() if k != "annotation"} for p in packets])
    manifest = {
        "run": "refusal_v5_new_train_diagnostic_v1", "status": "SELECTED_NOT_ANNOTATED",
        "seed": SEED, "algorithm": "lowest SHA256(seed:question_id), then question-ID order",
        "cases": CASES, "eligible_cases": population, "excluded_question_ids": sorted(excluded),
        "selected_question_ids": [p["question_id"] for p in packets],
        "source_hashes": {"snapshot": reuse.EXPECTED_SNAPSHOT_SHA256,
                          "results": reuse.EXPECTED_RESULTS_SHA256,
                          "prior_audit": reuse.EXPECTED_PRIOR_LABELS_SHA256,
                          **SELECTION_HASHES},
        "packet_sha256": reuse._sha256(packet_path), "inputs_sha256": reuse._sha256(inputs_path),
        "provenance": "New relative to prior refusal audit/v2/v3 selections; not independent of project research or reviewer",
        "controls": {"provider_calls": 0, "dev_opened": False, "gold_metadata_opened": False,
                     "class_balancing_or_replacement_after_annotation": False, "promotion_allowed": False},
    }
    _write_json(selection_path, manifest)
    return manifest


def freeze_annotations(annotated_path):
    selection = json.loads((REPORT_DIR / "v5_train_selection.json").read_text(encoding="utf-8"))
    packet_path = DATA_DIR / "annotation_packets.jsonl"
    if reuse._sha256(packet_path) != selection["packet_sha256"]:
        raise RuntimeError("annotation packet SHA mismatch")
    packets, annotations = reuse._read_jsonl(packet_path), reuse._read_jsonl(annotated_path)
    if [r["question_id"] for r in annotations] != selection["selected_question_ids"] or any(
        set(r) != {"question_id", "annotation"} for r in annotations
    ):
        raise RuntimeError("annotation IDs/order or compact schema changed")
    annotated_packets = [p | {"annotation": a["annotation"]}
                         for p, a in zip(packets, annotations, strict=True)]
    output_path = DATA_DIR / "annotation_packets_annotated_ai_draft.jsonl"
    if output_path.exists():
        raise RuntimeError("refusing to overwrite annotated packet")
    report = reuse.validate_and_freeze_annotations(
        packets, annotated_packets,
        annotation_source="ai-draft", minimum_per_class=5,
        targets_path=REPORT_DIR / "v5_train_targets.jsonl",
        freeze_path=REPORT_DIR / "v5_train_annotation_freeze.json",
        run="refusal_v5_current_assistant_ai_annotation_v1",
    )
    _write_jsonl(output_path, annotated_packets)
    return report


def prepare_contract():
    from experiments.evals.refusal_development_policy import (
        CLASSIFIER_SYSTEM_PROMPT_V4_DEV, build_classifier_messages_v4_dev,
    )
    from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
    from experiments.evals.refusal_structured_evidence import (
        CLASSIFIER_SYSTEM_PROMPT_V5_DEV, build_classifier_messages_v5_dev,
    )
    selection = json.loads((REPORT_DIR / "v5_train_selection.json").read_text(encoding="utf-8"))
    inputs_path, targets_path = DATA_DIR / "inputs.jsonl", REPORT_DIR / "v5_train_targets.jsonl"
    freeze_path = REPORT_DIR / "v5_train_annotation_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if reuse._sha256(inputs_path) != selection["inputs_sha256"] or (
        reuse._sha256(targets_path) != freeze["targets"]["sha256"]
    ):
        raise RuntimeError("frozen input/target SHA mismatch")
    rows = reuse._read_jsonl(inputs_path)
    if len(rows) != CASES or [r["question_id"] for r in rows] != selection["selected_question_ids"]:
        raise RuntimeError("frozen input population changed")
    arms = {}
    for name, prompt, builder, output_tokens in (
        ("v4", CLASSIFIER_SYSTEM_PROMPT_V4_DEV, build_classifier_messages_v4_dev, 512),
        ("v5", CLASSIFIER_SYSTEM_PROMPT_V5_DEV, build_classifier_messages_v5_dev, 2048),
    ):
        bounds = []
        for row in rows:
            messages = builder(ClassifierInput(row["question"], tuple(
                ContextSource(s["source_id"], "", "", s["content"]) for s in row["sources"])))
            bounds.append(sum(len(m["content"].encode("utf-8")) for m in messages) + 512)
        if max(bounds) > 256_000:
            raise RuntimeError("request exceeds pricing context band")
        arms[name] = {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_output_tokens": output_tokens, "required_predictions": CASES, "max_calls": CASES + 2,
            "output_dir": f"data/refusal_v5_train/paired_run_v1/{name}",
            "request_input_byte_bounds": bounds,
            "conservative_cost_bound_cny": (
                sum(bounds) * 3 + CASES * output_tokens * 18
                + 2 * (max(bounds) * 3 + output_tokens * 18)
            ) / 1_000_000,
        }
    total_bound = sum(a["conservative_cost_bound_cny"] for a in arms.values())
    if total_bound > 3:
        raise RuntimeError("preregistered maximum exceeds proposed accounting cap")
    contract = {
        "run": "refusal_v5_paired_train_diagnostic_v1", "status": "PREREGISTERED_NOT_RUN",
        "amendment": {"version": "1.1", "reason": "Git normalizes tracked JSON line endings; pin canonical JSON",
                      "supersedes_canonical_sha256": "bd0edf1e130bb625f93f078098382c5c896349e66dd7aca744ea19214a2bb76d"},
        "cases": CASES, "frozen_inputs": {
            "inputs_path": inputs_path.as_posix(), "inputs_sha256": reuse._sha256(inputs_path),
            "targets_path": targets_path.as_posix(), "targets_sha256": reuse._sha256(targets_path),
            "annotation_freeze_canonical_sha256": canonical_sha256(freeze),
            "selection_canonical_sha256": canonical_sha256(selection),
        },
        "classifier": {"provider": "Alibaba Cloud Model Studio", "region": "Singapore",
                       "model": "qwen3.5-plus-2026-04-20", "temperature": 0,
                       "enable_thinking": False, "response_format": "json_object"},
        "arms": arms, "required_predictions_total": 40, "max_calls_total": 44,
        "order": "question-ID order; even index v4 then v5, odd index v5 then v4",
        "v5_validator_normalized_source_sha256": hashlib.sha256(
            Path("experiments/evals/refusal_structured_evidence.py").read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "pricing_snapshot": {
            "input_usd_per_1m_tokens": 0.4, "output_usd_per_1m_tokens": 2.4,
            "accounting_cny_per_usd": 7.5, "input_cny_per_1m_tokens": 3,
            "output_cny_per_1m_tokens": 18, "hard_cost_cap_cny": 3,
            "conservative_cost_bound_cny": total_bound,
            "source": "https://www.alibabacloud.com/help/en/model-studio/model-pricing",
            "checked_date": "2026-10-06", "context_band": "0<Token<=256K",
            "caveat": "Fixed accounting conversion, not spot FX or invoice guarantee; no discounts assumed",
        },
        "label_provenance": "CURRENT_ASSISTANT_AI_PROXY_NOT_INDEPENDENT_NOT_HUMAN_GOLD",
        "metrics": ["all-20 format/provider failures per arm", "confusion on common valid pairs",
                    "sufficient recall", "insufficient recall", "paired error transitions",
                    "raw model admission vs final gate admission", "latency/tokens/cost per arm"],
        "controls": {"provider_calls_this_stage": 0, "authorization_required": True,
                     "runner_ready": False, "checkpoint_each_attempt": True,
                     "max_failed_attempts_per_arm": 2, "successful_predictions_never_retried": True,
                     "unknown_usage_stops_resume": True, "provider_or_schema_error_stops_immediately": True,
                     "save_raw_response_and_v5_checks": True, "targets_not_loaded_by_paid_loop": True,
                     "dev_opened": False, "promotion_allowed": False, "formal_pass_gate": None},
    }
    _write_json(REPORT_DIR / "v5_train_run_contract_v1_1.json", contract)
    return {"status": contract["status"], "cases": CASES, "required_calls": 40, "max_calls": 44,
            "hard_cost_cap_cny": 3, "conservative_cost_bound_cny": total_bound,
            "contract_canonical_sha256": canonical_sha256(contract)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path")
    parser.add_argument("--results-path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-annotations")
    mode.add_argument("--prepare-contract", action="store_true")
    args = parser.parse_args()
    if args.prepare_contract:
        result = prepare_contract()
    elif args.freeze_annotations:
        result = freeze_annotations(args.freeze_annotations)
    else:
        if not args.snapshot_path or not args.results_path:
            parser.error("explicit frozen TRAIN snapshot and result paths required")
        result = prepare(args.snapshot_path, args.results_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
