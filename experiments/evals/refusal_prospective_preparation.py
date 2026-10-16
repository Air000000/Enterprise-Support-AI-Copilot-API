"""Freeze fresh-to-refusal TRAIN inputs and an inert two-stage budget; no providers."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from experiments.evals import refusal_v5_2_annotation as annotation
from experiments.evals import refusal_v5_2_preparation as selection
from experiments.evals import refusal_v5_3_preparation as paired

CASES, SEED = 50, "refusal-prospective-train-50-v1"
PREVIOUS_SELECTION_SHA256 = "cdf6742bfc6b65625eec9eef37b04b773b3a3f2975c6fde826ae378ff762fd0d"
DATA = Path("data/refusal_prospective_train")
REPORT = selection.prior.REPORT_DIR / "prospective_train_preparation.json"
PROTOCOL = selection.prior.REPORT_DIR / "prospective_train_protocol.md"
canonical, shared = paired.canonical, annotation.shared
SOURCE_FILES = (*paired.SOURCE_FILES, "refusal_v5_2_annotation.py", "refusal_v2_confirmation_set.py",
                "refusal_v2_ai_proxy.py", "refusal_prospective_preparation.py")


def select_new_packets(snapshots, results, excluded, *, cases=CASES):
    if type(cases) is not int or cases < 1:
        raise ValueError("positive integer case count required")
    snapshot_index, result_index = [selection.prior.reuse._index(rows) for rows in (snapshots, results)]
    if set(snapshot_index) != set(result_index) or any(not re.fullmatch(r"TRAIN_Q\d{3}", qid) for qid in snapshot_index):
        raise RuntimeError("matching TRAIN-only frozen population required")
    family = selection.family_exclusions(snapshots, results, excluded)
    eligible = set(snapshot_index) - excluded - family
    chosen, questions, documents = [], set(), set()
    # ponytail: reuse the visible family proxy, greedily deduplicate before any labels exist.
    for qid in sorted(eligible, key=lambda q: selection.prior.reuse._selection_hash(q, seed=SEED)):
        question = " ".join(snapshot_index[qid]["question"].casefold().split())
        document = result_index[qid]["reranked_document_ids"][0]
        if question in questions or document in documents:
            continue
        chosen.append(qid)
        questions.add(question)
        documents.add(document)
        if len(chosen) == cases:
            break
    if len(chosen) != cases:
        raise RuntimeError("not enough fresh family-proxy groups; no smaller fallback")
    packets, _ = selection.prior.select_packets(snapshots, results, set(snapshot_index) - set(chosen), cases=cases, seed=SEED)
    return packets, {"eligible_before_within_batch_dedup": len(eligible),
                     "excluded_question_ids_canonical_sha256": canonical(sorted(excluded)),
                     "family_proxy_excluded_ids_canonical_sha256": canonical(sorted(family)),
                     "family_proxy_excluded_count": len(family)}


def preflight(rows):
    if len(rows) != CASES:
        raise RuntimeError("complete frozen 50-case population required")
    arms = {}
    builders = {"annotation": annotation.messages, **{
        arm: lambda row, builder=builder: builder(shared._classifier_input_from_row(row))
        for arm, builder in paired.BUILDERS.items()}}
    for arm, builder in builders.items():
        messages = [builder(row) for row in rows]
        bounds = [sum(len(m["content"].encode("utf-8")) for m in value) + 512 for value in messages]
        output, input_rate, output_rate = (4096, 3.75, 22.5) if arm == "annotation" else (2048, 3, 18)
        if max(bounds) + output > 256_000:
            raise RuntimeError("pricing context band exceeded")
        arms[arm] = {"prompt_sha256": hashlib.sha256(messages[0][0]["content"].encode()).hexdigest(),
                     "request_sha256": [canonical(value) for value in messages],
                     "input_byte_plus_framing_bounds": bounds, "max_output_tokens": output, "max_calls": CASES,
                     "planning_cost_cny": (sum(bounds) * input_rate + CASES * output * output_rate) / 1_000_000}
    costs = {"annotation": arms["annotation"]["planning_cost_cny"],
             "paired_classification": sum(arms[a]["planning_cost_cny"] for a in paired.BUILDERS)}
    if costs["annotation"] > 10 or costs["paired_classification"] > 15:
        raise RuntimeError("proposed stage budgets exceeded")
    return {"arms": arms, "stage_planning_cost_cny": costs, "planning_cost_cny": sum(costs.values()),
            "max_calls_total": 150, "stage_caps_cny": {"annotation": 10, "paired_classification": 15},
            "hard_accounting_cap_cny": 25, "region": "Singapore", "temperature": 0,
            "enable_thinking": False, "response_format": "json_object",
            "models": {"annotation": annotation.MODEL, "paired_classification": "qwen3.5-plus-2026-04-20"},
            "pricing": {"checked_date": "2026-10-16", "accounting_cny_per_usd": 7.5,
                        "annotation_usd_per_million_input_output": [0.5, 3],
                        "classification_usd_per_million_input_output": [0.4, 2.4],
                        "source": "https://www.alibabacloud.com/help/en/model-studio/model-pricing",
                        "caveat": "UTF-8 bytes + 512 framing and fixed FX are planning/accounting proxies, not certified tokens or a guaranteed invoice cap; no discounts"}}


def prepare(research_root):
    root = Path(research_root) / "experiments/evals/reports/r4_c1_hybrid_rerank"
    paths = [root / name for name in ("train_fused_snapshot.jsonl", "train_results.jsonl")]
    for path, digest in zip(paths, (selection.prior.reuse.EXPECTED_SNAPSHOT_SHA256, selection.prior.reuse.EXPECTED_RESULTS_SHA256), strict=True):
        if shared._sha256(path) != digest:
            raise RuntimeError("frozen TRAIN artifact changed")
    old = shared._read_json(selection.REPORT)
    if canonical(old) != PREVIOUS_SELECTION_SHA256:
        raise RuntimeError("previous frozen selection changed")
    excluded = set(old["recorded_excluded_question_ids"] + old["family_proxy_excluded_question_ids"] + old["selected_question_ids"])
    snapshots = shared._read_jsonl(paths[0])
    results = [{key: row[key] for key in ("question_id", "reranked_chunk_ids", "reranked_document_ids")}
               for row in shared._read_jsonl(paths[1])]
    if len(snapshots) != 450 or len(results) != 450:
        raise RuntimeError("frozen TRAIN population changed")
    packets, population = select_new_packets(snapshots, results, excluded)
    rows = [{k: v for k, v in packet.items() if k != "annotation"} for packet in packets]
    budget = preflight(rows)
    outputs = (DATA / "annotation_packets.jsonl", DATA / "inputs.jsonl", REPORT)
    if any(path.exists() for path in outputs):
        raise RuntimeError("refusing to overwrite frozen preparation")
    manifest = {"run": "prospective_refusal_train_50_v1", "status": "SELECTED_NOT_ANNOTATED_NOT_AUTHORIZED",
        "cases": CASES, "seed": SEED, "selection_algorithm": "SHA256(seed:question_id) priority; skip normalized-question/leading-document collisions; final question-ID order",
        "selected_question_ids": [row["question_id"] for row in rows], "population": population,
        "previous_selection_canonical_sha256": canonical(old),
        "snapshot_sha256": shared._sha256(paths[0]), "results_sha256": shared._sha256(paths[1]),
        "source_hashes": {name: hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() for name in SOURCE_FILES},
        "protocol_normalized_sha256": hashlib.sha256(PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest(),
        "preflight": budget, "checkpoint_root": DATA.as_posix(),
        "provider_calls": 0, "labels_frozen": False, "targets_sha256": None,
        "controls": {"authorization_required": True, "executable_run_contract": False, "runner_ready": False,
                     "dev_opened": False, "gold_fields_used": False, "promotion_allowed": False,
                     "no_paid_retries": True, "raw_before_parse_required": True, "unknown_charge_blocks_resume": True,
                     "all_labels_reviewed_and_frozen_before_predictions": True, "no_post_label_resampling": True,
                     "prompts_and_validator_frozen_before_labels": True, "targets_outside_classifier_loop": True,
                     "class_support_counts_reported_not_repaired": True, "formal_pass_gate": None},
        "provenance": "Fresh relative to earlier refusal selections/known family proxy only; all TRAIN already used in project retrieval/automated analysis; exact-question/leading-document proxy is not semantic-family independence; Qwen annotation plus current-assistant review remains AI proxy, not independent expert gold or unseen DEV acceptance"}
    selection.prior._write_jsonl(outputs[0], packets)
    selection.prior._write_jsonl(outputs[1], rows)
    manifest.update(packet_sha256=shared._sha256(outputs[0]), inputs_sha256=shared._sha256(outputs[1]))
    selection.prior._write_json(REPORT, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True)
    value = prepare(parser.parse_args().research_root)
    print(json.dumps({"status": value["status"], "cases": value["cases"], "planning_cost_cny": value["preflight"]["planning_cost_cny"], "provider_calls": 0}))
