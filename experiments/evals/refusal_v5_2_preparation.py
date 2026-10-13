"""Prepare TRAIN-only diagnostic packets and an inert paired budget; no providers."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from experiments.evals import refusal_v5_preparation as prior
from experiments.evals.refusal_citation_selection import build_classifier_messages_v5_1_dev
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
from experiments.evals.refusal_requirement_coverage import build_classifier_messages_v5_2_dev

SEED = "refusal-v5-2-additional-train-diagnostic-v1"
DATA_DIR = Path("data/refusal_v5_2_train")
REPORT = prior.REPORT_DIR / "v5_2_train_preparation.json"
EXPOSURE_FILES = (
    "e0_dense/task7_causal_audit_sample_manifest.json",
    "e1_rerank/train_residual_review.jsonl",
    "r4_chunk_bm25_audit/diagnostic_cases.json",
    "g1_document_local/evidence_sufficiency_30case_preregistration_v1_1.json",
    "g0_generation/pilot_document_aware_v1/train_generation_results.jsonl",
    "final_context_budget/residual_cases.jsonl",
    "structure_preserving_synthesis/failure_attribution.json",
    "techqa_structure_forensic/six_case_forensic.json",
)


def family_exclusions(snapshots, results, excluded):
    """Conservative visible-context proxy, NOT a semantic family guarantee."""
    def normalize(text):
        return " ".join(text.casefold().split())
    questions = {normalize(r["question"]) for r in snapshots if r["question_id"] in excluded}
    documents = {r["reranked_document_ids"][0] for r in results if r["question_id"] in excluded}
    # ponytail: exact question/leading retrieved document proxy; semantic review remains separate.
    return {r["question_id"] for r in snapshots if normalize(r["question"]) in questions} | {
        r["question_id"] for r in results if r["reranked_document_ids"][0] in documents
    }


def paired_preflight(packets):
    arms = {}
    for name, builder, module in (
        ("v5_1", build_classifier_messages_v5_1_dev, "refusal_citation_selection.py"),
        ("v5_2", build_classifier_messages_v5_2_dev, "refusal_requirement_coverage.py"),
    ):
        bounds, payload_hashes = [], []
        for packet in packets:
            value = ClassifierInput(packet["question"], tuple(
                ContextSource(s["source_id"], "", "", s["content"]) for s in packet["sources"]))
            messages = builder(value)
            bounds.append(sum(len(m["content"].encode("utf-8")) for m in messages) + 512)
            payload_hashes.append(prior.canonical_sha256(messages))
        if not bounds or max(bounds) + 2048 > 256_000:
            raise RuntimeError("empty population or pricing context band exceeded")
        arms[name] = {
            "prompt_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
            "validator_normalized_source_sha256": hashlib.sha256(
                (Path("experiments/evals") / module).read_text(encoding="utf-8").encode()).hexdigest(),
            "request_sha256": payload_hashes, "input_byte_plus_framing_bounds": bounds,
            "max_output_tokens": 2048, "planned_calls": len(packets),
            "planning_cost_usd": (sum(bounds) * 0.4 + len(packets) * 2048 * 2.4) / 1_000_000,
        }
    return {"arms": arms, "planned_calls": 2 * len(packets),
            "planning_cost_usd": sum(a["planning_cost_usd"] for a in arms.values()),
            "shared_dependency_sha256": {name: hashlib.sha256(
                (Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest()
                for name in ("refusal_structured_evidence.py", "refusal_development_policy.py")},
            "model": "qwen3.5-plus-2026-04-20", "region": "Singapore",
            "temperature": 0, "enable_thinking": False, "response_format": "json_object",
            "pricing": {"checked_date": "2026-10-13", "input_usd_per_million": 0.4,
                        "output_usd_per_million": 2.4, "source":
                        "https://www.alibabacloud.com/help/en/model-studio/model-pricing"},
            "caveat": "UTF-8 bytes + 512 framing is a planning proxy, not certified tokens or invoice cap; annotation and retries excluded",
            "authorization_required": True, "executable_run_contract": False}


def prepare(research_root):
    root = Path(research_root) / "experiments/evals/reports"
    snapshot_path = root / "r4_c1_hybrid_rerank/train_fused_snapshot.jsonl"
    result_path = root / "r4_c1_hybrid_rerank/train_results.jsonl"
    for path, expected in ((snapshot_path, prior.reuse.EXPECTED_SNAPSHOT_SHA256),
                           (result_path, prior.reuse.EXPECTED_RESULTS_SHA256)):
        if prior.reuse._sha256(path) != expected:
            raise RuntimeError("frozen TRAIN artifact SHA mismatch")
    old = json.loads((prior.REPORT_DIR / "v5_train_selection.json").read_text(encoding="utf-8"))
    excluded = set(old["excluded_question_ids"] + old["selected_question_ids"])
    exposure = {}
    for relative in EXPOSURE_FILES:
        path = root / relative
        # IDs only are used. Existing research files may contain gold fields; none are used as labels.
        ids = set(re.findall(r'"(TRAIN_Q\d{3})"', path.read_text(encoding="utf-8")))
        excluded.update(ids)
        exposure[relative] = {"sha256": prior.reuse._sha256(path), "question_ids": sorted(ids)}
    snapshots = prior.reuse._read_jsonl(snapshot_path)
    results = [{key: row[key] for key in (
        "question_id", "reranked_chunk_ids", "reranked_document_ids"
    )} for row in prior.reuse._read_jsonl(result_path)]
    if len(snapshots) != 450 or len(results) != 450:
        raise RuntimeError("frozen TRAIN population changed")
    families = family_exclusions(snapshots, results, excluded)
    packets, population = prior.select_packets(snapshots, results, excluded | families, cases=30, seed=SEED)
    budget = paired_preflight(packets)
    paths = (DATA_DIR / "annotation_packets.jsonl", DATA_DIR / "inputs.jsonl", REPORT)
    if any(path.exists() for path in paths):
        raise RuntimeError("refusing to overwrite preparation")
    prior._write_jsonl(paths[0], packets)
    prior._write_jsonl(paths[1], [{k: v for k, v in p.items() if k != "annotation"} for p in packets])
    manifest = {
        "status": "SELECTED_NOT_ANNOTATED_NOT_AUTHORIZED", "seed": SEED, "cases": 30,
        "eligible_cases": population, "selected_question_ids": [p["question_id"] for p in packets],
        "recorded_excluded_question_ids": sorted(excluded), "family_proxy_excluded_question_ids": sorted(families - excluded),
        "exposure_sources": exposure, "prior_selection_canonical_sha256": prior.canonical_sha256(old),
        "snapshot_sha256": prior.reuse.EXPECTED_SNAPSHOT_SHA256,
        "results_sha256": prior.reuse.EXPECTED_RESULTS_SHA256,
        "packet_sha256": prior.reuse._sha256(paths[0]), "inputs_sha256": prior.reuse._sha256(paths[1]),
        "provenance": "Additional refusal diagnostic, not independent holdout: all TRAIN already used in project retrieval/automated failure analysis; family proxy is not semantic clearance",
        "provider_calls": 0, "dev_opened": False, "gold_fields_used": False,
        "labels_frozen": False, "promotion_allowed": False, "paired_preflight": budget,
    }
    prior._write_json(REPORT, manifest)
    return {k: manifest[k] for k in ("status", "cases", "eligible_cases", "provider_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True)
    print(json.dumps(prepare(parser.parse_args().research_root), indent=2))
