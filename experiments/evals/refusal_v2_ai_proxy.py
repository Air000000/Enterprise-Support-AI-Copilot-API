from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.evals.refusal_evidence_sufficiency import ClassifierInput

DEFAULT_PACKET_PATH = Path(
    "data/refusal_v2_confirmation_set/annotation_packets.jsonl"
)
DEFAULT_TARGETS_SOURCE_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v2_ai_draft_targets.jsonl"
)
DEFAULT_INPUTS_PATH = Path(
    "data/refusal_v2_ai_proxy/holdout_inputs.jsonl"
)
DEFAULT_TARGETS_PATH = Path(
    "data/refusal_v2_ai_proxy/holdout_targets.jsonl"
)
DEFAULT_MANIFEST_PATH = Path(
    "experiments/evals/reports/refusal_evidence_sufficiency/"
    "v2_ai_proxy_holdout_manifest.json"
)

EXPECTED_PACKET_SHA256 = (
    "3029acd916bfff10def1448b28181e1d5341513232f6edcb4fafa8dc578092f4"
)
EXPECTED_TARGETS_SOURCE_SHA256 = (
    "0d08f8f06b6acdaa3da7b36e04c795571e8f25fd8de22a9fdb7a89a6c81b6c67"
)
SELECTION_SEED = "refusal-v2-ai-proxy-holdout-v1"
PER_CLASS = 20
DETAILS_PREVIOUSLY_OPENED = {"TRAIN_Q000", "TRAIN_Q005"}

CLASSIFIER_SYSTEM_PROMPT_V2 = """
You are a conservative evidence-completeness gate for a technical-support
RAG system.

Judge only whether the supplied Context directly supports a complete answer
to the exact Question, without outside knowledge.

First identify every material requirement in the Question. SUFFICIENT
requires the cited sources, considered together, to support all material
claims needed by the answer, including requested causes, conditions,
versions, values, and resolution steps when applicable.

Return INSUFFICIENT if the Context only discusses the same product, error,
or topic; supports a neighboring problem; omits any material requested
claim; conflicts; is ambiguous; or requires a plausible but unsupported
inference. When uncertain, return INSUFFICIENT.

Do not answer the Question. Return only the required structured decision.
""".strip()


def build_classifier_messages_v2(
    classifier_input: ClassifierInput,
) -> list[dict[str, str]]:
    context = "\n\n---\n\n".join(
        f"[{source.source_id}]\n{source.content}"
        for source in classifier_input.sources
    )
    user_prompt = (
        "Question:\n"
        f"{classifier_input.question}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Return a JSON object with exactly these fields:\n"
        '- decision: "SUFFICIENT" or "INSUFFICIENT"\n'
        "- reason: a short evidence-based explanation\n"
        "- supporting_source_ids: a JSON array using Source N identifiers only"
    )
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_prompt},
    ]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _selection_hash(question_id: str, *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def build_ai_proxy_holdout(
    packet_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    per_class: int = PER_CLASS,
    seed: str = SELECTION_SEED,
    excluded_question_ids: set[str] = DETAILS_PREVIOUSLY_OPENED,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    packets = {str(row["question_id"]): row for row in packet_rows}
    targets = {str(row["question_id"]): row for row in target_rows}
    if len(packets) != len(packet_rows) or len(targets) != len(target_rows):
        raise RuntimeError("duplicate question_id")
    if set(packets) != set(targets):
        raise RuntimeError("packet and target populations differ")

    candidates: dict[str, list[tuple[str, str]]] = {
        "SUFFICIENT": [],
        "INSUFFICIENT": [],
    }
    questionable = 0
    for question_id, target in targets.items():
        target_class = str(target["target_class"])
        if target_class == "QUESTIONABLE":
            questionable += 1
            continue
        if target_class not in candidates:
            raise RuntimeError(f"invalid target class: {question_id}")
        if question_id in excluded_question_ids:
            continue
        candidates[target_class].append(
            (_selection_hash(question_id, seed=seed), question_id)
        )

    selected_ids: set[str] = set()
    for target_class, values in candidates.items():
        values.sort()
        if len(values) < per_class:
            raise RuntimeError(f"not enough {target_class} cases")
        selected_ids.update(question_id for _, question_id in values[:per_class])

    inputs: list[dict[str, Any]] = []
    holdout_targets: list[dict[str, str]] = []
    for question_id in sorted(selected_ids):
        packet = packets[question_id]
        sources = packet["sources"]
        if len(sources) != 14:
            raise RuntimeError(f"expected Top14 sources: {question_id}")
        inputs.append(
            {
                "question_id": question_id,
                "question": str(packet["question"]),
                "sources": [
                    {
                        "source_id": str(source["source_id"]),
                        "content": str(source["content"]),
                    }
                    for source in sources
                ],
            }
        )
        holdout_targets.append(
            {
                "question_id": question_id,
                "target_class": str(targets[question_id]["target_class"]),
            }
        )

    manifest = {
        "selection_seed": seed,
        "algorithm": "lowest_sha256(seed + ':' + question_id) per class",
        "per_class": per_class,
        "selected_total": len(inputs),
        "questionable_excluded": questionable,
        "detail_opened_exclusions": sorted(excluded_question_ids),
        "selected_question_ids": sorted(selected_ids),
        "classifier_payload_fields": ["question", "sources"],
        "source_payload_fields": ["source_id", "content"],
    }
    return inputs, holdout_targets, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the refusal v2 AI-proxy holdout inputs."
    )
    parser.add_argument("--packet-path", default=str(DEFAULT_PACKET_PATH))
    parser.add_argument(
        "--targets-source-path",
        default=str(DEFAULT_TARGETS_SOURCE_PATH),
    )
    parser.add_argument("--inputs-path", default=str(DEFAULT_INPUTS_PATH))
    parser.add_argument("--targets-path", default=str(DEFAULT_TARGETS_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    packet_path = Path(args.packet_path)
    targets_source_path = Path(args.targets_source_path)
    if _sha256(packet_path) != EXPECTED_PACKET_SHA256:
        raise RuntimeError("frozen blank packet SHA mismatch")
    if _sha256(targets_source_path) != EXPECTED_TARGETS_SOURCE_SHA256:
        raise RuntimeError("frozen AI-draft targets SHA mismatch")

    inputs, targets, selection = build_ai_proxy_holdout(
        _read_jsonl(packet_path),
        _read_jsonl(targets_source_path),
    )
    inputs_path = Path(args.inputs_path)
    targets_path = Path(args.targets_path)
    manifest_path = Path(args.manifest_path)
    existing = [
        str(path)
        for path in (inputs_path, targets_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise RuntimeError("refusing to overwrite: " + ", ".join(existing))

    _write_jsonl(inputs_path, inputs)
    _write_jsonl(targets_path, targets)
    manifest = {
        "schema_version": 1,
        "run": "refusal_v2_ai_proxy_holdout_v1",
        "status": "PREREGISTERED_INPUTS_FROZEN",
        "label_provenance": "AI_DRAFT_DEVELOPMENT_ONLY",
        "selection": selection,
        "artifacts": {
            "blank_packet_sha256": EXPECTED_PACKET_SHA256,
            "source_targets_sha256": EXPECTED_TARGETS_SOURCE_SHA256,
            "holdout_inputs_path": str(inputs_path).replace("\\", "/"),
            "holdout_inputs_sha256": _sha256(inputs_path),
            "holdout_targets_path": str(targets_path).replace("\\", "/"),
            "holdout_targets_sha256": _sha256(targets_path),
            "prompt_sha256": hashlib.sha256(
                CLASSIFIER_SYSTEM_PROMPT_V2.encode()
            ).hexdigest(),
        },
        "controls": {
            "provider_calls": 0,
            "dev_artifact_opened": False,
            "classifier_predictions_created": False,
            "targets_must_not_be_loaded_by_paid_loop": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("REFUSAL_V2_AI_PROXY_PREFLIGHT=PASS")
    print(f"HOLDOUT_CASES={len(inputs)}")
    print(f"INPUTS_SHA256={_sha256(inputs_path)}")
    print(f"TARGETS_SHA256={_sha256(targets_path)}")
    print(f"PROMPT_SHA256={manifest['artifacts']['prompt_sha256']}")
    print("PROVIDER_CALLS=0")
    print("DEV_ARTIFACT_OPENED=NO")


if __name__ == "__main__":
    main()
