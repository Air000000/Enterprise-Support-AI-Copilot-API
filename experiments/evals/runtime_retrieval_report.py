"""Full160-only local IR report; no dataset loader, credentials or model client."""

import hashlib
import math
import random
from statistics import fmean

from ranx import Run

from experiments.evals.ir.ranx_adapter import build_ranx_qrels, evaluate_ir_run

ARMS = ("D", "DR", "H")
METRICS = ("recall@5", "recall@20", "mrr@10")


def document_id(chunk_id):
    parts = chunk_id.split(":")
    if (len(parts) != 5 or parts[0] != "doc" or not parts[1] or parts[2:4] != ["v1", "chunk"]
            or not parts[4].isdigit() or str(int(parts[4])) != parts[4]):
        raise ValueError("Invalid archival managed chunk ID")
    return parts[1]


def build_report(plan, cases, qrels, events):
    """Same 160 denominators and bootstrap samples; no partial-run promotion."""
    from experiments.evals.runtime_retrieval import accounting
    from experiments.evals.runtime_transport import digest

    pairs = plan["identity"]["questions"]
    ids = [cid for cid, _ in pairs]
    if (len(ids) != 160 or len(set(ids)) != 160 or len(cases) != 160 or set(qrels) != set(ids)
            or any(not isinstance(value, str) or not value for value in qrels.values())
            or [event["call_id"] for event in events] != list(plan["contract"]["calls"])
            or any(event["stage"] != plan["contract"]["calls"][event["call_id"]]
                   or event["status"] != "complete" for event in events)):
        raise ValueError("Full160 qrels/cases and complete480 receipts required")
    runs, vectors, context = {arm: {} for arm in ARMS}, {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
    for index, (cid, question) in enumerate(pairs):
        artifact = cases[index]
        row = artifact["result"]
        receipts = [{key: event[key] for key in ("call_id", "request_sha256", "response_sha256")}
                    for event in events if event["call_id"].startswith(cid + ":")]
        if (artifact["index"] != index or artifact["plan_sha256"] != digest(plan)
                or artifact["calls"] != receipts
                or artifact["result_sha256"] != digest(row) or row["question_id"] != cid
                or row["question_sha256"] != hashlib.sha256(question.encode()).hexdigest()
                or set(row["chunk_rankings"]) != set(ARMS) or set(row["document_rankings"]) != set(ARMS)
                or row["top14_chunk_ids"] != row["chunk_rankings"]["H"][:14]
                or len(row["fused_chunk_ids"]) != 100 or len(set(row["fused_chunk_ids"])) != 100
                or set(row["fused_chunk_ids"]) != set(row["chunk_rankings"]["H"])
                or set(row["chunk_rankings"]["D"]) != set(row["chunk_rankings"]["DR"])):
            raise ValueError("Case identity/ranking projection changed")
        for arm in ARMS:
            chunks = row["chunk_rankings"][arm]
            if len(chunks) != 100 or len(set(chunks)) != 100:
                raise ValueError("Each arm requires its full unique chunk100")
            documents = list(dict.fromkeys(document_id(key) for key in chunks))
            if documents != row["document_rankings"][arm]:
                raise ValueError("Document collapse differs from chunk ranking")
            rank = documents.index(qrels[cid]) + 1 if qrels[cid] in documents else 101
            vectors[arm].append((float(rank <= 5), float(rank <= 20), 1 / rank if rank <= 10 else 0.))
            context[arm].append(float(qrels[cid] in [document_id(key) for key in chunks[:14]]))
            runs[arm][cid] = {did: float(len(documents) - i) for i, did in enumerate(documents)}
    labels = build_ranx_qrels({cid: {qrels[cid]: 1} for cid in ids})
    metrics = {arm: evaluate_ir_run(labels, Run(runs[arm])) for arm in ARMS}
    for arm in ARMS:
        if any(abs(metrics[arm][metric] - fmean(row[i] for row in vectors[arm])) > 1e-12
               for i, metric in enumerate(METRICS)):
            raise RuntimeError("Per-query/bootstrap and ranx metrics disagree")
    comparisons = {"H-DR": ("H", "DR"), "DR-D": ("DR", "D"), "H-D": ("H", "D")}
    differences = {f"{name}:{metric}": [a[i] - b[i] for a, b in zip(vectors[left], vectors[right], strict=True)]
                   for name, (left, right) in comparisons.items() for i, metric in enumerate(METRICS)}
    samples = {key: [] for key in differences}
    rng = random.Random(20261023)
    for _ in range(10000):
        indices = [rng.randrange(160) for _ in range(160)]
        for key, values in differences.items():
            samples[key].append(sum(values[i] for i in indices) / 160)
    intervals = {key: [sorted(values)[249], sorted(values)[9749]] for key, values in samples.items()}
    latency = {}
    for stage in ("embedding", "rerank"):
        values = [event["latency_ms"] for event in events if event["stage"] == stage]
        ordered = sorted(values)
        latency[stage] = {"n": len(values), "first_attempt_ms": values[0],
                          "p50_ms": ordered[math.ceil(.5 * len(values)) - 1],
                          "p95_ms": ordered[math.ceil(.95 * len(values)) - 1]}
    return {"status": ("SYNTHETIC_FULL160_COMPONENT_CHECK" if plan["identity"]["planned_mode"] == "synthetic"
                       else "FULL160_ARCHIVAL_RETRIEVAL_COMPARISON"), "execution_mode": plan["identity"]["planned_mode"],
            "plan_sha256": digest(plan), "cases": 160, "metrics": metrics,
            "context_hit_at_chunk14": {arm: fmean(values) for arm, values in context.items()},
            "paired_differences": {key: {"absolute_delta": fmean(values), "delta_x100": 100 * fmean(values), "ci95": intervals[key]}
                                   for key, values in differences.items()},
            "primary": {"contrast": "H-DR", "metric": "mrr@10",
                        "conclusion": "PAIRED_IMPROVEMENT" if intervals["H-DR:mrr@10"][0] > 0 else "IMPROVEMENT_NOT_ESTABLISHED"},
            "bootstrap": {"seed": 20261023, "resamples": 10000, "method": "160 randrange draws per shared resample",
                          "interval_sorted_indices": [249, 9749]},
            "provider_attempts": len(events), "usage_tokens": sum(event["total_tokens"] for event in events),
            "accounting_cny_not_invoice": str(accounting(events)), "wire_attempt_latency": latency,
            "quality_acceptance_or_default_promotion": False,
            "limitations": ["Reused DEV and rekeyed/rebuilt archival HNSW; not fresh independent or ordinary-upload evaluation.",
                "Reference tokenizer is not attested cloud tokenization; accounting is not an invoice cap.",
                "Chunk14 context-hit is not document Recall@20 or the ordinary Dense Top3 answer profile.",
                "Wire attempt latency includes local response persistence, not pure provider or HTTP end-to-end latency.",
                "Synthetic execution_mode results must never be presented as real model/DEV quality evidence."]}
