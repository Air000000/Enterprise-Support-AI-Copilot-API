"""Offline v5 candidate: audit requirements, then derive the binary decision.

No provider calls or serving integration. Exact quotes prove provenance, not
entailment; requirement coverage and support statuses still depend on the model.
"""

import json

from experiments.evals.refusal_development_policy import CLASSIFIER_SYSTEM_PROMPT_V4_DEV
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput


CLASSIFIER_SYSTEM_PROMPT_V5_DEV = CLASSIFIER_SYSTEM_PROMPT_V4_DEV + """

Instead of an overall decision, return exactly this JSON object:
{"requirements": [{"requirement": "one necessary question requirement",
"status": "SUPPORTED|MISSING|CONFLICTING|CONDITIONAL",
"evidence": [{"source_id": "Source N", "quote": "exact contiguous source text"}],
"reason": "brief explanation of support or the specific gap"}],
"unresolved_ambiguities": ["material ambiguity, if any"]}

List ALL necessary question requirements separately, including each requested
outcome and necessary step. Do not omit a hard requirement or add invented ones.
SUPPORTED means direct support for this exact requirement in the applicable
scope; it needs at least one exact quote. CONFLICTING needs quoted evidence too.
MISSING means necessary support is absent. CONDITIONAL means support depends on
a material condition not established by the question or visible sources.
For missing evidence use an empty evidence array, not an invented quotation.
Quotes must be copied exactly, without ellipses, translation or paraphrase.
Check requested effect as well as parameter names: advice to increase a wait
does not support a request to decrease it. A matching error alone does not prove
the user's cause. One activation workaround does not establish a requested
license-transfer procedure. Useful conditional guidance is not full support.
Report unresolved target/applicability ambiguity in unresolved_ambiguities.
Do not output an overall decision; the caller derives it from your checks.
"""


def build_classifier_messages_v5_dev(classifier_input: ClassifierInput) -> list[dict[str, str]]:
    # ponytail: same input dataclass, no metadata/labels or second judging framework.
    payload = {
        "question": classifier_input.question,
        "sources": [{"source_id": s.source_id, "content": s.content}
                    for s in classifier_input.sources],
    }
    return [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT_V5_DEV},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_evidence_response(raw: str, classifier_input: ClassifierInput) -> dict:
    """Reject invalid output; never substitute a permissive decision on errors.

    The returned decision is a candidate model judgment, not verified truth.
    Callers must stop/audit ValueError, not route the input to answer generation.
    """
    checks = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(checks, dict) or set(checks) != {
        "requirements", "unresolved_ambiguities"
    }:
        raise ValueError("evidence response schema mismatch")
    requirements = checks["requirements"]
    ambiguities = checks["unresolved_ambiguities"]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("at least one material requirement is required")
    if not isinstance(ambiguities, list) or not all(map(_nonempty_text, ambiguities)):
        raise ValueError("invalid unresolved ambiguities")
    sources = {s.source_id: s.content for s in classifier_input.sources}
    if len(sources) != len(classifier_input.sources):
        raise ValueError("duplicate input source IDs")
    seen_requirements, supporting_ids, gaps = set(), [], []
    for item in requirements:
        if not isinstance(item, dict) or set(item) != {
            "requirement", "status", "evidence", "reason"
        }:
            raise ValueError("requirement schema mismatch")
        if not _nonempty_text(item["requirement"]) or not _nonempty_text(item["reason"]):
            raise ValueError("requirement and reason must be nonempty text")
        identity = item["requirement"].strip().casefold()
        if identity in seen_requirements:
            raise ValueError("duplicate requirement")
        seen_requirements.add(identity)
        status, evidence = item["status"], item["evidence"]
        if not isinstance(status, str) or status not in {
            "SUPPORTED", "MISSING", "CONFLICTING", "CONDITIONAL"
        } or not isinstance(evidence, list):
            raise ValueError("invalid support status or evidence")
        if status in {"SUPPORTED", "CONFLICTING"} and not evidence:
            raise ValueError("supported/conflicting requirement needs evidence")
        for citation in evidence:
            if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
                raise ValueError("citation schema mismatch")
            sid, quote = citation["source_id"], citation["quote"]
            if not isinstance(sid, str) or sid not in sources:
                raise ValueError("unknown source ID")
            if not _nonempty_text(quote) or quote not in sources[sid]:
                raise ValueError("quote is not exact source text")
            if status == "SUPPORTED" and sid not in supporting_ids:
                supporting_ids.append(sid)
        if status != "SUPPORTED":
            gaps.append({"requirement": item["requirement"], "status": status,
                         "reason": item["reason"]})
    # ponytail: model omissions/misclassified support cannot be caught by substring checks.
    return {"decision": "INSUFFICIENT" if gaps or ambiguities else "SUFFICIENT",
            "requirements": requirements, "unresolved_ambiguities": ambiguities,
            "supporting_source_ids": supporting_ids, "gaps": gaps}
