"""Offline v5.1 candidate: select source lines; code extracts untouched evidence.

No provider calls. Provenance is mechanical; relevance/coverage are model judgments.
"""

import json

from experiments.evals import refusal_structured_evidence as v5
from experiments.evals.refusal_development_policy import CLASSIFIER_SYSTEM_PROMPT_V4_DEV
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput


CLASSIFIER_SYSTEM_PROMPT_V5_1_DEV = CLASSIFIER_SYSTEM_PROMPT_V4_DEV + """

Instead of an overall decision, return exactly this JSON object:
{"requirements": [{"requirement": "one necessary question requirement",
"status": "SUPPORTED|MISSING|CONFLICTING|CONDITIONAL",
"evidence": [{"source_id": "Source N", "segment_ids": [1, 2]}],
"reason": "brief explanation of support or the specific gap"}],
"unresolved_ambiguities": ["material ambiguity, if any"]}

List ALL necessary question requirements separately, including each requested
outcome and necessary step. Do not omit a hard requirement or add invented ones.
SUPPORTED means direct support for this exact requirement in the applicable
scope; it needs at least one evidence reference. CONFLICTING needs evidence too.
MISSING means necessary support is absent. CONDITIONAL means support depends on
a material condition not established by the question or visible sources.
For missing evidence use an empty evidence array, not an invented reference.

Each source contains numbered segments: exact original lines, not new documents
or retrieved chunks. Select only integer segment_ids shown inside that source.
IDs are local to each source. Select all lines needed to retain material
conditions, limitations and qualifications; a nearby topic is not support.
Do not write a quote, character offset, summary or merged citation text. The
caller extracts each selected line unchanged as a separate evidence quote.
Blank lines carry no evidence; the caller ignores them, not their original text.
Numbering changes presentation only: judge the ENTIRE supplied context, not
just selected lines. Do not hide conflicting or missing support by selection.

Check requested effect as well as parameter names: advice to increase a wait
does not support a request to decrease it. A matching error alone does not prove
the user's cause. One activation workaround does not establish a requested
license-transfer procedure. Useful conditional guidance is not full support.
Report unresolved target/applicability ambiguity in unresolved_ambiguities.
Do not output an overall decision; the caller derives it from your checks.
"""


def numbered_sources(classifier_input: ClassifierInput) -> list[dict]:
    """Source-local, 1-based line IDs; concatenation reproduces the original text."""
    sources, seen = [], set()
    for source in classifier_input.sources:
        sid = source.source_id
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("invalid or duplicate input source ID")
        if not isinstance(source.content, str):
            raise ValueError("source content must be text")
        seen.add(sid)
        # ponytail: original line boundaries, no NLP sentence splitter or truncation.
        sources.append({"source_id": sid, "segments": [
            {"segment_id": index, "content": line}
            for index, line in enumerate(source.content.splitlines(keepends=True), start=1)
        ]})
    return sources


def build_classifier_messages_v5_1_dev(classifier_input: ClassifierInput) -> list[dict[str, str]]:
    if not isinstance(classifier_input.question, str) or not classifier_input.question.strip():
        raise ValueError("question must be nonempty text")
    payload = {"question": classifier_input.question, "sources": numbered_sources(classifier_input)}
    return [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT_V5_1_DEV},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_citation_selection_response(raw: str, classifier_input: ClassifierInput) -> dict:
    """Resolve only valid IDs; an invalid selection is an error, never a refusal."""
    checks = json.loads(raw, object_pairs_hook=v5._unique_object)
    if not isinstance(checks, dict) or set(checks) != {"requirements", "unresolved_ambiguities"}:
        raise ValueError("citation selection schema mismatch")
    requirements = checks["requirements"]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("at least one material requirement is required")
    lines = {(s["source_id"], part["segment_id"]): part["content"]
             for s in numbered_sources(classifier_input) for part in s["segments"]}
    sources = {s.source_id for s in classifier_input.sources}
    expanded = []
    for item in requirements:
        if not isinstance(item, dict) or set(item) != {"requirement", "status", "evidence", "reason"}:
            raise ValueError("requirement schema mismatch")
        if not isinstance(item["evidence"], list):
            raise ValueError("evidence must be a list")
        quotes, seen = [], set()
        for reference in item["evidence"]:
            if not isinstance(reference, dict) or set(reference) != {"source_id", "segment_ids"}:
                raise ValueError("citation reference schema mismatch")
            sid, ids = reference["source_id"], reference["segment_ids"]
            if not isinstance(sid, str) or sid not in sources:
                raise ValueError("unknown source ID")
            if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
                raise ValueError("segment_ids must be nonempty integer IDs")
            for segment_id in ids:
                identity = sid, segment_id
                if identity not in lines:
                    raise ValueError("unknown source-local segment ID")
                if identity not in seen and lines[identity].strip():
                    quotes.append({"source_id": sid, "quote": lines[identity]})
                    seen.add(identity)
        expanded.append({**item, "evidence": quotes})
    # ponytail: reuse the frozen v5 schema/status/provenance gate, do not modify it.
    result = v5.validate_evidence_response(json.dumps({
        "requirements": expanded, "unresolved_ambiguities": checks["unresolved_ambiguities"],
    }, ensure_ascii=False), classifier_input)
    result["citation_selections"] = [item["evidence"] for item in requirements]
    return result
