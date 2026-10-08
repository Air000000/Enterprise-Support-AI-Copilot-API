"""Offline v5.2 candidate: mandatory coverage audits, not verified semantic truth."""

import json

from experiments.evals import refusal_citation_selection as citation
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput

COVERAGE_FACETS = ("requested_outcomes", "applicability", "version_and_time")

CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV = citation.CLASSIFIER_SYSTEM_PROMPT_V4_DEV + """

Return exactly this JSON object, not an overall decision:
{"requirements": [{"requirement": "one necessary question requirement",
"status": "SUPPORTED|MISSING|CONFLICTING|CONDITIONAL",
"evidence": [{"source_id": "Source N", "segment_ids": [1, 2]}],
"reason": "brief support or gap explanation"}],
"coverage_checks": {
"requested_outcomes": {"status": "SUPPORTED|MISSING|CONFLICTING|CONDITIONAL",
"evidence": [], "reason": "audit every requested outcome and necessary step"},
"applicability": {"status": "SUPPORTED|MISSING|CONFLICTING|CONDITIONAL",
"evidence": [], "reason": "audit the applicable product/environment/incident scope"},
"version_and_time": {"status": "SUPPORTED|MISSING|CONFLICTING|CONDITIONAL|NOT_REQUIRED",
"evidence": [], "reason": "audit explicit version, current/latest or time requirements"}},
"unresolved_ambiguities": ["material ambiguity, if any"]}

List ALL necessary requirements separately. Coverage checks are mandatory even
if already discussed in requirements. SUPPORTED needs direct evidence for that
exact claim; SUPPORTED and CONFLICTING need evidence references. MISSING means
support is absent. CONDITIONAL means an unresolved material condition. Every
check needs a nonempty reason. Missing support is not a schema error or success.

requested_outcomes: a general mechanism or permission does not establish every
requested deliverable, procedure, availability claim or version-specific result.
applicability: distinguish possible/general causes from an established cause
or fix for a particular incident. General explanation may be supported without
a unique diagnosis or literal repetition of the target's name. Do not invent
an exact-name requirement; explain any material applicability bridge or gap.
version_and_time: retain explicitly requested versions, latest/current status
and dates. Historical or family-level availability alone does not establish a
current/latest release. Missing repetition of a version string is not enough
to refuse when visible compatible applicability evidence covers the request.
NOT_REQUIRED is allowed ONLY here, with empty evidence and a reason explaining
why the question imposes no material version/time requirement. Do not use it
to discard an explicit constraint. It is a model judgment, not outside proof.

Use the entire ordered context. Sources contain source-local, 1-based original
line segment_ids. Select only displayed integer IDs, including lines needed to
retain conditions and limitations. Blank lines carry no evidence. Do not write
quotes, offsets or merged text: the caller extracts each selected line unchanged.
For absent evidence return an empty list, never invent a reference. A true quote
does not prove relevance or coverage. Report unresolved material ambiguity.
"""


def build_classifier_messages_v5_2_dev(classifier_input: ClassifierInput) -> list[dict[str, str]]:
    # ponytail: reuse full-context numbering; no keyword parser, new retrieval or client.
    messages = citation.build_classifier_messages_v5_1_dev(classifier_input)
    messages[0]["content"] = CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV
    return messages


def validate_requirement_coverage_response(raw: str, classifier_input: ClassifierInput) -> dict:
    checks = json.loads(raw, object_pairs_hook=citation.v5._unique_object)
    if not isinstance(checks, dict) or set(checks) != {
        "requirements", "coverage_checks", "unresolved_ambiguities"
    }:
        raise ValueError("coverage response schema mismatch")
    requirements, coverage = checks["requirements"], checks["coverage_checks"]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("at least one material requirement is required")
    if not isinstance(coverage, dict) or set(coverage) != set(COVERAGE_FACETS):
        raise ValueError("all coverage facets are required")
    combined = list(requirements)
    for facet in COVERAGE_FACETS:
        item = coverage[facet]
        if not isinstance(item, dict) or set(item) != {"status", "evidence", "reason"}:
            raise ValueError("coverage check schema mismatch")
        if item["status"] == "NOT_REQUIRED":
            if (facet != "version_and_time" or item["evidence"] != []
                    or not citation.v5._nonempty_text(item["reason"])):
                raise ValueError("invalid NOT_REQUIRED coverage check")
            continue
        combined.append({"requirement": f"Coverage audit: {facet}", **item})
    # ponytail: one existing locator/status gate for material requirements AND audits.
    result = citation.validate_citation_selection_response(json.dumps({
        "requirements": combined, "unresolved_ambiguities": checks["unresolved_ambiguities"],
    }, ensure_ascii=False), classifier_input)
    result["coverage_checks"] = coverage
    return result
