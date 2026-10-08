"""Synthetic offline gate demo, not retrieval, model evaluation or serving."""

import copy
import json

from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
from experiments.evals.refusal_requirement_coverage import (
    build_classifier_messages_v5_2_dev,
    validate_requirement_coverage_response,
)


def demo():
    # ponytail: hand-authored responses isolate gate behavior; real quality needs independent evaluation.
    sources = (ContextSource(
        "Source 1", "synthetic_chunk", "synthetic_doc",
        "Atlas cache warnings can be caused by network congestion.\n"
        "Atlas release 12 permits source downloads.\n",
    ),)
    support = {
        "status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}],
        "reason": "The source describes a possible cause, not a diagnosis of this incident.",
    }
    general = {
        "requirements": [{"requirement": "Describe a possible cause", **copy.deepcopy(support)}],
        "coverage_checks": {
            "requested_outcomes": copy.deepcopy(support),
            "applicability": copy.deepcopy(support),
            "version_and_time": {
                "status": "NOT_REQUIRED", "evidence": [], "reason": "No version or date is requested.",
            },
        },
        "unresolved_ambiguities": [],
    }
    latest = copy.deepcopy(general)
    historical = {
        "status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [2]}],
        "reason": "Source download permission is documented for release 12 only.",
    }
    latest["requirements"] = [{"requirement": "Source download permission", **copy.deepcopy(historical)}]
    for facet in ("requested_outcomes", "applicability"):
        latest["coverage_checks"][facet] = copy.deepcopy(historical)
    latest["coverage_checks"]["version_and_time"] = {
        "status": "MISSING", "evidence": [], "reason": "Release 12 permission does not prove latest availability.",
    }
    incident = copy.deepcopy(general)
    incident["requirements"][0]["requirement"] = "Establish the cause of this incident"
    for facet in ("requested_outcomes", "applicability"):
        incident["coverage_checks"][facet] = {
            "status": "CONDITIONAL", "evidence": [],
            "reason": "A possible cause is not an established diagnosis of this incident.",
        }
    malformed = copy.deepcopy(general)
    del malformed["coverage_checks"]["applicability"]
    wrong_support = copy.deepcopy(latest)
    wrong_support["coverage_checks"]["version_and_time"] = {
        **copy.deepcopy(historical),
        "reason": "DELIBERATELY WRONG: historical permission treated as latest-release proof.",
    }
    general_question = "What can cause an Atlas cache warning?"
    latest_question = "Can I download the latest Atlas release source?"
    cases = (
        ("1. General explanation", general_question, general, "SUFFICIENT"),
        ("2. Declared latest-version gap", latest_question, latest, "INSUFFICIENT"),
        ("3. Undiagnosed incident", "What caused this Atlas incident?", incident, "INSUFFICIENT"),
        ("4. Missing mandatory audit", general_question, malformed, "VALIDATION_ERROR"),
        ("5. KNOWN SEMANTIC FAILURE (not a quality pass)", latest_question, wrong_support, "SUFFICIENT"),
    )
    print("OFFLINE SYNTHETIC DEMO | hand-authored responses | calls=0 | cost=0 | DEV=NO")
    print("No retrieval, model call, answer generation or serving integration.\n")
    for name, question, response, expected in cases:
        classifier_input = ClassifierInput(question, sources)
        payload = json.loads(build_classifier_messages_v5_2_dev(classifier_input)[1]["content"])
        print(name)
        print("Question:", payload["question"])
        print("Numbered sources:", json.dumps(payload["sources"], ensure_ascii=False))
        print("Hand-authored response:", json.dumps(response, ensure_ascii=False))
        try:
            result = validate_requirement_coverage_response(json.dumps(response), classifier_input)
        except ValueError as exc:
            assert expected == "VALIDATION_ERROR", str(exc)
            print("Outcome: VALIDATION_ERROR (stop/audit; not a refusal prediction)", str(exc))
        else:
            assert result["decision"] == expected, result
            for requirement in result["requirements"]:
                for evidence in requirement["evidence"]:
                    assert evidence["quote"] in sources[0].content
            print("Outcome:", result["decision"])
            print("Gate result:", json.dumps(result, ensure_ascii=False))
        print()
    print("Self-check: 5/5 expected mechanical outcomes reproduced; NOT an accuracy score.")
    print("True citations cannot detect false SUPPORTED or false NOT_REQUIRED judgments.")


if __name__ == "__main__":
    demo()
