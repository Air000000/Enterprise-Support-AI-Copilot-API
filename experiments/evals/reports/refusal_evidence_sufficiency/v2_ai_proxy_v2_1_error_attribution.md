# Refusal v2.1 AI-proxy disagreement attribution

Status: `POST_HOC_DIAGNOSTIC_NOT_RESCORING`

Date: `2026-09-23`

The preregistered v2.1 decision remains
`REJECT_EVIDENCE_SUFFICIENCY_V2_AI_PROXY`. This analysis does not change a
target, threshold, prediction, or reported metric.

## Primary finding

The proxy target and classifier contracts measure different constructs.

- The annotation contract labels context `SUFFICIENT` only when it supports
  every material claim in the hidden gold answer.
- The classifier receives no gold answer and judges whether the visible
  context can completely answer the exact visible question.

Those tasks diverge whenever the question underspecifies the gold answer.
The classifier cannot infer that a short "where can I find this?" question
will be scored against hidden CVSS details, or that a generic troubleshooting
question requires one particular configuration setting.

This is systematic rather than isolated: eight of the 40 holdout questions
contain "where can I find"; four of their nine total disagreements are cases
where the context supplies the requested bulletin or link but the proxy label
requires additional hidden-gold details.

## Nine disagreements

| ID | Target | Prediction | Diagnostic attribution |
| --- | --- | --- | --- |
| `TRAIN_Q033` | insufficient | sufficient | Contract mismatch: the context explains the cause and manual resolution asked by the visible question; only the gold-specific automatic-renew setting is absent. |
| `TRAIN_Q043` | sufficient | insufficient | Likely classifier over-refusal: Source 2 directly identifies `pdcollect` for the IBM i OS agent and gives its location. |
| `TRAIN_Q269` | sufficient | insufficient | Broken question/gold pairing: the question asks how to issue a graceful shutdown; the gold answer and cited sources instead configure behavior after repeated aborted batches. |
| `TRAIN_Q291` | insufficient | sufficient | Ambiguous duplicate-title case: the context gives the exact products, RHEL version, action, and other CVE sets, while the underspecified question does not identify the gold bulletin's CVE set. |
| `TRAIN_Q324` | insufficient | sufficient | Contract mismatch: the question asks where to find the CVE-2015-0138 bulletin; the exact bulletin and URL are present, while hidden CVSS details are absent. |
| `TRAIN_Q343` | sufficient | insufficient | Likely classifier over-refusal: Source 1 identifies the exact WSRR CVE and Source 3 states that the same CVE affects WebSphere Liberty. |
| `TRAIN_Q406` | insufficient | sufficient | Contract mismatch: the exact API Connect bulletin, CVE, remediation, and Java bulletin link are present; hidden vulnerability-score details are absent. |
| `TRAIN_Q493` | insufficient | sufficient | Contract mismatch: the exact ITNM CVE bulletin and direct WAS bulletin URL are present; hidden CVSS details are absent. |
| `TRAIN_Q584` | insufficient | sufficient | Likely classifier over-admission: the context links to remediation but does not contain the requested PI89498 or Fix Pack 8.0.0.15 recommendation. |

The diagnostic split is six likely target/construct problems and three likely
classifier errors. These are hypotheses for human adjudication, not replacement
labels. Reviewing only disagreements and then rescoring would be biased; agreement
cases may contain the same target-contract defect.

## Consequence

The v2.1 result is a valid failure against its frozen AI-draft proxy labels.
It is not clean evidence that the runtime evidence gate has 0.775 balanced
accuracy, because the proxy labels partly test access to hidden information
that the runtime classifier intentionally never receives.

Prompt tuning, threshold changes, or another run on these 40 cases would not
repair that construct-validity failure.

## Minimum valid v3

1. Define the target as answerability from the exact visible question and
   Top14 context, which matches runtime inputs.
2. Do not expose the gold answer during the initial label. Use it only during
   later adjudication to detect broken or underspecified question/gold pairs.
3. Mark materially ambiguous cases `QUESTIONABLE` and exclude them before
   freezing the scored holdout.
4. Use fresh TRAIN cases that were not in the v2 80-case annotation set.
5. Keep the classifier blind to targets and case-specific annotation notes.
6. Treat any AI annotation as development-only; production promotion still
   requires independent human confirmation.
