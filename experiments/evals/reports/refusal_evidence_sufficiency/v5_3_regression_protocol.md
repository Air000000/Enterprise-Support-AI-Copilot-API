# v5.2.1 / v5.3 observed-TRAIN development regression

Status: preparation only, not authorized, zero provider calls. No serving change.

## Population and interpretation

Reuse all 30 frozen natural TRAIN questions and each complete ordered Top14 context,
without retrieval, truncation, reranking, label edits or new annotation. Preserve
the existing input/target hashes and AI-proxy counts: 21 SUFFICIENT, 4 INSUFFICIENT,
5 AMBIGUOUS. These are not independent human/expert gold labels. This population
was already observed, including targeted evidence inspection used to design v5.3.
This is development regression, not unseen validation or representative enterprise
accuracy. The original minimum five examples per binary class is still unmet;
do not change that rule, resample, relabel or claim formal acceptance.

## Frozen comparison

Run fresh concurrent v5.2.1 and v5.3 arms on the identical 30 inputs. Historical
v5.2.1 predictions establish provenance, not a substitute baseline. Alternate arm
order by frozen question index, reversing at odd indices. Only the system prompt
differs; user payload, schema and strict citation/coverage validator stay unchanged.
No keyword patches, post-hoc audit deletion, inferred citations, response repair,
answer generation, judge or online integration.

Hypothesis: question-scoped requirements reduce unnecessary refusals, while explicit
incident applicability reduces unsupported admissions. Count all fixes and new
errors, not only the exposed problem cases; neither hypothesis is an acceptance gate.

Report counts before percentages: all30 decisions and terminal failures per arm;
binary agreement/confusion with denominator25 including failures; class denominators
21/4; AMBIGUOUS5 decisions descriptive only. Report common-valid binary pairs with
their actual denominator/classes, paired agreement and transitions separately.
Distinguish all30 admission coverage from admitted-binary proxy agreement, showing
ambiguous admissions separately. Preserve null failures, with possible binary
agreement bounds; an undefined denominator is null, never zero. Recompute metrics
dropping each member of the already-known correlated pair separately, without new
calls. This does not establish independent-family validation. Include every charged
outcome in cost/tokens/latency. No IID confidence interval, significance claim,
formal pass flag, production promotion or run-until-pass repeats.

## Execution boundary and resumability

Proposed future scope: Singapore `qwen3.5-plus-2026-04-20`, temperature0,
thinking disabled, JSON-object responses, max2048 output tokens. At most60 calls,
30 per arm, SDK retries0; format/provider failures are terminal. No DEV access.
Send only questions and source bodies, not target labels, annotation reasons,
retrieval metadata or secrets. Public outputs contain aggregates/fingerprints only.

Pricing checked 2026-10-15 against [official pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing):
International input USD0.4/M and output USD2.4/M within the <=256K input band.
Use fixed accounting FX7.5, no discounts, proposed CNY8 ledger cap. UTF-8 bytes plus
512 framing per request is a conservative planning proxy, not certified tokens or
a guaranteed invoice cap. Freeze the generated estimate before execution; stop if
remaining budget cannot reserve the next request. No automatic recharge.

Use a fresh `data/refusal_natural_v5_3/run_v1` checkpoint and reuse existing durable
execution logic, not a new checkpoint framework. Bind input, target, request,
prompt, source and protocol hashes before paid execution. Exclusive local lock;
pending marker before POST; durable raw response/usage before parsing and journal
append. Recover saved raw responses without another call; unknown-charge pending
state blocks automatic resume. Never repeat completed calls. Targets remain outside
the paid classifier loop and are read only for preflight/final aggregate evaluation.

This preparation has no paid entrypoint or executable run contract. A separate
explicit authorization for this data/model/call/budget scope is required before
adding a thin runner and executing; previous run authorizations are exhausted.
