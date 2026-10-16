# Prospective refusal comparison on 50 fresh-to-refusal TRAIN questions

Preparation only: zero calls, no labels, no executable paid contract or runner.
Freeze v5.2.1 and v5.3 unchanged before annotation. The question is whether the
observed small gain persists outside the old30 refusal examples, not how to obtain
a passing score by changing prompts, labels or samples.

## Selection and scope

Reuse the frozen450 TRAIN retrieval snapshot/results, original ordered full Top14,
without retrieval, reranking, truncation or gold-answer/qrel-based selection. Bind
the previous selection's canonical hash; exclude all its recorded exclusions,
family-proxy exclusions and selected30. Reuse normalized-question/leading-document
family exclusions against those old cases. In fixed SHA256(seed:question_id) priority,
greedily skip the same normalized question or leading document within the new batch,
take exactly50, then freeze question-ID order. No smaller fallback or post-label
class balancing/replacement. The seed is `refusal-prospective-train-50-v1`.

These are fresh relative to recorded refusal selections and this visible family
proxy, not untouched project data: all TRAIN participated in retrieval/automated
analysis. Different lead documents can still describe the same issue. Record any
later-discovered correlation without silently replacing cases. Neither the family
proxy nor the annotation process establishes independent semantic families/gold.
Do not open DEV or pool the old30, fictional controls or other experiments into scores.

## Two stages, in this order

1. At most50 Singapore `qwen3.6-plus-2026-04-02` annotation calls, max4096 output
   tokens each, using the existing blind annotation prompt and full sources. The
   current assistant reviews every question/draft/cited evidence, inspecting full
   context where needed and recording review depth. Preserve original responses,
   parser failures and review changes. Local citation-format recovery must be
   audited against the saved response, never overwrite it or alter its class/source
   IDs. Separate evidence review can change draft labels only before predictions.
   No paid retries or automatic assignment of failures to AMBIGUOUS. If complete
   defensible reviewed labels cannot be frozen, stop before classification.
2. Freeze all50 reviewed target rows, class/ambiguity counts and hashes before any
   classifier prediction. Then at most100 fresh paired calls to Singapore
   `qwen3.5-plus-2026-04-20`,50 per arm, max2048 output tokens each; fixed question-ID
   order, even index v5.2.1 then v5.3, odd index reversed. Requests contain identical
   question/source payloads, not labels, reviewer reasons, IDs or retrieval metadata.
   Keep system prompts, schema and strict coverage/citation validator unchanged.

Both stages: temperature0, thinking off, JSON-object responses, SDK retries0.
Use existing durable machinery with new isolated checkpoints and pinned inputs,
prompts, requests, dependencies, protocol and execution contracts; no new framework.
Exclusive lock, pending before POST, durable raw/usage before parsing; recover saved
responses without new calls, never repeat completed requests, stop on unknown charge.
Classifier format failures are terminal null outcomes, not refusals or successes.
No post-result repair, answers, judge, prompt tuning, automatic recharge or serving.

## Metrics and interpretation fixed before labels

Report counts first: all50 decisions/failures, actual target counts, binary agreement
and confusion with all binary targets (including failed predictions) as denominator,
per-class recalls and failure counts, ambiguous decisions descriptive only. Separately
report common-valid binary actual denominators/classes and paired fixes/regressions,
all50 admission coverage, admitted-binary proxy agreement and ambiguous admissions.
Retain null failures with possible binary-agreement lower/upper bounds; undefined
denominators are null, never zero. Include every charged outcome in stage/arm usage,
cost and provider latency. Report any known correlation; no IID significance claim.

Labels remain same-family Qwen draft plus current-assistant review: AI proxy, not
independent human/expert gold or real answer accuracy. Report whether each binary
class reaches the original five-example support rule; no relabeling/resampling to
meet it. Regardless of class counts or scores, no formal pass/production gate applies
to this development comparison. New apparent gains do not independently justify
online promotion; prior retrieval exposure and reviewer/model coupling remain.

## Proposed authorization, not execution

Only question/full Top14 bodies go to the two named Singapore models. No labels,
review reasons or secrets in classifier inputs. At most150 calls total; no retries.
Proposed accounting caps: annotation CNY10, paired classification CNY15, total CNY25;
no transfer between stage caps or automatic continuation after a cap/unknown charge.
The generated preparation freezes actual request-bound planning estimates.

[Official International pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
checked2026-10-16, <=256K input band: annotation USD0.5/3 per million input/output
tokens; classification USD0.4/2.4. Fixed accounting FX7.5, no discounts. UTF-8 bytes
plus512 framing is a planning proxy, not certified token counting or a guaranteed
invoice cap. Require fresh explicit authorization for both stages before paid work;
all previous execution permissions are exhausted. Labels and predictions do not yet
exist, and runner readiness must not be described as measured model improvement.
