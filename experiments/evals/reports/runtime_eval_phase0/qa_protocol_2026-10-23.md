# Phase 2 — complete HTTP answering and refusal comparison

Status: executable preparation, **not execution consent or a quality result**.
Phase 1 measured retrieval, not answer correctness or refusal quality. No default
policy promotion, deployment, real tickets, new dependency or daily-service change.

## Fixed comparison and data boundary

- Same phase-0 archival managed corpus/tenant/category and generation metadata SHA
  as [the parent protocol](evaluation_protocol_2026-10-23.md). Corpus embedding
  provenance remains unknown. DEV has been reused: not fresh independent testing.
- Exactly 310 generation questions (`question.rstrip()`), sorted by ID; historical
  160 answerable / 150 impossible labels are **strata, not present-context truth**.
  Do not use reference answers, qrels, or historical phase-1 outcomes to build
  provider payloads or tune a policy. Local file only, no automatic download.
  Loading the hash-bound JSON necessarily parses unused fields; only DEV ID,
  question and historical stratum are retained. No reference-answer projection.
- D: real loopback TCP `/rag/ask`, server-decoded eval-tenant JWT, Dense Top3,
  distance .9, unchanged Dense prompt. H: same route/JWT, Hybrid Top14, frozen
  `v5_3_1_experimental`, no distance argument, unchanged full-context prompt.
  Alternate D/H and H/D order by question index. Generation both non-thinking,
  max1024. This is a **whole-system comparison**, not refusal-only attribution.
- Dedicated serial container, at most 4CPU/5GiB, read-only archival source mount;
  persistent native Docker volume for the owned index copy, HTTP SQL, and receipts.
  Fresh database migrated to head, then copy only documents/document_chunks.
  No source SQL migration, uploads, daily API, .env mount or credentials in argv.

## Blind evidence and answer review

Exact prompts/schemas are `EVIDENCE_PROMPT`, `ANSWER_PROMPT` and
`validate_verdict` in [runtime_qa.py](../../runtime_qa.py). Model for both judge
phases: **Beijing qwen3.6-plus-2026-04-02**, temperature0, thinking false,
max2048, 30-second HTTP phase timeouts, retries0, redirects off.

1. Capture all 620 HTTP outcomes and their **full actual selected contexts**.
2. Project question plus ordered Source N/full content only. No arm, gate, answer,
   historical label, gold answer or document metadata in evidence packets. Packet
   IDs are hashes. Exact duplicate packets share one judgment, never partial matching.
   Source-count differences can reveal an arm; this is blinding of outputs and
   explicit identity, not a guarantee that arm is impossible to infer.
3. Judge evidence SUFFICIENT / INSUFFICIENT / AMBIGUOUS, checking requested
   outcomes, applicability, versions/time, missing conditions and conflicts.
   Assistant reviews **every unique packet** without opening output/mapping files,
   then explicitly freezes reviewed verdicts and reasons. Empty context cannot be
   sufficient. Do not force ambiguity into either class. No domain-expert claim.
   An HTTP failure before context capture is marked unobserved/ambiguous locally,
   not sent as a fabricated empty-context judgment; failures still count as zero.
4. Only after evidence freeze, reveal answers and judge non-exact-refusal HTTP200
   outputs. Exact fixed refusals and HTTP failures need no answer-model call.
   Paraphrased refusals still undergo behavior classification. Judge support,
   completeness and actual citation entailment, not just ID syntax. Partial answers
   with disclaimers are answers. Assistant reviews/finalizes all unique verdicts.
5. Raw requests/responses, private mapping, packets, reviews, SQL and answers remain
   local ignored artifacts. Report command does not publish anything. Any public
   private-eval-derived result needs publication consent separately.

This is Qwen-family AI proxy plus assistant review, **not independent expert gold**.
Even the different pinned judge model shares a family/provider with the system.
Correctness means grounded question-complete correctness against supplied evidence;
it does not certify real-world truth outside that corpus. No gold reference answers
are sent. No per-question adaptation, prompt repair, judge retry or extra model call.

## Metrics fixed before DEV reads

- Primary H−D: **grounded complete answer rate over all 310**. Requires ANSWER,
  all material claims supported, every necessary requirement addressed correctly,
  and valid supporting citations. Refusals and HTTP/schema failures score zero.
  This prevents an always-refuse system getting a perfect main score.
- Also publish answered-only correctness **together with coverage**, failures,
  ambiguous behavior, evidence class counts, refusal on sufficient context and
  answer on insufficient context (each with explicit class denominators).
  A complete-answer judgment conflicting with frozen insufficient/ambiguous
  evidence is reported as a review conflict and scores zero, not silently relabeled.
- Context-appropriate success (correct answer if sufficient, refusal if insufficient)
  is secondary, with ambiguity non-success. Context differs between arms, so this
  conditional score alone cannot establish better end-to-end service.
- Gate admission and final behavior are different: report gate availability,
  false allow/refuse counts separately. Dense has no classifier gate. Model/format
  failure is not an evidence-based refusal. Historical impossible answers are not
  automatically hallucinations. An answer can be partial but fully supported and
  still fail completeness; safety and usefulness are separate.
- All metrics also stratified by historical 160/150. Full310 paired bootstrap on
  main binary outcomes: seed20261023, 10000 paired samples, sorted indices249/9749.
  Lower bound >0 establishes main paired improvement on this reused benchmark only;
  otherwise report improvement not established. No automatic acceptance/promotion.
- HTTP wall-clock P50/P95 include failure outcomes. Resume reconstruction latency
  is missing, not a new real-time observation; report missing n. Provider-attempt
  timings remain in raw receipts. Persist first-run/ordering to interpret cold start.
  Loopback client timeout600s allows cold build plus four model stages; external
  HTTP phase timeouts remain 30s (rerank15s), not a strict total deadline.

## Complete budget, not service-only price

Rates verified 2026-10-23 against [official pricing](https://help.aliyun.com/zh/model-studio/model-pricing).
Rates below are CNY per million input/output tokens, without cache/free quota:

| Stage | Region/model | Calls at most | Input/output ceilings | Rates | Full reserve CNY |
| --- | --- | ---: | --- | --- | ---: |
| Embedding | Beijing text-embedding-v4 | 620 | 8192/0 | .5/0 | 2.539520 |
| Rerank | Singapore qwen3-rerank | 310 | 120000/0 | .74942/0 | 27.878424 |
| Classifier | Singapore qwen3.5-plus-2026-04-20 | 310 | 65536/2048 | 2.936/17.614 | 70.83102208 |
| Generation | Beijing qwen3.5-plus | 620 | 65536/1024 | .8/4.8 | 35.553280 |
| Evidence judge | Beijing qwen3.6-plus-2026-04-02 | 620 | 65536/2048 | 2/12 | 96.501760 |
| Answer judge | Beijing qwen3.6-plus-2026-04-02 | 620 | 65536/2048 | 2/12 | 96.501760 |

Service full reserve **136.80224608**, evidence **96.501760**, answer
**96.501760**; total **329.80576608** across at most **3100** provider calls.
The original **150 + 105 + 105 = 360** proposal is superseded on 2026-10-24 by
execution accounting caps **35 + 22 + 23 = 80** (service/evidence/answer).
This is a deliberately smaller stop budget, not a promise all310 will finish,
an expected invoice or a provider hard billing guarantee. Each send must fit its
phase's remaining cap including its full next-call reservation. No automatic cap
increase or unused-budget transfer. Original full-profile theoretical cost above
remains unchanged, not rewritten as an observed estimate. Exact duplicates/refusals can reduce
judge calls, never increase the fixed limits. Assistant review adds no provider call
here, but does consume the assistant session's normal quota.

The pinned local Qwen3 reference tokenizer is not an attested tokenizer for any
cloud endpoint. E/R reuse the parent reference profile; chat counts full messages
plus 25% +128 framing tokens, maximum65536. Both raw byte bounds and estimates
must pass. Never truncate/drop a source or question. Actual usage above a
reservation, missing usage, unknown non-stop finish, unknown billing or pricing change
stops the run. Reconfirm pricing and get **new full phase-2 consent** before live use.
Exception for service answer generation only: a fully received valid envelope with
`finish_reason=length`, exactly1024 completion tokens and valid bounded usage is
accounted once and recorded as a failed HTTP outcome. Never return or grade its
partial answer, retry it, increase its token limit, or remove it from denominators.
Classifier/judge truncation and malformed or unknown usage still stop the run.

## Recovery and commands

`python -m experiments.evals.runtime_qa_runner` only prints help. No default execute.
In the isolated image, local `prepare --metadata ... --snapshot ... --tokenizer ...
--output ...` explicitly reads DEV and binds code/dependencies/questions/assets.
This preparation PR does **not** run that command on private DEV.

Approval JSON must bind prepared SHA, scope `DEV310_HTTP_AND_TWO_BLIND_REVIEW_PHASES`,
max_provider_calls3100, accounting_limits_cny(service35/evidence22/answer23),
accept_reference_estimate/current_prices_reconfirmed/allow_resume_completed_receipts_only
all true, and an actual human_approval_reference. File contents do not manufacture consent.

`execute --phase service|evidence|answer --output ... --approval ...
--approval-sha256 ...` consumes only four selected regional keys on stdin, named
embedding/rerank/classifier/generation. Beijing generation key is used for the
Beijing judge. Do not mount/persist credentials or print stdin. Region URLs are
fixed, never inherited from caller env. No automatic paid connectivity probe.

For each phase, run only once or explicitly continue after inspecting checkpoint
state. Each request has a unique question/arm/stage ID, durable request+pending
before send, full response+usage and completion receipt before release. Completed
stages replay exact recorded bodies. HNSW selections are persisted **before** the
next model call, not re-queried to reconstruct history. A lost HTTP response is
reconstructed through the same service with saved selections/model receipts.
No additional paid call for completed stages. Native-volume per-case ledgers avoid
rescanning all large provider payloads on every call.
Ordinary provider/index getters are blocked for the entire local server lifetime;
only the active case may attach guarded clients, including during late-worker shutdown.

Unknown pending, stale locks, guard/structure/billing failures require explicit
audit; no automatic deletion or retry, and timeouts do not imply free calls.
Local-copy initialization failure can require rebuilding **only owned_work** after
target verification; provider receipts stay intact. Partial write/hash mismatch
requires audit. A format-invalid classifier is a known HTTP failure if its valid
wire envelope/usage is captured; it counts as failure, not refusal. Malformed judge
output stops before a report; do not silently replace it with a paid retry.
Failures print a bounded type/filename/function/line chain, never exception text,
source lines, local variables, HTTP headers, credentials or provider body text.

Use `freeze --kind evidence|answer --output ... --reviewed ... --reviewer ...`
only after actual assistant review of all packets; it does not perform that review.
Then `report --output ...` requires both frozen complete review sets and all310
pairs. No partial result may masquerade as a complete acceptance report.
