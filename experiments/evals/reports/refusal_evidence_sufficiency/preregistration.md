# Portfolio-v1 Refusal / Evidence-Sufficiency Preregistration

Date: 2026-09-22  
Status: **PREREGISTERED DESIGN — NOT YET RUN**  
Scope: refusal / abstention only  
Frozen upstream contract: portfolio-v1 retrieval + context

## 1. Question

The portfolio-v1 retrieval/context line is already frozen:

```text
Dense Top100 + BM25 Top100
    -> equal-weight chunk RRF (k=60)
    -> fused Top100
    -> qwen3-rerank
    -> Flat Top14
```

The unresolved question is:

> Given the frozen Top14 context, is there enough evidence to answer the user's technical-support question without relying on unsupported outside knowledge?

This stage treats refusal as an **evidence-sufficiency classification problem**.

It does not inherit the historical rule:

```text
dense_top1_distance > 0.9 -> refuse
```

because Dense Top1 distance describes only one first-stage retrieval signal, while the frozen portfolio candidate is Hybrid + rerank.

## 2. Historical behavior being replaced

The historical generation harness performs a hard pre-generation refusal when the Dense Top1 distance exceeds 0.9. The online Dense runtime uses the same style of heuristic.

That heuristic was useful as an early engineering guardrail, but it is not semantically aligned with the frozen portfolio-v1 retrieval route:

- BM25 can recover lexical evidence that Dense alone misses;
- RRF combines multiple candidate rankings;
- qwen3-rerank changes the final ordering;
- Flat Top14 is the final evidence unit presented downstream.

Therefore the final refusal decision must be based on the **final evidence context**, not on a privileged Dense-only score.

## 3. Important TechQA label boundary

TechQA's original paper defines answerable / unanswerable relative to the candidate Technotes supplied for the reading-comprehension task. The NVIDIA RAG-Eval conversion exposes 610 answerable and 300 impossible records, and impossible records have empty provided `contexts`.

For this project, however, the RAG retriever searches the project's frozen TechQA corpus rather than replaying the original 50-document candidate list.

Therefore:

> `is_impossible=true` is a benchmark abstention label, but it is not automatically treated as proof that no supporting evidence can exist anywhere in the project's current retrieval corpus.

This is why the existing abstention-audit vocabulary already allows:

- `corpus_supported_impossible`
- `semantic_abstention`
- `true_unsafe_answer`
- `correct_abstain`

Dataset answerability and final-context evidence sufficiency must be reported separately.

References:

- TechQA paper: https://arxiv.org/abs/1911.02984
- NVIDIA TechQA-RAG-Eval: https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval

## 4. Frozen classifier semantics

Classifier identifier:

```text
evidence_sufficiency_v1
```

Input:

- user question;
- exactly the frozen `flat_rerank_top14_v1` context;
- source boundaries / source IDs already available from the frozen context.

The classifier must not receive:

- gold answer;
- `is_impossible`;
- relevant document IDs;
- evidence labels;
- qrels;
- question ID as a semantic feature.

Output schema:

```json
{
  "decision": "SUFFICIENT | INSUFFICIENT",
  "reason": "short evidence-based explanation",
  "supporting_source_ids": ["Source 1", "Source 4"]
}
```

Decision rule:

### SUFFICIENT

Return `SUFFICIENT` only when the provided context contains enough evidence to answer the material parts of the question directly and specifically.

Topic similarity alone is insufficient.

### INSUFFICIENT

Return `INSUFFICIENT` when any of the following applies:

- context is only topically related;
- the requested version / error / procedure / configuration is not supported;
- evidence is partial and answering would require a material unsupported inference;
- evidence is contradictory or too ambiguous to resolve;
- the context contains symptoms but not the requested resolution;
- the answer would require outside technical knowledge not present in the context.

When uncertain, prefer `INSUFFICIENT`.

## 5. Classifier prompt contract

The first paid probe must use one frozen classifier prompt.

The prompt must instruct the model to judge **evidence sufficiency**, not answer correctness and not semantic similarity.

Required language-level semantics:

```text
You are an evidence-sufficiency gate for a technical-support RAG system.

Decide whether the supplied Context is sufficient to answer the Question
without relying on outside knowledge.

SUFFICIENT requires direct support for the material claims needed by the
answer. Topic relevance, keyword overlap, or a related product/version alone
is not enough.

If the evidence is partial, ambiguous, conflicting, or would require a
material unsupported inference, return INSUFFICIENT.

Do not answer the user's question.
Return only the required structured decision.
```

The exact serialized prompt, model, temperature, region, and output parser must be frozen in the run manifest before the first paid classification result is inspected.

No prompt patching is allowed after results are unblinded within the same gated sample.

## 6. Phase A — zero-provider preflight

Before any classifier call:

1. verify current Git SHA and frozen retrieval/context contract;
2. verify `freeze.json` still says:
   - Dense K=100
   - BM25 K=100
   - RRF k=60
   - fused K=100
   - reranker=`qwen3-rerank`
   - context policy=`flat_rerank_top14_v1`
   - TopK=14
3. verify the 54-case evidence audit still contains:
   - 35 answer-bearing hits at Top14
   - 19 answer-bearing misses at Top14
4. construct classifier input records without exposing evidence labels to the classifier path;
5. add tripwire tests proving gold answer / qrels / evidence labels cannot enter classifier input;
6. perform zero provider calls.

Exit:

```text
REFUSAL_PREFLIGHT=PASS
PROVIDER_CALLS=0
DEV_ARTIFACT_OPENED=NO
```

## 7. Phase B — 54-case mechanism probe

Population:

- the existing 54 usable TRAIN evidence-audit cases;
- final context = frozen Flat Top14;
- no retrieval, rerank, generation, or judge rerun.

Ground-truth proxy for this **mechanism probe only**:

```text
evidence_sufficient = at least one label=2 answer-bearing chunk is present in Flat Top14
```

Known class counts before classifier execution:

- sufficient: 35
- insufficient: 19

This is not a fresh held-out test. The 54-case audit already influenced context-policy development. The probe answers only whether an explicit evidence-sufficiency classifier is mechanically aligned with the audited evidence labels.

### Phase B metrics

Report:

- accuracy;
- balanced accuracy;
- sufficient recall;
- insufficient recall;
- sufficient -> insufficient count (over-refusal);
- insufficient -> sufficient count (unsafe-pass proxy);
- confusion matrix;
- provider calls / tokens / latency.

### Phase B preregistered gate

All conditions must pass:

```text
balanced_accuracy >= 0.80
sufficient_recall >= 0.85
insufficient_recall >= 0.70
sufficient_to_insufficient <= 5
```

Rationale:

- over-refusal must remain bounded because 35 cases already contain audited answer-bearing evidence;
- insufficient recall must be materially better than chance before paying to extend the policy;
- this is a portfolio engineering gate, not a statistical theorem.

If any condition fails:

```text
DECISION=REJECT_EVIDENCE_SUFFICIENCY_V1
```

Stop. Do not tune the prompt on these 54 labels and rerun under the same experiment label.

If all conditions pass:

```text
DECISION=ADMIT_TRAIN_ABSTENTION_PROBE
```

## 8. Phase C — TRAIN impossible abstention probe

Phase C is admitted only after Phase B passes.

Population:

- all 150 TRAIN `is_impossible=true` records.

Required upstream processing:

```text
question
 -> frozen Hybrid candidate construction
 -> qwen3-rerank
 -> Flat Top14
 -> evidence_sufficiency_v1
```

This phase may require new query embeddings and reranker calls because the historical retrieval-qrels artifacts did not cover impossible records.

No generation is run in Phase C.

### Required snapshot fields

Persist enough information to audit refusal later:

- question_id
- question
- Dense Top100 chunk IDs and distances
- BM25 Top100 chunk IDs
- fused Top100 chunk IDs and RRF scores where available
- reranked Top100 chunk IDs
- reranker relevance scores
- final Top14 chunk IDs
- final Top14 document IDs
- final Top14 contents
- evidence-sufficiency decision
- supporting source IDs
- classifier reason
- retrieval/rerank/classifier latency
- provider request IDs and token usage where available

This intentionally improves on the older C1 result schema, which preserved the reranked identities but not the reranker relevance scores.

## 9. Interpreting impossible cases

For Phase C, report two layers separately.

### Dataset-level abstention agreement

```text
dataset_abstention_rate =
classifier INSUFFICIENT / 150 impossible records
```

Do not call every `SUFFICIENT` decision a hallucination.

### Disagreement audit

Every impossible case classified `SUFFICIENT` must be reviewed using the existing abstention-audit categories.

At minimum classify each disagreement as one of:

- `corpus_supported_impossible`
- `semantic_abstention`
- `true_unsafe_answer`
- `correct_abstain`
- `needs_review`

The purpose is to distinguish:

1. benchmark-label disagreement caused by changed retrieval scope;
2. classifier error;
3. genuinely corpus-supported evidence.

No generation is needed to perform this context-level audit.

## 10. Phase C gate

Phase C is a diagnostic/label-compatibility stage, not the final refusal promotion gate.

Do **not** set a final abstention threshold from raw `is_impossible` agreement alone.

Exit must produce:

- impossible `SUFFICIENT` count;
- impossible `INSUFFICIENT` count;
- all `SUFFICIENT` disagreements reviewed;
- category counts;
- actual retrieval/rerank/classifier cost;
- a recommendation whether the TechQA impossible label is clean enough for final policy validation.

Possible decisions:

```text
IMPOSSSIBLE_LABEL_COMPATIBLE_FOR_FINAL_GATE
MANUAL_CONTEXT_SUFFICIENCY_SET_REQUIRED
REFUSAL_POLICY_REJECTED
```

## 11. Final confirmation set — only if required

If Phase C reveals non-trivial `corpus_supported_impossible` or ambiguous cases, create a fresh TRAIN confirmation set rather than forcing the original impossible label to serve as final truth.

Recommended bounded set:

- 30 answerable TRAIN cases not in the 54-case evidence audit;
- 30 impossible TRAIN cases;
- deterministic hash selection fixed before annotation;
- annotation view shows question + frozen Top14 context only;
- annotator assigns:
  - `SUFFICIENT`
  - `INSUFFICIENT`
  - `AMBIGUOUS`
- gold answer / `is_impossible` may be revealed only during adjudication, not initial sufficiency labeling.

This set is the preferred confirmation surface for a final refusal promotion decision if the benchmark label is not sufficiently aligned with the project's retrieval scope.

## 12. Cost and stop rules

No paid stage may start without a run manifest containing:

- model / provider / region;
- expected calls;
- token/cost estimate;
- hard monetary cap;
- resume/checkpoint path;
- stop condition.

Phase B maximum classifier calls:

```text
54
```

Phase C maximum new records:

```text
150
```

No generation or LLM-as-Judge calls are authorized by this preregistration.

A provider error must be checkpointed and the run must remain resumable.

## 13. DEV governance

DEV remains closed during refusal design.

Prohibited:

- DEV classifier calibration;
- DEV threshold selection;
- DEV disagreement inspection;
- DEV generation.

Only after the complete final contract is frozen may the project run the previously defined:

> frozen DEV generation validation after prior retrieval-side DEV exposure

## 14. Promotion boundary

Passing Phase B does **not** freeze the final refusal policy.

The final refusal contract can be frozen only after:

1. evidence-sufficiency mechanism probe passes;
2. impossible-label compatibility is understood;
3. any required fresh TRAIN confirmation passes;
4. model/prompt/output schema are frozen;
5. cost/latency trade-off is accepted.

Only then may:

```text
freeze.json
unresolved.refusal_policy
```

change from `true` to `false`.

## 15. What this stage is allowed to claim

If successful:

> The final refusal decision is based on explicit evidence sufficiency over the frozen final context rather than a Dense-only distance heuristic.

Not allowed:

- “the classifier knows whether the corpus contains the answer”;
- “TechQA impossible means no answer exists anywhere in the corpus”;
- “the refusal policy is production-safe”;
- “DEV validated” before the final DEV run.

## 16. Immediate next action

Implement **Phase A only** first.

Do not call any provider during implementation/preflight.

Expected stop point:

```text
NEXT_ACTION=STOP_FOR_REFUSAL_PREFLIGHT_REVIEW
```
