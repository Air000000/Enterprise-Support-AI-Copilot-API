# Refusal Phase A Review

Date: 2026-09-22  
Status: **PASS WITH TARGET-COMPATIBILITY AMENDMENT**  
Provider calls: **0**  
DEV artifacts opened: **no**

## 1. Executed preflight result

The first Phase A implementation was executed locally against the SHA-locked
R4 C1 artifacts after main was fast-forwarded to the merged Phase A code.

It was run twice and produced identical outputs:

```text
REFUSAL_PREFLIGHT=PASS
USABLE_CASES=54
SUFFICIENT_CASES=35
INSUFFICIENT_CASES=19
CONTEXT_TOP_K=14
PROVIDER_CALLS=0
DEV_ARTIFACT_OPENED=NO
CLASSIFIER_INPUTS_SHA256=bf0164d096758bf0c4cdda3ea0e106a3914e6e43aa1b627f13e8d6cd7efb9fa9
TARGETS_SHA256=15b355a7fa33c18a9dcf5f26773e11a4db8bdaeaf40fedea520a0d44f1b6e1df
```

This confirmed that the frozen C1 snapshot/results and the evidence-audit
labels still reconstruct the expected Flat Top14 population deterministically.

The two hashes above belong to the **superseded binary target encoding** and
must not be used for the paid Phase B run.

## 2. Target-compatibility issue found before any paid run

The first implementation encoded:

```text
Top14 contains a label=2 chunk -> sufficient
otherwise -> insufficient
```

A zero-provider review of the already-frozen manual labels found that three of
the 54 usable rows have **no label=2 candidate anywhere in the annotation
set**, while their frozen notes explicitly describe evidence distributed
across multiple chunks:

- `TRAIN_Q526` — version header and ReporterPlus row are split across chunks;
- `TRAIN_Q365` — two CVEs and remediation details are split across chunks;
- `TRAIN_Q427` — configuration locations and precedence are split across chunks.

The original evidence audit was chunk-level. The refusal classifier is
context-level and receives all 14 chunks at once. Therefore these three rows
cannot be defended as hard context-level negatives merely because no single
chunk was labeled answer-bearing.

No model output had been produced when this issue was found.

## 3. Amended Phase B target contract

The frozen amendment is:

```text
SUFFICIENT_PROXY
    Top14 contains at least one label=2 chunk

INSUFFICIENT_PROXY
    Top14 contains no label=2 chunk,
    but the frozen annotation set has at least one label=2 chunk elsewhere

AMBIGUOUS_MULTI_CHUNK
    the frozen usable annotation set has no label=2 chunk anywhere
```

Expected counts:

```text
SUFFICIENT_PROXY=35
INSUFFICIENT_PROXY=16
AMBIGUOUS_MULTI_CHUNK=3
BINARY_GATED_CASES=51
TOTAL=54
```

The three ambiguous cases remain in the paid classifier run for descriptive
inspection but are excluded from the binary promotion gate.

## 4. Why this is not post-hoc model tuning

This amendment changes only how already-existing manual evidence annotations
are mapped to the context-level evaluation target.

It does not change:

- retrieval;
- RRF;
- reranker;
- Flat Top14;
- classifier prompt;
- model;
- threshold;
- DEV state.

Most importantly, the amendment was made before any Phase B provider call or
classifier output existed.

## 5. Required rerun

After this amendment is merged, rerun:

```bash
python -m experiments.evals.refusal_evidence_sufficiency
```

The expected structural result is:

```text
REFUSAL_PREFLIGHT=PASS
USABLE_CASES=54
SUFFICIENT_PROXY_CASES=35
INSUFFICIENT_PROXY_CASES=16
AMBIGUOUS_MULTI_CHUNK_CASES=3
GATED_CASES=51
PROVIDER_CALLS=0
DEV_ARTIFACT_OPENED=NO
```

The regenerated target SHA is the only target identity eligible for Phase B.

## 6. Decision

```text
PHASE_A_INFRASTRUCTURE=PASS
TARGET_COMPATIBILITY_AMENDMENT=REQUIRED
PAID_PHASE_B_AUTHORIZED=NO
NEXT_ACTION=MERGE_AMENDMENT_AND_RERUN_PHASE_A
```
