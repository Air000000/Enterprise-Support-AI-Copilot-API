# G2 Rerank-Informed Document Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate G2-A, where the first five candidate documents are admitted by the already-computed global rerank order rather than Dense order, while all G1 downstream document-local evidence mechanics remain fixed.

**Architecture:** Add a focused offline G2 evaluation module that consumes the historical E0 Dense Top100 and an injected/replayed global `RerankResult`. G1 and G2-A share the same Dense input and downstream document-local expansion / merged rerank / Top16 contracts; only the ranking used for document admission differs. Formal evaluation uses a fresh preregistered TRAIN sample and a two-method blinded evidence-sufficiency review.

**Tech Stack:** Python 3.11, pytest, dataclasses, existing TechQA frozen artifacts, existing `qwen3-rerank` provider adapter.

**Spec:** `docs/superpowers/specs/2026-09-08-g2-rerank-informed-document-admission-design.md`

## Global Constraints

- Baseline commit: `7caf1a1245bdc73698c9fb218c863425ae3b7f8f`.
- Work only on `feat/g2-rerank-informed-admission`.
- Historical E0 Dense Top100 remains the input candidate pool.
- Global reranker remains `qwen3-rerank` with instruction exactly `Rank the candidate passages by relevance to resolving the technical support query.`.
- Candidate document limit remains exactly `5`.
- Candidate pool safety ceiling remains `500` chunks.
- Evidence limit remains `16`; final context limit remains `16`.
- G2-A performs no continuity/sibling-preservation change.
- G2-A performs no query rewrite, BM25/Hybrid addition, alternate reranker, per-document rerank, second-pass rerank, embedding, generation, or judge call.
- The global Top100 rerank result is injected/reused; G2 orchestration must not issue a second equivalent global rerank call.
- No Q346/Q492-specific logic.
- Previous G1 30 cases and prior manually labeled evidence cases are excluded from fresh formal sampling.
- Paid execution cannot begin until sample, contracts, and GO/NO-GO gates are frozen in-repo.

---

## File map

**Create**

- `experiments/evals/eval_techqa_g2_admission.py` — G2-A admission/result validation and pure G1-vs-G2 comparison orchestration.
- `tests/test_eval_techqa_g2_admission.py` — deterministic unit/contract tests for the new orchestration.
- `experiments/evals/reports/g2_rerank_informed_admission/preregistration_v1.json` — frozen fresh-sample/provider/metric/gate contract, created only after deterministic offline generation.
- `experiments/evals/reports/g2_rerank_informed_admission/README.md` — experiment scope, artifact lineage, and result boundary.

**Modify only if required by typing reuse**

- `experiments/evals/eval_techqa_generation.py` — behavior-preserving type generalization for `select_candidate_document_ids()`; no G1 semantic change.

**Reuse without changing behavior**

- `experiments/evals/rerankers/qwen3_reranker.py` — `RerankResult`, `RerankedCandidate`, `rerank_candidates`.
- `experiments/evals/eval_techqa_generation.py` — `G0RetrievedChunk`, `select_candidate_document_ids`, `build_document_local_candidate_pool`, `select_document_local_evidence`, `assemble_selected_evidence_context`.
- `experiments/evals/reports/g1_document_local/evidence_sufficiency_30case_preregistration_v1_1.json` — prior 30-case exclusion set.
- `experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl` — prior manual-label exclusion set.

---

### Task 1: Pure rerank-informed document admission

**Files:**
- Create: `experiments/evals/eval_techqa_g2_admission.py`
- Test: `tests/test_eval_techqa_g2_admission.py`
- Modify if necessary: `experiments/evals/eval_techqa_generation.py`

**Interfaces:**
- Consumes: `RerankResult.results: tuple[RerankedCandidate, ...]`.
- Produces: `select_rerank_informed_document_ids(shared_rerank: RerankResult, *, limit: int) -> tuple[str, ...]`.

- [ ] **Step 1: Write RED tests for rerank-order admission**

```python
from experiments.evals.eval_techqa_g2_admission import (
    select_rerank_informed_document_ids,
)
from experiments.evals.rerankers.qwen3_reranker import (
    RerankResult,
    RerankedCandidate,
)


def _shared_rerank(*doc_ids: str) -> RerankResult:
    return RerankResult(
        results=tuple(
            RerankedCandidate(
                chunk_id=f"c{index}",
                document_id=document_id,
                content=f"content-{index}",
                original_index=index,
                relevance_score=1.0 - index / 100.0,
            )
            for index, document_id in enumerate(doc_ids)
        ),
        request_id="shared",
        total_tokens=123,
    )


def test_rerank_informed_admission_uses_first_unique_docs_in_rerank_order():
    result = _shared_rerank("D3", "D3", "D1", "D5", "D2", "D4", "D6")
    assert select_rerank_informed_document_ids(result, limit=5) == (
        "D3", "D1", "D5", "D2", "D4"
    )


def test_rerank_informed_admission_requires_positive_limit():
    with pytest.raises(ValueError):
        select_rerank_informed_document_ids(_shared_rerank("D1"), limit=0)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py -q
```

Expected: collection/import failure because the new G2 module/function does not exist.

- [ ] **Step 3: Implement the minimal selector**

```python
from experiments.evals.rerankers.qwen3_reranker import RerankResult


def select_rerank_informed_document_ids(
    shared_rerank: RerankResult,
    *,
    limit: int,
) -> tuple[str, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    seen: set[str] = set()
    selected: list[str] = []
    for candidate in shared_rerank.results:
        document_id = str(candidate.document_id)
        if document_id in seen:
            continue
        seen.add(document_id)
        selected.append(document_id)
        if len(selected) >= limit:
            break
    return tuple(selected)
```

- [ ] **Step 4: Run focused tests and existing G1 selector tests**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py tests/test_generation_eval_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
git commit -m "feat: add rerank-informed document admission"
```

---

### Task 2: G2-A diagnostic path reusing the shared global rerank

**Files:**
- Modify: `experiments/evals/eval_techqa_g2_admission.py`
- Test: `tests/test_eval_techqa_g2_admission.py`

**Interfaces:**
- Consumes: historical E0 `Sequence[G0RetrievedChunk]`, injected `RerankResult`, offline document loader, merged reranker.
- Produces:

```python
@dataclass(frozen=True)
class G2AdmissionDiagnostic:
    question_id: str
    question: str
    dense_ranked_chunks: tuple[G0RetrievedChunk, ...]
    candidate_document_ids: tuple[str, ...]
    candidate_pool: tuple[G0RetrievedChunk, ...]
    selected_evidence: tuple[G0RetrievedChunk, ...]
    final_context: tuple[G0RetrievedChunk, ...]
    merged_rerank_invocations: int


def run_g2_admission_diagnostic(
    question_id: str,
    question: str,
    dense_ranked_chunks: Sequence[G0RetrievedChunk],
    *,
    shared_global_rerank: RerankResult,
    candidate_document_limit: int,
    load_document_chunks: Callable[[str], Sequence[G0RetrievedChunk]],
    candidate_pool_max_chunks: int,
    merged_reranker: Callable[..., Any],
    evidence_limit: int,
    max_context_chunks: int,
) -> G2AdmissionDiagnostic:
    ...
```

- [ ] **Step 1: Add RED contract tests**

Tests must prove all of the following with synthetic chunks:

```text
Dense first5 docs != global-rerank first5 docs
G2 candidate docs follow global-rerank order
G2 loads only those five documents
G2 calls only the merged reranker once
G2 never calls a global reranker because the global result is injected
G2 reuses build_document_local_candidate_pool()
G2 reuses select_document_local_evidence()
G2 reuses assemble_selected_evidence_context()
final context preserves merged-rerank order and max16 semantics
invalid budgets fail before loader/reranker side effects
```

Use a fake merged reranker returning a complete deterministic permutation. Include a regression test where Dense order begins `D1,D2,D3,D4,D5,...` but shared rerank begins chunks from `D9,D8,D7,D6,D5,...`; assert G2 admits `D9,D8,D7,D6,D5`.

- [ ] **Step 2: Run focused test file and verify RED**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py -q
```

Expected: FAIL on missing diagnostic dataclass/function.

- [ ] **Step 3: Implement the minimal diagnostic runner**

Implementation sequence must be exactly:

```python
candidate_document_ids = select_rerank_informed_document_ids(
    shared_global_rerank,
    limit=candidate_document_limit,
)
candidate_pool = build_document_local_candidate_pool(
    candidate_document_ids,
    load_document_chunks=load_document_chunks,
    max_chunks=candidate_pool_max_chunks,
)
selected_evidence = select_document_local_evidence(
    question,
    candidate_pool,
    reranker=tracked_merged_reranker,
    limit=evidence_limit,
)
final_context = assemble_selected_evidence_context(
    selected_evidence,
    max_context_chunks=max_context_chunks,
)
```

Do not call `rerank_candidates()` for the global stage inside this function.

- [ ] **Step 4: Run G2 + G1 focused tests**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py tests/test_generation_eval_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
git commit -m "feat: add G2 admission diagnostic path"
```

---

### Task 3: One-variable G1-vs-G2 comparison contract

**Files:**
- Modify: `experiments/evals/eval_techqa_g2_admission.py`
- Test: `tests/test_eval_techqa_g2_admission.py`

**Interfaces:**
- Produces a pure comparison record containing G1/G2 admitted docs and contexts from the same Dense Top100.

```python
@dataclass(frozen=True)
class G1G2AdmissionComparison:
    question_id: str
    dense_candidate_chunk_ids: tuple[str, ...]
    g1_candidate_document_ids: tuple[str, ...]
    g2_candidate_document_ids: tuple[str, ...]
    g1_context: tuple[G0RetrievedChunk, ...]
    g2_context: tuple[G0RetrievedChunk, ...]
```

- [ ] **Step 1: Add RED tests proving causal isolation**

Construct one Dense ranking and one shared global rerank. Assert:

```python
assert g1_candidate_document_ids == select_candidate_document_ids(
    dense_ranked_chunks, limit=5
)
assert g2_candidate_document_ids == select_rerank_informed_document_ids(
    shared_global_rerank, limit=5
)
```

Also assert both branches receive identical values for:

```text
candidate_pool_max_chunks=500
evidence_limit=16
max_context_chunks=16
same offline loader
same merged-reranker contract shape
```

Do not require identical merged-rerank outputs because candidate pools may differ.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py -q
```

- [ ] **Step 3: Implement the comparison function**

The G1 branch must use existing `select_candidate_document_ids(dense_ranked_chunks, limit=5)` and existing downstream helpers. The G2 branch must use the injected global rerank result and the same downstream helpers. Keep the comparison offline and dependency-injected.

- [ ] **Step 4: Run focused tests and static checks**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py tests/test_generation_eval_runner.py -q
conda run -n enterprise-g1-verify ruff check experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
```

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
git commit -m "test: freeze G1 versus G2 causal contract"
```

---

### Task 4: Deterministic fresh-sample preregistration builder

**Files:**
- Modify: `experiments/evals/eval_techqa_g2_admission.py`
- Test: `tests/test_eval_techqa_g2_admission.py`
- Create after offline generation: `experiments/evals/reports/g2_rerank_informed_admission/preregistration_v1.json`

**Interfaces:**

```python
G2_SAMPLE_SEED = "techqa-g2-rerank-informed-admission-v1"
G2_SAMPLE_SIZE = 30


def select_g2_preregistered_cases(
    cases: Sequence[TechQAGenerationCase],
    *,
    e0_trace_by_question_id: Mapping[str, Sequence[G0RetrievedChunk]],
    relevant_document_by_question_id: Mapping[str, str],
    excluded_question_ids: Collection[str],
    seed: str = G2_SAMPLE_SEED,
    sample_size: int = G2_SAMPLE_SIZE,
) -> tuple[str, ...]:
    ...
```

- [ ] **Step 1: Add RED tests for eligibility and deterministic selection**

Tests must show:

- impossible cases excluded;
- non-TRAIN cases excluded;
- missing E0 trace excluded;
- formal relevant document absent from Dense Top100 excluded;
- formal relevant doc may be outside Dense first5 and the case remains eligible;
- prior exclusion IDs are removed before hashing;
- sorting is exactly SHA256 of `seed + ':' + question_id`, then `question_id` as tiebreaker;
- same inputs produce exactly the same 30 IDs;
- insufficient eligible population raises `ValueError` rather than silently shrinking/resampling.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py -q
```

- [ ] **Step 3: Implement deterministic eligibility/selection**

No random module or shuffle. Use `hashlib.sha256` only.

- [ ] **Step 4: Build the exclusion set offline**

Programmatically union:

```text
historical G0 12-case pilot IDs
G1 preregistration selected_case_ids (30)
question_id values from reports/r1_evidence_audit/evidence_labels.jsonl
explicit G1 forensic IDs not already covered
```

Persist the exact exclusion count and SHA256 of the sorted exclusion list in the preregistration artifact.

- [ ] **Step 5: Generate the preregistration JSON with no provider calls**

The artifact must freeze at minimum:

```json
{
  "experiment_id": "techqa_g2_rerank_informed_admission_v1",
  "artifact_state": "PREREGISTRATION_BEFORE_PAID_EXECUTION",
  "baseline_commit": "7caf1a1245bdc73698c9fb218c863425ae3b7f8f",
  "sample_seed": "techqa-g2-rerank-informed-admission-v1",
  "sample_size": 30,
  "selected_case_ids": [],
  "eligibility_rule": {},
  "exclusion_contract": {},
  "g1_contract": {},
  "g2_contract": {},
  "rerank_contract": {},
  "provider_call_budget": {
    "shared_global_calls_per_case": 1,
    "g1_merged_calls_per_case": 1,
    "g2_merged_calls_per_case": 1,
    "max_real_rerank_calls": 90,
    "max_embedding_calls": 0,
    "max_generation_calls": 0,
    "max_judge_calls": 0
  },
  "go_no_go_rule": {}
}
```

`go_no_go_rule` must exactly freeze the spec gates before any provider call.

- [ ] **Step 6: Run tests, JSON parse check, and commit**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py -q
conda run -n enterprise-g1-verify python -m json.tool experiments/evals/reports/g2_rerank_informed_admission/preregistration_v1.json > $null
```

```bash
git add experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py experiments/evals/reports/g2_rerank_informed_admission/preregistration_v1.json
git commit -m "eval: preregister fresh G2 admission comparison"
```

---

### Task 5: Offline preflight and call-graph tripwires

**Files:**
- Modify: `experiments/evals/eval_techqa_g2_admission.py`
- Test: `tests/test_eval_techqa_g2_admission.py`
- Create: `experiments/evals/reports/g2_rerank_informed_admission/README.md`

**Interfaces:**
- Preflight accepts only frozen E0 traces, frozen corpus chunk loader, and replay/fake rerank results.
- It must be runnable with no network/API credentials.

- [ ] **Step 1: Add RED tripwire tests**

Tests must fail if preflight attempts any of:

```text
live Dense/Chroma query
provider global rerank
generation
judge
embedding
```

Also verify per case:

```text
Dense Top100 count/identity preserved
G1 admission = Dense first5 unique docs
G2 admission = replayed shared-rerank first5 unique docs
candidate pool <= 500
final context <= 16
no duplicate chunk IDs
all loaded chunks belong to admitted documents
```

- [ ] **Step 2: Implement minimal offline preflight summary**

Persist descriptive capacity stats only; do not tune thresholds from them.

- [ ] **Step 3: Run preflight twice and compare deterministic fingerprints**

Fingerprint canonical per-case fields:

```text
question_id
Dense chunk IDs
G1 admitted doc IDs
G2 admitted doc IDs
G1 candidate chunk IDs
G2 candidate chunk IDs
```

Two offline runs must produce identical SHA256.

- [ ] **Step 4: Update experiment README**

Document:

- G1 is still the reference/evaluated predecessor;
- G2-A is not serving code;
- previous 30 G1 cases are excluded from confirmation;
- G2-A changes only admission ranking;
- continuity remains explicitly out of scope;
- paid execution is authorized only after preflight passes.

- [ ] **Step 5: Run full local verification before paid execution**

```powershell
conda run -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py tests/test_generation_eval_runner.py -q
conda run -n enterprise-g1-verify ruff check experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
conda run -n enterprise-g1-verify python -m compileall -q experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
```

- [ ] **Step 6: Commit**

```bash
git add experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py experiments/evals/reports/g2_rerank_informed_admission/README.md
git commit -m "eval: add G2 offline preflight gates"
```

---

### Task 6: Paid rerank execution

**Files:**
- No code changes unless execution exposes a contract bug that reproduces offline first.
- Raw paid artifacts stay outside Git; compact frozen summary hashes may be recorded later.

**Interfaces:**
- Per case real call graph is exactly `1 shared global + 1 G1 merged + 1 G2 merged = 3` rerank calls.

- [ ] **Step 1: Verify exact preregistration and preflight fingerprints before provider access**

Abort if worktree is dirty, HEAD differs from the preregistered execution commit, sample differs, or any budget differs.

- [ ] **Step 2: Execute only the 30 frozen cases**

Provider failure policy:

```text
STOP immediately
preserve partial artifacts
no retry
no model change
no candidate truncation
no case replacement
```

- [ ] **Step 3: Validate execution ledger**

Required totals on success:

```text
cases = 30
shared_global_calls = 30
g1_merged_calls = 30
g2_merged_calls = 30
total_real_rerank_calls = 90
embedding_calls = 0
generation_calls = 0
judge_calls = 0
```

Record token usage separately for each of the three call classes.

- [ ] **Step 4: Produce method contexts and sealed A/B mapping**

Do not expose method labels to the evidence reviewer.

---

### Task 7: Two-method blinded evidence-sufficiency review

**Files:**
- Raw reviewer artifacts stay outside Git until final compact result is frozen.

**Interfaces:**
- 30 questions × 2 anonymous contexts = 60 review units.
- Claim decomposition is frozen before contexts are opened.

- [ ] **Step 1: Freeze material claims from question + gold answer only**

Reuse G1 support semantics exactly. Do not inspect A/B contexts while decomposing claims.

- [ ] **Step 2: Build sanitized A/B review packets**

Allowed fields:

```text
opaque unit_id
question
frozen claim IDs/text/required identifiers
anonymous context passages P1..Pn
```

Forbidden fields:

```text
G1/G2 names
raw document/chunk IDs
rerank scores/ranks
method provenance
provider request IDs
```

- [ ] **Step 3: Review every claim-context pair**

Labels:

```text
EXPLICIT_SUPPORT
COMPOSABLE_SUPPORT
INFERENCE_ONLY
ABSENT
UNRESOLVED
```

- [ ] **Step 4: Deterministically aggregate anonymous contexts**

Compute `COMPLETE/PARTIAL/INSUFFICIENT/UNRESOLVED` and per-case claim coverage before unblinding.

---

### Task 8: Unblind, decision, and closeout

**Files:**
- Create: `experiments/evals/reports/g2_rerank_informed_admission/final_decision.md`
- Modify: `experiments/evals/reports/g2_rerank_informed_admission/README.md`
- Modify root/eval README only if the result materially changes the documented evaluation lineage.

**Interfaces:**
- Input is the frozen anonymous aggregation plus sealed mapping.

- [ ] **Step 1: Unblind exactly once after review freeze**

Calculate:

```text
G1/G2 COMPLETE counts
G1/G2 PARTIAL counts
G1/G2 INSUFFICIENT counts
macro mean per-case claim coverage
G2 wins / ties / losses
catastrophic regressions: G1 COMPLETE -> G2 INSUFFICIENT
provider token/call descriptors
```

- [ ] **Step 2: Apply preregistered gates without relaxation**

GO requires all of:

```text
30/30 executed under frozen contract
UNRESOLVED = 0
G2 COMPLETE >= G1 COMPLETE
G2 INSUFFICIENT <= G1 INSUFFICIENT
G2 macro claim coverage >= G1 macro claim coverage
G2 wins > G2 losses
catastrophic regressions = 0
```

- [ ] **Step 3: Write bounded final decision**

If GO, wording is limited to authorization for the next design/evaluation stage. If NO_GO, preserve the result and diagnose before any new architecture change. Never call this production generation uplift or frozen DEV improvement.

- [ ] **Step 4: Run fresh full-suite verification**

```powershell
conda run -n enterprise-g1-verify python -m pytest -q
conda run -n enterprise-g1-verify ruff check experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
```

Expected baseline remains no test failures; any new failure blocks merge.

- [ ] **Step 5: Commit closeout, open PR, review, and merge only after CI success**

```bash
git add experiments/evals/reports/g2_rerank_informed_admission README.md experiments/evals/README.md
git commit -m "docs: close G2 admission experiment"
```

PR title:

```text
Evaluate G2 rerank-informed document admission
```

Do not merge until fresh local verification and GitHub Actions both succeed.

---

## Self-review

- Spec coverage: every frozen invariant, sampling rule, call budget, leakage rule, metric, and GO/NO-GO gate maps to Tasks 1-8.
- Placeholder scan: no implementation step depends on `TBD`, `TODO`, or outcome-dependent tuning.
- Type consistency: G2 consumes existing `RerankResult`, `G0RetrievedChunk`, and G1 downstream helper contracts; new names are fixed in Tasks 1-3.
- One-variable check: G1 uses Dense order for document admission; G2-A uses shared global-rerank order. Candidate document count, document expansion, merged rerank, Top16, context cap, model, and instruction stay fixed.
- Anti-overfitting check: Q346/Q492 are diagnostic only; previous G1 30 and prior manual evidence-label cases are excluded from fresh formal sampling.
- Cost check: formal 30-case comparison has a hard maximum of 90 real rerank calls and zero embedding/generation/judge calls.
