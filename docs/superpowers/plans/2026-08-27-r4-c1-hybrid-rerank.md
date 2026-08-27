# R4 C1 Hybrid + Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-cost frozen Hybrid Dense+BM25 chunk snapshot for 450 TechQA TRAIN queries, then evaluate that exact snapshot with the frozen E1 qwen3-rerank contract under explicit paid-run safety controls.

**Architecture:** C1 has a strict zero-cost / paid boundary. The zero-cost phase reconstructs the frozen TechQA chunk universe, takes the current frozen E0 Dense Top-100 chunk IDs, retrieves BM25 Top-100 chunks, fuses both rankings with equal-weight chunk-ID RRF(k=60), and persists all 450 fused Top-100 candidate pools as one immutable snapshot. The paid phase only reads and validates that snapshot, calls the unchanged E1 reranker, checkpoints each completed query, and computes the pre-registered C1 gate.

**Tech Stack:** Python 3.11, pytest, Ruff, datasets 5.0.1, bm25s 0.3.10, ranx 0.3.21, OpenAI-compatible DashScope client, qwen3-rerank.

**Spec:** `docs/superpowers/specs/2026-08-27-r4-c1-hybrid-rerank-design.md`

## Global Constraints

- Work only on branch `feat/r4-c1-hybrid-rerank`.
- Strict TDD, but only three behavior-level C1 contract tests.
- Do not add field-by-field tests.
- TRAIN only: exactly 450 answerable TechQA queries.
- DEV must not enter any C1 code path.
- Dense input is the current frozen E0 TRAIN Top-100 chunk ranking.
- BM25 input is chunk-level Top-100 using the already frozen R4 chunk-BM25 configuration.
- Each Dense and BM25 source ranking must contain exactly 100 unique chunk IDs.
- Cross-source duplicates are preserved as two independent RRF contributions.
- Fusion is equal-weight chunk-ID RRF with `rrf_k=60`.
- Fusion output is exactly 100 unique chunks.
- No per-document cap.
- No adaptive BM25 depth.
- No source weighting.
- No query routing.
- No RRF tuning.
- Reranker model remains `qwen3-rerank`.
- Reranker instruction remains exactly the E1 instruction.
- Reranker input remains exactly 100 chunks.
- Reranker query normalization remains `rstrip`.
- Singapore provider deployment remains unchanged from E1.
- SDK automatic retry must be disabled with `max_retries=0`.
- Provider-token stop threshold is `13_500_000`.
- Monetary safety envelope is USD `1.50`.
- Completed checkpoint queries must never be sent again.
- An unresolved inflight/uncertain query must not be automatically replayed.
- No provider call is allowed during implementation, tests, snapshot preparation, or preflight.
- The real paid run requires explicit human approval after zero-cost acceptance.
- Do not rerun E1 solely to recreate missing paired diagnostics.
- Do not use broad staging such as `git add .`.

## Existing Components to Reuse

Reuse without re-testing their internals:

- `experiments/evals/ir/rrf.py`
  - `fuse_rrf(rankings, rrf_k=60, top_k=100)`
- `experiments/evals/retrievers/bm25_techqa_chunks.py`
  - `TechQAChunk`
  - `build_techqa_chunks`
  - `TechQAChunkBM25Retriever`
- `experiments/evals/rerankers/qwen3_reranker.py`
  - `RerankCandidate`
  - `rerank_candidates`
  - `DEFAULT_RERANK_MODEL`
  - `DEFAULT_RERANK_INSTRUCTION`
- `experiments/evals/ir/ranx_adapter.py`
  - document-collapse and retrieval metric helpers
- `experiments/evals/eval_techqa_chunk_bm25.py`
  - frozen TechQA / splitter / chunk-count provenance constants where appropriate
- `experiments/evals/reports/e1_rerank/train_metrics.json`
  - frozen E1 aggregate baseline

## File Structure

Create:

- `experiments/evals/eval_techqa_hybrid_rerank.py`
  - all C1 experiment-specific snapshot preparation, paid orchestration,
    checkpoint safety, metrics, CLI and manifests.
- `tests/test_eval_techqa_hybrid_rerank.py`
  - exactly three C1 contract test functions.

Modify:

- `experiments/evals/rerankers/qwen3_reranker.py`
  - set SDK `max_retries=0`.
- `docs/superpowers/specs/2026-08-27-r4-c1-hybrid-rerank-design.md`
  - clarify that missing full E1 per-query evidence must not trigger a paid E1 rerun.
- `experiments/evals/reports/cost_ledger.md`
  - only after the real C1 paid run completes.

C1 local/experimental output directory:

`experiments/evals/reports/r4_c1_hybrid_rerank/`

Expected artifacts:

Zero-cost:
- `train_fused_snapshot.jsonl`
- `train_snapshot_manifest.json`
- `train_preflight.json`

Paid:
- `train_checkpoint.jsonl`
- `train_inflight.json` only while a request is unresolved
- `train_results.jsonl`
- `train_manifest.json`
- `train_metrics.json`
- `comparison.md`

Large candidate-heavy snapshot/checkpoint/result files do not need to be
committed when their SHA256 and provenance are persisted in compact
versioned evidence.

---

### Task 1: Freeze the 450-query Hybrid candidate snapshot

**Files:**
- Create: `experiments/evals/eval_techqa_hybrid_rerank.py`
- Create: `tests/test_eval_techqa_hybrid_rerank.py`

**Interfaces:**

Produces:

```python
@dataclass(frozen=True)
class HybridCandidate:
    chunk_id: str
    document_id: str
    content: str

@dataclass(frozen=True)
class HybridSnapshotRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    dense_chunk_ids: tuple[str, ...]
    bm25_chunk_ids: tuple[str, ...]
    fused_candidates: tuple[HybridCandidate, ...]

def build_hybrid_snapshot_record(
    record: Mapping[str, Any],
    *,
    chunks_by_id: Mapping[str, TechQAChunk],
    bm25_searcher: Callable[..., list[TechQAChunk]],
) -> HybridSnapshotRecord

def prepare_frozen_snapshot(...) -> dict[str, Any]
```

#### Contract 1 — candidate construction

- [ ] **Step 1: Write one RED covering the entire candidate contract**

Create `tests/test_eval_techqa_hybrid_rerank.py`.

The first test is:

```python
def test_candidate_snapshot_contract_preserves_cross_source_rrf_credit():
    module = _load_module()

    dense_ids = [f"d{i}" for i in range(99)]
    bm25_ids = [f"b{i}" for i in range(99)]

    # shared is rank 50 in both sources.
    dense_ids.insert(49, "shared")
    bm25_ids.insert(49, "shared")

    all_ids = set(dense_ids) | set(bm25_ids)

    chunks_by_id = {
        chunk_id: module.TechQAChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            chunk_index=0,
            content=f"content for {chunk_id}",
        )
        for chunk_id in all_ids
    }

    record = {
        "question_id": "TRAIN_Q1",
        "question": "shared technical query",
        "relevant_document_ids": ["doc-shared"],
        "raw_chunk_ids": dense_ids,
    }

    bm25_chunks = [chunks_by_id[chunk_id] for chunk_id in bm25_ids]

    result = module.build_hybrid_snapshot_record(
        record,
        chunks_by_id=chunks_by_id,
        bm25_searcher=lambda query, top_k: bm25_chunks[:top_k],
    )

    assert len(result.dense_chunk_ids) == 100
    assert len(set(result.dense_chunk_ids)) == 100
    assert len(result.bm25_chunk_ids) == 100
    assert len(set(result.bm25_chunk_ids)) == 100

    fused_ids = [candidate.chunk_id for candidate in result.fused_candidates]

    assert len(fused_ids) == 100
    assert len(set(fused_ids)) == 100

    # rank 50 + rank 50 receives two RRF contributions and therefore
    # outranks chunks seen in only one source at rank 1.
    assert fused_ids[0] == "shared"
```

The module loader may fail with a clear message if the new evaluator module
does not yet exist.

- [ ] **Step 2: Run the RED and stop**

Run:

```powershell
pytest tests/test_eval_techqa_hybrid_rerank.py::test_candidate_snapshot_contract_preserves_cross_source_rrf_credit -v
```

Expected:

FAIL because `build_hybrid_snapshot_record` does not exist.

Do not add more candidate-construction unit tests.

- [ ] **Step 3: Implement only the candidate contract**

Implement in `experiments/evals/eval_techqa_hybrid_rerank.py`:

```python
EXPECTED_TRAIN_COUNT = 450
CANDIDATE_K = 100
RRF_K = 60

def _require_unique_exact(
    values: Sequence[str],
    *,
    expected: int,
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)

    if len(normalized) != expected:
        raise RuntimeError(
            f"{label} must contain exactly {expected} chunks: "
            f"actual={len(normalized)}"
        )

    if len(set(normalized)) != expected:
        raise RuntimeError(f"{label} contains duplicate chunk_id values")

    return normalized


def build_hybrid_snapshot_record(
    record,
    *,
    chunks_by_id,
    bm25_searcher,
):
    question_id = str(record["question_id"])

    if not question_id.startswith("TRAIN_"):
        raise RuntimeError("C1 requires TRAIN-only input")

    dense_chunk_ids = _require_unique_exact(
        record["raw_chunk_ids"],
        expected=CANDIDATE_K,
        label="Dense candidate ranking",
    )

    bm25_chunks = bm25_searcher(
        str(record["question"]).rstrip(),
        top_k=CANDIDATE_K,
    )

    bm25_chunk_ids = _require_unique_exact(
        [chunk.chunk_id for chunk in bm25_chunks],
        expected=CANDIDATE_K,
        label="BM25 candidate ranking",
    )

    fused_chunk_ids = tuple(
        fuse_rrf(
            [dense_chunk_ids, bm25_chunk_ids],
            rrf_k=RRF_K,
            top_k=CANDIDATE_K,
        )
    )

    fused_chunk_ids = _require_unique_exact(
        fused_chunk_ids,
        expected=CANDIDATE_K,
        label="Hybrid fused ranking",
    )

    missing = [
        chunk_id
        for chunk_id in fused_chunk_ids
        if chunk_id not in chunks_by_id
    ]
    if missing:
        raise RuntimeError(
            "fused chunk(s) missing from frozen chunk universe: "
            + ", ".join(missing[:5])
        )

    return HybridSnapshotRecord(
        question_id=question_id,
        question=str(record["question"]),
        relevant_document_ids=tuple(
            str(value)
            for value in record["relevant_document_ids"]
        ),
        dense_chunk_ids=dense_chunk_ids,
        bm25_chunk_ids=bm25_chunk_ids,
        fused_candidates=tuple(
            HybridCandidate(
                chunk_id=chunks_by_id[chunk_id].chunk_id,
                document_id=chunks_by_id[chunk_id].document_id,
                content=chunks_by_id[chunk_id].content,
            )
            for chunk_id in fused_chunk_ids
        ),
    )
```

Use the existing `fuse_rrf`; do not write another RRF implementation.

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/test_eval_techqa_hybrid_rerank.py::test_candidate_snapshot_contract_preserves_cross_source_rrf_credit -v
ruff check experiments/evals/eval_techqa_hybrid_rerank.py tests/test_eval_techqa_hybrid_rerank.py
```

Expected: PASS and Ruff clean.

- [ ] **Step 5: Add zero-cost snapshot materialization**

In the same module implement:

```text
load current E0 TRAIN JSONL
→ require exactly 450 TRAIN rows
→ load frozen TechQA corpus from manifest revision
→ build 28,481 TechQA documents
→ build deterministic 172,614 chunks
→ build one chunks_by_id map
→ build one TechQAChunkBM25Retriever
→ build HybridSnapshotRecord for all 450 rows
→ write train_fused_snapshot.jsonl
→ SHA256 snapshot
→ write train_snapshot_manifest.json
→ write train_preflight.json
```

The manifest must record at minimum:

```json
{
  "run": "r4_c1_hybrid_rerank",
  "split": "train",
  "query_count": 450,
  "dense_candidate_k": 100,
  "bm25_candidate_k": 100,
  "rrf_k": 60,
  "fused_candidate_k": 100,
  "provider_calls": 0,
  "dev_artifact_opened": false,
  "snapshot_sha256": "...",
  "reranker_model": "qwen3-rerank",
  "token_stop_threshold": 13500000,
  "monetary_safety_envelope_usd": 1.50
}
```

Also record:
- current E0 input SHA
- historical E1 E0 source SHA
- TechQA frozen revision
- corpus SHA
- splitter blob SHA
- chunk count
- bm25s version/config
- project SHA

Do not call any provider from the `prepare` path.

CLI:

```powershell
python -m experiments.evals.eval_techqa_hybrid_rerank prepare
```

- [ ] **Step 6: Commit Task 1**

Stage exact files only:

```powershell
git add experiments/evals/eval_techqa_hybrid_rerank.py tests/test_eval_techqa_hybrid_rerank.py
git commit -m "feat(eval): build frozen R4 C1 hybrid snapshot"
```

---

### Task 2: Rerank one frozen Hybrid query without changing E1 behavior

**Files:**
- Modify: `experiments/evals/eval_techqa_hybrid_rerank.py`
- Modify: `tests/test_eval_techqa_hybrid_rerank.py`

**Interfaces:**

Produces:

```python
@dataclass(frozen=True)
class HybridRerankResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    fused_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    reranked_document_ids: tuple[str, ...]
    document_ranking: tuple[str, ...]
    rerank_latency_ms: float
    request_id: str | None
    total_tokens: int | None

def rerank_snapshot_record(
    record: HybridSnapshotRecord,
    *,
    reranker: Reranker = rerank_candidates,
    clock: Clock = time.perf_counter,
) -> HybridRerankResult
```

#### Contract 2 — one-query orchestration

- [ ] **Step 1: Append exactly one new RED**

```python
def test_one_query_orchestration_preserves_frozen_candidates_and_provider_identity():
    module = _load_module()

    candidates = tuple(
        module.HybridCandidate(
            chunk_id=f"c{i}",
            document_id=("shared-doc" if i < 2 else f"doc{i}"),
            content=f"content {i}",
        )
        for i in range(100)
    )

    record = module.HybridSnapshotRecord(
        question_id="TRAIN_Q1",
        question="technical query",
        relevant_document_ids=("shared-doc",),
        dense_chunk_ids=tuple(f"d{i}" for i in range(100)),
        bm25_chunk_ids=tuple(f"b{i}" for i in range(100)),
        fused_candidates=candidates,
    )

    class FakeResult:
        def __init__(self):
            self.results = tuple(
                module.RerankedCandidate(
                    chunk_id=candidates[index].chunk_id,
                    document_id=candidates[index].document_id,
                    content=candidates[index].content,
                    original_index=index,
                    relevance_score=float(100 - index),
                )
                for index in reversed(range(100))
            )
            self.request_id = "req-c1"
            self.total_tokens = 12345

    seen = {}

    def fake_reranker(query, rerank_candidates):
        seen["query"] = query
        seen["chunk_ids"] = [item.chunk_id for item in rerank_candidates]
        return FakeResult()

    result = module.rerank_snapshot_record(
        record,
        reranker=fake_reranker,
        clock=iter([1.0, 1.25]).__next__,
    )

    assert seen["query"] == "technical query"
    assert seen["chunk_ids"] == [f"c{i}" for i in range(100)]
    assert len(result.reranked_chunk_ids) == 100
    assert result.document_ranking[0] == "doc99"
    assert len(result.document_ranking) == 99
    assert result.request_id == "req-c1"
    assert result.total_tokens == 12345
    assert result.rerank_latency_ms == 250.0
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_eval_techqa_hybrid_rerank.py::test_one_query_orchestration_preserves_frozen_candidates_and_provider_identity -v
```

Expected: FAIL because `rerank_snapshot_record` is absent.

- [ ] **Step 3: Implement minimal orchestration**

Implementation must:

```text
validate TRAIN question_id
validate exactly 100 fused candidates
convert them to existing RerankCandidate
call existing rerank_candidates-compatible callable
retain complete provider permutation
collapse document IDs by first occurrence
record latency/request_id/total_tokens
```

Do not alter:
- model
- instruction
- query normalization
- candidate count
- document-collapse rule.

Use the existing reranker and ranx/document-collapse helpers instead of
reimplementing provider parsing.

- [ ] **Step 4: GREEN + Ruff**

```powershell
pytest tests/test_eval_techqa_hybrid_rerank.py::test_one_query_orchestration_preserves_frozen_candidates_and_provider_identity -v
ruff check experiments/evals/eval_techqa_hybrid_rerank.py tests/test_eval_techqa_hybrid_rerank.py
```

- [ ] **Step 5: Commit Task 2**

```powershell
git add experiments/evals/eval_techqa_hybrid_rerank.py tests/test_eval_techqa_hybrid_rerank.py
git commit -m "feat(eval): add R4 C1 hybrid rerank orchestration"
```

---

### Task 3: Add the paid-run safety boundary and resumable runner

**Files:**
- Modify: `experiments/evals/eval_techqa_hybrid_rerank.py`
- Modify: `experiments/evals/rerankers/qwen3_reranker.py`
- Modify: `tests/test_eval_techqa_hybrid_rerank.py`

**Interfaces:**

Produces:

```python
TOKEN_STOP_THRESHOLD = 13_500_000

def validate_paid_snapshot(...) -> list[HybridSnapshotRecord]

def run_resumable_paid_eval(
    records,
    *,
    evaluator,
    checkpoint_path,
    inflight_path,
    token_stop_threshold=TOKEN_STOP_THRESHOLD,
) -> HybridRerankSummary
```

#### Contract 3 — paid-run safety

- [ ] **Step 1: Append one final C1 contract test**

Use one test function containing a small set of sequential safety scenarios.
Do not split these into separate tests.

The test must verify:

```text
1. completed checkpoint qid -> provider evaluator not called
2. wrong snapshot SHA -> failure before provider evaluator
3. fused candidate count != 100 -> failure before provider evaluator
4. cumulative checkpoint tokens >= 13.5M -> next provider call not sent
5. evaluator/provider exception -> run stops and inflight marker remains
6. unresolved inflight qid not in checkpoint -> resume refuses automatic replay
7. DEV_* qid -> failure before provider use
8. get_rerank_client constructs OpenAI with max_retries=0
```

For the SDK assertion monkeypatch `OpenAI` and inspect constructor kwargs;
do not contact the network.

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_eval_techqa_hybrid_rerank.py::test_paid_runner_safety_contract -v
```

Expected: FAIL on missing paid-run safety behavior.

- [ ] **Step 3: Disable SDK retries**

Modify:

`experiments/evals/rerankers/qwen3_reranker.py`

from:

```python
return OpenAI(api_key=api_key, base_url=base_url)
```

to:

```python
return OpenAI(
    api_key=api_key,
    base_url=base_url,
    max_retries=0,
)
```

No other reranker behavior changes.

- [ ] **Step 4: Implement snapshot verification before provider use**

Paid startup order must be:

```text
load snapshot manifest
→ SHA256 actual snapshot
→ compare expected SHA
→ load snapshot
→ require 450 unique TRAIN question IDs
→ require exactly 100 unique fused chunks per record
→ inspect inflight marker
→ load completed checkpoint
→ only then allow evaluator/provider use
```

Any failure before the final step must result in zero provider calls.

- [ ] **Step 5: Implement minimal inflight safety**

Before each provider request:

```json
{"question_id": "TRAIN_Q123"}
```

is written to `train_inflight.json`.

After a successful provider result:

```text
append durable checkpoint result
→ then remove inflight marker
```

Resume behavior:

```text
no inflight marker
    -> normal resume

inflight qid already present in completed checkpoint
    -> stale marker may be cleared safely

inflight qid NOT present in completed checkpoint
    -> uncertain provider outcome
    -> stop
    -> do not automatically replay
```

Do not create a general state-machine framework.

- [ ] **Step 6: Implement token stop semantics**

At startup:

```python
provider_total_tokens = sum(
    result.total_tokens or 0
    for result in completed_results
)
```

Before every new request:

```python
if provider_total_tokens >= token_stop_threshold:
    break
```

After a successful response is checkpointed:

```python
provider_total_tokens += result.total_tokens or 0
```

Then evaluate the threshold before the next request.

Do not claim that 13.5M can never be exceeded by the last already-issued
request.

- [ ] **Step 7: Implement metrics and C1 gate**

Final TRAIN metrics:

- Recall@5
- Recall@20
- MRR@10
- rerank latency p50
- rerank latency p95
- provider total tokens
- completed query count

Frozen E1 baseline:

```text
Recall@5  = 0.6911111111111111
Recall@20 = 0.8155555555555556
MRR@10    = 0.5672063492063492
```

Gate:

```python
success = (
    metrics["mrr@10"] >= 0.577206
    and metrics["recall@20"] >= 0.810556
)
```

Write both observed metrics and gate outcome.

Do not retrofit thresholds after seeing results.

Do not rerun E1 for per-query comparison.

- [ ] **Step 8: Add explicit CLI separation**

CLI must expose only:

```powershell
python -m experiments.evals.eval_techqa_hybrid_rerank prepare
python -m experiments.evals.eval_techqa_hybrid_rerank paid
```

`prepare`:
- may load corpus
- may build BM25
- may RRF
- must never construct/call the provider client.

`paid`:
- must never rerun Dense
- must never rebuild BM25
- must never rerun RRF
- consumes only the frozen snapshot.

There is no `--split dev` option.

- [ ] **Step 9: GREEN the three C1 contracts**

```powershell
pytest tests/test_eval_techqa_hybrid_rerank.py -v
```

Expected:

```text
3 passed
```

Do not add more C1 tests unless a real defect discovered later requires one
regression test.

- [ ] **Step 10: Run relevant existing regression tests**

```powershell
pytest `
  tests/test_rrf.py `
  tests/test_bm25_techqa_chunks.py `
  tests/test_qwen3_reranker.py `
  tests/test_eval_techqa_rerank.py `
  tests/test_eval_techqa_hybrid_rerank.py `
  -q
```

Expected: all pass.

- [ ] **Step 11: Run full verification**

```powershell
pytest -q
ruff check .
```

Expected:
- zero failures
- Ruff clean.

- [ ] **Step 12: Commit Task 3**

```powershell
git add `
  experiments/evals/eval_techqa_hybrid_rerank.py `
  experiments/evals/rerankers/qwen3_reranker.py `
  tests/test_eval_techqa_hybrid_rerank.py

git commit -m "feat(eval): enforce R4 C1 paid-run safety"
```

---

### Task 4: Zero-provider-call real snapshot acceptance

This task does not add another unit test.

- [ ] **Step 1: Run snapshot preparation**

```powershell
python -m experiments.evals.eval_techqa_hybrid_rerank prepare
```

This command must make zero provider calls.

Expected high-level evidence:

```text
split=TRAIN
query_count=450
documents=28481
chunks=172614
dense_candidate_k=100
bm25_candidate_k=100
rrf_k=60
fused_candidate_k=100
provider_calls=0
dev_artifact_opened=false
```

- [ ] **Step 2: Verify frozen artifacts**

Check:

```powershell
Get-Content experiments/evals/reports/r4_c1_hybrid_rerank/train_preflight.json
Get-Content experiments/evals/reports/r4_c1_hybrid_rerank/train_snapshot_manifest.json
```

Verify:
- 450 queries
- every qid starts `TRAIN_`
- snapshot SHA exists
- actual snapshot SHA equals manifest SHA
- all Dense rankings are 100 unique chunks
- all BM25 rankings are 100 unique chunks
- all fused rankings are 100 unique chunks
- RRF k=60
- provider_calls=0
- DEV unopened
- qwen3-rerank identity matches E1
- token threshold=13.5M
- monetary envelope=$1.50.

- [ ] **Step 3: Re-run verification after the real snapshot exists**

```powershell
pytest -q
ruff check .
git status --short
```

No provider call.

- [ ] **Step 4: STOP**

Do NOT run:

```powershell
python -m experiments.evals.eval_techqa_hybrid_rerank paid
```

until the human explicitly reviews the zero-cost preflight and approves the
paid run.

This is the mandatory paid boundary.

---

### Task 5: Execute the paid C1 run after explicit approval

This task begins only after explicit human approval.

- [ ] **Step 1: Run the frozen paid experiment**

```powershell
python -m experiments.evals.eval_techqa_hybrid_rerank paid
```

Expected behavior:
- reads frozen snapshot only
- validates SHA before provider use
- resumes completed qids
- no automatic SDK retry
- no automatic uncertain-qid replay
- stops on provider error
- stops before the next request once cumulative returned tokens reach 13.5M.

- [ ] **Step 2: Inspect final evidence**

Read:

```powershell
Get-Content experiments/evals/reports/r4_c1_hybrid_rerank/train_metrics.json
Get-Content experiments/evals/reports/r4_c1_hybrid_rerank/train_manifest.json
Get-Content experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md
```

Primary decision:

```text
PASS C1:
MRR@10 >= 0.577206
AND
Recall@20 >= 0.810556

otherwise:
C1 does not satisfy the pre-registered success gate.
```

A negative result is valid evidence and must not be patched post-hoc by
changing RRF, BM25 depth, source weighting, rerank depth or thresholds.

- [ ] **Step 3: Update the cost ledger with actual evidence**

Modify:

`experiments/evals/reports/cost_ledger.md`

Record:
- actual returned provider tokens
- actual estimated/known spend
- whether the run completed all 450 queries
- C1 gate decision
- question answered.

- [ ] **Step 4: Final verification and exact staging**

```powershell
pytest -q
ruff check .
git diff --check
git status --short
```

Stage only compact evidence and source/test changes intended for the PR.
Do not blindly commit large local snapshot/checkpoint files.

---

## Execution Order Summary

```text
Task 1
candidate snapshot contract
        |
        v
Task 2
one-query rerank contract
        |
        v
Task 3
paid-run safety contract
        |
        v
full pytest + Ruff
        |
        v
Task 4
REAL 450-query zero-cost snapshot
        |
        v
snapshot SHA + preflight review
        |
        v
HUMAN PAID APPROVAL
        |
        v
Task 5
qwen3-rerank paid C1
        |
        v
frozen success gate
```

## Testing Budget

C1-specific test count is intentionally capped at three behavior-level
contract tests before the real experiment:

1. candidate construction
2. one-query orchestration
3. paid-run safety

No fourth test is added merely to increase coverage.

A new test beyond those three is allowed only when:
- an actual defect is observed during implementation/acceptance, and
- the test is a regression test for that concrete defect.
