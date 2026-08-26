# Evaluation Governance r4 — R1.5 through R3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the already-earned E1 rerank results into repository-auditable evidence, characterize E1 TRAIN residuals, pre-register the Hybrid admission gate, and run a zero-model-cost Dense/BM25/RRF TRAIN pilot without consuming generation DEV or starting another paid rerank run.

**Architecture:** Preserve all completed E0/E1 artifacts as immutable inputs. R1.5 reads local formal rerank artifacts and materializes only compact evidence plus hashes; R2 joins frozen E0/E1 TRAIN retrieval results to separate candidate misses from rerank residuals and creates a deterministic human-review sample; R3 indexes the frozen TechQA document corpus with BM25S at document level and fuses Dense/BM25 document rankings using equal-weight RRF. Paid Hybrid+Rerank is explicitly outside this plan and is admitted only after the pre-registered R3 gate is evaluated.

**Tech Stack:** Python 3.11, pytest 9, Ruff, datasets 5.0.1, ranx 0.3.21, bm25s 0.3.10, frozen TechQA artifacts.

**Spec:** `docs/superpowers/specs/2026-08-25-eval-governance-r4-design.md`

## Global Constraints

- Branch remains `feat/eval-techqa-baseline`; do not merge PR #11 during these stages.
- R1 is closed by fresh local evidence: `9 passed in 6.67s` and Ruff `All checks passed!`.
- Do not execute `python -m experiments.evals.eval_techqa_generation --split dev`.
- E0/E1 retrieval DEV is historical frozen evidence and must not influence Hybrid parameter selection.
- R1.5 may package already-frozen DEV aggregate evidence, but R2/R3 use TRAIN only and never inspect DEV failure cases.
- No embedding, rerank, generator, Judge, DashScope, or OpenAI provider call is allowed in R1.5, R2, or R3. Unexpected provider-client construction is a stop condition.
- Existing Singapore E1 identity remains `qwen3-rerank`, frozen Dense Top-100 chunks, `rstrip()` query normalization, first-occurrence document collapse.
- Large E1 `*_checkpoint.jsonl` and candidate-heavy `*_results.jsonl` remain local; repository evidence stores SHA256 references instead.
- Every qualitative admission condition is converted to an explicit numeric gate before the experiment it gates.
- R3 is deliberately document-level: Dense starts from the frozen E0 Top-100 **chunk** pool and collapses it to up to 100 unique documents; BM25 returns Top-100 documents; RRF produces Top-100 documents. R3 is a complementarity pilot, not the final chunk-level Hybrid+Rerank implementation.
- RRF is equal-weight with `rrf_k=60`.
- BM25 configuration is frozen to `bm25s==0.3.10`, `method="lucene"`, `k1=1.5`, `b=0.75`, `backend="numpy"`, and the project-owned technical tokenizer defined in Task 4.
- If R3 does not satisfy the committed gate, do not implement or run paid Hybrid+Rerank; move to a separate Generator-ablation plan.

---

### Task 1: Materialize compact E1 evidence and cost ledger (R1.5)

**Files:**
- Create: `experiments/evals/rerank_evidence.py`
- Create: `tests/test_rerank_evidence.py`
- Produce: `experiments/evals/reports/e1_rerank/artifact_hashes.json`
- Produce: `experiments/evals/reports/e1_rerank/comparison.md`
- Create: `experiments/evals/reports/cost_ledger.md`
- Modify: `.gitignore`
- Version after validation: `experiments/evals/reports/e1_rerank/train_manifest.json`, `train_metrics.json`, `dev_manifest.json`, `dev_metrics.json`

**Interfaces:**
- Consumes: frozen E0 `train/dev_{manifest,metrics,results}.json*` and local E1 `train/dev_{manifest,metrics,checkpoint,results}.json*`.
- Produces: `materialize_rerank_evidence(...) -> dict[str, Any]` and compact repository evidence used by later reports.

- [ ] **Step 1: Write the failing evidence tests**

Create `tests/test_rerank_evidence.py`. Tests must prove: paired fixed/regressed counts are correct; full retrieval latency uses per-query `dense + rerank` sums; checkpoint/results hashes are emitted; `comparison.md` contains aggregate DEV counts but no DEV QIDs.

```python
from experiments.evals.rerank_evidence import build_paired_evidence


def test_paired_evidence_uses_per_query_latency_sums():
    e0 = [
        {
            "question_id": "Q1",
            "relevant_document_ids": ["g1"],
            "document_ranking": ["x", "g1"],
            "latency_ms": 100.0,
        },
        {
            "question_id": "Q2",
            "relevant_document_ids": ["g2"],
            "document_ranking": ["x", "y"],
            "latency_ms": 900.0,
        },
    ]
    e1 = [
        {
            "question_id": "Q1",
            "relevant_document_ids": ["g1"],
            "document_ranking": ["g1"],
            "rerank_latency_ms": 900.0,
        },
        {
            "question_id": "Q2",
            "relevant_document_ids": ["g2"],
            "document_ranking": ["g2"],
            "rerank_latency_ms": 100.0,
        },
    ]

    evidence = build_paired_evidence(e0, e1)

    assert evidence["hit5_fixed"] == 1
    assert evidence["hit5_regressed"] == 0
    assert evidence["full_retrieval_latency_ms"] == [1000.0, 1000.0]
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_rerank_evidence.py -v
```

Expected: FAIL because `experiments.evals.rerank_evidence` does not exist.

- [ ] **Step 3: Implement the minimal evidence module**

Create `experiments/evals/rerank_evidence.py` with these boundaries:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

E0_DIR = Path("experiments/evals/reports/e0_dense")
E1_DIR = Path("experiments/evals/reports/e1_rerank")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_relevant_rank(row: dict[str, Any]) -> int | None:
    gold = {str(value) for value in row["relevant_document_ids"]}
    for rank, document_id in enumerate(row["document_ranking"], start=1):
        if str(document_id) in gold:
            return rank
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_paired_evidence(
    e0_rows: list[dict[str, Any]],
    e1_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    e0_by_id = {str(row["question_id"]): row for row in e0_rows}
    e1_by_id = {str(row["question_id"]): row for row in e1_rows}
    if set(e0_by_id) != set(e1_by_id):
        raise ValueError("E0/E1 question_id mismatch")

    fixed5 = regressed5 = fixed20 = regressed20 = 0
    full_latency: list[float] = []
    for question_id in sorted(e0_by_id):
        before = first_relevant_rank(e0_by_id[question_id])
        after = first_relevant_rank(e1_by_id[question_id])
        before5 = before is not None and before <= 5
        after5 = after is not None and after <= 5
        before20 = before is not None and before <= 20
        after20 = after is not None and after <= 20
        fixed5 += int((not before5) and after5)
        regressed5 += int(before5 and (not after5))
        fixed20 += int((not before20) and after20)
        regressed20 += int(before20 and (not after20))
        full_latency.append(
            float(e0_by_id[question_id]["latency_ms"])
            + float(e1_by_id[question_id]["rerank_latency_ms"])
        )

    return {
        "hit5_fixed": fixed5,
        "hit5_regressed": regressed5,
        "hit20_fixed": fixed20,
        "hit20_regressed": regressed20,
        "full_retrieval_latency_ms": full_latency,
        "full_retrieval_p50_ms": percentile(full_latency, 0.50),
        "full_retrieval_p95_ms": percentile(full_latency, 0.95),
    }
```

Add JSON/JSONL loaders and `materialize_rerank_evidence(...)`. It must validate formal row counts `450/160`, semantic equality of checkpoint/results per split, identical E0/E1 QID sets, and existing manifest identities. Write `artifact_hashes.json` for the local large artifacts and aggregate-only `comparison.md`. Do not write DEV failure QIDs.

- [ ] **Step 4: Run GREEN and Ruff**

```powershell
pytest tests/test_rerank_evidence.py -v
ruff check experiments/evals/rerank_evidence.py tests/test_rerank_evidence.py
```

- [ ] **Step 5: Materialize from the local formal artifacts**

```powershell
python -m experiments.evals.rerank_evidence
```

The command must fail closed if an expected artifact is absent or identities/counts differ. Expected aggregate DEV facts include Recall@5 `0.643750 -> 0.725000`, Recall@20 `0.818750 -> 0.843750`, MRR@10 `0.518931 -> 0.560841`, paired Top-5 `21 fixed / 8 regressed`, Top-20 `5 fixed / 1 regressed`, and full retrieval p95 about `4657.189 ms`.

- [ ] **Step 6: Add the cost ledger**

Create `experiments/evals/reports/cost_ledger.md`:

```markdown
# Evaluation Cost Ledger

| Stage | Question answered | API | Estimated cap | Actual spend | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Completed E0/E1 | establish baseline and test rerank ranking-error hypothesis | mixed historical | historical | ≈ CNY 41 cumulative | frozen |
| R2 | characterize E1 residuals | local | CNY 0 | CNY 0 | pending |
| R3 | test lexical complementarity with BM25/RRF | local | CNY 0 API | CNY 0 API | pending |
| R4 | test Hybrid candidate pool + rerank | qwen3-rerank | freeze before run | — | gated |
| G1 | test generator bottleneck on frozen context | candidate LLM | freeze before run | — | gated |
| D1 | validate final frozen generation configuration | final LLM/eval | final approval | — | gated |
```

- [ ] **Step 7: Ignore only large E1 local payloads**

Append to `.gitignore`:

```gitignore
experiments/evals/reports/e1_rerank/*_checkpoint.jsonl
experiments/evals/reports/e1_rerank/*_results.jsonl
```

Do not stage large rerank checkpoint/results files.

- [ ] **Step 8: Commit R1.5**

```bash
git add .gitignore experiments/evals/rerank_evidence.py tests/test_rerank_evidence.py experiments/evals/reports/e1_rerank/train_manifest.json experiments/evals/reports/e1_rerank/train_metrics.json experiments/evals/reports/e1_rerank/dev_manifest.json experiments/evals/reports/e1_rerank/dev_metrics.json experiments/evals/reports/e1_rerank/artifact_hashes.json experiments/evals/reports/e1_rerank/comparison.md experiments/evals/reports/cost_ledger.md
git commit -m "docs(eval): version frozen E1 rerank evidence"
```

**R1.5 exit gate:** compact TRAIN/DEV E1 evidence is repository-auditable, large local payload hashes are recorded, and no provider call occurred.

---

### Task 2: Build deterministic E1 TRAIN residual analysis (R2a)

**Files:**
- Create: `experiments/evals/rerank_residual_analysis.py`
- Create: `tests/test_rerank_residual_analysis.py`
- Produce: `experiments/evals/reports/e1_rerank/train_residual_summary.json`
- Produce: `experiments/evals/reports/e1_rerank/train_residual_review.jsonl`

**Interfaces:**
- Consumes: frozen E0 TRAIN `train_results.jsonl`, local E1 TRAIN `train_results.jsonl`, frozen TechQA corpus from `load_frozen_techqa_documents()`.
- Produces: deterministic residual counts and a SHA256-stable human-review sample containing only E1 candidate-miss cases.

- [ ] **Step 1: Write failing residual-classification tests**

```python
from experiments.evals.rerank_residual_analysis import build_residual_records


def test_residual_buckets_separate_candidate_miss_from_rerank_residual():
    e0 = [
        {"question_id": "Q1", "question": "q1", "relevant_document_ids": ["g1"], "raw_chunk_ids": ["c1"], "raw_document_ids": ["g1"]},
        {"question_id": "Q2", "question": "q2", "relevant_document_ids": ["g2"], "raw_chunk_ids": ["c2", "c3"], "raw_document_ids": ["g2", "x"]},
        {"question_id": "Q3", "question": "q3", "relevant_document_ids": ["g3"], "raw_chunk_ids": ["c4"], "raw_document_ids": ["x"]},
    ]
    e1 = [
        {"question_id": "Q1", "dense_chunk_ids": ["c1"], "document_ranking": ["g1"]},
        {"question_id": "Q2", "dense_chunk_ids": ["c2", "c3"], "document_ranking": ["x"]},
        {"question_id": "Q3", "dense_chunk_ids": ["c4"], "document_ranking": ["x"]},
    ]

    records = build_residual_records(e0, e1)

    assert [record.residual_bucket for record in records] == [
        "resolved_top20",
        "rerank_residual_top20",
        "dense_candidate_miss_top100",
    ]
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_rerank_residual_analysis.py -v
```

Expected: FAIL because the new module/function is absent.

- [ ] **Step 3: Implement residual records and identity validation**

```python
ResidualBucket = Literal[
    "resolved_top20",
    "rerank_residual_top20",
    "dense_candidate_miss_top100",
]


@dataclass(frozen=True)
class RerankResidualRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    dense_candidate_document_ids: tuple[str, ...]
    e1_document_ranking: tuple[str, ...]
    residual_bucket: ResidualBucket
```

`build_residual_records()` must validate identical QID sets and exact `E1 dense_chunk_ids == E0 raw_chunk_ids` for every query. Reconstruct Dense candidate documents by first occurrence over all E0 Top-100 raw chunk document IDs. Classify:
- `resolved_top20`: gold appears in E1 Top-20;
- `rerank_residual_top20`: gold exists in the frozen Dense Top-100 document candidates but is absent E1 Top-20;
- `dense_candidate_miss_top100`: gold is absent from the frozen Dense Top-100 document candidates.

- [ ] **Step 4: Add deterministic manual-review sampling**

```python
def select_candidate_miss_review_sample(
    records: Iterable[RerankResidualRecord],
    *,
    sample_size: int = 30,
) -> list[RerankResidualRecord]:
    misses = [
        record
        for record in records
        if record.residual_bucket == "dense_candidate_miss_top100"
    ]
    return sorted(
        misses,
        key=lambda record: hashlib.sha256(
            record.question_id.encode("utf-8")
        ).hexdigest(),
    )[: min(sample_size, len(misses))]
```

Review rows contain question, gold document ID/text excerpt, Dense Top-5 document IDs/text excerpts, E1 Top-5 document IDs/text excerpts, and blank `manual_label` / `notes`. Allowed labels are exactly:

```text
lexical_candidate
semantic_or_indirect_miss
qrel_or_query_ambiguity
```

- [ ] **Step 5: Run GREEN and Ruff**

```powershell
pytest tests/test_rerank_residual_analysis.py -v
ruff check experiments/evals/rerank_residual_analysis.py tests/test_rerank_residual_analysis.py
```

- [ ] **Step 6: Materialize R2a with zero model calls**

```powershell
python -m experiments.evals.rerank_residual_analysis prepare
```

Expected outputs:
- `train_residual_summary.json` partitioning all 450 TRAIN queries into exactly the three buckets;
- `train_residual_review.jsonl` containing at most 30 deterministic candidate-miss rows.

Stop on any E0/E1 identity mismatch. Do not inspect DEV rows.

- [ ] **Step 7: Commit R2a**

```bash
git add experiments/evals/rerank_residual_analysis.py tests/test_rerank_residual_analysis.py experiments/evals/reports/e1_rerank/train_residual_summary.json experiments/evals/reports/e1_rerank/train_residual_review.jsonl
git commit -m "feat(eval): materialize E1 TRAIN residuals"
```

---

### Task 3: Complete R2 human review and freeze the R3 admission gate

**Files:**
- Modify manually: `experiments/evals/reports/e1_rerank/train_residual_review.jsonl`
- Produce: `experiments/evals/reports/e1_rerank/train_residual_review_summary.json`
- Produce: `experiments/evals/reports/e1_rerank/r3_gate.json`
- Test: extend `tests/test_rerank_residual_analysis.py`

**Interfaces:**
- Consumes: deterministic R2 review template and measured Dense Top-100 candidate-miss population `M`.
- Produces: diagnostic lexical-miss counts and a numeric gate committed before any R3 result exists.

- [ ] **Step 1: Add failing tests for review completeness and gate calculation**

The summarizer rejects blank/unknown labels. Freeze this gate formula before R3:

```python
required_recovered_cases = max(5, math.ceil(0.15 * M))
required_net_gain_cases = max(5, math.ceil(0.10 * M))
required_net_gain_pp = required_net_gain_cases / 450 * 100.0
```

Paid R4 is admitted only if both conditions hold after R3:
1. Hybrid recovers at least `required_recovered_cases` Dense Top-100 misses into Hybrid Top-100.
2. Hybrid Top-100 has at least `required_net_gain_cases` more gold hits than Dense Top-100.

Concrete test for `M=40`: thresholds are `6` recovered, `5` net new hits, `1.111111...` absolute Recall@100 percentage points.

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_rerank_residual_analysis.py -v
```

- [ ] **Step 3: Implement review summary and `build_r3_gate()`**

The review summary reports raw counts/sample rates and includes `"population_rate_claim_allowed": false`.

For the concrete unit-test input `M=40`, `build_r3_gate(40)` must produce:

```json
{
  "split": "train",
  "query_count": 450,
  "dense_candidate_chunk_k": 100,
  "bm25_candidate_document_k": 100,
  "hybrid_candidate_document_k": 100,
  "rrf_k": 60,
  "dense_candidate_miss_count": 40,
  "required_recovered_dense_misses": 6,
  "required_net_gain_cases": 5,
  "required_net_gain_pp": 1.1111111111111112,
  "admission_logic": "recovered_dense_misses >= required_recovered_dense_misses AND hybrid_hit100 - dense_hit100 >= required_net_gain_cases"
}
```

Formal `r3_gate.json` is generated from the actual R2 value `M`; it is never hand-edited after R3 starts.

- [ ] **Step 4: Run GREEN and Ruff**

```powershell
pytest tests/test_rerank_residual_analysis.py -v
ruff check experiments/evals/rerank_residual_analysis.py tests/test_rerank_residual_analysis.py
```

- [ ] **Step 5: Human-label the frozen sample**

For each row in `train_residual_review.jsonl`, modify only `manual_label` and `notes`. Do not modify QID, question, gold text, or candidate excerpts.

- [ ] **Step 6: Materialize summary and gate before any R3 run**

```powershell
python -m experiments.evals.rerank_residual_analysis summarize
python -m experiments.evals.rerank_residual_analysis freeze-gate
```

Read the numeric `r3_gate.json` and explicitly approve it before Task 6 formal R3 execution. Once approved, do not edit it based on R3 metrics.

- [ ] **Step 7: Commit reviewed R2 evidence and frozen gate**

```bash
git add experiments/evals/reports/e1_rerank/train_residual_review.jsonl experiments/evals/reports/e1_rerank/train_residual_review_summary.json experiments/evals/reports/e1_rerank/r3_gate.json
git commit -m "docs(eval): freeze R3 hybrid admission gate"
```

**R2 exit gate:** deterministic residual population is known, diagnostic lexical evidence is recorded, and the numeric R3→R4 gate is committed before R3 results exist.

---

### Task 4: Add the frozen BM25S document retriever

**Files:**
- Modify: `experiments/evals/requirements-eval.txt`
- Create: `experiments/evals/retrievers/__init__.py`
- Create: `experiments/evals/retrievers/bm25_techqa.py`
- Create: `tests/test_bm25_techqa.py`

**Interfaces:**
- Consumes: `Sequence[TechQADocument]` from `load_frozen_techqa_documents()`.
- Produces: `TechQABM25Retriever.search(query: str, top_k: int = 100) -> list[str]` returning ordered unique document IDs.

- [ ] **Step 1: Pin BM25S**

Append exactly:

```text
bm25s==0.3.10
```

to `experiments/evals/requirements-eval.txt`. Do not add PyStemmer or numba.

- [ ] **Step 2: Write failing retriever tests**

```python
from experiments.evals.adapters.techqa import TechQADocument
from experiments.evals.retrievers.bm25_techqa import (
    TechQABM25Retriever,
    tokenize_technical,
)


def test_technical_tokenizer_preserves_error_codes_and_versions():
    assert tokenize_technical(
        "Error 0x80070005 on v1.2.3 / CVE-2026-1234"
    ) == ["error", "0x80070005", "on", "v1.2.3", "cve-2026-1234"]


def test_bm25_retriever_returns_document_ids():
    retriever = TechQABM25Retriever(
        [
            TechQADocument("a", "permission denied error 0x80070005"),
            TechQADocument("b", "printer configuration"),
        ]
    )
    assert retriever.search("0x80070005", top_k=2)[0] == "a"
```

Also test case normalization, deterministic ranking, `top_k<=0`, and empty-token queries.

- [ ] **Step 3: Run RED**

```powershell
pytest tests/test_bm25_techqa.py -v
```

Expected: FAIL because the retriever module is absent.

- [ ] **Step 4: Implement the frozen BM25 contract**

```python
TECHNICAL_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
)


def tokenize_technical(text: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in TECHNICAL_TOKEN_RE.finditer(text)
    ]


class TechQABM25Retriever:
    def __init__(self, documents: Sequence[TechQADocument]) -> None:
        ordered = sorted(documents, key=lambda document: document.document_id)
        self.document_ids = [document.document_id for document in ordered]
        corpus_tokens = [tokenize_technical(document.text) for document in ordered]
        self._retriever = bm25s.BM25(
            k1=1.5,
            b=0.75,
            method="lucene",
            corpus=self.document_ids,
            backend="numpy",
        )
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 100) -> list[str]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        query_tokens = tokenize_technical(query.rstrip())
        if not query_tokens:
            return []
        result = self._retriever.retrieve(
            [query_tokens],
            k=min(top_k, len(self.document_ids)),
            show_progress=False,
        )
        return [str(document_id) for document_id in result.documents[0]]
```

- [ ] **Step 5: Install the pinned local-eval dependency**

In the active `customer_profile` environment:

```powershell
python -m pip install -r experiments/evals/requirements-eval.txt
python -c "import bm25s; print(bm25s.__version__)"
```

Expected version: `0.3.10`. This is a package download only, not a model/provider call.

- [ ] **Step 6: Run GREEN and Ruff**

```powershell
pytest tests/test_bm25_techqa.py -v
ruff check experiments/evals/retrievers/bm25_techqa.py tests/test_bm25_techqa.py
```

- [ ] **Step 7: Commit**

```bash
git add experiments/evals/requirements-eval.txt experiments/evals/retrievers/__init__.py experiments/evals/retrievers/bm25_techqa.py tests/test_bm25_techqa.py
git commit -m "feat(eval): add frozen TechQA BM25 retriever"
```

---

### Task 5: Add deterministic equal-weight RRF and configurable ranx metrics

**Files:**
- Create: `experiments/evals/ir/rrf.py`
- Create: `tests/test_rrf.py`
- Modify: `experiments/evals/ir/ranx_adapter.py`
- Modify: `tests/test_ranx_eval_adapter.py`

**Interfaces:**
- Produces: `fuse_rrf(rankings: Sequence[Sequence[str]], *, rrf_k: int = 60, top_k: int = 100) -> list[str]`.
- Produces: `evaluate_ir_metrics(qrels: Qrels, run: Run, metrics: Sequence[str]) -> dict[str, float]`; existing `evaluate_ir_run()` remains backward compatible.

- [ ] **Step 1: Write RRF RED tests**

```python
from experiments.evals.ir.rrf import fuse_rrf


def test_rrf_rewards_documents_present_in_both_rankings():
    fused = fuse_rrf(
        [["a", "b", "c"], ["b", "d", "a"]],
        rrf_k=60,
        top_k=4,
    )
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}
```

Add a tie test requiring deterministic order by descending RRF score, then best source rank, then document ID.

- [ ] **Step 2: Write dynamic-metric RED test**

Extend `tests/test_ranx_eval_adapter.py` so `evaluate_ir_metrics(..., ("recall@20", "recall@100", "mrr@10"))` returns all three while existing `evaluate_ir_run()` still returns exactly the frozen primary metrics.

- [ ] **Step 3: Run RED**

```powershell
pytest tests/test_rrf.py tests/test_ranx_eval_adapter.py -v
```

- [ ] **Step 4: Implement RRF**

```python
def fuse_rrf(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = 60,
    top_k: int = 100,
) -> list[str]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            document_id = str(document_id)
            scores[document_id] = (
                scores.get(document_id, 0.0) + 1.0 / (rrf_k + rank)
            )
            best_rank[document_id] = min(
                best_rank.get(document_id, rank),
                rank,
            )

    ordered = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            best_rank[document_id],
            document_id,
        ),
    )
    return ordered[:top_k]
```

- [ ] **Step 5: Implement configurable ranx evaluation without changing E0/E1 semantics**

```python
def evaluate_ir_metrics(
    qrels: Qrels,
    run: Run,
    metrics: Sequence[str],
) -> dict[str, float]:
    results = evaluate(qrels, run, list(metrics))
    return {metric: float(results[metric]) for metric in metrics}


def evaluate_ir_run(qrels: Qrels, run: Run) -> dict[str, float]:
    return evaluate_ir_metrics(qrels, run, PRIMARY_IR_METRICS)
```

- [ ] **Step 6: Run GREEN and Ruff**

```powershell
pytest tests/test_rrf.py tests/test_ranx_eval_adapter.py -v
ruff check experiments/evals/ir/rrf.py experiments/evals/ir/ranx_adapter.py tests/test_rrf.py tests/test_ranx_eval_adapter.py
```

- [ ] **Step 7: Commit**

```bash
git add experiments/evals/ir/rrf.py experiments/evals/ir/ranx_adapter.py tests/test_rrf.py tests/test_ranx_eval_adapter.py
git commit -m "feat(eval): add deterministic RRF fusion"
```

---

### Task 6: Implement and run the zero-API R3 Dense/BM25/RRF TRAIN pilot

**Files:**
- Create: `experiments/evals/eval_techqa_hybrid.py`
- Create: `tests/test_eval_techqa_hybrid.py`
- Produce: `experiments/evals/reports/r3_hybrid/train_manifest.json`
- Produce: `experiments/evals/reports/r3_hybrid/train_metrics.json`
- Produce locally: `experiments/evals/reports/r3_hybrid/train_results.jsonl`
- Produce: `experiments/evals/reports/r3_hybrid/admission_decision.md`

**Interfaces:**
- Consumes: frozen E0 TRAIN results, frozen TechQA documents, committed `r3_gate.json`, `TechQABM25Retriever`, and `fuse_rrf()`.
- Produces: Dense/BM25/Hybrid metrics at `recall@20`, `recall@100`, `mrr@10`; direct complementarity counts; local p50/p95 BM25/fusion latency; deterministic paid-R4 admission decision.

- [ ] **Step 1: Write failing evaluator tests**

```python
from experiments.evals.eval_techqa_hybrid import collapse_document_ids


def test_collapse_document_ids_preserves_first_occurrence():
    assert collapse_document_ids(["a", "a", "b", "c", "b"]) == [
        "a",
        "b",
        "c",
    ]
```

Add a two-query synthetic evaluation with one Dense-only gold hit and one BM25-only gold hit. Assert `dense_only_hits == 1`, `bm25_only_hits == 1`, and the fused run is evaluated at Recall@20 and Recall@100.

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_eval_techqa_hybrid.py -v
```

- [ ] **Step 3: Implement the R3 result/summary contract**

```python
@dataclass(frozen=True)
class TechQAHybridResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    dense_document_ids: tuple[str, ...]
    bm25_document_ids: tuple[str, ...]
    hybrid_document_ids: tuple[str, ...]
    bm25_latency_ms: float
    fusion_latency_ms: float


@dataclass(frozen=True)
class TechQAHybridSummary:
    query_count: int
    dense_metrics: dict[str, float]
    bm25_metrics: dict[str, float]
    hybrid_metrics: dict[str, float]
    dense_hit100: int
    bm25_hit100: int
    hybrid_hit100: int
    dense_only_hits: int
    bm25_only_hits: int
    both_hits: int
    neither_hits: int
    recovered_dense_misses: int
    lost_dense_hits: int
    bm25_latency_p50_ms: float
    bm25_latency_p95_ms: float
    fusion_latency_p50_ms: float
    fusion_latency_p95_ms: float
```

For each TRAIN query:
1. collapse all frozen E0 `raw_document_ids` from the Top-100 **chunk** pool into the Dense document ranking by first occurrence;
2. query BM25 Top-100 documents using `question.rstrip()`;
3. fuse the two document rankings with equal-weight `rrf_k=60`, output Top-100;
4. evaluate each run with exactly `("recall@20", "recall@100", "mrr@10")`;
5. compute `dense_hit100`, `bm25_hit100`, and `hybrid_hit100` directly from persisted per-query rankings, not by rounding aggregate recall;
6. compute Dense-only, BM25-only, both, neither, recovered Dense misses, and lost Dense hits directly from per-query gold membership.

- [ ] **Step 4: Implement a manifest that locks the local pilot identity**

The formal manifest must include these exact semantic fields, with SHA values computed from local frozen files at materialization time:

```json
{
  "benchmark": "TechQA-RAG-Eval",
  "run": "r3_hybrid_pilot",
  "split": "train",
  "query_count": 450,
  "dense_source": {
    "candidate_chunk_k": 100,
    "document_rule": "first occurrence over frozen E0 raw_document_ids"
  },
  "bm25": {
    "library": "bm25s",
    "version": "0.3.10",
    "method": "lucene",
    "k1": 1.5,
    "b": 0.75,
    "backend": "numpy",
    "candidate_document_k": 100,
    "query_normalization": "rstrip",
    "tokenizer_regex": "[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
  },
  "rrf": {
    "rrf_k": 60,
    "top_k": 100,
    "weights": "equal"
  },
  "provider_calls": 0
}
```

In addition, persist `dense_source.results_sha256`, `gate_sha256`, and frozen TechQA corpus/query/qrels hashes from `datasets/techqa/manifest.json`.

- [ ] **Step 5: Implement gate evaluation from direct hit counts**

```python
net_gain_cases = summary.hybrid_hit100 - summary.dense_hit100
admitted = (
    summary.recovered_dense_misses
    >= gate["required_recovered_dense_misses"]
    and net_gain_cases >= gate["required_net_gain_cases"]
)
```

`admission_decision.md` prints the committed thresholds first, then observed direct counts/metrics, then exactly one status: `ADMIT_PAID_R4` or `SKIP_PAID_R4`. It never mutates thresholds.

- [ ] **Step 6: Run GREEN and Ruff**

```powershell
pytest tests/test_eval_techqa_hybrid.py tests/test_bm25_techqa.py tests/test_rrf.py tests/test_ranx_eval_adapter.py -v
ruff check experiments/evals/eval_techqa_hybrid.py experiments/evals/retrievers/bm25_techqa.py experiments/evals/ir/rrf.py experiments/evals/ir/ranx_adapter.py tests/test_eval_techqa_hybrid.py tests/test_bm25_techqa.py tests/test_rrf.py tests/test_ranx_eval_adapter.py
```

- [ ] **Step 7: Zero-provider preflight**

```powershell
python -m experiments.evals.eval_techqa_hybrid --preflight
```

Preflight must report: TRAIN count `450`; TechQA corpus count `28481`; frozen E0 input hash; frozen gate hash; `bm25s=0.3.10`; `provider_calls=0`; no DEV artifact path opened.

- [ ] **Step 8: Run the formal local R3 pilot**

```powershell
python -m experiments.evals.eval_techqa_hybrid
```

This may use the frozen Hugging Face dataset/cache and local CPU/RAM. It must not call embedding, rerank, generator, or Judge APIs.

- [ ] **Step 9: Verify the formal artifacts independently**

```powershell
python -m experiments.evals.eval_techqa_hybrid --verify
```

Verification must rebuild all three ranx metric sets from `train_results.jsonl`, recompute direct hit/complementarity counts and the gate status, validate manifest/input hashes, and end with:

```text
R3 HYBRID ARTIFACT VERIFICATION = OK
```

- [ ] **Step 10: Version compact R3 evidence and update the ledger**

Keep `train_results.jsonl` local if large and record its SHA256 in `train_manifest.json`. Commit the compact artifacts and set R3 actual spend to `CNY 0 API`.

```bash
git add experiments/evals/eval_techqa_hybrid.py tests/test_eval_techqa_hybrid.py experiments/evals/reports/r3_hybrid/train_manifest.json experiments/evals/reports/r3_hybrid/train_metrics.json experiments/evals/reports/r3_hybrid/admission_decision.md experiments/evals/reports/cost_ledger.md
git commit -m "feat(eval): complete local TechQA hybrid pilot"
```

**R3 exit gate:** zero-provider verification passes and `admission_decision.md` is frozen. Stop here. If status is `ADMIT_PAID_R4`, write a separate Hybrid+Rerank implementation plan with fresh provider pricing/cost cap before any rerank call. If status is `SKIP_PAID_R4`, do not implement paid Hybrid+Rerank; write the G1 frozen-context Generator Ablation plan instead.

---

## Final verification for this plan

After Task 6:

```powershell
pytest `
  tests/test_rerank_evidence.py `
  tests/test_rerank_residual_analysis.py `
  tests/test_bm25_techqa.py `
  tests/test_rrf.py `
  tests/test_ranx_eval_adapter.py `
  tests/test_eval_techqa_hybrid.py `
  -v

ruff check `
  experiments/evals/rerank_evidence.py `
  experiments/evals/rerank_residual_analysis.py `
  experiments/evals/retrievers/bm25_techqa.py `
  experiments/evals/ir/rrf.py `
  experiments/evals/ir/ranx_adapter.py `
  experiments/evals/eval_techqa_hybrid.py `
  tests/test_rerank_evidence.py `
  tests/test_rerank_residual_analysis.py `
  tests/test_bm25_techqa.py `
  tests/test_rrf.py `
  tests/test_ranx_eval_adapter.py `
  tests/test_eval_techqa_hybrid.py
```

No completion claim is allowed without fresh output from this focused suite plus the formal R3 `--verify` result.
