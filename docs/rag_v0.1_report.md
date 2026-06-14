# RAG v0.1 Baseline Report

## 1. Scope

This report records the current RAG v0.1 baseline before moving into the next project phase.

Current project branch:

```text
learn-rag
```

Current RAG components:

```text
/rag/search
/rag/ask
JSON index retrieval
Chroma retrieval
RAG service layer
RAG API tests
RAG service tests
Retrieval eval scripts
Enterprise support documents with metadata
Tenant and category metadata filtering
```

The purpose of this report is to freeze the current baseline, record the known evaluation results, document the root cause of the previous invalid Chroma evaluation result, and summarize the current Enterprise RAG retrieval baseline.

---

## 2. Eval Dataset Separation

The eval datasets are now separated by corpus:

| Eval Script                                         | Eval Dataset                      | Target Index | Corpus Type             |
| --------------------------------------------------- | --------------------------------- | ------------ | ----------------------- |
| `python -m experiments.evals.eval_retrieval`        | `eval_learning_questions.jsonl`   | JSON index   | Learning docs           |
| `python -m experiments.evals.eval_chroma_retrieval` | `eval_enterprise_questions.jsonl` | Chroma index | Enterprise support docs |

This separation avoids evaluating a retriever against a corpus that does not contain the expected documents.

The core rule is:

```text
Learning questions should be evaluated against the learning document index.
Enterprise support questions should be evaluated against the enterprise support document index.
```

---

## 3. JSON Index Baseline

Command:

```powershell
python -m experiments.evals.eval_retrieval
```

Result:

| Metric      | Value |
| ----------- | ----: |
| Total cases |    15 |
| hit@1       |  0.93 |
| hit@3       |  1.00 |

Main top1 miss:

```text
Question: embedding 在 RAG 中有什么作用？
Expected document: doc_embedding_notes
Top1 document: doc_rag_notes
Top2 document: doc_embedding_notes
```

Analysis:

```text
This is an understandable semantic overlap.

The RAG notes document also mentions embeddings as part of the RAG pipeline, while the embedding notes document focuses on embedding itself. Since the expected document appears in top3, this is a top1 miss rather than a retrieval failure.
```

Current conclusion:

```text
The JSON index learning baseline is healthy.
```

---

## 4. Chroma Enterprise Baseline

Command:

```powershell
python -m experiments.evals.eval_chroma_retrieval
```

The enterprise Chroma retrieval evaluation dataset was expanded from 10 cases to 30 cases.

The updated dataset covers 10 enterprise support documents, with approximately 3 questions per document. The covered categories include:

```text
IT
HR
finance
admin
security
```

Updated result:

| Metric          | Value |
| --------------- | ----: |
| Total cases     |    30 |
| hit@1           |  0.97 |
| hit@3           |  1.00 |
| top1 miss cases |     1 |
| failed cases    |     0 |

Raw summary:

```text
Total: 30
hit@1: 0.97
hit@3: 1.00

top1_miss_cases: 1
Failed cases: 0
```

Main top1 miss:

```text
Question: 发票报销有哪些要求？
Expected document: doc_invoice_rules
Top1 document: doc_travel_reimbursement
Top2 document: doc_invoice_rules
```

Top result distances:

```text
doc_travel_reimbursement distance=0.6810
doc_invoice_rules        distance=0.6811
```

Analysis:

```text
This is a reasonable ambiguity case because the query combines "invoice" and "reimbursement".

The travel reimbursement document contains detailed reimbursement material requirements, including invoices, receipts, travel records, hotel bills, payment records, and approval screenshots.

The invoice rules document focuses more specifically on supplier invoices, procurement, payment rules, tax information, contract matching, and acceptance responsibilities.

The top two distances are nearly identical, so this is effectively a near-tie between two semantically close finance documents.
```

Current conclusion:

```text
The Chroma enterprise support baseline is healthy.

All 30 enterprise eval cases retrieved the expected document within top3.
The only top1 miss is a document-boundary ambiguity between two finance-related policies.
```

---

## 5. Root Cause of the Previous Chroma 0.00 Result

Earlier Chroma evaluation showed:

```text
hit@1 = 0.00
hit@3 = 0.00
```

This was not caused by a Chroma retriever failure.

Root cause:

```text
The Chroma collection contained only enterprise support documents, while the eval script was still using learning-doc questions.

Therefore, the expected documents such as doc_fastapi_notes, doc_docker_notes, doc_rag_notes, doc_embedding_notes, and doc_sqlmodel_notes were not present in the Chroma collection.
```

Observed Chroma collection documents:

```text
doc_data_access_policy
doc_device_borrowing
doc_email_login_faq
doc_invoice_rules
doc_leave_policy
doc_meeting_room
doc_onboarding_process
doc_outsource_account
doc_travel_reimbursement
doc_vpn_guide
```

Missing learning documents in Chroma:

```text
doc_fastapi_notes
doc_docker_notes
doc_rag_notes
doc_embedding_notes
doc_sqlmodel_notes
```

Final diagnosis:

```text
The previous Chroma 0.00 result reflected a corpus-eval mismatch, not a retriever failure.
```

---

## 6. Fix Applied

The eval setup was adjusted as follows:

```text
eval_learning_questions.jsonl
- Used by JSON index eval
- Covers FastAPI, Docker, RAG, Embedding, SQLModel learning documents

eval_enterprise_questions.jsonl
- Used by Chroma eval
- Covers enterprise support documents such as VPN, email, leave policy, onboarding, reimbursement, invoice rules, meeting rooms, device borrowing, data access, and outsource accounts
```

The eval scripts now follow this mapping:

| Script                     | Eval File                        |
| -------------------------- | -------------------------------- |
| `eval_retrieval.py`        | `eval_core.LEARNING_EVAL_FILE`   |
| `eval_chroma_retrieval.py` | `eval_core.ENTERPRISE_EVAL_FILE` |

This makes the evaluation target explicit and avoids silently reusing the wrong dataset.

---

## 7. Metadata Filter Validation

Enterprise support documents now carry metadata through the RAG pipeline.

Current metadata fields:

```text
tenant_id
category
```

The metadata flow is:

```text
Document
↓
Chunk
↓
Chroma metadata
↓
search_chroma filter
↓
ask_rag sources
↓
service layer
↓
/rag/search and /rag/ask API responses
```

Manual Chroma filter verification showed that category filtering works as expected.

Example behavior:

```text
VPN query + category=None
-> returns doc_vpn_guide from category=it

VPN query + category=it
-> returns doc_vpn_guide from category=it

VPN query + category=hr
-> returns only HR documents, not doc_vpn_guide

Leave policy query + category=hr
-> returns HR documents

Leave policy query + category=it
-> returns only IT documents, not doc_leave_policy
```

Current conclusion:

```text
Chroma metadata filtering is working.
tenant_id and category can restrict the retrieval scope.
```

---

## 8. Test Coverage Added

The project now includes tests for metadata filter behavior.

### Chroma Filter Unit Tests

`tests/test_query_chroma.py` validates the `build_where_filter` behavior.

Covered cases:

```text
No filters
tenant_id only
category only
tenant_id + category
```

Expected Chroma `where` filter behavior:

```text
No filters:
None

tenant_id only:
{"tenant_id": "tenant_demo"}

category only:
{"category": "it"}

tenant_id + category:
{
  "$and": [
    {"tenant_id": "tenant_demo"},
    {"category": "it"}
  ]
}
```

### RAG API Metadata Filter Tests

`tests/test_rag_api.py` validates that API requests pass category metadata through the router layer.

Covered API paths:

```text
/rag/search
/rag/ask
```

Validated behavior:

```text
request.category is passed into the service layer
tenant_id is provided by the server-side router context
response results include tenant_id and category
source previews are normalized
service failures are converted into HTTP 500 responses
```

### Regression Test Set

The current relevant regression test set includes:

```powershell
pytest tests/test_query_chroma.py
pytest tests/test_rag_api.py
pytest tests/test_rag_service.py
```

Expected result:

```text
tests/test_query_chroma.py  4 passed
tests/test_rag_api.py       8 passed
tests/test_rag_service.py   2 passed
```

---

## 9. Current Baseline Summary

| Retriever  | Corpus                  | Total Cases | hit@1 | hit@3 | Status  |
| ---------- | ----------------------- | ----------: | ----: | ----: | ------- |
| JSON index | Learning docs           |          15 |  0.93 |  1.00 | Healthy |
| Chroma     | Enterprise support docs |          30 |  0.97 |  1.00 | Healthy |

The current RAG baseline is valid.

The expanded enterprise eval confirms that the Chroma retriever is stable enough for the Enterprise RAG v1 cleanup stage.

---

## 10. Current Limitations

The current baseline is still intentionally minimal.

Known limitations:

```text
1. The enterprise eval set has been expanded to 30 cases, but it is still a small curated dataset.
2. The Chroma enterprise corpus currently has 10 enterprise support documents.
3. The current eval mainly measures document-level retrieval hit rate.
4. MRR and latency metrics are not yet included in the report output.
5. The invoice-related top1 miss shows that semantically close finance documents can still compete with each other.
6. tenant_id is currently provided by a server-side mock tenant context rather than a real authentication or tenant-resolution system.
7. The current tests validate metadata propagation and filter construction, but not a full production multi-tenant authorization model.
```

These are acceptable at the current stage because the goal of this step is baseline validation and Enterprise RAG cleanup, not final RAG optimization.

---

## 11. Next Step

The project can now continue from baseline validation into the next phase:

```text
Enterprise RAG v1 cleanup
```

Completed cleanup items:

```text
1. Verify tenant_id and category metadata consistency.
2. Confirm Chroma category filtering behavior.
3. Confirm /rag/search passes category to the service layer.
4. Confirm /rag/ask passes category to the service layer.
5. Add tests for Chroma metadata filter construction.
6. Add API tests for metadata filter propagation.
7. Expand enterprise eval cases from 10 to 30.
8. Record the expanded enterprise eval result in this report.
```

Recommended next tasks:

```text
1. Add MRR or recall@k metrics to the eval report.
2. Add simple latency timing to retrieval eval output.
3. Review the known finance ambiguity case and decide whether it should be solved by query wording, document boundaries, reranking, or answer-level citation handling.
4. Consider adding category-specific eval slices, such as IT-only, HR-only, finance-only, admin-only, and security-only.
5. Prepare the project for the next feature phase.
```

After this cleanup, the project should move into:

```text
Ticket CRUD
Ticket Agent preview / confirm
AgentOps audit records
```

---

## 12. Final Conclusion

The RAG v0.1 baseline is now stable.

The key engineering lesson from this phase is:

```text
Evaluation results are only meaningful when the eval dataset, expected documents, and target index corpus are aligned.
```

The current Enterprise RAG baseline is also stronger than the initial baseline because it now includes:

```text
separated eval datasets
expanded enterprise eval coverage
document-level retrieval metrics
known ambiguity analysis
tenant_id and category metadata propagation
Chroma filter validation
API-level metadata propagation tests
```

This baseline can be used as the starting point for the next project stage.
