"""Synthetic receipts: complete replay and interrupted-case Dense decisions."""
import json
from types import SimpleNamespace

import pytest
from sqlmodel import Session, select

from test_runtime_retrieval import KEYS, plan, providers, runtime  # noqa: F401
from experiments.evals import runtime_retrieval as evaluation
from experiments.evals import runtime_retrieval_recovery as recovery
from models.document import DocumentChunk


def test_completed_case_uses_actual_receipts_not_a_new_hnsw_query(runtime, tmp_path, monkeypatch):  # noqa: F811
    value, seen, root = plan(), [], tmp_path / 'run'
    saved = evaluation.save_case(value, 0, root, KEYS, providers(seen))
    monkeypatch.setattr(evaluation.hybrid, 'get_collection', lambda: pytest.fail('No completed-case HNSW query'))
    assert recovery.recover_case(value, 0, root, recovery.history(value, root)) == saved
    assert len(seen) == 3
    with Session(evaluation.hybrid.engine) as session:
        row = session.exec(select(DocumentChunk).where(DocumentChunk.embedding_id == saved['result']['chunk_rankings']['D'][0])).one()
        row.content += ' changed'
        session.add(row)
        session.commit()
    with pytest.raises(ValueError, match='SQL bodies'):
        recovery.recover_case(value, 0, root, recovery.history(value, root))
    assert len(seen) == 3


@pytest.mark.parametrize('field', ['rerank_order', 'fusion', 'question'])
def test_recomputed_case_digest_cannot_hide_wire_projection_changes(runtime, tmp_path, field):  # noqa: F811
    value, seen, root = plan(), [], tmp_path / 'run'
    saved = evaluation.save_case(value, 0, root, KEYS, providers(seen))
    row = saved['result']
    if field == 'rerank_order':
        row['chunk_rankings']['DR'].reverse()
        row['document_rankings']['DR'].reverse()
    elif field == 'fusion':
        row['fused_chunk_ids'].reverse()
    else:
        row['question_sha256'] = 'changed'
    saved['result_sha256'] = evaluation.wire.digest(row)
    (root / 'cases/000.json').write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        recovery.recover_case(value, 0, root, recovery.history(value, root))
    assert len(seen) == 3


def test_partial_case_replays_dense_selection_without_resending_e_or_dr(runtime, tmp_path, monkeypatch):  # noqa: F811
    value, seen, root = plan(), [], tmp_path / 'run'
    captures = tmp_path / 'captures'
    captures.mkdir()
    delegates = providers(seen)
    original = evaluation.make_client

    def stop_before_h(*args, **kwargs):
        if args[2].endswith(':H'):
            raise RuntimeError('synthetic_stop_before_new_send')
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, 'make_client', stop_before_h)
    with pytest.raises(RuntimeError, match='synthetic_stop_before_new_send'):
        recovery.save_unfinished(value, 0, root, KEYS, delegates, capture_root=captures, adapter_sha='synthetic')
    assert len(seen) == 2 and not (root / 'wire/pending.json').exists()
    monkeypatch.setattr(evaluation, 'make_client', original)
    monkeypatch.setattr(evaluation.hybrid, 'get_collection', lambda: SimpleNamespace(query=lambda **kw: pytest.fail('Do not rerun a captured Dense decision')))
    saved = recovery.save_unfinished(value, 0, root, KEYS, delegates, capture_root=captures, adapter_sha='synthetic')
    assert len(seen) == 3 and recovery.recover_case(value, 0, root, recovery.history(value, root)) == saved
    capture = captures / '000.json'
    record = json.loads(capture.read_text())
    record['identity']['adapter_sha256'] = 'changed'
    capture.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='Dense decision identity/hash'):
        recovery.save_unfinished(value, 0, root, KEYS, delegates, capture_root=captures, adapter_sha='synthetic')
    assert len(seen) == 3


def test_explicit_adapter_keeps_legacy_plan_and_only_sends_unfinished_case(runtime, tmp_path, monkeypatch):  # noqa: F811
    value, seen, output = plan(), [], tmp_path / 'bundle'
    output.mkdir()
    root = output / 'run'
    delegates = providers(seen)
    evaluation.save_case(value, 0, root, KEYS, delegates)
    prepared = output / 'prepared.json'
    prepared.write_text(json.dumps({'plan': value}), encoding='utf-8')
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'human_approval_reference':'SYNTHETIC_ONLY',
        'prepared_sha256':recovery.runner.sha(prepared), 'scope':'DEV160_RETRIEVAL_ONLY',
        'max_provider_calls':480, 'accounting_limit_cny':'35', 'accept_reference_estimate':True,
        'current_prices_reconfirmed':True}), encoding='utf-8')
    monkeypatch.setattr(recovery.runner, 'require_container', lambda: None)
    def legacy_loop(*args):
        return [evaluation.save_case(value, i, root, KEYS, delegates) for i in range(2)]
    monkeypatch.setattr(recovery.runner, 'execute', legacy_loop)
    results = recovery.execute(output, approval, recovery.runner.sha(approval), recovery.runner.sha(recovery.__file__))
    assert len(results) == 2 and len(seen) == 6
    assert len(recovery.history(value, root)) == 6
    assert len(list(tmp_path.glob('receipt_recovery_*.json'))) == 1
