"""Explicit receipt recovery for an unchanged legacy phase-1 plan.

Completed rankings are historical observations, not new HNSW queries. Validate
them against their actual wire inputs/outputs and the approved SQL snapshot.
Unfinished cases durably capture Dense selection before any rerank is sent.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from experiments.evals import runtime_retrieval as evaluation
from experiments.evals import runtime_retrieval_runner as runner
from experiments.evals.runtime_retrieval_report import document_id


def history(plan, root):
    transport = evaluation.Phase1Transport(plan['contract'], root / 'wire',
        next(iter(plan['contract']['calls'])), httpx.MockTransport(lambda request: None))
    try:
        return transport._history()
    finally:
        transport.close()


def recover_case(plan, index, root, events):
    """No query/model client: verify original captured selections and responses."""
    artifact = runner.read_json(root / f'cases/{index:03d}.json')
    row = artifact['result']
    cid, question = plan['identity']['questions'][index]
    receipts = [event for event in events if event['call_id'].startswith(cid + ':')]
    if (set(artifact) != {'plan_sha256', 'index', 'result', 'result_sha256', 'calls'}
            or artifact['plan_sha256'] != evaluation.wire.digest(plan) or artifact['index'] != index
            or artifact['result_sha256'] != evaluation.wire.digest(row)
            or row['question_id'] != cid or row['question_sha256'] != hashlib.sha256(question.encode()).hexdigest()
            or [v['call_id'] for v in receipts] != [key for key in plan['contract']['calls'] if key.startswith(cid + ':')]
            or artifact['calls'] != [{key: v[key] for key in ('call_id', 'request_sha256', 'response_sha256')} for v in receipts]):
        raise ValueError('Saved case identity or actual wire receipts differ')
    ranks = row['chunk_rankings']
    if (set(ranks) != {'D', 'DR', 'H'} or set(row['document_rankings']) != set(ranks)
            or any(len(ids) != 100 or len(set(ids)) != 100 for ids in ranks.values())
            or set(ranks['D']) != set(ranks['DR']) or set(row['fused_chunk_ids']) != set(ranks['H'])
            or len(row['fused_chunk_ids']) != 100 or len(set(row['bm25_chunk_ids'])) != len(row['bm25_chunk_ids'])
            or len(row['bm25_chunk_ids']) > 100 or row['top14_chunk_ids'] != ranks['H'][:14]
            or evaluation.hybrid.fuse_rrf([ranks['D'], row['bm25_chunk_ids']], rrf_k=60, top_k=100) != row['fused_chunk_ids']):
        raise ValueError('Captured Dense/BM25/fusion/context rankings differ')
    ids = set(row['fused_chunk_ids'] + row['bm25_chunk_ids'] + ranks['D'])
    corpus = evaluation.hybrid.load_visible_chunks(evaluation.TENANT, evaluation.CATEGORY, embedding_ids=list(ids))
    if set(corpus) != ids:
        raise ValueError('Captured selections are not current authorized snapshot chunks')
    for arm, chunks in ranks.items():
        if list(dict.fromkeys(document_id(key) for key in chunks)) != row['document_rankings'][arm]:
            raise ValueError('Captured document collapse differs')
    for event in receipts:
        position = events.index(event)
        body = json.loads(runner.read_json(root / f'wire/requests/{position}.json')['wire_utf8'])
        payload = json.loads(runner.read_json(root / 'wire' / event['response_file'])['wire_utf8'])
        evaluation.validate_response(event['stage'], body, payload)
        if event['stage'] == 'embedding':
            if body != {'model': 'text-embedding-v4', 'input': [question], 'dimensions': 1024, 'encoding_format': 'float'}:
                raise ValueError('Captured query embedding input differs')
            continue
        arm = event['call_id'].rsplit(':', 1)[1]
        input_ids = ranks['D'] if arm == 'DR' else row['fused_chunk_ids']
        if (body != {'model': 'qwen3-rerank', 'query': question,
                     'documents': [corpus[key].content for key in input_ids], 'top_n': 100,
                     'instruct': evaluation.DEFAULT_RERANK_INSTRUCTION}
                or ranks[arm] != [input_ids[result['index']] for result in payload['results']]):
            raise ValueError('Captured ranking does not match actual provider input/output and SQL bodies')
    return artifact


def save_unfinished(plan, index, root, keys, delegates, *, capture_root, adapter_sha, **kwargs):
    """Capture once, then replay the exact Dense decision after interruptions."""
    collection = evaluation.hybrid.get_collection()
    path = capture_root / f'{index:03d}.json'

    def query(**query_args):
        identity = {'plan_sha256': evaluation.wire.digest(plan), 'adapter_sha256': adapter_sha,
                    'index': index, 'query_sha256': evaluation.wire.digest(query_args)}
        if path.exists():
            record = runner.read_json(path)
            if (set(record) != {'identity', 'result', 'result_sha256'} or record['identity'] != identity
                    or record['result_sha256'] != evaluation.wire.digest(record['result'])):
                raise ValueError('Persisted Dense decision identity/hash changed')
            return record['result']
        raw = collection.query(**query_args)
        result = {key: raw[key] for key in ('ids', 'distances')}
        evaluation.wire.write_json(path, {'identity': identity, 'result': result,
                                         'result_sha256': evaluation.wire.digest(result)})
        return result

    with patch.object(evaluation.hybrid, 'get_collection', lambda: SimpleNamespace(query=query)):
        return evaluation.save_case(plan, index, root, keys, delegates, **kwargs)


def execute(output, authorization, authorization_sha, adapter_sha):
    if runner.sha(__file__) != adapter_sha:
        raise ValueError('Explicit committed recovery-adapter SHA required')
    runner.require_container()
    output = Path(output).resolve()
    runner.validate_approval(authorization, authorization_sha, output / 'prepared.json')
    plan = runner.read_json(output / 'prepared.json')['plan']
    root = output / 'run'
    if (root / 'wire/pending.json').exists() or (root / 'halt.json').exists():
        raise RuntimeError('Pending/halt requires separate audit; recovery never clears it')
    events = history(plan, root)
    files = sorted((root / 'cases').iterdir())
    if [p.name for p in files] != [f'{i:03d}.json' for i in range(len(files))]:
        raise ValueError('Contiguous saved-case prefix required')
    hashes = [runner.sha(path) for path in files]
    capture_root = output.parent / ('dense_decisions_' + evaluation.wire.digest(plan))
    capture_root.mkdir(exist_ok=True)
    binding = {'adapter_sha256': adapter_sha, 'plan_sha256': evaluation.wire.digest(plan),
               'saved_case_sha256': hashes, 'completed_wire_prefix_sha256': runner.sha(root / 'wire/attempts.jsonl'),
               'completed_calls': len(events), 'recovery_only_no_algorithm_or_model_change': True}
    receipt = output.parent / f'receipt_recovery_{len(files):03d}_{len(events):03d}.json'
    if receipt.exists():
        if runner.read_json(receipt) != binding:
            raise ValueError('Recovery binding changed')
    else:
        evaluation.wire.write_json(receipt, binding)
    original_save = evaluation.save_case

    def resume_case(current_plan, index, current_root, keys, delegates, **kwargs):
        if evaluation.wire.digest(current_plan) != binding['plan_sha256'] or Path(current_root) != root:
            raise ValueError('Legacy plan or checkpoint root changed')
        if index < len(files):
            if runner.sha(files[index]) != hashes[index]:
                raise ValueError('Saved case changed during recovery')
            return recover_case(plan, index, root, events)
        # Call the unchanged legacy saver without recursing through this wrapper.
        with patch.object(evaluation, 'save_case', original_save):
            return save_unfinished(plan, index, root, keys, delegates,
                                   capture_root=capture_root, adapter_sha=adapter_sha, **kwargs)

    with patch.object(evaluation, 'save_case', resume_case):
        return runner.execute(output, authorization, authorization_sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('output', 'authorization', 'authorization-sha256', 'adapter-sha256'):
        parser.add_argument('--' + argument, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.output, args.authorization, args.authorization_sha256, args.adapter_sha256)), flush=True)
    except Exception as error:
        # No SDK messages, source bodies, credentials or local variables.
        chain = []
        while error is not None:
            frames, tb = [], error.__traceback__
            while tb is not None:
                frames.append([Path(tb.tb_frame.f_code.co_filename).name, tb.tb_frame.f_code.co_name, tb.tb_lineno])
                tb = tb.tb_next
            chain.append({'type': type(error).__name__, 'frames': frames})
            error = error.__cause__ or error.__context__
        print(json.dumps({'status': 'STOPPED_NO_AUTOMATIC_RETRY', 'safe_error_chain': chain}), flush=True)
        raise SystemExit(1) from None
