import argparse
import collections
import contextlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import protocol_en
import protocol_zh
from transport import (
    API, DESTINATION, atomic, batch_matches, check, collect, cost_bound,
    digest, encoded, load, now, reconcile, remote_id, request_bytes,
    rows, safe, save, submit,
)

HERE = Path(__file__).resolve().parent
PROTOCOLS = {'en': protocol_en, 'zh': protocol_zh}
CRITERIA = protocol_en.CRITERIA
MODEL = protocol_en.MODEL
SCHEMA = 'trace-ctrs-v1'
CODE_FILES = ('ctrs.py', 'transport.py', 'protocol_en.py', 'protocol_zh.py')


def body(turns, rubric, language):
    return {'model': MODEL, 'temperature': 0, 'reasoning_effort': 'none',
            'prompt_cache_options': {'mode': 'explicit'}, 'max_completion_tokens': 1024,
            'messages': [{'role': 'user', 'content': rubric.format(
                conversation=PROTOCOLS[language].render_transcript(turns))}]}


def requests_for(records, root, language):
    prompts = {c: (root / 'prompts' / (c + '.txt')).read_text(encoding='utf-8') for c in CRITERIA}
    requests, mapping = [], []
    for record in records:
        turns = load(safe(root, record['input']))
        for criterion in CRITERIA:
            identity = '|'.join((record['cohort'], record['session_id'], criterion, record['transcript_sha256']))
            cid = 'ctrs-' + digest(identity.encode())[:40]
            mapping.append({k: record[k] for k in (
                'cohort', 'session_id', 'profile_id', 'dataset', 'backbone',
                'source_sha256', 'transcript_sha256')}
                | {'custom_id': cid, 'criterion': criterion, 'language': language})
            requests.append({'custom_id': cid, 'method': 'POST', 'url': '/v1/chat/completions',
                             'body': body(turns, prompts[criterion], language)})
    check(len({r['custom_id'] for r in requests}) == len(requests), 'Duplicate request ID')
    return requests, mapping


def identifier(value):
    check(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value), 'Invalid identifier')
    return value


def prepare(config_path, root, language, pricing_path):
    config = load(config_path)
    identifier(config['run_id'])
    check(not root.exists(), 'Run exists; choose a new output directory')
    check(config['cohorts'], 'No cohorts')
    root.mkdir(parents=True)
    pricing = load(pricing_path)
    check(pricing['model'] == MODEL and pricing['checked_utc'], 'Pricing model/date mismatch')
    check(all(isinstance(pricing.get(k), (int, float)) and math.isfinite(pricing[k]) and pricing[k] >= 0
              for k in ('batch_input_usd_per_million', 'batch_cached_input_usd_per_million',
                        'batch_cache_write_usd_per_million', 'batch_output_usd_per_million')), 'Invalid pricing rates')
    save(root / 'pricing.json', pricing)
    for name in CODE_FILES:
        atomic(root / 'runner' / name, (HERE / name).read_bytes())
    for c in CRITERIA:
        atomic(root / 'prompts' / (c + '.txt'), (HERE / 'prompts' / language / (c + '.txt')).read_bytes())
    records, cohorts = [], []
    for cohort in config['cohorts']:
        name = identifier(cohort['cohort'])
        dataset, backbone = identifier(cohort['dataset']), identifier(cohort['backbone'])
        check(cohort['sessions'], 'Empty cohort')
        ids, profiles = set(), set()
        for session in cohort['sessions']:
            sid, pid = identifier(session['session_id']), identifier(session['profile_id'])
            check(sid not in ids and pid not in profiles, 'Duplicate session/profile in cohort')
            ids.add(sid)
            profiles.add(pid)
            source = (config_path.parent / session['dialogue']).resolve()
            raw = source.read_bytes()
            if session.get('sha256'):
                check(digest(raw) == session['sha256'].lower(), 'Source dialogue hash mismatch')
            full = json.loads(raw)
            PROTOCOLS[language].render_transcript(full)
            check(len(full) % 2 == 0, 'Dialogue must contain complete counsellor/patient exchanges')
            turns = full[:80]
            full_rel = f'full/{name}/{sid}.json'
            rel = f'inputs/{name}/{sid}.json'
            atomic(safe(root, full_rel), raw)
            view_raw = raw if len(full) <= 80 else encoded(turns)
            atomic(safe(root, rel), view_raw)
            transcript = PROTOCOLS[language].render_transcript(turns)
            records.append({'cohort': name, 'session_id': sid, 'profile_id': pid,
                            'dataset': dataset, 'backbone': backbone, 'input': rel,
                            'full_dialogue': full_rel, 'full_sha256': digest(raw),
                            'source_sha256': digest(view_raw).upper(),
                            'transcript_sha256': digest(transcript.encode()).upper(),
                            'full_exchanges': len(full) // 2, 'judge_exchanges': len(turns) // 2})
        cohorts.append({'cohort': name, 'dataset': dataset, 'backbone': backbone,
                        'session_ids': sorted(ids), 'profile_ids': sorted(profiles), 'language': language})
    check(len({c['cohort'] for c in cohorts}) == len(cohorts), 'Duplicate cohort')
    records.sort(key=lambda r: (r['cohort'], r['session_id']))
    requests, mapping = requests_for(records, root, language)
    save(root / 'source-index.json', records)
    save(root / 'cohorts.json', cohorts)
    save(root / 'mapping.json', mapping)
    atomic(root / 'requests.jsonl', request_bytes(requests))
    manifest = {'schema': SCHEMA, 'run_id': config['run_id'], 'model': MODEL, 'language': language,
                'destination': DESTINATION, 'request_layout': 'six_independent_criteria_per_session',
                'created_utc': now().isoformat(), 'sessions': len(records), 'requests': len(requests),
                'maximum_exchanges': 40, 'cost': cost_bound(requests, pricing),
                'files': {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size,
                          'sha256': digest(p.read_bytes())} for p in sorted(root.rglob('*')) if p.is_file()}}
    manifest['binding'] = digest(encoded(manifest))
    save(root / 'manifest.json', manifest)
    save(root / 'ledger.json', {'binding': manifest['binding'], 'state': 'not_submitted'})
    validate(root)
    return {'status': 'PREPARED', 'sessions': len(records), 'requests': len(requests), 'cost': manifest['cost']}


def validate(root):
    manifest = load(root / 'manifest.json')
    check(manifest['schema'] == SCHEMA and manifest['model'] == MODEL and
          manifest['destination'] == DESTINATION and manifest['language'] in PROTOCOLS and
          manifest['request_layout'] == 'six_independent_criteria_per_session', 'Protocol contract mismatch')
    without = dict(manifest)
    binding = without.pop('binding')
    check(binding == digest(encoded(without)), 'Manifest binding mismatch')
    for rel, entry in manifest['files'].items():
        raw = safe(root, rel).read_bytes()
        check(len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], 'Protected file mismatch: ' + rel)
    for filename in CODE_FILES:
        check((root / 'runner' / filename).read_bytes() == (HERE / filename).read_bytes(), 'Runner differs from frozen run')
    language = manifest['language']
    protocol = PROTOCOLS[language]
    for criterion in CRITERIA:
        check((root / 'prompts' / (criterion + '.txt')).read_bytes() ==
              (HERE / 'prompts' / language / (criterion + '.txt')).read_bytes(), 'Rubric mismatch')
    index = load(root / 'source-index.json')
    check(len({(r['cohort'], r['session_id']) for r in index}) == len(index) == manifest['sessions'], 'Duplicate source')
    for record in index:
        raw = safe(root, record['input']).read_bytes()
        full_raw = safe(root, record['full_dialogue']).read_bytes()
        turns, full = json.loads(raw), json.loads(full_raw)
        check(0 < len(turns) <= 80 and len(turns) % 2 == 0 and len(full) % 2 == 0 and turns == full[:80], 'Invalid first-40 view')
        protocol.render_transcript(full)
        check(digest(full_raw) == record['full_sha256'] and digest(raw).upper() == record['source_sha256'] and
              digest(protocol.render_transcript(turns).encode()).upper() == record['transcript_sha256'], 'Dialogue binding mismatch')
    cohorts = load(root / 'cohorts.json')
    check({(c['cohort'], s) for c in cohorts for s in c['session_ids']} ==
          {(r['cohort'], r['session_id']) for r in index}, 'Cohort membership mismatch')
    for c in cohorts:
        subset = [r for r in index if r['cohort'] == c['cohort']]
        check(sorted(r['profile_id'] for r in subset) == c['profile_ids'] and
              len(set(c['profile_ids'])) == len(c['profile_ids']), 'Profile membership mismatch')
    requests, mapping = requests_for(index, root, language)
    check(request_bytes(requests) == (root / 'requests.jsonl').read_bytes() and
          mapping == load(root / 'mapping.json'), 'Request reconstruction mismatch')
    check(len(requests) == manifest['requests'] == manifest['sessions'] * 6 <= 50000 and
          (root / 'requests.jsonl').stat().st_size <= 200_000_000, 'Invalid Batch size')
    check(cost_bound(requests, load(root / 'pricing.json')) == manifest['cost'], 'Cost bound changed')
    return manifest


@contextlib.contextmanager
def lock(root):
    path = root / '.running'
    try:
        path.mkdir()
    except FileExistsError:
        raise RuntimeError('Run is locked; ensure its prior process has ended before removing .running') from None
    try:
        yield
    finally:
        path.rmdir()


def summarize(root, manifest):
    mapping = {r['custom_id']: r for r in load(root / 'mapping.json')}
    ledger = load(root / 'ledger.json')
    check(ledger['binding'] == manifest['binding'], 'Ledger binding mismatch')
    output = []
    for stem, entry in ledger.get('collected_files', {}).items():
        path = root / (stem + '.jsonl')
        check(digest(path.read_bytes()) == entry['sha256'], 'Raw output hash mismatch')
        output.extend(rows(path))
    seen, successes, failures = set(), [], []
    usage = dict(input=0, cached_input=0, cache_write=0, output=0, reasoning=0, total=0)
    for result in output:
        cid = result.get('custom_id')
        check(cid in mapping and cid not in seen, 'Unknown/duplicate output ID')
        seen.add(cid)
        try:
            score, explanation, response = PROTOCOLS[manifest['language']].parse_result(result)
            check(response['model'] == MODEL, 'Returned judge mismatch')
            u = response['usage']
            details = u.get('prompt_tokens_details') or {}
            reasoning = (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
            check(reasoning == 0, 'Unexpected reasoning tokens')
            check(0 <= details.get('cached_tokens', 0) + details.get('cache_write_tokens', 0) <= u['prompt_tokens'], 'Invalid cache usage')
            check(all(type(u.get(k)) is int and u[k] >= 0 for k in
                      ('prompt_tokens', 'completion_tokens', 'total_tokens')), 'Invalid usage fields')
            check(u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'], 'Usage total mismatch')
            usage['input'] += u['prompt_tokens']
            usage['output'] += u['completion_tokens']
            usage['total'] += u['total_tokens']
            usage['cached_input'] += details.get('cached_tokens', 0)
            usage['cache_write'] += details.get('cache_write_tokens', 0)
            successes.append(dict(mapping[cid], score=score, explanation=explanation))
        except (KeyError, ValueError, IndexError, TypeError) as error:
            failures.append({'custom_id': cid, 'error': str(error) if type(error) is ValueError else 'invalid_response'})
    failures += [{'custom_id': cid, 'error': 'missing_response'} for cid in sorted(set(mapping) - seen)]
    save(root / 'results/scores.json', successes)
    save(root / 'results/failures.json', failures)
    missing = {r['custom_id'] for r in failures}
    atomic(root / 'results/failed-requests.jsonl', request_bytes([r for r in rows(root / 'requests.jsonl') if r['custom_id'] in missing]))
    by_cohort = collections.defaultdict(list)
    for row in successes:
        by_cohort[row['cohort']].append(row)
    summaries = []
    for c in load(root / 'cohorts.json'):
        scores = by_cohort[c['cohort']]
        coverage = collections.defaultdict(set)
        for row in scores:
            check(row['session_id'] in c['session_ids'] and row['criterion'] not in coverage[row['session_id']], 'Invalid merged score')
            coverage[row['session_id']].add(row['criterion'])
        complete_ids = sorted(sid for sid in c['session_ids'] if coverage[sid] == set(CRITERIA))
        complete = len(complete_ids) == len(c['session_ids'])
        item = {'cohort': c['cohort'], 'expected_sessions': len(c['session_ids']), 'scores_available': len(scores),
                'complete_sessions': len(complete_ids), 'incomplete_session_ids': sorted(set(c['session_ids']) - set(complete_ids)),
                'status': 'COMPLETE' if complete else 'PENDING'}
        if complete:
            item['overall_mean'] = statistics.mean(r['score'] for r in scores)
            item['criteria'] = {k: {'mean': statistics.mean(r['score'] for r in scores if r['criterion'] == k),
                'standard_deviation': statistics.stdev(r['score'] for r in scores if r['criterion'] == k)
                if len(c['session_ids']) > 1 else None, 'n': len(c['session_ids'])} for k in CRITERIA}
        summaries.append(item)
    pricing = load(root / 'pricing.json')
    ordinary = usage['input'] - usage['cached_input'] - usage['cache_write']
    observed = (ordinary * pricing['batch_input_usd_per_million'] +
                usage['cached_input'] * pricing['batch_cached_input_usd_per_million'] +
                usage['cache_write'] * pricing['batch_cache_write_usd_per_million'] +
                usage['output'] * pricing['batch_output_usd_per_million']) / 1e6
    report = {'status': 'COMPLETE' if not failures else 'PARTIAL', 'new_scores': len(successes),
              'failed_or_missing': len(failures), 'usage_for_valid_responses': usage,
              'calculated_cost_for_valid_responses_usd': observed, 'cohorts': summaries}
    save(root / 'results/summary.json', report)
    return report


def execute(root, action, maximum_cost):
    with lock(root):
        manifest = validate(root)
        ledger = load(root / 'ledger.json')
        check(ledger['binding'] == manifest['binding'], 'Ledger binding mismatch')
        if action == 'summarize':
            return summarize(root, manifest)
        if action == 'submit':
            check(not ledger.get('batch_id') and ledger['state'] in {'not_submitted', 'uploaded'}, 'Existing/ambiguous submission; reconcile first')
            check(maximum_cost is not None and maximum_cost >= manifest['cost']['reservation_usd'], 'Provide --maximum-cost-usd covering the prepared cost bound')
            pricing = load(root / 'pricing.json')
            checked = now().fromisoformat(pricing['checked_utc'])
            check(checked.tzinfo is not None and 0 <= (now() - checked).total_seconds() <= 7 * 86400,
                  'Verify model pricing/endpoint support and prepare with pricing checked within seven days')
            ledger['maximum_cost_usd'] = maximum_cost
            save(root / 'ledger.json', ledger)
        elif action in {'status', 'collect'}:
            remote_id(ledger.get('batch_id'), 'batch')
        elif action == 'reconcile':
            check(ledger.get('batch_id') or ledger['state'] in {'uploading', 'creating'}, 'Nothing to reconcile')
        key = os.environ.get('OPENAI_API_KEY')
        check(key, 'Set OPENAI_API_KEY in your own environment')
        api = API(key)
        try:
            if action == 'submit':
                check(api.call('GET', '/models/' + MODEL).get('id') == MODEL, 'Configured judge unavailable')
                return submit(root, manifest, ledger, api)
            if action == 'reconcile':
                return reconcile(root, manifest, ledger, api)
            if action == 'status':
                batch = api.call('GET', '/batches/' + remote_id(ledger['batch_id'], 'batch'))
                batch_matches(batch, ledger, manifest)
                save(root / 'batch-status.json', batch)
                return {'batch_id': batch['id'], 'status': batch['status'], 'request_counts': batch.get('request_counts')}
            result = collect(root, manifest, ledger, api)
            return summarize(root, manifest) if 'collected_files' in result else result
        finally:
            api.key = None


def main():
    parser = argparse.ArgumentParser(description='Six independent CACTUS CTRS criterion requests per dialogue.')
    parser.add_argument('action', choices=['prepare', 'validate', 'submit', 'status', 'collect', 'reconcile', 'summarize'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--language', choices=['en', 'zh'])
    parser.add_argument('--pricing', type=Path)
    parser.add_argument('--maximum-cost-usd', type=float)
    args = parser.parse_args()
    root = args.run.resolve()
    if args.action == 'prepare':
        check(args.config and args.language and args.pricing, 'Provide --config, --language and --pricing')
        result = prepare(args.config.resolve(), root, args.language, args.pricing.resolve())
    elif args.action == 'validate':
        m = validate(root)
        result = {'status': 'PASS', 'sessions': m['sessions'], 'requests': m['requests'], 'language': m['language']}
    else:
        result = execute(root, args.action, args.maximum_cost_usd)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        message = str(error) if type(error) in (ValueError, RuntimeError) else 'Operation failed; no automatic retry.'
        print(json.dumps({'status': 'STOPPED', 'error_type': type(error).__name__, 'message': message}))
        sys.exit(1)
