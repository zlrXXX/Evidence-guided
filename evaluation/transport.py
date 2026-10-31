import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid

DESTINATION = "https://api.openai.com/v1"

def check(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def atomic(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('xb') as handle:
            os.chmod(temporary, 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def save(path, value):
    atomic(path, encoded(value))


def safe(root, relative):
    p = root / relative
    check(p.resolve().is_relative_to(root.resolve()) and p.resolve() == p,
          'Invalid or symlinked package path')
    return p


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def request_bytes(requests):
    return ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n'
                   for r in requests).encode()


def cost_bound(requests, pricing):
    
    counts = [len(r['body']['messages'][0]['content'].encode()) + 256 for r in requests]
    check(counts and max(counts) < pricing['long_context_threshold'], 'Long-context pricing needs review')
    upper = (sum(counts) * pricing['batch_input_usd_per_million'] +
             len(requests) * 1024 * pricing['batch_output_usd_per_million']) / 1e6
    return {'method': 'UTF-8 bytes + 256 framing tokens per request; full 1024 output allowance; 25% reserve',
            'input_token_bound': sum(counts), 'output_token_bound': len(requests) * 1024,
            'uncached_full_output_bound_usd': round(upper, 6),
            'reservation_usd': math.ceil(upper * 1.25 * 100) / 100,
            'cache_savings_assumed': False, 'automatic_retries': 0}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def tls_opener():
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                      urllib.request.HTTPSHandler(context=context))


class API:
    def __init__(self, key):
        self.key = key
        self.opener = tls_opener()

    def call(self, method, endpoint, payload=None, content_type='application/json', raw=False):
        check(endpoint.startswith('/') and not endpoint.startswith('//'), 'Invalid endpoint')
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode()
        headers = {'Content-Type': content_type}
        if self.key is not None:
            headers['Authorization'] = 'Bearer ' + self.key
        request = urllib.request.Request(DESTINATION + endpoint, data=payload, method=method, headers=headers)
        try:
            with self.opener.open(request, timeout=90) as response:
                result = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'OpenAI HTTP {error.code}; body suppressed; no automatic retry') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError('HTTPS request failed; POST outcome may be ambiguous; no automatic retry') from None
        return result if raw else json.loads(result)


def remote_id(value, kind):
    check(isinstance(value, str) and re.fullmatch(('file-' if kind == 'file' else 'batch_') + r'[A-Za-z0-9_-]+', value), 'Invalid remote object ID')
    return value


def batch_matches(batch, ledger, manifest):
    check(batch.get('input_file_id') == ledger.get('input_file_id') and
          batch.get('endpoint') == '/v1/chat/completions' and
          batch.get('metadata', {}).get('request_binding') == manifest['binding'] and
          batch.get('metadata', {}).get('evaluation_id') == manifest['run_id'], 'Remote batch binding mismatch')
    if ledger.get('batch_id'):
        check(batch['id'] == ledger['batch_id'], 'Remote batch identity mismatch')


def submit(root, manifest, ledger, api):
    
    if not ledger.get('input_file_id'):
        boundary = 'ctrs-' + uuid.uuid4().hex
        filename = manifest['run_id'] + '.' + manifest['binding'][:12] + '.jsonl'
        raw = (root / 'requests.jsonl').read_bytes()
        payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                   f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                   'Content-Type: application/jsonl\r\n\r\n').encode() + raw + f'\r\n--{boundary}--\r\n'.encode()
        ledger.update(state='uploading', upload_filename=filename)
        save(root / 'ledger.json', ledger)
        uploaded = api.call('POST', '/files', payload, f'multipart/form-data; boundary={boundary}')
        ledger.update(input_file_id=remote_id(uploaded['id'], 'file'), state='uploaded')
        save(root / 'ledger.json', ledger)
    uploaded_raw = api.call('GET', '/files/' + remote_id(ledger['input_file_id'], 'file') + '/content', raw=True)
    check(uploaded_raw == (root / 'requests.jsonl').read_bytes(), 'Uploaded bytes differ; batch not created')
    ledger.update(state='creating', reserved_usd=manifest['cost']['reservation_usd'])
    save(root / 'ledger.json', ledger)
    batch = api.call('POST', '/batches', {'input_file_id': ledger['input_file_id'],
        'endpoint': '/v1/chat/completions', 'completion_window': '24h',
        'metadata': {'evaluation_id': manifest['run_id'], 'request_binding': manifest['binding'], 'judge': 'terra'}})
    
    ledger.update(batch_id=remote_id(batch['id'], 'batch'), state='created')
    save(root / 'ledger.json', ledger)
    batch_matches(batch, ledger, manifest)
    save(root / 'batch-status.json', batch)
    return {'state': 'created', 'batch_id': ledger['batch_id']}


def list_remote(api, endpoint):
    after = None
    seen = set()
    for _ in range(1000):
        query = {'limit': 100}
        if endpoint == '/files':
            query['purpose'] = 'batch'
        if after:
            query['after'] = after
        page = api.call('GET', endpoint + '?' + urllib.parse.urlencode(query))
        yield from page['data']
        if not page.get('has_more'):
            return
        check(page['data'] and page['data'][-1]['id'] not in seen, 'Remote pagination did not advance')
        after = page['data'][-1]['id']
        seen.add(after)
    raise RuntimeError('Reconciliation listing incomplete; no new submission permitted')


def reconcile(root, manifest, ledger, api):
    if ledger.get('batch_id'):
        batch = api.call('GET', '/batches/' + remote_id(ledger['batch_id'], 'batch'))
        batch_matches(batch, ledger, manifest)
        save(root / 'batch-status.json', batch)
        return {'state': ledger['state'], 'batch_id': ledger['batch_id']}
    if ledger['state'] == 'uploading':
        matches = [f for f in list_remote(api, '/files') if f.get('filename') == ledger['upload_filename']]
        check(len(matches) == 1, 'Upload remains ambiguous; zero/multiple matching files; no automatic retry')
        fid = remote_id(matches[0]['id'], 'file')
        check(api.call('GET', '/files/' + fid + '/content', raw=True) == (root / 'requests.jsonl').read_bytes(), 'Reconciled input bytes differ')
        ledger.update(state='uploaded', input_file_id=fid)
        save(root / 'ledger.json', ledger)
        return {'state': 'uploaded', 'inference_submitted': False}
    check(ledger['state'] == 'creating', 'No ambiguous operation to reconcile')
    matches = [b for b in list_remote(api, '/batches') if
               (b.get('metadata') or {}).get('request_binding') == manifest['binding'] and
               (b.get('metadata') or {}).get('evaluation_id') == manifest['run_id']]
    check(len(matches) == 1, 'Batch creation remains ambiguous; no automatic retry')
    batch = matches[0]
    batch_matches(batch, ledger, manifest)
    ledger.update(state='created', batch_id=remote_id(batch['id'], 'batch'))
    save(root / 'ledger.json', ledger)
    save(root / 'batch-status.json', batch)
    return {'state': 'created', 'batch_id': ledger['batch_id']}


def collect(root, manifest, ledger, api):
    batch = api.call('GET', '/batches/' + remote_id(ledger.get('batch_id'), 'batch'))
    batch_matches(batch, ledger, manifest)
    save(root / 'batch-status.json', batch)
    if batch['status'] not in {'completed', 'failed', 'expired', 'cancelled'}:
        return {'state': batch['status'], 'batch_id': ledger['batch_id']}
    artifacts = ledger.setdefault('collected_files', {})
    for stem in ('output', 'error'):
        fid = batch.get(stem + '_file_id')
        if not fid:
            continue
        remote_id(fid, 'file')
        dest = root / (stem + '.jsonl')
        if stem in artifacts:
            check(artifacts[stem]['file_id'] == fid and digest(dest.read_bytes()) == artifacts[stem]['sha256'], 'Collected output changed')
        else:
            raw = api.call('GET', '/files/' + fid + '/content', raw=True)
            if dest.exists():
                check(dest.read_bytes() == raw, 'Existing output differs; not overwritten')
            else:
                atomic(dest, raw)
            artifacts[stem] = {'file_id': fid, 'sha256': digest(raw)}
            save(root / 'ledger.json', ledger)
    ledger['state'] = batch['status']
    save(root / 'ledger.json', ledger)
    return {"state": batch["status"], "batch_id": ledger["batch_id"], "collected_files": artifacts}
