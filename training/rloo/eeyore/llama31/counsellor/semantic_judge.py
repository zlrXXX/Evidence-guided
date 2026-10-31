from __future__ import annotations
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence
MODEL = 'gpt-5.6-luna'
REASONING_EFFORT = 'high'
TOKEN_SCHEDULE = (2600, 8000, 10000)
INPUT_USD_PER_MILLION = 0.2
OUTPUT_USD_PER_MILLION = 1.2

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()

def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(dict(value)) + '\n')
        handle.flush()

def response_text(body: Mapping[str, Any]) -> str:
    return ''.join((str(part.get('text', '')) for item in body.get('output', []) for part in item.get('content', []) if part.get('type') == 'output_text'))

def incomplete_reason(body: Mapping[str, Any]) -> str | None:
    details = body.get('incomplete_details')
    return str(details.get('reason')) if isinstance(details, Mapping) and details.get('reason') else None
ROLE_PROMPT = 'You are a source-hidden evaluator of synthetic CBT counsellor trajectories.\nJudge only the supplied visible causal context, sampled legal decision, official CBT step,\nand candidate counsellor utterance. Do not infer hidden targets or reward agreement,\noptimism, verbosity, symptom reduction, technique completion, or session ending.\n\nFor each candidate, assign integer scores from 0 to 4:\n- framework: how faithfully the utterance executes the sampled decision and official step,\n  respects the current resistance/progress state, preserves continuity, and stays within\n  the supplied information. A locally illegal decision cannot score above 0.\n- empathy: specific, proportionate, non-assumptive acknowledgement/support grounded in\n  what the patient actually expressed. Do not reward invented feelings or generic praise.\n- naturalness: believable counsellor speech with ordinary conversational wording, a\n  manageable focus, and natural turn-taking. Penalize parroting, stacked questions,\n  dense lists, repeated praise, premature conclusions, unintroduced jargon, or a manual tone.\n\nAlso mark hard_failure=true only for an unsafe or role-breaking output: instructions to\nignore the simulation, diagnosis as fact, clinical-effectiveness promises, fabricated\nhigh-risk facts, metadata/JSON/hidden reasoning, or content that contradicts the committed\ndecision. Evidence strings are optional support: use an exact non-empty span of the\ncandidate when possible, otherwise return an empty string. Evidence exactness never changes\nan otherwise valid numeric score. These are synthetic-proxy annotations, not clinical or\nhuman ground truth.'

def output_schema(candidate_ids: Sequence[str]) -> dict[str, Any]:
    score = {'type': 'object', 'additionalProperties': False, 'properties': {'candidate_id': {'type': 'string', 'enum': list(candidate_ids)}, 'framework_score': {'type': 'integer', 'minimum': 0, 'maximum': 4}, 'framework_evidence': {'type': 'string'}, 'empathy_score': {'type': 'integer', 'minimum': 0, 'maximum': 4}, 'empathy_evidence': {'type': 'string'}, 'naturalness_score': {'type': 'integer', 'minimum': 0, 'maximum': 4}, 'naturalness_evidence': {'type': 'string'}, 'hard_failure': {'type': 'boolean'}, 'hard_failure_reason': {'type': 'string'}}, 'required': ['candidate_id', 'framework_score', 'framework_evidence', 'empathy_score', 'empathy_evidence', 'naturalness_score', 'naturalness_evidence', 'hard_failure', 'hard_failure_reason']}
    return {'type': 'object', 'additionalProperties': False, 'properties': {'scores': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': score}}, 'required': ['scores']}

class OnlineCounsellorJudge:

    def __init__(self, output_dir: Path, max_cost_usd: float, workers: int=4) -> None:
        if not os.environ.get('OPENAI_API_KEY', '').strip():
            raise RuntimeError('OPENAI_API_KEY is required for counsellor semantic rewards')
        if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
            raise ValueError('max_cost_usd must be positive')
        if workers < 1 or workers > 16:
            raise ValueError('workers must be in [1,16]')
        self.max_cost_usd = float(max_cost_usd)
        self.workers = int(workers)
        self.contract_sha256 = sha256_bytes(ROLE_PROMPT.encode('utf-8'))
        self.cache_path = output_dir / 'semantic_score_cache.jsonl'
        self.attempt_path = output_dir / 'semantic_api_attempts.jsonl'
        self._cache: dict[str, dict[str, Any]] = {}
        self._spent = 0.0
        self._reserved = 0.0
        self._attempts = 0
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._load()

    def _load(self) -> None:
        if self.attempt_path.is_file():
            for line in self.attempt_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._spent += float(row.get('estimated_cost_usd', 0.0))
                    self._attempts += 1
        if self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('contract_sha256') != self.contract_sha256:
                    raise RuntimeError('semantic cache belongs to another judge contract')
                previous = self._cache.get(row['cache_key'])
                if previous is not None and previous != row:
                    raise RuntimeError('conflicting semantic cache rows')
                self._cache[row['cache_key']] = row
        if self._spent > self.max_cost_usd:
            raise RuntimeError('recorded Luna cost already exceeds this run cap')

    def _client(self):
        client = getattr(self._thread_local, 'client', None)
        if client is None:
            from openai import OpenAI
            client = OpenAI(max_retries=0, timeout=300.0)
            self._thread_local.client = client
        return client

    @staticmethod
    def _cost(usage: Mapping[str, Any]) -> float:
        return (float(usage.get('input_tokens', 0)) * INPUT_USD_PER_MILLION + float(usage.get('output_tokens', 0)) * OUTPUT_USD_PER_MILLION) / 1000000.0

    def _body(self, case: Mapping[str, Any], candidate_ids: Sequence[str], tokens: int) -> dict[str, Any]:
        return {'model': MODEL, 'input': [{'role': 'developer', 'content': ROLE_PROMPT}, {'role': 'user', 'content': 'EVALUATION_CASE=\n' + canonical(case)}], 'reasoning': {'effort': REASONING_EFFORT}, 'store': False, 'max_output_tokens': tokens, 'text': {'format': {'type': 'json_schema', 'name': 'counsellor_semantic_scores', 'strict': True, 'schema': output_schema(candidate_ids)}}}

    def _reserve(self, body: Mapping[str, Any]) -> float:
        bound = (len(TOKEN_SCHEDULE) * len(canonical(body).encode('utf-8')) * INPUT_USD_PER_MILLION + sum(TOKEN_SCHEDULE) * OUTPUT_USD_PER_MILLION) / 1000000.0
        with self._lock:
            if self._spent + self._reserved + bound > self.max_cost_usd:
                raise RuntimeError(f'Luna scoring would exceed ${self.max_cost_usd:.2f}; spent=${self._spent:.4f}, reserved=${self._reserved:.4f}')
            self._reserved += bound
        return bound

    def _record_attempt(self, key: str, request: Mapping[str, Any], body: Mapping[str, Any], index: int) -> None:
        usage = body.get('usage') or {}
        cost = self._cost(usage)
        row = {'cache_key': key, 'attempt_index': index, 'model': MODEL, 'resolved_model': body.get('model'), 'status': body.get('status'), 'incomplete_reason': incomplete_reason(body), 'max_output_tokens': request['max_output_tokens'], 'usage': usage, 'estimated_cost_usd': cost}
        with self._lock:
            append_jsonl(self.attempt_path, row)
            self._spent += cost
            self._attempts += 1

    def _validate(self, body: Mapping[str, Any], candidate_ids: Sequence[str], utterances: Sequence[str]) -> list[dict[str, Any]]:
        resolved = body.get('model')
        if not isinstance(resolved, str) or not resolved.casefold().startswith(MODEL):
            raise ValueError('semantic response did not resolve to selected Luna')
        parsed = json.loads(response_text(body))
        scores = parsed.get('scores')
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError('judge must return four scores')
        by_id = {item.get('candidate_id'): item for item in scores if isinstance(item, Mapping)}
        if set(by_id) != set(candidate_ids) or len(by_id) != 4:
            raise ValueError('candidate IDs were lost or duplicated')
        output: list[dict[str, Any]] = []
        for candidate_id, utterance in zip(candidate_ids, utterances):
            item = by_id[candidate_id]
            checked = {'candidate_id': candidate_id}
            for dimension in ('framework', 'empathy', 'naturalness'):
                score = item.get(f'{dimension}_score')
                evidence = item.get(f'{dimension}_evidence')
                if type(score) is not int or score not in range(5) or (not isinstance(evidence, str)):
                    raise ValueError(f'invalid {dimension} score')
                present = bool(evidence.strip())
                exact = present and evidence in utterance
                checked[f'{dimension}_score'] = score
                checked[f'{dimension}_reward'] = (score - 2) / 2.0
                checked[f'{dimension}_evidence'] = evidence
                checked[f'{dimension}_evidence_status'] = 'EXACT_RESPONSE_SPAN' if exact else 'NON_EXACT_OPTIONAL_SUPPORT' if present else 'NOT_PROVIDED'
            hard = item.get('hard_failure')
            reason = item.get('hard_failure_reason')
            if type(hard) is not bool or not isinstance(reason, str):
                raise ValueError('invalid hard-failure fields')
            checked['hard_failure'] = hard
            checked['hard_failure_reason'] = reason
            output.append(checked)
        return output

    def _score_one(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        key = prepared['cache_key']
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        standard = self._body(prepared['case'], prepared['candidate_ids'], TOKEN_SCHEDULE[0])
        reservation = self._reserve(standard)
        try:
            final: dict[str, Any] | None = None
            attempts = 0
            for index, limit in enumerate(TOKEN_SCHEDULE):
                request = self._body(prepared['case'], prepared['candidate_ids'], limit)
                response = self._client().responses.create(**request)
                body = response.model_dump(mode='json')
                attempts += 1
                self._record_attempt(key, request, body, index)
                if body.get('status') == 'completed':
                    final = body
                    break
                if body.get('status') != 'incomplete' or incomplete_reason(body) != 'max_output_tokens':
                    raise RuntimeError(f'Luna judge failed: status={body.get('status')} reason={incomplete_reason(body)}')
            if final is None:
                raise RuntimeError('Luna judge did not complete')
            scores = self._validate(final, prepared['candidate_ids'], prepared['utterances'])
            row = {'schema_version': 'certify-cbt-counsellor-semantic-cache-v1', 'cache_key': key, 'phase': prepared['phase'], 'prompt_id': prepared['prompt_id'], 'prompt_position': prepared['prompt_position'], 'contract_sha256': self.contract_sha256, 'model': MODEL, 'reasoning_effort': REASONING_EFFORT, 'attempts_used': attempts, 'utterance_sha256': [sha256_bytes(value.encode('utf-8')) for value in prepared['utterances']], 'scores': scores}
            with self._lock:
                append_jsonl(self.cache_path, row)
                self._cache[key] = row
            return row
        finally:
            with self._lock:
                self._reserved -= reservation

    def _prepare(self, items: Sequence[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
        if not items or len(items) > self.workers:
            raise ValueError('semantic step size must be in [1,workers]')
        prepared: list[dict[str, Any]] = []
        for item in items:
            utterances = list(item['utterances'])
            if len(utterances) != 4:
                raise ValueError('each RLOO group requires four candidates')
            order = sorted(range(4), key=lambda index: sha256_bytes(f'CRL5-JUDGE|{item['row']['prompt_id']}|{index}'.encode('utf-8')))
            candidate_ids = [''] * 4
            candidates: list[dict[str, Any]] = []
            for display, original in enumerate(order, 1):
                candidate_id = f'candidate_{display}'
                candidate_ids[original] = candidate_id
                candidates.append({'candidate_id': candidate_id, 'sampled_decision': item['decisions'][original], 'local_legality': item['local_legality'][original], 'counsellor_utterance': utterances[original]})
            visible = copy.deepcopy(item['row']['counsellor_visible'])
            case = {'simulation_scope': 'single-session CBT research simulation', 'visible_context': visible, 'candidates': candidates}
            key = sha256_bytes(canonical({'contract': self.contract_sha256, 'phase': phase, 'position': item['prompt_position'], 'prompt_id': item['row']['prompt_id'], 'case': case}).encode('utf-8'))
            prepared.append({'cache_key': key, 'phase': phase, 'prompt_id': item['row']['prompt_id'], 'prompt_position': int(item['prompt_position']), 'case': case, 'candidate_ids': candidate_ids, 'utterances': utterances})
        return prepared

    def submit_step(self, items: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
        prepared = self._prepare(items, phase)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(prepared)))
        return {'pool': pool, 'futures': [pool.submit(self._score_one, item) for item in prepared], 'consumed': False}

    @staticmethod
    def finish_step(job: dict[str, Any]) -> list[list[dict[str, Any]]]:
        if job.get('consumed') is not False:
            raise RuntimeError('semantic job already consumed')
        job['consumed'] = True
        try:
            rows = [future.result() for future in job['futures']]
        finally:
            job['pool'].shutdown(wait=True)
        return [row['scores'] for row in rows]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {'model': MODEL, 'reasoning_effort': REASONING_EFFORT, 'parallel_workers': self.workers, 'recorded_attempts': self._attempts, 'cached_groups': len(self._cache), 'estimated_recorded_cost_usd': self._spent, 'maximum_cost_usd': self.max_cost_usd, 'contract_sha256': self.contract_sha256}
