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
STANDARD_MAX_OUTPUT_TOKENS = 2600
MAX_OUTPUT_TOKEN_SCHEDULE = (STANDARD_MAX_OUTPUT_TOKENS, 8000, 10000)
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
    if isinstance(details, Mapping):
        reason = details.get('reason')
        return str(reason) if reason is not None else None
    return None

class OnlineSemanticJudge:
    """Score one four-response group per prompt with a frozen Luna-high judge."""

    def __init__(self, bundle_root: Path, output_dir: Path, max_cost_usd: float, workers: int=4) -> None:
        if not math.isfinite(max_cost_usd) or max_cost_usd <= 0.0:
            raise ValueError('the Luna judge needs one positive total USD cap')
        if workers < 1 or workers > 16:
            raise ValueError('semantic API workers must be between 1 and 16')
        if not os.environ.get('OPENAI_API_KEY', '').strip():
            raise RuntimeError('OPENAI_API_KEY is required for online semantic rewards')
        judge_root = bundle_root / 'judge'
        self.prompt_contract_path = judge_root / 'prompt_contract.json'
        self.selection_path = judge_root / 'selected_judge.json'
        self.prompt_contract = json.loads(self.prompt_contract_path.read_text(encoding='utf-8'))
        self.selection = json.loads(self.selection_path.read_text(encoding='utf-8'))
        self.prompt_contract_sha256 = sha256_bytes(canonical(self.prompt_contract).encode('utf-8'))
        if self.selection.get('selected_model') != MODEL or self.selection.get('selected_reasoning_effort') != REASONING_EFFORT or self.selection.get('prompt_contract_sha256') != self.prompt_contract_sha256:
            raise RuntimeError('the frozen semantic-judge selection contract changed')
        fusion = self.prompt_contract.get('semantic_to_sft_alignment_fusion', {})
        if fusion != {'profile_and_action': {'llm_weight': 0.6, 'sft_alignment_weight': 0.4}, 'naturalness': {'llm_weight': 0.8, 'sft_alignment_weight': 0.2}}:
            raise RuntimeError('the frozen semantic-to-SFT reward fusion changed')
        self.role_prompt = str(self.prompt_contract['role_and_task_prompt'])
        self.output_schema = self.prompt_contract['output_schema']
        dimensions = self.prompt_contract.get('semantic_dimensions', ['profile', 'action'])
        if not isinstance(dimensions, list) or not all((isinstance(dimension, str) for dimension in dimensions)) or len(dimensions) != len(set(dimensions)) or (not {'profile', 'action'}.issubset(dimensions)) or (not set(dimensions).issubset({'profile', 'action', 'naturalness'})):
            raise RuntimeError('invalid semantic reward dimensions in the frozen contract')
        self.dimensions = tuple(dimensions)
        self.max_cost_usd = float(max_cost_usd)
        self.workers = int(workers)
        self.cache_path = output_dir / 'semantic_score_cache.jsonl'
        self.attempt_path = output_dir / 'semantic_api_attempts.jsonl'
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._reserved_usd = 0.0
        self._cache: dict[str, dict[str, Any]] = {}
        self._spent_usd = 0.0
        self._attempts = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if self.attempt_path.is_file():
            with self.attempt_path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self._spent_usd += float(row.get('estimated_cost_usd', 0.0))
                    self._attempts += 1
        if self.cache_path.is_file():
            with self.cache_path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get('model') != MODEL or row.get('reasoning_effort') != REASONING_EFFORT or row.get('prompt_contract_sha256') != self.prompt_contract_sha256:
                        raise RuntimeError('semantic score cache belongs to another judge contract')
                    key = row['cache_key']
                    previous = self._cache.get(key)
                    if previous is not None and previous != row:
                        raise RuntimeError('conflicting semantic scores exist for one frozen request')
                    self._cache[key] = row
        if self._spent_usd > self.max_cost_usd:
            raise RuntimeError('existing semantic API usage already exceeds the configured cap')

    def _client(self):
        client = getattr(self._thread_local, 'client', None)
        if client is None:
            from openai import OpenAI
            client = OpenAI(max_retries=0, timeout=300.0)
            self._thread_local.client = client
        return client

    def _case(self, row: Mapping[str, Any], responses: Sequence[str], phase: str, prompt_position: int) -> tuple[dict[str, Any], list[str]]:
        if len(responses) != 4 or any((not response.strip() for response in responses)):
            raise ValueError('every semantic request requires four non-empty patient responses')
        visible = copy.deepcopy(row['patient_visible'])
        visible['current_resistance_state'] = row['runtime_assigned_resistance_state']
        response_order = sorted(range(4), key=lambda index: sha256_bytes(f'PRL-JUDGE-V2|{phase}|{prompt_position}|{row['prompt_id']}|{index}'.encode('utf-8')))
        candidate_ids = [''] * 4
        presented_candidates: list[dict[str, str]] = []
        for display_position, response_index in enumerate(response_order, start=1):
            candidate_id = f'candidate_{display_position}'
            candidate_ids[response_index] = candidate_id
            presented_candidates.append({'candidate_id': candidate_id, 'patient_response': responses[response_index]})
        reward_context = row['reward_context']
        case = {'simulation_scope': 'single-session CBT dialogue; research simulation only', 'patient_context': {'patient_profile': visible['patient_profile'], 'counselling_progress': visible.get('counselling_progress', {}), 'symptom_level': visible.get('symptom_level', 'NOT_PROVIDED'), 'current_resistance_state': visible['current_resistance_state'], 'dialogue_history': visible['dialogue_history']}, 'current_counsellor_turn': {'selected_action': reward_context['selected_counsellor_action'], 'utterance': visible['current_counsellor_utterance']}, 'candidate_responses': presented_candidates}
        return (case, candidate_ids)

    def _body(self, case: Mapping[str, Any], max_output_tokens: int) -> dict[str, Any]:
        candidate_ids = [str(item['candidate_id']) for item in case['candidate_responses']]
        if len(candidate_ids) != 4 or len(set(candidate_ids)) != 4:
            raise ValueError('semantic request must contain four unique candidate IDs')
        schema = copy.deepcopy(self.output_schema)
        schema['properties']['scores']['items']['properties']['candidate_id']['enum'] = candidate_ids
        return {'model': MODEL, 'input': [{'role': 'developer', 'content': self.role_prompt}, {'role': 'user', 'content': 'EVALUATION_CASE=\n' + json.dumps(case, ensure_ascii=False, sort_keys=True)}], 'reasoning': {'effort': REASONING_EFFORT}, 'store': False, 'max_output_tokens': max_output_tokens, 'text': {'format': {'type': 'json_schema', 'name': 'patient_semantic_scores', 'strict': True, 'schema': schema}}}

    @staticmethod
    def _cost(usage: Mapping[str, Any]) -> float:
        return (float(usage.get('input_tokens', 0)) * INPUT_USD_PER_MILLION + float(usage.get('output_tokens', 0)) * OUTPUT_USD_PER_MILLION) / 1000000.0

    def _reserve_bound(self, body: Mapping[str, Any]) -> float:
        input_bound = len(canonical(body).encode('utf-8'))
        return (len(MAX_OUTPUT_TOKEN_SCHEDULE) * input_bound * INPUT_USD_PER_MILLION + sum(MAX_OUTPUT_TOKEN_SCHEDULE) * OUTPUT_USD_PER_MILLION) / 1000000.0

    def _reserve(self, amount: float) -> None:
        with self._lock:
            if self._spent_usd + self._reserved_usd + amount > self.max_cost_usd:
                raise RuntimeError(f'Luna semantic scoring would exceed the ${self.max_cost_usd:.2f} cap; recorded=${self._spent_usd:.4f}, reserved=${self._reserved_usd:.4f}')
            self._reserved_usd += amount

    def _release(self, amount: float) -> None:
        with self._lock:
            self._reserved_usd -= amount

    def _record_attempt(self, cache_key: str, body: Mapping[str, Any], response_body: Mapping[str, Any], attempt_index: int) -> None:
        usage = response_body.get('usage') or {}
        cost = self._cost(usage)
        row = {'schema_version': 'certify-cbt-prl-semantic-api-attempt-v2', 'cache_key': cache_key, 'attempt_index': attempt_index, 'model': MODEL, 'resolved_model': response_body.get('model'), 'reasoning_effort': REASONING_EFFORT, 'max_output_tokens': body['max_output_tokens'], 'status': response_body.get('status'), 'incomplete_reason': incomplete_reason(response_body), 'response_id': response_body.get('id'), 'usage': usage, 'estimated_cost_usd': cost}
        with self._lock:
            append_jsonl(self.attempt_path, row)
            self._spent_usd += cost
            self._attempts += 1

    def _validate_scores(self, body: Mapping[str, Any], candidate_ids: Sequence[str], responses: Sequence[str]) -> list[dict[str, Any]]:
        resolved = body.get('model')
        if not isinstance(resolved, str) or not resolved.casefold().startswith(MODEL):
            raise ValueError('semantic score did not resolve to the selected Luna model')
        parsed = json.loads(response_text(body))
        items = parsed.get('scores')
        if not isinstance(items, list) or len(items) != 4:
            raise ValueError('semantic judge did not return four score objects')
        by_id = {item.get('candidate_id'): item for item in items if isinstance(item, Mapping)}
        if len(by_id) != 4 or set(by_id) != set(candidate_ids):
            raise ValueError('semantic score candidate IDs are missing, duplicated, or changed')
        output: list[dict[str, Any]] = []
        for candidate_id, patient_response in zip(candidate_ids, responses):
            item = by_id[candidate_id]
            checked: dict[str, Any] = {'candidate_id': candidate_id}
            for dimension in self.dimensions:
                score = item.get(f'{dimension}_score')
                evidence = item.get(f'{dimension}_evidence')
                if type(score) is not int or score not in range(5) or (not isinstance(evidence, str)):
                    raise ValueError(f'invalid {dimension} semantic score or evidence')
                evidence_present = bool(evidence.strip())
                evidence_exact = evidence_present and evidence in patient_response
                checked[f'{dimension}_reported_score'] = score
                checked[f'{dimension}_score'] = score
                checked[f'{dimension}_evidence'] = evidence
                checked[f'{dimension}_evidence_present'] = evidence_present
                checked[f'{dimension}_evidence_exact'] = evidence_exact
                checked[f'{dimension}_evidence_support_status'] = 'EXACT_RESPONSE_SPAN' if evidence_exact else 'NON_EXACT_OPTIONAL_SUPPORT' if evidence_present else 'NOT_PROVIDED'
                checked[f'{dimension}_semantic_reward'] = (score - 2) / 2.0
            output.append(checked)
        return output

    def _score_one(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        cache_key = prepared['cache_key']
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        standard_body = prepared['standard_body']
        reservation = self._reserve_bound(standard_body)
        self._reserve(reservation)
        try:
            final_body: dict[str, Any] | None = None
            attempts_used = 0
            request = standard_body
            for attempt_index, token_limit in enumerate(MAX_OUTPUT_TOKEN_SCHEDULE):
                if int(request['max_output_tokens']) != token_limit:
                    raise RuntimeError('semantic retry token schedule changed unexpectedly')
                response = self._client().responses.create(**request)
                body = response.model_dump(mode='json')
                attempts_used += 1
                self._record_attempt(cache_key, request, body, attempt_index)
                if body.get('status') == 'completed':
                    final_body = body
                    break
                if attempt_index + 1 < len(MAX_OUTPUT_TOKEN_SCHEDULE) and body.get('status') == 'incomplete' and (incomplete_reason(body) == 'max_output_tokens'):
                    request = self._body(prepared['case'], MAX_OUTPUT_TOKEN_SCHEDULE[attempt_index + 1])
                    continue
                raise RuntimeError(f'Luna semantic request failed: status={body.get('status')}, reason={incomplete_reason(body)}')
            if final_body is None:
                raise RuntimeError('Luna semantic request did not produce a completed response')
            scores = self._validate_scores(final_body, prepared['candidate_ids'], prepared['responses'])
            row = {'schema_version': 'certify-cbt-prl-semantic-score-cache-v4', 'cache_key': cache_key, 'phase': prepared['phase'], 'prompt_position': prepared['prompt_position'], 'prompt_id': prepared['prompt_id'], 'response_sha256': prepared['response_sha256'], 'request_body_sha256': sha256_bytes(canonical(standard_body).encode('utf-8')), 'prompt_contract_sha256': self.prompt_contract_sha256, 'model': MODEL, 'reasoning_effort': REASONING_EFFORT, 'attempts_used': attempts_used, 'used_expanded_token_retry': attempts_used > 1, 'max_output_token_attempt_schedule': list(MAX_OUTPUT_TOKEN_SCHEDULE[:attempts_used]), 'scores': scores}
            with self._lock:
                append_jsonl(self.cache_path, row)
                self._cache[cache_key] = row
            return row
        finally:
            self._release(reservation)

    def _prepare_step(self, items: Sequence[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
        if not items or len(items) > self.workers:
            raise ValueError('one semantic step must contain one to workers prompt groups')
        prepared: list[dict[str, Any]] = []
        for item in items:
            row = item['row']
            responses = list(item['responses'])
            position = int(item['prompt_position'])
            case, candidate_ids = self._case(row, responses, phase, position)
            standard_body = self._body(case, STANDARD_MAX_OUTPUT_TOKENS)
            response_hashes = [sha256_bytes(response.encode('utf-8')) for response in responses]
            cache_key = sha256_bytes(canonical({'prompt_contract_sha256': self.prompt_contract_sha256, 'phase': phase, 'prompt_position': position, 'prompt_id': row['prompt_id'], 'case': case}).encode('utf-8'))
            prepared.append({'cache_key': cache_key, 'phase': phase, 'prompt_position': position, 'prompt_id': row['prompt_id'], 'case': case, 'candidate_ids': candidate_ids, 'responses': responses, 'response_sha256': response_hashes, 'standard_body': standard_body})
        return prepared

    def submit_step(self, items: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
        prepared = self._prepare_step(items, phase)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(prepared)))
        return {'pool': pool, 'futures': [pool.submit(self._score_one, item) for item in prepared], 'consumed': False}

    @staticmethod
    def finish_step(job: dict[str, Any]) -> list[list[dict[str, Any]]]:
        if job.get('consumed') is not False:
            raise RuntimeError('semantic step was already collected')
        job['consumed'] = True
        pool = job['pool']
        try:
            rows = [future.result() for future in job['futures']]
        finally:
            pool.shutdown(wait=True)
        return [row['scores'] for row in rows]

    def score_step(self, items: Sequence[Mapping[str, Any]], phase: str) -> list[list[dict[str, Any]]]:
        return self.finish_step(self.submit_step(items, phase))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {'model': MODEL, 'reasoning_effort': REASONING_EFFORT, 'standard_max_output_tokens': STANDARD_MAX_OUTPUT_TOKENS, 'max_output_token_retry_schedule': list(MAX_OUTPUT_TOKEN_SCHEDULE[1:]), 'maximum_attempts_per_uncached_group': len(MAX_OUTPUT_TOKEN_SCHEDULE), 'parallel_workers': self.workers, 'recorded_api_attempts': self._attempts, 'cached_prompt_groups': len(self._cache), 'estimated_recorded_cost_usd': self._spent_usd, 'maximum_cost_usd': self.max_cost_usd, 'prompt_contract_sha256': self.prompt_contract_sha256}
