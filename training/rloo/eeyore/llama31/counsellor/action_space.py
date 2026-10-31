from __future__ import annotations
import json
from typing import Any, Mapping, Sequence
ACTION_KEY_ORDER = ('action', 'delivery_mode', 'retrieve_or_reuse', 'technique_id', 'step_id', 'session_control')
ACTION_KEYS = frozenset(ACTION_KEY_ORDER)
ADVANCEMENT_STATES = frozenset({'GUIDED_APPLICATION', 'INDEPENDENT_APPLICATION'})
CBT_PROGRESS_STATES = frozenset({'NO_DEMONSTRATION', 'ECHOED_COUNSELLOR_DEPENDENT', 'GUIDED_APPLICATION', 'INDEPENDENT_APPLICATION'})
DELIVERY_MODES = ('WITHOUT_GUIDANCE', 'WITH_GUIDANCE', 'ADDRESS_RESISTANCE')
SESSION_CONTROLS = ('CONTINUE_SESSION', 'END_SESSION')

def action_text(action: Mapping[str, str]) -> str:
    return json.dumps({key: action[key] for key in ACTION_KEY_ORDER}, ensure_ascii=False, separators=(',', ':'))

def legal_token_successors(prefix: Sequence[int], candidates: Sequence[Sequence[int]], pad_token_id: int) -> list[int]:
    prefix_values = [int(value) for value in prefix]
    allowed = sorted({int(candidate[len(prefix_values)]) for candidate in candidates if len(candidate) > len(prefix_values) and list(candidate[:len(prefix_values)]) == prefix_values})
    if allowed:
        return allowed
    if any((len(prefix_values) >= len(candidate) and prefix_values[:len(candidate)] == list(candidate) and all((value == pad_token_id for value in prefix_values[len(candidate):])) for candidate in candidates)):
        return [pad_token_id]
    raise RuntimeError('constrained legal action trie reached an invalid prefix')

def _action(name: str, delivery_mode: str, retrieve_or_reuse: str, technique_id: str, step_id: str, session_control: str) -> dict[str, str]:
    return {'action': name, 'delivery_mode': delivery_mode, 'retrieve_or_reuse': retrieve_or_reuse, 'technique_id': technique_id, 'step_id': step_id, 'session_control': session_control}

def active_step_progress_state(visible: Mapping[str, Any]) -> str:
    active = visible.get('active_cbt_rag')
    if active is None:
        return 'NO_ACTIVE_TECHNIQUE'
    key = f'{active['technique_id']}:{active['current_step']['step_id']}'
    progress_map = visible.get('cbt_progress_by_step')
    if not isinstance(progress_map, Mapping):
        raise ValueError('persistent CBT progress map is missing')
    state = str(progress_map.get(key, 'NO_DEMONSTRATION'))
    if state not in CBT_PROGRESS_STATES:
        raise ValueError('active-step persistent CBT progress state is invalid')
    return state

def action_prompt_context(visible: Mapping[str, Any], candidate_texts: list[str]) -> dict[str, Any]:
    context = dict(visible)
    context.pop('current_cbt_progress_state', None)
    context['active_step_progress_state_before_decision'] = active_step_progress_state(visible)
    context['legal_action_candidates'] = list(candidate_texts)
    return context

def legal_action_candidates(visible: Mapping[str, Any], rag: Mapping[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    active = visible.get('active_cbt_rag')
    option_ids = sorted({str(item['technique_id']) for item in visible['cbt_rag_options']})
    if active is None:
        for technique_id in option_ids:
            if technique_id not in rag:
                raise ValueError('retrieved option is absent from official CBT RAG')
            candidates.append(_action('START_TECHNIQUE', 'NOT_APPLICABLE', 'RETRIEVE_NEW_TECHNIQUE', technique_id, str(rag[technique_id]['entry']['first_step_id']), 'CONTINUE_SESSION'))
    else:
        technique_id = str(active['technique_id'])
        current_step_id = str(active['current_step']['step_id'])
        if technique_id not in rag or current_step_id not in rag[technique_id]['steps']:
            raise ValueError('active technique or step is absent from official CBT RAG')
        official_step = rag[technique_id]['steps'][current_step_id]
        next_step_id = official_step.get('next_step_id_or_none')
        progress = active_step_progress_state(visible)
        for session_control in SESSION_CONTROLS:
            candidates.append(_action('CHECK_PATIENT_EVIDENCE', 'NOT_APPLICABLE', 'REUSE_ACTIVE_TECHNIQUE', technique_id, current_step_id, session_control))
            for delivery_mode in DELIVERY_MODES:
                candidates.append(_action('CONTINUE_TECHNIQUE', delivery_mode, 'REUSE_ACTIVE_TECHNIQUE', technique_id, current_step_id, session_control))
            if next_step_id is not None and progress in ADVANCEMENT_STATES:
                for delivery_mode in DELIVERY_MODES:
                    candidates.append(_action('CONTINUE_TECHNIQUE', delivery_mode, 'REUSE_ACTIVE_TECHNIQUE', technique_id, str(next_step_id), session_control))
            if next_step_id is None and progress == 'INDEPENDENT_APPLICATION':
                candidates.append(_action('COMPLETE_TECHNIQUE', 'NOT_APPLICABLE', 'REUSE_ACTIVE_TECHNIQUE', technique_id, current_step_id, session_control))
        for option_id in option_ids:
            if option_id == technique_id:
                continue
            if option_id not in rag:
                raise ValueError('retrieved option is absent from official CBT RAG')
            candidates.append(_action('SWITCH_TECHNIQUE', 'NOT_APPLICABLE', 'RETRIEVE_NEW_TECHNIQUE', option_id, str(rag[option_id]['entry']['first_step_id']), 'CONTINUE_SESSION'))
    texts = [action_text(candidate) for candidate in candidates]
    if not candidates:
        raise ValueError('current causal state has no legal counsellor action')
    if len(texts) != len(set(texts)):
        raise ValueError('legal counsellor action candidates are not unique')
    return candidates

def validate_action(action: Mapping[str, str], visible: Mapping[str, Any], rag: Mapping[str, Any]) -> None:
    if dict(action) not in legal_action_candidates(visible, rag):
        raise ValueError('action is not in the deterministic legal action space')
