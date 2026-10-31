from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from reward_math import expected_resistance_utility
RESISTANCE_STATES = ['REJECTING_DISENGAGED', 'AMBIVALENT', 'COOPERATIVE', 'COLLABORATIVE_EXPLORATION']
RESISTANCE_DIRECTIONS = ['WORSE', 'STAY', 'BETTER']
LEGAL_DIRECTIONS = {'REJECTING_DISENGAGED': ['STAY', 'BETTER'], 'AMBIVALENT': ['WORSE', 'STAY', 'BETTER'], 'COOPERATIVE': ['WORSE', 'STAY', 'BETTER'], 'COLLABORATIVE_EXPLORATION': ['WORSE', 'STAY']}
DIRECTION_DELTA = {'WORSE': -1, 'STAY': 0, 'BETTER': 1}
CBT_STATES = ['NO_DEMONSTRATION', 'ECHOED_COUNSELLOR_DEPENDENT', 'GUIDED_APPLICATION', 'INDEPENDENT_APPLICATION']
CBT_RANK = {state: index for index, state in enumerate(CBT_STATES)}
from action_space import ACTION_KEYS, active_step_progress_state, legal_action_candidates, validate_action
EXPECTED_RESISTANCE_SHA256 = '135BEAD553FACF6D4922D9F03EF97707DAA0A426C41077415CDED80FAF30A0F7'
EXPECTED_EVIDENCE_SHA256 = 'D683F9298F5801D29869328949F934F5614B46688FC2CE1B7DE6C8B465AC799A'
EXPECTED_EVIDENCE_SOURCE_BASE_CONFIG_SHA256 = '649F6A1EC33C6BDD9A6486D5C66019D461139E54957073EAFE9BBC2D34C75B0B'
EXPECTED_EVIDENCE_PACKAGED_BASE_CONFIG_SHA256 = 'D62F646A17F0C097CAB99D561EA61DC21E6386887A4DD56227BA303A76A0AA30'
EXPECTED_EVIDENCE_BASE_CONFIG_SEMANTIC_SHA256 = 'A811F3E25660A59B54EB686129818DAE7CED0C279CF006ED9C976486A989CF2C'
EXPECTED_RESISTANCE_TEMPERATURE = 1.846444010734558
EXPECTED_EVIDENCE_TEMPERATURE = 1.3515501022338867
EXPECTED_EVIDENCE_TRACKER_CONFIG_SHA256 = 'AD29A8AE9A63C0BB9591779EFC4CC9F84A76D2972E2B94A37425E678E438D5C1'
EXPECTED_EVIDENCE_TEMPERATURE_SHA256 = '50D02776AA31655A3F5E9091A8207154B3F224D341DB61585BB6D51289FF6098'
EXPECTED_EVIDENCE_TOKENIZER_SHA256 = {'added_tokens.json': 'DC046D04C9B0ADA7AE6F1DC89C465801799ACDF0C9A6AAB8C15A1B2D5CA4E91F', 'special_tokens_map.json': 'B2F1B2F15F29A6B6D9D6EA4ECA1675D2C231A71477F151D48F79CC83A625BA21', 'spm.model': 'C679FBF93643D19AAB7EE10C0B99E460BDBC02FEDF34B92B05AF343B4AF586FD', 'tokenizer.json': '0A7EFAD4DA0E8014F186D51A46D90FD10A75914DE79C7D5CAD6B82F658609DB9', 'tokenizer_config.json': 'E6C4C771911C211618A2D46488DC7E9E499F051773CD4FF4FAECA6FE55FDF569'}
PROGRESS_POLICY = {'schema_version': 'certify-cbt-progress-proxy-calibration-v1', 'status': 'FROZEN_SIMULATION_PROXY_THRESHOLDS', 'adjacent': {'enabled': True, 'threshold': 0.40264633297920227}, 'jump': {'enabled': True, 'threshold': 0.5608643293380737}, 'confidence_is_gate_not_reward_multiplier': True, 'exact_nonfallback_patient_span_required': True}

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def load_rag(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != 15:
        raise ValueError('official CBT RAG must contain 15 techniques')
    output: dict[str, Any] = {}
    for entry in rows:
        steps = {step['step_id']: step for step in entry['procedure_steps']}
        output[entry['technique_id']] = {'entry': entry, 'steps': steps}
    if sum((len(item['steps']) for item in output.values())) != 60:
        raise ValueError('official CBT RAG must contain 60 steps')
    return output

def parse_action(text: str) -> dict[str, str]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError('action must be exactly one JSON object without surrounding text') from error
    if not isinstance(value, dict) or set(value) != ACTION_KEYS:
        raise ValueError('action keys differ from the six-field contract')
    if any((not isinstance(value[key], str) or not value[key] for key in ACTION_KEYS)):
        raise ValueError('action values must be non-empty strings')
    return value

def selected_rag(action: Mapping[str, str], rag: Mapping[str, Any]) -> dict[str, Any]:
    entry = rag[action['technique_id']]['entry']
    return {'technique_id': entry['technique_id'], 'technique_name': entry['technique_name'], 'when_to_use': entry['when_to_use'], 'prerequisites': entry['prerequisites'], 'completion_condition': entry['completion_condition'], 'stop_or_switch_condition': entry['stop_or_switch_condition'], 'current_step': copy.deepcopy(rag[action['technique_id']]['steps'][action['step_id']])}

def response_context(visible: Mapping[str, Any], action: Mapping[str, Any], rag: Mapping[str, Any]) -> dict[str, Any]:
    context = {key: copy.deepcopy(value) for key, value in visible.items() if key not in {'cbt_rag_options', 'current_provisional_resistance_state', 'current_cbt_progress_state'}}
    context['active_step_progress_state_before_decision'] = active_step_progress_state(visible)
    context['natural_engagement_cue'] = COUNSELLOR_ENGAGEMENT_CUES[visible['current_provisional_resistance_state']]
    context['selected_controller_decision'] = copy.deepcopy(action)
    context['selected_official_cbt_step'] = selected_rag(action, rag)
    if action['action'] == 'SWITCH_TECHNIQUE':
        context['response_phase'] = 'SWITCH_INTRODUCTION'
    elif action['session_control'] == 'END_SESSION':
        context['response_phase'] = 'END_PROPOSAL'
    else:
        context['response_phase'] = 'IN_SESSION_RESPONSE'
    return context
FIXED_OPENING = 'Hello, and thank you for being here. To begin, could you share what has been troubling you recently and what has been going through your mind?'
COUNSELLOR_ENGAGEMENT_CUES = {'REJECTING_DISENGAGED': 'The patient is guarded, doubtful, and only minimally willing to engage; do not force agreement.', 'AMBIVALENT': 'The patient is willing to engage in part but remains uncertain and may hold reservations.', 'COOPERATIVE': 'The patient is willing to work with the counsellor while remaining realistic rather than automatically agreeing.', 'COLLABORATIVE_EXPLORATION': 'The patient is actively exploring and contributing while remaining capable of uncertainty or correction.'}
ENGAGEMENT_CUES = {'REJECTING_DISENGAGED': 'The patient is currently strongly guarded or rejecting; respond with limited engagement unless the visible dialogue itself supports a small shift.', 'AMBIVALENT': 'The patient is currently mixed or uncertain; express both hesitation and some willingness where supported by the visible dialogue.', 'COOPERATIVE': 'The patient is currently willing to engage with the counsellor while retaining realistic uncertainty or difficulty.', 'COLLABORATIVE_EXPLORATION': 'The patient is currently exploring the issue collaboratively while still speaking from their own experience and without assumed independent mastery.'}

def patient_context(visible: Mapping[str, Any], counsellor_utterance: str, next_resistance_state: str) -> dict[str, Any]:
    turns = visible['dialogue_history']
    if not turns or len(turns) % 2 != 0:
        raise ValueError('EEYORE causal history must contain complete counsellor-patient exchanges')
    dialogue = []
    for index in range(0, len(turns), 2):
        counsellor_turn, patient_turn = (turns[index], turns[index + 1])
        if counsellor_turn.get('role') != 'counsellor' or patient_turn.get('role') != 'patient':
            raise ValueError('EEYORE causal turn order is invalid')
        dialogue.append({'counsellor': str(counsellor_turn['utterance']), 'patient': str(patient_turn['utterance'])})
    return {'patient_profile': copy.deepcopy(visible['patient_profile']), 'counselling_progress': copy.deepcopy(visible['supported_session_information']), 'current_resistance_state': next_resistance_state, 'dialogue_history': dialogue, 'current_counsellor_utterance': counsellor_utterance, 'response_phase': 'IN_SESSION_RESPONSE'}

def alternating_history(dialogue: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [{'role': str(turn['role']), 'content': str(turn['utterance'])} for turn in dialogue[-4:]]

def resistance_input(row: Mapping[str, Any]) -> str:
    history = '\n'.join((f'{turn['role']}: {turn['content']}' for turn in alternating_history(row['dialogue_history'])))
    return f'Current resistance: {row['current_resistance_state']}\nActive CBT technique: {row['active_technique_id']}\nActive CBT step: {row['active_step_id']}\nCounsellor: {row['counsellor_utterance']}\nPatient profile: {canonical(row['patient_profile'])}\nCounselling progress: {canonical(row['counselling_progress'])}\nDialogue context:\n{history}'

class FrozenResistanceTracker:

    def __init__(self, path: Path):
        if sha256_file(path / 'tracker_state.pt') != EXPECTED_RESISTANCE_SHA256:
            raise ValueError('wrong selected Resistance State Tracker')
        config = read_json(path / 'tracker_config.json')
        if config['labels'] != RESISTANCE_DIRECTIONS or config['resistance_states'] != RESISTANCE_STATES or config.get('seed') != 20260818 or (config.get('task') != 'resistance') or (config.get('prediction_target') != 'transition_direction_conditioned_on_current_resistance_state'):
            raise ValueError('Resistance State Tracker label order mismatch')
        base_config = AutoConfig.from_pretrained(path / 'base_model_config.json', local_files_only=True)

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_config(base_config)
                self.dropout = nn.Dropout(0.1)
                self.state_head = nn.Linear(base_config.hidden_size, len(RESISTANCE_DIRECTIONS))
                legal = torch.zeros(len(RESISTANCE_STATES), len(RESISTANCE_DIRECTIONS), dtype=torch.bool)
                for state_index, state in enumerate(RESISTANCE_STATES):
                    for direction in LEGAL_DIRECTIONS[state]:
                        legal[state_index, RESISTANCE_DIRECTIONS.index(direction)] = True
                self.register_buffer('legal_transition_mask', legal)

            def forward(self, input_ids, attention_mask, previous_state_id, token_type_ids=None):
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
                logits = self.state_head(self.dropout(hidden[:, 0]))
                return logits.masked_fill(~self.legal_transition_mask[previous_state_id], -10000.0)
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer', use_fast=True)
        self.model = Model()
        self.model.load_state_dict(torch.load(path / 'tracker_state.pt', map_location='cpu', weights_only=True), strict=True)
        self.model.to('cuda').eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_length = int(config['max_length'])
        self.temperature = float(read_json(path / 'temperature.json')['temperature'])
        if self.temperature != EXPECTED_RESISTANCE_TEMPERATURE:
            raise ValueError('Resistance State Tracker calibration temperature mismatch')

    @torch.inference_mode()
    def predict(self, rows: Sequence[Mapping[str, Any]], batch_size: int=64) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = self.tokenizer([resistance_input(row) for row in batch], truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
            encoded = {key: value.cuda() for key, value in encoded.items()}
            previous = torch.tensor([RESISTANCE_STATES.index(row['current_resistance_state']) for row in batch], dtype=torch.long, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = self.model(previous_state_id=previous, **encoded)
            probabilities = (logits.float() / self.temperature).softmax(-1).cpu()
            for row, probs in zip(batch, probabilities):
                direction = RESISTANCE_DIRECTIONS[int(probs.argmax())]
                current = row['current_resistance_state']
                next_state = RESISTANCE_STATES[RESISTANCE_STATES.index(current) + DIRECTION_DELTA[direction]]
                probability_map = {label: float(probs[index]) for index, label in enumerate(RESISTANCE_DIRECTIONS)}
                expected_delta = sum((float(probs[index]) * DIRECTION_DELTA[label] for index, label in enumerate(RESISTANCE_DIRECTIONS)))
                output.append({'direction': direction, 'next_state': next_state, 'expected_direction_reward': expected_delta, 'expected_state_aware_utility': expected_resistance_utility(current, probability_map), 'probabilities': probability_map})
        return output

def evidence_text(technique_id: str, step_id: str, requirement: str, counsellor: str, patient: str) -> tuple[str, int]:
    prefix = f'Technique: {technique_id}\nStep: {step_id}\nRequired patient operation: {requirement}\nCounsellor: {counsellor}\nPatient: '
    return (prefix + patient, len(prefix))

def select_span(text: str, response: str, response_start: int, offsets: list[list[int]], probabilities: list[float], attention: list[int]) -> tuple[dict[str, Any] | None, bool]:
    candidates = [(index, float(probability)) for index, ((start, end), probability, attended) in enumerate(zip(offsets, probabilities, attention)) if attended and end > start and (end > response_start) and (start < len(text))]
    if not candidates:
        return (None, True)
    selected = [index for index, probability in candidates if probability >= 0.5]
    fallback = not selected
    if not selected:
        selected = [max(candidates, key=lambda value: value[1])[0]]
    groups: list[list[int]] = []
    for index in selected:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    by_index = dict(candidates)
    group = max(groups, key=lambda values: (sum((by_index[index] for index in values)), len(values)))
    start = max(0, offsets[group[0]][0] - response_start)
    end = min(len(response), offsets[group[-1]][1] - response_start)
    return ({'text': response[start:end], 'start': start, 'end': end}, fallback) if response[start:end] else (None, True)

class FrozenEvidenceTracker:

    def __init__(self, path: Path, packaged_base_config: Path):
        if sha256_file(path / 'tracker_state.pt') != EXPECTED_EVIDENCE_SHA256:
            raise ValueError('wrong selected CBT Evidence Tracker')
        if sha256_file(path / 'tracker_config.json') != EXPECTED_EVIDENCE_TRACKER_CONFIG_SHA256:
            raise ValueError('CBT Evidence Tracker config hash mismatch')
        if sha256_file(path / 'temperature.json') != EXPECTED_EVIDENCE_TEMPERATURE_SHA256:
            raise ValueError('CBT Evidence Tracker temperature artifact hash mismatch')
        for relative, expected in EXPECTED_EVIDENCE_TOKENIZER_SHA256.items():
            if sha256_file(path / 'tokenizer' / relative) != expected:
                raise ValueError(f'CBT Evidence Tracker tokenizer artifact mismatch: {relative}')
        config = read_json(path / 'tracker_config.json')
        if config['labels'] != CBT_STATES:
            raise ValueError('CBT Evidence Tracker label order mismatch')
        tracker_base_config = path / 'base_model_config.json'
        base_config_path = tracker_base_config if tracker_base_config.is_file() else packaged_base_config
        if not base_config_path.is_file():
            raise FileNotFoundError('CBT Evidence Tracker architecture config is absent from both the tracker export and the sealed RLOO bundle')
        expected_base_hash = EXPECTED_EVIDENCE_SOURCE_BASE_CONFIG_SHA256 if base_config_path == tracker_base_config else EXPECTED_EVIDENCE_PACKAGED_BASE_CONFIG_SHA256
        if sha256_file(base_config_path) != expected_base_hash:
            raise ValueError('CBT Evidence Tracker architecture config hash mismatch')
        base_payload = read_json(base_config_path)
        if hashlib.sha256(canonical(base_payload).encode('utf-8')).hexdigest().upper() != EXPECTED_EVIDENCE_BASE_CONFIG_SEMANTIC_SHA256:
            raise ValueError('CBT Evidence Tracker architecture config is not semantically identical to the selected source')
        model_type = base_payload.pop('model_type')
        base_config = AutoConfig.for_model(model_type, **base_payload)

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_config(base_config)
                self.dropout = nn.Dropout(0.1)
                self.state_head = nn.Linear(base_config.hidden_size, len(CBT_STATES))
                self.span_head = nn.Linear(base_config.hidden_size, 1)

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
                return (self.state_head(self.dropout(hidden[:, 0])), self.span_head(self.dropout(hidden)).squeeze(-1))
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer', use_fast=True)
        self.model = Model()
        self.model.load_state_dict(torch.load(path / 'tracker_state.pt', map_location='cpu', weights_only=True), strict=True)
        self.model.to('cuda').eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.temperature = float(read_json(path / 'temperature.json')['temperature'])
        if self.temperature != EXPECTED_EVIDENCE_TEMPERATURE:
            raise ValueError('CBT Evidence Tracker calibration temperature mismatch')
        self.max_length = int(config['max_length'])

    @torch.inference_mode()
    def predict(self, rows: Sequence[Mapping[str, Any]], batch_size: int=32) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prepared = [evidence_text(row['technique_id'], row['step_id'], row['evidence_requirement'], row['counsellor_utterance'], row['patient_response']) for row in batch]
            encoded = self.tokenizer([value[0] for value in prepared], truncation=True, padding=True, max_length=self.max_length, return_offsets_mapping=True, return_tensors='pt')
            offsets = encoded.pop('offset_mapping')
            attention = encoded['attention_mask'].clone()
            model_inputs = {key: value.cuda() for key, value in encoded.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state_logits, span_logits = self.model(**model_inputs)
            state_probs = (state_logits.float() / self.temperature).softmax(-1).cpu()
            span_probs = span_logits.float().sigmoid().cpu()
            for index, (row, (text, patient_start)) in enumerate(zip(batch, prepared)):
                probs = state_probs[index]
                label_index = int(probs.argmax())
                span, fallback = select_span(text, row['patient_response'], patient_start, offsets[index].tolist(), span_probs[index].tolist(), attention[index].tolist())
                output.append({'evidence_state': CBT_STATES[label_index], 'confidence': float(probs[label_index]), 'evidence_span': span, 'span_was_fallback': fallback})
        return output

def progress_reward(progress_before: Mapping[str, str], action: Mapping[str, str], evidence: Mapping[str, Any]) -> dict[str, Any]:
    before = copy.deepcopy(dict(progress_before))
    after = copy.deepcopy(before)
    key = f'{action['technique_id']}:{action['step_id']}'
    before_state = before.get(key, CBT_STATES[0])
    observed = str(evidence['evidence_state'])
    distance = CBT_RANK[observed] - CBT_RANK[before_state]
    exact = evidence['evidence_span'] is not None and (not evidence['span_was_fallback'])
    if distance <= 0:
        accepted, reason = (False, 'STAY_AT_HIGHEST_CERTIFIED_STATE')
    else:
        gate = PROGRESS_POLICY['adjacent'] if distance == 1 else PROGRESS_POLICY['jump']
        accepted = bool(gate['enabled']) and exact and (float(evidence['confidence']) >= float(gate['threshold']))
        reason = 'ADJACENT_CERTIFIED' if accepted and distance == 1 else 'JUMP_CERTIFIED' if accepted else 'ADVANCE_REJECTED_BY_CERTIFICATE'
    if accepted:
        after[key] = observed
    potential_before = sum((CBT_RANK[state] / 3.0 for state in before.values()))
    potential_after = sum((CBT_RANK[state] / 3.0 for state in after.values()))
    reward = potential_after - potential_before
    if reward < -1e-12 or reward > 1.0 + 1e-12:
        raise RuntimeError('progress reward violates monotonic bounded contract')
    return {'step_key': key, 'stored_state_before': before_state, 'observed_evidence_state': observed, 'certificate_accepted': accepted, 'transition_reason': reason, 'progress_map_before': before, 'progress_map_after': after, 'cbt_progress_reward': reward}
