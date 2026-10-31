from __future__ import annotations
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
IMPLEMENTATION_VERSION = 'CERTIFY_CBT_PATIENT_RLOO_V5_2202_USER_PROMPT_RATE_LIMITED_WEIGHTS_V4'
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
EXPECTED_BASE_MODEL_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
BUNDLE_ROOT = Path(os.environ['PRL5_BUNDLE_ROOT'])
sys.path.insert(0, str(BUNDLE_ROOT))
from checkpoint_binding import bind_sft
from prompt_runtime import patient_context as shown_patient_context, patient_system as select_patient_system
from semantic_judge import MAX_OUTPUT_TOKEN_SCHEDULE, STANDARD_MAX_OUTPUT_TOKENS, OnlineSemanticJudge
OUTPUT_ROOT = Path('Path to corresponding file')
PATIENT_SFT_ROOT = Path('Path to corresponding file')
RESISTANCE_TRACKER_ROOT = Path(os.environ['CERTIFY_CBT_RESISTANCE_TRACKER_DIR'])
EXPECTED_PATIENT_SFT_SHA256 = None
EXPECTED_COUNSELLOR_SFT_SHA256 = '1B7FEA0448569BF2EA94E9DCCF1DD908E06A4415BBA6A28C08603D55E47D82D0'
EXPECTED_EXPRESSION_SHA256 = 'AE4BAC0D7DA7B5F57B8B793DA83D090A681111D0B5345D2CA8F9A22FE46661F6'
SEED = 20260830
NUM_GENERATIONS = 4
TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-06
WEIGHT_DECAY = 0.0
KL_BETA = 0.05
GRADIENT_ACCUMULATION_PROMPTS = 4
WARMUP_RATIO = 0.05
MAX_PROMPT_TOKENS = 5632
MAX_NEW_TOKENS = 160
MIN_NEW_TOKENS = 8
GENERATION_TEMPERATURE = 0.8
TRAIN_GENERATION_TOP_P = 1.0
DEVELOPMENT_GENERATION_TOP_P = 0.9
TRAIN_GENERATION_TOP_K = 0
DEVELOPMENT_GENERATION_TOP_K = 50
MAX_GRAD_NORM = 1.0
SAVE_EVERY_PROMPTS = 252
DEV_PROMPTS = 64
VALIDATE_EVERY_STEPS = 250
PROFILE_ACTION_LLM_WEIGHT = 0.6
PROFILE_ACTION_SFT_WEIGHT = 0.4
NATURALNESS_LLM_WEIGHT = 0.8
NATURALNESS_SFT_WEIGHT = 0.2
SEMANTIC_API_WORKERS = int(os.environ.get('PRL5_LUNA_WORKERS', '4'))
TRAIN_PROMPTS = 3000
DEVELOPMENT_PROMPTS = 500
PATIENT_VISIBLE_KEYS = {'patient_profile', 'symptom_level', 'counselling_progress', 'current_resistance_state', 'dialogue_history', 'current_counsellor_utterance'}
INITIAL_REWARD_WEIGHTS = {'profile_consistency': 0.3, 'counsellor_action_dependence': 0.3, 'naturalness': 0.35, 'intended_resistance_realization': 0.05}
REWARD_WEIGHT_UPDATE_PROMPTS = 64
REWARD_WEIGHT_UPDATE_RESPONSES = REWARD_WEIGHT_UPDATE_PROMPTS * NUM_GENERATIONS
FAIR_STABLE_STEP_SIZE = 0.1
MAX_REWARD_WEIGHT_DELTA_PER_UPDATE = 0.002
REWARD_WEIGHT_BOUNDS = {'profile_consistency': (0.27, 0.33), 'counsellor_action_dependence': (0.27, 0.33), 'naturalness': (0.32, 0.38), 'intended_resistance_realization': (0.03, 0.07)}
FORBIDDEN_REWARDS = {'CBT_PROGRESS', 'COUNSELLING_QUALITY', 'SYMPTOM_REDUCTION', 'SHARED_THERAPY_SUCCESS'}
PATIENT_SYSTEM = "You are a client attending counselling. Use the preceding dialogue, patient profile, symptom level, and provisional engagement cue. Output only the next natural patient utterance. Respond to the counsellor's latest utterance while maintaining continuity with the preceding counselling history. Stay profile-consistent and do not invent facts. Express the supplied engagement cue through openness, uncertainty, or disagreement without naming a state. Use natural first-person speech. When the counsellor asks more than one related question, weave the relevant parts into one natural response rather than replying as a numbered or question-by-question list. When supported, include one relevant thought, feeling, worry, bodily reaction, example, preference, uncertainty, or barrier; do not force detail. You may hesitate, correct yourself, disagree, or answer partly when appropriate. If an answered question is repeated, do not invent novelty or simply agree; clarify, correct, identify a barrier, or say there is nothing more to add. Do not claim that elapsed time passed, an external task was completed, or another person responded within one generated turn; keep such agreed tasks untried. Avoid copying the counsellor, repeated replies, profile summaries, canned therapy language, sudden insight or compliance, tidy conclusions, and clinical explanations. Session control is selected separately, so do not output an ending token or control label. Do not output speaker or state labels, JSON, metadata, stage directions, hidden reasoning, clinical claims, or evaluator commentary."
RESISTANCE_STATES = ['REJECTING_DISENGAGED', 'AMBIVALENT', 'COOPERATIVE', 'COLLABORATIVE_EXPLORATION']
RESISTANCE_DIRECTIONS = ['WORSE', 'STAY', 'BETTER']
LEGAL_DIRECTIONS = {'REJECTING_DISENGAGED': ['STAY', 'BETTER'], 'AMBIVALENT': ['WORSE', 'STAY', 'BETTER'], 'COOPERATIVE': ['WORSE', 'STAY', 'BETTER'], 'COLLABORATIVE_EXPLORATION': ['WORSE', 'STAY']}
DIRECTION_DELTA = {'WORSE': -1, 'STAY': 0, 'BETTER': 1}

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def select_development_rows(development_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_indices = sorted(range(len(development_rows)), key=lambda index: sha256_bytes(f'{SEED}|{development_rows[index]['prompt_id']}'.encode('utf-8')))[:DEV_PROMPTS]
    return [development_rows[index] for index in expected_indices]

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(dict(payload)) + '\n')

def show_training_progress(completed_steps: int, total_steps: int, completed_prompts: int, total_prompts: int, mean_reward: float, llm_cost_usd: float) -> None:
    if total_steps <= 0 or not 0 <= completed_steps <= total_steps:
        raise ValueError('invalid patient-RLOO progress counters')
    width = 30
    filled = round(width * completed_steps / total_steps)
    bar = '=' * filled + '-' * (width - filled)
    percent = 100.0 * completed_steps / total_steps
    line = f'\r[patient RLOO] [{bar}] {completed_steps}/{total_steps} steps ({percent:5.1f}%; {completed_prompts}/{total_prompts} prompts) reward={mean_reward:.4f} cost=${llm_cost_usd:.4f}'
    print(line, end='\n' if completed_steps == total_steps else '', flush=True)

def verify_bundle() -> dict[str, Any]:
    manifest_path = BUNDLE_ROOT / 'bundle_manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('status') != 'LOCAL_STRUCTURE_VALIDATED_GPU_AND_API_NOT_RUN':
        raise ValueError('unexpected bundle status')
    expected_manifest_sha = os.environ.get('PRL5_BUNDLE_MANIFEST_SHA256', '').strip()
    if not expected_manifest_sha or sha256_file(manifest_path) != expected_manifest_sha:
        raise RuntimeError('the trainer was not launched from the verified PRL5 bundle')
    for relative, metadata in manifest['members'].items():
        path = BUNDLE_ROOT / relative
        if not path.is_file() or path.stat().st_size != metadata['bytes'] or sha256_file(path) != metadata['sha256']:
            raise ValueError(f'verified bundle extraction is incomplete: {relative}')
    if manifest['counts'] != {'train': TRAIN_PROMPTS, 'development': DEVELOPMENT_PROMPTS, 'test': 0}:
        raise ValueError('unexpected prompt split counts')
    return manifest

def locate_patient_adapter() -> Path:
    global EXPECTED_PATIENT_SFT_SHA256
    expected = read_json(BUNDLE_ROOT / 'sft_checkpoint_contracts.json')['patient']
    adapter, EXPECTED_PATIENT_SFT_SHA256 = bind_sft(PATIENT_SFT_ROOT / 'best_adapter', expected)
    return adapter

def validate_hardware() -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 1024 ** 3
    if 'A100' not in props.name or gib < 75:
        raise RuntimeError(f'This full cell requires an A100 80GB-class GPU; found {props.name} with {gib:.1f} GiB')
    return {'name': props.name, 'total_memory_gib': gib, 'bf16': True}

def disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

def next_state(current: str, direction: str) -> str:
    if direction not in LEGAL_DIRECTIONS[current]:
        raise ValueError(f'illegal resistance direction {direction} from {current}')
    return RESISTANCE_STATES[RESISTANCE_STATES.index(current) + DIRECTION_DELTA[direction]]

def alternating_history(dialogue: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for exchange in dialogue:
        turns.append({'role': 'counsellor', 'content': str(exchange['counsellor'])})
        turns.append({'role': 'patient', 'content': str(exchange['patient'])})
    return turns[-4:]

def resistance_input(row: Mapping[str, Any]) -> str:
    visible = row['patient_visible']
    history = '\n'.join((f'{turn['role']}: {turn['content']}' for turn in alternating_history(visible['dialogue_history'])))
    profile = json.dumps(visible['patient_profile'], ensure_ascii=False, sort_keys=True)
    progress = json.dumps(visible['counselling_progress'], ensure_ascii=False, sort_keys=True)
    decision = row['reward_context']['selected_counsellor_action']
    return f'Current resistance: {row['previous_resistance_state']}\nActive CBT technique: {decision.get('technique_id')}\nActive CBT step: {decision.get('step_id')}\nCounsellor: {visible['current_counsellor_utterance']}\nPatient profile: {profile}\nCounselling progress: {progress}\nDialogue context:\n{history}'

class FrozenResistanceTracker:

    def __init__(self, checkpoint_dir: Path):
        state_path = checkpoint_dir / 'tracker_state.pt'
        required = [state_path, checkpoint_dir / 'tracker_config.json', checkpoint_dir / 'base_model_config.json', checkpoint_dir / 'temperature.json', checkpoint_dir / 'tokenizer' / 'tokenizer.json']
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'set CERTIFY_CBT_RESISTANCE_TRACKER_DIR to the completed selected tracker; missing {missing}')
        self.checkpoint_sha256 = sha256_file(state_path)
        config = read_json(checkpoint_dir / 'tracker_config.json')
        base_config = AutoConfig.from_pretrained(checkpoint_dir / 'base_model_config.json', local_files_only=True)

        class Model(nn.Module):

            def __init__(self) -> None:
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
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / 'tokenizer', use_fast=True)
        self.model = Model()
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to('cuda').eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_length = int(config['max_length'])
        self.temperature = float(read_json(checkpoint_dir / 'temperature.json')['temperature'])

    @torch.inference_mode()
    def resolve(self, rows: list[dict[str, Any]], batch_size: int=64) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = self.tokenizer([resistance_input(row) for row in batch], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            encoded = {key: value.cuda() for key, value in encoded.items()}
            previous = torch.tensor([RESISTANCE_STATES.index(row['previous_resistance_state']) for row in batch], dtype=torch.long, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = self.model(previous_state_id=previous, **encoded)
            probabilities = (logits.float() / self.temperature).softmax(-1).cpu()
            for row, probs in zip(batch, probabilities):
                direction = RESISTANCE_DIRECTIONS[int(probs.argmax())]
                assigned = next_state(row['previous_resistance_state'], direction)
                copied = copy.deepcopy(row)
                copied['runtime_transition_direction'] = direction
                copied['runtime_assigned_resistance_state'] = assigned
                copied['patient_visible']['current_resistance_state'] = assigned
                copied['resistance_tracker_audit'] = {'direction_probabilities': {label: float(probs[index]) for index, label in enumerate(RESISTANCE_DIRECTIONS)}, 'source_proxy_state_agreement': assigned == row['source_proxy_assigned_state_audit_only']}
                resolved.append(copied)
        return resolved

class FrozenExpressionChecker:

    def __init__(self, checkpoint_dir: Path):
        model_dir = checkpoint_dir / 'model'
        if sha256_file(model_dir / 'model.safetensors') != EXPECTED_EXPRESSION_SHA256:
            raise ValueError('wrong Resistance Expression Checker checkpoint')
        contract = read_json(checkpoint_dir / 'rl_inference_contract.json')
        if contract['model_input'] != ['patient_response']:
            raise ValueError('expression checker is not response-only')
        self.labels = list(contract['labels_in_order'])
        if self.labels != RESISTANCE_STATES:
            raise ValueError('expression-checker state order mismatch')
        self.temperature = float(contract['temperature'])
        self.threshold = float(contract['consistency_threshold']['threshold'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to('cuda').eval()
        id2label = {int(index): label for index, label in self.model.config.id2label.items()}
        if [id2label.get(index) for index in range(4)] != self.labels:
            raise ValueError('expression-checker model labels do not match its frozen contract')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def scores(self, responses: Sequence[str], assigned_state: str) -> list[dict[str, Any]]:
        encoded = self.tokenizer(list(responses), truncation=True, padding='max_length', max_length=192, return_tensors='pt')
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(**encoded).logits.float() / self.temperature
        probabilities = logits.softmax(-1)
        assigned_id = self.labels.index(assigned_state)
        output: list[dict[str, Any]] = []
        for probs in probabilities:
            assigned = float(probs[assigned_id])
            others = torch.cat([probs[:assigned_id], probs[assigned_id + 1:]])
            margin = assigned - float(others.max())
            output.append({'margin': margin, 'assigned_probability': assigned, 'predicted_state': self.labels[int(probs.argmax())], 'consistent_audit_only': margin >= self.threshold})
        return output

def patient_prompt(tokenizer, visible: Mapping[str, Any]) -> tuple[str, int]:
    shown = shown_patient_context(visible, rloo=True)
    prompts = read_json(BUNDLE_ROOT / 'patient_prompts.json')
    system = select_patient_system(prompts, shown['response_phase'])
    user = 'VISIBLE_PATIENT_RUNTIME_CONTEXT=\n' + json.dumps(shown, ensure_ascii=False, sort_keys=True)
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    if length > MAX_PROMPT_TOKENS:
        raise ValueError(f'Patient prompt requires {length} tokens; no field is truncated')
    return (text, length)

def ablated_visible(visible: Mapping[str, Any], kind: str) -> dict[str, Any]:
    output = copy.deepcopy(visible)
    if kind == 'profile':
        output['patient_profile'] = {'presenting_situation': 'UNKNOWN', 'cognitive_distortions': [], 'demographics': {'age_group': 'UNKNOWN', 'gender': 'UNKNOWN', 'occupation': 'UNKNOWN', 'marital_status': 'UNKNOWN'}}
    elif kind == 'action':
        output['current_counsellor_utterance'] = ''
    else:
        raise ValueError(kind)
    return output

def reward_text_completion_ids(tokenizer, response: str) -> list[int]:
    """Reproduce the frozen text-based NLL reward representation.

    This representation is used only for the calibrated naturalness/profile/
    action rewards.  It is never used for the RLOO policy gradient or KL term.
    """
    ids = tokenizer(response, add_special_tokens=False)['input_ids']
    if not ids:
        ids = [tokenizer.eos_token_id]
    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    return ids[:MAX_NEW_TOKENS + 1]

def exact_sampled_completion_ids(generated_sequence: Sequence[int], prompt_length: int, eos_token_id: int) -> list[int]:
    """Return the exact sampled suffix, including its first sampled EOS."""
    suffix = [int(token_id) for token_id in generated_sequence[prompt_length:]]
    if eos_token_id in suffix[:MIN_NEW_TOKENS]:
        raise ValueError('sampled EOS violates the configured minimum completion length')
    if eos_token_id in suffix:
        suffix = suffix[:suffix.index(eos_token_id) + 1]
    if not suffix:
        raise ValueError('patient generation produced no sampled completion tokens')
    if len(suffix) > MAX_NEW_TOKENS:
        raise ValueError('sampled patient completion exceeds MAX_NEW_TOKENS')
    return suffix

def conditional_logps(model: nn.Module, tokenizer, prompt_texts: Sequence[str], completion_token_ids: Sequence[Sequence[int]], require_grad: bool, sampling_temperature: float | None=None, sampling_min_new_tokens: int=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Score supplied token trajectories without decoding or re-tokenizing them.

    The first output is the ordinary model log-probability used by the frozen
    text-NLL reward.  When ``sampling_temperature`` is supplied, the third
    output is the log-probability under the same temperature-scaled distribution
    used to sample the RLOO trajectory.
    """
    if len(prompt_texts) != len(completion_token_ids):
        raise ValueError('prompt/completion-token batch mismatch')
    if sampling_temperature is not None and (not math.isfinite(sampling_temperature) or sampling_temperature <= 0.0):
        raise ValueError('sampling_temperature must be positive and finite')
    if type(sampling_min_new_tokens) is not int or sampling_min_new_tokens < 0:
        raise ValueError('sampling_min_new_tokens must be a non-negative integer')
    if sampling_temperature is None and sampling_min_new_tokens:
        raise ValueError('sampling_min_new_tokens requires a sampling temperature')
    sequences: list[list[int]] = []
    masks: list[list[bool]] = []
    for prompt, target_values in zip(prompt_texts, completion_token_ids):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target = list(target_values)
        if not target or len(target) > MAX_NEW_TOKENS + 1 or any((type(token_id) is not int or token_id < 0 for token_id in target)):
            raise ValueError('invalid supplied completion-token trajectory')
        sequence = prompt_ids + target
        if len(sequence) > MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 1:
            raise ValueError('combined sequence exceeds the frozen token limit')
        sequences.append(sequence)
        masks.append([False] * (len(prompt_ids) - 1) + [True] * len(target))
    maximum = max(map(len, sequences))
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([sequence + [pad] * (maximum - len(sequence)) for sequence in sequences], dtype=torch.long, device='cuda')
    attention = torch.tensor([[1] * len(sequence) + [0] * (maximum - len(sequence)) for sequence in sequences], dtype=torch.long, device='cuda')
    token_mask = torch.tensor([mask + [False] * (maximum - 1 - len(mask)) for mask in masks], dtype=torch.bool, device='cuda')
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
        labels = input_ids[:, 1:]
        raw_token_logps = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        raw_token_logps = raw_token_logps.masked_fill(~token_mask, 0.0)
        raw_sequence_logps = raw_token_logps.sum(-1)
        sampling_sequence_logps = None
        if sampling_temperature is not None:
            sampling_logits = logits / sampling_temperature
            if sampling_min_new_tokens:
                completion_positions = token_mask.long().cumsum(-1)
                suppress_eos = token_mask & (completion_positions <= sampling_min_new_tokens)
                sampling_logits = sampling_logits.clone()
                sampling_logits[..., tokenizer.eos_token_id] = sampling_logits[..., tokenizer.eos_token_id].masked_fill(suppress_eos, float('-inf'))
            sampling_token_logps = sampling_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            sampling_token_logps = sampling_token_logps.masked_fill(~token_mask, 0.0)
            sampling_sequence_logps = sampling_token_logps.sum(-1)
    lengths = token_mask.sum(-1).clamp_min(1)
    return (raw_sequence_logps, lengths, sampling_sequence_logps)

def generate_responses(model, tokenizer, prompt_text: str, seed: int, top_p: float, top_k: int) -> list[dict[str, Any]]:
    if not 0.0 < top_p <= 1.0:
        raise ValueError('top_p must be in (0, 1]')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a non-negative integer')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt')
    encoded = {key: value.cuda() for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        generated = model.generate(**encoded, do_sample=True, temperature=GENERATION_TEMPERATURE, top_p=top_p, top_k=top_k, num_return_sequences=NUM_GENERATIONS, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    prompt_length = encoded['input_ids'].shape[1]
    samples: list[dict[str, Any]] = []
    for sequence in generated:
        token_ids = exact_sampled_completion_ids(sequence.tolist(), prompt_length, tokenizer.eos_token_id)
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        samples.append({'text': text, 'completion_token_ids': token_ids})
    if any((not sample['text'] for sample in samples)):
        raise ValueError('patient generation produced an empty response')
    return samples
AI_PATTERNS = re.compile('\\b(as an ai|language model|i cannot provide therapy|consult a professional|state label|metadata)\\b', re.IGNORECASE)
THERAPIST_PATTERNS = re.compile('\\b(as your therapist|my client|i recommend that you|you should tell me)\\b', re.IGNORECASE)

def explicit_profile_contradiction(profile: Mapping[str, Any], response: str) -> bool:
    demographics = profile.get('demographics', {})
    text = response.lower()
    marital = str(demographics.get('marital_status', 'UNKNOWN')).lower()
    direct = {'married': ['\\bi am single\\b', "\\bi'm single\\b", '\\bi am unmarried\\b'], 'single': ['\\bi am married\\b', "\\bi'm married\\b"]}
    if marital in direct and any((re.search(pattern, text) for pattern in direct[marital])):
        return True
    age_group = str(demographics.get('age_group', 'UNKNOWN'))
    match = re.search("\\bi(?: am|'m) (\\d{1,3}) years old\\b", text)
    if match and '~' in age_group:
        low, high = map(int, age_group.split('~'))
        if not low <= int(match.group(1)) <= high:
            return True
    return False

def deterministic_penalty(profile: Mapping[str, Any], response: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    words = response.split()
    if AI_PATTERNS.search(response):
        reasons.append('AI_OR_METADATA_STYLE')
    if THERAPIST_PATTERNS.search(response):
        reasons.append('ROLE_REVERSAL')
    if explicit_profile_contradiction(profile, response):
        reasons.append('EXPLICIT_PROFILE_CONTRADICTION')
    if len(words) < 6 or response.lower().strip(' .!?') in {"i don't know", 'not sure', 'okay'}:
        reasons.append('GENERIC_OR_EMPTY')
    unique_ratio = len(set((word.lower() for word in words))) / max(1, len(words))
    if len(words) >= 16 and unique_ratio < 0.45:
        reasons.append('REPETITIVE')
    return (min(0.6, 0.2 * len(reasons)), reasons)

@torch.inference_mode()
def local_reward_components(reference, tokenizer, checker: FrozenExpressionChecker, row: Mapping[str, Any], prompt_text: str, responses: Sequence[str], sampled_completion_token_ids: Sequence[Sequence[int]]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    visible = row['patient_visible']
    profile_prompt, _ = patient_prompt(tokenizer, ablated_visible(visible, 'profile'))
    action_prompt, _ = patient_prompt(tokenizer, ablated_visible(visible, 'action'))
    reward_completion_token_ids = [reward_text_completion_ids(tokenizer, response) for response in responses]
    full_logp, lengths, _ = conditional_logps(reference, tokenizer, [prompt_text] * len(responses), reward_completion_token_ids, require_grad=False)
    profile_logp, _, _ = conditional_logps(reference, tokenizer, [profile_prompt] * len(responses), reward_completion_token_ids, require_grad=False)
    action_logp, _, _ = conditional_logps(reference, tokenizer, [action_prompt] * len(responses), reward_completion_token_ids, require_grad=False)
    _, _, reference_sampling_logps = conditional_logps(reference, tokenizer, [prompt_text] * len(responses), sampled_completion_token_ids, require_grad=False, sampling_temperature=GENERATION_TEMPERATURE, sampling_min_new_tokens=MIN_NEW_TOKENS)
    if reference_sampling_logps is None:
        raise RuntimeError('reference sampling log-probabilities were not calculated')
    full_nll = -full_logp / lengths
    profile_nll = -profile_logp / lengths
    action_nll = -action_logp / lengths
    natural_raw = -full_nll
    profile_nll_score = ((profile_nll - full_nll) / 0.5).tanh()
    action_nll_score = ((action_nll - full_nll) / 0.5).tanh()
    expression = checker.scores(responses, row['runtime_assigned_resistance_state'])
    records: list[dict[str, Any]] = []
    for index, response in enumerate(responses):
        penalty, penalty_reasons = deterministic_penalty(visible['patient_profile'], response)
        records.append({'raw_components': {'naturalness': float(natural_raw[index]), 'intended_resistance_realization': float(expression[index]['margin'])}, 'nll_components': {'profile_consistency': float(profile_nll_score[index]), 'counsellor_action_dependence': float(action_nll_score[index])}, 'deterministic_penalty': penalty, 'penalty_reasons': penalty_reasons, 'reference_mean_nll': float(full_nll[index]), 'expression_audit': expression[index]})
    return (records, reference_sampling_logps.detach())

def add_semantic_reward_components(local_records: Sequence[Mapping[str, Any]], semantic_scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(local_records) != len(semantic_scores):
        raise ValueError('semantic score/response batch mismatch')
    records: list[dict[str, Any]] = []
    for local_record, semantic in zip(local_records, semantic_scores):
        record = copy.deepcopy(dict(local_record))
        nll = record['nll_components']
        profile_semantic = float(semantic['profile_semantic_reward'])
        action_semantic = float(semantic['action_semantic_reward'])
        record['raw_components']['profile_consistency'] = PROFILE_ACTION_SFT_WEIGHT * float(nll['profile_consistency']) + PROFILE_ACTION_LLM_WEIGHT * profile_semantic
        record['raw_components']['counsellor_action_dependence'] = PROFILE_ACTION_SFT_WEIGHT * float(nll['counsellor_action_dependence']) + PROFILE_ACTION_LLM_WEIGHT * action_semantic
        record['semantic_components'] = {'profile_consistency': profile_semantic, 'counsellor_action_dependence': action_semantic}
        if 'naturalness_semantic_reward' not in semantic:
            raise ValueError('Luna naturalness score is mandatory')
        record['semantic_components']['naturalness'] = float(semantic['naturalness_semantic_reward'])
        record['semantic_judge_audit'] = dict(semantic)
        record['profile_action_fusion'] = {'llm_weight': PROFILE_ACTION_LLM_WEIGHT, 'sft_alignment_weight': PROFILE_ACTION_SFT_WEIGHT}
        records.append(record)
    return records

def finalize_reward_records(raw_records: Sequence[Mapping[str, Any]], reward_weights: Mapping[str, float], naturalness_calibration: Mapping[str, float]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if set(reward_weights) != set(INITIAL_REWARD_WEIGHTS):
        raise ValueError('reward-weight keys changed')
    if not math.isclose(sum(reward_weights.values()), 1.0, abs_tol=1e-08):
        raise ValueError('reward weights must sum to one')
    mean = float(naturalness_calibration['mean'])
    standard_deviation = float(naturalness_calibration['standard_deviation'])
    if not math.isfinite(mean) or not math.isfinite(standard_deviation) or standard_deviation < 0.1:
        raise ValueError('invalid frozen naturalness calibration')
    records: list[dict[str, Any]] = []
    totals: list[float] = []
    for raw_record in raw_records:
        raw = raw_record['raw_components']
        nll_naturalness = math.tanh((float(raw['naturalness']) - mean) / standard_deviation)
        semantic_naturalness = float(raw_record['semantic_components']['naturalness'])
        naturalness = NATURALNESS_SFT_WEIGHT * nll_naturalness + NATURALNESS_LLM_WEIGHT * semantic_naturalness
        naturalness_fusion = {'mode': 'LUNA_HIGH_NATURALNESS_80_PLUS_FROZEN_PATIENT_SFT_ALIGNMENT_20', 'llm_weight': NATURALNESS_LLM_WEIGHT, 'sft_alignment_weight': NATURALNESS_SFT_WEIGHT, 'semantic_naturalness': semantic_naturalness, 'sft_alignment_naturalness': nll_naturalness}
        components = {'profile_consistency': float(raw['profile_consistency']), 'counsellor_action_dependence': float(raw['counsellor_action_dependence']), 'naturalness': naturalness, 'intended_resistance_realization': float(raw['intended_resistance_realization'])}
        penalty = float(raw_record['deterministic_penalty'])
        total = sum((reward_weights[key] * value for key, value in components.items())) - penalty
        total = max(-1.5, min(1.5, total))
        record = dict(raw_record)
        record.pop('raw_components')
        record['components'] = components
        record['naturalness_fusion'] = naturalness_fusion
        record['total_reward'] = total
        records.append(record)
        totals.append(total)
    return (records, torch.tensor(totals, dtype=torch.float32, device='cuda'))

def bounded_simplex_projection(values: Mapping[str, float]) -> dict[str, float]:
    keys = list(INITIAL_REWARD_WEIGHTS)
    lower = min((values[key] - REWARD_WEIGHT_BOUNDS[key][1] for key in keys))
    upper = max((values[key] - REWARD_WEIGHT_BOUNDS[key][0] for key in keys))
    for _ in range(80):
        offset = (lower + upper) / 2.0
        projected = {key: min(REWARD_WEIGHT_BOUNDS[key][1], max(REWARD_WEIGHT_BOUNDS[key][0], values[key] - offset)) for key in keys}
        if sum(projected.values()) > 1.0:
            lower = offset
        else:
            upper = offset
    projected = {key: min(REWARD_WEIGHT_BOUNDS[key][1], max(REWARD_WEIGHT_BOUNDS[key][0], values[key] - (lower + upper) / 2.0)) for key in keys}
    if not math.isclose(sum(projected.values()), 1.0, abs_tol=1e-08):
        raise RuntimeError('bounded reward weights do not sum to one')
    return projected

def update_reward_weights(previous: Mapping[str, float], component_means: Mapping[str, float]) -> dict[str, float]:
    if set(previous) != set(INITIAL_REWARD_WEIGHTS) or set(component_means) != set(previous):
        raise ValueError('Fair-and-Stable reward keys changed')
    unnormalized = {key: float(previous[key]) * math.exp(-FAIR_STABLE_STEP_SIZE * math.tanh(float(component_means[key]))) for key in previous}
    normalizer = sum(unnormalized.values())
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError('invalid Fair-and-Stable reward-weight update')
    normalized = {key: value / normalizer for key, value in unnormalized.items()}
    bounded = bounded_simplex_projection(normalized)
    largest_delta = max((abs(bounded[key] - float(previous[key])) for key in previous))
    if largest_delta <= MAX_REWARD_WEIGHT_DELTA_PER_UPDATE:
        return bounded
    scale = MAX_REWARD_WEIGHT_DELTA_PER_UPDATE / largest_delta
    limited = {key: float(previous[key]) + scale * (bounded[key] - float(previous[key])) for key in previous}
    if not math.isclose(sum(limited.values()), 1.0, abs_tol=1e-08):
        raise RuntimeError('rate-limited reward weights do not sum to one')
    for key, value in limited.items():
        lower, upper = REWARD_WEIGHT_BOUNDS[key]
        if not lower <= value <= upper:
            raise RuntimeError(f'rate-limited reward weight is outside its bound: {key}')
    return limited

def validate_reward_composition_state(reward_weights: Mapping[str, float], component_sums: Mapping[str, float], response_count: int) -> None:
    keys = set(INITIAL_REWARD_WEIGHTS)
    if set(reward_weights) != keys or set(component_sums) != keys:
        raise RuntimeError('resume reward-composition keys changed')
    if type(response_count) is not int or not 0 <= response_count < REWARD_WEIGHT_UPDATE_RESPONSES:
        raise RuntimeError('resume reward-window count is out of range')
    if any((not math.isfinite(float(value)) for value in component_sums.values())):
        raise RuntimeError('resume reward-window sums are not finite')
    if not math.isclose(sum((float(value) for value in reward_weights.values())), 1.0, abs_tol=1e-08):
        raise RuntimeError('resume reward weights do not sum to one')
    for key, value in reward_weights.items():
        numeric = float(value)
        lower, upper = REWARD_WEIGHT_BOUNDS[key]
        if not math.isfinite(numeric) or not lower <= numeric <= upper:
            raise RuntimeError(f'resume reward weight is outside its bound: {key}')

def leave_one_out_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.numel() != NUM_GENERATIONS:
        raise ValueError('RLOO reward group has the wrong size')
    return rewards - (rewards.sum() - rewards) / (NUM_GENERATIONS - 1)

def load_policy_and_reference(policy_adapter: Path, reference_adapter: Path, revision: str):
    common = {'token': os.environ.get('HF_TOKEN') or None, 'revision': revision, 'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa', 'low_cpu_mem_usage': True}
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get('CPRL_MODEL_LOAD_PATH', MODEL_ID), token=os.environ.get('HF_TOKEN') or None, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    policy_base = AutoModelForCausalLM.from_pretrained(os.environ.get('CPRL_MODEL_LOAD_PATH', MODEL_ID), **common)
    policy = PeftModel.from_pretrained(policy_base, policy_adapter, is_trainable=True)
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.config.use_cache = False
    disable_dropout(policy)
    policy.to('cuda')
    reference_base = AutoModelForCausalLM.from_pretrained(os.environ.get('CPRL_MODEL_LOAD_PATH', MODEL_ID), **common)
    reference = PeftModel.from_pretrained(reference_base, reference_adapter, is_trainable=False)
    reference.config.use_cache = False
    disable_dropout(reference)
    reference.to('cuda').eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainable = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not trainable or any(('lora_' not in name for name, _ in trainable)):
        raise ValueError('only patient LoRA parameters may be trainable')
    return (tokenizer, policy, reference, trainable)

def aggregate_reward_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError('empty reward record set')
    keys = list(INITIAL_REWARD_WEIGHTS)
    result = {key: float(np.mean([record['components'][key] for record in records])) for key in keys}
    result['total_reward'] = float(np.mean([record['total_reward'] for record in records]))
    result['deterministic_penalty'] = float(np.mean([record['deterministic_penalty'] for record in records]))
    return result

@torch.inference_mode()
def collect_policy_reward_records(policy, reference, tokenizer, checker, rows: Sequence[dict[str, Any]], seed: int, semantic_judge: OnlineSemanticJudge, phase: str) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    policy.eval()
    for start in range(0, len(rows), SEMANTIC_API_WORKERS):
        group: list[dict[str, Any]] = []
        for index in range(start, min(start + SEMANTIC_API_WORKERS, len(rows))):
            row = rows[index]
            prompt_text, _ = patient_prompt(tokenizer, row['patient_visible'])
            samples = generate_responses(policy, tokenizer, prompt_text, seed + index, top_p=DEVELOPMENT_GENERATION_TOP_P, top_k=DEVELOPMENT_GENERATION_TOP_K)
            group.append({'row': row, 'prompt_text': prompt_text, 'responses': [sample['text'] for sample in samples], 'sampled_completion_token_ids': [sample['completion_token_ids'] for sample in samples], 'prompt_position': index})
        semantic_job = semantic_judge.submit_step(group, phase=phase)
        local_groups: list[list[dict[str, Any]]] = []
        try:
            for item in group:
                local_records, _ = local_reward_components(reference, tokenizer, checker, item['row'], item['prompt_text'], item['responses'], item['sampled_completion_token_ids'])
                local_groups.append(local_records)
        except BaseException:
            try:
                semantic_judge.finish_step(semantic_job)
            except BaseException:
                pass
            raise
        semantic_groups = semantic_judge.finish_step(semantic_job)
        for local_records, semantic_scores in zip(local_groups, semantic_groups):
            raw_records = add_semantic_reward_components(local_records, semantic_scores)
            all_records.extend(raw_records)
    return all_records

@torch.inference_mode()
def evaluate_policy(policy, reference, tokenizer, checker, rows: Sequence[dict[str, Any]], seed: int, naturalness_calibration: Mapping[str, float], semantic_judge: OnlineSemanticJudge, phase: str, reward_weights: Mapping[str, float]=INITIAL_REWARD_WEIGHTS) -> dict[str, Any]:
    raw_records = collect_policy_reward_records(policy, reference, tokenizer, checker, rows, seed, semantic_judge, phase)
    all_records, _ = finalize_reward_records(raw_records, reward_weights, naturalness_calibration)
    return {'contexts': len(rows), 'completions': len(all_records), 'mean_rewards': aggregate_reward_records(all_records)}

def create_or_load_sft_development_baseline(policy, reference, tokenizer, checker, dev_rows: Sequence[dict[str, Any]], semantic_judge: OnlineSemanticJudge, output_dir: Path, run_config_sha256: str) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    baseline_path = output_dir / 'development_baseline.json'
    calibration_path = output_dir / 'naturalness_calibration.json'
    binding_path = output_dir / 'development_baseline_binding.json'
    prompt_ids = [row['prompt_id'] for row in dev_rows]
    protocol = {'schema_version': 'certify-cbt-patient-rloo-sft-baseline-binding-v1', 'run_config_sha256': run_config_sha256, 'patient_sft_adapter_sha256': EXPECTED_PATIENT_SFT_SHA256, 'semantic_prompt_contract_sha256': semantic_judge.prompt_contract_sha256, 'context_prompt_ids_sha256': sha256_bytes(canonical(prompt_ids).encode('utf-8')), 'contexts': len(dev_rows), 'responses_per_context': NUM_GENERATIONS, 'seed': SEED + 10000000, 'test_split_used': False, 'naturalness_fusion': {'llm': NATURALNESS_LLM_WEIGHT, 'sft_alignment': NATURALNESS_SFT_WEIGHT}, 'cbt_evidence_reward': False}
    if baseline_path.is_file() or calibration_path.is_file() or binding_path.is_file():
        if not all((path.is_file() for path in (baseline_path, calibration_path, binding_path))):
            raise RuntimeError('partial frozen SFT development baseline found')
        binding = read_json(binding_path)
        baseline = read_json(baseline_path)
        calibration = read_json(calibration_path)
        expected = dict(protocol)
        expected.update({'baseline_sha256': sha256_file(baseline_path), 'naturalness_calibration_sha256': sha256_file(calibration_path)})
        if binding != expected:
            raise RuntimeError('stored SFT development baseline binding changed')
        return (baseline, calibration, binding)
    raw_records = collect_policy_reward_records(policy, reference, tokenizer, checker, dev_rows, SEED + 10000000, semantic_judge, phase='sft_development_baseline')
    natural_values = np.asarray([float(record['raw_components']['naturalness']) for record in raw_records], dtype=np.float64)
    calibration = {'schema_version': 'certify-cbt-patient-rloo-naturalness-calibration-v1', 'source': 'FROZEN_CURRENT_PATIENT_SFT_64_CONTEXT_256_RESPONSE_BASELINE', 'mean': float(natural_values.mean()), 'standard_deviation': float(natural_values.std(ddof=0)), 'responses': int(natural_values.size)}
    if calibration['standard_deviation'] < 0.1:
        raise RuntimeError('SFT naturalness calibration is degenerate')
    finalized, _ = finalize_reward_records(raw_records, INITIAL_REWARD_WEIGHTS, calibration)
    baseline = {'schema_version': 'certify-cbt-patient-rloo-development-baseline-v1', 'contexts': len(dev_rows), 'completions': len(finalized), 'mean_rewards': aggregate_reward_records(finalized)}
    write_json(baseline_path, baseline)
    write_json(calibration_path, calibration)
    binding = dict(protocol)
    binding.update({'baseline_sha256': sha256_file(baseline_path), 'naturalness_calibration_sha256': sha256_file(calibration_path)})
    write_json(binding_path, binding)
    return (baseline, calibration, binding)

def print_development_table(step: int, baseline_rewards: Mapping[str, float], current_rewards: Mapping[str, float]) -> None:
    criteria = ['profile_consistency', 'counsellor_action_dependence', 'naturalness', 'intended_resistance_realization', 'total_reward']
    if any((key not in baseline_rewards or key not in current_rewards for key in criteria)):
        raise ValueError('development result is missing a required reward criterion')
    print(f'\n[development check] step {step}', flush=True)
    print(f'{'criterion':<36} {'baseline':>10} {'current':>10} {'delta':>10}')
    print('-' * 69)
    for key in criteria:
        baseline_value = float(baseline_rewards[key])
        current_value = float(current_rewards[key])
        print(f'{key:<36} {baseline_value:>10.4f} {current_value:>10.4f} {current_value - baseline_value:>+10.4f}')

def run_development_check(policy, reference, tokenizer, checker, dev_rows: Sequence[dict[str, Any]], naturalness_calibration: Mapping[str, float], semantic_judge: OnlineSemanticJudge, baseline: Mapping[str, Any], checks_dir: Path, step: int, completed_prompts: int, run_config_sha256: str, policy_checkpoint_sha256: str) -> dict[str, Any]:
    path = checks_dir / f'step_{step:04d}.json'
    if path.is_file():
        stored = read_json(path)
        if stored.get('run_config_sha256') != run_config_sha256 or stored.get('training_step') != step or stored.get('completed_prompts') != completed_prompts or (stored.get('policy_checkpoint_sha256') != policy_checkpoint_sha256) or (not isinstance(stored.get('result'), Mapping)):
            raise RuntimeError(f'stored development check is inconsistent: {path}')
        result = stored['result']
    else:
        result = evaluate_policy(policy, reference, tokenizer, checker, dev_rows, SEED + 10000000, naturalness_calibration, semantic_judge, phase=f'development_step_{step:04d}', reward_weights=INITIAL_REWARD_WEIGHTS)
        write_json(path, {'schema_version': 'certify-cbt-patient-rloo-development-check-v1', 'run_config_sha256': run_config_sha256, 'training_step': step, 'completed_prompts': completed_prompts, 'policy_checkpoint_sha256': policy_checkpoint_sha256, 'fixed_contexts': DEV_PROMPTS, 'responses_per_context': NUM_GENERATIONS, 'evaluation_seed': SEED + 10000000, 'reward_weights': INITIAL_REWARD_WEIGHTS, 'test_split_used': False, 'result': result})
    print_development_table(step, baseline['mean_rewards'], result['mean_rewards'])
    return dict(result)

def save_resume(path: Path, policy, optimizer, scheduler, next_prompt_index: int, order: Sequence[int], run_config_sha256: str, reward_weights: Mapping[str, float], weight_window_component_sums: Mapping[str, float], weight_window_response_count: int, reward_weight_updates: int) -> None:
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    policy.save_pretrained(temporary / 'policy_adapter', safe_serialization=True)
    saved_adapter = temporary / 'policy_adapter' / 'adapter_model.safetensors'
    saved_adapter_config = temporary / 'policy_adapter' / 'adapter_config.json'
    if not saved_adapter.is_file() or not saved_adapter_config.is_file():
        raise RuntimeError('policy checkpoint adapter files were not written')
    torch.save({'next_prompt_index': next_prompt_index, 'order': list(order), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'run_config_sha256': run_config_sha256, 'policy_adapter_sha256': sha256_file(saved_adapter), 'policy_adapter_config_sha256': sha256_file(saved_adapter_config), 'reward_weights': dict(reward_weights), 'weight_window_component_sums': dict(weight_window_component_sums), 'weight_window_response_count': int(weight_window_response_count), 'reward_weight_updates': int(reward_weight_updates)}, temporary / 'training_state.pt')
    if path.exists():
        shutil.rmtree(path)
    temporary.rename(path)

def main() -> None:
    started = time.time()
    print('[stage 1/7] validating A100, bundle, adapter, and secrets', flush=True)
    hardware = validate_hardware()
    verify_bundle()
    adapter = locate_patient_adapter()
    hf_token = os.environ.get('HF_TOKEN', '').strip()
    if not hf_token and not os.environ.get('CPRL_MODEL_LOAD_PATH'):
        raise RuntimeError('HF_TOKEN is required for the gated Llama base model')
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        raise RuntimeError('OPENAI_API_KEY is required for Luna-high semantic rewards')
    max_llm_cost_usd = float(os.environ.get('PRL5_MAX_LLM_COST_USD', '10.0'))
    if not math.isfinite(max_llm_cost_usd) or max_llm_cost_usd <= 0.0:
        raise ValueError('PRL5_MAX_LLM_COST_USD must be one positive total cost cap')
    resolved_revision = EXPECTED_BASE_MODEL_REVISION
    train_rows = read_jsonl(BUNDLE_ROOT / 'data' / 'patient_rloo_train.jsonl')
    development_rows = read_jsonl(BUNDLE_ROOT / 'data' / 'patient_rloo_development.jsonl')
    if len(train_rows) != TRAIN_PROMPTS or len(development_rows) != DEVELOPMENT_PROMPTS:
        raise ValueError('prompt counts changed')
    if any((row.get('patient_target_present') is not False for row in train_rows + development_rows)):
        raise ValueError('patient targets are forbidden in RLOO prompts')
    if any((row.get('test_row') is not False for row in train_rows + development_rows)):
        raise ValueError('test rows are forbidden in the RLOO bundle')
    groups_train = {row['source_group_id'] for row in train_rows}
    groups_dev = {row['source_group_id'] for row in development_rows}
    if groups_train & groups_dev:
        raise ValueError('train/development source-group leakage')
    print('[stage 2/7] assigning resistance states for 3,500 causal contexts', flush=True)
    resistance = FrozenResistanceTracker(RESISTANCE_TRACKER_ROOT)
    train_rows = resistance.resolve(train_rows)
    development_rows = resistance.resolve(development_rows)
    resistance_tracker_sha256 = resistance.checkpoint_sha256
    tracker_resolution = {'train_assigned_states': dict(Counter((row['runtime_assigned_resistance_state'] for row in train_rows))), 'development_assigned_states': dict(Counter((row['runtime_assigned_resistance_state'] for row in development_rows))), 'source_proxy_agreement': float(np.mean([row['resistance_tracker_audit']['source_proxy_state_agreement'] for row in train_rows + development_rows]))}
    del resistance
    gc.collect()
    torch.cuda.empty_cache()
    print('[stage 3/7] resistance assignment complete; preparing the run', flush=True)
    run_config = {'implementation_version': IMPLEMENTATION_VERSION, 'algorithm': 'RLOO_REINFORCE_LEAVE_ONE_OUT', 'model_id': MODEL_ID, 'base_model_revision': resolved_revision, 'patient_sft_adapter_sha256': EXPECTED_PATIENT_SFT_SHA256, 'counsellor_sft_context_checkpoint_sha256': EXPECTED_COUNSELLOR_SFT_SHA256, 'resistance_state_tracker_sha256': resistance_tracker_sha256, 'resistance_state_tracker_path_supplied_at_runtime': True, 'resistance_expression_checker_sha256': EXPECTED_EXPRESSION_SHA256, 'bundle_manifest_sha256': sha256_file(BUNDLE_ROOT / 'bundle_manifest.json'), 'train_file_sha256': sha256_file(BUNDLE_ROOT / 'data' / 'patient_rloo_train.jsonl'), 'development_file_sha256': sha256_file(BUNDLE_ROOT / 'data' / 'patient_rloo_development.jsonl'), 'train_prompts': TRAIN_PROMPTS, 'development_prompts': DEVELOPMENT_PROMPTS, 'selection_development_prompts': DEV_PROMPTS, 'development_validation_every_steps': VALIDATE_EVERY_STEPS, 'development_validation_steps': list(range(VALIDATE_EVERY_STEPS, len(train_rows) // GRADIENT_ACCUMULATION_PROMPTS + 1, VALIDATE_EVERY_STEPS)), 'test_prompts': 0, 'seed': SEED, 'epochs': TRAIN_EPOCHS, 'num_generations': NUM_GENERATIONS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'kl_beta': KL_BETA, 'kl_reward_shaping': 'TASK_REWARD_MINUS_BETA_LOG_POLICY_OVER_REFERENCE_BEFORE_RLOO_BASELINE', 'training_sampling': {'temperature': GENERATION_TEMPERATURE, 'top_p': TRAIN_GENERATION_TOP_P, 'top_k': TRAIN_GENERATION_TOP_K, 'min_new_tokens': MIN_NEW_TOKENS, 'trajectory_binding': 'EXACT_SAMPLED_TOKEN_IDS', 'policy_logps': 'SAME_TEMPERATURE_AND_MIN_NEW_TOKEN_EOS_MASK'}, 'development_sampling': {'temperature': GENERATION_TEMPERATURE, 'top_p': DEVELOPMENT_GENERATION_TOP_P, 'top_k': DEVELOPMENT_GENERATION_TOP_K, 'min_new_tokens': MIN_NEW_TOKENS}, 'gradient_accumulation_prompts': GRADIENT_ACCUMULATION_PROMPTS, 'checkpoint_storage': {'directory_inside_run': 'checkpoints/latest', 'save_every_prompts': SAVE_EVERY_PROMPTS, 'retention': 'LATEST_ONLY'}, 'training_rollout_storage': {'path_inside_run': 'training_log.jsonl', 'row_unit': 'ONE_PROMPT_WITH_FOUR_RLOO_RESPONSES', 'full_patient_response_text': True, 'exact_sampled_completion_token_ids': True, 'prompt_join_key': 'prompt_id'}, 'reward_weight_update_prompts': REWARD_WEIGHT_UPDATE_PROMPTS, 'reward_weight_update_responses': REWARD_WEIGHT_UPDATE_RESPONSES, 'reward_weight_method': 'BOUNDED_FAIR_AND_STABLE_MIRROR_DESCENT', 'fair_stable_step_size': FAIR_STABLE_STEP_SIZE, 'maximum_reward_weight_delta_per_update': MAX_REWARD_WEIGHT_DELTA_PER_UPDATE, 'reward_weight_bounds': {key: {'minimum': bounds[0], 'maximum': bounds[1]} for key, bounds in REWARD_WEIGHT_BOUNDS.items()}, 'naturalness_calibration': 'CURRENT_PATIENT_SFT_64_CONTEXT_256_RESPONSE_BASELINE', 'frozen_development_baseline_protocol': {'contexts': DEV_PROMPTS, 'responses_per_context': NUM_GENERATIONS, 'seed': SEED + 10000000, 'generated_once_then_hash_bound': True}, 'semantic_reward': {'model': 'gpt-5.6-luna', 'reasoning_effort': 'high', 'profile_action_fusion': {'llm_weight': PROFILE_ACTION_LLM_WEIGHT, 'sft_alignment_weight': PROFILE_ACTION_SFT_WEIGHT}, 'naturalness_fusion': {'llm_weight': NATURALNESS_LLM_WEIGHT, 'sft_alignment_weight': NATURALNESS_SFT_WEIGHT}, 'parallel_workers': SEMANTIC_API_WORKERS, 'responses_per_request': NUM_GENERATIONS, 'standard_max_output_tokens': STANDARD_MAX_OUTPUT_TOKENS, 'max_output_token_retry_schedule': list(MAX_OUTPUT_TOKEN_SCHEDULE[1:]), 'retry_rule': 'ONLY_INCOMPLETE_MAX_OUTPUT_TOKENS_TWICE_AT_8000_THEN_10000', 'prompt_contract_sha256': sha256_bytes(canonical(read_json(BUNDLE_ROOT / 'judge' / 'prompt_contract.json')).encode('utf-8'))}, 'max_prompt_tokens': MAX_PROMPT_TOKENS, 'max_new_tokens': MAX_NEW_TOKENS, 'initial_reward_weights': INITIAL_REWARD_WEIGHTS, 'development_selection_weights': INITIAL_REWARD_WEIGHTS, 'forbidden_rewards': sorted(FORBIDDEN_REWARDS), 'updated_parameters': 'PATIENT_LORA_ONLY', 'counsellor_partner': 'FROZEN_ACCEPTED_V5_CAUSAL_SESSION_CONTEXTS', 'resistance_timing': 'AFTER_COUNSELLOR_BEFORE_PATIENT', 'patient_receives': 'NATURAL_ENGAGEMENT_CUE_FROM_COMMITTED_STATE', 'cbt_evidence_tracker_is_patient_reward': False, 'cbt_evidence_tracker_packaged': False, 'cbt_progress_reward': False, 'symptom_reduction_is_reward': False, 'shared_outcome_reward': False, 'evaluation_scope': 'SYNTHETIC_SIMULATION_NOT_CLINICAL_EFFICACY'}
    run_config_sha = sha256_bytes(canonical(run_config).encode('utf-8'))
    output_dir = OUTPUT_ROOT / f'run_{run_config_sha[:12].lower()}'
    require_resume = os.environ.get('PRL5_REQUIRE_RESUME', '0') == '1'
    if require_resume and (not (output_dir / 'checkpoints' / 'latest' / 'training_state.pt').is_file()) and (not (output_dir / 'COMPLETED.json').is_file()):
        raise RuntimeError('this launcher requires an existing checkpoint for this exact RLOO v5 run')
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / 'COMPLETED.json'
    if completed_path.is_file():
        completed = read_json(completed_path)
        selected = output_dir / 'selected_patient_rl_adapter' / 'adapter_model.safetensors'
        stored_summary = output_dir / 'training_summary.json'
        stored_rollouts = output_dir / 'training_log.jsonl'
        if completed.get('status') == 'COMPLETE' and completed.get('run_config_sha256') == run_config_sha and selected.is_file() and (sha256_file(selected) == completed.get('selected_adapter_sha256')) and stored_summary.is_file() and (sha256_file(stored_summary) == completed.get('training_summary_sha256')) and stored_rollouts.is_file() and (sha256_file(stored_rollouts) == completed.get('training_rollout_dataset_sha256')) and (completed.get('training_rollout_prompt_rows') == TRAIN_PROMPTS) and (completed.get('training_rollout_responses') == TRAIN_PROMPTS * NUM_GENERATIONS):
            print('[stage 7/7] intact completed run already exists', flush=True)
            print(json.dumps(completed, indent=2, sort_keys=True))
            return
        raise RuntimeError('existing completion marker is not bound to intact outputs')
    config_path = output_dir / 'RUN_CONFIG.json'
    if config_path.is_file():
        stored = read_json(config_path)
        if stored.get('run_config_sha256') != run_config_sha:
            raise RuntimeError('existing output belongs to a different run configuration')
    else:
        write_json(config_path, {'run_config_sha256': run_config_sha, 'config': run_config, 'hardware': hardware, 'tracker_resolution': tracker_resolution})
    print('[stage 4/7] loading the expression checker and two frozen-start Llama policies', flush=True)
    expression = FrozenExpressionChecker(BUNDLE_ROOT / 'trackers' / 'resistance_expression')
    semantic_judge = OnlineSemanticJudge(BUNDLE_ROOT, output_dir, max_cost_usd=max_llm_cost_usd, workers=SEMANTIC_API_WORKERS)
    resume_dir = output_dir / 'checkpoints' / 'latest'
    resume_temporary = resume_dir.with_name(resume_dir.name + '.tmp')
    if resume_temporary.exists():
        raise RuntimeError('interrupted checkpoints/latest.tmp detected; inspect or remove it before resuming')
    resume_adapter_dir = resume_dir / 'policy_adapter'
    resume_state_path = resume_dir / 'training_state.pt'
    resume_adapter_complete = resume_adapter_dir.is_dir() and (resume_adapter_dir / 'adapter_config.json').is_file() and (resume_adapter_dir / 'adapter_model.safetensors').is_file()
    resume_state_present = resume_state_path.is_file()
    if resume_dir.exists() and (not (resume_adapter_complete and resume_state_present)):
        raise RuntimeError('partial checkpoints/latest detected; both the complete policy adapter and training_state.pt are required')
    if resume_adapter_complete != resume_state_present:
        raise RuntimeError('resume adapter/state presence mismatch')
    resume_state: dict[str, Any] | None = None
    if resume_state_present:
        resume_state = torch.load(resume_state_path, map_location='cpu', weights_only=False)
        if resume_state.get('run_config_sha256') != run_config_sha or resume_state.get('policy_adapter_sha256') != sha256_file(resume_adapter_dir / 'adapter_model.safetensors') or resume_state.get('policy_adapter_config_sha256') != sha256_file(resume_adapter_dir / 'adapter_config.json'):
            raise RuntimeError('resume checkpoint adapter/state hash binding failed')
    policy_adapter = resume_adapter_dir if resume_state_present else adapter
    tokenizer, policy, reference, trainable = load_policy_and_reference(policy_adapter, adapter, resolved_revision)
    print('[stage 5/7] model loading complete; freezing/checking the current-SFT baseline', flush=True)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_optimizer_steps = math.ceil(len(train_rows) * TRAIN_EPOCHS / GRADIENT_ACCUMULATION_PROMPTS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_optimizer_steps * WARMUP_RATIO)), num_training_steps=total_optimizer_steps)
    rng = random.Random(SEED)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    next_prompt_index = 0
    current_reward_weights = dict(INITIAL_REWARD_WEIGHTS)
    weight_window_component_sums = {key: 0.0 for key in INITIAL_REWARD_WEIGHTS}
    weight_window_response_count = 0
    reward_weight_updates = 0
    if resume_state is not None:
        state = resume_state
        order = state['order']
        next_prompt_index = int(state['next_prompt_index'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        current_reward_weights = {key: float(value) for key, value in state['reward_weights'].items()}
        weight_window_component_sums = {key: float(value) for key, value in state['weight_window_component_sums'].items()}
        weight_window_response_count = int(state['weight_window_response_count'])
        reward_weight_updates = int(state['reward_weight_updates'])
        validate_reward_composition_state(current_reward_weights, weight_window_component_sums, weight_window_response_count)
        if weight_window_response_count != next_prompt_index % REWARD_WEIGHT_UPDATE_PROMPTS * NUM_GENERATIONS or reward_weight_updates != next_prompt_index // REWARD_WEIGHT_UPDATE_PROMPTS:
            raise RuntimeError('resume reward-composition state is inconsistent')
        if next_prompt_index % GRADIENT_ACCUMULATION_PROMPTS:
            raise RuntimeError('resume must start at a completed four-prompt optimizer step')
        print(f'[resume] continuing at prompt {next_prompt_index}/{len(order)}')
    else:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    dev_order = sorted(range(len(development_rows)), key=lambda index: sha256_bytes(f'{SEED}|{development_rows[index]['prompt_id']}'.encode('utf-8')))[:DEV_PROMPTS]
    dev_rows = [development_rows[index] for index in dev_order]
    baseline, naturalness_calibration, baseline_binding = create_or_load_sft_development_baseline(policy, reference, tokenizer, expression, dev_rows, semantic_judge, output_dir, run_config_sha)
    print('[stage 5/7] frozen current-SFT baseline ready; starting patient RLOO', flush=True)
    development_checks_dir = output_dir / 'development_checks'
    development_checks_dir.mkdir(exist_ok=True)
    development_baseline_record = {'schema_version': 'certify-cbt-patient-rloo-development-baseline-record-v1', 'run_config_sha256': run_config_sha, 'training_step': 0, 'completed_prompts': 0, 'policy_checkpoint_sha256': EXPECTED_PATIENT_SFT_SHA256, 'fixed_contexts': DEV_PROMPTS, 'responses_per_context': NUM_GENERATIONS, 'evaluation_seed': SEED + 10000000, 'reward_weights': INITIAL_REWARD_WEIGHTS, 'test_split_used': False, 'result': baseline}
    development_baseline_record_path = development_checks_dir / 'step_0000_baseline.json'
    if development_baseline_record_path.is_file():
        if read_json(development_baseline_record_path) != development_baseline_record:
            raise RuntimeError('stored development baseline record changed')
    else:
        write_json(development_baseline_record_path, development_baseline_record)
    log_path = output_dir / 'training_log.jsonl'
    if next_prompt_index and log_path.is_file():
        retained = [line for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip() and int(json.loads(line)['prompt_position']) < next_prompt_index]
        log_path.write_text('\n'.join(retained) + ('\n' if retained else ''), encoding='utf-8')
    weight_log_path = output_dir / 'reward_weight_updates.jsonl'
    if next_prompt_index and weight_log_path.is_file():
        retained = [line for line in weight_log_path.read_text(encoding='utf-8').splitlines() if line.strip() and int(json.loads(line)['completed_prompts']) <= next_prompt_index]
        weight_log_path.write_text('\n'.join(retained) + ('\n' if retained else ''), encoding='utf-8')
    optimizer.zero_grad(set_to_none=True)
    recent_rewards: list[float] = []
    show_training_progress(next_prompt_index // GRADIENT_ACCUMULATION_PROMPTS, total_optimizer_steps, next_prompt_index, len(order), 0.0, semantic_judge.snapshot()['estimated_recorded_cost_usd'])
    completed_resume_step = next_prompt_index // GRADIENT_ACCUMULATION_PROMPTS
    if completed_resume_step > 0 and completed_resume_step < total_optimizer_steps and (completed_resume_step % VALIDATE_EVERY_STEPS == 0):
        run_development_check(policy, reference, tokenizer, expression, dev_rows, naturalness_calibration, semantic_judge, baseline, development_checks_dir, completed_resume_step, next_prompt_index, run_config_sha, sha256_file(resume_dir / 'policy_adapter' / 'adapter_model.safetensors'))
        show_training_progress(completed_resume_step, total_optimizer_steps, next_prompt_index, len(order), 0.0, semantic_judge.snapshot()['estimated_recorded_cost_usd'])
    for step_start in range(next_prompt_index, len(order), GRADIENT_ACCUMULATION_PROMPTS):
        step_end = min(step_start + GRADIENT_ACCUMULATION_PROMPTS, len(order))
        if step_end - step_start != GRADIENT_ACCUMULATION_PROMPTS:
            raise RuntimeError('training prompt count must form complete four-prompt steps')
        generated_step: list[dict[str, Any]] = []
        for position in range(step_start, step_end):
            row = train_rows[order[position]]
            prompt_text, prompt_tokens = patient_prompt(tokenizer, row['patient_visible'])
            samples = generate_responses(policy, tokenizer, prompt_text, SEED * 1000000 + position, top_p=TRAIN_GENERATION_TOP_P, top_k=TRAIN_GENERATION_TOP_K)
            generated_step.append({'row': row, 'prompt_text': prompt_text, 'prompt_tokens': prompt_tokens, 'responses': [sample['text'] for sample in samples], 'sampled_completion_token_ids': [sample['completion_token_ids'] for sample in samples], 'prompt_position': position})
        semantic_job = semantic_judge.submit_step(generated_step, phase='training')
        local_groups: list[tuple[list[dict[str, Any]], torch.Tensor]] = []
        try:
            for item in generated_step:
                local_groups.append(local_reward_components(reference, tokenizer, expression, item['row'], item['prompt_text'], item['responses'], item['sampled_completion_token_ids']))
        except BaseException:
            try:
                semantic_judge.finish_step(semantic_job)
            except BaseException:
                pass
            raise
        try:
            semantic_groups = semantic_judge.finish_step(semantic_job)
        except BaseException:
            save_resume(resume_dir, policy, optimizer, scheduler, step_start, order, run_config_sha, current_reward_weights, weight_window_component_sums, weight_window_response_count, reward_weight_updates)
            print(f'[resume] Luna failure checkpoint saved at optimizer step {step_start // GRADIENT_ACCUMULATION_PROMPTS} (prompt {step_start})', flush=True)
            raise
        weights_used = dict(current_reward_weights)
        step_logs: list[dict[str, Any]] = []
        for item, semantic_scores, local_group in zip(generated_step, semantic_groups, local_groups):
            row = item['row']
            prompt_text = item['prompt_text']
            responses = item['responses']
            sampled_completion_token_ids = item['sampled_completion_token_ids']
            local_records, reference_sampling_logps = local_group
            raw_records = add_semantic_reward_components(local_records, semantic_scores)
            reward_records, rewards = finalize_reward_records(raw_records, weights_used, naturalness_calibration)
            policy.train()
            _, _, policy_sampling_logps = conditional_logps(policy, tokenizer, [prompt_text] * NUM_GENERATIONS, sampled_completion_token_ids, require_grad=True, sampling_temperature=GENERATION_TEMPERATURE, sampling_min_new_tokens=MIN_NEW_TOKENS)
            if policy_sampling_logps is None:
                raise RuntimeError('policy sampling log-probabilities were not calculated')
            sampled_log_ratio = policy_sampling_logps.detach() - reference_sampling_logps
            kl_shaped_rewards = rewards - KL_BETA * sampled_log_ratio
            advantages = leave_one_out_advantages(kl_shaped_rewards)
            policy_loss = -(advantages.detach() * policy_sampling_logps).mean()
            loss = policy_loss
            (loss / GRADIENT_ACCUMULATION_PROMPTS).backward()
            recent_rewards.append(float(rewards.mean()))
            for reward_record in reward_records:
                for key, value in reward_record['components'].items():
                    weight_window_component_sums[key] += float(value)
                weight_window_response_count += 1
            step_logs.append({'schema_version': 'certify-cbt-patient-rloo-training-rollout-v1', 'split': 'train', 'prompt_position': item['prompt_position'], 'prompt_id': row['prompt_id'], 'source_group_id': row['source_group_id'], 'profile_id': row['profile_id'], 'session_id': row['session_id'], 'exchange_index': row['exchange_index'], 'runtime_transition_direction': row['runtime_transition_direction'], 'runtime_assigned_resistance_state': row['runtime_assigned_resistance_state'], 'prompt_tokens': item['prompt_tokens'], 'patient_responses': list(responses), 'response_sha256': [sha256_bytes(response.encode('utf-8')) for response in responses], 'sampled_completion_token_ids': [list(token_ids) for token_ids in sampled_completion_token_ids], 'sampled_completion_token_ids_sha256': [sha256_bytes(canonical(token_ids).encode('utf-8')) for token_ids in sampled_completion_token_ids], 'rewards': reward_records, 'reward_weights': weights_used, 'task_rewards': [float(value) for value in rewards], 'kl_shaped_rewards': [float(value) for value in kl_shaped_rewards], 'advantages': [float(value) for value in advantages], 'policy_loss': float(policy_loss.detach()), 'mean_sampled_log_ratio': float(sampled_log_ratio.mean()), 'mean_kl_reward_penalty': float((-KL_BETA * sampled_log_ratio).mean())})
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        processed = step_end
        for step_log in step_logs:
            step_log['optimizer_step'] = step_log['prompt_position'] == step_end - 1
            step_log['learning_rate'] = float(scheduler.get_last_lr()[0])
            append_jsonl(log_path, step_log)
        if weight_window_response_count == REWARD_WEIGHT_UPDATE_RESPONSES:
            component_means = {key: value / weight_window_response_count for key, value in weight_window_component_sums.items()}
            updated_weights = update_reward_weights(current_reward_weights, component_means)
            reward_weight_updates += 1
            append_jsonl(weight_log_path, {'update_index': reward_weight_updates, 'completed_prompts': processed, 'responses_in_window': weight_window_response_count, 'component_means': component_means, 'previous_weights': current_reward_weights, 'updated_weights': updated_weights})
            current_reward_weights = updated_weights
            weight_window_component_sums = {key: 0.0 for key in INITIAL_REWARD_WEIGHTS}
            weight_window_response_count = 0
        elif weight_window_response_count > REWARD_WEIGHT_UPDATE_RESPONSES:
            raise RuntimeError('reward-weight update boundary was crossed')
        show_training_progress(processed // GRADIENT_ACCUMULATION_PROMPTS, total_optimizer_steps, processed, len(order), float(np.mean(recent_rewards[-GRADIENT_ACCUMULATION_PROMPTS:])), semantic_judge.snapshot()['estimated_recorded_cost_usd'])
        completed_steps = processed // GRADIENT_ACCUMULATION_PROMPTS
        if completed_steps < total_optimizer_steps and completed_steps % VALIDATE_EVERY_STEPS == 0:
            save_resume(resume_dir, policy, optimizer, scheduler, processed, order, run_config_sha, current_reward_weights, weight_window_component_sums, weight_window_response_count, reward_weight_updates)
            run_development_check(policy, reference, tokenizer, expression, dev_rows, naturalness_calibration, semantic_judge, baseline, development_checks_dir, completed_steps, processed, run_config_sha, sha256_file(resume_dir / 'policy_adapter' / 'adapter_model.safetensors'))
            show_training_progress(completed_steps, total_optimizer_steps, processed, len(order), float(np.mean(recent_rewards[-GRADIENT_ACCUMULATION_PROMPTS:])), semantic_judge.snapshot()['estimated_recorded_cost_usd'])
        if processed % SAVE_EVERY_PROMPTS == 0:
            save_resume(resume_dir, policy, optimizer, scheduler, processed, order, run_config_sha, current_reward_weights, weight_window_component_sums, weight_window_response_count, reward_weight_updates)
    expected_weight_updates = len(order) // REWARD_WEIGHT_UPDATE_PROMPTS
    expected_remainder_responses = len(order) % REWARD_WEIGHT_UPDATE_PROMPTS * NUM_GENERATIONS
    if reward_weight_updates != expected_weight_updates or weight_window_response_count != expected_remainder_responses:
        raise RuntimeError('final reward-weight window accounting is inconsistent')
    save_resume(resume_dir, policy, optimizer, scheduler, len(order), order, run_config_sha, current_reward_weights, weight_window_component_sums, weight_window_response_count, reward_weight_updates)
    candidate_dir = output_dir / 'final_candidate_adapter'
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    policy.save_pretrained(candidate_dir, safe_serialization=True)
    tokenizer.save_pretrained(candidate_dir)
    print('[stage 6/7] training complete; running the fixed 64-context development gate', flush=True)
    final_development = run_development_check(policy, reference, tokenizer, expression, dev_rows, naturalness_calibration, semantic_judge, baseline, development_checks_dir, total_optimizer_steps, len(order), run_config_sha, sha256_file(candidate_dir / 'adapter_model.safetensors'))
    before = baseline['mean_rewards']
    after = final_development['mean_rewards']
    component_floor_ok = all((after[key] >= before[key] - 0.05 for key in ('profile_consistency', 'counsellor_action_dependence', 'naturalness', 'intended_resistance_realization')))
    accepted = after['total_reward'] >= before['total_reward'] + 0.005 and component_floor_ok
    selected_dir = output_dir / 'selected_patient_rl_adapter'
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    if accepted:
        shutil.copytree(candidate_dir, selected_dir)
        selected_stage = 'PATIENT_RLOO_ACCEPTED'
    else:
        shutil.copytree(adapter, selected_dir)
        selected_stage = 'PATIENT_SFT_RETAINED_RLOO_CANDIDATE_REJECTED'
    selected_model = selected_dir / 'adapter_model.safetensors'
    if not selected_model.is_file():
        raise FileNotFoundError(selected_model)
    semantic_snapshot = semantic_judge.snapshot()
    semantic_artifacts = {'score_cache_sha256': sha256_file(semantic_judge.cache_path), 'attempt_ledger_sha256': sha256_file(semantic_judge.attempt_path)}
    with log_path.open('r', encoding='utf-8') as handle:
        training_rollout_prompt_rows = sum((1 for line in handle if line.strip()))
    if training_rollout_prompt_rows != len(order):
        raise RuntimeError('completed training rollout dataset has the wrong prompt count')
    training_rollout_dataset_sha256 = sha256_file(log_path)
    summary = {'status': 'COMPLETE', 'selected_stage': selected_stage, 'rloo_candidate_accepted': accepted, 'run_config_sha256': run_config_sha, 'baseline_development': baseline, 'final_development': final_development, 'development_checks': {'steps': [250, 500, 750], 'directory': str(development_checks_dir), 'files': {path.name: sha256_file(path) for path in sorted(development_checks_dir.glob('*.json'))}}, 'tracker_resolution': tracker_resolution, 'selected_adapter_sha256': sha256_file(selected_model), 'candidate_adapter_sha256': sha256_file(candidate_dir / 'adapter_model.safetensors'), 'elapsed_seconds': time.time() - started, 'trainable_parameters': sum((parameter.numel() for _, parameter in trainable)), 'initial_reward_weights': INITIAL_REWARD_WEIGHTS, 'final_training_reward_weights': current_reward_weights, 'reward_weight_updates_applied': reward_weight_updates, 'incomplete_final_weight_window_prompts': weight_window_response_count // NUM_GENERATIONS, 'naturalness_calibration': naturalness_calibration, 'frozen_sft_development_baseline_binding': baseline_binding, 'semantic_judge': semantic_snapshot, 'semantic_artifacts': semantic_artifacts, 'training_rollout_dataset': {'path': str(log_path), 'sha256': training_rollout_dataset_sha256, 'prompt_rows': training_rollout_prompt_rows, 'responses': training_rollout_prompt_rows * NUM_GENERATIONS, 'full_patient_response_text': True, 'exact_sampled_completion_token_ids': True, 'prompt_join_key': 'prompt_id'}, 'scope': 'simulated patient behaviour; no clinical efficacy claim', 'next_stage': 'freeze/hash selected patient adapter, then train counsellor RLOO while this patient remains frozen'}
    write_json(output_dir / 'training_summary.json', summary)
    completed = {'status': 'COMPLETE', 'task': 'CERTIFY_CBT_PATIENT_RLOO', 'selected_stage': selected_stage, 'run_config_sha256': run_config_sha, 'selected_adapter_sha256': summary['selected_adapter_sha256'], 'training_summary_sha256': sha256_file(output_dir / 'training_summary.json'), 'training_rollout_dataset_sha256': training_rollout_dataset_sha256, 'training_rollout_prompt_rows': training_rollout_prompt_rows, 'training_rollout_responses': training_rollout_prompt_rows * NUM_GENERATIONS, 'output_dir': str(output_dir)}
    write_json(completed_path, completed)
    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    print('[stage 7/7] run complete', flush=True)
    print(json.dumps(completed, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
