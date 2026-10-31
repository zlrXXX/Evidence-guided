from __future__ import annotations
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(os.environ['PRL5_BUNDLE_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_patient_rloo as legacy
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MODEL_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
ALLOWED_VISIBLE_KEYS = {'patient_profile', 'symptom_level', 'counselling_progress', 'current_resistance_state', 'dialogue_history', 'current_counsellor_utterance', 'response_phase'}
REQUIRED_VISIBLE_KEYS = {'patient_profile', 'current_resistance_state', 'dialogue_history', 'current_counsellor_utterance', 'response_phase'}
ENGAGEMENT_CUES = {'REJECTING_DISENGAGED': 'The patient is currently strongly guarded or rejecting; respond with limited engagement unless the visible dialogue itself supports a small shift.', 'AMBIVALENT': 'The patient is currently mixed or uncertain; express both hesitation and some willingness where supported by the visible dialogue.', 'COOPERATIVE': 'The patient is currently willing to engage with the counsellor while retaining realistic uncertainty or difficulty.', 'COLLABORATIVE_EXPLORATION': 'The patient is currently exploring the issue collaboratively while still speaking from their own experience and without assumed independent mastery.'}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))
PROMPTS = read_json(ROOT / 'contracts' / 'prompt_contracts.json')

def locate_patient_adapter() -> Path:
    adapter = Path(os.environ['CPRL_PATIENT_SFT_DIR'])
    model_path = adapter / 'adapter_model.safetensors'
    config_path = adapter / 'adapter_config.json'
    for path in (model_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(model_path) != os.environ['CPRL_PATIENT_SFT_SHA256']:
        raise ValueError('the staged CACTUS patient SFT adapter hash changed')
    config = read_json(config_path)
    if config.get('base_model_name_or_path') != MODEL_ID:
        raise ValueError('the patient adapter is not based on Llama-3.1-8B-Instruct')
    return adapter

def patient_prompt(tokenizer, visible: Mapping[str, Any]) -> tuple[str, int]:
    keys = set(visible)
    if not REQUIRED_VISIBLE_KEYS <= keys or not keys <= ALLOWED_VISIBLE_KEYS:
        raise ValueError('patient-visible context violates the frozen causal boundary')
    phase = str(visible['response_phase'])
    if phase == 'PATIENT_END_ACCEPTANCE':
        system = PROMPTS['patient_end_acceptance_system']
    elif phase == 'PATIENT_END_CONTINUE':
        system = PROMPTS['patient_end_continue_system']
    elif phase in {'IN_SESSION_RESPONSE', 'POST_CLOSING_ACKNOWLEDGEMENT'}:
        system = PROMPTS['patient_system']
    else:
        raise ValueError(f'unknown patient response phase: {phase}')
    state = str(visible['current_resistance_state'])
    if state not in ENGAGEMENT_CUES:
        raise ValueError(f'unknown committed resistance state: {state}')
    shown = copy.deepcopy(dict(visible))
    shown.pop('current_resistance_state')
    shown['provisional_engagement_cue'] = ENGAGEMENT_CUES[state]
    user = 'VISIBLE_PATIENT_RUNTIME_CONTEXT=\n' + json.dumps(shown, ensure_ascii=False, sort_keys=True)
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    if length > legacy.MAX_PROMPT_TOKENS:
        raise ValueError(f'patient prompt requires {length} tokens; no field is truncated')
    return (text, length)

def resistance_input(row: Mapping[str, Any]) -> str:
    visible = row['patient_visible']
    history = '\n'.join((f'{turn['role']}: {turn['content']}' for turn in legacy.alternating_history(visible['dialogue_history'])))
    progress = visible.get('counselling_progress') or row['tracker_context'].get('counselling_progress', {})
    decision = row['reward_context']['selected_counsellor_action']
    return f'Current resistance: {row['previous_resistance_state']}\nActive CBT technique: {decision.get('technique_id')}\nActive CBT step: {decision.get('step_id')}\nCounsellor: {visible['current_counsellor_utterance']}\nPatient profile: {json.dumps(visible['patient_profile'], ensure_ascii=False, sort_keys=True)}\nCounselling progress: {json.dumps(progress, ensure_ascii=False, sort_keys=True)}\nDialogue context:\n{history}'

def load_policy_and_reference(policy_adapter: Path, reference_adapter: Path, revision: str):
    token = os.environ.get('HF_TOKEN') or None
    common = {'token': token, 'revision': revision, 'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa', 'low_cpu_mem_usage': True}
    model_path = os.environ['CPRL_MODEL_LOAD_PATH']
    common['local_files_only'] = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, revision=revision, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    policy_base = AutoModelForCausalLM.from_pretrained(model_path, **common)
    policy = PeftModel.from_pretrained(policy_base, policy_adapter, is_trainable=True)
    policy.peft_config['default'].base_model_name_or_path = MODEL_ID
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.config.use_cache = False
    legacy.disable_dropout(policy)
    policy.to('cuda')
    reference_base = AutoModelForCausalLM.from_pretrained(model_path, **common)
    reference = PeftModel.from_pretrained(reference_base, reference_adapter, is_trainable=False)
    reference.config.use_cache = False
    legacy.disable_dropout(reference)
    reference.to('cuda').eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainable = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not trainable or any(('lora_' not in name for name, _ in trainable)):
        raise ValueError('only the CACTUS patient LoRA parameters may be trainable')
    return (tokenizer, policy, reference, trainable)

def deterministic_penalty(profile: Mapping[str, Any], response: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    words = response.split()
    if legacy.AI_PATTERNS.search(response):
        reasons.append('AI_OR_METADATA_STYLE')
    if legacy.THERAPIST_PATTERNS.search(response):
        reasons.append('ROLE_REVERSAL')
    if legacy.explicit_profile_contradiction(profile, response):
        reasons.append('EXPLICIT_PROFILE_CONTRADICTION')
    unique_ratio = len({word.casefold() for word in words}) / max(1, len(words))
    if len(words) >= 16 and unique_ratio < 0.45:
        reasons.append('REPETITIVE')
    return (min(0.6, 0.2 * len(reasons)), reasons)

def checkpoint_complete(path: Path) -> bool:
    return all((item.is_file() for item in (path / 'policy_adapter' / 'adapter_model.safetensors', path / 'policy_adapter' / 'adapter_config.json', path / 'training_state.pt')))

def repair_checkpoint_pair(latest: Path) -> None:
    previous = latest.with_name('previous')
    staged = latest.with_name('next_complete')
    if checkpoint_complete(latest):
        return
    if latest.exists():
        shutil.rmtree(latest)
    if checkpoint_complete(staged):
        staged.rename(latest)
    elif checkpoint_complete(previous):
        shutil.copytree(previous, latest)
    elif staged.exists():
        shutil.rmtree(staged)

def save_resume(path: Path, policy, optimizer, scheduler, next_prompt_index: int, order: Sequence[int], run_config_sha256: str, reward_weights: Mapping[str, float], weight_window_component_sums: Mapping[str, float], weight_window_response_count: int, reward_weight_updates: int) -> None:
    local = Path(os.environ['CPRL_CHECKPOINT_STAGE']) / 'latest'
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    legacy._original_save_resume(local, policy, optimizer, scheduler, next_prompt_index, order, run_config_sha256, reward_weights, weight_window_component_sums, weight_window_response_count, reward_weight_updates)
    if not checkpoint_complete(local):
        raise RuntimeError('local recovery checkpoint is incomplete')
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.with_name('previous')
    staged = path.with_name('next_complete')
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(local, staged)
    if not checkpoint_complete(staged):
        raise RuntimeError('Drive staging checkpoint is incomplete')
    if previous.exists():
        shutil.rmtree(previous)
    if path.exists():
        path.rename(previous)
    staged.rename(path)

def configure() -> None:
    output_root = Path(os.environ['CPRL_OUTPUT_ROOT'])
    patient_hash = os.environ['CPRL_PATIENT_SFT_SHA256']
    context_hash = os.environ['CPRL_FROZEN_COUNSELLOR_CONTEXTS_SHA256']
    legacy.IMPLEMENTATION_VERSION = 'CERTIFY_CBT_CACTUS_PATIENT_RLOO_LLAMA31_2002_V1_LUNA_CONTEXTS'
    legacy.MODEL_ID = MODEL_ID
    legacy.EXPECTED_BASE_MODEL_REVISION = MODEL_REVISION
    legacy.OUTPUT_ROOT = output_root
    legacy.PATIENT_SFT_ROOT = Path(os.environ['CPRL_PATIENT_SFT_DIR'])
    legacy.EXPECTED_PATIENT_SFT_SHA256 = patient_hash
    legacy.EXPECTED_COUNSELLOR_CONTEXTS_SHA256 = context_hash
    legacy.LEARNING_RATE = 1e-06
    legacy.NATURALNESS_LLM_WEIGHT = 0.8
    legacy.NATURALNESS_SFT_WEIGHT = 0.2
    legacy.INITIAL_REWARD_WEIGHTS = {'profile_consistency': 0.3, 'counsellor_action_dependence': 0.3, 'naturalness': 0.35, 'intended_resistance_realization': 0.05}
    legacy.REWARD_WEIGHT_BOUNDS = {'profile_consistency': (0.27, 0.33), 'counsellor_action_dependence': (0.27, 0.33), 'naturalness': (0.32, 0.38), 'intended_resistance_realization': (0.03, 0.07)}
    legacy.PATIENT_VISIBLE_KEYS = ALLOWED_VISIBLE_KEYS
    legacy.patient_prompt = patient_prompt
    legacy.resistance_input = resistance_input
    legacy.locate_patient_adapter = locate_patient_adapter
    legacy.load_policy_and_reference = load_policy_and_reference
    legacy.deterministic_penalty = deterministic_penalty
    legacy._original_save_resume = legacy.save_resume
    legacy.save_resume = save_resume
    if output_root.is_dir():
        for run in output_root.glob('run_*'):
            repair_checkpoint_pair(run / 'checkpoints' / 'latest')
if __name__ == '__main__':
    configure()
    legacy.main()
