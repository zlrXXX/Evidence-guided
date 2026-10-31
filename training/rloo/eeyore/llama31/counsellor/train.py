from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
BUNDLE = Path(os.environ['CRL5_BUNDLE_ROOT'])
sys.path.insert(0, str(BUNDLE))
from semantic_judge import OnlineCounsellorJudge
from reward_math import gated_resistance_reward
from checkpoint_binding import bind_sft, bind_patient_rloo
from prompt_runtime import counsellor_context as shown_counsellor_context, patient_context as shown_patient_context, action_candidates
from trajectory_runtime import FrozenEvidenceTracker, FrozenResistanceTracker, PROGRESS_POLICY, canonical, load_rag, parse_action, patient_context, progress_reward, response_context, selected_rag, sha256_file, validate_action
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MODEL_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
EXPECTED_COUNSELLOR_SFT_SHA256 = None
EXPECTED_PATIENT_RLOO_SHA256 = None
EXPECTED_PATIENT_RUN_COMPLETED_SHA256 = None
COMPATIBLE_BASELINE_MANIFEST_SHA256 = None
POLICY_ADAPTER_NAME = 'default'
REFERENCE_ADAPTER_NAME = 'reference'
SEED = 20260831
NUM_GENERATIONS = 4
TRAIN_PROMPTS = 3000
DEVELOPMENT_PROMPTS = 500
FIXED_DEVELOPMENT_PROMPTS = 64
PROMPTS_PER_OPTIMIZER_STEP = 4
TOTAL_OPTIMIZER_STEPS = TRAIN_PROMPTS // PROMPTS_PER_OPTIMIZER_STEP
VALIDATE_EVERY_STEPS = 250
LEARNING_RATE = 1e-06
KL_BETA = 0.05
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
MAX_PROMPT_TOKENS = 6144
ACTION_MAX_NEW_TOKENS = 144
RESPONSE_MAX_NEW_TOKENS = 192
PATIENT_MAX_NEW_TOKENS = 160
MIN_NEW_TOKENS = 4
TEMPERATURE = 0.8
TRAIN_TOP_P = 1.0
DEVELOPMENT_TOP_P = 0.9
TRAIN_TOP_K = 0
DEVELOPMENT_TOP_K = 50
SEMANTIC_WORKERS = int(os.environ.get('CRL5_LUNA_WORKERS', '4'))
MAX_LUNA_COST_USD = float(os.environ.get('CRL5_MAX_LUNA_COST_USD', '30.0'))
INITIAL_REWARD_WEIGHTS = {'framework_fidelity': 0.3, 'cbt_progress': 0.25, 'empathy': 0.2, 'naturalness': 0.1, 'resistance_transition': 0.15}
REWARD_WEIGHT_BOUNDS = {'framework_fidelity': (0.27, 0.33), 'cbt_progress': (0.21, 0.29), 'empathy': (0.17, 0.23), 'naturalness': (0.08, 0.12), 'resistance_transition': (0.12, 0.18)}
REWARD_WEIGHT_UPDATE_PROMPTS = 64
REWARD_WEIGHT_UPDATE_TRAJECTORIES = REWARD_WEIGHT_UPDATE_PROMPTS * NUM_GENERATIONS
FAIR_STABLE_STEP_SIZE = 0.1
MAX_REWARD_WEIGHT_DELTA_PER_UPDATE = 0.002
NATURALNESS_LLM_WEIGHT = 0.7
NATURALNESS_SFT_WEIGHT = 0.3
HARD_FAILURE_PENALTY = 1.0
ACTION_SYSTEM = "You are an expert in Cognitive Behavior Therapy focused mental health counselling. Use only the provided patient profile, supported session information, current provisional resistance state, persistent CBT progress, completed techniques, verified prior patient evidence, preceding counselling dialogue, active official CBT technique, and retrieved official CBT options. Treat the provisional resistance state as contextual guidance, not verified patient evidence or a mandatory action rule. The supplied legal_action_candidates are the complete procedurally legal action space for the current state. Choose exactly one supplied candidate. In every candidate, step_id is the official CBT step that the upcoming counsellor response will perform: the current step means remain there, while the immediate next step means advance. The active-step progress state is derived from the persistent CBT progress map before this decision. Advancement candidates are supplied only after guided or independent application; completion candidates are supplied only after independent application of the final step. Preserve the active therapeutic target, official technique step order, and saved technique position. Use CHECK_PATIENT_EVIDENCE only when evidence genuinely required by the active step is missing or needs verification. Switch techniques only when an official stop-or-switch condition is supported by the dialogue and current target. A pending off-session task may support pausing or ending the current session, but it cannot satisfy a completion condition requiring the task's later execution or outcome. Do not treat an untried activity, behavioural experiment, or plan as completed. Use COMPLETE_TECHNIQUE only when supported patient evidence satisfies the official final-step requirement. Completing a technique does not automatically end the session. Choose END_SESSION only when the dialogue supports that enough has been done for today. END_SESSION is an ending proposal rather than the final counsellor closing. Technique completion and session ending must each be independently justified. Return only the selected compact JSON object with exactly these keys in this order: action, delivery_mode, retrieve_or_reuse, technique_id, step_id, and session_control. Copy one supplied candidate exactly; do not combine values from different candidates. Do not output placeholders, explanations, markdown, metadata, hidden reasoning, or additional keys."
RESPONSE_SYSTEM = "You are an expert in Cognitive Behavior Therapy focused mental health counselling. Output only the next counsellor utterance. Follow the selected legal decision and selected official CBT step exactly. Apply the official step guidance and its evidence requirement naturally rather than naming or reciting them. Respond mainly to the latest patient utterance with natural warmth and make one coherent move. Use relevant verified prior patient evidence rather than asking the patient to repeat it. When a genuinely missing part of the selected step remains, ask at most one focused question about only that part; otherwise acknowledge, reflect, explain, guide, or consolidate. When switching techniques, introduce only the high-level new direction and why it may fit; leave the new technique's first procedural step for the next counsellor turn. Do not ask the patient to simulate elapsed time, an external action, or another person's later response; keep off-session tasks untried. When the selected decision proposes ending the session, make a collaborative ending proposal rather than a final closing because the patient has not yet responded. Avoid lecturing, persuasion, cross-examination, over-summary, mechanical paraphrasing, repeated information seeking, question checklists, unsupported facts, diagnosis, promises, or clinical-effectiveness claims. Output only the natural counsellor utterance, with no JSON, identifiers, labels, metadata, stage directions, or hidden reasoning."
PATIENT_SYSTEM = "You are a client attending counselling. Use the preceding dialogue, patient profile, symptom level, and provisional engagement cue. Output only the next natural patient utterance. Respond to the counsellor's latest utterance while maintaining continuity with the preceding counselling history. Stay profile-consistent and do not invent facts. Express the supplied engagement cue through openness, uncertainty, or disagreement without naming a state. Use natural first-person speech. When the counsellor asks more than one related question, weave the relevant parts into one natural response rather than replying as a numbered or question-by-question list. When supported, include one relevant thought, feeling, worry, bodily reaction, example, preference, uncertainty, or barrier; do not force detail. You may hesitate, correct yourself, disagree, or answer partly when appropriate. If an answered question is repeated, do not invent novelty or simply agree; clarify, correct, identify a barrier, or say there is nothing more to add. Do not claim that elapsed time passed, an external task was completed, or another person responded within one generated turn; keep such agreed tasks untried. Avoid copying the counsellor, repeated replies, profile summaries, canned therapy language, sudden insight or compliance, tidy conclusions, and clinical explanations. Do not output an ending token or control label. Do not output speaker or state labels, JSON, metadata, stage directions, hidden reasoning, clinical claims, or evaluator commentary."

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()

def directory_contract_sha256(path: Path) -> str:
    rows = [[str(item.relative_to(path)).replace('\\', '/'), item.stat().st_size, sha256_file(item)] for item in sorted(path.rglob('*')) if item.is_file()]
    if not rows:
        raise FileNotFoundError(f'artifact directory is empty: {path}')
    return sha256_bytes(canonical(rows).encode('utf-8'))

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(dict(value)) + '\n')
        handle.flush()

def rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(canonical(dict(row)) + '\n')

def validate_bundle() -> dict[str, Any]:
    manifest_path = BUNDLE / 'bundle_manifest.json'
    completed = read_json(BUNDLE / 'COMPLETED.json')
    if completed.get('bundle_manifest_sha256') != sha256_file(manifest_path):
        raise ValueError('bundle completion seal mismatch')
    manifest = read_json(manifest_path)
    for relative, expected in manifest['members'].items():
        path = BUNDLE / relative
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256_file(path) != expected['sha256']:
            raise ValueError(f'bundle member mismatch: {relative}')
    return manifest

def validate_hardware() -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is required')
    properties = torch.cuda.get_device_properties(0)
    gib = properties.total_memory / 1024 ** 3
    supported_a100 = 'A100' in properties.name and gib >= 75
    supported_g4 = 'RTX PRO 6000' in properties.name and 'Blackwell' in properties.name and (gib >= 90)
    if not (supported_a100 or supported_g4):
        raise RuntimeError(f'A100 80GB or Colab G4 RTX PRO 6000 Blackwell 96GB required; found {properties.name} ({gib:.1f} GiB)')
    return {'name': properties.name, 'memory_gib': gib, 'bf16': True}

def locate_adapter(path: Path, expected_sha: str, role: str) -> Path:
    required = [path / 'adapter_model.safetensors', path / 'adapter_config.json']
    if any((not item.is_file() for item in required)):
        raise FileNotFoundError(f'missing {role} adapter files under {path}')
    if sha256_file(required[0]) != expected_sha:
        raise ValueError(f'{role} adapter hash is not the frozen selected checkpoint')
    if read_json(required[1]).get('base_model_name_or_path') != MODEL_ID:
        raise ValueError(f'{role} adapter base model mismatch')
    return path

def locate_patient_adapter(run_root: Path) -> Path:
    global EXPECTED_PATIENT_RLOO_SHA256
    expected = read_json(BUNDLE / 'contracts' / 'checkpoint_bindings.json')
    adapter, EXPECTED_PATIENT_RLOO_SHA256 = bind_patient_rloo(run_root, expected['patient_rloo_manifest_sha256'])
    return adapter

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get('CRL5_MODEL_LOAD_PATH', MODEL_ID), revision=MODEL_REVISION, token=os.environ.get('HF_TOKEN') or None, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer

def load_base_model():
    base = AutoModelForCausalLM.from_pretrained(os.environ.get('CRL5_MODEL_LOAD_PATH', MODEL_ID), revision=MODEL_REVISION, token=os.environ.get('HF_TOKEN') or None, torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    base.config.use_cache = False
    return base

def zero_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

def adapter_parameter(name: str, adapter_name: str) -> bool:
    return adapter_name in name.split('.')

def activate_counsellor_adapter(model, adapter_name: str, trainable_forward: bool) -> None:
    if adapter_name not in {POLICY_ADAPTER_NAME, REFERENCE_ADAPTER_NAME}:
        raise ValueError(f'unknown counsellor adapter: {adapter_name}')
    if adapter_name not in model.peft_config:
        raise RuntimeError(f'counsellor adapter is not loaded: {adapter_name}')
    model.set_adapter(adapter_name)
    expected_trainable = trainable_forward and adapter_name == POLICY_ADAPTER_NAME
    trainable_count = 0
    for name, parameter in model.named_parameters():
        should_train = expected_trainable and adapter_parameter(name, POLICY_ADAPTER_NAME)
        parameter.requires_grad_(should_train)
        trainable_count += int(should_train)
    model.train(trainable_forward and adapter_name == POLICY_ADAPTER_NAME)
    active = model.active_adapter
    if isinstance(active, (list, tuple)):
        active = active[0] if len(active) == 1 else active
    if active != adapter_name:
        raise RuntimeError(f'adapter activation failed: expected {adapter_name}, found {active}')
    if expected_trainable and (not trainable_count):
        raise RuntimeError('policy adapter activation found no trainable parameters')

def adapter_tensor_sha256(model, adapter_name: str) -> str:
    digest = hashlib.sha256()
    found = 0
    for name, parameter in sorted(model.named_parameters()):
        if not adapter_parameter(name, adapter_name):
            continue
        found += 1
        value = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode('utf-8') + b'\x00' + value)
    if not found:
        raise RuntimeError(f'no parameters found for adapter {adapter_name}')
    return digest.hexdigest().upper()

def adapter_parameter_map(model, adapter_name: str) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if adapter_parameter(name, adapter_name):
            normalized = '.'.join(('<adapter>' if part == adapter_name else part for part in name.split('.')))
            output[normalized] = parameter
    if not output:
        raise RuntimeError(f'no parameters found for adapter {adapter_name}')
    return output

def initialize_fresh_reference_from_policy(model) -> None:
    policy_values = adapter_parameter_map(model, POLICY_ADAPTER_NAME)
    reference_values = adapter_parameter_map(model, REFERENCE_ADAPTER_NAME)
    if policy_values.keys() != reference_values.keys() or any((policy_values[name].shape != reference_values[name].shape for name in policy_values)):
        raise RuntimeError('fresh policy/reference adapter structures differ')
    with torch.no_grad():
        for name, source in policy_values.items():
            reference_values[name].data = source.detach().clone()

def enable_policy_gradient_checkpointing(model) -> None:
    base_model = model.get_base_model()
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base_model.enable_input_require_grads()
    base_model.config.use_cache = False
    if not base_model.is_gradient_checkpointing:
        raise RuntimeError('policy gradient checkpointing failed to activate')

def validate_shared_counsellor(model, tokenizer, identical_start_adapters: bool) -> None:
    if set(model.peft_config) != {POLICY_ADAPTER_NAME, REFERENCE_ADAPTER_NAME}:
        raise RuntimeError('shared counsellor does not contain exactly policy and reference adapters')
    if identical_start_adapters:
        policy_values = adapter_parameter_map(model, POLICY_ADAPTER_NAME)
        reference_values = adapter_parameter_map(model, REFERENCE_ADAPTER_NAME)
        if policy_values.keys() != reference_values.keys() or any((not torch.equal(policy_values[name], reference_values[name]) for name in policy_values)):
            raise RuntimeError('fresh policy and frozen reference adapters are not identical')
        del policy_values, reference_values
    probe = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device='cuda')
    activate_counsellor_adapter(model, REFERENCE_ADAPTER_NAME, trainable_forward=False)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        reference_first = model(input_ids=probe).logits[:, -1].float().cpu()
    activate_counsellor_adapter(model, POLICY_ADAPTER_NAME, trainable_forward=False)
    activate_counsellor_adapter(model, REFERENCE_ADAPTER_NAME, trainable_forward=False)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        reference_second = model(input_ids=probe).logits[:, -1].float().cpu()
    if not torch.equal(reference_first, reference_second):
        raise RuntimeError('reference adapter output changed after adapter switching')
    del reference_first, reference_second, probe
    activate_counsellor_adapter(model, POLICY_ADAPTER_NAME, trainable_forward=True)

def load_shared_counsellor(policy_adapter: Path, reference_adapter: Path, initialize_identical_reference: bool):
    model = PeftModel.from_pretrained(load_base_model(), policy_adapter, adapter_name=POLICY_ADAPTER_NAME, is_trainable=True)
    model.load_adapter(reference_adapter, adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
    if initialize_identical_reference:
        initialize_fresh_reference_from_policy(model)
    model = model.to('cuda')
    zero_dropout(model)
    enable_policy_gradient_checkpointing(model)
    activate_counsellor_adapter(model, POLICY_ADAPTER_NAME, trainable_forward=True)
    return model

def load_frozen_peft(adapter: Path):
    model = PeftModel.from_pretrained(load_base_model(), adapter, is_trainable=False).to('cuda')
    zero_dropout(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

def save_policy_adapter(model, directory: Path) -> None:
    activate_counsellor_adapter(model, POLICY_ADAPTER_NAME, trainable_forward=False)
    model.save_pretrained(directory, safe_serialization=True, selected_adapters=[POLICY_ADAPTER_NAME])
    if not (directory / 'adapter_model.safetensors').is_file():
        raise RuntimeError('policy adapter was not saved at the resumable root path')

def messages_prompt(tokenizer, system: str, prefix: str, payload: Mapping[str, Any]) -> tuple[str, int]:
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prefix + '\n' + canonical(payload)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if len(tokens) > MAX_PROMPT_TOKENS:
        raise ValueError(f'prompt has {len(tokens)} tokens, exceeding {MAX_PROMPT_TOKENS}; silent truncation is forbidden')
    return (prompt, len(tokens))

def exact_suffix(sequence: Sequence[int], prompt_length: int, eos: int, maximum: int) -> list[int]:
    suffix = [int(value) for value in sequence[prompt_length:]]
    if eos in suffix[:MIN_NEW_TOKENS]:
        raise ValueError('sampled EOS violates minimum token count')
    if eos in suffix:
        suffix = suffix[:suffix.index(eos) + 1]
    if not suffix or len(suffix) > maximum:
        raise ValueError('invalid sampled completion length')
    return suffix

@torch.inference_mode()
def generate_group(model, tokenizer, prompt: str, seed: int, maximum: int, top_p: float, top_k: int) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    encoded = {key: value.cuda() for key, value in encoded.items()}
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        generated = model.generate(**encoded, do_sample=True, temperature=TEMPERATURE, top_p=top_p, top_k=top_k, num_return_sequences=NUM_GENERATIONS, min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=maximum, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    prompt_length = int(encoded['input_ids'].shape[1])
    output = []
    for sequence in generated:
        token_ids = exact_suffix(sequence.tolist(), prompt_length, tokenizer.eos_token_id, maximum)
        output.append({'text': tokenizer.decode(token_ids, skip_special_tokens=True).strip(), 'token_ids': token_ids})
    return output

@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, seed: int, maximum: int, top_p: float, top_k: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    encoded = {key: value.cuda() for key, value in encoded.items()}
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        generated = model.generate(**encoded, do_sample=True, temperature=TEMPERATURE, top_p=top_p, top_k=top_k, min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=maximum, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    prompt_length = int(encoded['input_ids'].shape[1])
    token_ids = exact_suffix(generated[0].tolist(), prompt_length, tokenizer.eos_token_id, maximum)
    return {'text': tokenizer.decode(token_ids, skip_special_tokens=True).strip(), 'token_ids': token_ids}

@torch.inference_mode()
def generate_batch_one(model, tokenizer, prompts: Sequence[str], seed: int, maximum: int, top_p: float, top_k: int) -> list[dict[str, Any]]:
    if not prompts:
        return []
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(list(prompts), add_special_tokens=False, padding=True, return_tensors='pt')
    encoded = {key: value.cuda() for key, value in encoded.items()}
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        generated = model.generate(**encoded, do_sample=True, temperature=TEMPERATURE, top_p=top_p, top_k=top_k, min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=maximum, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    padded_prompt_length = int(encoded['input_ids'].shape[1])
    output = []
    for sequence in generated:
        token_ids = exact_suffix(sequence.tolist(), padded_prompt_length, tokenizer.eos_token_id, maximum)
        output.append({'text': tokenizer.decode(token_ids, skip_special_tokens=True).strip(), 'token_ids': token_ids})
    return output

def conditional_logps(model: nn.Module, tokenizer, prompt_texts: Sequence[str], completions: Sequence[Sequence[int]], require_grad: bool, counsellor_adapter: str | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    if counsellor_adapter is not None:
        activate_counsellor_adapter(model, counsellor_adapter, trainable_forward=require_grad)
    sequences, masks = ([], [])
    for prompt, target_values in zip(prompt_texts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target = list(target_values)
        if not target:
            raise ValueError('empty sampled completion')
        sequences.append(prompt_ids + target)
        masks.append([False] * (len(prompt_ids) - 1) + [True] * len(target))
    maximum = max(map(len, sequences))
    input_ids = torch.tensor([row + [tokenizer.pad_token_id] * (maximum - len(row)) for row in sequences], dtype=torch.long, device='cuda')
    attention = torch.tensor([[1] * len(row) + [0] * (maximum - len(row)) for row in sequences], dtype=torch.long, device='cuda')
    token_mask = torch.tensor([row + [False] * (maximum - 1 - len(row)) for row in masks], dtype=torch.bool, device='cuda')
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context, torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1]
    labels = input_ids[:, 1:]
    completion_positions = token_mask.long().cumsum(-1)
    selected_logits = logits[token_mask].float() / TEMPERATURE
    selected_labels = labels[token_mask]
    selected_suppress_eos = (completion_positions <= MIN_NEW_TOKENS)[token_mask]
    if selected_suppress_eos.any():
        selected_logits = selected_logits.clone()
        selected_logits[selected_suppress_eos, tokenizer.eos_token_id] = float('-inf')
    selected_logps = selected_logits.log_softmax(-1).gather(-1, selected_labels.unsqueeze(-1)).squeeze(-1)
    row_ids = torch.arange(input_ids.shape[0], device='cuda').unsqueeze(1).expand_as(token_mask)[token_mask]
    sequence_logps = torch.zeros(input_ids.shape[0], dtype=torch.float32, device='cuda')
    sequence_logps.scatter_add_(0, row_ids, selected_logps)
    return (sequence_logps, token_mask.sum(-1).clamp_min(1))

def joint_logps(model, tokenizer, item: Mapping[str, Any], require_grad: bool, counsellor_adapter: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_logps, action_lengths = conditional_logps(model, tokenizer, [item['action_prompt']] * NUM_GENERATIONS, [sample['token_ids'] for sample in item['action_samples']], require_grad, counsellor_adapter)
    joint = action_logps.clone()
    response_values = torch.zeros(NUM_GENERATIONS, device='cuda', dtype=torch.float32)
    response_lengths = torch.zeros(NUM_GENERATIONS, device='cuda', dtype=torch.long)
    valid = [index for index, sample in enumerate(item['response_samples']) if sample is not None]
    if valid:
        response_logps, lengths = conditional_logps(model, tokenizer, [item['response_prompts'][index] for index in valid], [item['response_samples'][index]['token_ids'] for index in valid], require_grad, counsellor_adapter)
        for local, original in enumerate(valid):
            joint[original] = joint[original] + response_logps[local]
            response_values[original] = response_logps[local]
            response_lengths[original] = lengths[local]
    return (joint, action_lengths + response_lengths, response_values, response_lengths)

def sample_counsellor_group(counsellor, counsellor_adapter: str, tokenizer, row: Mapping[str, Any], rag: Mapping[str, Any], seed: int, top_p: float, top_k: int) -> dict[str, Any]:
    activate_counsellor_adapter(counsellor, counsellor_adapter, trainable_forward=False)
    visible = row['counsellor_visible']
    action_context = shown_counsellor_context(visible, rloo=True)
    action_context['legal_action_candidates'] = action_candidates(visible, rag)
    action_prompt, action_prompt_tokens = messages_prompt(tokenizer, ACTION_SYSTEM, 'VISIBLE_COUNSELLOR_ACTION_CONTEXT=', action_context)
    action_samples = generate_group(counsellor, tokenizer, action_prompt, seed, ACTION_MAX_NEW_TOKENS, top_p, top_k)
    decisions: list[dict[str, Any]] = []
    legality: list[dict[str, Any]] = []
    response_prompts: list[str | None] = []
    response_samples: list[dict[str, Any] | None] = [None] * NUM_GENERATIONS
    valid_response_indices: list[int] = []
    for index, sample in enumerate(action_samples):
        try:
            decision = parse_action(sample['text'])
            validate_action(decision, visible, rag)
            legal = {'valid': True, 'reason': 'LEGAL'}
            context = response_context(visible, decision, rag)
            response_prompt, _ = messages_prompt(tokenizer, RESPONSE_SYSTEM, 'VISIBLE_COUNSELLOR_RESPONSE_CONTEXT=', shown_counsellor_context(context, response=True, rloo=True))
            response = None
            valid_response_indices.append(index)
        except Exception as error:
            decision = {'invalid_action_text': sample['text']}
            legal = {'valid': False, 'reason': f'{type(error).__name__}: {error}'}
            response_prompt, response = (None, None)
        decisions.append(decision)
        legality.append(legal)
        response_prompts.append(response_prompt)
    if valid_response_indices:
        generated_responses = generate_batch_one(counsellor, tokenizer, [response_prompts[index] for index in valid_response_indices], seed + 10000, RESPONSE_MAX_NEW_TOKENS, top_p, top_k)
        for index, response in zip(valid_response_indices, generated_responses):
            if not response['text']:
                legality[index] = {'valid': False, 'reason': 'empty counsellor response'}
                continue
            response_samples[index] = response
    return {'row': row, 'action_prompt': action_prompt, 'action_prompt_tokens': action_prompt_tokens, 'action_samples': action_samples, 'decisions': decisions, 'local_legality': legality, 'response_prompts': response_prompts, 'response_samples': response_samples, 'utterances': [sample['text'] if sample is not None else '' for sample in response_samples]}
ROLE_BREAK = re.compile('\\b(as an ai|language model|clinical effectiveness|metadata|hidden reasoning)\\b', re.I)

def local_trajectory_rewards(item: Mapping[str, Any], counsellor, patient, tokenizer, resistance_tracker: FrozenResistanceTracker, evidence_tracker: FrozenEvidenceTracker, rag: Mapping[str, Any], top_p: float, top_k: int, seed: int) -> tuple[list[dict[str, Any]], torch.Tensor]:
    visible = item['row']['counsellor_visible']
    reference_joint, _, response_logps, response_lengths = joint_logps(counsellor, tokenizer, item, require_grad=False, counsellor_adapter=REFERENCE_ADAPTER_NAME)
    records: list[dict[str, Any]] = [{'framework_fidelity': -1.0, 'cbt_progress': 0.0, 'empathy': 0.0, 'naturalness_reference_raw': -20.0, 'resistance_transition_raw': 0.0, 'hard_penalty': HARD_FAILURE_PENALTY, 'penalty_reasons': ['ILLEGAL_ACTION'], 'patient_response': '', 'tracker': None, 'evidence': None, 'progress': None} for _ in range(NUM_GENERATIONS)]
    valid = [index for index, legal in enumerate(item['local_legality']) if legal['valid']]
    if not valid:
        return (records, reference_joint.detach())
    tracker_rows = []
    for index in valid:
        decision = item['decisions'][index]
        utterance = item['utterances'][index]
        active = visible.get('active_cbt_rag')
        tracker_rows.append({'current_resistance_state': visible['current_resistance_state'], 'active_technique_id': decision['technique_id'] if decision else active['technique_id'] if active else 'NONE', 'active_step_id': decision['step_id'] if decision else active['current_step']['step_id'] if active else 'NONE', 'counsellor_utterance': utterance, 'patient_profile': visible['patient_profile'], 'counselling_progress': visible['counselling_progress'], 'dialogue_history': visible['dialogue_history']})
    tracker_predictions = resistance_tracker.predict(tracker_rows)
    evidence_rows = []
    generated_patients: dict[int, str] = {}
    patient_prompts: list[str] = []
    for local, index in enumerate(valid):
        prediction = tracker_predictions[local]
        p_context = patient_context(visible, item['utterances'][index], prediction['next_state'])
        patient_prompt, _ = messages_prompt(tokenizer, PATIENT_SYSTEM, 'VISIBLE_PATIENT_RUNTIME_CONTEXT=', shown_patient_context(p_context, rloo=True))
        patient_prompts.append(patient_prompt)
    patient_samples = generate_batch_one(patient, tokenizer, patient_prompts, seed + 20000, PATIENT_MAX_NEW_TOKENS, top_p, top_k)
    for local, index in enumerate(valid):
        patient_sample = patient_samples[local]
        generated_patients[index] = patient_sample['text']
        decision = item['decisions'][index]
        step = rag[decision['technique_id']]['steps'][decision['step_id']]
        evidence_rows.append({'technique_id': decision['technique_id'], 'step_id': decision['step_id'], 'evidence_requirement': step['evidence_requirement'], 'counsellor_utterance': item['utterances'][index], 'patient_response': patient_sample['text']})
    evidence_predictions = evidence_tracker.predict(evidence_rows)
    for local, index in enumerate(valid):
        decision = item['decisions'][index]
        utterance = item['utterances'][index]
        progress = progress_reward(visible.get('cbt_progress_by_step', {}), decision, evidence_predictions[local])
        penalties: list[str] = []
        if ROLE_BREAK.search(utterance):
            penalties.append('ROLE_OR_METADATA_BREAK')
        if len(utterance.split()) < 5:
            penalties.append('EMPTY_OR_FRAGMENTARY_RESPONSE')
        penalty = HARD_FAILURE_PENALTY if penalties else 0.0
        reference_mean_logp = float(response_logps[index] / response_lengths[index].clamp_min(1))
        records[index] = {'framework_fidelity': 0.0, 'cbt_progress': float(progress['cbt_progress_reward']), 'empathy': 0.0, 'naturalness_reference_raw': reference_mean_logp, 'resistance_transition_raw': float(tracker_predictions[local]['expected_state_aware_utility']), 'hard_penalty': penalty, 'penalty_reasons': penalties, 'patient_response': generated_patients[index], 'tracker': tracker_predictions[local], 'evidence': evidence_predictions[local], 'progress': progress}
    return (records, reference_joint.detach())

def bounded_simplex_projection(values: Mapping[str, float]) -> dict[str, float]:
    keys = list(INITIAL_REWARD_WEIGHTS)
    lower = min((values[key] - REWARD_WEIGHT_BOUNDS[key][1] for key in keys))
    upper = max((values[key] - REWARD_WEIGHT_BOUNDS[key][0] for key in keys))
    for _ in range(80):
        offset = (lower + upper) / 2
        projected = {key: min(REWARD_WEIGHT_BOUNDS[key][1], max(REWARD_WEIGHT_BOUNDS[key][0], values[key] - offset)) for key in keys}
        if sum(projected.values()) > 1:
            lower = offset
        else:
            upper = offset
    return {key: min(REWARD_WEIGHT_BOUNDS[key][1], max(REWARD_WEIGHT_BOUNDS[key][0], values[key] - (lower + upper) / 2)) for key in keys}

def update_reward_weights(previous: Mapping[str, float], means: Mapping[str, float]) -> dict[str, float]:
    raw = {key: previous[key] * math.exp(-FAIR_STABLE_STEP_SIZE * math.tanh(float(means[key]))) for key in previous}
    total = sum(raw.values())
    bounded = bounded_simplex_projection({key: value / total for key, value in raw.items()})
    largest = max((abs(bounded[key] - previous[key]) for key in previous))
    if largest <= MAX_REWARD_WEIGHT_DELTA_PER_UPDATE:
        return bounded
    scale = MAX_REWARD_WEIGHT_DELTA_PER_UPDATE / largest
    return {key: previous[key] + scale * (bounded[key] - previous[key]) for key in previous}

def leave_one_out(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - (rewards.sum() - rewards) / (NUM_GENERATIONS - 1)

def fuse_rewards(local: Sequence[Mapping[str, Any]], semantic: Sequence[Mapping[str, Any]], weights: Mapping[str, float], calibration: Mapping[str, float]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records, totals = ([], [])
    for base, judge in zip(local, semantic):
        framework_reward = float(judge['framework_reward']) if base['tracker'] is not None else -1.0
        empathy_reward = float(judge['empathy_reward']) if base['tracker'] is not None else -1.0
        resistance_reward, resistance_gate = gated_resistance_reward(float(base['resistance_transition_raw']), framework_reward, empathy_reward)
        reference_naturalness = math.tanh((float(base['naturalness_reference_raw']) - calibration['mean']) / calibration['standard_deviation'])
        components = {'framework_fidelity': framework_reward, 'cbt_progress': float(base['cbt_progress']), 'empathy': empathy_reward, 'naturalness': NATURALNESS_LLM_WEIGHT * float(judge['naturalness_reward']) + NATURALNESS_SFT_WEIGHT * reference_naturalness if base['tracker'] is not None else -1.0, 'resistance_transition': resistance_reward}
        penalty = float(base['hard_penalty'])
        reasons = list(base['penalty_reasons'])
        if judge['hard_failure']:
            penalty = max(penalty, HARD_FAILURE_PENALTY)
            reasons.append('LUNA_HARD_FAILURE')
        total = max(-1.5, min(1.5, sum((weights[key] * value for key, value in components.items())) - penalty))
        records.append({'components': components, 'total_reward': total, 'hard_penalty': penalty, 'penalty_reasons': sorted(set(reasons)), 'naturalness_fusion': {'llm_weight': NATURALNESS_LLM_WEIGHT, 'sft_alignment_weight': NATURALNESS_SFT_WEIGHT, 'sft_alignment_reward': reference_naturalness}, 'resistance_transition_gate': {'raw_state_aware_utility': float(base['resistance_transition_raw']), 'positive_credit_gate': resistance_gate, 'final_reward': resistance_reward, 'negative_utility_is_ungated': True}, 'semantic': dict(judge), 'patient_response': base['patient_response'], 'tracker': base['tracker'], 'evidence': base['evidence'], 'progress': base['progress']})
        totals.append(total)
    return (records, torch.tensor(totals, dtype=torch.float32, device='cuda'))

def reward_calibration(local_groups: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    values = [float(record['naturalness_reference_raw']) for group in local_groups for record in group if record['tracker'] is not None]
    if len(values) < 32:
        raise RuntimeError('too few valid SFT trajectories for naturalness calibration')
    standard_deviation = max(0.1, float(np.std(values)))
    return {'mean': float(np.mean(values)), 'standard_deviation': standard_deviation, 'rows': len(values)}

def run_groups(counsellor, generation_adapter: str, patient, tokenizer, rows: Sequence[Mapping[str, Any]], judge: OnlineCounsellorJudge, resistance_tracker, evidence_tracker, rag, phase: str, position_offset: int, calibration: Mapping[str, float] | None, weights: Mapping[str, float], top_p: float, top_k: int) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
    pending: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], torch.Tensor]] = []
    for start in range(0, len(rows), SEMANTIC_WORKERS):
        batch = rows[start:start + SEMANTIC_WORKERS]
        generated = [sample_counsellor_group(counsellor, generation_adapter, tokenizer, row, rag, SEED * 1000000 + start + index, top_p, top_k) for index, row in enumerate(batch)]
        for index, item in enumerate(generated):
            item['prompt_position'] = position_offset + start + index
        semantic_job = judge.submit_step(generated, phase=phase)
        local_groups, reference_logps = ([], [])
        try:
            for index, item in enumerate(generated):
                local, reference_joint = local_trajectory_rewards(item, counsellor, patient, tokenizer, resistance_tracker, evidence_tracker, rag, top_p, top_k, SEED * 2000000 + start + index)
                local_groups.append(local)
                reference_logps.append(reference_joint)
            semantic_groups = judge.finish_step(semantic_job)
        except BaseException:
            try:
                judge.finish_step(semantic_job)
            except BaseException:
                pass
            raise
        for item, local, semantic, reference_joint in zip(generated, local_groups, semantic_groups, reference_logps):
            pending.append((item, local, semantic, reference_joint))
    if calibration is None:
        calibration = reward_calibration([local for _, local, _, _ in pending])
    all_logs: list[dict[str, Any]] = []
    for item, local, semantic, reference_joint in pending:
        reward_records, rewards = fuse_rewards(local, semantic, weights, calibration)
        all_logs.append({'item': item, 'rewards': reward_records, 'task_rewards': rewards, 'reference_joint_logps': reference_joint})
    return (all_logs, dict(calibration) if calibration is not None else None)

def summarize(logs: Sequence[Mapping[str, Any]], weights: Mapping[str, float]) -> dict[str, Any]:
    rewards = [record for group in logs for record in group['rewards']]
    return {'prompt_groups': len(logs), 'trajectories': len(rewards), 'mean_total_reward': float(np.mean([record['total_reward'] for record in rewards])), 'component_means': {key: float(np.mean([record['components'][key] for record in rewards])) for key in INITIAL_REWARD_WEIGHTS}, 'hard_failure_rate': float(np.mean([record['hard_penalty'] > 0 for record in rewards])), 'selection_weights': dict(weights)}

def baseline_compatible_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    comparable = dict(value)
    for key in ('bundle_manifest_sha256', 'implementation', 'runtime', 'hardware', 'baseline_source_binding', 'artifact_identity'):
        comparable.pop(key, None)
    return comparable

def validate_baseline_payload(baseline: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    if baseline.get('prompt_groups') != FIXED_DEVELOPMENT_PROMPTS:
        raise ValueError('frozen SFT baseline prompt count mismatch')
    if baseline.get('trajectories') != FIXED_DEVELOPMENT_PROMPTS * NUM_GENERATIONS:
        raise ValueError('frozen SFT baseline trajectory count mismatch')
    if set(baseline.get('component_means', {})) != set(INITIAL_REWARD_WEIGHTS):
        raise ValueError('frozen SFT baseline reward components mismatch')
    if baseline.get('selection_weights') != INITIAL_REWARD_WEIGHTS:
        raise ValueError('frozen SFT baseline selection weights mismatch')
    finite_values = [baseline.get('mean_total_reward'), baseline.get('hard_failure_rate'), calibration.get('mean'), calibration.get('standard_deviation')] + list(baseline['component_means'].values())
    if any((not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in finite_values)):
        raise ValueError('frozen SFT baseline contains non-finite metrics')
    if not 0.0 <= float(baseline['hard_failure_rate']) <= 1.0:
        raise ValueError('frozen SFT baseline hard-failure rate is invalid')
    if int(calibration.get('rows', 0)) < 32 or float(calibration['standard_deviation']) < 0.1:
        raise ValueError('frozen SFT naturalness calibration is invalid')

def current_sft_baseline(output, run_contract, counsellor, patient, tokenizer, fixed_rows, judge, resistance_tracker, evidence_tracker, rag):
    path = output / 'SFT_BASELINE_STATE.json'
    contract_sha = sha256_bytes(canonical(run_contract).encode('utf-8'))
    if path.is_file():
        state = read_json(path)
        if state.get('run_config_sha256') != contract_sha:
            raise ValueError('SFT baseline belongs to a different prompt/checkpoint configuration')
        baseline, calibration = (state['baseline'], state['calibration'])
    else:
        if (output / 'resume_latest' / 'state.pt').is_file() or (output / 'TRAINING_TRAJECTORIES.jsonl').is_file():
            raise ValueError('Cannot resume without the bound initial SFT baseline')
        logs, calibration = run_groups(counsellor, REFERENCE_ADAPTER_NAME, patient, tokenizer, fixed_rows, judge, resistance_tracker, evidence_tracker, rag, 'sft_baseline', 0, None, INITIAL_REWARD_WEIGHTS, DEVELOPMENT_TOP_P, DEVELOPMENT_TOP_K)
        baseline = summarize(logs, INITIAL_REWARD_WEIGHTS)
        validate_baseline_payload(baseline, calibration)
        write_json(path, {'run_config_sha256': contract_sha, 'baseline': baseline, 'calibration': calibration})
    validate_baseline_payload(baseline, calibration)
    write_json(output / 'SFT_BASELINE.json', baseline)
    write_json(output / 'NATURALNESS_CALIBRATION.json', calibration)
    return (baseline, calibration)

def save_resume(resume: Path, policy, optimizer, scheduler, next_prompt: int, order: Sequence[int], weights: Mapping[str, float], sums: Mapping[str, float], count: int, config_sha: str, parameter_signature: str) -> None:
    staging = resume.with_name(resume.name + '.staging')
    previous = resume.with_name(resume.name + '.previous')
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'adapter').mkdir(parents=True)
    save_policy_adapter(policy, staging / 'adapter')
    adapter_sha = sha256_file(staging / 'adapter' / 'adapter_model.safetensors')
    torch.save({'next_prompt': next_prompt, 'order': list(order), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'reward_weights': dict(weights), 'window_sums': dict(sums), 'window_count': count, 'config_sha256': config_sha, 'policy_adapter_sha256': adapter_sha, 'policy_parameter_signature': parameter_signature}, staging / 'state.pt')
    if previous.exists():
        shutil.rmtree(previous)
    if resume.exists():
        resume.rename(previous)
    staging.rename(resume)
    if previous.exists():
        shutil.rmtree(previous)

def policy_parameter_signature(model) -> str:
    rows = [[name, list(parameter.shape), str(parameter.dtype)] for name, parameter in model.named_parameters() if adapter_parameter(name, POLICY_ADAPTER_NAME)]
    if not rows:
        raise RuntimeError('policy adapter parameter signature is empty')
    return sha256_bytes(canonical(rows).encode('utf-8'))

def restore_interrupted_resume(output: Path) -> Path:
    resume = output / 'resume_latest'
    previous = output / 'resume_latest.previous'
    required = (Path('state.pt'), Path('adapter/adapter_model.safetensors'), Path('adapter/adapter_config.json'))
    current_complete = all(((resume / item).is_file() for item in required))
    previous_complete = all(((previous / item).is_file() for item in required))
    if current_complete:
        if previous.exists():
            shutil.rmtree(previous)
        return resume
    if resume.exists():
        shutil.rmtree(resume)
    if previous_complete:
        previous.rename(resume)
        return resume
    if previous.exists():
        raise RuntimeError('both current and previous resume checkpoints are incomplete')
    return resume

def validate_resume_state(state: Mapping[str, Any], resume: Path, config_sha: str, order: Sequence[int], parameter_signature: str) -> None:
    if state.get('config_sha256') != config_sha or state.get('order') != list(order):
        raise RuntimeError('resume belongs to another frozen run')
    adapter_path = resume / 'adapter' / 'adapter_model.safetensors'
    if state.get('policy_adapter_sha256') != sha256_file(adapter_path):
        raise RuntimeError('resume policy adapter hash mismatch')
    if state.get('policy_parameter_signature') != parameter_signature:
        raise RuntimeError('resume policy parameter signature mismatch')
    next_prompt = state.get('next_prompt')
    if type(next_prompt) is not int or not 0 <= next_prompt <= TRAIN_PROMPTS or next_prompt % PROMPTS_PER_OPTIMIZER_STEP:
        raise RuntimeError('resume prompt boundary is invalid')
    weights = state.get('reward_weights')
    sums = state.get('window_sums')
    count = state.get('window_count')
    if not isinstance(weights, Mapping) or set(weights) != set(INITIAL_REWARD_WEIGHTS):
        raise RuntimeError('resume reward weights are incomplete')
    if any((not isinstance(weights[key], (int, float)) or not REWARD_WEIGHT_BOUNDS[key][0] <= float(weights[key]) <= REWARD_WEIGHT_BOUNDS[key][1] for key in INITIAL_REWARD_WEIGHTS)) or abs(sum((float(value) for value in weights.values())) - 1.0) > 1e-06:
        raise RuntimeError('resume reward weights are outside the frozen bounded simplex')
    if not isinstance(sums, Mapping) or set(sums) != set(INITIAL_REWARD_WEIGHTS) or any((not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in sums.values())):
        raise RuntimeError('resume dynamic-weight window is invalid')
    if type(count) is not int or not 0 <= count < REWARD_WEIGHT_UPDATE_TRAJECTORIES or count % NUM_GENERATIONS:
        raise RuntimeError('resume dynamic-weight count is invalid')

def validate_existing_completion(output: Path, config_sha: str) -> dict[str, Any] | None:
    path = output / 'COMPLETED.json'
    if not path.is_file():
        return None
    completed = read_json(path)
    if completed.get('task') != 'CERTIFY_CBT_COUNSELLOR_RLOO' or completed.get('run_config_sha256') != config_sha or completed.get('bundle_manifest_sha256') != sha256_file(BUNDLE / 'bundle_manifest.json') or (completed.get('status') not in {'COMPLETE_NO_RL_CHECKPOINT_BEAT_FROZEN_SFT_BASELINE', 'COMPLETE_COUNSELLOR_RLOO_ACCEPTED'}):
        raise RuntimeError('existing completion seal is not bound to this frozen run')
    selected = completed.get('selected_adapter')
    if selected:
        adapter_path = Path(selected) / 'adapter_model.safetensors'
        if not adapter_path.is_file() or completed.get('selected_adapter_sha256') != sha256_file(adapter_path):
            raise RuntimeError('existing completion selected-adapter binding is invalid')
    elif completed.get('selected_adapter_sha256') is not None:
        raise RuntimeError('existing completion records an adapter hash without an adapter')
    return completed

def load_existing_best(output: Path, config_sha: str, baseline: Mapping[str, Any]) -> dict[str, Any]:
    path = output / 'BEST_SELECTION.json'
    if not path.is_file():
        return {'mean_total_reward': baseline['mean_total_reward'], 'optimizer_step': 0, 'status': 'FROZEN_SFT_BASELINE'}
    best = read_json(path)
    step = best.get('optimizer_step')
    adapter_dir = Path(str(best.get('adapter_path', '')))
    adapter_path = adapter_dir / 'adapter_model.safetensors'
    if best.get('status') != 'CURRENT_BEST_ACCEPTABLE' or best.get('config_sha256') != config_sha or step not in {250, 500, 750} or (adapter_dir.parent.resolve() != output.resolve()) or (not adapter_path.is_file()) or (best.get('adapter_sha256') != sha256_file(adapter_path)):
        raise RuntimeError('existing best-selection artifact is not bound to this frozen run')
    return best

def main() -> None:
    started = time.time()
    if not os.environ.get('HF_TOKEN') and not os.environ.get('CRL5_MODEL_LOAD_PATH'):
        raise RuntimeError('HF_TOKEN is required')
    if not 4 <= SEMANTIC_WORKERS <= 16:
        raise RuntimeError('CRL5_LUNA_WORKERS must be in [4,16] for four concurrent prompt groups')
    validate_bundle()
    gpu = validate_hardware()
    train_rows = read_jsonl(BUNDLE / 'data' / 'tr.jsonl')
    development_rows = read_jsonl(BUNDLE / 'data' / 'dv.jsonl')
    fixed_rows = read_jsonl(BUNDLE / 'data' / 'dv64.jsonl')
    if (len(train_rows), len(development_rows), len(fixed_rows)) != (TRAIN_PROMPTS, DEVELOPMENT_PROMPTS, FIXED_DEVELOPMENT_PROMPTS):
        raise ValueError('dataset row count mismatch')
    if any((row['trajectory_target_present'] or row['test_row'] for row in train_rows + development_rows)):
        raise ValueError('target/test leakage in RLOO contexts')
    global EXPECTED_COUNSELLOR_SFT_SHA256
    expected_sft = read_json(BUNDLE / 'contracts' / 'checkpoint_bindings.json')['counsellor']
    counsellor_sft, EXPECTED_COUNSELLOR_SFT_SHA256 = bind_sft(Path(os.environ['CRL5_COUNSELLOR_SFT_DIR']), expected_sft)
    patient_adapter = locate_patient_adapter(Path(os.environ['CRL5_PATIENT_RLOO_RUN']))
    resistance_path = Path(os.environ['CRL5_RESISTANCE_TRACKER_DIR'])
    evidence_path = Path(os.environ['CRL5_CBT_EVIDENCE_TRACKER_DIR'])
    baseline_source_binding = {'mode': 'CURRENT_PROMPTS_AND_CHECKPOINTS'}
    run_contract = {'task': 'CERTIFY_CBT_COUNSELLOR_RLOO_V5_2202', 'base_model': MODEL_ID, 'base_model_revision': MODEL_REVISION, 'seed': SEED, 'counsellor_sft_sha256': EXPECTED_COUNSELLOR_SFT_SHA256, 'patient_rloo_sha256': EXPECTED_PATIENT_RLOO_SHA256, 'resistance_tracker_sha256': '135BEAD553FACF6D4922D9F03EF97707DAA0A426C41077415CDED80FAF30A0F7', 'cbt_evidence_tracker_sha256': 'D683F9298F5801D29869328949F934F5614B46688FC2CE1B7DE6C8B465AC799A', 'bundle_manifest_sha256': sha256_file(BUNDLE / 'bundle_manifest.json'), 'reward_contract_sha256': sha256_file(BUNDLE / 'contracts' / 'reward_contract.json'), 'cbt_evidence_base_config_sha256': sha256_file(BUNDLE / 'contracts' / 'cbt_evidence_base_model_config.json'), 'train_prompts': TRAIN_PROMPTS, 'trajectories_per_prompt': NUM_GENERATIONS, 'training_trajectories': TRAIN_PROMPTS * NUM_GENERATIONS, 'prompts_per_optimizer_step': PROMPTS_PER_OPTIMIZER_STEP, 'effective_trajectories_per_optimizer_step': PROMPTS_PER_OPTIMIZER_STEP * NUM_GENERATIONS, 'optimizer_steps': TOTAL_OPTIMIZER_STEPS, 'development_fixed_prompts': FIXED_DEVELOPMENT_PROMPTS, 'test_prompts_used': 0, 'learning_rate': LEARNING_RATE, 'kl_beta': KL_BETA, 'initial_reward_weights': INITIAL_REWARD_WEIGHTS, 'reward_weight_bounds': REWARD_WEIGHT_BOUNDS, 'maximum_weight_delta_per_64_prompt_update': MAX_REWARD_WEIGHT_DELTA_PER_UPDATE, 'naturalness_fusion': {'luna_high': NATURALNESS_LLM_WEIGHT, 'frozen_sft_alignment': NATURALNESS_SFT_WEIGHT}, 'progress_policy': PROGRESS_POLICY, 'baseline_source_binding': baseline_source_binding, 'artifact_identity': {'counsellor_sft_adapter_config_sha256': sha256_file(counsellor_sft / 'adapter_config.json'), 'patient_rloo_adapter_config_sha256': sha256_file(patient_adapter / 'adapter_config.json'), 'resistance_tracker_config_sha256': sha256_file(resistance_path / 'tracker_config.json'), 'resistance_temperature_sha256': sha256_file(resistance_path / 'temperature.json'), 'resistance_tokenizer_contract_sha256': directory_contract_sha256(resistance_path / 'tokenizer'), 'cbt_evidence_tracker_config_sha256': sha256_file(evidence_path / 'tracker_config.json'), 'cbt_evidence_temperature_sha256': sha256_file(evidence_path / 'temperature.json'), 'cbt_evidence_tokenizer_contract_sha256': directory_contract_sha256(evidence_path / 'tokenizer')}, 'hardware': gpu, 'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': importlib.metadata.version('transformers'), 'peft': importlib.metadata.version('peft'), 'accelerate': importlib.metadata.version('accelerate'), 'openai': importlib.metadata.version('openai')}, 'implementation': {'counsellor_base_copies': 1, 'counsellor_policy_adapter': POLICY_ADAPTER_NAME, 'counsellor_reference_adapter': REFERENCE_ADAPTER_NAME, 'completion_logits_fp32_only': True, 'fresh_reference_initialized_by_exact_policy_tensor_clone': True, 'policy_gradient_checkpointing': {'enabled': True, 'use_reentrant': False, 'policy_logprob_candidate_batch_size': NUM_GENERATIONS}, 'patient_simulator_base_is_separate': True}, 'scope': 'SYNTHETIC_SIMULATION_PROXY_NOT_CLINICAL_EFFICACY'}
    config_sha = sha256_bytes(canonical(run_contract).encode('utf-8'))
    output_root = Path(os.environ.get('CRL5_OUTPUT_ROOT', 'Path to corresponding file'))
    output = output_root / f'run_{config_sha[:12].lower()}'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'RUN_CONFIG.json', run_contract)
    completed = validate_existing_completion(output, config_sha)
    if completed is not None:
        print('[complete] existing sealed run found', flush=True)
        print(json.dumps(completed, indent=2), flush=True)
        return
    order = list(range(TRAIN_PROMPTS))
    random.Random(SEED).shuffle(order)
    resume = restore_interrupted_resume(output)
    policy_source = resume / 'adapter' if (resume / 'state.pt').is_file() else counsellor_sft
    print('[1/7] loading tokenizer and one shared counsellor base with two isolated adapters', flush=True)
    tokenizer = load_tokenizer()
    fresh_policy = policy_source.resolve() == counsellor_sft.resolve()
    counsellor = load_shared_counsellor(policy_source, counsellor_sft, initialize_identical_reference=fresh_policy)
    validate_shared_counsellor(counsellor, tokenizer, identical_start_adapters=fresh_policy)
    reference_tensor_sha256 = adapter_tensor_sha256(counsellor, REFERENCE_ADAPTER_NAME)
    policy_signature = policy_parameter_signature(counsellor)
    print('[2/7] loading frozen accepted Patient RLOO simulator', flush=True)
    patient = load_frozen_peft(patient_adapter)
    print('[3/7] loading frozen selected trackers and official CBT graph', flush=True)
    resistance_tracker = FrozenResistanceTracker(resistance_path)
    evidence_tracker = FrozenEvidenceTracker(evidence_path, BUNDLE / 'contracts' / 'cbt_evidence_base_model_config.json')
    rag = load_rag(BUNDLE / 'data' / 'rag.jsonl')
    judge = OnlineCounsellorJudge(output, MAX_LUNA_COST_USD, SEMANTIC_WORKERS)
    baseline, calibration = current_sft_baseline(output, run_contract, counsellor, patient, tokenizer, fixed_rows, judge, resistance_tracker, evidence_tracker, rag)
    activate_counsellor_adapter(counsellor, POLICY_ADAPTER_NAME, trainable_forward=True)
    trainable = [parameter for parameter in counsellor.parameters() if parameter.requires_grad]
    trainable_ids = {id(parameter) for parameter in trainable}
    expected_trainable_ids = {id(parameter) for name, parameter in counsellor.named_parameters() if adapter_parameter(name, POLICY_ADAPTER_NAME)}
    if not trainable or trainable_ids != expected_trainable_ids:
        raise RuntimeError('optimizer parameters are not exactly the policy adapter')
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    warmup = round(TOTAL_OPTIMIZER_STEPS * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, TOTAL_OPTIMIZER_STEPS)
    current_weights = dict(INITIAL_REWARD_WEIGHTS)
    window_sums = {key: 0.0 for key in INITIAL_REWARD_WEIGHTS}
    window_count = 0
    next_prompt = 0
    if (resume / 'state.pt').is_file():
        state = torch.load(resume / 'state.pt', map_location='cpu', weights_only=False)
        validate_resume_state(state, resume, config_sha, order, policy_signature)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        next_prompt = int(state['next_prompt'])
        current_weights = {key: float(value) for key, value in state['reward_weights'].items()}
        window_sums = {key: float(value) for key, value in state['window_sums'].items()}
        window_count = int(state['window_count'])
        print(f'[resume] continuing at prompt {next_prompt}', flush=True)
    print('[4/7] frozen SFT baseline and naturalness calibration verified', flush=True)
    best = load_existing_best(output, config_sha, baseline)
    trajectory_log = output / 'TRAINING_TRAJECTORIES.jsonl'
    weight_log = output / 'REWARD_WEIGHT_UPDATES.jsonl'
    dev_log = output / 'DEVELOPMENT_CHECKS.jsonl'
    if next_prompt:
        if trajectory_log.is_file():
            rewrite_jsonl(trajectory_log, [row for row in read_jsonl(trajectory_log) if int(row['prompt_position']) < next_prompt])
        if weight_log.is_file():
            rewrite_jsonl(weight_log, [row for row in read_jsonl(weight_log) if int(row['completed_prompts']) <= next_prompt])
        if dev_log.is_file():
            rewrite_jsonl(dev_log, [row for row in read_jsonl(dev_log) if int(row['optimizer_step']) <= next_prompt // PROMPTS_PER_OPTIMIZER_STEP])
    recent = deque(maxlen=32)
    optimizer.zero_grad(set_to_none=True)
    print(f'[5/7] training {TRAIN_PROMPTS} prompts / {TRAIN_PROMPTS * NUM_GENERATIONS} trajectories / {TOTAL_OPTIMIZER_STEPS} steps', flush=True)
    for step_start in range(next_prompt, TRAIN_PROMPTS, PROMPTS_PER_OPTIMIZER_STEP):
        step_end = step_start + PROMPTS_PER_OPTIMIZER_STEP
        rows = [train_rows[order[position]] for position in range(step_start, step_end)]
        generated = [sample_counsellor_group(counsellor, POLICY_ADAPTER_NAME, tokenizer, row, rag, SEED * 1000000 + position, TRAIN_TOP_P, TRAIN_TOP_K) for position, row in zip(range(step_start, step_end), rows)]
        for position, item in zip(range(step_start, step_end), generated):
            item['prompt_position'] = position
        semantic_job = judge.submit_step(generated, phase='training')
        local_groups, reference_groups = ([], [])
        try:
            for position, item in zip(range(step_start, step_end), generated):
                local, reference_joint = local_trajectory_rewards(item, counsellor, patient, tokenizer, resistance_tracker, evidence_tracker, rag, TRAIN_TOP_P, TRAIN_TOP_K, SEED * 2000000 + position)
                local_groups.append(local)
                reference_groups.append(reference_joint)
            semantic_groups = judge.finish_step(semantic_job)
        except BaseException:
            save_resume(resume, counsellor, optimizer, scheduler, step_start, order, current_weights, window_sums, window_count, config_sha, policy_signature)
            raise
        step_records = []
        weights_used = dict(current_weights)
        for item, local, semantic, reference_joint in zip(generated, local_groups, semantic_groups, reference_groups):
            reward_records, task_rewards = fuse_rewards(local, semantic, weights_used, calibration)
            policy_joint, _, _, _ = joint_logps(counsellor, tokenizer, item, require_grad=True, counsellor_adapter=POLICY_ADAPTER_NAME)
            log_ratio = policy_joint.detach() - reference_joint
            shaped = task_rewards - KL_BETA * log_ratio
            advantages = leave_one_out(shaped)
            policy_loss = -(advantages.detach() * policy_joint).mean()
            (policy_loss / PROMPTS_PER_OPTIMIZER_STEP).backward()
            for record in reward_records:
                for key, value in record['components'].items():
                    window_sums[key] += float(value)
                window_count += 1
            recent.append(float(task_rewards.mean()))
            step_records.append({'schema_version': 'certify-cbt-counsellor-rloo-trajectory-v1', 'split': 'train', 'prompt_position': item['prompt_position'], 'prompt_id': item['row']['prompt_id'], 'session_id': item['row']['session_id'], 'exchange_index': item['row']['exchange_index'], 'action_texts': [sample['text'] for sample in item['action_samples']], 'action_token_ids': [sample['token_ids'] for sample in item['action_samples']], 'decisions': item['decisions'], 'local_legality': item['local_legality'], 'counsellor_utterances': item['utterances'], 'response_token_ids': [sample['token_ids'] if sample else [] for sample in item['response_samples']], 'patient_responses': [record['patient_response'] for record in reward_records], 'rewards': reward_records, 'reward_weights': weights_used, 'task_rewards': [float(value) for value in task_rewards], 'kl_shaped_rewards': [float(value) for value in shaped], 'advantages': [float(value) for value in advantages], 'policy_loss': float(policy_loss.detach()), 'mean_joint_log_ratio': float(log_ratio.mean())})
        torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        for record in step_records:
            record['optimizer_step'] = step_end // PROMPTS_PER_OPTIMIZER_STEP
            record['learning_rate'] = float(scheduler.get_last_lr()[0])
            append_jsonl(trajectory_log, record)
        if window_count == REWARD_WEIGHT_UPDATE_TRAJECTORIES:
            means = {key: value / window_count for key, value in window_sums.items()}
            updated = update_reward_weights(current_weights, means)
            append_jsonl(weight_log, {'completed_prompts': step_end, 'trajectories_in_window': window_count, 'component_means': means, 'previous_weights': current_weights, 'updated_weights': updated})
            current_weights = updated
            window_sums = {key: 0.0 for key in INITIAL_REWARD_WEIGHTS}
            window_count = 0
        elif window_count > REWARD_WEIGHT_UPDATE_TRAJECTORIES:
            raise RuntimeError('reward-weight window boundary crossed')
        completed_step = step_end // PROMPTS_PER_OPTIMIZER_STEP
        print(f'[train] {completed_step}/{TOTAL_OPTIMIZER_STEPS} steps; prompt {step_end}/{TRAIN_PROMPTS}; recent_reward={np.mean(recent):.4f}; Luna=${judge.snapshot()['estimated_recorded_cost_usd']:.4f}', flush=True)
        if completed_step % VALIDATE_EVERY_STEPS == 0:
            print(f'[6/7] development check at step {completed_step}', flush=True)
            dev_logs, _ = run_groups(counsellor, POLICY_ADAPTER_NAME, patient, tokenizer, fixed_rows, judge, resistance_tracker, evidence_tracker, rag, f'development_step_{completed_step}', completed_step * FIXED_DEVELOPMENT_PROMPTS, calibration, INITIAL_REWARD_WEIGHTS, DEVELOPMENT_TOP_P, DEVELOPMENT_TOP_K)
            report = summarize(dev_logs, INITIAL_REWARD_WEIGHTS)
            report['optimizer_step'] = completed_step
            report['baseline_mean_total_reward'] = baseline['mean_total_reward']
            hard_keys = ('framework_fidelity', 'cbt_progress', 'empathy', 'naturalness')
            report['hard_nonregression_pass'] = all((report['component_means'][key] >= baseline['component_means'][key] - 0.02 for key in hard_keys))
            report['total_improvement'] = report['mean_total_reward'] - baseline['mean_total_reward']
            append_jsonl(dev_log, report)
            if report['hard_nonregression_pass'] and report['mean_total_reward'] > best['mean_total_reward']:
                best = {**report, 'status': 'CURRENT_BEST_ACCEPTABLE'}
                best_dir = output / f'best_counsellor_rl_adapter_step_{completed_step}'
                staging_best = best_dir.with_name(best_dir.name + '.staging')
                if staging_best.exists():
                    shutil.rmtree(staging_best)
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                save_policy_adapter(counsellor, staging_best)
                staging_best.rename(best_dir)
                best['adapter_sha256'] = sha256_file(best_dir / 'adapter_model.safetensors')
                best['adapter_path'] = str(best_dir)
                best['config_sha256'] = config_sha
                write_json(output / 'BEST_SELECTION.json', best)
            save_resume(resume, counsellor, optimizer, scheduler, step_end, order, current_weights, window_sums, window_count, config_sha, policy_signature)
    print('[7/7] sealing result', flush=True)
    if adapter_tensor_sha256(counsellor, REFERENCE_ADAPTER_NAME) != reference_tensor_sha256:
        raise RuntimeError('frozen reference adapter changed during training')
    if best['optimizer_step'] == 0:
        status = 'COMPLETE_NO_RL_CHECKPOINT_BEAT_FROZEN_SFT_BASELINE'
        selected_adapter = None
    else:
        status = 'COMPLETE_COUNSELLOR_RLOO_ACCEPTED'
        selected_adapter = best['adapter_path']
    selected_adapter_sha256 = sha256_file(Path(selected_adapter) / 'adapter_model.safetensors') if selected_adapter else None
    completed = {'status': status, 'task': 'CERTIFY_CBT_COUNSELLOR_RLOO', 'selected_stage': 'COUNSELLOR_RLOO_ACCEPTED' if selected_adapter else 'FROZEN_COUNSELLOR_SFT_RETAINED', 'selected_adapter': selected_adapter, 'selected_adapter_sha256': selected_adapter_sha256, 'best_selection': best, 'run_config_sha256': config_sha, 'bundle_manifest_sha256': sha256_file(BUNDLE / 'bundle_manifest.json'), 'frozen_patient_rloo_sha256': EXPECTED_PATIENT_RLOO_SHA256, 'frozen_counsellor_reference_tensor_sha256': reference_tensor_sha256, 'shared_counsellor_base': True, 'completion_logits_fp32_only': True, 'hardware': gpu, 'runtime': run_contract['runtime'], 'training_prompts': TRAIN_PROMPTS, 'training_trajectories': TRAIN_PROMPTS * NUM_GENERATIONS, 'optimizer_steps': TOTAL_OPTIMIZER_STEPS, 'test_rows_used': 0, 'semantic_judge': judge.snapshot(), 'elapsed_seconds': time.time() - started, 'scope_note': 'Synthetic CBT simulation/policy learning only; not psychotherapy quality or clinical efficacy.'}
    write_json(output / 'COMPLETED.json', completed)
    print(json.dumps(completed, indent=2), flush=True)
if __name__ == '__main__':
    main()
