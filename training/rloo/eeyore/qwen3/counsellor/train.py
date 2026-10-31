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
BUNDLE = Path(os.environ['CCRL_BUNDLE_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_judge import OnlineCounsellorJudge
from reward_math import gated_resistance_reward
from trajectory_runtime import FrozenEvidenceTracker, FrozenResistanceTracker, PROGRESS_POLICY, canonical, load_rag, parse_action, patient_context, progress_reward, response_context, selected_rag, sha256_file, validate_action
from action_space import action_prompt_context, action_text, legal_action_candidates, legal_token_successors
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
EXPECTED_COUNSELLOR_SFT_SHA256 = 'FABA7E8E12784C6F6E7EDF8843BFBE6C9A58B905FCBB9D4FBF37C2B64A1A5E61'
EXPECTED_PATIENT_RLOO_SHA256 = 'F59DD22E6CAB52A6E949C84ABD3DEFFCBF3DD3925386CB4947D88E3CA73D947D'
EXPECTED_PATIENT_RUN_CONFIG_SHA256 = 'C19D400C20A7A57DF5A863BF47CC3AEE7196DA98F5464C664232A814E4F26163'
EXPECTED_PATIENT_TRAINING_SUMMARY_SHA256 = '4A8A72326DD17949660B20EE5D6E0F1EC5A91270A0DEDA04A51B3287019FBE4C'
POLICY_ADAPTER_NAME = 'default'
REFERENCE_ADAPTER_NAME = 'reference'
SEED = 20260831
NUM_GENERATIONS = 4
TRAIN_PROMPTS = 3000
DEVELOPMENT_PROMPTS = 500
FIXED_DEVELOPMENT_PROMPTS = 64
PROMPTS_PER_OPTIMIZER_STEP = 4
TOTAL_OPTIMIZER_STEPS = TRAIN_PROMPTS // PROMPTS_PER_OPTIMIZER_STEP
RESUME_SAVE_EVERY_STEPS = 50
VALIDATE_EVERY_STEPS = 250
LEARNING_RATE = 1e-06
KL_BETA = 0.05
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
MAX_PROMPT_TOKENS = 6144
ACTION_CANDIDATE_MAX_TOKENS = 144
RESPONSE_MAX_NEW_TOKENS = 192
PATIENT_MAX_NEW_TOKENS = 160
MIN_NEW_TOKENS = 4
TEMPERATURE = 0.8
TRAIN_TOP_P = 1.0
DEVELOPMENT_TOP_P = 0.9
TRAIN_TOP_K = 0
DEVELOPMENT_TOP_K = 50
SEMANTIC_WORKERS = int(os.environ.get('CCRL_LUNA_WORKERS', '4'))
MAX_LUNA_COST_USD = float(os.environ.get('CCRL_MAX_LUNA_COST_USD', '30.0'))
INFERENCE_MICROBATCH = 8
LOGPROB_MICROBATCH = 8
INITIAL_REWARD_WEIGHTS = {'framework_fidelity': 0.3, 'cbt_progress': 0.25, 'empathy': 0.2, 'naturalness': 0.1, 'resistance_transition': 0.15}
REWARD_WEIGHT_BOUNDS = {'framework_fidelity': (0.27, 0.33), 'cbt_progress': (0.21, 0.29), 'empathy': (0.17, 0.23), 'naturalness': (0.08, 0.12), 'resistance_transition': (0.12, 0.18)}
REWARD_WEIGHT_UPDATE_PROMPTS = 64
REWARD_WEIGHT_UPDATE_TRAJECTORIES = REWARD_WEIGHT_UPDATE_PROMPTS * NUM_GENERATIONS
FAIR_STABLE_STEP_SIZE = 0.1
MAX_REWARD_WEIGHT_DELTA_PER_UPDATE = 0.002
NATURALNESS_LLM_WEIGHT = 0.7
NATURALNESS_SFT_WEIGHT = 0.3
HARD_FAILURE_PENALTY = 1.0
PROMPTS = json.loads((BUNDLE / 'contracts' / 'prompt_contracts.json').read_text(encoding='utf-8'))
ACTION_SYSTEM = PROMPTS['action_system']
RESPONSE_SYSTEM = PROMPTS['response_system']
PATIENT_SYSTEM = PROMPTS['patient_system']
COUNSELLOR_ENGAGEMENT_CUES = PROMPTS['counsellor_engagement_cues']
PATIENT_ENGAGEMENT_CUES = PROMPTS['patient_engagement_cues']

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

def append_jsonl_many(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    if not values:
        return
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        for value in values:
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
    completed_path = run_root / 'COMPLETED.json'
    summary_path = run_root / 'training_summary.json'
    config_path = run_root / 'RUN_CONFIG.json'
    if not all((path.is_file() for path in (completed_path, summary_path, config_path))):
        raise FileNotFoundError('the staged EEYORE patient RLOO run is incomplete')
    completed = read_json(completed_path)
    if completed.get('status') != 'COMPLETE' or completed.get('task') != 'CERTIFY_CBT_PATIENT_RLOO' or completed.get('selected_stage') != 'PATIENT_RLOO_ACCEPTED' or (completed.get('run_config_sha256') != EXPECTED_PATIENT_RUN_CONFIG_SHA256) or (completed.get('selected_adapter_sha256') != EXPECTED_PATIENT_RLOO_SHA256) or (completed.get('training_summary_sha256') != EXPECTED_PATIENT_TRAINING_SUMMARY_SHA256) or (sha256_file(summary_path) != EXPECTED_PATIENT_TRAINING_SUMMARY_SHA256) or (read_json(config_path).get('run_config_sha256') != EXPECTED_PATIENT_RUN_CONFIG_SHA256):
        raise ValueError('the staged patient run is not the accepted EEYORE patient RLOO checkpoint')
    return locate_adapter(run_root / 'selected_patient_rl_adapter', EXPECTED_PATIENT_RLOO_SHA256, 'Patient RLOO')

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.environ['CCRL_MODEL_LOAD_PATH'], local_files_only=True, token=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id != 151645:
        raise ValueError('Expected native Qwen end-of-turn token')
    tokenizer.padding_side = 'left'
    return tokenizer

def load_base_model():
    base = AutoModelForCausalLM.from_pretrained(os.environ['CCRL_MODEL_LOAD_PATH'], local_files_only=True, token=False, torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    base.config.use_cache = False
    base.config._name_or_path = MODEL_ID
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
    model.peft_config[POLICY_ADAPTER_NAME].base_model_name_or_path = MODEL_ID
    model.save_pretrained(directory, safe_serialization=True, selected_adapters=[POLICY_ADAPTER_NAME])
    if not (directory / 'adapter_model.safetensors').is_file():
        raise RuntimeError('policy adapter was not saved at the resumable root path')

def messages_prompt(tokenizer, system: str, prefix: str, payload: Mapping[str, Any]) -> tuple[str, int]:
    shown = dict(payload)
    if prefix == 'VISIBLE_PATIENT_RUNTIME_CONTEXT=':
        state = str(shown.pop('current_resistance_state'))
        shown['provisional_engagement_cue'] = PATIENT_ENGAGEMENT_CUES[state]
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prefix + '\n' + canonical(shown)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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

def conditional_logps(model: nn.Module, tokenizer, prompt_texts: Sequence[str], completions: Sequence[Sequence[int]], require_grad: bool, counsellor_adapter: str | None=None, allowed_completion_sets: Sequence[Sequence[Sequence[int]]] | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    if counsellor_adapter is not None:
        activate_counsellor_adapter(model, counsellor_adapter, trainable_forward=require_grad)
    prompt_rows, sequences, masks = ([], [], [])
    for prompt, target_values in zip(prompt_texts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target = list(target_values)
        if not target:
            raise ValueError('empty sampled completion')
        prompt_rows.append(prompt_ids)
        sequences.append(prompt_ids + target)
        masks.append([False] * (len(prompt_ids) - 1) + [True] * len(target))
    maximum = max(map(len, sequences))
    input_ids = torch.tensor([row + [tokenizer.pad_token_id] * (maximum - len(row)) for row in sequences], dtype=torch.long, device='cuda')
    attention = torch.tensor([[1] * len(row) + [0] * (maximum - len(row)) for row in sequences], dtype=torch.long, device='cuda')
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context, torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1]
    lengths = torch.tensor([len(target) for target in completions], dtype=torch.long, device='cuda')
    if allowed_completion_sets is not None:
        if len(allowed_completion_sets) != len(completions):
            raise ValueError('constrained action candidate batch mismatch')
        sequence_logps = []
        for row_index, (prompt_ids, target_values, candidate_values) in enumerate(zip(prompt_rows, completions, allowed_completion_sets)):
            target = list(target_values)
            candidates = [list(candidate) for candidate in candidate_values]
            total = logits[row_index, 0, 0].float() * 0.0
            for position, label in enumerate(target):
                prefix = target[:position]
                allowed = sorted({candidate[position] for candidate in candidates if len(candidate) > position and candidate[:position] == prefix})
                if label not in allowed:
                    raise ValueError('sampled action is outside the constrained legal candidate trie')
                step_logits = logits[row_index, len(prompt_ids) - 1 + position].float() / TEMPERATURE
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=step_logits.device)
                total = total + step_logits[label] - torch.logsumexp(step_logits.index_select(0, allowed_tensor), dim=0)
            sequence_logps.append(total)
        return (torch.stack(sequence_logps), lengths)
    token_mask = torch.tensor([row + [False] * (maximum - 1 - len(row)) for row in masks], dtype=torch.bool, device='cuda')
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
    return (sequence_logps, lengths)

@torch.inference_mode()
def generate_legal_action_group(model, tokenizer, prompt: str, candidates: Sequence[Mapping[str, str]], seed: int, top_p: float, top_k: int) -> tuple[list[dict[str, Any]], list[int], list[list[int]]]:
    candidate_texts = [action_text(candidate) for candidate in candidates]
    candidate_tokens = [tokenizer(text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id] for text in candidate_texts]
    if any((len(tokens) > ACTION_CANDIDATE_MAX_TOKENS for tokens in candidate_tokens)):
        raise ValueError('legal action candidate exceeds the sealed token limit')
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_ids)

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prefix = input_ids[prompt_length:].tolist()
        return legal_token_successors(prefix, candidate_tokens, tokenizer.pad_token_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    encoded = {key: value.cuda() for key, value in encoded.items()}
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        generated = model.generate(**encoded, do_sample=True, temperature=TEMPERATURE, top_p=top_p, top_k=top_k, num_return_sequences=NUM_GENERATIONS, min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=max(map(len, candidate_tokens)), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, prefix_allowed_tokens_fn=allowed_tokens, use_cache=True)
    output, selected_indices = ([], [])
    for sequence in generated:
        suffix = exact_suffix(sequence.tolist(), prompt_length, tokenizer.eos_token_id, max(map(len, candidate_tokens)))
        matches = [index for index, candidate in enumerate(candidate_tokens) if candidate == suffix]
        if len(matches) != 1:
            raise RuntimeError('constrained action generation did not resolve to one legal candidate')
        index = matches[0]
        selected_indices.append(index)
        output.append({'text': candidate_texts[index], 'token_ids': suffix})
    return (output, selected_indices, candidate_tokens)

def conditional_logps_chunked(model: nn.Module, tokenizer, prompt_texts: Sequence[str], completions: Sequence[Sequence[int]], require_grad: bool, counsellor_adapter: str, allowed_completion_sets: Sequence[Sequence[Sequence[int]]] | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    if len(prompt_texts) != len(completions):
        raise ValueError('log-probability prompt/completion batch mismatch')
    if allowed_completion_sets is not None and len(allowed_completion_sets) != len(completions):
        raise ValueError('log-probability legal-candidate batch mismatch')
    value_chunks, length_chunks = ([], [])
    for start in range(0, len(completions), LOGPROB_MICROBATCH):
        end = min(start + LOGPROB_MICROBATCH, len(completions))
        values, lengths = conditional_logps(model, tokenizer, prompt_texts[start:end], completions[start:end], require_grad, counsellor_adapter, allowed_completion_sets[start:end] if allowed_completion_sets is not None else None)
        value_chunks.append(values)
        length_chunks.append(lengths)
    if not value_chunks:
        return (torch.empty(0, dtype=torch.float32, device='cuda'), torch.empty(0, dtype=torch.long, device='cuda'))
    return (torch.cat(value_chunks), torch.cat(length_chunks))

def joint_logps_many(model, tokenizer, items: Sequence[Mapping[str, Any]], require_grad: bool, counsellor_adapter: str) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    action_prompts = [item['action_prompt'] for item in items for _ in range(NUM_GENERATIONS)]
    action_completions = [sample['token_ids'] for item in items for sample in item['action_samples']]
    action_candidate_sets = [item['legal_action_candidate_token_ids'] for item in items for _ in range(NUM_GENERATIONS)]
    action_logps, action_lengths = conditional_logps_chunked(model, tokenizer, action_prompts, action_completions, require_grad, counsellor_adapter, action_candidate_sets)
    total = len(action_completions)
    response_flat_indices = [group_index * NUM_GENERATIONS + candidate_index for group_index, item in enumerate(items) for candidate_index, sample in enumerate(item['response_samples']) if sample is not None]
    response_values = torch.zeros(total, device='cuda', dtype=torch.float32)
    response_lengths = torch.zeros(total, device='cuda', dtype=torch.long)
    if response_flat_indices:
        response_prompts = [items[index // NUM_GENERATIONS]['response_prompts'][index % NUM_GENERATIONS] for index in response_flat_indices]
        response_completions = [items[index // NUM_GENERATIONS]['response_samples'][index % NUM_GENERATIONS]['token_ids'] for index in response_flat_indices]
        response_logps, lengths = conditional_logps_chunked(model, tokenizer, response_prompts, response_completions, require_grad, counsellor_adapter)
        indices = torch.tensor(response_flat_indices, dtype=torch.long, device='cuda')
        response_values = response_values.index_copy(0, indices, response_logps)
        response_lengths = response_lengths.index_copy(0, indices, lengths)
    joint = action_logps + response_values
    return [(joint[start:start + NUM_GENERATIONS], (action_lengths + response_lengths)[start:start + NUM_GENERATIONS], response_values[start:start + NUM_GENERATIONS], response_lengths[start:start + NUM_GENERATIONS]) for start in range(0, total, NUM_GENERATIONS)]

def generate_inference_chunks(model, tokenizer, prompts: Sequence[str], seed: int, maximum: int, top_p: float, top_k: int) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for start in range(0, len(prompts), INFERENCE_MICROBATCH):
        generated.extend(generate_batch_one(model, tokenizer, prompts[start:start + INFERENCE_MICROBATCH], seed + start, maximum, top_p, top_k))
    return generated

def sample_counsellor_groups(counsellor, counsellor_adapter: str, tokenizer, rows: Sequence[Mapping[str, Any]], rag: Mapping[str, Any], seeds: Sequence[int], top_p: float, top_k: int) -> list[dict[str, Any]]:
    if len(rows) != len(seeds):
        raise ValueError('counsellor row/seed batch mismatch')
    activate_counsellor_adapter(counsellor, counsellor_adapter, trainable_forward=False)
    items: list[dict[str, Any]] = []
    pending_locations: list[tuple[int, int]] = []
    pending_prompts: list[str] = []
    for row, seed in zip(rows, seeds):
        visible = row['counsellor_visible']
        legal_candidates = legal_action_candidates(visible, rag)
        candidate_texts = [action_text(candidate) for candidate in legal_candidates]
        action_visible = action_prompt_context(visible, candidate_texts)
        action_prompt, action_prompt_tokens = messages_prompt(tokenizer, ACTION_SYSTEM, 'VISIBLE_COUNSELLOR_ACTION_CONTEXT=', action_visible)
        action_samples, selected_indices, candidate_token_ids = generate_legal_action_group(counsellor, tokenizer, action_prompt, legal_candidates, seed, top_p, top_k)
        decisions: list[dict[str, Any]] = []
        legality: list[dict[str, Any]] = []
        response_contracts: list[dict[str, Any] | None] = []
        response_prompts: list[str | None] = []
        for candidate_index, sample in enumerate(action_samples):
            try:
                decision = parse_action(sample['text'])
                validate_action(decision, visible, rag)
                legal = {'valid': True, 'reason': 'LEGAL_CANDIDATE'}
                context = response_context(visible, decision, rag)
                response_contract = {'selected_controller_decision': context['selected_controller_decision'], 'selected_official_cbt_step': context['selected_official_cbt_step'], 'response_phase': context['response_phase'], 'natural_engagement_cue': context['natural_engagement_cue']}
                response_prompt, _ = messages_prompt(tokenizer, RESPONSE_SYSTEM, 'VISIBLE_COUNSELLOR_RESPONSE_CONTEXT=', context)
                pending_locations.append((len(items), candidate_index))
                pending_prompts.append(response_prompt)
            except Exception as error:
                decision = {'invalid_action_text': sample['text']}
                legal = {'valid': False, 'reason': f'{type(error).__name__}: {error}'}
                response_contract = None
                response_prompt = None
            decisions.append(decision)
            legality.append(legal)
            response_contracts.append(response_contract)
            response_prompts.append(response_prompt)
        items.append({'row': row, 'action_prompt': action_prompt, 'action_prompt_tokens': action_prompt_tokens, 'legal_action_candidate_texts': candidate_texts, 'legal_action_candidate_token_ids': candidate_token_ids, 'selected_action_candidate_indices': selected_indices, 'action_samples': action_samples, 'decisions': decisions, 'local_legality': legality, 'response_contracts': response_contracts, 'response_prompts': response_prompts, 'response_samples': [None] * NUM_GENERATIONS})
    if pending_prompts:
        generated_responses = generate_inference_chunks(counsellor, tokenizer, pending_prompts, seeds[0] + 10000, RESPONSE_MAX_NEW_TOKENS, top_p, top_k)
        for (group_index, candidate_index), response in zip(pending_locations, generated_responses):
            if not response['text']:
                items[group_index]['local_legality'][candidate_index] = {'valid': False, 'reason': 'empty counsellor response'}
                continue
            items[group_index]['response_samples'][candidate_index] = response
    for item in items:
        item['utterances'] = [sample['text'] if sample is not None else '' for sample in item['response_samples']]
    return items
ROLE_BREAK = re.compile('\\b(as an ai|language model|clinical effectiveness|metadata|hidden reasoning)\\b', re.I)

def local_trajectory_rewards_many(items: Sequence[Mapping[str, Any]], counsellor, patient, tokenizer, resistance_tracker: FrozenResistanceTracker, evidence_tracker: FrozenEvidenceTracker, rag: Mapping[str, Any], top_p: float, top_k: int, seeds: Sequence[int]) -> tuple[list[list[dict[str, Any]]], list[torch.Tensor]]:
    if len(items) != len(seeds):
        raise ValueError('local-reward item/seed batch mismatch')
    reference_outputs = joint_logps_many(counsellor, tokenizer, items, require_grad=False, counsellor_adapter=REFERENCE_ADAPTER_NAME)
    records = [[{'framework_fidelity': -1.0, 'cbt_progress': 0.0, 'empathy': 0.0, 'naturalness_reference_raw': -20.0, 'resistance_transition_raw': 0.0, 'hard_penalty': HARD_FAILURE_PENALTY, 'penalty_reasons': ['ILLEGAL_ACTION'], 'patient_response': '', 'tracker': None, 'evidence': None, 'progress': None} for _ in range(NUM_GENERATIONS)] for _ in items]
    locations = [(group_index, candidate_index) for group_index, item in enumerate(items) for candidate_index, legal in enumerate(item['local_legality']) if legal['valid']]
    if not locations:
        return (records, [output[0].detach() for output in reference_outputs])
    tracker_rows = []
    for group_index, candidate_index in locations:
        item = items[group_index]
        visible = item['row']['counsellor_visible']
        decision = item['decisions'][candidate_index]
        utterance = item['utterances'][candidate_index]
        active = visible.get('active_cbt_rag')
        tracker_rows.append({'current_resistance_state': visible['current_provisional_resistance_state'], 'active_technique_id': decision['technique_id'] if decision else active['technique_id'] if active else 'NONE', 'active_step_id': decision['step_id'] if decision else active['current_step']['step_id'] if active else 'NONE', 'counsellor_utterance': utterance, 'patient_profile': visible['patient_profile'], 'counselling_progress': visible['supported_session_information'], 'dialogue_history': visible['dialogue_history']})
    tracker_predictions = resistance_tracker.predict(tracker_rows)
    evidence_rows = []
    patient_prompts: list[str] = []
    for local, (group_index, candidate_index) in enumerate(locations):
        item = items[group_index]
        visible = item['row']['counsellor_visible']
        prediction = tracker_predictions[local]
        p_context = patient_context(visible, item['utterances'][candidate_index], prediction['next_state'])
        patient_prompt, _ = messages_prompt(tokenizer, PATIENT_SYSTEM, 'VISIBLE_PATIENT_RUNTIME_CONTEXT=', p_context)
        patient_prompts.append(patient_prompt)
    patient_samples = generate_inference_chunks(patient, tokenizer, patient_prompts, seeds[0] + 20000, PATIENT_MAX_NEW_TOKENS, top_p, top_k)
    for local, (group_index, candidate_index) in enumerate(locations):
        item = items[group_index]
        patient_sample = patient_samples[local]
        decision = item['decisions'][candidate_index]
        step = rag[decision['technique_id']]['steps'][decision['step_id']]
        evidence_rows.append({'technique_id': decision['technique_id'], 'step_id': decision['step_id'], 'evidence_requirement': step['evidence_requirement'], 'counsellor_utterance': item['utterances'][candidate_index], 'patient_response': patient_sample['text']})
    evidence_predictions = evidence_tracker.predict(evidence_rows)
    for local, (group_index, candidate_index) in enumerate(locations):
        item = items[group_index]
        visible = item['row']['counsellor_visible']
        decision = item['decisions'][candidate_index]
        utterance = item['utterances'][candidate_index]
        progress = progress_reward(visible.get('cbt_progress_by_step', {}), decision, evidence_predictions[local])
        penalties: list[str] = []
        if ROLE_BREAK.search(utterance):
            penalties.append('ROLE_OR_METADATA_BREAK')
        if len(utterance.split()) < 5:
            penalties.append('EMPTY_OR_FRAGMENTARY_RESPONSE')
        penalty = HARD_FAILURE_PENALTY if penalties else 0.0
        _, _, response_logps, response_lengths = reference_outputs[group_index]
        reference_mean_logp = float(response_logps[candidate_index] / response_lengths[candidate_index].clamp_min(1))
        records[group_index][candidate_index] = {'framework_fidelity': 0.0, 'cbt_progress': float(progress['cbt_progress_reward']), 'empathy': 0.0, 'naturalness_reference_raw': reference_mean_logp, 'resistance_transition_raw': float(tracker_predictions[local]['expected_state_aware_utility']), 'hard_penalty': penalty, 'penalty_reasons': penalties, 'patient_response': patient_samples[local]['text'], 'tracker': tracker_predictions[local], 'evidence': evidence_predictions[local], 'progress': progress}
    return (records, [output[0].detach() for output in reference_outputs])

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
        generation_seeds = [SEED * 1000000 + start + index for index in range(len(batch))]
        generated = sample_counsellor_groups(counsellor, generation_adapter, tokenizer, batch, rag, generation_seeds, top_p, top_k)
        for index, item in enumerate(generated):
            item['prompt_position'] = position_offset + start + index
        semantic_job = judge.submit_step(generated, phase=phase)
        try:
            local_groups, reference_logps = local_trajectory_rewards_many(generated, counsellor, patient, tokenizer, resistance_tracker, evidence_tracker, rag, top_p, top_k, [SEED * 2000000 + start + index for index in range(len(batch))])
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

def validate_baseline_payload(baseline: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    if baseline.get('prompt_groups') != FIXED_DEVELOPMENT_PROMPTS:
        raise ValueError('frozen SFT baseline prompt count mismatch')
    if baseline.get('trajectories') != FIXED_DEVELOPMENT_PROMPTS * NUM_GENERATIONS:
        raise ValueError('frozen SFT baseline trajectory count mismatch')
    if set(baseline.get('component_means', {})) != set(INITIAL_REWARD_WEIGHTS):
        raise ValueError('frozen SFT baseline reward components mismatch')
    if baseline.get('selection_weights') != INITIAL_REWARD_WEIGHTS:
        raise ValueError('frozen SFT baseline selection weights mismatch')
    finite = [baseline.get('mean_total_reward'), baseline.get('hard_failure_rate'), calibration.get('mean'), calibration.get('standard_deviation')] + list(baseline['component_means'].values())
    if any((not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in finite)):
        raise ValueError('frozen SFT baseline contains non-finite metrics')
    if not 0.0 <= float(baseline['hard_failure_rate']) <= 1.0:
        raise ValueError('frozen SFT baseline hard-failure rate is invalid')
    if int(calibration.get('rows', 0)) < 32 or float(calibration['standard_deviation']) < 0.1:
        raise ValueError('frozen SFT naturalness calibration is invalid')

def create_or_load_sft_baseline(output: Path, config_sha: str, counsellor, patient, tokenizer, judge, resistance_tracker, evidence_tracker, rag, fixed_rows) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_path = output / 'SFT_BASELINE.json'
    calibration_path = output / 'NATURALNESS_CALIBRATION.json'
    binding_path = output / 'SFT_BASELINE_BINDING.json'
    paths = (baseline_path, calibration_path, binding_path)
    if any((path.is_file() for path in paths)):
        if not all((path.is_file() for path in paths)):
            raise RuntimeError('partial frozen SFT baseline artifacts')
        binding = read_json(binding_path)
        if binding.get('run_config_sha256') != config_sha or binding.get('baseline_sha256') != sha256_file(baseline_path) or binding.get('calibration_sha256') != sha256_file(calibration_path):
            raise RuntimeError('frozen SFT baseline binding mismatch')
        baseline, calibration = (read_json(baseline_path), read_json(calibration_path))
        validate_baseline_payload(baseline, calibration)
        return (baseline, calibration)
    logs, calibration = run_groups(counsellor, REFERENCE_ADAPTER_NAME, patient, tokenizer, fixed_rows, judge, resistance_tracker, evidence_tracker, rag, 'frozen_sft_baseline', 0, None, INITIAL_REWARD_WEIGHTS, DEVELOPMENT_TOP_P, DEVELOPMENT_TOP_K)
    baseline = summarize(logs, INITIAL_REWARD_WEIGHTS)
    validate_baseline_payload(baseline, calibration)
    write_json(baseline_path, baseline)
    write_json(calibration_path, calibration)
    write_json(binding_path, {'status': 'GENERATED_ONCE_FROM_HASH_BOUND_FROZEN_EEYORE_COUNSELLOR_SFT', 'run_config_sha256': config_sha, 'baseline_sha256': sha256_file(baseline_path), 'calibration_sha256': sha256_file(calibration_path), 'test_used': False})
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
    if completed.get('task') != 'CERTIFY_CBT_EEYORE_COUNSELLOR_RLOO' or completed.get('run_config_sha256') != config_sha or completed.get('bundle_manifest_sha256') != sha256_file(BUNDLE / 'bundle_manifest.json') or (completed.get('status') not in {'COMPLETE_NO_RL_CHECKPOINT_BEAT_FROZEN_SFT_BASELINE', 'COMPLETE_COUNSELLOR_RLOO_ACCEPTED'}):
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
    if not 4 <= SEMANTIC_WORKERS <= 16:
        raise RuntimeError('CCRL_LUNA_WORKERS must be in [4,16] for four concurrent prompt groups')
    validate_bundle()
    gpu = validate_hardware()
    train_rows = read_jsonl(BUNDLE / 'data' / 'tr.jsonl')
    development_rows = read_jsonl(BUNDLE / 'data' / 'dv.jsonl')
    fixed_rows = read_jsonl(BUNDLE / 'data' / 'dv64.jsonl')
    if (len(train_rows), len(development_rows), len(fixed_rows)) != (TRAIN_PROMPTS, DEVELOPMENT_PROMPTS, FIXED_DEVELOPMENT_PROMPTS):
        raise ValueError('dataset row count mismatch')
    if any((row['trajectory_target_present'] or row['test_row'] for row in train_rows + development_rows)):
        raise ValueError('target/test leakage in RLOO contexts')
    counsellor_sft = locate_adapter(Path(os.environ['CCRL_COUNSELLOR_SFT_DIR']), EXPECTED_COUNSELLOR_SFT_SHA256, 'Counsellor SFT')
    patient_adapter = locate_patient_adapter(Path(os.environ['CCRL_PATIENT_RLOO_RUN']))
    resistance_path = Path(os.environ['CCRL_RESISTANCE_TRACKER_DIR'])
    evidence_path = Path(os.environ['CCRL_CBT_EVIDENCE_TRACKER_DIR'])
    run_contract = {'task': 'CERTIFY_CBT_EEYORE_COUNSELLOR_RLOO_QWEN331_2002', 'base_model': MODEL_ID, 'executable_integrity_sha256': sha256_file(Path(__file__).parent / 'code_integrity.json'), 'base_model_revision': MODEL_REVISION, 'seed': SEED, 'counsellor_sft_sha256': EXPECTED_COUNSELLOR_SFT_SHA256, 'patient_rloo_sha256': EXPECTED_PATIENT_RLOO_SHA256, 'resistance_tracker_sha256': '135BEAD553FACF6D4922D9F03EF97707DAA0A426C41077415CDED80FAF30A0F7', 'cbt_evidence_tracker_sha256': 'D683F9298F5801D29869328949F934F5614B46688FC2CE1B7DE6C8B465AC799A', 'bundle_manifest_sha256': sha256_file(BUNDLE / 'bundle_manifest.json'), 'reward_contract_sha256': sha256_file(BUNDLE / 'contracts' / 'reward_contract.json'), 'cbt_evidence_base_config_sha256': sha256_file(BUNDLE / 'contracts' / 'cbt_evidence_base_model_config.json'), 'train_prompts': TRAIN_PROMPTS, 'trajectories_per_prompt': NUM_GENERATIONS, 'training_trajectories': TRAIN_PROMPTS * NUM_GENERATIONS, 'prompts_per_optimizer_step': PROMPTS_PER_OPTIMIZER_STEP, 'effective_trajectories_per_optimizer_step': PROMPTS_PER_OPTIMIZER_STEP * NUM_GENERATIONS, 'optimizer_steps': TOTAL_OPTIMIZER_STEPS, 'development_fixed_prompts': FIXED_DEVELOPMENT_PROMPTS, 'test_prompts_used': 0, 'learning_rate': LEARNING_RATE, 'kl_beta': KL_BETA, 'initial_reward_weights': INITIAL_REWARD_WEIGHTS, 'reward_weight_bounds': REWARD_WEIGHT_BOUNDS, 'maximum_weight_delta_per_64_prompt_update': MAX_REWARD_WEIGHT_DELTA_PER_UPDATE, 'naturalness_fusion': {'luna_high': NATURALNESS_LLM_WEIGHT, 'frozen_sft_alignment': NATURALNESS_SFT_WEIGHT}, 'progress_policy': PROGRESS_POLICY, 'artifact_identity': {'counsellor_sft_adapter_config_sha256': sha256_file(counsellor_sft / 'adapter_config.json'), 'patient_rloo_adapter_config_sha256': sha256_file(patient_adapter / 'adapter_config.json'), 'resistance_tracker_config_sha256': sha256_file(resistance_path / 'tracker_config.json'), 'resistance_temperature_sha256': sha256_file(resistance_path / 'temperature.json'), 'resistance_tokenizer_contract_sha256': directory_contract_sha256(resistance_path / 'tokenizer'), 'cbt_evidence_tracker_config_sha256': sha256_file(evidence_path / 'tracker_config.json'), 'cbt_evidence_temperature_sha256': sha256_file(evidence_path / 'temperature.json'), 'cbt_evidence_tokenizer_contract_sha256': directory_contract_sha256(evidence_path / 'tokenizer')}, 'hardware': gpu, 'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': importlib.metadata.version('transformers'), 'peft': importlib.metadata.version('peft'), 'accelerate': importlib.metadata.version('accelerate'), 'openai': importlib.metadata.version('openai')}, 'implementation': {'counsellor_base_copies': 1, 'counsellor_policy_adapter': POLICY_ADAPTER_NAME, 'counsellor_reference_adapter': REFERENCE_ADAPTER_NAME, 'completion_logits_fp32_only': True, 'action_sampling': 'TOKEN_TRIE_CONSTRAINED_TO_DETERMINISTIC_LEGAL_JSON_CANDIDATES', 'action_step_id_semantics': 'UPCOMING_COUNSELLOR_OPERATION', 'fresh_reference_initialized_by_exact_policy_tensor_clone': True, 'policy_gradient_checkpointing': {'enabled': True, 'use_reentrant': False, 'policy_logprob_candidate_batch_size': NUM_GENERATIONS}, 'patient_simulator_base_is_separate': True, 'execution_only_batching': {'inference_microbatch': INFERENCE_MICROBATCH, 'logprob_microbatch': LOGPROB_MICROBATCH, 'rloo_group_size': NUM_GENERATIONS, 'optimizer_prompt_groups': PROMPTS_PER_OPTIMIZER_STEP, 'tracker_rows_batched_across_optimizer_step': True, 'trajectory_log_single_flush_per_optimizer_step': True}, 'recovery_checkpointing': {'interval_optimizer_steps': RESUME_SAVE_EVERY_STEPS, 'save_before_development': True, 'resume_replays_missing_development_at_saved_boundary': True, 'storage': 'ROLLING_ATOMIC_RESUME_LATEST'}, 'initialization': 'FRESH_FROM_QWEN3_COUNSELLOR_SFT_OR_SAME_RUN_RESUME'}, 'scope': 'SYNTHETIC_SIMULATION_PROXY_NOT_CLINICAL_EFFICACY'}
    config_sha = sha256_bytes(canonical(run_contract).encode('utf-8'))
    output_root = Path(os.environ['CCRL_OUTPUT_ROOT'])
    output = output_root / f'run_{config_sha[:12].lower()}'
    if output_root.exists() and any((p.name != output.name for p in output_root.glob('run_*'))):
        raise RuntimeError('Existing run has a different frozen configuration; use a new run ID')
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
    baseline, calibration = create_or_load_sft_baseline(output, config_sha, counsellor, patient, tokenizer, judge, resistance_tracker, evidence_tracker, rag, fixed_rows)
    print('[baseline] frozen EEYORE counsellor-SFT baseline ready', flush=True)
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

    def perform_development_check(completed_step: int, best_state: dict[str, Any]) -> dict[str, Any]:
        print(f'[6/7] development check at step {completed_step}', flush=True)
        dev_logs, _ = run_groups(counsellor, POLICY_ADAPTER_NAME, patient, tokenizer, fixed_rows, judge, resistance_tracker, evidence_tracker, rag, f'development_step_{completed_step}', completed_step * FIXED_DEVELOPMENT_PROMPTS, calibration, INITIAL_REWARD_WEIGHTS, DEVELOPMENT_TOP_P, DEVELOPMENT_TOP_K)
        report = summarize(dev_logs, INITIAL_REWARD_WEIGHTS)
        report['optimizer_step'] = completed_step
        report['baseline_mean_total_reward'] = baseline['mean_total_reward']
        hard_keys = ('framework_fidelity', 'cbt_progress', 'empathy', 'naturalness')
        report['hard_nonregression_pass'] = all((report['component_means'][key] >= baseline['component_means'][key] - 0.02 for key in hard_keys))
        report['total_improvement'] = report['mean_total_reward'] - baseline['mean_total_reward']
        if report['hard_nonregression_pass'] and report['mean_total_reward'] > best_state['mean_total_reward']:
            best_state = {**report, 'status': 'CURRENT_BEST_ACCEPTABLE'}
            best_dir = output / f'best_counsellor_rl_adapter_step_{completed_step}'
            staging_best = best_dir.with_name(best_dir.name + '.staging')
            if staging_best.exists():
                shutil.rmtree(staging_best)
            if best_dir.exists():
                shutil.rmtree(best_dir)
            save_policy_adapter(counsellor, staging_best)
            staging_best.rename(best_dir)
            best_state['adapter_sha256'] = sha256_file(best_dir / 'adapter_model.safetensors')
            best_state['adapter_path'] = str(best_dir)
            best_state['config_sha256'] = config_sha
            write_json(output / 'BEST_SELECTION.json', best_state)
        append_jsonl(dev_log, report)
        return best_state
    if next_prompt:
        if trajectory_log.is_file():
            rewrite_jsonl(trajectory_log, [row for row in read_jsonl(trajectory_log) if int(row['prompt_position']) < next_prompt])
        if weight_log.is_file():
            rewrite_jsonl(weight_log, [row for row in read_jsonl(weight_log) if int(row['completed_prompts']) <= next_prompt])
        if dev_log.is_file():
            rewrite_jsonl(dev_log, [row for row in read_jsonl(dev_log) if int(row['optimizer_step']) <= next_prompt // PROMPTS_PER_OPTIMIZER_STEP])
    completed_development_steps = {int(row['optimizer_step']) for row in read_jsonl(dev_log)} if dev_log.is_file() else set()
    resumed_step = next_prompt // PROMPTS_PER_OPTIMIZER_STEP
    if resumed_step > 0 and resumed_step % VALIDATE_EVERY_STEPS == 0 and (resumed_step not in completed_development_steps):
        print(f'[resume] recovery checkpoint at step {resumed_step} has a pending development check', flush=True)
        best = perform_development_check(resumed_step, best)
    recent = deque(maxlen=32)
    optimizer.zero_grad(set_to_none=True)
    print(f'[5/7] training {TRAIN_PROMPTS} prompts / {TRAIN_PROMPTS * NUM_GENERATIONS} trajectories / {TOTAL_OPTIMIZER_STEPS} steps', flush=True)
    for step_start in range(next_prompt, TRAIN_PROMPTS, PROMPTS_PER_OPTIMIZER_STEP):
        step_end = step_start + PROMPTS_PER_OPTIMIZER_STEP
        rows = [train_rows[order[position]] for position in range(step_start, step_end)]
        generated = sample_counsellor_groups(counsellor, POLICY_ADAPTER_NAME, tokenizer, rows, rag, [SEED * 1000000 + position for position in range(step_start, step_end)], TRAIN_TOP_P, TRAIN_TOP_K)
        for position, item in zip(range(step_start, step_end), generated):
            item['prompt_position'] = position
        semantic_job = judge.submit_step(generated, phase='training')
        try:
            local_groups, reference_groups = local_trajectory_rewards_many(generated, counsellor, patient, tokenizer, resistance_tracker, evidence_tracker, rag, TRAIN_TOP_P, TRAIN_TOP_K, [SEED * 2000000 + position for position in range(step_start, step_end)])
            semantic_groups = judge.finish_step(semantic_job)
        except BaseException:
            save_resume(resume, counsellor, optimizer, scheduler, step_start, order, current_weights, window_sums, window_count, config_sha, policy_signature)
            raise
        weights_used = dict(current_weights)
        fused_groups = [fuse_rewards(local, semantic, weights_used, calibration) for local, semantic in zip(local_groups, semantic_groups)]
        step_records = []
        policy_group_microbatch = LOGPROB_MICROBATCH // NUM_GENERATIONS
        for group_start in range(0, PROMPTS_PER_OPTIMIZER_STEP, policy_group_microbatch):
            group_end = min(group_start + policy_group_microbatch, PROMPTS_PER_OPTIMIZER_STEP)
            policy_outputs = joint_logps_many(counsellor, tokenizer, generated[group_start:group_end], require_grad=True, counsellor_adapter=POLICY_ADAPTER_NAME)
            chunk_losses = []
            for item, (reward_records, task_rewards), policy_output, reference_joint in zip(generated[group_start:group_end], fused_groups[group_start:group_end], policy_outputs, reference_groups[group_start:group_end]):
                policy_joint = policy_output[0]
                log_ratio = policy_joint.detach() - reference_joint
                shaped = task_rewards - KL_BETA * log_ratio
                advantages = leave_one_out(shaped)
                policy_loss = -(advantages.detach() * policy_joint).mean()
                chunk_losses.append(policy_loss)
                for record in reward_records:
                    for key, value in record['components'].items():
                        window_sums[key] += float(value)
                    window_count += 1
                recent.append(float(task_rewards.mean()))
                step_records.append({'schema_version': 'certify-cbt-counsellor-rloo-trajectory-v1', 'split': 'train', 'prompt_position': item['prompt_position'], 'prompt_id': item['row']['prompt_id'], 'session_id': item['row']['session_id'], 'exchange_index': item['row']['exchange_index'], 'legal_action_candidates': item['legal_action_candidate_texts'], 'selected_action_candidate_indices': item['selected_action_candidate_indices'], 'action_texts': [sample['text'] for sample in item['action_samples']], 'action_token_ids': [sample['token_ids'] for sample in item['action_samples']], 'decisions': item['decisions'], 'local_legality': item['local_legality'], 'counsellor_utterances': item['utterances'], 'response_token_ids': [sample['token_ids'] if sample else [] for sample in item['response_samples']], 'patient_responses': [record['patient_response'] for record in reward_records], 'rewards': reward_records, 'reward_weights': weights_used, 'task_rewards': [float(value) for value in task_rewards], 'kl_shaped_rewards': [float(value) for value in shaped], 'advantages': [float(value) for value in advantages], 'policy_loss': float(policy_loss.detach()), 'mean_joint_log_ratio': float(log_ratio.mean())})
            (torch.stack(chunk_losses).sum() / PROMPTS_PER_OPTIMIZER_STEP).backward()
        torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        for record in step_records:
            record['optimizer_step'] = step_end // PROMPTS_PER_OPTIMIZER_STEP
            record['learning_rate'] = float(scheduler.get_last_lr()[0])
        append_jsonl_many(trajectory_log, step_records)
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
        if completed_step % RESUME_SAVE_EVERY_STEPS == 0:
            print(f'[checkpoint] saving recovery state at step {completed_step}', flush=True)
            save_resume(resume, counsellor, optimizer, scheduler, step_end, order, current_weights, window_sums, window_count, config_sha, policy_signature)
        if completed_step % VALIDATE_EVERY_STEPS == 0:
            best = perform_development_check(completed_step, best)
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
    completed = {'status': status, 'task': 'CERTIFY_CBT_EEYORE_COUNSELLOR_RLOO', 'selected_stage': 'COUNSELLOR_RLOO_ACCEPTED' if selected_adapter else 'FROZEN_COUNSELLOR_SFT_RETAINED', 'selected_adapter': selected_adapter, 'selected_adapter_sha256': selected_adapter_sha256, 'best_selection': best, 'run_config_sha256': config_sha, 'bundle_manifest_sha256': sha256_file(BUNDLE / 'bundle_manifest.json'), 'frozen_patient_rloo_sha256': EXPECTED_PATIENT_RLOO_SHA256, 'frozen_counsellor_reference_tensor_sha256': reference_tensor_sha256, 'shared_counsellor_base': True, 'completion_logits_fp32_only': True, 'hardware': gpu, 'runtime': run_contract['runtime'], 'training_prompts': TRAIN_PROMPTS, 'training_trajectories': TRAIN_PROMPTS * NUM_GENERATIONS, 'optimizer_steps': TOTAL_OPTIMIZER_STEPS, 'test_rows_used': 0, 'semantic_judge': judge.snapshot(), 'elapsed_seconds': time.time() - started, 'scope_note': 'Synthetic CBT simulation/policy learning only; not psychotherapy quality or clinical efficacy.'}
    write_json(output / 'COMPLETED.json', completed)
    print(json.dumps(completed, indent=2), flush=True)
if __name__ == '__main__':
    main()
