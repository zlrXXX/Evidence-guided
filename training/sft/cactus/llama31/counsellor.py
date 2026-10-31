from __future__ import annotations
import collections, gc, hashlib, importlib.metadata, json, math, os, platform, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MODEL_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
EXPECTED_RUNTIME_DEPENDENCIES = {'accelerate': '1.10.1', 'bitsandbytes': '0.47.0', 'datasets': '4.0.0', 'numpy': '2.1.3', 'peft': '0.17.1', 'safetensors': '0.6.2', 'sentencepiece': '0.2.1', 'tensorboard': '2.20.0', 'torch': '2.8.0', 'transformers': '4.56.2'}
SEED = 42
USE_4BIT = False
MAX_LENGTH = 16384
NUM_TRAIN_EPOCHS = 1.0
MID_EPOCH_EVAL_FRACTION = 0.5
LEARNING_RATE = 2e-05
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 12
RECOVERY_SAVE_STEPS = 500
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05
MAX_CLASS_WEIGHT = 3.0
EXTRA_CALLBACKS = []

def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest().upper()

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path.name}:{line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected object at {path.name}:{line_number}')
            rows.append(row)
    return rows

def normalized_inverse_sqrt_weights(counts: Mapping[str, int]) -> Dict[str, float]:
    if not counts or any((value <= 0 for value in counts.values())):
        raise ValueError('All class counts must be positive')
    total = sum(counts.values())
    classes = len(counts)
    raw = {key: min(MAX_CLASS_WEIGHT, math.sqrt(total / (classes * value))) for key, value in counts.items()}
    weighted_mean = sum((counts[key] * raw[key] for key in counts)) / total
    return {key: raw[key] / weighted_mean for key in raw}

def counsellor_example_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[List[float], Dict[str, Any]]:
    parsed_actions: List[Optional[Dict[str, Any]]] = []
    action_counts: collections.Counter[str] = collections.Counter()
    delivery_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row['task'] == 'ACTION':
            target = json.loads(row['target'])
            parsed_actions.append(target)
            action_counts[target['action']] += 1
            if target['action'] == 'CONTINUE_TECHNIQUE':
                delivery_counts[target['delivery_mode']] += 1
        else:
            parsed_actions.append(None)
    action_weights = normalized_inverse_sqrt_weights(action_counts)
    delivery_weights = normalized_inverse_sqrt_weights(delivery_counts)
    raw_action_row_weights: List[float] = []
    for target in parsed_actions:
        if target is None:
            continue
        weight = action_weights[target['action']]
        if target['action'] == 'CONTINUE_TECHNIQUE':
            weight *= delivery_weights[target['delivery_mode']]
        raw_action_row_weights.append(min(MAX_CLASS_WEIGHT, weight))
    action_mean = sum(raw_action_row_weights) / len(raw_action_row_weights)
    weights: List[float] = []
    action_position = 0
    final_by_stratum: Dict[str, List[float]] = collections.defaultdict(list)
    for target in parsed_actions:
        if target is None:
            weights.append(1.0)
            continue
        value = raw_action_row_weights[action_position] / action_mean
        action_position += 1
        weights.append(value)
        stratum = target['action']
        if stratum == 'CONTINUE_TECHNIQUE':
            stratum = f'{stratum}::{target['delivery_mode']}'
        final_by_stratum[stratum].append(value)
    report = {'method': 'inverse-sqrt class weighting; action and response views remain one-to-one', 'action_counts': dict(sorted(action_counts.items())), 'continue_delivery_counts': dict(sorted(delivery_counts.items())), 'action_base_weights': {key: round(value, 6) for key, value in sorted(action_weights.items())}, 'delivery_relative_weights': {key: round(value, 6) for key, value in sorted(delivery_weights.items())}, 'final_mean_weight_by_action_stratum': {key: round(sum(values) / len(values), 6) for key, values in sorted(final_by_stratum.items())}, 'overall_action_weight_mean': round(sum((weight for weight, target in zip(weights, parsed_actions) if target is not None)) / sum((target is not None for target in parsed_actions)), 6), 'response_weight': 1.0}
    return (weights, report)

def train_role(role: str, data_dir: Path, bundle_hash: str, bundle_manifest: Mapping[str, Any], hf_token: str, gpu_summary: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(SEED)
    runtime_dependencies = {name: importlib.metadata.version(name) for name in EXPECTED_RUNTIME_DEPENDENCIES}
    if runtime_dependencies != EXPECTED_RUNTIME_DEPENDENCIES:
        raise ValueError(f'Runtime dependency mismatch: expected {EXPECTED_RUNTIME_DEPENDENCIES}, observed {runtime_dependencies}')
    runtime_environment = {'packages': runtime_dependencies, 'python': platform.python_version(), 'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu_name': str(gpu_summary['name']), 'gpu_memory_gib': float(gpu_summary['memory_gib']), 'bf16_supported': bool(gpu_summary['bf16_supported'])}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    split_rows = {split: load_jsonl(data_dir / f'{role}_{split}.jsonl') for split in ('train', 'development', 'test')}
    validation = {'mode': 'SEALED_HASH_AND_JSON_PARSE_ONLY', 'reason': 'The pre-seal validator already checked semantics, causality, schema, and split isolation; runtime rechecks exact file hashes before parsing.', 'rows': {split: len(rows) for split, rows in split_rows.items()}}
    print(f'\n{role.upper()} SEALED DATA LOAD')
    print(json.dumps(validation, indent=2, sort_keys=True))
    train_rows = split_rows['train']
    if role == 'counsellor':
        train_weights, balance_report = counsellor_example_weights(train_rows)
    else:
        train_weights = [1.0] * len(train_rows)
        balance_report = {'method': 'natural patient-response distribution; no class reweighting', 'response_weight': 1.0}
    for row, weight in zip(train_rows, train_weights):
        row['loss_weight'] = weight
    for split in ('development', 'test'):
        for row in split_rows[split]:
            row['loss_weight'] = 1.0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, local_files_only=True, token=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def tokenize_record(row: Mapping[str, Any]) -> Dict[str, Any]:
        prompt_text = tokenizer.apply_chat_template([{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['user']}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        target_ids = tokenizer(row['target'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        total_length = len(prompt_ids) + len(target_ids)
        if len(target_ids) >= MAX_LENGTH:
            raise ValueError(f'Target alone exceeds MAX_LENGTH: {row['example_id']}')
        if total_length > MAX_LENGTH:
            raise ValueError(f'Fail-closed: {row['example_id']} requires {total_length} tokens, above MAX_LENGTH={MAX_LENGTH}. No runtime field will be silently truncated.')
        return {'input_ids': prompt_ids + target_ids, 'attention_mask': [1] * total_length, 'labels': [-100] * len(prompt_ids) + target_ids, 'loss_weight': float(row['loss_weight']), 'sequence_length': total_length, 'target_tokens': len(target_ids)}
    datasets: Dict[str, Dataset] = {}
    length_reports: Dict[str, Dict[str, Any]] = {}
    for split, rows in split_rows.items():
        raw_dataset = Dataset.from_list(rows)
        tokenized = raw_dataset.map(tokenize_record, remove_columns=raw_dataset.column_names, num_proc=None, desc=f'Tokenizing {role} {split}')
        lengths = tokenized['sequence_length']
        targets = tokenized['target_tokens']
        length_reports[split] = {'examples': len(tokenized), 'max_sequence_tokens': int(max(lengths)), 'p50_sequence_tokens': int(np.percentile(lengths, 50)), 'p95_sequence_tokens': int(np.percentile(lengths, 95)), 'p99_sequence_tokens': int(np.percentile(lengths, 99)), 'max_target_tokens': int(max(targets)), 'prompt_truncations': 0}
        datasets[split] = tokenized.remove_columns(['sequence_length', 'target_tokens'])
    run_flavor = 'qlora4' if USE_4BIT else 'bf16_lora'
    run_name = f'{role}_full_seed{SEED}_{run_flavor}_r{LORA_R}_ep1_save{RECOVERY_SAVE_STEPS}'
    output_dir = Path(OUTPUT_ROOT) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_contract = {'role': role, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'executing_trainer_canonical_sha256': EXPECTED_EXECUTING_TRAINER_CANONICAL_SHA256, 'runtime_environment': runtime_environment, 'bundle_sha256': bundle_hash, 'source_corpus_sha256': EXPECTED_SOURCE_CORPUS_SHA256, 'rag_sha256': EXPECTED_RAG_SHA256, 'seed': SEED, 'use_4bit': USE_4BIT, 'max_length': MAX_LENGTH, 'num_train_epochs': NUM_TRAIN_EPOCHS, 'learning_rate': LEARNING_RATE, 'micro_batch_size': MICRO_BATCH_SIZE, 'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS, 'recovery_save_steps': RECOVERY_SAVE_STEPS, 'mid_epoch_eval_fraction': MID_EPOCH_EVAL_FRACTION, 'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT, 'weight_decay': WEIGHT_DECAY, 'warmup_ratio': WARMUP_RATIO, 'loss_normalization': 'trainer_managed_gradient_accumulation_v1', 'train_file_sha256': EXPECTED_FILES[f'{role}_train.jsonl']['sha256'], 'development_file_sha256': EXPECTED_FILES[f'{role}_development.jsonl']['sha256'], 'test_file_sha256': EXPECTED_FILES[f'{role}_test.jsonl']['sha256']}
    config_hash = sha256_json(config_contract)
    run_config_path = output_dir / 'RUN_CONFIG.json'
    if run_config_path.is_file():
        stored_contract = json.loads(run_config_path.read_text(encoding='utf-8'))
        if stored_contract.get('config_sha256') != config_hash:
            raise ValueError(f'Existing {role} checkpoint directory belongs to a different config')
    else:
        stale_items = [path.name for path in output_dir.iterdir() if path.name != 'runs']
        if stale_items:
            raise FileExistsError(f'Refusing unbound resume for {role}; RUN_CONFIG.json is missing: {stale_items}')
        run_config_path.write_text(json.dumps({'config_sha256': config_hash, 'config': config_contract}, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    completion_path = output_dir / 'COMPLETED.json'
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        if completion.get('config_sha256') != config_hash:
            raise ValueError(f'Existing {role} completion marker has a different config')
        adapter_path = output_dir / 'best_adapter' / 'adapter_model.safetensors'
        summary_path = output_dir / 'training_summary.json'
        if not adapter_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f'Completed {role} run is missing its adapter or summary')
        if sha256_file(adapter_path) != completion.get('adapter_sha256'):
            raise ValueError(f'Completed {role} adapter hash mismatch')
        if sha256_file(summary_path) != completion.get('training_summary_sha256'):
            raise ValueError(f'Completed {role} training-summary hash mismatch')
        stored_summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if stored_summary.get('config_sha256') != config_hash or stored_summary.get('adapter_sha256') != completion.get('adapter_sha256'):
            raise ValueError(f'Completed {role} summary provenance mismatch')
        print(f'{role} is already complete and hash-compatible; skipping retraining.')
        return completion
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_LOAD_PATH, local_files_only=True, token=False, torch_dtype=torch.bfloat16, quantization_config=quantization_config, device_map={'': 0}, attn_implementation='sdpa')
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_LOAD_PATH, local_files_only=True, token=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa')
    model.name_or_path = MODEL_ID
    model.config._name_or_path = MODEL_ID
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config._name_or_path = BASE_MODEL_ID
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model.print_trainable_parameters()
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')

    class WeightedCollator:

        def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            weights = torch.tensor([float(feature['loss_weight']) for feature in features], dtype=torch.float32)
            stripped = [{key: value for key, value in feature.items() if key != 'loss_weight'} for feature in features]
            batch = base_collator(stripped)
            batch['example_weight'] = weights
            return batch

    class TargetOnlyWeightedTrainer(Trainer):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool=False, num_items_in_batch: Optional[Any]=None) -> Any:
            weights = inputs.pop('example_weight').to(model.device)
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            token_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none', ignore_index=-100).view(shift_labels.shape)
            valid = shift_labels.ne(-100)
            valid_counts = valid.sum(dim=1).clamp_min(1)
            per_example = (token_losses * valid).sum(dim=1) / valid_counts
            loss = (per_example * weights).mean()
            return (loss, outputs) if return_outputs else loss

    class FixedStepRecoveryAndMidpointEvalCallback(TrainerCallback):

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            midpoint_step = max(1, int(state.max_steps * MID_EPOCH_EVAL_FRACTION + 0.5))
            is_midpoint = state.global_step == midpoint_step
            if is_midpoint or (state.global_step > 0 and state.global_step % RECOVERY_SAVE_STEPS == 0):
                control.should_save = True
            if is_midpoint:
                control.should_evaluate = True
            return control
    training_args = TrainingArguments(output_dir=str(output_dir), run_name=run_name, seed=SEED, data_seed=SEED, num_train_epochs=NUM_TRAIN_EPOCHS, learning_rate=LEARNING_RATE, lr_scheduler_type='cosine', warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY, max_grad_norm=1.0, per_device_train_batch_size=MICRO_BATCH_SIZE, per_device_eval_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, bf16=True, bf16_full_eval=True, fp16=False, tf32=True, optim='adamw_torch_fused', eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, logging_strategy='steps', logging_steps=10, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, report_to=['tensorboard'], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, save_safetensors=True)
    trainer = TargetOnlyWeightedTrainer(model=model, args=training_args, train_dataset=datasets['train'], eval_dataset=datasets['development'], processing_class=tokenizer, data_collator=WeightedCollator(), callbacks=[FixedStepRecoveryAndMidpointEvalCallback(), *EXTRA_CALLBACKS])
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f'Resuming {role} from {last_checkpoint}')
    elif any(output_dir.iterdir()):
        allowed = {'runs', 'RUN_CONFIG.json'}
        unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
        if unexpected:
            raise FileExistsError(f'Refusing to overwrite incomplete {role} output without a checkpoint: {unexpected}')
    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    elapsed_seconds = time.time() - started
    best_adapter = output_dir / 'best_adapter'
    trainer.save_model(str(best_adapter))
    tokenizer.save_pretrained(str(best_adapter))
    development_metrics = trainer.evaluate(datasets['development'], metric_key_prefix='development')
    test_metrics = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    adapter_path = best_adapter / 'adapter_model.safetensors'
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    summary = {'artifact_type': 'CERTIFY_CBT_FULL_SFT_ADAPTER', 'status': 'COMPLETE', 'role': role, 'config': config_contract, 'config_sha256': config_hash, 'bundle_manifest': bundle_manifest, 'bundle_sha256': bundle_hash, 'data_validation': validation, 'class_balance': balance_report, 'token_lengths': length_reports, 'train_metrics': train_result.metrics, 'development_metrics': development_metrics, 'test_metrics': test_metrics, 'development_perplexity': math.exp(min(20.0, float(development_metrics['development_loss']))), 'test_perplexity': math.exp(min(20.0, float(test_metrics['test_loss']))), 'best_checkpoint': trainer.state.best_model_checkpoint, 'best_metric': trainer.state.best_metric, 'elapsed_seconds': elapsed_seconds, 'adapter_sha256': sha256_file(adapter_path), 'gpu': dict(gpu_summary), 'software': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'peft': __import__('peft').__version__}, 'scope_note': 'These are SFT optimization metrics, not psychotherapy quality, clinical efficacy, or final interactive-framework results.'}
    (output_dir / 'training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    completion = {'status': 'COMPLETE', 'role': role, 'config_sha256': config_hash, 'adapter_sha256': summary['adapter_sha256'], 'training_summary_sha256': sha256_file(output_dir / 'training_summary.json'), 'output_dir': str(output_dir)}
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'\n{role.upper()} COMPLETE')
    print(json.dumps(completion, indent=2, sort_keys=True))
    del trainer, model, tokenizer, datasets, split_rows, train_rows
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return completion
