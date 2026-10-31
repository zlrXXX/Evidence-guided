from __future__ import annotations
import collections, gc, hashlib, json, math, os, time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
EXPECTED_SOURCE_SESSIONS = 2201
EXTRA_CALLBACKS = []
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
SEED = 42
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

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Expected object at {path.name}:{line_number}')
            rows.append(row)
    return rows

def require_runtime() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if 'A100' not in name or memory < 70:
        raise RuntimeError(f'Expected A100 80GB, found {name} ({memory:.1f} GiB)')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required')
    print(f'Runtime: {name}, {memory:.1f} GiB')

def normalized_inverse_sqrt_weights(counts: Mapping[str, int]) -> dict[str, float]:
    if not counts or any((value <= 0 for value in counts.values())):
        raise ValueError('All class counts must be positive')
    total = sum(counts.values())
    classes = len(counts)
    raw = {key: min(MAX_CLASS_WEIGHT, math.sqrt(total / (classes * value))) for key, value in counts.items()}
    weighted_mean = sum((counts[key] * raw[key] for key in counts)) / total
    return {key: raw[key] / weighted_mean for key in raw}

def example_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    targets: list[Optional[dict[str, Any]]] = []
    action_counts: collections.Counter[str] = collections.Counter()
    delivery_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row['task'] == 'ACTION':
            target = json.loads(row['target'])
            targets.append(target)
            action_counts[target['action']] += 1
            if target['action'] == 'CONTINUE_TECHNIQUE':
                delivery_counts[target['delivery_mode']] += 1
        else:
            targets.append(None)
    action_weights = normalized_inverse_sqrt_weights(action_counts)
    delivery_weights = normalized_inverse_sqrt_weights(delivery_counts)
    raw_action = []
    for target in targets:
        if target is None:
            continue
        value = action_weights[target['action']]
        if target['action'] == 'CONTINUE_TECHNIQUE':
            value *= delivery_weights[target['delivery_mode']]
        raw_action.append(min(MAX_CLASS_WEIGHT, value))
    action_mean = sum(raw_action) / len(raw_action)
    weights, position = ([], 0)
    for target in targets:
        if target is None:
            weights.append(1.0)
        else:
            weights.append(raw_action[position] / action_mean)
            position += 1
    return (weights, {'method': 'inverse-sqrt action and continuation-delivery weighting; cap 3.0', 'action_counts': dict(sorted(action_counts.items())), 'continue_delivery_counts': dict(sorted(delivery_counts.items())), 'response_weight': 1.0})

def run_training(files: dict[str, Path]) -> None:
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    token = False
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, local_files_only=True, token=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOAD_PATH, local_files_only=True, token=False, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.name_or_path = MODEL_ID
    model.config._name_or_path = MODEL_ID
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config._name_or_path = BASE_MODEL_ID
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model.print_trainable_parameters()
    raw = {split: read_jsonl(files[f'counsellor_{split}.jsonl']) for split in ('train', 'development', 'test')}
    train_weights, balance_report = example_weights(raw['train'])
    for row, weight in zip(raw['train'], train_weights):
        row['loss_weight'] = weight
    for split in ('development', 'test'):
        for row in raw[split]:
            row['loss_weight'] = 1.0

    def tokenize(row: Mapping[str, Any]) -> dict[str, Any]:
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': str(row['system'])}, {'role': 'user', 'content': str(row['user'])}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target_ids = tokenizer(str(row['target']) + (tokenizer.eos_token or ''), add_special_tokens=False)['input_ids']
        total = len(prompt_ids) + len(target_ids)
        if total > MAX_LENGTH:
            raise ValueError(f'{row['example_id']} requires {total} tokens, above {MAX_LENGTH}; refusing silent truncation')
        return {'input_ids': prompt_ids + target_ids, 'attention_mask': [1] * total, 'labels': [-100] * len(prompt_ids) + target_ids, 'loss_weight': float(row['loss_weight'])}
    datasets = {split: Dataset.from_list([tokenize(row) for row in rows]) for split, rows in raw.items()}
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')

    class WeightedCollator:

        def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            weights = torch.tensor([float(feature['loss_weight']) for feature in features], dtype=torch.float32)
            batch = base_collator([{key: value for key, value in feature.items() if key != 'loss_weight'} for feature in features])
            batch['example_weight'] = weights
            return batch

    class TargetOnlyWeightedTrainer(Trainer):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool=False, num_items_in_batch: Optional[Any]=None) -> Any:
            weights = inputs.pop('example_weight').to(model.device)
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            logits = outputs.logits[..., :-1, :].contiguous().float()
            shifted = labels[..., 1:].contiguous()
            losses = functional.cross_entropy(logits.view(-1, logits.size(-1)), shifted.view(-1), reduction='none', ignore_index=-100).view(shifted.shape)
            valid = shifted.ne(-100)
            per_example = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            loss = (per_example * weights).mean()
            return (loss, outputs) if return_outputs else loss

    class RecoveryAndMidpointCallback(TrainerCallback):

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            midpoint = max(1, int(state.max_steps * MID_EPOCH_EVAL_FRACTION + 0.5))
            if state.global_step == midpoint:
                control.should_evaluate = True
                control.should_save = True
            elif state.global_step > 0 and state.global_step % RECOVERY_SAVE_STEPS == 0:
                control.should_save = True
            return control
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'seed': SEED, 'epochs': NUM_TRAIN_EPOCHS, 'learning_rate': LEARNING_RATE, 'max_length': MAX_LENGTH, 'micro_batch_size': MICRO_BATCH_SIZE, 'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS, 'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT, 'target_only_loss': True, 'qwen_enable_thinking': False, 'rloo_base_model_contract': 'same_qwen3_revision', 'precision': 'BF16_LORA', 'effective_batch_size': MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 'optimizer': 'adamw_torch_fused', 'scheduler': 'cosine', 'warmup_ratio': WARMUP_RATIO, 'weight_decay': WEIGHT_DECAY, 'max_grad_norm': 1.0, 'recovery_save_steps': RECOVERY_SAVE_STEPS, 'mid_epoch_eval_fraction': MID_EPOCH_EVAL_FRACTION, 'checkpoint_retention': 2, 'balance_report': balance_report, 'training_file_hashes': {name: sha256_file(path) for name, path in files.items()}}
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest().upper()
    config_path = OUTPUT_DIR / 'RUN_CONFIG.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing.get('config_sha256') != config_hash:
            raise RuntimeError('Existing output belongs to another frozen configuration')
    else:
        config_path.write_text(json.dumps({**run_config, 'config_sha256': config_hash}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    arguments = TrainingArguments(output_dir=str(OUTPUT_DIR / 'checkpoints'), run_name='eeyore-counsellor-qwen3-2201', seed=SEED, data_seed=SEED, num_train_epochs=NUM_TRAIN_EPOCHS, learning_rate=LEARNING_RATE, lr_scheduler_type='cosine', warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY, max_grad_norm=1.0, per_device_train_batch_size=MICRO_BATCH_SIZE, per_device_eval_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, bf16=True, bf16_full_eval=True, fp16=False, tf32=True, optim='adamw_torch_fused', eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, logging_strategy='steps', logging_steps=10, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, report_to=['tensorboard'], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, save_safetensors=True)
    trainer = TargetOnlyWeightedTrainer(model=model, args=arguments, train_dataset=datasets['train'], eval_dataset=datasets['development'], processing_class=tokenizer, data_collator=WeightedCollator(), callbacks=[RecoveryAndMidpointCallback(), *EXTRA_CALLBACKS])
    checkpoint = get_last_checkpoint(str(OUTPUT_DIR / 'checkpoints'))
    result = trainer.train(resume_from_checkpoint=checkpoint)
    development = trainer.evaluate(datasets['development'], metric_key_prefix='development')
    test = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    adapter = OUTPUT_DIR / 'adapter'
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))

    def add_perplexity(metrics: dict[str, Any], key: str) -> None:
        loss = metrics.get(key)
        metrics[key.replace('loss', 'perplexity')] = math.exp(float(loss)) if loss is not None and float(loss) < 20 else None
    add_perplexity(development, 'development_loss')
    add_perplexity(test, 'test_loss')
    summary = {'status': 'COMPLETE', 'scope': 'synthetic_counselling_simulation_only', 'source_sessions': EXPECTED_SOURCE_SESSIONS, 'best_model_selected_on': 'development_loss', 'test_used_for_selection': False, 'train_metrics': result.metrics, 'development_metrics': development, 'test_metrics': test, 'config_sha256': config_hash, 'completed_at_unix': time.time(), 'best_checkpoint': trainer.state.best_model_checkpoint, 'best_metric': trainer.state.best_metric, 'optimizer_steps': trainer.state.global_step}
    (OUTPUT_DIR / 'SFT_COMPLETE.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
