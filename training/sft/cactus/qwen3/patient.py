from __future__ import annotations
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
EXPECTED_SOURCE_SESSIONS = 2002
OUTPUT_DIR = Path('Path to corresponding file')
SEED = 42
MAX_LENGTH = 16384
NUM_TRAIN_EPOCHS = 2.0
LEARNING_RATE = 2e-05
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 12
RECOVERY_SAVE_STEPS = 250
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    device_name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if 'A100' not in device_name or total_gib < 70:
        raise RuntimeError(f'Expected an A100 80GB runtime, found {device_name} ({total_gib:.1f} GiB)')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required')
    print(f'Runtime: {device_name}, {total_gib:.1f} GiB')

def tokenize_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    eos = tokenizer.eos_token or ''
    for row in rows:
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': str(row['system'])}, {'role': 'user', 'content': str(row['user'])}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target_ids = tokenizer(str(row['target']) + eos, add_special_tokens=False)['input_ids']
        input_ids = prompt_ids + target_ids
        if len(input_ids) > MAX_LENGTH:
            raise ValueError(f'Example {row['example_id']} has {len(input_ids)} tokens, exceeding {MAX_LENGTH}; refusing silent truncation')
        encoded.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids), 'labels': [-100] * len(prompt_ids) + target_ids})
    return encoded

def run_training(files: dict[str, Path]) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed
    from getpass import getpass
    from huggingface_hub import login
    token = (os.environ.get('HF_TOKEN') or '').strip() or None
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config._name_or_path = BASE_MODEL_ID
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    raw = {split: read_jsonl(files[f'patient_{split}.jsonl']) for split in ('train', 'development', 'test')}
    tokenized = {split: Dataset.from_list(tokenize_rows(rows, tokenizer)) for split, rows in raw.items()}
    config = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'seed': SEED, 'epochs': NUM_TRAIN_EPOCHS, 'learning_rate': LEARNING_RATE, 'max_length': MAX_LENGTH, 'micro_batch_size': MICRO_BATCH_SIZE, 'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS, 'recovery_save_steps': RECOVERY_SAVE_STEPS, 'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT, 'target_only_loss': True, 'qwen_enable_thinking': False, 'rloo_base_model_contract': 'same_qwen3_revision', 'training_file_hashes': {name: sha256_file(path) for name, path in files.items()}}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest().upper()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / 'RUN_CONFIG.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous.get('config_sha256') != config_hash:
            raise RuntimeError('Existing output belongs to a different frozen run configuration')
    else:
        config_path.write_text(json.dumps({**config, 'config_sha256': config_hash}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')

    class RecoveryCheckpointCallback(TrainerCallback):

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step > 0 and state.global_step % RECOVERY_SAVE_STEPS == 0:
                control.should_save = True
            return control

    def latest_complete_checkpoint(checkpoint_dir: Path) -> Path | None:
        checkpoints = sorted(checkpoint_dir.glob('checkpoint-*'), key=lambda path: int(path.name.split('-')[-1]), reverse=True)
        for checkpoint in checkpoints:
            required = (checkpoint / 'trainer_state.json', checkpoint / 'optimizer.pt', checkpoint / 'scheduler.pt', checkpoint / 'adapter_config.json')
            adapter_present = any(((checkpoint / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')))
            if adapter_present and all((path.is_file() for path in required)):
                return checkpoint
        return None
    arguments = TrainingArguments(output_dir=str(OUTPUT_DIR / 'checkpoints'), num_train_epochs=NUM_TRAIN_EPOCHS, per_device_train_batch_size=MICRO_BATCH_SIZE, per_device_eval_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, lr_scheduler_type='cosine', bf16=True, tf32=True, gradient_checkpointing=True, eval_strategy='epoch', eval_steps=500, save_strategy='epoch', save_steps=RECOVERY_SAVE_STEPS, save_only_model=False, save_total_limit=2, logging_steps=10, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, report_to=['tensorboard'], seed=SEED, data_seed=SEED, remove_unused_columns=False, optim='adamw_torch_fused')
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized['train'], eval_dataset=tokenized['development'], data_collator=collator, callbacks=[RecoveryCheckpointCallback()])
    checkpoint_dir = OUTPUT_DIR / 'checkpoints'
    resume_checkpoint = latest_complete_checkpoint(checkpoint_dir)
    if resume_checkpoint is not None:
        print(f'Resuming from complete recovery checkpoint: {resume_checkpoint}')
    result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None)
    development = trainer.evaluate(tokenized['development'], metric_key_prefix='development')
    test = trainer.evaluate(tokenized['test'], metric_key_prefix='test')
    adapter_dir = OUTPUT_DIR / 'adapter'
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    def add_perplexity(metrics: dict[str, Any], key: str) -> None:
        loss = metrics.get(key)
        metrics[key.replace('loss', 'perplexity')] = math.exp(float(loss)) if loss is not None and float(loss) < 20 else None
    add_perplexity(development, 'development_loss')
    add_perplexity(test, 'test_loss')
    summary = {'status': 'COMPLETE', 'scope': 'synthetic_counselling_simulation_only', 'source_sessions': EXPECTED_SOURCE_SESSIONS, 'learning_rate': LEARNING_RATE, 'recovery_save_steps': RECOVERY_SAVE_STEPS, 'target_only_loss': True, 'qwen_enable_thinking': False, 'rloo_base_model_contract': 'same_qwen3_revision', 'train_metrics': result.metrics, 'development_metrics': development, 'test_metrics': test, 'config_sha256': config_hash, 'completed_at_unix': time.time()}
    (OUTPUT_DIR / 'SFT_COMPLETE.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
