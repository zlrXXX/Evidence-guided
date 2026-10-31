import hashlib
import json
from pathlib import Path

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest().upper()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def config_digest(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest().upper()

def bind_sft(adapter, expected):
    adapter = Path(adapter)
    run = adapter.parent
    stored = read(run / 'RUN_CONFIG.json')
    config = stored['config']
    config_sha = config_digest(config)
    completion = read(run / 'COMPLETED.json')
    if stored.get('config_sha256') != config_sha or completion.get('config_sha256') != config_sha:
        raise ValueError('SFT completion/config binding mismatch')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'SFT configuration mismatch: {key}')
    summary = run / 'training_summary.json'
    if digest(summary) != completion.get('training_summary_sha256'):
        raise ValueError('SFT training summary hash mismatch')
    details = read(summary)
    if details.get('status') != 'COMPLETE' or details.get('config_sha256') != config_sha:
        raise ValueError('SFT training is not complete for this configuration')
    weights = digest(adapter / 'adapter_model.safetensors')
    if weights != completion.get('adapter_sha256') or weights != details.get('adapter_sha256'):
        raise ValueError('SFT adapter hash mismatch')
    if read(adapter / 'adapter_config.json').get('base_model_name_or_path') != expected['model_id']:
        raise ValueError('SFT adapter backbone mismatch')
    return (adapter, weights)

def bind_patient_rloo(root, expected_manifest):
    root = Path(root)
    candidates = [root] if (root / 'COMPLETED.json').is_file() else sorted(root.glob('run_*'))
    matches = []
    for run in candidates:
        if not (run / 'COMPLETED.json').is_file() or not (run / 'RUN_CONFIG.json').is_file():
            continue
        stored = read(run / 'RUN_CONFIG.json')
        config = stored['config']
        if config.get('bundle_manifest_sha256') != expected_manifest:
            continue
        config_sha = config_digest(config)
        completed = read(run / 'COMPLETED.json')
        if stored.get('run_config_sha256') != config_sha or completed.get('run_config_sha256') != config_sha:
            raise ValueError('Patient RLOO config binding mismatch')
        if completed.get('status') != 'COMPLETE':
            raise ValueError('Patient RLOO is not complete')
        adapter = run / 'selected_patient_rl_adapter'
        weights = digest(adapter / 'adapter_model.safetensors')
        if weights != completed.get('selected_adapter_sha256'):
            raise ValueError('Patient RLOO adapter hash mismatch')
        for file, field in [('training_summary.json', 'training_summary_sha256'), ('training_log.jsonl', 'training_rollout_dataset_sha256')]:
            if digest(run / file) != completed.get(field):
                raise ValueError(f'Patient RLOO output hash mismatch: {file}')
        if completed.get('training_rollout_prompt_rows') != 3000 or completed.get('training_rollout_responses') != 12000:
            raise ValueError('Patient RLOO rollout count mismatch')
        if read(adapter / 'adapter_config.json').get('base_model_name_or_path') != 'meta-llama/Llama-3.1-8B-Instruct':
            raise ValueError('Patient RLOO backbone mismatch')
        matches.append((adapter, weights))
    if len(matches) != 1:
        raise ValueError(f'Expected one completed patient RLOO from the current bundle, found {len(matches)}')
    return matches[0]
