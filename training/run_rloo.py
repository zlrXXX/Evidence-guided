import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from _inputs import verify_release_files

ROOT = Path(__file__).resolve().parent

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest().upper()

def read(path):
    return json.loads(Path(path).read_text())

def adapter(path, model):
    path = Path(path)
    config = read(path/'adapter_config.json')
    if config.get('base_model_name_or_path') != model:
        raise ValueError('Adapter base-model identity does not match the selected condition')
    return path, digest(path/'adapter_model.safetensors')

def patient_run(root, model):
    root = Path(root)
    completed = read(root/'COMPLETED.json')
    config = read(root/'RUN_CONFIG.json')
    summary = root/'training_summary.json'
    selected, selected_hash = adapter(root/'selected_patient_rl_adapter', model)
    if completed.get('status') != 'COMPLETE' or completed.get('selected_stage') != 'PATIENT_RLOO_ACCEPTED':
        raise ValueError('A completed, accepted patient RLOO checkpoint is required')
    if selected_hash != completed.get('selected_adapter_sha256') or digest(summary) != completed.get('training_summary_sha256'):
        raise ValueError('Patient checkpoint completion hashes do not match')
    if config.get('run_config_sha256') != completed.get('run_config_sha256'):
        raise ValueError('Patient run configuration does not match completion')
    return selected, selected_hash, completed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['cactus','eeyore'])
    parser.add_argument('--backbone', required=True, choices=['qwen3','llama31'])
    parser.add_argument('--role', required=True, choices=['counsellor','patient'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--sft-adapter', required=True, type=Path)
    parser.add_argument('--patient-rloo-run', type=Path)
    parser.add_argument('--resistance-tracker', required=True, type=Path)
    parser.add_argument('--expression-checker', type=Path)
    parser.add_argument('--evidence-tracker', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--context-checkpoint-sha256')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-api-cost', required=True, type=float)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    model = {'qwen3':'Qwen/Qwen3-8B','llama31':'meta-llama/Llama-3.1-8B-Instruct'}[args.backbone]
    source = ROOT/'rloo'/args.dataset/args.backbone/args.role
    data = (args.data_dir or ROOT.parent/'datasets'/args.dataset/'rloo'/args.role/args.backbone).resolve()
    names = ['patient_rloo_train.jsonl','patient_rloo_development.jsonl'] if args.role == 'patient' else ['tr.jsonl','dv.jsonl','dv64.jsonl','rag.jsonl']
    expected_rows = [3000,500] if args.role == 'patient' else [3000,500,64,None]
    metadata = {}
    for name, count in zip(names, expected_rows):
        path = data/name
        rows = sum(1 for line in path.open() if line.strip())
        if count is not None and rows != count:
            raise ValueError(f'{name}: expected {count} rows, received {rows}')
        metadata[name] = {'bytes':path.stat().st_size,'sha256':digest(path),'rows':rows}
    verify_release_files(args.dataset, 'rloo', args.role, args.backbone, metadata)
    sft, sft_hash = adapter(args.sft_adapter.resolve(), model)
    if args.role == 'patient' and not args.context_checkpoint_sha256:
        parser.error('--context-checkpoint-sha256 is required for the frozen counsellor that produced the patient training contexts')
    if args.role == 'patient' and not args.expression_checker:
        parser.error('--expression-checker is required for patient RLOO')
    if args.role == 'counsellor' and (not args.evidence_tracker or not args.patient_rloo_run):
        parser.error('--evidence-tracker and --patient-rloo-run are required for counsellor RLOO')
    if not args.model_dir and (args.backbone == 'llama31' and args.dataset == 'cactus' or args.backbone == 'qwen3' and args.dataset == 'eeyore'):
        parser.error('--model-dir is required for this training core')
    binding = {'dataset':args.dataset,'backbone':args.backbone,'role':args.role,'data':metadata,'sft_adapter_sha256':sft_hash,'code':{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'},'max_api_cost':args.max_api_cost}
    selected = selected_hash = completed = None
    if args.role == 'counsellor':
        selected, selected_hash, completed = patient_run(args.patient_rloo_run.resolve(), model)
        binding['patient_rloo_adapter_sha256'] = selected_hash
    if args.validate_only:
        print(json.dumps(binding, indent=2))
        return
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY for the semantic reward judge')
    if args.max_api_cost <= 0:
        raise ValueError('A positive API cost ceiling is required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bind_path = output/'input_binding.json'
    encoded = json.dumps(binding, sort_keys=True, indent=2)+'\n'
    if bind_path.exists() and bind_path.read_text() != encoded:
        raise ValueError('Existing output is bound to different inputs/settings')
    bind_path.write_text(encoded)
    bundle = output/'runtime_bundle'
    shutil.copytree(source, bundle, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (bundle/'data').mkdir(exist_ok=True)
    for name in names:
        link = bundle/'data'/name
        if not link.exists():
            link.symlink_to(data/name)
    if args.expression_checker:
        target = bundle/'trackers'/'resistance_expression'
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.symlink_to(args.expression_checker.resolve())
    members = {str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and p.name not in ['bundle_manifest.json','COMPLETED.json','code_integrity.json']}
    manifest = {'status':'LOCAL_STRUCTURE_VALIDATED_GPU_AND_API_NOT_RUN','counts':{'train':3000,'development':500,'test':0},'members':members}
    (bundle/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (bundle/'COMPLETED.json').write_text(json.dumps({'bundle_manifest_sha256':digest(bundle/'bundle_manifest.json')})+'\n')
    (bundle/'code_integrity.json').write_text(json.dumps(binding['code'],indent=2)+'\n')
    for prefix in ['CCRL','CRL5']:
        os.environ[prefix+'_BUNDLE_ROOT'] = str(bundle)
        os.environ[prefix+'_OUTPUT_ROOT'] = str(output/'runs')
        os.environ[prefix+'_COUNSELLOR_SFT_DIR'] = str(sft)
        os.environ[prefix+'_RESISTANCE_TRACKER_DIR'] = str(args.resistance_tracker.resolve())
        os.environ[prefix+'_MAX_LUNA_COST_USD'] = str(args.max_api_cost)
        if args.patient_rloo_run:os.environ[prefix+'_PATIENT_RLOO_RUN'] = str(args.patient_rloo_run.resolve())
        if args.evidence_tracker:os.environ[prefix+'_CBT_EVIDENCE_TRACKER_DIR'] = str(args.evidence_tracker.resolve())
        if args.model_dir:os.environ[prefix+'_MODEL_LOAD_PATH'] = str(args.model_dir.resolve())
    os.environ.update(PRL5_BUNDLE_ROOT=str(bundle),PRL5_BUNDLE_MANIFEST_SHA256=digest(bundle/'bundle_manifest.json'),CPRL_OUTPUT_ROOT=str(output/'runs'),CPRL_PATIENT_SFT_DIR=str(sft),CPRL_PATIENT_SFT_SHA256=sft_hash,CPRL_FROZEN_COUNSELLOR_CONTEXTS_SHA256=args.context_checkpoint_sha256 or '',CERTIFY_CBT_RESISTANCE_TRACKER_DIR=str(args.resistance_tracker.resolve()),CPRL_CHECKPOINT_STAGE=str(output/'checkpoint_stage'),PRL5_MAX_LLM_COST_USD=str(args.max_api_cost))
    if args.model_dir:os.environ['CPRL_MODEL_LOAD_PATH']=str(args.model_dir.resolve())
    sys.path.insert(0,str(bundle))
    script = bundle/('train_patient_rloo.py' if (bundle/'train_patient_rloo.py').exists() else 'train.py')
    spec = importlib.util.spec_from_file_location('trace_training_run',script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.role == 'patient':
        if hasattr(module,'configure'):
            module.configure()
            module = module.legacy
        module.OUTPUT_ROOT = output/'runs'
        module.PATIENT_SFT_ROOT = sft
        module.EXPECTED_PATIENT_SFT_SHA256 = sft_hash
        module.EXPECTED_COUNSELLOR_SFT_SHA256 = args.context_checkpoint_sha256.upper()
        module.locate_patient_adapter = lambda:adapter(sft,model)[0]
    else:
        module.EXPECTED_COUNSELLOR_SFT_SHA256 = sft_hash
        module.EXPECTED_PATIENT_RLOO_SHA256 = selected_hash
        module.EXPECTED_PATIENT_RUN_CONFIG_SHA256 = completed['run_config_sha256']
        module.EXPECTED_PATIENT_TRAINING_SUMMARY_SHA256 = completed['training_summary_sha256']
        module.locate_patient_adapter = lambda root:patient_run(root,model)[0]
        if hasattr(module,'bind_sft'):
            module.bind_sft = lambda path,expected:adapter(path,model)
    module.main()

if __name__ == '__main__':
    main()
