import argparse
import hashlib
import importlib.util
import json
import os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['cactus', 'eeyore'])
    parser.add_argument('--backbone', required=True, choices=['qwen3', 'llama31'])
    parser.add_argument('--role', required=True, choices=['counsellor', 'patient'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    data = (args.data_dir or ROOT.parent/'datasets'/args.dataset/'sft'/args.role).resolve()
    files = {f'{args.role}_{split}.jsonl': data/f'{args.role}_{split}.jsonl' for split in ['train','development','test']}
    manifest = {'files': {name: {'sha256': digest(path), 'bytes': path.stat().st_size, 'rows': sum(1 for line in path.open() if line.strip())} for name,path in files.items()}}
    verify_release_files(args.dataset, 'sft', args.role, args.backbone, manifest['files'])
    settings = json.loads((ROOT/'sft_settings.json').read_text())[f'{args.dataset}/{args.backbone}/{args.role}']
    source = ROOT/'sft'/args.dataset/args.backbone/f'{args.role}.py'
    manifest.update({'training_source_sha256':digest(source),'settings':settings})
    if args.validate_only:
        print(json.dumps(manifest, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable CUDA device required')
    spec = importlib.util.spec_from_file_location('training_core', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = str(args.model_dir.resolve()) if args.model_dir else settings['MODEL_ID']
    if args.model_dir is None and 'MODEL_LOAD_PATH' in source.read_text():
        raise ValueError('This training core requires --model-dir pointing to the documented base-model snapshot')
    module.BASE_MODEL_ID = settings['MODEL_ID']
    module.MODEL_LOAD_PATH = model
    module.OUTPUT_DIR = args.output.resolve()
    module.OUTPUT_ROOT = str(args.output.resolve())
    module.EXPECTED_FILES = manifest['files']
    module.EXPECTED_EXECUTING_TRAINER_CANONICAL_SHA256 = digest(source)
    module.EXPECTED_SOURCE_SESSIONS = 2002 if args.dataset == 'cactus' else 2201
    module.EXPECTED_SOURCE_CORPUS_SHA256 = None
    module.EXPECTED_RAG_SHA256 = None
    if args.model_dir and 'MODEL_LOAD_PATH' not in source.read_text():
        module.MODEL_ID = model
        module.MODEL_REVISION = None
    args.output.mkdir(parents=True, exist_ok=True)
    binding = args.output/'input_binding.json'
    encoded = json.dumps(manifest, sort_keys=True, indent=2)+'\n'
    if binding.exists() and binding.read_text() != encoded:
        raise ValueError('Existing output is bound to different inputs/settings')
    binding.write_text(encoded)
    if hasattr(module, 'run_training'):
        if hasattr(module, 'require_runtime'):
            module.require_runtime()
        module.run_training(files)
    else:
        gpu = {'name':torch.cuda.get_device_name(0),'memory_gib':torch.cuda.get_device_properties(0).total_memory/(1024**3),'bf16_supported':True}
        module.train_role(args.role, data, digest(binding), manifest, os.environ.get('HF_TOKEN'), gpu)

if __name__ == '__main__':
    main()
