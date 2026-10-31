import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def verify_release_files(dataset, stage, role, backbone, observed):
    root = Path(__file__).resolve().parent.parent/'datasets'
    manifest = json.loads((root/'data_manifest.json').read_text())
    entries = {item['output']:item for item in manifest['files']}
    aliases = {item['output']:item['source'] for item in manifest.get('aliases',[])}
    directory = f'{dataset}/{stage}/{role}'
    if stage == 'rloo':
        directory += '/'+backbone
    for name, value in observed.items():
        key = directory+'/'+name
        key = aliases.get(key,key)
        if key not in entries:
            raise FileNotFoundError(f'No released data record for {key}')
        expected = entries[key]
        if value['sha256'].lower() != expected['sha256'].lower() or value['bytes'] != expected['bytes'] or value['rows'] != expected['rows']:
            raise ValueError(f'Released data integrity mismatch: {key}')


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def within(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a relative dataset path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its directory")
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError("Dataset paths must not contain symbolic links")
    return path


def unpack(source_root, output_root, item, verify_only):
    value = hashlib.sha256()
    size = 0
    lines = 0
    final_byte = b""
    target = within(output_root, item["output"])
    temporary = None
    stream = None
    try:
        if not verify_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".unpack-", dir=target.parent)
            stream = os.fdopen(descriptor, "wb")
        for part in item["parts"]:
            path = within(source_root, part["path"])
            if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                raise ValueError("Compressed dataset hash mismatch: " + part["path"])
            with gzip.open(path, "rb") as archive:
                for block in iter(lambda: archive.read(1024 * 1024), b""):
                    value.update(block)
                    size += len(block)
                    lines += block.count(b"\n")
                    final_byte = block[-1:]
                    if stream is not None:
                        stream.write(block)
        lines += int(bool(final_byte) and final_byte != b"\n")
        if value.hexdigest() != item["sha256"] or size != item["bytes"] or lines != item["rows"]:
            raise ValueError("Decompressed dataset mismatch: " + item["output"])
        if stream is not None:
            stream.close()
            stream = None
            if target.exists():
                if digest(target) != item["sha256"]:
                    raise FileExistsError("Existing dataset differs: " + str(target))
            else:
                os.replace(temporary, target)
                temporary = None
    finally:
        if stream is not None:
            stream.close()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent.parent / "datasets" / "data_manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_root = args.manifest.resolve().parent
    output_root = (args.output or source_root).resolve()
    if manifest["format"] != "deterministic-gzip-jsonl-v1":
        raise ValueError("Unknown dataset format")
    outputs = [item["output"] for item in manifest["files"]]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate dataset destinations")
    total = sum(unpack(source_root, output_root, item, args.verify_only) for item in manifest["files"])
    if not args.verify_only:
        for alias in manifest.get("aliases", []):
            source = within(output_root, alias["source"])
            target = within(output_root, alias["output"])
            if target.exists():
                if digest(source) != digest(target):
                    raise FileExistsError("Existing dataset alias differs: " + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    print(json.dumps({"status": "verified" if args.verify_only else "prepared", "files": len(outputs), "rows": total}))


if __name__ == "__main__":
    main()
