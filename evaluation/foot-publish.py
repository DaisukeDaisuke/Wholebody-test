#!/usr/bin/env python3
"""Upload already exported ONNX artifacts; no PyTorch checkpoint is opened here."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Model publication runs in GitHub Actions only.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    token = next((os.environ.get(key) for key in ('HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACE_HUB_TOKEN') if os.environ.get(key)), None)
    if not token:
        raise RuntimeError('The repository needs a Hugging Face write token in Actions secrets.')
    api = HfApi(token=token)
    repo = 'RabbitDaisuke/Foot-Refinement-onnx'
    info = api.model_info(repo)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    if card.get('license') != 'agpl-3.0':
        raise RuntimeError('The existing target repository must declare agpl-3.0.')
    manifest_path = args.models / 'foot-models.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('repository') != repo or manifest.get('license') != 'agpl-3.0' or not manifest.get('models'):
        raise ValueError('Invalid publication manifest.')
    for model in manifest['models']:
        filename = model['filename']
        if Path(filename).name != filename or not filename.endswith('.onnx'):
            raise ValueError('Only top-level ONNX files are allowed.')
        content = (args.models / filename).read_bytes()
        if len(content) != model['bytes'] or hashlib.sha256(content).hexdigest() != model['sha256']:
            raise ValueError('Model checksum differs from the exported artifact.')
    commit = api.upload_folder(repo_id=repo, folder_path=str(args.models),
                               allow_patterns=['*.onnx', 'foot-models.json', 'README.md', 'LICENSE'],
                               commit_message='Publish verified FOCUS ONNX exports from GitHub Actions')
    for model in manifest['models']:
        model['url'] = f"https://huggingface.co/{repo}/resolve/{commit.oid}/{model['filename']}"
        model['revision'] = commit.oid
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    metadata = api.upload_file(repo_id=repo, path_or_fileobj=str(manifest_path), path_in_repo='foot-models.json',
                               commit_message='Pin foot model URLs to the uploaded weight revision')
    result = {'status': 'published', 'repository': repo, 'license': 'agpl-3.0',
              'weightsRevision': commit.oid, 'manifestRevision': metadata.oid,
              'gitCommit': os.environ.get('GITHUB_SHA'), 'workflowRunId': os.environ.get('GITHUB_RUN_ID'),
              'models': manifest['models']}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'published', 'repository': repo, 'weightsRevision': commit.oid, 'models': len(manifest['models'])}))


if __name__ == '__main__':
    main()
