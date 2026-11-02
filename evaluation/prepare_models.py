#!/usr/bin/env python3
"""Acquire upstream ONNX unchanged and publish only through GitHub Actions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
import onnx
from benchmark import BODY_MODEL, HAND_MODEL, resolve_model, fetch

REPOSITORY = 'RabbitDaisuke/Wholebody-Refinement-onnx'
SPECS = [('rtmw-l', BODY_MODEL, 'rtmw-l-384x288-fp32.onnx', [288,384], 133),
         ('hand5', HAND_MODEL, 'rtmpose-hand5-256x256-fp32.onnx', [256,256], 21)]
ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'
VENDOR_FILES = ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']

def prepare(cache: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'schemaVersion': 1, 'repository': REPOSITORY, 'license': 'Apache-2.0',
                'modelOrigin': 'Existing MMPose pretrained ONNX; no new training is claimed.', 'models': []}
    for model_id, source, filename, expected_wh, expected_keypoints in SPECS:
        path, metadata = resolve_model(source, cache, model_id)
        graph = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(graph)
        shape = [d.dim_value if d.dim_value else d.dim_param for d in graph.graph.input[0].type.tensor_type.shape.dim]
        if shape[-2:] != expected_wh[::-1]:
            raise ValueError(f'{model_id}: unexpected image shape {shape}')
        outputs = [{'name': value.name, 'shape': [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]}
                   for value in graph.graph.output]
        if len(outputs) != 2 or any(value['shape'][1] != expected_keypoints for value in outputs):
            raise ValueError(f'{model_id}: unexpected keypoint outputs {outputs}')
        shutil.copyfile(path, destination / filename)
        manifest['models'].append({'id': model_id, 'filename': filename, 'inputWH': expected_wh,
                                  'keypointCount': expected_keypoints, 'dimensions': 2, 'dtype': 'float32',
                                  'inputName': graph.graph.input[0].name, 'outputs': outputs,
                                  'sha256': metadata['onnxSha256'], 'bytes': metadata['bytes'],
                                  'upstreamURL': source, 'upstreamArchiveSha256': metadata['archiveSha256'],
                                  'license': 'Apache-2.0', 'color': 'RGB', 'mean': [123.675,116.28,103.53],
                                  'std': [58.395,57.12,57.375], 'padding': 1.25, 'simccSplitRatio': 2})
    (destination / 'models.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    fetch('https://raw.githubusercontent.com/open-mmlab/mmpose/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/LICENSE', destination / 'LICENSE')
    card = '''---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: keypoint-detection
tags: [onnx, mmpose, wholebody, hand-pose]
---
# Wholebody Refinement ONNX
Unmodified official MMPose ONNX models for full-body and native-resolution hand ROI inference.
The source URLs, SHA-256 hashes, input/output tensors, RGB normalization and crop contract are in models.json.
RTMW-l: 133 keypoints, 384x288. RTMPose Hand5: 21 keypoints, 256x256.
These are existing pretrained models, not newly trained models.
Foot landmarks are big toe, small toe and heel per foot. Individual toe articulation is not observed.
All experiments and uploads run through https://github.com/DaisukeDaisuke/Wholebody-test .
The repository license covers the original MMPose model distribution. Evaluation images retain their individual licenses and are not included here.
'''
    (destination / 'README.md').write_text(card, encoding='utf-8')
    return manifest

def publish(destination: Path, manifest: dict, report: Path) -> dict:
    from huggingface_hub import HfApi
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not token:
        raise RuntimeError('The Actions repository needs an HF_TOKEN secret with write access to ' + REPOSITORY)
    api = HfApi(token=token)
    info = api.model_info(REPOSITORY)
    license_id = (info.card_data or {}).get('license')
    if license_id != 'apache-2.0':
        raise RuntimeError('The existing Hugging Face repository must explicitly use Apache-2.0.')
    uploaded = api.upload_folder(repo_id=REPOSITORY, repo_type='model', folder_path=str(destination),
                                 ignore_patterns=['models.json'], commit_message='Add unchanged official RTMW-l and Hand5 ONNX')
    revision = uploaded.oid
    for model in manifest['models']:
        model['revision'] = revision
        model['url'] = f'https://huggingface.co/{REPOSITORY}/resolve/{revision}/{model["filename"]}'
    (destination / 'models.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    manifest_commit = api.upload_file(repo_id=REPOSITORY, repo_type='model', path_or_fileobj=str(destination / 'models.json'),
                                      path_in_repo='models.json', commit_message='Pin model manifest to uploaded weight revision')
    result = {'repository': REPOSITORY, 'weightsRevision': revision, 'manifestRevision': manifest_commit.oid,
              'published': True, 'models': manifest['models']}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'repository': REPOSITORY, 'published': True, 'revision': revision}))
    return result

def vendor(dist: Path):
    directory = dist / 'vendor'
    directory.mkdir(parents=True, exist_ok=True)
    for name in VENDOR_FILES:
        fetch(ORT_BASE + name, directory / name, limit=50 * 1024 * 1024)
    fetch('https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/ThirdPartyNotices.txt', directory / 'ThirdPartyNotices.txt')
    fetch('https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE', directory / 'LICENSE.txt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('.cache/wholebody'))
    parser.add_argument('--output', type=Path, default=Path('model-package'))
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    manifest = prepare(args.cache, args.output)
    vendor(args.dist)
    if args.publish:
        publish(args.output, manifest, Path('evaluation/results/publication.json'))
    shutil.copyfile(args.output / 'models.json', args.dist / 'models-manifest.json')

if __name__ == '__main__':
    main()
