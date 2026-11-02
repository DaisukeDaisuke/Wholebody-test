"""Prepare the pinned public TAPIR inputs without expiring Actions artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REPOSITORY = 'RabbitDaisuke/Foot-Point-Tracker-onnx'
MANIFEST_REVISION = 'ac66a7e5df280cb9986a417de79bc2c1b997c233'
WEIGHTS_REVISION = '4751f96d5413e48c05e63da133a5b4b1e6074c12'
EXPORT_COMMIT = '274407d44a7fbdcb8411f9dc78286b1f49409c6d'
PUBLIC_IMAGE = 'https://raw.githubusercontent.com/open-mmlab/mmpose/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/tests/data/coco/000000000785.jpg'
SOURCE_ROOT = Path(__file__).resolve().parent.parent


def fetch(url, destination, maximum_bytes, expected_hash=None, expected_bytes=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read(maximum_bytes + 1)
    if len(content) > maximum_bytes or (expected_bytes is not None and len(content) != expected_bytes):
        raise ValueError('Unexpected byte count for ' + destination.name)
    if expected_hash is not None and hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError('Checksum mismatch for ' + destination.name)
    destination.write_bytes(content)
    return content


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Prepare this public compatibility fixture in GitHub Actions only.')
    import cv2
    if cv2.__version__ != '4.11.0':
        raise RuntimeError('Use the pinned OpenCV decoder for the reference RGB bytes.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cache = (SOURCE_ROOT / '.cache').resolve()
    if cache not in root.parents:
        raise ValueError('Public preparation must write beneath the repository .cache directory.')
    models = root / 'models'
    fixture_root = root / 'results' / 'codec-fixture'
    fixture_root.mkdir(parents=True, exist_ok=True)
    pins = (SOURCE_ROOT / 'dist/tapir-models.mjs').read_text(encoding='utf-8')
    if "TAPIR_REPOSITORY='" + REPOSITORY + "'" not in pins or "TAPIR_MANIFEST_REVISION='" + MANIFEST_REVISION + "'" not in pins:
        raise ValueError('CI preparation and the shipped app must use the same published model revision.')
    manifest = json.loads(fetch(f'https://huggingface.co/{REPOSITORY}/resolve/{MANIFEST_REVISION}/tapir-models.json', models / 'tapir-models.json', 1024 * 1024))
    if manifest.get('repository') != REPOSITORY or manifest.get('license') != 'apache-2.0' or manifest.get('gitCommit') != EXPORT_COMMIT or manifest.get('parityPassed') is not True or manifest.get('codecParityPassed') is not True:
        raise ValueError('The published model does not match the verified public export.')
    if {model.get('id') for model in manifest.get('models', [])} != {'encoder', 'query', 'step'} or len(manifest['models']) != 3:
        raise ValueError('Exactly three model graphs are required.')
    for model in manifest['models']:
        name = f"causal-bootstapir-{model['id']}-256-q3-fp32.onnx"
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{WEIGHTS_REVISION}/{name}'
        if model.get('filename') != name or model.get('revision') != WEIGHTS_REVISION or model.get('url') != url:
            raise ValueError('Unexpected published model location.')
        fetch(url, models / name, 160 * 1024 * 1024, model['sha256'], model['bytes'])
    fixture = json.loads((SOURCE_ROOT / 'evaluation/fixtures/tapir-public-reference.json').read_text(encoding='utf-8'))
    attribution = fixture.get('attribution', {})
    if attribution.get('id') != 785 or attribution.get('license') != 4 or attribution.get('privateMedia') is not False or attribution.get('retrievedFrom') != PUBLIC_IMAGE:
        raise ValueError('Only the fixed public CC BY image is allowed.')
    image_path = root / 'public-reference.jpg'
    fetch(PUBLIC_IMAGE, image_path, 5 * 1024 * 1024, attribution['sha256'])
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError('The public image cannot be decoded.')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if list(rgb.shape) != [fixture['height'], fixture['width'], 3]:
        raise ValueError('Unexpected public image shape.')
    content = rgb.tobytes()
    if hashlib.sha256(content).hexdigest() != fixture['rawRgbSha256']:
        raise ValueError('RGB bytes differ from the original official comparison fixture.')
    (fixture_root / 'public-rgb.u8').write_bytes(content)
    (fixture_root / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'publicOnly': True, 'modelManifestRevision': MANIFEST_REVISION, 'modelWeightsRevision': WEIGHTS_REVISION,
                      'referenceWorkflowRun': '41384892970', 'imageId': 785, 'rgbSha256': fixture['rawRgbSha256']}))


if __name__ == '__main__':
    main()
