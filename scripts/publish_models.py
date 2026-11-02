"""Publish verified upstream models, never user media, from GitHub Actions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

import cv2
from huggingface_hub import HfApi
import mediapipe as mp
import numpy as np
import onnx
import onnxruntime as ort


REPOSITORY = "RabbitDaisuke/Wholebody-Video-Tracking"
GOOGLE = "https://storage.googleapis.com/mediapipe-models"
MP_REVISION = "v0.10.21"
MM_REVISION = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
SOURCES = {
    "pose_full": {
        "file": "pose_landmarker_full.task", "format": "mediapipe-task",
        "url": f"{GOOGLE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "kind": "pose", "variant": "full", "license": "Apache-2.0",
    },
    "pose_lite": {
        "file": "pose_landmarker_lite.task", "format": "mediapipe-task",
        "url": f"{GOOGLE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "kind": "pose", "variant": "lite", "license": "Apache-2.0",
    },
    "hands": {
        "file": "hand_landmarker.task", "format": "mediapipe-task",
        "url": f"{GOOGLE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "kind": "hands", "license": "Apache-2.0",
    },
    "dwpose_m_wholebody": {
        "file": "dwpose_m_wholebody.onnx", "format": "onnx",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
        "kind": "wholebody2d", "variant": "rtmpose-m-dwpose", "archive": True,
        "license": "Apache-2.0",
    },
    "rtmpose_s_wholebody": {
        "file": "rtmpose_s_wholebody.onnx", "format": "onnx",
        "url": "https://huggingface.co/RabbitDaisuke/RTMPose-Wholebody-onnx/resolve/9d7f318c43452baf33f1b39d39c1f8c03512aeb5/rtmpose-s-256x192-fp32.onnx",
        "kind": "wholebody2d", "variant": "rtmpose-s-dwpose",
        "sha256": "0dd72a9932d5f6eb539d02b3f5825e2ac5c15aeada5c0ff2dd9d5b34ab4103e8",
        "license": "Apache-2.0",
    },
}
LICENSES = {
    "LICENSE-MEDIAPIPE.txt": f"https://raw.githubusercontent.com/google-ai-edge/mediapipe/{MP_REVISION}/LICENSE",
    "LICENSE-MMPose.txt": f"https://raw.githubusercontent.com/open-mmlab/mmpose/{MM_REVISION}/LICENSE",
}
FIXTURES = {
    "pose": "https://storage.googleapis.com/mediapipe-assets/pose.jpg",
    "hands": "https://storage.googleapis.com/mediapipe-assets/thumb_up.jpg",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, limit: int = 250_000_000) -> bytes:
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "wholebody-video-model-publication/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f"Public download HTTP {response.status}")
                data = response.read(limit + 1)
            if not data or len(data) > limit:
                raise RuntimeError("Invalid public asset size")
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def public_image(kind: str) -> tuple[mp.Image, np.ndarray, dict]:
    data = download(FIXTURES[kind], 10_000_000)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Cannot decode public fixture")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), rgb, {
        "url": FIXTURES[kind], "sha256": digest(data),
        "source": f"https://github.com/google-ai-edge/mediapipe/tree/{MP_REVISION}/mediapipe/tasks/python/test/vision",
        "groundTruthAccuracyMeasured": False,
    }


def verify_tasks(path: Path, kind: str, image: mp.Image) -> tuple[dict, list]:
    base = mp.tasks.BaseOptions(model_asset_path=str(path), delegate=mp.tasks.BaseOptions.Delegate.CPU)
    start = time.perf_counter()
    if kind == "pose":
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base, running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=.5,
            min_pose_presence_confidence=.5, output_segmentation_masks=False)
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
            result = model.detect(image)
        points, world, expected = result.pose_landmarks, result.pose_world_landmarks, 33
    else:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base, running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=.5, min_hand_presence_confidence=.5)
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as model:
            result = model.detect(image)
        points, world, expected = result.hand_landmarks, result.hand_world_landmarks, 21
    elapsed = (time.perf_counter() - start) * 1000
    if not points or len(points) != len(world):
        raise RuntimeError(f"No {kind} detected in upstream public fixture")
    for group in points + world:
        if len(group) != expected or not all(np.isfinite([p.x, p.y, p.z]).all() for p in group):
            raise RuntimeError(f"Invalid {kind} landmark output")
    return {
        "runtime": "MediaPipe Python Tasks CPU", "detectedInstances": len(points),
        "landmarksPerInstance": expected, "worldLandmarksFinite": True,
        "initializationAndSingleInferenceMs": round(elapsed, 3),
        "groundTruthAccuracyMeasured": False, "browserGpuMeasured": False,
    }, points[0]


def wholebody_crop(rgb: np.ndarray, body: list) -> np.ndarray:
    height, width = rgb.shape[:2]
    observed = [p for p in body if (p.visibility or 0) >= .4]
    if len(observed) < 8:
        raise RuntimeError("Public fixture has insufficient visible body points")
    xs, ys = [p.x * width for p in observed], [p.y * height for p in observed]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    crop_w, crop_h = (max(xs) - min(xs)) * 1.25, (max(ys) - min(ys)) * 1.25
    if crop_w / crop_h > .75:
        crop_h = crop_w / .75
    else:
        crop_w = crop_h * .75
    transform = np.array([[192 / crop_w, 0, (crop_w / 2 - cx) * 192 / crop_w],
                          [0, 256 / crop_h, (crop_h / 2 - cy) * 256 / crop_h]], np.float32)
    cropped = cv2.warpAffine(rgb, transform, (192, 256), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    array = (cropped.astype(np.float32) - np.array([123.675, 116.28, 103.53], np.float32))
    array /= np.array([58.395, 57.12, 57.375], np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def verify_onnx(path: Path, image: np.ndarray) -> tuple[dict, dict]:
    graph = onnx.load(str(path), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise RuntimeError("External tensor file is outside the publication contract")
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if len(runtime.get_inputs()) != 1:
        raise RuntimeError("Expected one RGB image input")
    input_spec = runtime.get_inputs()[0]
    if input_spec.shape[1:] != [3, 256, 192] or input_spec.type != "tensor(float)":
        raise RuntimeError(f"Unexpected model input {input_spec.shape}")
    start = time.perf_counter()
    outputs = runtime.run(None, {input_spec.name: image})
    elapsed = (time.perf_counter() - start) * 1000
    shapes = [list(t.shape) for t in outputs]
    if sorted(shapes) != [[1, 133, 384], [1, 133, 512]] or not all(np.isfinite(t).all() for t in outputs):
        raise RuntimeError(f"Unexpected wholebody SimCC output {shapes}")
    axes = {"x" if t.shape[-1] == 384 else "y": {"name": meta.name, "shape": list(t.shape), "dtype": "float32"}
            for meta, t in zip(runtime.get_outputs(), outputs)}
    maxima = [t.max(axis=-1)[0] for t in outputs]
    scores = np.minimum(*maxima)
    if np.count_nonzero(scores[:17] > .1) < 8:
        raise RuntimeError("Public body fixture has no meaningful model response")
    contract = {
        "input": {"name": input_spec.name, "shape": [1, 3, 256, 192], "layout": "NCHW",
                  "color": "RGB", "dtype": "float32", "mean": [123.675, 116.28, 103.53],
                  "std": [58.395, 57.12, 57.375], "cropPadding": 1.25},
        "outputs": axes, "keypointCount": 133, "keypointOrder": "COCO-WholeBody-133",
        "splitRatio": 2, "dimensions": 2,
        "score": {"formula": "min(max(simcc_x),max(simcc_y))", "calibratedProbability": False},
    }
    return contract, {
        "runtime": "ONNX Runtime CPU", "finite": True, "outputShapes": shapes,
        "bodyPointsAboveSmokeThreshold01": int(np.count_nonzero(scores[:17] > .1)),
        "singleInferenceMs": round(elapsed, 3), "groundTruthAccuracyMeasured": False,
        "browserGpuMeasured": False,
    }


def prepare(output: Path) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    pose_image, rgb, pose_fixture = public_image("pose")
    hand_image, _, hand_fixture = public_image("hands")
    models, verification, crop = {}, {}, None
    for key, source in SOURCES.items():
        data = download(source["url"])
        source_digest = digest(data)
        archive_member = None
        if source.get("archive"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                candidates = [i for i in archive.infolist() if i.filename.rsplit("/", 1)[-1] == "end2end.onnx"]
                if len(candidates) != 1 or candidates[0].file_size > 250_000_000:
                    raise RuntimeError("Official SDK archive has no unique bounded ONNX model")
                archive_member = candidates[0].filename
                data = archive.read(candidates[0])
        expected = source.get("sha256")
        if expected and digest(data) != expected:
            raise RuntimeError(f"Model SHA-256 mismatch: {key}")
        path = output / source["file"]
        path.write_bytes(data)
        record = {"id": key, "file": source["file"], "format": source["format"],
                  "bytes": len(data), "sha256": digest(data), "license": source["license"],
                  "sourceUrl": source["url"], "sourceSha256": source_digest,
                  "trainedInThisProject": False, "privateInputsUsed": False}
        if archive_member:
            record["sourceArchiveMember"] = archive_member
        if source["format"] == "mediapipe-task":
            with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                members = [i for i in bundle.infolist() if i.filename.endswith(".tflite")]
                if len(members) < 2:
                    raise RuntimeError("Task bundle must include detector and landmarker")
                record["members"] = [{"name": i.filename, "bytes": i.file_size} for i in members]
            measured, points = verify_tasks(path, source["kind"], hand_image if source["kind"] == "hands" else pose_image)
            if key == "pose_full":
                crop = wholebody_crop(rgb, points)
            count = 21 if source["kind"] == "hands" else 33
            record.update({"input": {"type": "decoded RGB video frame", "normalization": "MediaPipe Tasks internal"},
                           "outputs": {"normalizedLandmarks": {"points": count, "fields": ["x", "y", "z"]},
                                       "worldLandmarks": {"points": count, "fields": ["x", "y", "z"],
                                                          "units": "estimated metres relative to hand or hips"}},
                           "dimensions": 3, "absoluteDistanceMeasured": False,
                           "api": "@mediapipe/tasks-vision"})
            measured["fixture"] = hand_fixture if source["kind"] == "hands" else pose_fixture
        else:
            if crop is None:
                raise RuntimeError("Pose fixture crop unavailable")
            contract, measured = verify_onnx(path, crop)
            record.update(contract)
            measured["fixture"] = pose_fixture
        models[key], verification[key] = record, measured
        print(json.dumps({"prepared": key, "bytes": len(data), "sha256": digest(data)}), flush=True)
    for name, url in LICENSES.items():
        data = download(url, 100_000)
        text = data.decode("utf-8")
        if "Apache License" not in text or "Version 2.0" not in text:
            raise RuntimeError("Unexpected upstream license content")
        (output / name).write_text(text, encoding="utf-8")
    manifest = {"schemaVersion": 1, "repository": REPOSITORY, "sourceCommit": os.environ.get("GITHUB_SHA"),
                "defaultPose": "pose_full", "models": models,
                "accuracyCertified": False, "privateVideoEvaluationIncluded": False,
                "note": "Real model execution and output contracts checked; this is not ground-truth tracking accuracy certification."}
    report = {"schemaVersion": 1, "models": verification, "containsUserMedia": False,
              "groundTruthAccuracyMeasured": False, "gpuUtilizationMeasured": False}
    return manifest, report


def publish(output: Path, manifest: dict, report: dict) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the explicitly requested model publication")
    api = HfApi(token=token)
    api.create_repo(repo_id=REPOSITORY, repo_type="model", private=False, exist_ok=True)
    if api.model_info(REPOSITORY).private:
        raise RuntimeError("Target model repository exists privately; refusing to expose unrelated contents")
    asset_names = [source["file"] for source in SOURCES.values()] + list(LICENSES)
    asset_commit = api.upload_folder(repo_id=REPOSITORY, folder_path=str(output),
                                    allow_patterns=asset_names, delete_patterns=None,
                                    commit_message="Publish upstream wholebody tracking models " + os.environ.get("GITHUB_SHA", ""))
    revision = asset_commit.oid
    manifest["revision"] = revision
    for model in manifest["models"].values():
        model["url"] = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{model['file']}"
    manifest["licenses"] = [{"file": name, "sourceUrl": url,
                             "url": f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{name}"}
                            for name, url in LICENSES.items()]
    write_json(output / "manifest.json", manifest)
    write_json(output / "verification.json", report)
    (output / "README.md").write_text(
        "---\nlanguage: [en, ja]\nlicense: apache-2.0\npipeline_tag: keypoint-detection\n---\n"
        "# Wholebody Video Tracking models\n"
        "Verified mirrors of Google MediaPipe GHUM Full/Lite Pose and Hand task bundles, plus OpenMMLab RTMPose-s/m 133-point ONNX models.\n"
        "All original pretrained weights are retained. This project does not retrain on a small image subset.\n"
        "The default body model is GHUM Full. Lite trades accuracy for compute. RTMPose provides alternate 2D wholebody observations; it does not emit measured depth.\n"
        "`manifest.json` records SHA-256, original source URLs, model contracts, and immutable Hugging Face asset revisions.\n"
        "`verification.json` records actual CPU inference on upstream public fixtures. Passing these checks proves executable model contracts, not accuracy on arbitrary videos or low GPU utilization on a particular device.\n"
        "No user videos, frames, audio, personal landmarks, or credentials are published. Monocular world coordinates and camera-root translation remain estimates.\n"
        "MediaPipe source: https://github.com/google-ai-edge/mediapipe/tree/v0.10.21 (Apache-2.0).\n"
        f"MMPose source: https://github.com/open-mmlab/mmpose/tree/{MM_REVISION} (Apache-2.0).\n",
        encoding="utf-8")
    final_commit = api.upload_folder(repo_id=REPOSITORY, folder_path=str(output),
                                    allow_patterns=["manifest.json", "verification.json", "README.md"],
                                    delete_patterns=None, commit_message="Record pinned model provenance and execution results")
    result = {"repository": REPOSITORY, "assetRevision": revision, "manifestRevision": final_commit.oid,
              "manifestUrl": f"https://huggingface.co/{REPOSITORY}/resolve/{final_commit.oid}/manifest.json",
              "modelCount": len(manifest["models"]), "privateInputsPublished": False}
    write_json(output / "publication.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Model preparation and publication run in GitHub Actions.")
    manifest, report = prepare(args.output)
    write_json(args.output / "verification.json", report)
    if args.prepare_only:
        write_json(args.output / "manifest.unpublished.json", manifest)
        print(json.dumps({"prepared": True, "published": False}))
    else:
        print(json.dumps(publish(args.output, manifest, report)))


if __name__ == "__main__":
    main()
