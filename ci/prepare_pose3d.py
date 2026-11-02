"""Fetch a pinned Apache-2.0 learned 3D body model for Actions publication.

Only public model/licence bytes are read. This script does not train, run
inference, read private media, or publish anything by itself.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODEL_NAME = "pose_estimation_mediapipe_2023mar.onnx"
SHA256 = "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
SIZE = 5557238
BASE = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/pose_estimation_mediapipe/"
LICENSE_URL = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{REVISION}/models/pose_estimation_mediapipe/LICENSE"


def fetch(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "vrc-public-pose3d-model/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError("pose3d_public_download_failed")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("pose3d_public_download_oversized")
    return data


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Run model preparation in GitHub Actions.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/pose3d"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = fetch(BASE + MODEL_NAME, SIZE)
    if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("pose3d_model_integrity_failed")
    license_text = fetch(LICENSE_URL, 20000)
    if b"Apache License" not in license_text or b"Version 2.0" not in license_text:
        raise RuntimeError("pose3d_license_integrity_failed")
    (args.output / "pose3d.onnx").write_bytes(data)
    (args.output / "LICENSE-POSE3D.txt").write_bytes(license_text)
    manifest = {
        "id": "mediapipe-pose3d-lite-opencv-2023mar",
        "file": "pose3d.onnx", "filename": "pose3d.onnx",
        "sha256": SHA256, "bytes": SIZE,
        "inputShape": [1, 256, 256, 3], "inputWH": [256, 256],
        "inputNamePolicy": "session.inputNames[0]",
        "color": "RGB", "layout": "NHWC", "scale": 1 / 255,
        "input": {"shape": [1, 256, 256, 3], "layout": "NHWC", "color": "RGB", "scale": 1 / 255, "crop": {"padding": 1.25, "rotation": "hip-to-shoulder axis when observed"}},
        "outputContract": {"screenLandmarks": [1, 195], "poseConfidence": [1, 1], "worldLandmarks": [1, 117], "segmentation": [1, 256, 256, 1], "heatmap": [1, 64, 64, 39]},
        "worldUnits": "estimated-metres-relative-to-hips",
        "worldAxes": "x right, y down, z away from camera",
        "landmarks": 33, "learnedDepth": True,
        "absoluteDistanceMeasured": False,
        "upstreamUrl": BASE + MODEL_NAME,
        "upstreamRevision": REVISION,
        "source": f"https://github.com/opencv/opencv_zoo/tree/{REVISION}/models/pose_estimation_mediapipe",
        "license": "Apache-2.0", "licenseFile": "LICENSE-POSE3D.txt",
        "provenance": {"weightOrigin": "Google MediaPipe Pose, ONNX conversion published by OpenCV Zoo", "customTrainingPerformed": False, "privateInputsUsed": False},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "sha256": SHA256, "bytes": SIZE, "privateInputsUsed": False}))


if __name__ == "__main__":
    main()
