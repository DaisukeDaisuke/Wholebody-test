#!/usr/bin/env python3
"""Export the pinned FOCUS foot detector in GitHub Actions, without upload secrets.

This script verifies the checkpoint's Git object before opening it with PyTorch.
It exports both available inference heads. Constant-image inference verifies the
tensor contract only; foot-evaluate.py measures accuracy on annotated images.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

from benchmark import fetch, sha256


SOURCE_COMMIT = "ed2ee8e394162b823b575f8d2fd5b7d625dcd74c"
SOURCE_GIT_BLOB = "aaab7daf96fd2253a64268c3f897391156e5864a"
SOURCE_BYTES = 6396455
SOURCE_URL = f"https://raw.githubusercontent.com/Logimade/FOCUS/{SOURCE_COMMIT}/weights/best.pt"
ULTRALYTICS_VERSION = "8.4.198"
HF_REPOSITORY = "RabbitDaisuke/Foot-Refinement-onnx"
KEYPOINT_NAMES = ["left_big_toe", "left_small_toe", "left_heel", "left_ankle",
                  "right_big_toe", "right_small_toe", "right_heel", "right_ankle"]
COCO_INDICES = [17, 18, 19, 15, 20, 21, 22, 16]


def require_actions():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Checkpoint loading, export and inference must run in GitHub Actions.")
    if any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("Do not pass a Hugging Face upload token to the checkpoint export job.")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_checkpoint(path: Path) -> dict:
    content = path.read_bytes()
    git_blob = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
    if len(content) != SOURCE_BYTES or git_blob != SOURCE_GIT_BLOB:
        raise ValueError("The FOCUS checkpoint does not match the reviewed pinned Git object.")
    return {"repository": "Logimade/FOCUS", "commit": SOURCE_COMMIT, "url": SOURCE_URL,
            "gitBlob": git_blob, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def inspect_export(path: Path, expected_head: str, threads: int) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or inputs[0].type != "tensor(float)":
        raise ValueError(f"Unexpected input contract: {[(item.name, item.shape, item.type) for item in inputs]}")
    output = session.run(None, {inputs[0].name: np.full([1, 3, 640, 640], 114 / 255, dtype=np.float32)})
    if len(output) != 1 or output[0].ndim != 3 or not np.isfinite(output[0]).all():
        raise ValueError(f"Unexpected foot output: {[item.shape for item in output]}")
    shape = list(output[0].shape)
    if shape[0] == 1 and shape[1] == 29 and shape[2] > 300:
        head, layout = "one2many", "N,4+1+8*3,anchors"
        box_format = "cxcywh"
    elif shape[0] == 1 and shape[-1] == 30 and shape[1] <= 300:
        head, layout = "one2one", "N,max_det,6+8*3"
        box_format = "xyxy"
    else:
        raise ValueError(f"Unrecognized 8-point/one-class output shape: {shape}")
    if head != expected_head:
        raise ValueError(f"Requested {expected_head}, but exported tensor is {head}: {shape}")
    return {"inputName": inputs[0].name, "inputShape": inputs[0].shape,
            "outputs": [{"name": session.get_outputs()[0].name,
                         "shape": session.get_outputs()[0].shape, "runtimeShape": shape}],
            "head": head, "layout": layout, "boxFormat": box_format,
            "scoreOffset": 4, "keypointOffset": 5 if head == "one2many" else 6,
            "classOffset": None if head == "one2many" else 5,
            "coordinateSpace": "letterboxed input pixels; already decoded by ONNX",
            "nms": "external for multiple detections; top-one selection needs no NMS" if head == "one2many" else "not required",
            "tensorContractChecked": True, "accuracyMeasuredByThisCheck": False,
            "onnxruntime": ort.__version__, "onnx": onnx.__version__}


def main():
    require_actions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("foot-model-package"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody-foot"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    installed = importlib.metadata.version("ultralytics")
    if installed != ULTRALYTICS_VERSION:
        raise ValueError(f"Install ultralytics=={ULTRALYTICS_VERSION}; found {installed}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = fetch(SOURCE_URL, args.cache / f"focus-{SOURCE_COMMIT}.pt", limit=8 * 1024 * 1024)
    source = verify_checkpoint(checkpoint)

    # Loading happens only after exact object verification, in a job without HF_TOKEN.
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    manifest = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
                "repository": HF_REPOSITORY, "license": "agpl-3.0",
                "modelOrigin": "Existing FOCUS YOLO26n-pose checkpoint; exported, not newly trained.",
                "source": source, "keypointNames": KEYPOINT_NAMES, "cocoIndices": COCO_INDICES,
                "individualToeArticulationObserved": False, "models": [], "failedExports": [],
                "environment": {"python": platform.python_version(), "ultralytics": installed,
                                "torch": torch.__version__, "gitCommit": os.environ.get("GITHUB_SHA"),
                                "workflowRunId": os.environ.get("GITHUB_RUN_ID")},
                "sources": [f"https://github.com/Logimade/FOCUS/tree/{SOURCE_COMMIT}",
                            "https://docs.ultralytics.com/models/yolo26/",
                            "https://docs.ultralytics.com/modes/export/", "https://www.ultralytics.com/license"]}
    for head_name, nms in (("one2many", None), ("one2one", False)):
        try:
            model = YOLO(str(checkpoint), task="pose")
            head = model.model.model[-1]
            if tuple(getattr(head, "kpt_shape", ())) != (8, 3) or getattr(head, "nc", None) != 1:
                raise ValueError(f"Expected one class and eight x/y/confidence points; got {type(head).__name__}")
            export_options = {"format": "onnx", "imgsz": 640, "batch": 1,
                              "dynamic": False, "simplify": False, "opset": 17,
                              "device": "cpu", "nms": nms}
            exported = Path(model.export(**export_options))
            filename = f"focus-foot8-{head_name}-640-fp32.onnx"
            destination = args.output / filename
            shutil.copyfile(exported, destination)
            contract = inspect_export(destination, head_name, args.threads)
            manifest["models"].append({"id": f"focus-foot8-{head_name}", "filename": filename,
                                       "license": "agpl-3.0", "sha256": sha256(destination),
                                       "bytes": destination.stat().st_size, "dimensions": 2,
                                       "keypointCount": 8, "inputWH": [640, 640], "dtype": "float32",
                                       "architecture": type(head).__name__, "cocoIndices": COCO_INDICES,
                                       "color": "RGB", "normalization": "divide by 255",
                                       "letterbox": {"fill": 114, "center": True, "auto": False,
                                                     "rounding": "round resize; round(half-pad - 0.1) left/top"},
                                       "exportOptions": export_options, **contract})
            print(json.dumps({"exported": filename, "head": head_name, "bytes": destination.stat().st_size}), flush=True)
        except Exception as error:
            (args.output / f"focus-foot8-{head_name}-640-fp32.onnx").unlink(missing_ok=True)
            manifest["failedExports"].append({"head": head_name, "error": f"{type(error).__name__}: {error}"})
            print(json.dumps({"head": head_name, "exportFailed": str(error)}), flush=True)
    write_json(args.output / "foot-models.json", manifest)
    if not manifest["models"]:
        raise RuntimeError("Neither FOCUS inference head exported successfully; see foot-models.json.")
    fetch(f"https://raw.githubusercontent.com/ultralytics/ultralytics/v{ULTRALYTICS_VERSION}/LICENSE",
          args.output / "LICENSE", limit=128 * 1024)
    card = """---
license: agpl-3.0
library_name: onnxruntime
pipeline_tag: keypoint-detection
tags: [onnx, foot-pose, keypoints, yolo26]
---
# FOCUS Foot Refinement ONNX

Export of Logimade/FOCUS's existing YOLO26n-pose checkpoint. No new training is claimed.
The upstream checkpoint is pinned and its Git object is verified before export.
The model weights retain Ultralytics AGPL-3.0 terms; the FOCUS README's code license
does not replace the pretrained model license. This is separate from the Apache-2.0
Wholebody-Refinement-onnx repository.

Eight points: left big toe, small toe, heel, ankle; right big toe, small toe, heel, ankle.
COCO-WholeBody indices: 17,18,19,15,20,21,22,16. These are 2D estimated landmarks.
Five individual toes, toe articulation, depth and true visibility are not measured.

Input: RGB float32 NCHW 1x3x640x640, letterbox padding 114, divide by 255.
Output coordinates are already decoded in input-image pixels. Undo letterbox and
add any crop origin; do not apply a second YOLO head decoder. See foot-models.json
for actual measured output shapes and export settings. The one-to-many head needs
external NMS when retaining multiple detections. Argmax selects one foot pair.

Source and reproducible Actions evaluation:
- https://github.com/Logimade/FOCUS
- https://github.com/DaisukeDaisuke/Wholebody-test
- https://www.ultralytics.com/license

The small COCO-WholeBody diagnostic evaluation is not an independent generalization
benchmark. Images and annotations are not distributed in this model repository.
"""
    (args.output / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps({"models": len(manifest["models"]), "failedExports": manifest["failedExports"],
                      "output": str(args.output), "uploaded": False}), flush=True)


if __name__ == "__main__":
    main()
