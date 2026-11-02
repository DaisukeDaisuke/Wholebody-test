"""One actual-image RTMDet input-contract check; run in GitHub Actions only.

This verifies the shipped detector path with real predictions, not a benchmark
of detector recall or a browser/GPU performance measurement.
"""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from evaluate import CACHE, contract, download, load_dataset, session


MODEL = {
    "url": "https://huggingface.co/ziq/rtm/resolve/6bdd0110307167f5f003e92e34099a21bd61283c/rtmdet-s.onnx",
    "sha256": "3bb7246a8b5104dd94e9f9bbfcf92d2cd9e55d344a9a5e8b7b84ebcfeedafe14",
    "official_config": "https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "inherited_config": "https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py",
}
MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def infer(detector, image, normalize):
    height, width = image.shape[:2]
    ratio = min(640 / width, 640 / height)
    # Math.round in the browser; positive dimensions only.
    resized_width = int(width * ratio + .5)
    resized_height = int(height * ratio + .5)
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    values = padded.astype(np.float32)
    if normalize:
        values = (values - MEAN_BGR) / STD_BGR
    tensor = np.ascontiguousarray(values.transpose(2, 0, 1)[None])
    began = time.perf_counter()
    output_values = detector.run(None, {detector.get_inputs()[0].name: tensor})
    elapsed = (time.perf_counter() - began) * 1000
    outputs = dict(zip([output.name for output in detector.get_outputs()], output_values))
    boxes_name = next((name for name, value in outputs.items() if value.ndim >= 2 and value.shape[-1] == 5), None)
    if boxes_name is None:
        raise ValueError(f"No exported xyxy+score output: {contract(detector)}")
    rows = outputs[boxes_name].reshape(-1, 5)
    labels_name = next((name for name, value in outputs.items() if name != boxes_name and value.size == len(rows)), None)
    if labels_name is None:
        raise ValueError("Detector must return COCO labels; assuming all objects are people is invalid")
    labels = outputs[labels_name].reshape(-1)
    people = []
    for row, label in zip(rows, labels):
        if not np.isfinite(row).all() or not np.isfinite(label) or label != 0 or row[4] < .35:
            continue
        box = row[:4].astype(float) / ratio
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        people.append({"bbox_xyxy": box.tolist(), "score": float(row[4])})
    people.sort(key=lambda detection: (detection["bbox_xyxy"][2] - detection["bbox_xyxy"][0])
                * (detection["bbox_xyxy"][3] - detection["bbox_xyxy"][1]), reverse=True)
    return {"input_normalized": normalize, "channel_order": "BGR", "input_shape": list(tensor.shape),
            "input_range": [float(tensor.min()), float(tensor.max())],
            "input_channel_means": tensor.mean(axis=(0, 2, 3)).tolist(),
            "resized_size": [resized_width, resized_height], "nominal_scale": ratio,
            "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
            "boxes_output": boxes_name, "labels_output": labels_name, "cpu_inference_ms": elapsed,
            "person_detections": people}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/detector.json"))
    args = parser.parse_args()
    # One fixed official example image; error measurements never select it.
    first, images, source = load_dataset("official-example", 1)
    image_id = first[0]["image_id"]
    image, image_source = images[image_id]
    annotations = json.loads((CACHE / "official-example.json").read_text(encoding="utf-8"))
    gt = []
    for annotation in annotations["annotations"]:
        if annotation["image_id"] != image_id or annotation.get("iscrowd") or annotation.get("category_id", 1) != 1:
            continue
        x, y, width, height = annotation["bbox"]
        if width > 0 and height > 0:
            gt.append([x, y, x + width, y + height])
    if not gt:
        raise ValueError("No person boxes in the selected annotated image")
    detector = session(download(MODEL["url"], CACHE / "rtmdet-s.onnx", MODEL["sha256"]))
    shape = detector.get_inputs()[0].shape
    expected = [1, 3, 640, 640]
    if len(shape) != 4 or any(isinstance(actual, int) and actual != wanted for actual, wanted in zip(shape, expected)):
        raise ValueError(f"Detector input shape disagrees with 640x640 NCHW: {shape}")
    report = {"kind": "one-image-detector-contract", "git_sha": os.environ.get("GITHUB_SHA"),
              "model": {**MODEL, "contract": contract(detector)}, "image": image_source,
              "annotation_source": source["url"], "ground_truth_person_boxes": len(gt),
              "conditions": {},
              "limitations": ["One real image checks a usable inference path, not general detection recall.",
                              "IoU and detections do not inspect graph nodes or prove the graph has no normalization operations.",
                              "Raw-BGR is a diagnostic control; the official normalized-BGR pipeline is the acceptance condition.",
                              "CPU inference does not measure browser WASM latency or camera throughput."]}
    for name, normalize in [("as_shipped_normalized_bgr", True), ("diagnostic_raw_bgr", False)]:
        result = infer(detector, image, normalize)
        for detection in result["person_detections"]:
            detection["best_person_iou"] = max(iou(detection["bbox_xyxy"], box) for box in gt)
        result["selected_largest_person_iou"] = (result["person_detections"][0]["best_person_iou"]
                                                 if result["person_detections"] else 0)
        result["best_person_iou"] = max((detection["best_person_iou"] for detection in result["person_detections"]), default=0)
        report["conditions"][name] = result
    report["ok"] = report["conditions"]["as_shipped_normalized_bgr"]["selected_largest_person_iou"] >= .5
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not report["ok"]:
        raise RuntimeError("The shipped detector did not localize a real GT person at IoU >= 0.5")


if __name__ == "__main__":
    main()
