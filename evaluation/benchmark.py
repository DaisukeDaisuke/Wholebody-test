#!/usr/bin/env python3
"""Small, real-image ONNX accuracy experiment. Run this in GitHub Actions.

This is a diagnostic sample, not a COCO AP evaluation or a realtime/GPU claim.
The body crop is a ground-truth person box. Hand refinement receives predictions
only. Ground truth is used for scoring and for controlled occluder placement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SOURCE_REVISION = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
DATA_ROOT = f"https://raw.githubusercontent.com/open-mmlab/mmpose/{SOURCE_REVISION}/tests/data/coco"
BODY_MODEL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
HAND_MODEL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
              (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
              (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13),
              (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19),
              (19, 20)]
SELECTIVE_HAND_POLICY = {
    "rawWristScoreMinimum": 0.2,
    "maximumWristShiftFraction": 0.35,
    "rawJointScoreMinimum": 0.2,
    "minimumConfidentJoints": 11,
    "cropBoundaryMarginPixels": 1.0,
    "maximumBoundaryJoints": 10,
    "cropPadding": 1.25,
    "designEvidenceWorkflowRunId": "41375521773",
    "evaluationStatus": "Exploratory post-hoc design on this diagnostic set; not independent validation.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, limit: int = 512 * 1024 * 1024) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Wholebody-test-accuracy/1.0"})
    for attempt in range(3):
        try:
            size = 0
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(f"Download exceeds {limit} bytes: {url}")
                    target.write(chunk)
            if size == 0:
                raise ValueError(f"Empty download: {url}")
            temporary.replace(destination)
            return destination
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable download state")


def resolve_model(source: str, cache: Path, label: str) -> tuple[Path, dict]:
    if source.startswith(("https://", "http://")):
        suffix = ".zip" if source.split("?", 1)[0].endswith(".zip") else ".onnx"
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        path = fetch(source, cache / f"{label}-{key}{suffix}")
    else:
        path = Path(source).resolve(strict=True)
    archive_hash = sha256(path)
    if zipfile.is_zipfile(path):
        folder = cache / f"{label}-{archive_hash[:16]}"
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            members = [member for member in archive.infolist() if member.filename.endswith(".onnx")]
            preferred = [member for member in members if Path(member.filename).name == "end2end.onnx"]
            if len(preferred) == 1:
                member = preferred[0]
            elif len(members) == 1:
                member = members[0]
            else:
                raise ValueError(f"Expected one unambiguous ONNX in {path}, found {len(members)}")
            model_path = folder / "model.onnx"
            if not model_path.is_file():
                with archive.open(member) as src, model_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            sidecars = [item for item in archive.infolist() if item.filename.endswith(".onnx.data")]
            for item in sidecars:
                with archive.open(item) as src, (folder / Path(item.filename).name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    else:
        model_path = path
    return model_path, {"source": source, "archiveSha256": archive_hash,
                        "onnxSha256": sha256(model_path), "bytes": model_path.stat().st_size}


class PoseModel:
    def __init__(self, source: str, cache: Path, label: str, threads: int):
        path, self.metadata = resolve_model(source, cache, label)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f"Expected fixed NCHW image dimensions, received {shape}")
        self.height, self.width = shape[2:]
        self.name = model_input.name
        self.output_names = [value.name for value in self.session.get_outputs()]
        self.dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32
        self.times: list[float] = []
        self.capture_next = False
        self.codec_fixture = None
        self.metadata.update({"input": {"name": self.name, "shape": shape, "type": model_input.type},
                              "outputs": [{"name": value.name, "shape": value.shape, "type": value.type}
                                          for value in self.session.get_outputs()],
                              "provider": "CPUExecutionProvider", "color": "RGB",
                              "normalizationMean": MEAN.tolist(), "normalizationStd": STD.tolist(),
                              "simccSplitRatio": 2, "cropPadding": 1.25})

    def predict(self, image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, width, height = [float(value) for value in box]
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive crop {box}")
        center = np.array([x + width / 2, y + height / 2], dtype=np.float32)
        scale = np.array([width, height], dtype=np.float32) * 1.25
        aspect = self.width / self.height
        if scale[0] > scale[1] * aspect:
            scale[1] = scale[0] / aspect
        else:
            scale[0] = scale[1] * aspect
        matrix = np.array([[self.width / scale[0], 0, self.width * (0.5 - center[0] / scale[0])],
                           [0, self.height / scale[1], self.height * (0.5 - center[1] / scale[1])]],
                          dtype=np.float32)
        crop = cv2.warpAffine(image, matrix, (self.width, self.height),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None], dtype=self.dtype)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.name: tensor})
        self.times.append((time.perf_counter() - started) * 1000)
        if len(outputs) != 2 or any(value.ndim != 3 for value in outputs):
            raise ValueError(f"Expected two SimCC outputs, got {[value.shape for value in outputs]}")
        named = dict(zip(self.output_names, outputs))
        x_axis = named.get("simcc_x", outputs[0])
        y_axis = named.get("simcc_y", outputs[1])
        if self.width != self.height and x_axis.shape[-1] != self.width * 2:
            x_axis, y_axis = y_axis, x_axis
        if x_axis.shape[-1] != self.width * 2 or y_axis.shape[-1] != self.height * 2:
            raise ValueError(f"Unexpected SimCC dimensions: {x_axis.shape}, {y_axis.shape}")
        if not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all():
            raise ValueError("SimCC model produced non-finite values")
        xy = np.stack([x_axis[0].argmax(axis=-1), y_axis[0].argmax(axis=-1)], axis=-1).astype(np.float32) / 2
        score = np.minimum(x_axis[0].max(axis=-1), y_axis[0].max(axis=-1)).astype(np.float32)
        # Official get_simcc_maximum sets both indices to -1 for a
        # non-positive peak, before dividing by simcc_split_ratio=2.
        xy[score <= 0] = -0.5
        xy = xy / np.array([self.width, self.height]) * scale + center - scale / 2
        if self.capture_next:
            self.codec_fixture = {"rgba": cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
                                  "tensor": tensor.astype(np.float32),
                                  "outputs": list(zip(self.output_names, outputs)),
                                  "box": [float(value) for value in box],
                                  "center": center.tolist(), "scale": scale.tolist(),
                                  "inputWH": [self.width, self.height],
                                  "xy": xy, "score": score}
            self.capture_next = False
        return xy, score

    def timing(self) -> dict:
        samples = self.times[1:] if len(self.times) > 1 else self.times
        return {"calls": len(self.times), "warmupExcluded": len(self.times) > 1,
                "meanMs": float(np.mean(samples)) if samples else None,
                "p95Ms": float(np.percentile(samples, 95)) if samples else None,
                "scope": "CPU session.run only; not browser/GPU or end-to-end realtime"}


def predicted_hand_box(xy: np.ndarray, scores: np.ndarray, side: str,
                       person_box: np.ndarray) -> np.ndarray | None:
    offset, wrist, elbow = (91, 9, 7) if side == "left" else (112, 10, 8)
    hand = xy[offset:offset + 21]
    confidence = scores[offset:offset + 21]
    valid = (confidence >= 0.2) & np.isfinite(hand).all(axis=1)
    arm_valid = scores[wrist] >= 0.2 and scores[elbow] >= 0.2
    forearm = float(np.linalg.norm(xy[wrist] - xy[elbow])) if arm_valid else 0.0
    if np.count_nonzero(valid) >= 5:
        points = hand[valid]
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        size = max(float(np.max(high - low)), forearm * 0.5, person_box[3] * 0.045, 10.0)
    elif arm_valid:
        center = xy[wrist] + (xy[wrist] - xy[elbow]) * 0.25
        size = max(forearm * 0.8, person_box[3] * 0.08, 12.0)
    else:
        return None
    size = min(size, max(person_box[2], person_box[3]) * 0.55)
    return np.array([center[0] - size / 2, center[1] - size / 2, size, size], dtype=np.float32)


def refine_hands(model: PoseModel, image: np.ndarray, xy: np.ndarray,
                 scores: np.ndarray, person_box: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    refined, refined_scores = xy.copy(), scores.copy()
    records = []
    for side, offset in (("left", 91), ("right", 112)):
        box = predicted_hand_box(xy, scores, side, person_box)
        if box is None:
            records.append({"side": side, "accepted": False, "reason": "no_predicted_roi"})
            continue
        hand, confidence = model.predict(image, box)
        if hand.shape != (21, 2):
            raise ValueError(f"Hand model returned {hand.shape}, expected 21 keypoints")
        reference_wrist = 9 if side == "left" else 10
        wrist_distance = float(np.linalg.norm(hand[0] - xy[reference_wrist]))
        finite = bool(np.isfinite(hand).all() and np.isfinite(confidence).all())
        accepted = finite and float(np.mean(confidence)) >= 0.2
        if scores[reference_wrist] >= 0.3:
            accepted = accepted and wrist_distance <= float(box[2]) * 0.8
        if accepted:
            refined[offset:offset + 21] = hand
            refined_scores[offset:offset + 21] = confidence
        records.append({"side": side, "roi": box.tolist(), "accepted": bool(accepted),
                        "wristDistancePx": wrist_distance, "meanScore": float(np.mean(confidence))})
    return refined, refined_scores, records


def selective_refinement(xy: np.ndarray, scores: np.ndarray, refined: np.ndarray,
                         refined_scores: np.ndarray, records: list[dict],
                         hand_input_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Conservative acceptance of the same Hand5 candidates, with no extra run.

    This policy uses predicted geometry and Hand5's own score scale only. The
    scoring annotations and image/annotation IDs are deliberately not inputs.
    """
    selected, selected_scores, decisions = xy.copy(), scores.copy(), []
    input_wh = np.asarray(hand_input_wh, dtype=np.float64)
    for record in records:
        decision = {**record, "policy": "hand_selective"}
        if not record["accepted"]:
            decision.update({"accepted": False, "rejectedBy": ["initial_candidate_gate"]})
            decisions.append(decision)
            continue
        offset, wrist = (91, 9) if record["side"] == "left" else (112, 10)
        hand = refined[offset:offset + 21]
        confidence = refined_scores[offset:offset + 21]
        box = np.asarray(record["roi"], dtype=np.float64)
        center = box[:2] + box[2:] / 2
        crop_scale = box[2:] * SELECTIVE_HAND_POLICY["cropPadding"]
        aspect = input_wh[0] / input_wh[1]
        if crop_scale[0] > crop_scale[1] * aspect:
            crop_scale[1] = crop_scale[0] / aspect
        else:
            crop_scale[0] = crop_scale[1] * aspect
        input_xy = (hand - center + crop_scale / 2) / crop_scale * input_wh
        margin = SELECTIVE_HAND_POLICY["cropBoundaryMarginPixels"]
        boundary = ((input_xy <= margin) | (input_xy >= input_wh - margin)).any(axis=1)
        boundary_count = int(np.count_nonzero(boundary))
        confident_count = int(np.count_nonzero(confidence >= SELECTIVE_HAND_POLICY["rawJointScoreMinimum"]))
        wrist_shift = float(np.linalg.norm(hand[0] - xy[wrist]) / box[2])
        checks = {
            "wrist_confidence": float(confidence[0]) >= SELECTIVE_HAND_POLICY["rawWristScoreMinimum"],
            "wrist_agreement": wrist_shift <= SELECTIVE_HAND_POLICY["maximumWristShiftFraction"],
            "majority_confident": confident_count >= SELECTIVE_HAND_POLICY["minimumConfidentJoints"],
            "not_boundary_collapsed": boundary_count <= SELECTIVE_HAND_POLICY["maximumBoundaryJoints"],
        }
        accepted = all(checks.values())
        decision.update({"accepted": accepted, "rejectedBy": [name for name, passed in checks.items() if not passed],
                         "rawWristScore": float(confidence[0]), "wristShiftFraction": wrist_shift,
                         "confidentJointCount": confident_count, "cropBoundaryJointCount": boundary_count})
        if accepted:
            selected[offset:offset + 21] = hand
            selected_scores[offset:offset + 21] = confidence
        decisions.append(decision)
    return selected, selected_scores, decisions


def target_points(annotation: dict) -> np.ndarray:
    fields = ("keypoints", "foot_kpts", "face_kpts", "lefthand_kpts", "righthand_kpts")
    values = np.concatenate([np.asarray(annotation[field]).reshape(-1, 3) for field in fields]).astype(np.float32)
    for flag, start, end in (("foot_valid", 17, 23), ("face_valid", 23, 91),
                              ("lefthand_valid", 91, 112), ("righthand_valid", 112, 133)):
        if not annotation.get(flag, False):
            values[start:end, 2] = 0
    return values


def part_groups(annotation: dict, target: np.ndarray):
    yield "body", np.arange(17), max(annotation["bbox"][2:])
    for label, start, end in (("left_foot", 17, 20), ("right_foot", 20, 23)):
        points = target[start:end]
        valid = points[points[:, 2] > 0, :2]
        scale = float(np.max(np.ptp(valid, axis=0))) if len(valid) >= 2 else 0.0
        yield label, np.arange(start, end), scale
    for side, offset in (("left", 91), ("right", 112)):
        box = annotation[f"{side}hand_box"]
        yield f"{side}_hand", np.arange(offset, offset + 21), max(box[2:])


def add_metrics(bucket: dict, annotation: dict, target: np.ndarray, prediction: np.ndarray,
                confidence: np.ndarray, changed: np.ndarray | None = None):
    for name, indices, scale in part_groups(annotation, target):
        valid = target[indices, 2] > 0
        if changed is not None:
            valid &= changed[indices]
        selected = indices[valid]
        if not len(selected) or scale <= 0:
            continue
        error = np.linalg.norm(prediction[selected] - target[selected, :2], axis=1)
        error[~np.isfinite(error)] = float("inf")
        combined = "hand" if "hand" in name else "foot" if "foot" in name else "body"
        for group in dict.fromkeys((name, combined)):
            value = bucket.setdefault(group, {"errors": [], "normalized": [], "confidences": [], "people": set()})
            value["errors"].extend(error.tolist())
            value["normalized"].extend((error / scale).tolist())
            value["confidences"].extend(confidence[selected].tolist())
            value["people"].add(annotation["id"])


def summarize_metrics(bucket: dict) -> dict:
    summary = {}
    for group, measurements in bucket.items():
        error = np.array(measurements["errors"])
        normalized = np.array(measurements["normalized"])
        confidence = np.array(measurements["confidences"])
        finite = np.isfinite(error)
        summary[group] = {"points": len(error), "people": len(measurements["people"]),
                          "meanErrorPx": float(error[finite].mean()) if finite.any() else None,
                          "p95ErrorPx": float(np.percentile(error[finite], 95)) if finite.any() else None,
                          "normalizedMeanError": float(normalized[finite].mean()) if finite.any() else None,
                          "invalidPredictionCount": int((~finite).sum()),
                          "pck005": float(np.mean(normalized <= 0.05)),
                          "pck010": float(np.mean(normalized <= 0.10)),
                          "pck020": float(np.mean(normalized <= 0.20)),
                          "coverageAtScore020": float(np.mean(confidence >= 0.20)),
                          "coverageAtScore030": float(np.mean(confidence >= 0.30)),
                          "rawScoreP05": float(np.percentile(confidence, 5)),
                          "rawScoreMedian": float(np.median(confidence)),
                          "rawScoreP95": float(np.percentile(confidence, 95)),
                          "denominator": "all annotated points, including low-confidence predictions"}
    return summary


def occlude(image: np.ndarray, annotation: dict, target: np.ndarray,
            kind: str) -> tuple[np.ndarray, np.ndarray]:
    masked = image.copy()
    changed = np.zeros(133, dtype=bool)
    rectangles = []
    if kind == "hand":
        for side in ("left", "right"):
            if annotation.get(f"{side}hand_valid", False):
                x, y, width, height = annotation[f"{side}hand_box"]
                rectangles.append((x, y, x + width * 0.6, y + height))
    elif kind == "foot":
        for start in (17, 20):
            points = target[start:start + 3]
            valid = points[points[:, 2] > 0, :2]
            if len(valid) >= 2:
                low, high = valid.min(axis=0) - 4, valid.max(axis=0) + 4
                rectangles.append((low[0], low[1], (low[0] + high[0]) / 2, high[1]))
    elif kind == "torso":
        x, y, width, height = annotation["bbox"]
        rectangles.append((x + width * 0.2, y + height * 0.30, x + width * 0.8, y + height * 0.60))
    for rectangle in rectangles:
        x1, y1, x2, y2 = [int(round(value)) for value in rectangle]
        x1, x2 = max(x1, 0), min(x2, image.shape[1])
        y1, y2 = max(y1, 0), min(y2, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        masked[y1:y2, x1:x2] = (114, 114, 114)
        changed |= ((target[:, 0] >= x1) & (target[:, 0] < x2) &
                    (target[:, 1] >= y1) & (target[:, 1] < y2) & (target[:, 2] > 0))
    return masked, changed


def overlay(image: np.ndarray, prediction: np.ndarray, score: np.ndarray,
            target: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    edges = BODY_EDGES + [(offset + a, offset + b) for offset in (91, 112) for a, b in HAND_EDGES]
    for a, b in edges:
        if min(score[a], score[b]) >= 0.2 and np.isfinite(prediction[[a, b]]).all():
            cv2.line(out, tuple(np.rint(prediction[a]).astype(int)),
                     tuple(np.rint(prediction[b]).astype(int)), (255, 230, 40), 1, cv2.LINE_AA)
    for index in list(range(23)) + list(range(91, 133)):
        if target[index, 2] > 0:
            cv2.circle(out, tuple(np.rint(target[index, :2]).astype(int)), 2, (30, 230, 60), 1, cv2.LINE_AA)
        if score[index] >= 0.2 and np.isfinite(prediction[index]).all():
            cv2.circle(out, tuple(np.rint(prediction[index]).astype(int)), 1, (255, 230, 40), -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def json_write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def emit_codec_fixture(model: PoseModel, folder: Path, image_info: dict, annotation: dict):
    sample = model.codec_fixture
    if sample is None:
        raise ValueError("No real inference captured for the codec comparison")
    folder.mkdir(parents=True, exist_ok=True)
    sample["rgba"].tofile(folder / "image-rgba.u8")
    np.asarray(sample["tensor"], dtype="<f4").tofile(folder / "expected-input.f32")
    outputs = []
    for index, (name, array) in enumerate(sample["outputs"]):
        filename = f"output-{index}.f32"
        np.asarray(array, dtype="<f4").tofile(folder / filename)
        outputs.append({"name": name, "shape": list(array.shape), "file": filename, "dtype": "float32-le"})
    manifest = {"schemaVersion": 1, "source": "real COCO image, OpenCV INTER_LINEAR crop, real ONNX outputs",
                "imageId": image_info["id"], "annotationId": annotation["id"],
                "image": {"width": sample["rgba"].shape[1], "height": sample["rgba"].shape[0],
                          "file": "image-rgba.u8", "dtype": "uint8", "channels": "RGBA"},
                "boxXYWH": sample["box"], "inputWH": sample["inputWH"],
                "transform": {"center": sample["center"], "scale": sample["scale"]},
                "expectedTensor": {"file": "expected-input.f32", "shape": list(sample["tensor"].shape),
                                   "dtype": "float32-le"},
                "outputs": outputs, "keypointCount": len(sample["xy"]),
                "expectedPoints": [{"x": float(point[0]), "y": float(point[1]),
                                    "rawScore": float(score), "score": float(np.clip(score, 0, 1))}
                                   for point, score in zip(sample["xy"], sample["score"])],
                "modelSha256": model.metadata["onnxSha256"],
                "opencvVersion": cv2.__version__, "normalizationMean": MEAN.tolist(),
                "normalizationStd": STD.tolist(), "padding": 1.25,
                "scope": "One actual frame/crop. Frontend numerical parity, not tracking accuracy or browser execution."}
    json_write(folder / "manifest.json", manifest)
    model.codec_fixture = None


def predicted_person_box(prediction: np.ndarray, scores: np.ndarray,
                         image: np.ndarray) -> tuple[np.ndarray | None, int]:
    """The first updateTracker call in dist/inference.mjs, with no history.

    The 23 body/foot predictions, not any annotation, determine this crop.
    A second pass over the same still image is only an initialization diagnostic.
    """
    points = prediction[:23]
    valid = (scores[:23] > 0.4) & np.isfinite(points).all(axis=1)
    count = int(np.count_nonzero(valid))
    if count < 8:
        return None, count
    low, high = points[valid].min(axis=0), points[valid].max(axis=0)
    width = max(float(high[0] - low[0]), image.shape[1] * 0.08)
    height = max(float(high[1] - low[1]), image.shape[0] * 0.18)
    return np.array([low[0] - width * 0.25, low[1] - height * 0.15,
                     width * 1.5, height * 1.3], dtype=np.float64), count


def nearest_annotated_body(prediction: np.ndarray, annotations: list[dict],
                           target_id: int) -> dict:
    """Interpret identity errors after inference; never use this to set the ROI.

    This diagnostic does not change the preselected scoring target or turn a
    nearest annotation into a verified identity match.
    """
    candidates = []
    for annotation in annotations:
        body = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        valid = body[:, 2] > 0
        if not valid.any():
            continue
        error = np.linalg.norm(prediction[:17][valid] - body[valid, :2], axis=1)
        candidates.append({"annotationId": annotation["id"],
                           "normalizedMeanBodyError": float(error.mean() / max(annotation["bbox"][2:])),
                           "bodyPoints": int(np.count_nonzero(valid))})
    candidates.sort(key=lambda value: (value["normalizedMeanBodyError"], value["annotationId"]))
    nearest = candidates[0] if candidates else None
    return {"nearestAnnotationId": nearest["annotationId"] if nearest else None,
            "nearestIsScoringTarget": nearest["annotationId"] == target_id if nearest else False,
            "candidates": candidates,
            "scope": "post-inference body-error diagnostic; target and input ROI remain unchanged"}


def initialization_experiment(body_model: PoseModel, hand_model: PoseModel,
                              annotations: list[dict], image_cache: dict,
                              images: dict, output: Path) -> dict:
    """Measure the actual full-frame -> predicted crop -> hand path on images."""
    by_image = defaultdict(list)
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)
    measurements, cases, predictions = {}, [], []
    body_calls_before, hand_calls_before = len(body_model.times), len(hand_model.times)
    method_order = ("full_frame", "full_frame_hand_refined", "full_frame_hand_selective",
                    "predicted_person_crop", "predicted_person_crop_hand_refined",
                    "predicted_person_crop_hand_selective")
    for image_id, candidates in sorted(by_image.items()):
        image = image_cache[image_id]
        # A fixed, reproducible scoring target. Its box is never an input here.
        annotation = max(candidates, key=lambda value: (value["bbox"][2] * value["bbox"][3], -value["id"]))
        target = target_points(annotation)
        full_box = np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float64)
        first_xy, first_scores = body_model.predict(image, full_box)
        next_box, reliable_count = predicted_person_box(first_xy, first_scores, image)
        crop_available = next_box is not None
        if next_box is None:
            next_box = full_box.copy()
        second_xy, second_scores = body_model.predict(image, next_box)
        first_refined, first_refined_scores, first_hands = refine_hands(
            hand_model, image, first_xy, first_scores, full_box)
        second_refined, second_refined_scores, second_hands = refine_hands(
            hand_model, image, second_xy, second_scores, next_box)
        first_selected, first_selected_scores, first_decisions = selective_refinement(
            first_xy, first_scores, first_refined, first_refined_scores, first_hands,
            (hand_model.width, hand_model.height))
        second_selected, second_selected_scores, second_decisions = selective_refinement(
            second_xy, second_scores, second_refined, second_refined_scores, second_hands,
            (hand_model.width, hand_model.height))
        stages = [
            ("full_frame", full_box, first_xy, first_scores, []),
            ("full_frame_hand_refined", full_box, first_refined, first_refined_scores, first_hands),
            ("full_frame_hand_selective", full_box, first_selected, first_selected_scores, first_decisions),
            ("predicted_person_crop", next_box, second_xy, second_scores, []),
            ("predicted_person_crop_hand_refined", next_box, second_refined, second_refined_scores, second_hands),
            ("predicted_person_crop_hand_selective", next_box, second_selected, second_selected_scores, second_decisions),
        ]
        case = {"imageId": image_id, "targetAnnotationId": annotation["id"],
                "targetBBoxXYWHForScoringOnly": annotation["bbox"],
                "annotatedPeople": len(candidates), "initialReliableBodyFootPoints": reliable_count,
                "predictedCropAvailable": crop_available,
                "fallbackReason": None if crop_available else "fewer_than_8_body_foot_points_above_score_0.4",
                "stages": []}
        for method, box, xy, scores, hand_records in stages:
            if xy.shape != (133, 2) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
                raise ValueError(f"Invalid wholebody result for initialization/{method}")
            add_metrics(measurements.setdefault(method, {}), annotation, target, xy, scores)
            per_image = {}
            add_metrics(per_image, annotation, target, xy, scores)
            stage = {"method": method, "inputBBoxXYWH": box.tolist(),
                     "metrics": summarize_metrics(per_image),
                     "identityDiagnostic": nearest_annotated_body(xy, candidates, annotation["id"]),
                     "refinement": hand_records}
            case["stages"].append(stage)
            predictions.append({"imageId": image_id, "targetAnnotationId": annotation["id"],
                                **stage, "prediction": xy.tolist(), "scores": scores.tolist()})
        cases.append(case)
        if images[image_id]["license"] == 4:
            panels = [overlay(image, xy, scores, target, title)
                      for (_, _, xy, scores, _), title in zip(stages, (
                          "full frame; green=fixed target annotation", "full frame + Hand5",
                          "full frame + selective Hand5", "predicted person crop",
                          "predicted person crop + Hand5", "predicted crop + selective Hand5"))]
            grid = np.concatenate([np.concatenate(panels[:3], axis=1),
                                   np.concatenate(panels[3:], axis=1)], axis=0)
            cv2.imwrite(str(output / "overlays" / f"{image_id}-initialization.jpg"),
                        grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"initialization image={image_id} target={annotation['id']} "
              f"predicted_crop={crop_available} reliable_points={reliable_count}", flush=True)
    json_write(output / "initialization-predictions.json", predictions)
    return {"images": len(cases), "methodOrder": list(method_order),
            "targetSelection": "largest annotated person bbox area per selected image; lowest id breaks ties",
            "groundTruthUsedForInference": False,
            "inputPath": "full image -> predicted body/foot extent -> original-image hand crops",
            "roiContract": {"pointIndices": "0..22", "scoreGreaterThan": 0.4, "minimumPoints": 8,
                            "minimumImageWidthFraction": 0.08, "minimumImageHeightFraction": 0.18,
                            "extentExpansionXYWH": [-0.25, -0.15, 1.5, 1.3],
                            "missingCropFallback": "full image", "historyBlending": False,
                            "modelCropPadding": 1.25},
            "bodyCalls": len(body_model.times) - body_calls_before,
            "handCalls": len(hand_model.times) - hand_calls_before,
            "metrics": {method: summarize_metrics(measurements[method]) for method in method_order},
            "cases": cases,
            "limits": ["The identical still image is used twice; this does not measure video tracking.",
                       "Multi-person initialization can select another person; fixed-target errors are retained.",
                       "Identity diagnostics never alter the target or model input.",
                       "No detector or ground-truth person box is supplied to the inference path."]}


def markdown_report(report: dict) -> str:
    rows = ["# 実画像・実ONNX推論の小規模評価", f"実行日時: {report['createdAt']}",
            f"対象: {report['dataset']['images']}画像・{report['dataset']['people']}人物。人物領域は正解BBoxを使用。",
            "この数値は診断用の小規模標本であり、COCO AP、実動画、多視点、利用者のGPU性能の証明ではない。",
            "手の細部再推論は予測された手首・肘・手指からROIを作る。正解手BBoxを入力しない。",
            "PCKは正解に対する誤差を部位サイズで割って採点。低信頼点も分母から除かない。",
            "身体は人物BBoxの最大辺、手は正解手BBoxの最大辺、足は注釈3点の最大幅/高さで正規化。",
            "|条件|処理|部位|点数|平均誤差px|PCK@0.05|PCK@0.10|PCK@0.20|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for condition, methods in report["metrics"].items():
        for method, parts in methods.items():
            for part in ("body", "hand", "foot"):
                if part not in parts:
                    continue
                value = parts[part]
                error = "—" if value["meanErrorPx"] is None else f"{value['meanErrorPx']:.3f}"
                rows.append(f"|{condition}|{method}|{part}|{value['points']}|{error}|{value['pck005']:.3f}|{value['pck010']:.3f}|{value['pck020']:.3f}|")
    rows.extend(["## 手の再推論による差", "改善を合格条件として偽装せず、符号付きの差をそのまま記録する。"])
    for condition, delta in report["handRefinementDifference"].items():
        rows.append(f"{condition}: 平均誤差差 {delta['meanErrorPxRefinedMinusBaseline']:.3f}px、PCK@0.20差 {delta['pck020RefinedMinusBaseline']:+.3f}。")
    rows.extend(["## 手の選択採用による差",
                 "hand_selectiveは同じHand5候補へ手首の信頼度・位置整合性・関節の信頼度・crop境界への集中を判定する。",
                 "このルールはrun 41375521773の失敗例から設計した探索的改善であり、この小標本での再測定を汎化性能の証明には使わない。"])
    for condition, delta in report["handSelectiveDifference"].items():
        rows.append(f"{condition}: 平均誤差差 {delta['meanErrorPxRefinedMinusBaseline']:.3f}px、PCK@0.20差 {delta['pck020RefinedMinusBaseline']:+.3f}。")
    initialization = report["initializationExperiment"]
    rows.extend(["## 実際の入力経路: 全画面から予測ROIへ",
                 f"{initialization['images']}画像で、全画面入力から予測した身体・足の23点で次の人物領域を作り、元画像の手を再推論する。",
                 "推論に正解BBoxは渡さない。採点対象は画像ごとの最大人物に事前固定し、別人を追った場合の誤差も残す。",
                 "同じ静止画を2回使う初期化の診断であり、実動画での追従・フレーム落ち・再捕捉の評価ではない。",
                 "|入力経路|部位|点数|平均誤差px|PCK@0.05|PCK@0.10|PCK@0.20|",
                 "|---|---|---:|---:|---:|---:|---:|"])
    for method in initialization["methodOrder"]:
        for part in ("body", "hand", "foot"):
            value = initialization["metrics"][method].get(part)
            if value is None:
                continue
            error = "—" if value["meanErrorPx"] is None else f"{value['meanErrorPx']:.3f}"
            rows.append(f"|{method}|{part}|{value['points']}|{error}|{value['pck005']:.3f}|{value['pck010']:.3f}|{value['pck020']:.3f}|")
    rows.extend(["画像ごとの採点・予測ROI・人物対応の診断はreport.jsonのinitializationExperiment.casesへ、",
                 "実座標と手ROIはinitialization-predictions.jsonへ保存する。CC BY画像785の比較はoverlays/785-initialization.jpg。"])
    rows.extend(["## 遮蔽の解釈", "hand/foot/torsoは元画像上へ矩形を置く制御実験。遮蔽で失われた点だけの成績はreport.jsonのmaskedPointMetricsへ分離。",
                 "これは時系列補完や別カメラでの回復を実証する評価ではない。全方向から隠れた指を観測したと扱わない。",
                 "## 出典とライセンス", "COCO-WholeBodyの注釈は研究・非商用用途（公式READMEのリンク先CC BY-NC 4.0）。画像は各画像のFlickr由来ライセンス。",
                 "公開オーバーレイはCC BYの画像785だけ。画像別情報・取得URL・ライセンスはlicense-attribution.json。",
                 f"MMPose fixture commit: {SOURCE_REVISION}", "https://github.com/jin-s13/COCO-WholeBody#terms-of-use", "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose"])
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody"))
    parser.add_argument("--body-model", default=BODY_MODEL)
    parser.add_argument("--body-model-id", default="RTMW-l-cocktail14-384x288")
    parser.add_argument("--hand-model", default=HAND_MODEL)
    parser.add_argument("--max-people", type=int, default=14)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--emit-codec-fixture", action="store_true",
                        help="Save one real frame/crop/input/output fixture for the Actions JavaScript comparison")
    args = parser.parse_args()
    if args.max_people < 1 or args.threads < 1:
        parser.error("--max-people and --threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overlays").mkdir(exist_ok=True)
    annotation_path = fetch(DATA_ROOT + "/test_coco_wholebody.json", args.cache / "test_coco_wholebody.json", 2 * 1024 * 1024)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {value["id"]: value for value in data["images"]}
    licenses = {value["id"]: value for value in data["licenses"]}
    eligible = [value for value in sorted(data["annotations"], key=lambda v: (v["image_id"], v["id"]))
                if not value.get("iscrowd", 0) and value.get("num_keypoints", 0) > 0
                and min(value["bbox"][2:]) >= 20]
    selected = eligible[:args.max_people]
    if not selected:
        raise ValueError("No annotated people were selected")
    body_model = PoseModel(args.body_model, args.cache, "wholebody", args.threads)
    body_model.capture_next = args.emit_codec_fixture
    hand_model = PoseModel(args.hand_model, args.cache, "hand5", args.threads)
    measurements, masked_measurements = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
    all_predictions, image_attributions, image_cache = [], [], {}
    for annotation in selected:
        image_id = annotation["image_id"]
        image_info = images[image_id]
        if image_id not in image_cache:
            image_path = fetch(DATA_ROOT + "/" + image_info["file_name"], args.cache / image_info["file_name"], 5 * 1024 * 1024)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Image decode failed: {image_path}")
            image_cache[image_id] = image
            image_attributions.append({**image_info, "licenseDetail": licenses[image_info["license"]],
                                       "retrievedFrom": DATA_ROOT + "/" + image_info["file_name"],
                                       "sha256": sha256(image_path), "publishedOverlay": image_info["license"] == 4})
        image = image_cache[image_id]
        target = target_points(annotation)
        box = np.array(annotation["bbox"], dtype=np.float32)
        for condition in ("clean", "hand", "foot", "torso"):
            frame, changed = occlude(image, annotation, target, condition)
            prediction, scores = body_model.predict(frame, box)
            if prediction.shape != (133, 2):
                raise ValueError(f"Wholebody model returned {prediction.shape}, expected 133 keypoints")
            if not np.isfinite(prediction).all() or not np.isfinite(scores).all():
                raise ValueError("Wholebody inference produced non-finite values")
            if body_model.codec_fixture is not None:
                emit_codec_fixture(body_model, args.output / "codec-fixture", image_info, annotation)
            refined, refined_scores, hand_details = refine_hands(hand_model, frame, prediction, scores, box)
            selective, selective_scores, selective_details = selective_refinement(
                prediction, scores, refined, refined_scores, hand_details, (hand_model.width, hand_model.height))
            for method, points, confidence in (("wholebody", prediction, scores),
                                               ("hand_refined", refined, refined_scores),
                                               ("hand_selective", selective, selective_scores)):
                add_metrics(measurements[condition][method], annotation, target, points, confidence)
                if changed.any():
                    add_metrics(masked_measurements[condition][method], annotation, target, points, confidence, changed)
            all_predictions.append({"imageId": image_id, "annotationId": annotation["id"], "condition": condition,
                                    "bboxXYWH": box.tolist(), "prediction": prediction.tolist(), "scores": scores.tolist(),
                                    "refined": refined.tolist(), "refinedScores": refined_scores.tolist(),
                                    "refinement": hand_details, "selective": selective.tolist(),
                                    "selectiveScores": selective_scores.tolist(), "selectiveRefinement": selective_details,
                                    "occludedPointIndices": np.flatnonzero(changed).tolist()})
            if image_info["license"] == 4:
                comparison = np.concatenate([overlay(frame, prediction, scores, target, f"{condition}: wholebody; green=annotation"),
                                             overlay(frame, refined, refined_scores, target, "predicted hand ROI + hand5"),
                                             overlay(frame, selective, selective_scores, target, "selective Hand5 (same candidates)")], axis=1)
                filename = args.output / "overlays" / f"{image_id}-{annotation['id']}-{condition}.jpg"
                cv2.imwrite(str(filename), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])
            print(f"inferred image={image_id} person={annotation['id']} condition={condition}", flush=True)
    initialization = initialization_experiment(
        body_model, hand_model, [value for value in eligible if value["image_id"] in image_cache],
        image_cache, images, args.output)
    metrics = {condition: {method: summarize_metrics(parts) for method, parts in methods.items()}
               for condition, methods in measurements.items()}
    masked_metrics = {condition: {method: summarize_metrics(parts) for method, parts in methods.items()}
                      for condition, methods in masked_measurements.items()}
    differences, selective_differences = {}, {}
    for condition, methods in metrics.items():
        if "hand" in methods["wholebody"] and "hand" in methods["hand_refined"]:
            before, after = methods["wholebody"]["hand"], methods["hand_refined"]["hand"]
            differences[condition] = {"meanErrorPxRefinedMinusBaseline": after["meanErrorPx"] - before["meanErrorPx"],
                                      "pck020RefinedMinusBaseline": after["pck020"] - before["pck020"]}
            selected_hand = methods["hand_selective"]["hand"]
            selective_differences[condition] = {
                "meanErrorPxRefinedMinusBaseline": selected_hand["meanErrorPx"] - before["meanErrorPx"],
                "pck020RefinedMinusBaseline": selected_hand["pck020"] - before["pck020"],
            }
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
              "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
              "dataset": {"name": "MMPose COCO-WholeBody diagnostic subset", "sourceRevision": SOURCE_REVISION,
                          "annotationSha256": sha256(annotation_path), "images": len(image_cache), "people": len(selected),
                          "personSelection": "sorted image_id/id; noncrowd; >0 body labels; bbox min side >=20px",
                          "bodyRoi": "ground-truth person bounding box; detector is not evaluated",
                          "handRoi": "predicted points only; ground-truth hand boxes never given to model"},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "onnxruntime": ort.__version__, "opencv": cv2.__version__, "threads": args.threads},
              "models": {"wholebody": {"id": args.body_model_id, **body_model.metadata, "timing": body_model.timing()},
                         "hand": {"id": "RTMPose-m-hand5-256x256", **hand_model.metadata, "timing": hand_model.timing()}},
              "metrics": metrics, "maskedPointMetrics": masked_metrics, "handRefinementDifference": differences,
              "handSelectiveDifference": selective_differences, "handSelectivePolicy": SELECTIVE_HAND_POLICY,
              "initializationExperiment": initialization,
              "limits": ["Small diagnostic subset, not representative COCO AP.",
                         "CPU runtime timings do not measure browser WebGPU or the user's GPU.",
                         "Still-image masks do not validate temporal tracking or true multiview reconstruction.",
                         "COCO feet contain big-toe tip, small-toe tip and heel per foot, not five articulated toes.",
                         "Body ROI comes from annotation; no claim of end-to-end detector accuracy."],
              "sources": ["https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
                          "https://github.com/jin-s13/COCO-WholeBody#terms-of-use"]}
    json_write(args.output / "report.json", report)
    json_write(args.output / "predictions.json", all_predictions)
    json_write(args.output / "license-attribution.json", {"dataset": "COCO-WholeBody research/non-commercial annotations",
                                                        "annotationLicense": "https://creativecommons.org/licenses/by-nc/4.0/legalcode",
                                                        "images": image_attributions})
    (args.output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"images": len(image_cache), "people": len(selected), "handRefinementDifference": differences,
                      "handSelectiveDifference": selective_differences,
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
