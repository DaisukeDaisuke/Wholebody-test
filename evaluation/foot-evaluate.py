#!/usr/bin/env python3
"""Compare FOCUS foot refinement with RTMW on the existing annotated image subset.

Inference runs only in GitHub Actions. This measures 2D landmarks on still images,
including controlled masks. It does not measure 3D, real temporal occlusion, or
five independently articulated toes. No ground-truth foot crop is supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from benchmark import (BODY_MODEL, DATA_ROOT, SOURCE_REVISION, PoseModel, add_metrics,
                       fetch, json_write, occlude, overlay, sha256, summarize_metrics,
                       target_points)


COCO_INDICES = np.array([17, 18, 19, 15, 20, 21, 22, 16], dtype=int)
FOOT_GROUPS = ("foot", "left_foot", "right_foot", "ankle")
DETECTION_MIN = 0.05
REFINE_DETECTION_MIN = 0.25
REFINE_KEYPOINT_MIN = 0.35


def require_actions():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Model inference and accuracy evaluation must run in GitHub Actions.")


def safe_box(box: np.ndarray, image: np.ndarray) -> np.ndarray:
    x, y, width, height = [float(value) for value in box]
    if not all(math.isfinite(value) for value in (x, y, width, height)) or min(width, height) <= 0:
        raise ValueError(f"Invalid crop: {box}")
    x1 = max(0, min(image.shape[1] - 1, math.floor(x)))
    y1 = max(0, min(image.shape[0] - 1, math.floor(y)))
    x2 = max(x1 + 1, min(image.shape[1], math.ceil(x + width)))
    y2 = max(y1 + 1, min(image.shape[0], math.ceil(y + height)))
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=int)


def context_box(box: np.ndarray) -> np.ndarray:
    x, y, width, height = box.astype(float)
    return np.array([x - width * 0.125, y - height * 0.125, width * 1.25, height * 1.25])


def predicted_foot_box(prediction: np.ndarray, scores: np.ndarray,
                       person_box: np.ndarray) -> tuple[np.ndarray, str]:
    """Use eight body-model predictions or a documented lower-person fallback."""
    valid = (scores[COCO_INDICES] >= 0.30) & np.isfinite(prediction[COCO_INDICES]).all(axis=1)
    selected = prediction[COCO_INDICES[valid]]
    x, y, width, height = person_box.astype(float)
    if len(selected) < 3:
        return np.array([x - width * 0.125, y + height * 0.50,
                         width * 1.25, height * 0.65]), "lower_person_box_fallback"
    low, high = selected.min(axis=0), selected.max(axis=0)
    center = (low + high) / 2
    crop_width = max(float(high[0] - low[0]) * 1.6, width * 0.30, 32.0)
    crop_height = max(float(high[1] - low[1]) * 1.8, height * 0.18, 32.0)
    return np.array([center[0] - crop_width / 2, center[1] - crop_height / 2,
                     crop_width, crop_height]), "predicted_ankle_and_foot_extent"


def letterbox(image: np.ndarray, requested_box: np.ndarray, size: int = 640):
    box = safe_box(requested_box, image)
    x, y, width, height = box.tolist()
    crop = image[y:y + height, x:x + width]
    ratio = min(size / width, size / height)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = round((size - resized_width) / 2 - 0.1)
    top = round((size - resized_height) / 2 - 0.1)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[top:top + resized_height, left:left + resized_width] = resized
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    transform = {"cropXYWH": box.tolist(), "ratio": ratio, "left": left, "top": top,
                 "resizedWH": [resized_width, resized_height], "inputWH": [size, size]}
    return tensor, transform


def decode_top_one(raw: np.ndarray, model: dict, transform: dict) -> dict:
    """The maximum score survives ordinary NMS, so top-one requires no NMS pass."""
    if raw.ndim != 3 or raw.shape[0] != 1 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid foot output: {raw.shape}")
    if model["head"] == "one2many":
        if raw.shape[1] != 29:
            raise ValueError(f"Expected 29 raw channels, got {raw.shape}")
        candidates = raw[0].T
        candidate_index = int(np.argmax(candidates[:, 4]))
        row = candidates[candidate_index]
        cx, cy, width, height = row[:4]
        box = np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
        point_offset = 5
    elif model["head"] == "one2one":
        if raw.shape[2] != 30:
            raise ValueError(f"Expected 30 decoded columns, got {raw.shape}")
        candidates = raw[0]
        candidate_index = int(np.argmax(candidates[:, 4]))
        row = candidates[candidate_index]
        if int(round(float(row[5]))) != 0:
            raise ValueError("FOCUS is a one-class model, but output class is not zero")
        box = row[:4].copy()
        point_offset = 6
    else:
        raise ValueError(f"Unknown output head: {model['head']}")
    score = float(row[4])
    points = row[point_offset:].reshape(8, 3).copy()
    if np.any(points[:, 2] < -0.001) or np.any(points[:, 2] > 1.001) or not -0.001 <= score <= 1.001:
        raise ValueError("Foot confidence is outside the declared probability-head range")
    origin = np.array(transform["cropXYWH"][:2], dtype=float)
    padding = np.array([transform["left"], transform["top"]], dtype=float)
    points[:, :2] = (points[:, :2] - padding) / transform["ratio"] + origin
    box = ((box.reshape(2, 2) - padding) / transform["ratio"] + origin).reshape(-1)
    return {"found": score >= DETECTION_MIN, "detectionScore": score, "points": points[:, :2],
            "keypointScores": np.clip(points[:, 2], 0, 1), "bboxXYXY": box,
            "candidateIndex": candidate_index, "candidateCount": len(candidates)}


class FootModel:
    def __init__(self, directory: Path, metadata: dict, threads: int):
        self.metadata = metadata
        path = directory / metadata["filename"]
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"ONNX checksum mismatch: {metadata['filename']}")
        if metadata["cocoIndices"] != COCO_INDICES.tolist():
            raise ValueError("Unexpected FOCUS keypoint ordering")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        if self.input.shape != [1, 3, 640, 640]:
            raise ValueError(f"Unsupported foot input: {self.input.shape}")
        self.times, self.total_times = [], []
        self.capture_next, self.fixture = False, None

    def predict(self, image: np.ndarray, box: np.ndarray) -> dict:
        total_start = time.perf_counter()
        tensor, transform = letterbox(image, box)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input.name: tensor})
        self.times.append((time.perf_counter() - started) * 1000)
        if len(outputs) != 1:
            raise ValueError("Expected one FOCUS output tensor")
        result = decode_top_one(outputs[0], self.metadata, transform)
        result["transform"] = transform
        self.total_times.append((time.perf_counter() - total_start) * 1000)
        if self.capture_next:
            self.fixture = {"rgba": cv2.cvtColor(image, cv2.COLOR_BGR2RGBA), "tensor": tensor,
                            "output": outputs[0], "transform": transform, "result": result}
            self.capture_next = False
        return result

    def timing(self):
        samples = self.times[1:] if len(self.times) > 1 else self.times
        total = self.total_times[1:] if len(self.total_times) > 1 else self.total_times
        return {"calls": len(self.times), "warmupExcluded": len(self.times) > 1,
                "sessionMeanMs": float(np.mean(samples)) if samples else None,
                "sessionP95Ms": float(np.percentile(samples, 95)) if samples else None,
                "preprocessInferDecodeMeanMs": float(np.mean(total)) if total else None,
                "scope": "GitHub Actions CPU; excludes body inference and video decoding; not browser/GPU timing"}


def insert_predictions(baseline: np.ndarray, baseline_scores: np.ndarray,
                       result: dict, selective: bool):
    prediction, scores = baseline.copy(), baseline_scores.copy()
    use = np.ones(8, dtype=bool)
    if selective:
        use &= result["detectionScore"] >= REFINE_DETECTION_MIN
        use &= result["keypointScores"] >= REFINE_KEYPOINT_MIN
    elif not result["found"]:
        prediction[COCO_INDICES] = np.nan
        scores[COCO_INDICES] = 0
        return prediction, scores, []
    prediction[COCO_INDICES[use]] = result["points"][use]
    scores[COCO_INDICES[use]] = result["detectionScore"] * result["keypointScores"][use]
    return prediction, scores, COCO_INDICES[use].tolist()


def record_metrics(bucket: dict, annotation: dict, target: np.ndarray,
                   prediction: np.ndarray, confidence: np.ndarray, changed=None):
    add_metrics(bucket, annotation, target, prediction, confidence, changed)
    indices = np.array([15, 16])
    valid = target[indices, 2] > 0
    if changed is not None:
        valid &= changed[indices]
    selected = indices[valid]
    if not len(selected):
        return
    error = np.linalg.norm(prediction[selected] - target[selected, :2], axis=1)
    error[~np.isfinite(error)] = float("inf")
    value = bucket.setdefault("ankle", {"errors": [], "normalized": [], "confidences": [], "people": set()})
    value["errors"].extend(error.tolist())
    value["normalized"].extend((error / max(annotation["bbox"][2:])).tolist())
    value["confidences"].extend(confidence[selected].tolist())
    value["people"].add(annotation["id"])


def summarize_foot(bucket: dict) -> dict:
    return {name: value for name, value in summarize_metrics(bucket).items() if name in FOOT_GROUPS}


def clean_array(array: np.ndarray):
    return [[float(value) if math.isfinite(float(value)) else None for value in row] for row in array]


def write_fixture(model: FootModel, directory: Path):
    sample = model.fixture
    if sample is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    sample["rgba"].tofile(directory / "image-rgba.u8")
    np.asarray(sample["tensor"], dtype="<f4").tofile(directory / "expected-input.f32")
    np.asarray(sample["output"], dtype="<f4").tofile(directory / "output.f32")
    result = sample["result"]
    json_write(directory / "fixture.json", {"schemaVersion": 1, "model": model.metadata,
               "image": {"file": "image-rgba.u8", "width": sample["rgba"].shape[1], "height": sample["rgba"].shape[0]},
               "transform": sample["transform"],
               "expectedInput": {"file": "expected-input.f32", "shape": list(sample["tensor"].shape)},
               "output": {"file": "output.f32", "shape": list(sample["output"].shape)},
               "expected": {"detectionScore": result["detectionScore"], "found": result["found"],
                            "points": clean_array(result["points"]),
                            "keypointScores": result["keypointScores"].tolist(), "bboxXYXY": result["bboxXYXY"].tolist()},
               "scope": "One real image; preprocessing/decoder parity only, not an extra accuracy dataset"})


def main():
    require_actions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("foot-model-package"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/foot-results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody"))
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--body-model", default=BODY_MODEL)
    parser.add_argument("--max-people", type=int, default=14)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--emit-codec-fixture", action="store_true")
    args = parser.parse_args()
    if min(args.max_people, args.threads) < 1:
        parser.error("--max-people and --threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overlays").mkdir(exist_ok=True)
    manifest = json.loads((args.models / "foot-models.json").read_text(encoding="utf-8"))
    models = [FootModel(args.models, metadata, args.threads) for metadata in manifest["models"]]
    if not models:
        raise ValueError("No exported foot models to evaluate")
    annotation_path = fetch(DATA_ROOT + "/test_coco_wholebody.json", args.cache / "test_coco_wholebody.json", 2 * 1024 * 1024)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    eligible = [value for value in sorted(data["annotations"], key=lambda value: (value["image_id"], value["id"]))
                if not value.get("iscrowd", 0) and value.get("num_keypoints", 0) > 0 and min(value["bbox"][2:]) >= 20]
    selected = eligible[:args.max_people]
    if not selected:
        raise ValueError("No eligible annotated people")
    images = {value["id"]: value for value in data["images"]}
    licenses = {value["id"]: value for value in data["licenses"]}
    baseline_records, baseline_report = {}, None
    if args.baseline_results:
        baseline_report = json.loads((args.baseline_results / "report.json").read_text(encoding="utf-8"))
        if baseline_report["dataset"]["sourceRevision"] != SOURCE_REVISION or baseline_report["dataset"]["annotationSha256"] != sha256(annotation_path):
            raise ValueError("Baseline and foot experiment do not use identical pinned annotations")
        records = json.loads((args.baseline_results / "predictions.json").read_text(encoding="utf-8"))
        baseline_records = {(value["annotationId"], value["condition"]): value for value in records}
        for annotation in selected:
            for condition in ("clean", "foot"):
                if (annotation["id"], condition) not in baseline_records:
                    raise ValueError("Baseline artifact is missing a selected person/condition")
        body = None
        body_metadata = baseline_report["models"]["wholebody"]
    else:
        body = PoseModel(args.body_model, args.cache, "wholebody", args.threads)
        body_metadata = {"id": "RTMW-l-cocktail14-384x288", **body.metadata}
    measurements = defaultdict(lambda: defaultdict(dict))
    masked_measurements = defaultdict(lambda: defaultdict(dict))
    image_cache, attributions, predictions = {}, [], []
    replacement_counts = defaultdict(lambda: {"footPoints": 0, "anklePoints": 0, "calls": 0})
    for annotation in selected:
        image_id = annotation["image_id"]
        info = images[image_id]
        if image_id not in image_cache:
            path = fetch(DATA_ROOT + "/" + info["file_name"], args.cache / info["file_name"], 5 * 1024 * 1024)
            image_cache[image_id] = cv2.imread(str(path))
            if image_cache[image_id] is None:
                raise ValueError(f"Image decode failed: {path}")
            attributions.append({**info, "licenseDetail": licenses[info["license"]], "sha256": sha256(path),
                                 "retrievedFrom": DATA_ROOT + "/" + info["file_name"], "publishedOverlay": info["license"] == 4})
        image = image_cache[image_id]
        box = np.asarray(annotation["bbox"], dtype=np.float32)
        target = target_points(annotation)
        for condition in ("clean", "foot"):
            frame, changed = occlude(image, annotation, target, condition)
            if body is None:
                record = baseline_records[(annotation["id"], condition)]
                if record["imageId"] != image_id or not np.allclose(record["bboxXYWH"], box, atol=1e-3):
                    raise ValueError("Baseline crop or identity mismatch")
                baseline = np.asarray(record["prediction"], dtype=np.float32)
                baseline_scores = np.asarray(record["scores"], dtype=np.float32)
            else:
                baseline, baseline_scores = body.predict(frame, box)
            if baseline.shape != (133, 2) or baseline_scores.shape != (133,):
                raise ValueError("Unexpected wholebody baseline shape")
            record_metrics(measurements[condition]["rtmw_l"], annotation, target, baseline, baseline_scores)
            if changed.any():
                record_metrics(masked_measurements[condition]["rtmw_l"], annotation, target, baseline, baseline_scores, changed)
            foot_box, crop_source = predicted_foot_box(baseline, baseline_scores, box)
            crop_choices = (("person_crop", context_box(box), "same_person_box_with_1.25_context"),
                            ("predicted_foot_crop", foot_box, crop_source))
            for model in models:
                model_id = model.metadata["id"]
                for crop_name, crop_box, source in crop_choices:
                    if args.emit_codec_fixture and model.fixture is None and condition == "clean" and crop_name == "predicted_foot_crop" and info["license"] == 4:
                        model.capture_next = True
                    result = model.predict(frame, crop_box)
                    method_results = {}
                    for selective in (False, True):
                        mode = "selective" if selective else "direct"
                        method = f"{model_id}/{crop_name}/{mode}"
                        points, scores, replaced = insert_predictions(baseline, baseline_scores, result, selective)
                        record_metrics(measurements[condition][method], annotation, target, points, scores)
                        if changed.any():
                            record_metrics(masked_measurements[condition][method], annotation, target, points, scores, changed)
                        count = replacement_counts[f"{condition}/{method}"]
                        count["footPoints"] += sum(17 <= index <= 22 for index in replaced)
                        count["anklePoints"] += sum(index in (15, 16) for index in replaced)
                        count["calls"] += 1
                        method_results[mode] = (points, scores)
                    predictions.append({"annotationId": annotation["id"], "imageId": image_id, "condition": condition,
                                        "modelId": model_id, "crop": crop_name, "cropSource": source,
                                        "transform": result["transform"], "found": result["found"],
                                        "detectionScore": result["detectionScore"],
                                        "keypointScores": result["keypointScores"].tolist(),
                                        "cocoIndices": COCO_INDICES.tolist(), "points": clean_array(result["points"]),
                                        "bboxXYXY": result["bboxXYXY"].tolist(),
                                        "candidateIndex": result["candidateIndex"], "candidateCount": result["candidateCount"]})
                    if info["license"] == 4 and crop_name == "predicted_foot_crop":
                        panels = [overlay(frame, baseline, baseline_scores, target, f"{condition}: RTMW-l; green=label"),
                                  overlay(frame, *method_results["direct"], target, f"FOCUS {model.metadata['head']} direct"),
                                  overlay(frame, *method_results["selective"], target, "FOCUS score-gated + RTMW fallback")]
                        filename = args.output / "overlays" / f"{image_id}-{annotation['id']}-{condition}-{model_id}.jpg"
                        cv2.imwrite(str(filename), np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 94])
            print(json.dumps({"footInferred": True, "image": image_id, "person": annotation["id"], "condition": condition}), flush=True)
    metrics = {condition: {method: summarize_foot(bucket) for method, bucket in methods.items()}
               for condition, methods in measurements.items()}
    masked_metrics = {condition: {method: summarize_foot(bucket) for method, bucket in methods.items()}
                      for condition, methods in masked_measurements.items()}
    differences = {}
    for condition, methods in metrics.items():
        baseline_foot = methods["rtmw_l"].get("foot")
        if not baseline_foot:
            continue
        differences[condition] = {}
        for name, parts in methods.items():
            if name == "rtmw_l" or "foot" not in parts:
                continue
            candidate = parts["foot"]
            differences[condition][name] = {
                "pck020RefinedMinusBaseline": candidate["pck020"] - baseline_foot["pck020"],
                "meanErrorPxRefinedMinusBaseline": (candidate["meanErrorPx"] - baseline_foot["meanErrorPx"])
                if candidate["meanErrorPx"] is not None and baseline_foot["meanErrorPx"] is not None else None,
                "points": candidate["points"], "sameDenominator": candidate["points"] == baseline_foot["points"]}
    ranked = sorted((name for name in metrics["clean"] if name.endswith("/predicted_foot_crop/selective")),
                    key=lambda name: metrics["clean"][name].get("foot", {}).get("pck020", -1), reverse=True)
    best = ranked[0] if ranked else None
    recommendation = {"method": best, "adoptForDiagnosticUse": False,
                      "decisionRule": "At least two additional correct clean foot points; no loss on masked-point PCK; no increase in clean mean pixel error.",
                      "independentGeneralizationEstablished": False}
    if best and "foot" in metrics["clean"]["rtmw_l"]:
        delta = differences["clean"][best]
        masked_base = masked_metrics.get("foot", {}).get("rtmw_l", {}).get("foot", {})
        masked_candidate = masked_metrics.get("foot", {}).get(best, {}).get("foot", {})
        extra_correct = round(delta["pck020RefinedMinusBaseline"] * delta["points"])
        masked_no_loss = bool(masked_base and masked_candidate and masked_candidate["pck020"] >= masked_base["pck020"] - 1e-9)
        recommendation.update({"additionalCorrectCleanPoints": extra_correct, "maskedPointsDoNotRegress": masked_no_loss,
                               "adoptForDiagnosticUse": extra_correct >= 2 and masked_no_loss and
                               delta["meanErrorPxRefinedMinusBaseline"] is not None and delta["meanErrorPxRefinedMinusBaseline"] <= 0})
    prior_baseline_match = None
    if baseline_report:
        prior = baseline_report["metrics"]["clean"]["wholebody"].get("foot")
        current = metrics["clean"]["rtmw_l"].get("foot")
        if prior and current and prior["points"] == current["points"]:
            prior_baseline_match = abs(prior["pck020"] - current["pck020"]) <= 1e-9
            if not prior_baseline_match:
                raise ValueError("Reused baseline failed to reproduce the prior foot PCK")
    report = {"schemaVersion": 1, "status": "measured", "createdAt": datetime.now(timezone.utc).isoformat(),
              "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
              "realImageAccuracyMeasured": True, "realVideoAccuracyMeasured": False,
              "individualToeArticulationAccuracyMeasured": False,
              "dataset": {"name": "MMPose COCO-WholeBody diagnostic subset", "sourceRevision": SOURCE_REVISION,
                          "annotationSha256": sha256(annotation_path), "people": len(selected), "images": len(image_cache),
                          "annotationIds": [value["id"] for value in selected],
                          "personSelection": "same benchmark.py eligibility and sorted image_id/id; first max_people",
                          "bodyRoi": "ground-truth person box; detector accuracy is excluded",
                          "footRoi": "whole-person context or predicted eight-point extent; no ground-truth foot crop",
                          "trainingOverlapExcluded": False},
              "policy": {"detectionMin": DETECTION_MIN, "refineDetectionMin": REFINE_DETECTION_MIN,
                         "refineKeypointMin": REFINE_KEYPOINT_MIN, "cocoIndices": COCO_INDICES.tolist(),
                         "targetSelection": "highest detector score inside the supplied crop; no ground-truth identity matching",
                         "predictionScore": "detection score times keypoint head score; not calibrated visibility probability",
                         "footNormalization": "each foot's annotated big/small/heel span, identical to benchmark.py",
                         "missingDetections": "direct method records failures; selective method retains RTMW fallback",
                         "denominator": "all annotated points, including low-score and missing predictions"},
              "environment": {"python": platform.python_version(), "onnxruntime": ort.__version__, "opencv": cv2.__version__,
                              "threads": args.threads, "provider": "CPUExecutionProvider"},
              "models": {"baseline": body_metadata, "foot": [{**model.metadata, "timing": model.timing()} for model in models]},
              "baselineArtifactReused": bool(baseline_report), "priorBaselinePckReproduced": prior_baseline_match,
              "metrics": metrics, "maskedPointMetrics": masked_metrics, "footRefinementDifference": differences,
              "replacementCounts": dict(replacement_counts), "recommendation": recommendation,
              "limits": ["Small image fixture, not representative COCO AP or an independent test split.",
                         "FOCUS is trained on COCO-WholeBody; training/validation overlap with this diagnostic fixture has not been excluded.",
                         "Controlled image masks do not establish performance under real motion, clothing, depth ambiguity or complete occlusion.",
                         "Eight foot/ankle points provide big-toe tip, small-toe tip, heel and ankle only, not five toe articulations.",
                         "Scores are model outputs, not proof that a landmark is directly visible.",
                         "CPU timings exclude body inference and do not measure the user's browser or GPU."],
              "sources": manifest["sources"] + ["https://github.com/jin-s13/COCO-WholeBody#terms-of-use"]}
    json_write(args.output / "foot-report.json", report)
    json_write(args.output / "foot-predictions.json", predictions)
    json_write(args.output / "license-attribution.json", {"annotationLicense": "https://creativecommons.org/licenses/by-nc/4.0/legalcode", "images": attributions})
    for model in models:
        if args.emit_codec_fixture:
            write_fixture(model, args.output / "codec-fixtures" / model.metadata["id"])
    rows = ["# FOCUS foot refinement: measured image results", "",
            f"{len(selected)} people / {len(image_cache)} images. Same annotations and PCK normalization as benchmark.py.",
            "", "| Condition | Method | Foot points | PCK@0.2 | Mean px | Invalid |",
            "|---|---|---:|---:|---:|---:|"]
    for condition, methods in metrics.items():
        for name, parts in methods.items():
            if "foot" in parts:
                value = parts["foot"]
                error = f"{value['meanErrorPx']:.3f}" if value["meanErrorPx"] is not None else "unavailable"
                rows.append(f"| {condition} | {name} | {value['points']} | {value['pck020']:.4f} | {error} | {value['invalidPredictionCount']} |")
    rows += ["", "## Scope", "", "All candidates, failures and mask-only results are retained in foot-report.json.",
             "The selected candidate is a diagnostic result, with no claim of five-toe articulation, independent generalization, or video accuracy.",
             "Ground-truth person boxes are shared by both baselines; foot crops come only from predictions."]
    (args.output / "foot-report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"footRefinementDifference": differences, "recommendation": recommendation,
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
