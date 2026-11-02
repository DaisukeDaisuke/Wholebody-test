#!/usr/bin/env python3
"""Evaluate actual person detection -> RTMW -> Hand5 on GitHub Actions only.

The largest person detection is chosen without looking at an annotation.
The largest annotated person is a fixed scoring target, never an oracle crop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from benchmark import (
    BODY_MODEL, DATA_ROOT, HAND_MODEL, SELECTIVE_HAND_POLICY, SOURCE_REVISION,
    PoseModel, add_metrics, fetch, json_write, nearest_annotated_body, overlay,
    refine_hands, resolve_model, selective_refinement, sha256, summarize_metrics,
    target_points,
)


DETECTOR_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
DETECTOR_ID = "person-yolox-tiny-416"
HF_REPOSITORY = "RabbitDaisuke/Wholebody-Refinement-onnx"
MODEL_FILENAME = "yolox-tiny-416.onnx"
SOURCES = [
    "https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/yolox/humanart/yolox_tiny_8xb8-300e_humanart.py",
    "https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/yolox/humanart/yolox_s_8xb8-300e_humanart.py",
    "https://github.com/Tau-J/rtmlib/blob/03a1693e59e4f7cd84582c0fb30459b3bf18ad42/rtmlib/tools/object_detection/yolox.py",
    "https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
]


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        if not len(rest):
            break
        low = np.maximum(boxes[current, :2], boxes[rest, :2])
        high = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        intersection = np.prod(np.maximum(0, high - low), axis=1)
        area = np.prod(np.maximum(0, boxes[current, 2:] - boxes[current, :2]))
        other_area = np.prod(np.maximum(0, boxes[rest, 2:] - boxes[rest, :2]), axis=1)
        iou = intersection / np.maximum(area + other_area - intersection, 1e-12)
        order = rest[iou <= threshold]
    return keep


class PersonDetector:
    def __init__(self, source: str, cache: Path, threads: int,
                 score_threshold: float = 0.3, nms_threshold: float = 0.45):
        self.path, self.metadata = resolve_model(source, cache, "person-yolox-tiny")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or (isinstance(shape[1], int) and shape[1] != 3):
            raise ValueError(f"Expected NCHW detector input, received {shape}")
        if model_input.type != "tensor(float)":
            raise ValueError(f"Expected original float32 detector input, received {model_input.type}")
        self.height = shape[2] if isinstance(shape[2], int) else 416
        self.width = shape[3] if isinstance(shape[3], int) else 416
        self.input_name = model_input.name
        self.score_threshold, self.nms_threshold = score_threshold, nms_threshold
        self.times = []
        self.runtime_outputs = None
        self.output_format = None
        self.capture_next = False
        self.codec_fixture = None
        graph = onnx.load(str(self.path), load_external_data=False)
        self.metadata.update({
            "id": DETECTOR_ID, "input": {"name": model_input.name, "shape": shape, "type": model_input.type},
            "outputs": [{"name": value.name, "shape": value.shape, "type": value.type}
                        for value in self.session.get_outputs()],
            "inputWH": [self.width, self.height], "color": "BGR", "inputRange": [0, 255],
            "normalization": "none", "layout": "NCHW float32",
            "resize": "OpenCV INTER_LINEAR; ratio=min(inputW/imageW,inputH/imageH); floor resized dimensions",
            "padding": {"value": 114, "placement": "image at top-left; pad right and bottom"},
            "personClassId": 0, "scoreThreshold": score_threshold, "nmsThreshold": nms_threshold,
            "onnxOperators": sorted({node.op_type for node in graph.graph.node}),
            "provider": "CPUExecutionProvider", "unmodifiedOfficialOnnx": True,
            "sources": SOURCES,
        })

    def predict(self, image: np.ndarray) -> list[dict]:
        height, width = image.shape[:2]
        ratio = min(self.width / width, self.height / height)
        resized_width, resized_height = max(1, int(width * ratio)), max(1, int(height * ratio))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        padded[:resized_height, :resized_width] = resized
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        self.times.append((time.perf_counter() - started) * 1000)
        if self.runtime_outputs is None:
            self.runtime_outputs = [{"name": value.name, "shape": list(array.shape), "dtype": str(array.dtype)}
                                    for value, array in zip(self.session.get_outputs(), outputs)]
        if any(not np.isfinite(array).all() for array in outputs):
            raise ValueError("Person detector produced non-finite outputs")
        packed = next((array for array in outputs if array.ndim == 3 and array.shape[0] == 1
                       and array.shape[-1] == 5), None)
        if packed is not None:
            boxes, scores = packed[0, :, :4].copy(), packed[0, :, 4].copy()
            labels = next((array.reshape(-1) for array in outputs if array is not packed
                           and array.size == len(scores) and array.ndim in (1, 2, 3)), None)
            if labels is None:
                raise ValueError("NMS output has boxes/scores but no class labels; cannot verify person class=0")
            valid = (labels == 0) & (scores >= self.score_threshold)
            boxes, scores = boxes[valid], scores[valid]
            self.output_format = "nms_xyxy_score_plus_labels"
        else:
            expected_rows = sum((self.height // stride) * (self.width // stride) for stride in (8, 16, 32))
            raw = next((array for array in outputs if array.ndim == 3 and array.shape[0] == 1
                        and array.shape[1] == expected_rows and array.shape[2] >= 6), None)
            if raw is None:
                raise ValueError(f"Unsupported YOLOX output contract: {[array.shape for array in outputs]}")
            prediction = raw[0].copy()
            grids, strides = [], []
            for stride in (8, 16, 32):
                gx, gy = np.meshgrid(np.arange(self.width // stride), np.arange(self.height // stride))
                grids.append(np.stack((gx, gy), axis=-1).reshape(-1, 2))
                strides.append(np.full((gx.size, 1), stride))
            grid, expanded_stride = np.concatenate(grids), np.concatenate(strides)
            center = (prediction[:, :2] + grid) * expanded_stride
            size = np.exp(np.clip(prediction[:, 2:4], -20, 20)) * expanded_stride
            boxes = np.concatenate((center - size / 2, center + size / 2), axis=1)
            # COCO class 0 is person. Never select the highest arbitrary class.
            scores = prediction[:, 4] * prediction[:, 5]
            valid = scores >= self.score_threshold
            boxes, scores = boxes[valid], scores[valid]
            self.output_format = "raw_grid_stride_xywh_objectness_class_scores"
        boxes = np.asarray(boxes, dtype=np.float64) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = nms(boxes, scores, self.nms_threshold)
        detections = []
        for index in keep:
            left, top, right, bottom = boxes[index]
            detections.append({"bboxXYWH": [float(left), float(top), float(right - left), float(bottom - top)],
                               "score": float(scores[index]), "classId": 0,
                               "area": float((right - left) * (bottom - top))})
        # Stable deterministic selection entirely from detector outputs.
        detections.sort(key=lambda value: (-value["area"], -value["score"],
                                           value["bboxXYWH"][0], value["bboxXYWH"][1]))
        if self.capture_next:
            self.codec_fixture = {
                "rgba": cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
                "input": tensor.copy(),
                "outputs": [(value.name, array.copy())
                            for value, array in zip(self.session.get_outputs(), outputs)],
                "transform": {"ratio": ratio, "imageWH": [width, height],
                              "inputWH": [self.width, self.height],
                              "resizedWH": [resized_width, resized_height],
                              "paddingValue": 114, "paddingLeft": 0, "paddingTop": 0,
                              "color": "BGR", "normalization": "none"},
                "detections": detections,
                "outputFormat": self.output_format,
            }
            self.capture_next = False
        return detections

    def summary(self) -> dict:
        samples = self.times[1:] if len(self.times) > 1 else self.times
        return {**self.metadata, "runtimeOutputs": self.runtime_outputs, "outputFormat": self.output_format,
                "timing": {"calls": len(self.times), "warmupExcluded": len(self.times) > 1,
                           "meanMs": float(np.mean(samples)) if samples else None,
                           "scope": "CPU session.run only; not browser, WebGPU, or end-to-end latency"}}


def box_iou(a: list, b: list) -> float:
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(np.asarray(a[:2]) + a[2:], np.asarray(b[:2]) + b[2:])
    intersection = float(np.prod(np.maximum(0, high - low)))
    return intersection / max(a[2] * a[3] + b[2] * b[3] - intersection, 1e-12)


def nullable_points(points: np.ndarray) -> list:
    return [[float(value) if np.isfinite(value) else None for value in point] for point in points]


def emit_detector_fixture(folder: Path, detector: PersonDetector, image_info: dict,
                          image_path: Path, license_detail: dict) -> dict:
    """Record one already-computed CC BY inference for the JavaScript codec check."""
    if image_info.get("license") != 4:
        raise ValueError("Detector fixture publication is restricted to the CC BY image")
    fixture = detector.codec_fixture
    if fixture is None:
        raise ValueError("No actual detector inference was captured")
    folder.mkdir(parents=True, exist_ok=True)
    rgba = np.ascontiguousarray(fixture["rgba"], dtype=np.uint8)
    rgba.tofile(folder / "image-rgba.u8")
    np.asarray(fixture["input"], dtype="<f4").tofile(folder / "expected-input.f32")
    outputs = []
    extensions = {"float32": "f32", "float64": "f64", "int64": "i64", "int32": "i32"}
    for index, (name, array) in enumerate(fixture["outputs"]):
        dtype = array.dtype.name
        if dtype not in extensions:
            raise ValueError(f"Unsupported fixture output dtype: {dtype}")
        filename = f"output-{index}.{extensions[dtype]}"
        np.asarray(array, dtype=array.dtype.newbyteorder("<")).tofile(folder / filename)
        outputs.append({"name": name, "shape": list(array.shape), "dtype": dtype,
                        "byteOrder": "little", "file": filename})
    manifest = {
        "schemaVersion": 1,
        "purpose": "One actual inference, shared by Python and JavaScript preprocessing/postprocessing parity checks",
        "source": {"imageId": image_info["id"], "imageUrl": DATA_ROOT + "/" + image_info["file_name"],
                   "imageSha256": sha256(image_path), "sourceRevision": SOURCE_REVISION,
                   "license": license_detail, "imageAttribution": image_info},
        "model": {"id": DETECTOR_ID, "onnxSha256": detector.metadata["onnxSha256"],
                  "archiveUrl": detector.metadata["source"], "outputFormat": fixture["outputFormat"]},
        "image": {"width": rgba.shape[1], "height": rgba.shape[0], "channels": 4,
                  "dtype": "uint8", "file": "image-rgba.u8"},
        "input": {"name": detector.input_name, "shape": list(fixture["input"].shape),
                  "dtype": "float32", "byteOrder": "little", "file": "expected-input.f32"},
        "outputs": outputs,
        "transform": fixture["transform"],
        "decoding": {"personClassId": 0, "scoreThreshold": detector.score_threshold,
                     "nmsThreshold": detector.nms_threshold,
                     "selection": "largest clipped person box area; descending score then x/y for ties"},
        "expectedDetections": fixture["detections"],
        "expectedChosen": fixture["detections"][0] if fixture["detections"] else None,
        "limits": "Uses the original Python model outputs for decoder comparison; browser model execution is not evaluated.",
    }
    json_write(folder / "manifest.json", manifest)
    return {"path": str(folder / "manifest.json"), "imageId": image_info["id"],
            "modelSha256": detector.metadata["onnxSha256"], "outputs": outputs}


def publish_model(detector: PersonDetector, cache: Path) -> dict:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face write token is available for the authorized model publication")
    folder = cache / "person-publication"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(detector.path, folder / MODEL_FILENAME)
    if sha256(folder / MODEL_FILENAME) != detector.metadata["onnxSha256"]:
        raise ValueError("Publication copy differs from the unmodified official ONNX")
    model = {"id": DETECTOR_ID, "kind": "personDetector", "filename": f"person/{MODEL_FILENAME}",
             "label": "YOLOX-tiny HumanArt person detector", "license": "Apache-2.0",
             "url": f"https://huggingface.co/{HF_REPOSITORY}/resolve/main/person/{MODEL_FILENAME}",
             "sha256": detector.metadata["onnxSha256"], "archiveUrl": DETECTOR_URL,
             **detector.summary()}
    json_write(folder / "manifest.json", {"schemaVersion": 1, "models": [model],
                                          "createdAt": datetime.now(timezone.utc).isoformat()})
    (folder / "NOTICE.md").write_text(
        "# YOLOX-tiny HumanArt person detector\n"
        "Unmodified official OpenMMLab ONNX, copied from the public SDK archive.\n"
        "No training, graph editing, quantization, or re-export was performed.\n"
        "OpenMMLab MMPose / MMDetection and YOLOX source code use Apache-2.0.\n"
        "Upstream model provenance and dataset conditions remain applicable independently.\n"
        f"Source: {DETECTOR_URL}\n"
        f"ONNX SHA-256: {detector.metadata['onnxSha256']}\n" + "\n".join(SOURCES) + "\n",
        encoding="utf-8")
    api = HfApi(token=token)
    commit = api.upload_folder(repo_id=HF_REPOSITORY, repo_type="model", folder_path=str(folder),
                               path_in_repo="person", commit_message="Publish verified original YOLOX-tiny person detector")
    return {"status": "published", "repo": HF_REPOSITORY, "revision": commit.oid,
            "url": f"https://huggingface.co/{HF_REPOSITORY}/resolve/{commit.oid}/person/{MODEL_FILENAME}",
            "manifestUrl": f"https://huggingface.co/{HF_REPOSITORY}/resolve/{commit.oid}/person/manifest.json",
            "sha256": detector.metadata["onnxSha256"]}


def make_markdown(report: dict) -> str:
    lines = ["# 人物検出からの実画像評価", f"実行: {report['createdAt']}",
             "最大のperson検出BBoxを選び、RTMWとHand5選択採用を実行する。正解BBoxはモデル入力・人物選択に使わない。",
             "採点は既存初期化評価と同じ4画像の最大注釈人物へ固定。人物を取り違えた誤差も残す。",
             "|入力経路|部位|点数|平均誤差px|PCK@0.10|PCK@0.20|", "|---|---|---:|---:|---:|---:|"]
    for method, parts in report["metrics"].items():
        for part in ("body", "hand", "foot"):
            if part not in parts:
                continue
            value = parts[part]
            error = "—" if value["meanErrorPx"] is None else f"{value['meanErrorPx']:.3f}"
            lines.append(f"|{method}|{part}|{value['points']}|{error}|{value['pck010']:.3f}|{value['pck020']:.3f}|")
    lines.extend(["検出がない場合も分母から除かずPCK失敗へ数える。平均px誤差は有限予測だけの平均なので、invalidPredictionCountも併記する。",
                  "同じ4画像を用いた診断であり、大規模精度保証・実動画・WebGPUの速度評価ではない。",
                  "手選択ルールは先行した同じ診断標本の失敗から設計したため、独立な汎化評価ではない。",
                  "## 取得元", *SOURCES])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/detector-report.json"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody"))
    parser.add_argument("--detector-model", default=DETECTOR_URL)
    parser.add_argument("--body-model", default=BODY_MODEL)
    parser.add_argument("--hand-model", default=HAND_MODEL)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-people", type=int, default=14)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--emit-codec-fixture", action="store_true",
                        help="Record one actual CC BY detector inference for the JavaScript codec comparison")
    args = parser.parse_args()
    if args.threads < 1 or args.max_people < 1:
        parser.error("--threads and --max-people must be positive")
    output = args.output.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    data_path = fetch(DATA_ROOT + "/test_coco_wholebody.json", args.cache / "test_coco_wholebody.json", 2 * 1024 * 1024)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    images = {value["id"]: value for value in data["images"]}
    eligible = [value for value in sorted(data["annotations"], key=lambda item: (item["image_id"], item["id"]))
                if not value.get("iscrowd", 0) and value.get("num_keypoints", 0) > 0
                and min(value["bbox"][2:]) >= 20]
    image_ids = {value["image_id"] for value in eligible[:args.max_people]}
    grouped = defaultdict(list)
    for annotation in eligible:
        if annotation["image_id"] in image_ids:
            grouped[annotation["image_id"]].append(annotation)
    if not grouped:
        raise ValueError("No images selected")
    detector = PersonDetector(args.detector_model, args.cache, args.threads)
    body = PoseModel(args.body_model, args.cache, "wholebody", args.threads)
    hand = PoseModel(args.hand_model, args.cache, "hand5", args.threads)
    baseline_path = output / "predictions.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else []
    baseline_index = {(value["imageId"], value["annotationId"]): value for value in baseline if value["condition"] == "clean"}
    measurements, cases, predictions, attributions = {}, [], [], []
    detector_fixture = None
    for image_id, candidates in sorted(grouped.items()):
        image_info = images[image_id]
        image_path = fetch(DATA_ROOT + "/" + image_info["file_name"], args.cache / image_info["file_name"], 5 * 1024 * 1024)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode image {image_id}")
        # This target is used only by metrics, diagnostics and overlay drawings.
        annotation = max(candidates, key=lambda value: (value["bbox"][2] * value["bbox"][3], -value["id"]))
        target = target_points(annotation)
        if args.emit_codec_fixture and image_info["license"] == 4 and detector_fixture is None:
            detector.capture_next = True
        detections = detector.predict(image)
        if detector.capture_next is False and detector.codec_fixture is not None and detector_fixture is None:
            detector_fixture = emit_detector_fixture(
                output / "detector-codec-fixture", detector, image_info, image_path,
                next(value for value in data["licenses"] if value["id"] == image_info["license"]))
        chosen = detections[0] if detections else None
        if chosen is not None:
            box = np.asarray(chosen["bboxXYWH"], dtype=np.float64)
            xy, scores = body.predict(image, box)
            refined, refined_scores, hand_records = refine_hands(hand, image, xy, scores, box)
            selected, selected_scores, decisions = selective_refinement(
                xy, scores, refined, refined_scores, hand_records, (hand.width, hand.height))
        else:
            xy = np.full((133, 2), np.nan)
            scores = np.zeros(133)
            refined, refined_scores = xy.copy(), scores.copy()
            selected, selected_scores, hand_records, decisions = xy.copy(), scores.copy(), [], []
        methods = [("detected_person", xy, scores), ("detected_person_hand_refined", refined, refined_scores),
                   ("detected_person_hand_selective", selected, selected_scores)]
        reference = baseline_index.get((image_id, annotation["id"]))
        if reference is not None:
            methods.extend([("gt_person_reference", np.asarray(reference["prediction"]), np.asarray(reference["scores"])),
                            ("gt_person_hand_refined_reference", np.asarray(reference["refined"]), np.asarray(reference["refinedScores"]))])
            if "selective" in reference:
                methods.append(("gt_person_hand_selective_reference", np.asarray(reference["selective"]),
                                np.asarray(reference["selectiveScores"])))
        per_case = {}
        for method, points, confidence in methods:
            add_metrics(measurements.setdefault(method, {}), annotation, target, points, confidence)
            bucket = {}
            add_metrics(bucket, annotation, target, points, confidence)
            per_case[method] = summarize_metrics(bucket)
        identity = nearest_annotated_body(xy, candidates, annotation["id"]) if chosen else {"nearestAnnotationId": None, "reason": "no detection"}
        case = {"imageId": image_id, "targetAnnotationId": annotation["id"],
                "targetBBoxForScoringOnly": annotation["bbox"], "detections": detections, "chosen": chosen,
                "chosenBoxIoUWithFixedTarget": box_iou(chosen["bboxXYWH"], annotation["bbox"]) if chosen else 0,
                "identityDiagnostic": identity, "metrics": per_case, "refinement": hand_records,
                "selectiveRefinement": decisions}
        cases.append(case)
        predictions.append({**case, "prediction": nullable_points(xy), "scores": scores.tolist(),
                            "refined": nullable_points(refined), "refinedScores": refined_scores.tolist(),
                            "selective": nullable_points(selected), "selectiveScores": selected_scores.tolist()})
        attributions.append({**image_info, "sha256": sha256(image_path), "retrievedFrom": DATA_ROOT + "/" + image_info["file_name"],
                             "licenseDetail": next(value for value in data["licenses"] if value["id"] == image_info["license"]),
                             "publishedOverlay": image_info["license"] == 4})
        if image_info["license"] == 4:
            panels = [overlay(image, points, confidence, target, title) for _, points, confidence, title in (
                (0, xy, scores, "detector largest person; green=fixed annotation"),
                (1, refined, refined_scores, "detector + Hand5 original gate"),
                (2, selected, selected_scores, "detector + selective Hand5"))]
            cv2.imwrite(str(output / "overlays" / f"{image_id}-person-detector.jpg"),
                        np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"person_detector image={image_id} detections={len(detections)} target={annotation['id']} "
              f"chosen_iou={case['chosenBoxIoUWithFixedTarget']:.4f}", flush=True)
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
              "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
              "dataset": {"sourceRevision": SOURCE_REVISION, "images": len(cases), "people": len(cases),
                          "annotationSha256": sha256(data_path), "groundTruthUsedForInference": False,
                          "targetSelection": "largest annotated person bbox per image; scoring only",
                          "detectorSelection": "largest clipped person bbox area, score>=0.3, classId=0, NMS IoU=0.45; no annotation access"},
              "models": {"detector": detector.summary(), "wholebody": body.metadata, "hand": hand.metadata},
              "metrics": {method: summarize_metrics(bucket) for method, bucket in measurements.items()},
              "cases": cases, "handSelectivePolicy": SELECTIVE_HAND_POLICY,
              "codecFixture": detector_fixture,
              "publication": {"status": "not_requested"},
              "limits": ["Four diagnostic images, not population-level detector or tracking accuracy.",
                         "Largest predicted box may differ from the largest annotated target; errors remain counted.",
                         "No person detection counts as a failure, not a skipped sample.",
                         "CPU inference does not demonstrate browser WebGPU support or throughput.",
                         "The hand gate was designed using earlier results from these same images."],
              "sources": SOURCES}
    baseline_report_path = output / "report.json"
    if baseline_report_path.is_file():
        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        initial = baseline_report.get("initializationExperiment", {})
        same_targets = [(value["imageId"], value["targetAnnotationId"]) for value in initial.get("cases", [])] == [
            (value["imageId"], value["targetAnnotationId"]) for value in cases]
        report["fullFrameComparison"] = {"sameTargets": same_targets,
                                         "metrics": initial.get("metrics", {}) if same_targets else {},
                                         "sourceGitCommit": baseline_report.get("gitCommit")}
    json_write(args.output, report)
    json_write(output / "detector-predictions.json", predictions)
    json_write(output / "detector-attribution.json", {"annotationLicense": "https://creativecommons.org/licenses/by-nc/4.0/legalcode", "images": attributions})
    args.output.with_suffix(".md").write_text(make_markdown(report), encoding="utf-8")
    if args.publish:
        try:
            report["publication"] = publish_model(detector, args.cache)
        except Exception as error:
            report["publication"] = {"status": "failed", "error": str(error)}
            json_write(args.output, report)
            raise
        json_write(args.output, report)
    print(json.dumps({"images": len(cases), "detector": detector.summary(),
                      "metrics": {method: {key: value for key, value in parts.items() if key in ("body", "hand", "foot")}
                                  for method, parts in report["metrics"].items()},
                      "publication": report["publication"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
