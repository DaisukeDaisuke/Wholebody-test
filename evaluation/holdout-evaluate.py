#!/usr/bin/env python3
"""Frozen-rule validation on 32 COCO-WholeBody images unused by this PoC.

Run model inference only in GitHub Actions. The original dataset images and
masked derivatives stay in the cache; only numerical results are published.
This is a holdout from our gate-design examples, not a claim that upstream
model training or model selection excluded these public validation images.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from benchmark import (
    BODY_MODEL, HAND_MODEL, SELECTIVE_HAND_POLICY, PoseModel, add_metrics,
    fetch, json_write, nearest_annotated_body, occlude, refine_hands,
    selective_refinement, sha256, summarize_metrics, target_points,
)


ANNOTATION_SHA256 = "f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e"
ANNOTATION_BYTES = 132970317
ANNOTATION_URLS = [
    "https://drive.usercontent.google.com/download?id=1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb&export=download&confirm=t",
    "https://huggingface.co/datasets/msdkhairi/coco2017/resolve/159fa91cbeccbb944e666a399e43b4cf25e2d3e9/coco_wholebody_val_v1.0.json",
]
SELECTION_SEED = "wholebody-phase2-4fc7ff60-holdout-v1"
EXCLUDED_IMAGE_IDS = {785, 40083, 196141, 197388}
CONDITIONS = ("clean", "hand", "foot")
ROUTES = ("detected_person", "gt_person_reference")
CORE_PARTS = ("body", "hand", "foot")
PUBLISHED_KEYPOINT_IDS = list(range(23)) + list(range(91, 133))
FOOT_REVISION = "c7a40271d743a6cef447ee63c6246399418aa96f"
FOOT_REPOSITORY = "RabbitDaisuke/Foot-Refinement-onnx"
FOOT_MODELS = [
    {"id": "focus-foot8-one2many", "head": "one2many",
     "filename": "focus-foot8-one2many-640-fp32.onnx",
     "sha256": "f4f58eac45f40870fe2c6bc774400423c11f8e9a64ae1d9daa186a72e7142f7b",
     "bytes": 10687517},
    {"id": "focus-foot8-one2one", "head": "one2one",
     "filename": "focus-foot8-one2one-640-fp32.onnx",
     "sha256": "7569eddfa65c943b9a2fe0e1d6e13a57fbd7c9f69678c59863daf67d9dd7988b",
     "bytes": 10700493},
]
SOURCES = [
    "https://github.com/jin-s13/COCO-WholeBody#download",
    "https://github.com/jin-s13/COCO-WholeBody#terms-of-use",
    "https://huggingface.co/datasets/msdkhairi/coco2017/blob/159fa91cbeccbb944e666a399e43b4cf25e2d3e9/coco_wholebody_val_v1.0.json",
    f"https://huggingface.co/{FOOT_REPOSITORY}/tree/{FOOT_REVISION}",
]


def import_evaluation(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import the frozen evaluation implementation: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acquire_annotations(cache: Path) -> tuple[dict, dict]:
    destination = cache / "holdout" / "coco_wholebody_val_v1.0.json"
    failures = []
    for url in ANNOTATION_URLS:
        try:
            path = fetch(url, destination, 150 * 1024 * 1024)
            actual_sha = sha256(path)
            if path.stat().st_size != ANNOTATION_BYTES or actual_sha != ANNOTATION_SHA256:
                raise ValueError(f"Annotation bytes/hash mismatch: {path.stat().st_size}, {actual_sha}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("info", {}).get("version") != "1.0":
                raise ValueError("Expected original COCO-WholeBody annotation version 1.0")
            return data, {"url": url, "sha256": actual_sha, "bytes": path.stat().st_size,
                          "officialAndMirrorHashVerified": True, "version": "1.0",
                          "officialDistribution": ANNOTATION_URLS[0], "mirror": ANNOTATION_URLS[1],
                          "previousDownloadFailures": failures}
        except Exception as error:
            failures.append({"url": url, "error": str(error)})
            # A failed or truncated transfer must never become an accepted cache.
            if destination.is_file() and (destination.stat().st_size != ANNOTATION_BYTES or
                                           sha256(destination) != ANNOTATION_SHA256):
                destination.unlink()
    raise RuntimeError(f"Could not obtain the hash-pinned official annotations: {failures}")


def has_scorable_hand_and_foot(annotation: dict) -> bool:
    hand = any(annotation.get(f"{side}hand_valid", False)
               and max(annotation[f"{side}hand_box"][2:]) > 0
               and any(value > 0 for value in annotation[f"{side}hand_kpts"][2::3])
               for side in ("left", "right"))
    foot = False
    if annotation.get("foot_valid", False):
        for offset in (0, 9):
            points = annotation["foot_kpts"][offset:offset + 9]
            labeled = [points[index:index + 2] for index in (0, 3, 6) if points[index + 2] > 0]
            if len(labeled) >= 2 and any(max(point[axis] for point in labeled) >
                                         min(point[axis] for point in labeled) for axis in (0, 1)):
                foot = True
    return hand and foot


def select_holdout(data: dict, count: int):
    images = {value["id"]: value for value in data["images"]}
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        if (annotation["image_id"] not in EXCLUDED_IMAGE_IDS and not annotation.get("iscrowd", 0)
                and annotation.get("num_keypoints", 0) > 0 and min(annotation["bbox"][2:]) >= 20):
            grouped[annotation["image_id"]].append(annotation)
    candidates = []
    for image_id, people in grouped.items():
        # Fix the target before checking whether its hand/foot labels are scorable.
        target = max(people, key=lambda value: (value["bbox"][2] * value["bbox"][3], -value["id"]))
        if not has_scorable_hand_and_foot(target):
            continue
        selection_hash = hashlib.sha256(f"{SELECTION_SEED}:{image_id}:{target['id']}".encode()).hexdigest()
        candidates.append({"image": images[image_id], "annotation": target,
                           "people": people, "selectionHash": selection_hash})
    candidates.sort(key=lambda value: (value["selectionHash"], value["image"]["id"], value["annotation"]["id"]))
    if len(candidates) < count:
        raise ValueError(f"Expected {count} holdout targets but only {len(candidates)} are eligible")
    selected = candidates[:count]
    signature = hashlib.sha256("\n".join(f"{value['image']['id']}:{value['annotation']['id']}"
                                          for value in selected).encode()).hexdigest()
    return selected, {
        "seed": SELECTION_SEED, "count": count, "eligibleImageCount": len(grouped),
        "eligibleHandFootTargetCount": len(candidates), "selectionSha256": signature,
        "excludedImageIds": sorted(EXCLUDED_IMAGE_IDS),
        "bodyEligibility": "not crowd; num_keypoints>0; bbox width and height>=20 pixels",
        "targetRule": "maximum eligible body bbox area per image; smaller annotation ID breaks ties",
        "labelEligibilityAfterTargetFixing": "at least one valid labeled hand with nonzero bbox; at least one valid foot with two distinct labeled points",
        "order": "ascending SHA256(seed:imageId:annotationId), then imageId and annotationId",
        "selectionUsesModelOutputs": False, "replaceUnavailableImages": False,
        "selectionMadeBeforeAnyHoldoutModelInference": True,
        "excludedFromThisPocGateDesign": True, "upstreamTrainingOverlapExcluded": False,
    }


def acquire_image(image_info: dict, cache: Path) -> tuple[Path, dict]:
    filename = image_info["file_name"]
    if Path(filename).name != filename or filename != f"{image_info['id']:012d}.jpg":
        raise ValueError(f"Unexpected COCO image filename: {filename}")
    destination = cache / "holdout" / "val2017" / filename
    urls = [f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{filename}",
            f"https://images.cocodataset.org/val2017/{filename}"]
    failures = []
    for url in urls:
        try:
            path = fetch(url, destination, 8 * 1024 * 1024)
            # The image is decoded later; check the public JPEG format now.
            with path.open("rb") as stream:
                if stream.read(2) != b"\xff\xd8":
                    raise ValueError("Image response is not a JPEG")
            return path, {"imageId": image_info["id"], "url": url, "sha256": sha256(path),
                          "bytes": path.stat().st_size, "originalCocoUrl": image_info.get("coco_url"),
                          "licenseId": image_info["license"], "previousDownloadFailures": failures,
                          "publishedPixels": False}
        except Exception as error:
            failures.append({"url": url, "error": str(error)})
            destination.unlink(missing_ok=True)
    raise RuntimeError(f"Selected image {image_info['id']} unavailable: {failures}")


def core_metrics(bucket: dict) -> dict:
    return {key: value for key, value in summarize_metrics(bucket).items() if key in CORE_PARTS}


def numerical(value):
    if isinstance(value, np.ndarray):
        return numerical(value.tolist())
    if isinstance(value, np.generic):
        return numerical(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: numerical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numerical(item) for item in value]
    return value


def association_summary(association: dict) -> dict:
    selected = association["selected"]
    reasons = Counter(side["reason"] for value in association["candidates"] for side in value["sides"])
    return {"policyId": association["policyId"], "candidateCountEvaluated": association["candidateCountEvaluated"],
            "selected": selected["audit"] if selected else None, "sideReasonCounts": dict(reasons)}


def predict_route(image: np.ndarray, person_box, body: PoseModel, hand: PoseModel,
                  feet: dict, foot_api):
    if person_box is None:
        xy, scores = np.full((133, 2), np.nan), np.zeros(133)
        selected, selected_scores, decisions = xy.copy(), scores.copy(), []
    else:
        xy, scores = body.predict(image, person_box)
        refined, refined_scores, records = refine_hands(hand, image, xy, scores, person_box)
        selected, selected_scores, decisions = selective_refinement(
            xy, scores, refined, refined_scores, records, (hand.width, hand.height))
    methods = {"wholebody": (xy, scores), "hand_selective": (selected, selected_scores)}
    foot_audit = {}
    for head, model in feet.items():
        if person_box is None:
            associated, associated_scores = selected.copy(), selected_scores.copy()
            blended, blended_scores, replaced = selected.copy(), selected_scores.copy(), []
            foot_audit[head] = {"reason": "person_detector_missing", "replacedCocoIndices": []}
        else:
            # Person context only. No labeled foot box or foot coordinates enter inference.
            requested_box = foot_api.context_box(person_box)
            result = model.predict(image, requested_box)
            association = foot_api.associate_foot_candidates(result["candidates"], xy, scores)
            associated, associated_scores, replaced = foot_api.insert_associated(
                selected, selected_scores, association, blend=False)
            blended, blended_scores, _ = foot_api.insert_associated(
                selected, selected_scores, association, blend=True)
            foot_audit[head] = {"requestedBoxXYWH": requested_box.tolist(), "transform": result["transform"],
                                "association": association_summary(association), "replacedCocoIndices": replaced}
        methods[f"hand_selective_foot_{head}_associated"] = (associated, associated_scores)
        methods[f"hand_selective_foot_{head}_blend"] = (blended, blended_scores)
    return methods, {"handSelective": decisions, "foot": foot_audit}


def markdown(report: dict) -> str:
    lines = ["# 開発例と分離した32画像での固定ルール検証", f"実行: {report['createdAt']}",
             f"注釈 SHA-256: {ANNOTATION_SHA256}",
             "旧4画像を除外し、各画像の最大有効人物を先に固定。その人物に手足注釈がある画像からhash順32件を選んだ。",
             "検出人物cropと正解人物cropのreferenceを別々に表示する。選択・閾値・重みはこの結果を見て変更していない。",
             "|条件|入力|手法|部位|点数|無効予測|平均誤差px|PCK@.10|PCK@.20|",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for condition, routes in report["metrics"].items():
        for route, methods in routes.items():
            for method, parts in methods.items():
                for part in CORE_PARTS:
                    if part not in parts:
                        continue
                    value = parts[part]
                    error = "—" if value["meanErrorPx"] is None else f"{value['meanErrorPx']:.3f}"
                    lines.append(f"|{condition}|{route}|{method}|{part}|{value['points']}|"
                                 f"{value['invalidPredictionCount']}|{error}|{value['pck010']:.3f}|{value['pck020']:.3f}|")
    lines.extend(["## 解釈の範囲", *[f"- {value}" for value in report["limits"]],
                  "## 出典", *[f"- {source}" for source in SOURCES]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/holdout-report.json"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody"))
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--body-model", default=BODY_MODEL)
    parser.add_argument("--hand-model", default=HAND_MODEL)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Holdout model inference is authorized only in GitHub Actions")
    if args.threads < 1 or not 16 <= args.max_images <= 32:
        parser.error("--threads must be positive; --max-images must be between 16 and 32")
    output = args.output.parent
    output.mkdir(parents=True, exist_ok=True)
    data, annotation_source = acquire_annotations(args.cache)
    selected, selection_rule = select_holdout(data, args.max_images)
    selection = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
                 "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
                 "annotationSource": annotation_source, "rule": selection_rule,
                 "annotationLicense": "https://creativecommons.org/licenses/by-nc/4.0/legalcode",
                 "images": [{"imageId": item["image"]["id"], "annotationId": item["annotation"]["id"],
                             "selectionHash": item["selectionHash"], "fileName": item["image"]["file_name"],
                             "license": next(value for value in data["licenses"] if value["id"] == item["image"]["license"])}
                            for item in selected]}
    # This artifact is written before any model is loaded or run.
    json_write(output / "holdout-selection.json", selection)
    acquisition_failures = []
    for item, record in zip(selected, selection["images"]):
        try:
            item["path"], record["retrieval"] = acquire_image(item["image"], args.cache)
        except Exception as error:
            acquisition_failures.append({"imageId": item["image"]["id"], "error": str(error)})
    selection["acquisitionFailures"] = acquisition_failures
    json_write(output / "holdout-selection.json", selection)
    if acquisition_failures:
        json_write(args.output, {"status": "unavailable", "selection": selection,
                                 "reason": "Fixed images are never replaced after selection; no partial metric is presented"})
        raise RuntimeError(f"Could not acquire all fixed holdout images: {acquisition_failures}")

    detector_api = import_evaluation("detector-evaluate.py", "holdout_person_detector")
    foot_api = import_evaluation("foot-evaluate.py", "holdout_foot_refinement")
    detector = detector_api.PersonDetector(detector_api.DETECTOR_URL, args.cache, args.threads)
    body = PoseModel(args.body_model, args.cache, "wholebody", args.threads)
    hand = PoseModel(args.hand_model, args.cache, "hand5", args.threads)
    foot_directory = args.cache / "holdout" / "foot-models"
    feet, foot_metadata = {}, []
    for item in FOOT_MODELS:
        metadata = {**item, "license": "agpl-3.0", "revision": FOOT_REVISION,
                    "inputWH": [640, 640], "cocoIndices": [17, 18, 19, 15, 20, 21, 22, 16],
                    "url": f"https://huggingface.co/{FOOT_REPOSITORY}/resolve/{FOOT_REVISION}/{item['filename']}"}
        path = fetch(metadata["url"], foot_directory / metadata["filename"], 20 * 1024 * 1024)
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Pinned foot model bytes/hash mismatch: {metadata['filename']}")
        feet[metadata["head"]] = foot_api.FootModel(foot_directory, metadata, args.threads)
        foot_metadata.append(metadata)

    measurements, masked_measurements, cases, predictions = {}, {}, [], []
    for index, item in enumerate(selected, 1):
        image = cv2.imread(str(item["path"]))
        image_info, annotation = item["image"], item["annotation"]
        if image is None or list(image.shape[:2]) != [image_info["height"], image_info["width"]]:
            raise ValueError(f"Decoded image does not match annotation dimensions: {image_info['id']}")
        target = target_points(annotation)
        for condition in CONDITIONS:
            frame, changed = occlude(image, annotation, target, condition)
            detections = detector.predict(frame)
            chosen = detections[0] if detections else None
            detector_box = np.asarray(chosen["bboxXYWH"], dtype=np.float64) if chosen else None
            for route in ROUTES:
                person_box = detector_box if route == "detected_person" else np.asarray(annotation["bbox"], dtype=np.float64)
                methods, audit = predict_route(frame, person_box, body, hand, feet, foot_api)
                per_case = {}
                for method, (xy, scores) in methods.items():
                    bucket = measurements.setdefault(condition, {}).setdefault(route, {}).setdefault(method, {})
                    add_metrics(bucket, annotation, target, xy, scores)
                    local = {}
                    add_metrics(local, annotation, target, xy, scores)
                    per_case[method] = core_metrics(local)
                    if condition != "clean":
                        masked_bucket = masked_measurements.setdefault(condition, {}).setdefault(route, {}).setdefault(method, {})
                        add_metrics(masked_bucket, annotation, target, xy, scores, changed)
                xy, scores = methods["wholebody"]
                identity = (nearest_annotated_body(xy, item["people"], annotation["id"])
                            if person_box is not None else {"nearestAnnotationId": None, "reason": "person detector missing"})
                case = {"imageId": image_info["id"], "annotationId": annotation["id"], "condition": condition,
                        "route": route, "groundTruthPersonBoxUsedForInference": route == "gt_person_reference",
                        "inputPersonBoxXYWH": person_box.tolist() if person_box is not None else None,
                        "detections": len(detections), "chosenDetection": chosen if route == "detected_person" else None,
                        "identityDiagnostic": identity, "maskedKeypointIds": np.flatnonzero(changed).tolist(),
                        "metrics": per_case, "refinement": audit}
                cases.append(numerical(case))
                predictions.append({"imageId": image_info["id"], "annotationId": annotation["id"],
                                    "condition": condition, "route": route,
                                    "keypointIds": PUBLISHED_KEYPOINT_IDS,
                                    "methods": {method: {"xy": numerical(xy[PUBLISHED_KEYPOINT_IDS]),
                                                         "rawScores": numerical(scores[PUBLISHED_KEYPOINT_IDS])}
                                                for method, (xy, scores) in methods.items()}})
        print(f"holdout {index}/{len(selected)} image={image_info['id']} target={annotation['id']} complete", flush=True)

    def aggregate(values):
        return {condition: {route: {method: core_metrics(bucket) for method, bucket in methods.items()}
                            for route, methods in routes.items()} for condition, routes in values.items()}

    report = {"schemaVersion": 1, "status": "complete", "createdAt": datetime.now(timezone.utc).isoformat(),
              "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
              "dataset": {"name": "COCO-WholeBody val v1.0", "images": len(selected), "targets": len(selected),
                          "annotationSource": annotation_source, "selectionRule": selection_rule,
                          "developmentImagesExcluded": True, "trainingOverlapExcluded": False,
                          "publishedPixels": False},
              "conditions": list(CONDITIONS), "routes": list(ROUTES),
              "models": {"detector": detector.summary(), "body": body.metadata, "hand": hand.metadata,
                         "foot": [{**value, "timing": feet[value['head']].timing()} for value in foot_metadata]},
              "policies": {"handSelective": SELECTIVE_HAND_POLICY, "footAssociation": foot_api.ASSOCIATION_POLICY,
                           "footContext": "1.25 times the current person crop, matching foot-evaluate.py context_box",
                           "footBlend": "fixed 0.5 of accepted same-side candidate coordinates; baseline score retained",
                           "frozenBeforeHoldoutInference": True},
              "metrics": aggregate(measurements), "maskedPointMetrics": aggregate(masked_measurements),
              "cases": cases, "sources": SOURCES,
              "limits": [
                  "32 images were excluded from this PoC's four gate-design images; upstream training/development overlap is not excluded.",
                  "Target selection requires scorable hand and foot annotations and is not a population-representative sample.",
                  "Detected-person and ground-truth-person reference routes are separate; the detector never uses ground truth to pick a box.",
                  "Controlled hand/foot masks use annotation locations only to define perturbations, not inference crops or refinement decisions.",
                  "A missing person detection remains an invalid prediction and a PCK failure for every labeled point.",
                  "Mean pixel error uses finite predictions; invalidPredictionCount must be read alongside it.",
                  "32 independent images support only limited validation, not production accuracy or temporal/multiview performance.",
                  "COCO foot landmarks are heel, big-toe tip and small-toe tip per foot; five articulated toes are not evaluated.",
                  "Raw scores have model-specific scales and are not calibrated visibility or correctness probabilities.",
                  "Source JPEGs and all masked frames remain in cache and are not included in the numerical artifacts.",
              ]}
    json_write(args.output, numerical(report))
    json_write(output / "holdout-predictions.json", predictions)
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "images": len(selected),
                      "selectionSha256": selection_rule["selectionSha256"], "metrics": report["metrics"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
