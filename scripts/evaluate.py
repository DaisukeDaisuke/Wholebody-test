"""Real annotated-image evaluation, executed by GitHub Actions only.

Ground-truth boxes isolate pose quality from person detection. No ground-truth
hand coordinates are provided to refinement: its ROI comes from the base model.
The mask experiment is single-frame corruption, not a temporal tracking test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".evaluation-cache"
COCO_COMMIT = "76f3b6331b6e15553fbee05951dbf49f1fad029c"
MMPOSE_COMMIT = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
EXAMPLE_URL = f"https://raw.githubusercontent.com/jin-s13/COCO-WholeBody/{COCO_COMMIT}/evaluation/example_gt.json"
VAL_URL = "https://huggingface.co/datasets/msdkhairi/coco2017/resolve/main/coco_wholebody_val_v1.0.json"
VAL_SHA = "f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e"
IMAGE_IDS = {785, 40083, 196141, 197388}
HF_RTMPOSE = "https://huggingface.co/RabbitDaisuke/RTMPose-Wholebody-onnx/resolve/9d7f318c43452baf33f1b39d39c1f8c03512aeb5"
HF_RTMW = "https://huggingface.co/RabbitDaisuke/rtmw3d-webgpu-onnx/resolve/2fc6f257743eb890a340a50bbaa3e46c5a313e83"
MODELS = {
    "rtmpose-s": {
        "url": f"{HF_RTMPOSE}/rtmpose-s-256x192-fp32.onnx",
        "sha256": "0dd72a9932d5f6eb539d02b3f5825e2ac5c15aeada5c0ff2dd9d5b34ab4103e8",
        "input": [192, 256],
    },
    "rtmpose-t": {
        "url": f"{HF_RTMPOSE}/rtmpose-t-256x192-fp32.onnx",
        "sha256": "fcaae4a08ea7610b4bcc5d0b9e8a7e80fd6f2d998be41438a1b2737056a7623f",
        "input": [192, 256],
    },
    "rtmw-l": {
        "url": f"{HF_RTMW}/rtmw2d/rtmw-l-256x192-fp32.onnx",
        "sha256": "8b6fb439dd02b02b0f71edaac647c8818e2335083ba1b3abc103f852a7c7c8d8",
        "input": [192, 256],
    },
    "rtmw-l-384": {
        "url": "https://huggingface.co/RabbitDaisuke/Wholebody-Refinement-onnx/resolve/0a5519d19e52477307916e99ef22024ac0faa241/rtmw-l-384x288-fp32.onnx",
        "sha256": "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054",
        "input": [288, 384],
    },
    "rtmw3d-x": {
        "url": f"{HF_RTMW}/onnx/rtmw3d-x-fp32.onnx",
        "sha256": "de7bc9dba71b4c4e2be91ec768db58d73afa78863cc834f6625e3a0b540f4c23",
        "input": [288, 384],
    },
}
HAND_MODEL = {
    "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
    "sha256": "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
}
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
GROUPS = {"body": np.arange(17), "feet": np.arange(17, 23), "hands": np.arange(91, 133)}
BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 18), (15, 19),
              (16, 20), (16, 21), (16, 22)]
HAND_EDGES = [(start, start + 1 + finger * 4) for start in [91, 112] for finger in range(5)]
HAND_EDGES += [(start + 1 + finger * 4 + j, start + 2 + finger * 4 + j)
               for start in [91, 112] for finger in range(5) for j in range(3)]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected=None, attempts=3, timeout=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected or sha256(path) == expected):
        return path
    for attempt in range(attempts):
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Wholebody-test-evaluation/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if expected and sha256(temporary) != expected:
                raise ValueError(f"SHA256 mismatch for {url}")
            temporary.replace(path)
            return path
        except (urllib.error.URLError, TimeoutError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def contract(sess):
    return {"inputs": [{"name": v.name, "shape": v.shape, "type": v.type} for v in sess.get_inputs()],
            "outputs": [{"name": v.name, "shape": v.shape, "type": v.type} for v in sess.get_outputs()]}


def ground_truth(annotation):
    point_groups = [annotation["keypoints"], annotation["foot_kpts"], annotation["face_kpts"],
                    annotation["lefthand_kpts"], annotation["righthand_kpts"]]
    points = np.array(sum(point_groups, []), dtype=np.float64).reshape(133, 3)
    valid = points[:, 2] > 0
    for begin, end, field in [(17, 23, "foot_valid"), (23, 91, "face_valid"),
                              (91, 112, "lefthand_valid"), (112, 133, "righthand_valid")]:
        valid[begin:end] &= bool(annotation[field])
    return points, valid


def load_dataset(kind, people):
    url = EXAMPLE_URL if kind == "official-example" else VAL_URL
    path = download(url, CACHE / (kind + ".json"), None if kind == "official-example" else VAL_SHA)
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    annotations = [a for a in data["annotations"]
                   if not a.get("iscrowd") and a.get("num_keypoints", 0) > 0
                   and (a["image_id"] in IMAGE_IDS if kind == "official-example" else a["image_id"] not in IMAGE_IDS)]
    # Stable selection independent of prediction quality and model identity.
    if kind == "official-example":
        annotations.sort(key=lambda a: (a["image_id"], a["id"]))
    else:
        annotations.sort(key=lambda a: hashlib.sha256(f"wholebody-phase2-holdout-v1:{a['id']}".encode()).hexdigest())
    annotations = annotations[:people]
    if not annotations:
        raise RuntimeError("No annotated people selected")
    selected_images = {}
    for image_id in sorted({a["image_id"] for a in annotations}):
        info = images[image_id]
        if image_id in IMAGE_IDS:
            image_urls = [f"https://raw.githubusercontent.com/open-mmlab/mmpose/{MMPOSE_COMMIT}/tests/data/coco/{info['file_name']}"]
        else:
            image_urls = [f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{info['file_name']}",
                          f"https://images.cocodataset.org/val2017/{info['file_name']}"]
        for source_index, image_url in enumerate(image_urls):
            try:
                path = download(image_url, CACHE / "images" / info["file_name"], attempts=1, timeout=20)
                break
            except (urllib.error.URLError, TimeoutError):
                if source_index == len(image_urls) - 1:
                    raise
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (info["height"], info["width"]):
            raise ValueError(f"Image dimensions do not match annotation: {image_id}")
        selected_images[image_id] = (image, {**info, "retrieved_url": image_url, "sha256": sha256(path)})
        print(json.dumps({"downloaded_image": image_id, "images_ready": len(selected_images)}), flush=True)
    return annotations, selected_images, {"url": url, "sha256": sha256(CACHE / (kind + ".json")),
                                         "selection": "image_id,id ascending for four example images; SHA256(annotation id) order with all four example images excluded for val-subset",
                                         "used_for_training": False, "used_for_threshold_tuning": False,
                                         "person_ids": [a["id"] for a in annotations], "licenses": data["licenses"]}


def pose_inference(sess, image, annotation, spec):
    width, height = spec["input"]
    x, y, bw, bh = annotation["bbox"]
    cx, cy = x + bw / 2, y + bh / 2
    rw, rh = bw * 1.25, bh * 1.25
    if rw / rh > width / height:
        rh = rw * height / width
    else:
        rw = rh * width / height
    roi = [cx - rw / 2, cy - rh / 2, rw, rh]
    # Inverse mapping uses the same full-image coordinate system as the browser.
    affine = np.array([[rw / width, 0, roi[0]], [0, rh / height, roi[1]]], dtype=np.float32)
    crop = cv2.warpAffine(image, affine, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    inputs = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])
    began = time.perf_counter()
    values = sess.run(None, {sess.get_inputs()[0].name: inputs})
    elapsed = (time.perf_counter() - began) * 1000
    outputs = dict(zip([o.name for o in sess.get_outputs()], values))
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite inference output")
    if "keypoints" in outputs:
        xy = outputs["keypoints"][0, :, :2].astype(np.float64)
        score = outputs["scores"][0].astype(np.float64)
    else:
        simcc_x = outputs.get("simcc_x", values[0])
        simcc_y = outputs.get("simcc_y", values[1])
        if simcc_x.shape != (1, 133, width * 2) or simcc_y.shape != (1, 133, height * 2):
            raise ValueError(f"Unexpected SimCC shape: {simcc_x.shape}, {simcc_y.shape}")
        xy = np.stack([simcc_x[0].argmax(-1), simcc_y[0].argmax(-1)], -1).astype(np.float64) / 2
        score = np.minimum(simcc_x[0].max(-1), simcc_y[0].max(-1)).astype(np.float64)
    xy[:, 0] = roi[0] + xy[:, 0] * rw / width
    xy[:, 1] = roi[1] + xy[:, 1] * rh / height
    return xy, score, elapsed, roi


def refine_hands(sess, image, xy, scores):
    refined, refined_scores = xy.copy(), scores.copy()
    notes = []
    milliseconds = 0.0
    for start, body_wrist in [(91, 9), (112, 10)]:
        palm_ids = np.array([0, 5, 9, 13, 17, 1, 2]) + start
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < image.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < image.shape[0])
        trusted = (scores >= .15) & inside
        if not trusted[start] or not trusted[start + 9] or np.count_nonzero(trusted[palm_ids]) < 4:
            notes.append({"start": start, "accepted": False, "reason": "weak-palm-observations"})
            continue
        ey = xy[start] - xy[start + 9]
        length = float(np.linalg.norm(ey))
        if length < 2:
            notes.append({"start": start, "accepted": False, "reason": "palm-too-small"})
            continue
        ey /= length
        ex = np.array([ey[1], -ey[0]])
        palm = xy[palm_ids[trusted[palm_ids]]]
        px, py = palm @ ex, palm @ ey
        span_x, span_y = float(np.ptp(px)), float(np.ptp(py))
        side = 3 * max(span_x, span_y)
        if side < 12 or side > np.hypot(*image.shape[:2]) * .8:
            notes.append({"start": start, "accepted": False, "reason": "invalid-crop-size"})
            continue
        center = ex * ((px.min() + px.max()) / 2) + ey * ((py.min() + py.max()) / 2 - .4 * span_y)
        origin = center - (ex + ey) * side / 2
        affine = np.array([[ex[0] * side / 224, ey[0] * side / 224, origin[0]],
                           [ex[1] * side / 224, ey[1] * side / 224, origin[1]]], dtype=np.float32)
        patch = cv2.warpAffine(image, affine, (224, 224), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        inp = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255
        began = time.perf_counter()
        out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(inp)})
        milliseconds += (time.perf_counter() - began) * 1000
        result = dict(zip([o.name for o in sess.get_outputs()], out))
        pixels = result.get("Identity", out[0]).reshape(21, 3)
        presence = float(result.get("Identity_1", out[1]).reshape(-1)[0])
        predicted = center + np.outer((pixels[:, 0] - 112) * side / 224, ex) + np.outer((pixels[:, 1] - 112) * side / 224, ey)
        wrist_anchor = xy[body_wrist] if trusted[body_wrist] else xy[start]
        wrist_error = float(np.linalg.norm(predicted[0] - wrist_anchor))
        all_inside = bool(((predicted[:, 0] >= 0) & (predicted[:, 0] < image.shape[1])
                           & (predicted[:, 1] >= 0) & (predicted[:, 1] < image.shape[0])).all())
        near_crop = bool((np.abs(predicted - center) <= side).all())
        accepted = bool(np.isfinite(predicted).all() and all_inside and near_crop and presence >= .65
                        and wrist_error <= max(18, .3 * side))
        notes.append({"start": start, "accepted": accepted, "presence": presence,
                      "wrist_disagreement_px": wrist_error, "crop_side_px": side})
        if accepted:
            refined[start:start + 21] = predicted
            # Presence is one hand-level score, not per-finger visibility.
            refined_scores[start:start + 21] = np.minimum(np.clip(scores[start:start + 21], 0, 1), presence)
    return refined, refined_scores, milliseconds, notes


def mask_frame(image, annotation):
    gt, valid = ground_truth(annotation)
    # Cover one valid hand if available, otherwise the valid feet. Ground truth
    # defines corruption location only; inference never receives the coordinates.
    target = next((np.arange(begin, end) for begin, end in [(112, 133), (91, 112), (17, 23)]
                   if np.count_nonzero(valid[begin:end]) >= 3), np.array([], dtype=int))
    affected = np.zeros(133, dtype=bool)
    if not target.size:
        return None, affected, None
    points = gt[target[valid[target]], :2]
    lo = np.floor(points.min(0) - 5).astype(int)
    hi = np.ceil(points.max(0) + 5).astype(int)
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [image.shape[1], image.shape[0]])
    masked = image.copy()
    masked[lo[1]:hi[1], lo[0]:hi[0]] = (114, 114, 114)
    affected = valid & (gt[:, 0] >= lo[0]) & (gt[:, 0] < hi[0]) & (gt[:, 1] >= lo[1]) & (gt[:, 1] < hi[1])
    return masked, affected, [*lo.tolist(), *hi.tolist()]


def metric_records(annotation, prediction, scores, affected=None):
    gt, valid = ground_truth(annotation)
    error = np.linalg.norm(prediction - gt[:, :2], axis=1)
    norm = float(max(annotation["bbox"][2:]))
    records = []
    for group, ids in GROUPS.items():
        chosen = ids[valid[ids]]
        if affected is not None:
            chosen = chosen[affected[chosen]]
        for idx in chosen:
            hand_norm = None
            if group == "hands":
                key = "lefthand_box" if idx < 112 else "righthand_box"
                hand_norm = max(annotation[key][2:])
            records.append({"group": group, "joint": int(idx), "error_px": float(error[idx]),
                            "normalized_error": float(error[idx] / norm),
                            "hand_normalized_error": float(error[idx] / hand_norm) if hand_norm and hand_norm > 0 else None,
                            "score": float(scores[idx]),
                            "visibility": int(gt[idx, 2]) if group == "body" else None})
    return records


def aggregate(records):
    results = {}
    for group in GROUPS:
        for visibility in ([None, 1, 2] if group == "body" else [None]):
            chosen = [r for r in records if r["group"] == group and (visibility is None or r["visibility"] == visibility)]
            label = group if visibility is None else group + ("_occluded" if visibility == 1 else "_visible")
            if not chosen:
                results[label] = {"joints": 0}
                continue
            error = np.array([r["error_px"] for r in chosen])
            normal = np.array([r["normalized_error"] for r in chosen])
            results[label] = {"joints": len(chosen), "mean_error_px": float(error.mean()),
                              "median_error_px": float(np.median(error)), "p95_error_px": float(np.percentile(error, 95)),
                              "pck_person_005": float((normal <= .05).mean()),
                              "pck_person_010": float((normal <= .10).mean()),
                              "coverage_score_025": float(np.mean([r["score"] >= .25 for r in chosen]))}
            if group == "hands":
                normalized = [r["hand_normalized_error"] for r in chosen if r["hand_normalized_error"] is not None]
                results[label]["pck_hand_020"] = float(np.mean(np.array(normalized) <= .20)) if normalized else None
    return results


def draw_evidence(image, annotation, baseline, refined, output):
    gt, valid = ground_truth(annotation)
    canvas = image.copy()
    for a, b in BODY_EDGES + HAND_EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, tuple(np.rint(gt[a, :2]).astype(int)), tuple(np.rint(gt[b, :2]).astype(int)), (60, 220, 60), 1)
    for i in np.r_[np.arange(23), np.arange(91, 133)]:
        if valid[i]:
            cv2.circle(canvas, tuple(np.rint(gt[i, :2]).astype(int)), 3, (60, 220, 60), 1)
            cv2.circle(canvas, tuple(np.rint(baseline[i]).astype(int)), 2, (0, 170, 255), -1)
            if i >= 91:
                cv2.circle(canvas, tuple(np.rint(refined[i]).astype(int)), 2, (255, 100, 255), -1)
    cv2.putText(canvas, "GT green / base orange / refined hand purple", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3)
    cv2.putText(canvas, "GT green / base orange / refined hand purple", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="rtmpose-s,rtmw-l")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--dataset", choices=["official-example", "val-subset"], default="official-example")
    parser.add_argument("--people", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlays").mkdir(exist_ok=True)
    if args.model_manifest:
        MODELS.update(json.loads(args.model_manifest.read_text(encoding="utf-8")))
    model_ids = args.models.split(",")
    unknown = set(model_ids) - set(MODELS)
    if unknown:
        raise ValueError(f"No pinned manifest for models: {sorted(unknown)}")
    annotations, images, data_source = load_dataset(args.dataset, args.people)
    hand = session(download(HAND_MODEL["url"], CACHE / "hand.onnx", HAND_MODEL["sha256"]))
    if hand.get_inputs()[0].shape != [1, 224, 224, 3]:
        raise ValueError(f"Unexpected hand input: {contract(hand)}")
    report = {"schema_version": 1, "git_sha": os.environ.get("GITHUB_SHA"),
              "backend": "CPUExecutionProvider", "runtime_version": ort.__version__,
              "dataset": {**data_source, "kind": args.dataset, "people": len(annotations),
                          "images": [info for _, info in images.values()]},
              "scope": "2D pose with ground-truth person boxes; no 3D, detector, temporal or browser/GPU accuracy claim",
              "hand_model": {**HAND_MODEL, "contract": contract(hand)}, "models": {},
              "limitations": ["Small fixed set is not a full benchmark.",
                              "Hand and foot visibility values are reliability labels, not binary occlusion ground truth.",
                              "Gray rectangles are controlled corruption, not real occlusion videos.",
                              "Scores never filter which annotated points count toward positional error.",
                              "GT body boxes are used; predicted hand landmarks alone determine hand crops.",
                              "CPU timings do not estimate RTX/WebGPU throughput."]}
    observations = {"schemaVersion": 1, "isTemporalSequence": False,
                    "description": "Independent annotated photographs, not consecutive frames or multi-view captures.",
                    "models": {}, "images": [info for _, info in images.values()]}
    for model_id in model_ids:
        spec = MODELS[model_id]
        pose = session(download(spec["url"], CACHE / (model_id + ".onnx"), spec["sha256"]))
        if pose.get_inputs()[0].shape != [1, 3, spec["input"][1], spec["input"][0]]:
            raise ValueError(f"Unexpected pose input: {contract(pose)}")
        collected = {key: [] for key in ["original", "hand_refined", "original_mask_target",
                                        "refined_mask_target", "masked_affected", "masked_refined_affected"]}
        frames, timings, hand_timings, model_observations = [], [], [], []
        accepted, attempted = 0, 0
        for ordinal, annotation in enumerate(annotations):
            image, _ = images[annotation["image_id"]]
            xy, score, milliseconds, roi = pose_inference(pose, image, annotation, spec)
            refined, refined_scores, hand_ms, notes = refine_hands(hand, image, xy, score)
            timings.append(milliseconds)
            hand_timings.append(hand_ms)
            attempted += len(notes)
            accepted += sum(n["accepted"] for n in notes)
            collected["original"].extend(metric_records(annotation, xy, score))
            collected["hand_refined"].extend(metric_records(annotation, refined, refined_scores))
            frames.append({"annotation_id": annotation["id"], "image_id": annotation["image_id"],
                           "roi": roi, "inference_ms": milliseconds, "hand_ms": hand_ms,
                           "hand_refinement": notes, "base_xy": xy.tolist(), "base_scores": score.tolist(),
                           "refined_xy": refined.tolist(), "refined_scores": refined_scores.tolist()})
            gt, gt_valid = ground_truth(annotation)
            model_observations.append({"type": "pose-observation", "sourceId": f"coco-{annotation['image_id']}",
                                       "annotationId": annotation["id"], "frameId": ordinal, "mediaTimeMs": None,
                                       "imageSize": {"width": image.shape[1], "height": image.shape[0]},
                                       "bboxXYWH": annotation["bbox"], "modelId": model_id,
                                       "keypoints": [{"id": i, "u": float(p[0]), "v": float(p[1]), "score": float(refined_scores[i]),
                                                      "relativeZm": None, "source": "image-inference"} for i, p in enumerate(refined)],
                                       "groundTruth": [{"id": i, "u": float(p[0]), "v": float(p[1]), "valid": bool(gt_valid[i])}
                                                       for i, p in enumerate(gt)],
                                       "handRefinement": notes})
            if ordinal < 6:
                draw_evidence(image, annotation, xy, refined, args.out / "overlays" / f"{model_id}-{annotation['id']}.png")
            masked, affected, mask_box = mask_frame(image, annotation)
            if masked is not None:
                collected["original_mask_target"].extend(metric_records(annotation, xy, score, affected))
                collected["refined_mask_target"].extend(metric_records(annotation, refined, refined_scores, affected))
                mxy, mscore, _, _ = pose_inference(pose, masked, annotation, spec)
                mrxy, mrscore, _, mask_notes = refine_hands(hand, masked, mxy, mscore)
                collected["masked_affected"].extend(metric_records(annotation, mxy, mscore, affected))
                collected["masked_refined_affected"].extend(metric_records(annotation, mrxy, mrscore, affected))
                frames[-1]["mask_experiment"] = {"rectangle": mask_box, "affected_joint_ids": np.flatnonzero(affected).tolist(),
                                                  "base_xy": mxy.tolist(), "refined_xy": mrxy.tolist(), "refinement": mask_notes}
            print(json.dumps({"model": model_id, "person": annotation["id"], "completed": ordinal + 1,
                              "total": len(annotations), "inference_ms": round(milliseconds, 2)}), flush=True)
        report["models"][model_id] = {**spec, "contract": contract(pose),
                                      "conditions": {key: aggregate(rows) for key, rows in collected.items()},
                                      "cpu_pose_ms": {"mean": float(np.mean(timings)), "p95": float(np.percentile(timings, 95))},
                                      "cpu_hand_ms_per_person": float(np.mean(hand_timings)),
                                      "refinement_accepted_hands": accepted, "refinement_attempted_hands": attempted}
        (args.out / f"{model_id}-predictions.json").write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")
        observations["models"][model_id] = model_observations
        del pose
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "observations.json").write_text(json.dumps(observations, ensure_ascii=False), encoding="utf-8")
    lines = ["# 実画像による姿勢評価", "", f"対象: {len(images)}画像 / {len(annotations)}人。COCO-WholeBody公式注釈、人物ROIは正解BBox。",
             "CPU実推論による2D位置誤差。ブラウザWebGPU速度、3D精度、実動画遮蔽追跡の合格結果ではありません。", "",
             "PCK(person=.05)は人物BBox長辺の5%以内。Hand PCK(.20)は手BBox長辺の20%以内。低信頼点も誤差の分母に含みます。", "",
             "| モデル | 条件 | 部位 | 関節数 | 平均px | PCK(person=.05) | Hand PCK(.20) |", "|---|---|---|---:|---:|---:|---:|"]
    for model_id, result in report["models"].items():
        for condition, metrics in result["conditions"].items():
            for group in GROUPS:
                row = metrics[group]
                if row["joints"]:
                    hp = row.get("pck_hand_020")
                    lines.append(f"| {model_id} | {condition} | {group} | {row['joints']} | {row['mean_error_px']:.2f} | {row['pck_person_005']:.1%} | {f'{hp:.1%}' if hp is not None else '—'} |")
    lines += ["", "masked_affectedは灰色矩形で隠した領域の元の正解点だけを集計。手専用モデルは人物モデルの予測掌点からROIを作り、正解の手位置は入力しません。",
              "", "原画像の権利は個々のCOCO/Flickr撮影者に帰属し、注釈はCOCO-WholeBodyの研究・非商用条件に従います。評価画像・注釈はアプリ本体へ同梱しません。",
              "", f"注釈: {data_source['url']}", f"実行コミット: {os.environ.get('GITHUB_SHA', 'unknown')}"]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["models"], indent=2), flush=True)


if __name__ == "__main__":
    main()
