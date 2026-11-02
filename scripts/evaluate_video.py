"""Measure current observations on one or two local videos without publishing media.

This diagnostic uses the same pretrained MediaPipe task assets as the web app.
It measures observability, loss, motion and estimated depth separately. It does
not certify anatomical accuracy without ground-truth labels, and it does not
claim that Python CPU measurements are browser/GPU performance measurements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


GROUPS = {
    "body": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    "leftArm": [11, 13, 15], "rightArm": [12, 14, 16],
    "leftLeg": [23, 25, 27], "rightLeg": [24, 26, 28],
    "leftFoot": [27, 29, 31], "rightFoot": [28, 30, 32],
}
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
         (31, 27), (28, 30), (30, 32), (32, 28)]


def summary(values: list[float]) -> dict:
    values = [float(x) for x in values if math.isfinite(x)]
    if not values:
        return {"count": 0, "mean": None, "p05": None, "median": None,
                "p95": None, "min": None, "max": None, "range": None}
    array = np.asarray(values)
    return {"count": len(values), "mean": float(array.mean()), "p05": float(np.quantile(array, .05)),
            "median": float(np.median(array)), "p95": float(np.quantile(array, .95)),
            "min": float(array.min()), "max": float(array.max()), "range": float(np.ptp(array))}


def fetch_public(url: str, limit: int) -> bytes:
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("Automatic asset fetch is restricted to public Hugging Face URLs")
    request = urllib.request.Request(url, headers={"User-Agent": "wholebody-video-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(limit + 1)
    if len(data) > limit or not data:
        raise ValueError("Unexpected public model response size")
    return data


def resolve_task(args, key: str) -> Path:
    explicit = args.hand_task if key == "hands" else args.pose_task
    if explicit:
        if not explicit.is_file():
            raise ValueError("Explicit model task does not exist")
        return explicit
    if args.manifest.startswith("https://"):
        manifest = json.loads(fetch_public(args.manifest, 1_000_000).decode("utf-8"))
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model = manifest["models"][key]
    if model.get("format") != "mediapipe-task":
        raise ValueError("Expected a MediaPipe task asset")
    filename = Path(model["file"]).name
    if filename != model["file"]:
        raise ValueError("Unexpected model filename")
    args.cache.mkdir(parents=True, exist_ok=True)
    path = args.cache / filename
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == model["sha256"]:
        return path
    data = fetch_public(model["url"], int(model["bytes"]) + 1)
    if len(data) != model["bytes"] or hashlib.sha256(data).hexdigest() != model["sha256"]:
        raise ValueError("Downloaded model integrity mismatch")
    path.write_bytes(data)
    return path


def score(point) -> float:
    visibility = point.visibility if point.visibility is not None else 0.0
    presence = point.presence if point.presence is not None else 1.0
    return float(min(visibility, presence))


def observed(point, threshold: float) -> bool:
    return bool(point and score(point) >= threshold and
                np.isfinite([point.x, point.y, point.z]).all() and
                0 <= point.x <= 1 and 0 <= point.y <= 1)


def perspective_root(points: list, world: list, width: int, height: int,
                     hfov: float, threshold: float) -> tuple[list | None, float | None]:
    """Fit camera translation of root-relative learned XYZ, not true metric depth."""
    if not points or not world:
        return None, None
    focal = .5 * width / math.tan(math.radians(hfov) / 2)
    ids = [i for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
           if observed(points[i], threshold) and np.isfinite([world[i].x, world[i].y, world[i].z]).all()]
    if len(ids) < 4:
        return None, None
    a, b, weights = [], [], []
    for i in ids:
        u = (points[i].x * width - width / 2) / focal
        v = (points[i].y * height - height / 2) / focal
        x, y, z = world[i].x, world[i].y, world[i].z
        a.extend([[1, 0, -u], [0, 1, -v]])
        b.extend([u * z - x, v * z - y])
        weights.extend([score(points[i]) ** 2] * 2)
    a, b, initial = np.asarray(a), np.asarray(b), np.asarray(weights)
    weights = initial.copy()
    translation, residual = None, None
    for _ in range(3):
        weighted = a * np.sqrt(weights)[:, None]
        if np.linalg.matrix_rank(weighted) < 3:
            return None, None
        translation = np.linalg.lstsq(weighted, b * np.sqrt(weights), rcond=None)[0]
        if not np.isfinite(translation).all() or not .2 < translation[2] < 50:
            return None, None
        residual = []
        for i in ids:
            depth = world[i].z + translation[2]
            if depth <= .05:
                return None, None
            px = focal * (world[i].x + translation[0]) / depth + width / 2
            py = focal * (world[i].y + translation[1]) / depth + height / 2
            residual.append(math.hypot(px - points[i].x * width, py - points[i].y * height))
        weights = initial * np.repeat(np.minimum(1, 12 / np.maximum(residual, 1e-6)), 2)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    span = max(np.ptp([points[i].x * width for i in ids]),
               np.ptp([points[i].y * height for i in ids]), 32)
    return (translation.tolist(), rms) if rms / span <= .1 else (None, rms)


def natural_key(path: Path):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r"(\d+)", path.name)]


class Input:
    def __init__(self, path: Path, image_fps: float | None):
        self.frames, self.capture = None, None
        self.index, self.last_time = -1, -1.0
        self.pending, self.ended = None, False
        self.rotation_degrees, self.orientation_auto = 0.0, False
        if path.is_dir():
            if not image_fps or image_fps <= 0:
                raise ValueError("Image sequence input requires --image-fps; timing is not guessed")
            self.frames = sorted([p for p in path.iterdir() if p.is_file() and
                                  p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")], key=natural_key)
            if not self.frames:
                raise ValueError("No directly contained image frames in the input directory")
            self.fps, self.count = image_fps, len(self.frames)
            self.width, self.height = 0, 0
            self.timing = "explicit image sequence FPS"
        else:
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise ValueError("Video could not be opened by OpenCV/FFmpeg")
            self.rotation_degrees = float(self.capture.get(cv2.CAP_PROP_ORIENTATION_META))
            self.capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            self.orientation_auto = self.capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) >= .5
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Video frame rate unavailable")
            self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.timing = "decoder timestamps, frame-rate fallback when missing"

    def read(self):
        self.index += 1
        if self.frames is not None:
            if self.index >= self.count:
                return None
            frame = cv2.imread(str(self.frames[self.index]))
            if frame is None:
                raise ValueError("Image sequence contains an undecodable frame")
            timestamp = self.index / self.fps * 1000
        else:
            ok, frame = self.capture.read()
            if not ok:
                return None
            if not self.orientation_auto:
                rotation = int(round(self.rotation_degrees)) % 360
                if rotation in (90, 180, 270):
                    mode = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]
                    frame = cv2.rotate(frame, mode)
                elif rotation:
                    raise ValueError("Unsupported non-right-angle video orientation metadata")
            timestamp = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(timestamp) or timestamp <= self.last_time:
                timestamp = max(self.last_time + 1000 / self.fps, self.index * 1000 / self.fps)
        self.last_time = timestamp
        self.height, self.width = frame.shape[:2]
        return timestamp, frame

    def at(self, target_ms: float):
        if target_ms < 0 or self.ended:
            return None
        while self.pending is None or self.pending[0] < target_ms:
            self.pending = self.read()
            if self.pending is None:
                self.ended = True
                return None
        return self.pending

    def close(self):
        if self.capture is not None:
            self.capture.release()


class Measurements:
    def __init__(self, threshold: float, sample_hz: float):
        self.threshold, self.period = threshold, 1000 / sample_hz
        self.samples = self.pose_count = self.hand_count = 0
        self.values = defaultdict(list)
        self.groups = {key: {"counts": [], "missing": 0, "missingStart": None,
                             "maxMissingMs": 0.0} for key in GROUPS}
        self.previous, self.previous_time = {}, None

    def add(self, timestamp: float, result, hands, frame, hfov: float, elapsed_ms: float):
        self.samples += 1
        points = result.pose_landmarks[0] if result.pose_landmarks else []
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
        self.pose_count += bool(points)
        self.hand_count += bool(hands and hands.hand_landmarks)
        self.values["handsDetectedPerFrame"].append(len(hands.hand_landmarks) if hands else 0)
        self.values["inferenceMs"].append(elapsed_ms)
        valid = [i for i, p in enumerate(points) if observed(p, self.threshold)]
        self.values["visibleBody33"].append(len(valid))
        self.values["confidenceAllBodyPoints"].extend(score(p) for p in points)
        dt_ms = timestamp - self.previous_time if self.previous_time is not None else self.period
        interval_ms = max(self.period, dt_ms)
        for group, ids in GROUPS.items():
            record = self.groups[group]
            count = sum(i in valid for i in ids)
            record["counts"].append(count)
            if count == 0:
                record["missing"] += 1
                if record["missingStart"] is None:
                    record["missingStart"] = timestamp
                record["maxMissingMs"] = max(record["maxMissingMs"],
                                              timestamp - record["missingStart"] + interval_ms)
            else:
                record["missingStart"] = None
        if world and valid:
            z = [world[i].z for i in valid]
            self.values["relativeJointZSpreadMetres"].append(max(z) - min(z))
            self.values["relativeJointZMetres"].extend(z)
        dt = (timestamp - self.previous_time) / 1000 if self.previous_time is not None else None
        current = {i: (points[i].x, points[i].y) for i in valid}
        if dt and 0 < dt < 1:
            for i, xy in current.items():
                if i in self.previous:
                    self.values["observedImageMotionPerSecond"].append(math.dist(xy, self.previous[i]) / dt)
        self.previous, self.previous_time = current, timestamp
        root, reprojection = perspective_root(points, world, frame.shape[1], frame.shape[0], hfov, self.threshold)
        if root is not None:
            self.values["estimatedRootXMetres"].append(root[0])
            self.values["estimatedRootYMetres"].append(root[1])
            self.values["estimatedRootZMetres"].append(root[2])
        if reprojection is not None:
            self.values["rootReprojectionRmsPixels"].append(reprojection)
        return points, valid

    def report(self):
        return {
            "sampledFrames": self.samples,
            "poseDetectedFraction": self.pose_count / self.samples if self.samples else None,
            "handDetectedFraction": self.hand_count / self.samples if self.samples else None,
            "groups": {name: {"jointCount": len(GROUPS[name]), "observedJoints": summary(item["counts"]),
                               "fullyObservedFraction": sum(n == len(GROUPS[name]) for n in item["counts"]) / self.samples if self.samples else None,
                               "completelyMissingFraction": item["missing"] / self.samples if self.samples else None,
                               "longestCompletelyMissingSeconds": item["maxMissingMs"] / 1000}
                       for name, item in self.groups.items()},
            "measurements": {name: summary(values) for name, values in self.values.items()},
            "estimatedRootValidFraction": len(self.values["estimatedRootZMetres"]) / self.samples if self.samples else None,
            "anatomicalAccuracyMeasured": False, "depthGroundTruthAvailable": False,
            "zeroMotionIsNotAnAccuracyPass": True,
        }


def save_overlay(directory: Path, view_index: int, sample_index: int, frame, points, valid):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"view{view_index + 1}-{sample_index:06d}.jpg"
    if destination.exists():
        raise ValueError("Diagnostic overlay already exists; use a new output directory")
    image = frame.copy()
    height, width = image.shape[:2]
    for a, b in EDGES:
        if a in valid and b in valid:
            cv2.line(image, (round(points[a].x * width), round(points[a].y * height)),
                     (round(points[b].x * width), round(points[b].y * height)), (80, 230, 100), 2)
    for i in valid:
        p = (round(points[i].x * width), round(points[i].y * height))
        cv2.circle(image, p, 4, (60, 210, 255), -1)
        cv2.putText(image, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    cv2.putText(image, f"view {view_index + 1} | current visible points {len(valid)}/33",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError("Cannot write local diagnostic overlay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--offset-ms", type=float, action="append", default=[])
    parser.add_argument("--image-fps", type=float)
    parser.add_argument("--pose-task", type=Path)
    parser.add_argument("--hand-task", type=Path)
    parser.add_argument("--pose-model", choices=["pose_full", "pose_lite"], default="pose_full")
    parser.add_argument("--manifest", default="https://huggingface.co/RabbitDaisuke/Wholebody-Video-Tracking/resolve/main/manifest.json")
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody-video"))
    parser.add_argument("--sample-hz", type=float, default=10)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--horizontal-fov", type=float, default=60)
    parser.add_argument("--hands", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/video-evaluation.json"))
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--overlay-every-seconds", type=float, default=5)
    args = parser.parse_args()
    if not 1 <= len(args.input) <= 2:
        parser.error("Supply one video or two simultaneous views")
    if not 0 < args.sample_hz <= 60 or not 0 <= args.confidence <= 1 or not 20 <= args.horizontal_fov <= 150:
        parser.error("Invalid sample frequency, confidence or field of view")
    if args.max_seconds <= 0 or args.overlay_every_seconds <= 0:
        parser.error("Explicit positive duration and overlay interval are required")
    if len(args.offset_ms) > len(args.input):
        parser.error("More offsets than input views")
    pose_path = resolve_task(args, args.pose_model)
    hand_path = resolve_task(args, "hands") if args.hands else None
    inputs = [Input(path, args.image_fps) for path in args.input]
    records = [Measurements(args.confidence, args.sample_hz) for _ in inputs]
    offsets = args.offset_ms + [0] * (len(inputs) - len(args.offset_ms))
    pose_models, hand_models = [], []
    sync = []
    last_timestamps = [-1] * len(inputs)
    started = time.perf_counter()
    try:
        for _ in inputs:
            base = mp.tasks.BaseOptions(model_asset_path=str(pose_path), delegate=mp.tasks.BaseOptions.Delegate.CPU)
            pose_models.append(mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(base_options=base, running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=1, min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
                    min_tracking_confidence=.5, output_segmentation_masks=False)))
            if hand_path:
                base = mp.tasks.BaseOptions(model_asset_path=str(hand_path), delegate=mp.tasks.BaseOptions.Delegate.CPU)
                hand_models.append(mp.tasks.vision.HandLandmarker.create_from_options(
                    mp.tasks.vision.HandLandmarkerOptions(base_options=base, running_mode=mp.tasks.vision.RunningMode.VIDEO,
                        num_hands=2, min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                        min_tracking_confidence=.5)))
        for sample in range(math.ceil(args.max_seconds * args.sample_hz)):
            target = sample / args.sample_hz * 1000
            captured = [source.at(target + offset) for source, offset in zip(inputs, offsets)]
            if all(source.ended for source in inputs):
                break
            aligned = [item[0] - offsets[i] for i, item in enumerate(captured) if item is not None]
            if len(aligned) == 2:
                sync.append(abs(aligned[0] - aligned[1]))
            for i, item in enumerate(captured):
                if item is None:
                    continue
                timestamp, frame = item
                stamp = round(timestamp)
                if stamp <= last_timestamps[i]:
                    continue
                last_timestamps[i] = stamp
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                start = time.perf_counter()
                result = pose_models[i].detect_for_video(image, stamp)
                hands = hand_models[i].detect_for_video(image, stamp) if hand_models else None
                elapsed = (time.perf_counter() - start) * 1000
                points, valid = records[i].add(timestamp, result, hands, frame, args.horizontal_fov, elapsed)
                if args.overlay_dir and sample % max(1, round(args.overlay_every_seconds * args.sample_hz)) == 0:
                    save_overlay(args.overlay_dir, i, sample, frame, points, valid)
            if sample and sample % max(1, round(args.sample_hz * 10)) == 0:
                print(json.dumps({"processedVideoSeconds": round(target / 1000, 2),
                                  "samplesPerView": [r.samples for r in records]}), flush=True)
    finally:
        for model in pose_models + hand_models:
            model.close()
        for source in inputs:
            source.close()
    report = {
        "schema": "wholebody-video-observation-v1", "poseModel": args.pose_model,
        "poseTaskSha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
        "runtime": "MediaPipe Python Tasks CPU", "runningMode": "VIDEO", "sampleHz": args.sample_hz,
        "confidenceThreshold": args.confidence, "horizontalFovAssumedDegrees": args.horizontal_fov,
        "views": [{"width": source.width, "height": source.height, "sourceFps": source.fps,
                   "rotationMetadataDegrees": source.rotation_degrees,
                   "decoderAutoRotationEnabled": source.orientation_auto,
                   "timestampSource": source.timing, "offsetMs": offsets[i], **records[i].report()}
                  for i, source in enumerate(inputs)],
        "pairedTimestampDifferenceMs": summary(sync), "triangulationPerformed": False,
        "elapsedSeconds": time.perf_counter() - started,
        "gpuUtilizationMeasured": False, "privateMediaUploaded": False,
        "stage4AccuracyCertified": False,
        "limitations": ["Detection and visibility measure observability, not anatomical ground-truth accuracy.",
                        "VIDEO tracking receives sampled frames; large time gaps can change tracking compared with the browser scheduler.",
                        "Relative joint Z and perspective-root Z are separate monocular estimates.",
                        "Two videos are sampled together using explicit offsets; camera calibration and 3D triangulation are not performed by this diagnostic.",
                        "Missing observations are counted, never replaced by frozen previous landmarks.",
                        "Python CPU timings do not measure WebGPU latency or GPU load.",
                        "Optional overlay images contain input pixels and are written locally only."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"reportWritten": True, "sampledFrames": [r.samples for r in records],
                      "accuracyCertified": False, "mediaUploaded": False}), flush=True)


if __name__ == "__main__":
    main()
