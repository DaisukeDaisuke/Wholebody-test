"""Small, bounded Panoptic data probe and real multi-view evaluation.

Run network downloads and inference in GitHub Actions, not in the local workspace.
The probe uses only Python's standard library and never downloads complete videos.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request


SEQUENCE = "171204_pose1_sample"
ENDPOINTS = [
    "https://domedb.perception.cs.cmu.edu",
    "http://domedb.perception.cs.cmu.edu",
    "http://vcl.snu.ac.kr/panoptic",
    "https://vcl.snu.ac.kr/panoptic",
]
USER_AGENT = "Wholebody-test-Panoptic-research/1.0"
TOOLBOX = "https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".evaluation-cache" / "panoptic"
FPS = 30000 / 1001
BODY_MAP = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]


def data_url(endpoint, sequence, relative):
    return f"{endpoint}/webdata/dataset/{sequence}/{relative}"


def bounded_get(url, maximum, timeout=20, byte_range=True):
    headers = {"User-Agent": USER_AGENT}
    if byte_range:
        headers["Range"] = f"bytes=0-{maximum - 1}"
    began = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        data = response.read(maximum)
        metadata = {
            "url": url, "resolved_url": response.url, "status": response.status,
            "content_length": response.headers.get("Content-Length"),
            "content_range": response.headers.get("Content-Range"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
            "content_type": response.headers.get("Content-Type"),
            "received_bytes": len(data), "prefix_sha256": hashlib.sha256(data).hexdigest(),
            "request_ms": round((time.perf_counter() - began) * 1000, 2),
        }
    return data, metadata


def schema_summary(value, depth=0):
    if isinstance(value, dict):
        return {key: schema_summary(item, depth + 1) for key, item in value.items() if depth < 4}
    if isinstance(value, list):
        return {"type": "list", "length": len(value),
                "item": schema_summary(value[0], depth + 1) if value and depth < 4 else None}
    return type(value).__name__


def first_annotation(prefix):
    first = None
    try:
        with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|*") as archive:
            for ordinal, member in enumerate(archive):
                if ordinal > 200:
                    break
                if not member.isfile() or not member.name.endswith(".json") or member.size > 1024 * 1024:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                annotation = json.loads(stream.read().decode("utf-8"))
                result = {"member": member.name, "size": member.size, "schema": schema_summary(annotation)}
                if first is None:
                    first = result
                if annotation.get("bodies") or annotation.get("people"):
                    return result
    except (tarfile.TarError, OSError, EOFError, json.JSONDecodeError):
        if first is None:
            raise
    if first is None:
        raise ValueError("No complete annotation JSON appeared within the bounded tar prefix")
    return first


def probe(args):
    report = {"schema_version": 1, "mode": "probe", "sequence": args.sequence,
              "git_sha": os.environ.get("GITHUB_SHA"), "toolbox": TOOLBOX,
              "maximum_bytes_per_archive": args.probe_bytes,
              "video_bytes_requested": 4096, "attempts": [], "ok": False}
    selected = None
    calibration = None
    endpoints = [args.endpoint] if args.endpoint else ENDPOINTS
    for endpoint in endpoints:
        url = data_url(endpoint, args.sequence, f"calibration_{args.sequence}.json")
        try:
            data, metadata = bounded_get(url, args.probe_bytes, args.timeout, byte_range=False)
            calibration = json.loads(data.decode("utf-8"))
            cameras = calibration.get("cameras", [])
            if not cameras:
                raise ValueError("Calibration JSON does not contain cameras")
            report["calibration"] = {**metadata, "camera_count": len(cameras),
                                     "hd_camera_names": [cam.get("name") for cam in cameras if cam.get("panel") == 0],
                                     "schema": schema_summary(cameras[0])}
            selected = endpoint
            break
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            report["attempts"].append({"url": url, "error": str(error)})
    if selected:
        report["endpoint"] = selected
        report["resources"] = {}
        candidates = [("body", "hdPose3d_stage1_coco19.tar"), ("hands", "hdHand3d.tar")]
        candidates += [(f"video_{node:02d}", f"videos/hd_shared_crf20/hd_00_{node:02d}.mp4") for node in args.views]
        for name, relative in candidates:
            url = data_url(selected, args.sequence, relative)
            try:
                limit = args.probe_bytes if name in {"body", "hands"} else 4096
                data, metadata = bounded_get(url, limit, args.timeout)
                if name in {"body", "hands"}:
                    metadata["annotation"] = first_annotation(data)
                elif b"ftyp" not in data[:64]:
                    raise ValueError("Video response does not start with an MP4 ftyp box")
                report["resources"][name] = metadata
            except (urllib.error.URLError, TimeoutError, ValueError, OSError, tarfile.TarError) as error:
                report["resources"][name] = {"url": url, "error": str(error)}
        report["ok"] = all("error" not in item for item in report["resources"].values())
    report["limitations"] = ["Reachability and schema probe only; no accuracy measurement.",
                             "No original video, image, calibration or ground-truth coordinate arrays are exported.",
                             "No toe or instep ground truth is provided by this body19/hand dataset."]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


class LimitedStream:
    """Bound the HTTP bytes consumed while reading selected tar members."""

    def __init__(self, stream, maximum):
        self.stream = stream
        self.maximum = maximum
        self.count = 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        if size < 0:
            size = self.maximum - self.count + 1
        size = min(size, self.maximum - self.count + 1)
        chunk = self.stream.read(size)
        self.count += len(chunk)
        if self.count > self.maximum:
            raise ValueError(f"Annotation archive exceeds the {self.maximum}-byte read budget")
        self.digest.update(chunk)
        return chunk


def selected_annotations(url, kind, indices, maximum, timeout):
    """Read only JSON members; never extract a remotely supplied filesystem path."""
    wanted = set(indices)
    found = {}
    pattern = (r"(?:^|/)body3DScene_(\d{8})\.json$" if kind == "body"
               else r"(?:^|/)handRecon3D_hd(\d{8})\.json$")
    began = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        stream = LimitedStream(response, maximum)
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                match = re.search(pattern, member.name)
                if not member.isfile() or not match or int(match.group(1)) not in wanted:
                    continue
                if member.size > 4 * 1024 * 1024:
                    raise ValueError(f"Unexpectedly large annotation member: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not read annotation member: {member.name}")
                found[int(match.group(1))] = json.loads(extracted.read().decode("utf-8"))
                if len(found) == len(wanted):
                    break
        metadata = {"url": url, "resolved_url": response.url,
                    "content_length": response.headers.get("Content-Length"),
                    "received_bytes": stream.count, "read_prefix_sha256": stream.digest.hexdigest(),
                    "selected_annotation_frames": sorted(found),
                    "missing_annotation_frames": sorted(wanted - set(found)),
                    "request_ms": round((time.perf_counter() - began) * 1000, 2)}
    if not found:
        raise ValueError(f"No requested {kind} annotation frames found in {url}")
    return found, metadata


def choose_endpoint(args):
    attempts = []
    for endpoint in ([args.endpoint] if args.endpoint else ENDPOINTS):
        url = data_url(endpoint, args.sequence, f"calibration_{args.sequence}.json")
        try:
            data, metadata = bounded_get(url, 4 * 1024 * 1024, args.timeout, byte_range=False)
            calibration = json.loads(data.decode("utf-8"))
            available = {(camera["panel"], camera["node"]) for camera in calibration.get("cameras", [])}
            if not all((0, node) in available for node in args.views):
                raise ValueError("Requested HD cameras are missing from calibration")
            return endpoint, calibration, metadata, attempts
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            attempts.append({"url": url, "error": str(error)})
    raise RuntimeError(f"No usable calibration endpoint: {json.dumps(attempts)}")


def camera_contract(camera, np):
    rotation = np.asarray(camera["R"], dtype=float).reshape(3, 3)
    translation_cm = np.asarray(camera["t"], dtype=float).reshape(3)
    intrinsics = np.asarray(camera["K"], dtype=float).reshape(3, 3)
    distortion = np.asarray(camera["distCoef"], dtype=float).reshape(-1)
    if (not all(np.isfinite(value).all() for value in [rotation, translation_cm, intrinsics, distortion])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or abs(np.linalg.det(rotation) - 1) > 1e-4 or distortion.size != 5):
        raise ValueError("Invalid Panoptic camera matrix or distortion contract")
    width, height = camera["resolution"]
    return {"id": camera["name"], "width": int(width), "height": int(height),
            "fx": float(intrinsics[0, 0]), "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]), "cy": float(intrinsics[1, 2]),
            "skew": float(intrinsics[0, 1]), "distortion": distortion.tolist(),
            "calibrated": True, "intrinsicsCalibrated": True,
            "worldFromCamera": {"rotation": rotation.T.reshape(-1).tolist(),
                                "translation": (-rotation.T @ translation_cm / 100).tolist()}}


def project_cm(xyz_cm, camera, np):
    """Official Panoptic K(RX+t), Brown distortion [k1,k2,p1,p2,k3]."""
    rotation = np.asarray(camera["R"], dtype=float).reshape(3, 3)
    translation = np.asarray(camera["t"], dtype=float).reshape(3)
    intrinsics = np.asarray(camera["K"], dtype=float).reshape(3, 3)
    coordinates = xyz_cm @ rotation.T + translation
    front = np.isfinite(coordinates).all(axis=1) & (coordinates[:, 2] > 1e-6)
    normalized = np.full((len(coordinates), 2), np.nan)
    normalized[front] = coordinates[front, :2] / coordinates[front, 2, None]
    x, y = normalized.T
    k1, k2, p1, p2, k3 = np.asarray(camera["distCoef"], dtype=float).reshape(5)
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv = np.column_stack([intrinsics[0, 0] * xd + intrinsics[0, 1] * yd + intrinsics[0, 2],
                          intrinsics[1, 0] * xd + intrinsics[1, 1] * yd + intrinsics[1, 2]])
    return uv, front


def frame_truth(body_frame, hand_frame, person_id, np):
    xyz = np.zeros((133, 3), dtype=float)
    valid = np.zeros(133, dtype=bool)
    quality = {"body": "confidence > 0.1", "hands": {}}
    body = next((body for body in body_frame.get("bodies", []) if body["id"] == person_id), None)
    if body is not None:
        joints = np.asarray(body["joints19"], dtype=float).reshape(19, 4)[BODY_MAP]
        xyz[:17] = joints[:, :3]
        valid[:17] = np.isfinite(joints).all(axis=1) & (joints[:, 3] > .1)
    person = next((person for person in hand_frame.get("people", []) if person["id"] == person_id), None)
    if person:
        for name, begin in [("left_hand", 91), ("right_hand", 112)]:
            hand = person.get(name) or {}
            landmarks = np.asarray(hand.get("landmarks", []), dtype=float)
            if landmarks.size != 63:
                quality["hands"][name] = "absent or invalid landmark array"
                continue
            landmarks = landmarks.reshape(21, 3)
            accepted = np.isfinite(landmarks).all(axis=1) & np.any(landmarks != 0, axis=1)
            scores = np.asarray(hand.get("scores", []), dtype=float).reshape(-1)
            if scores.size == 21:
                accepted &= np.isfinite(scores) & (scores > .1)
                quality["hands"][name] = "finite nonzero 3D coordinates and score > 0.1"
            else:
                quality["hands"][name] = "finite nonzero 3D coordinates; no per-joint score available"
            xyz[begin:begin + 21] = landmarks
            valid[begin:begin + 21] = accepted
    return xyz, valid, quality


def oracle_body_box(xyz_cm, valid, camera, np):
    uv, front = project_cm(xyz_cm[:17], camera, np)
    width, height = camera["resolution"]
    accepted = valid[:17] & front & np.isfinite(uv).all(axis=1)
    accepted &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if accepted.sum() < 6:
        return None
    low, high = uv[accepted].min(axis=0), uv[accepted].max(axis=0)
    if np.any(high - low < 8):
        return None
    # pose_inference supplies the same 1.25 expansion as the browser pose crop.
    # No ground-truth hand/finger coordinates participate in this body box.
    return [float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1])]


def decode_video(args, endpoint, node):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to decode bounded Panoptic video excerpts")
    url = data_url(endpoint, args.sequence, f"videos/hd_shared_crf20/hd_00_{node:02d}.mp4")
    directory = CACHE / args.sequence / f"00_{node:02d}" / f"{args.start:08d}-{args.frames}"
    directory.mkdir(parents=True, exist_ok=True)
    indices = list(range(args.start, args.start + args.frames))
    files = [directory / f"{index:08d}.png" for index in indices]
    if not all(path.is_file() for path in files):
        # Start decoding at video frame zero; there is no approximate timestamp
        # seek. ffmpeg can use HTTP byte ranges for the MP4 index and stops after
        # the requested prefix. The complete source video is never retained.
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-rw_timeout", str(int(args.timeout * 1_000_000)), "-threads", "2", "-i", url,
                   "-an", "-sn", "-dn", "-vf", f"select=gte(n\\,{args.start})", "-vsync", "0",
                   "-frames:v", str(args.frames), "-threads", "2", "-start_number", str(args.start),
                   str(directory / "%08d.png")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.video_timeout)
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed for HD camera {node}: {completed.stderr[-4000:]}")
    if not all(path.is_file() for path in files):
        raise ValueError(f"The video did not contain all {args.frames} requested frames for camera {node}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return files, {"url": url, "view": f"00_{node:02d}", "first_frame": indices[0],
                   "last_frame": indices[-1], "decoded_frames": len(files),
                   "decoded_frame_hashes_sha256": digest.hexdigest(),
                   "decode": "ffmpeg from frame zero; selected consecutive indices; no timestamp seek",
                   "download": "HTTP decoder stops after requested frames; no complete video retained",
                   "images_exported": False}


def pixel_records(xy, scores, truth_cm, truth_valid, camera, body_box, np):
    uv, front = project_cm(truth_cm, camera, np)
    width, height = camera["resolution"]
    valid = truth_valid & front & np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    rows = []
    for group, ids in [("body", range(17)), ("hands", range(91, 133))]:
        for index in ids:
            if not valid[index]:
                continue
            error = float(np.linalg.norm(xy[index] - uv[index]))
            hand_size = None
            if index >= 91:
                begin = 91 if index < 112 else 112
                selected = uv[begin:begin + 21][valid[begin:begin + 21]]
                if len(selected) >= 2:
                    hand_size = float(np.ptp(selected, axis=0).max())
            rows.append({"group": group, "error": error, "score": float(scores[index]),
                         "person_scale": max(body_box[2:]), "hand_scale": hand_size})
    return rows


def aggregate_pixels(rows, np):
    result = {}
    for group in ["body", "hands"]:
        selected = [row for row in rows if row["group"] == group]
        errors = np.asarray([row["error"] for row in selected], dtype=float)
        hand_rows = [row for row in selected if row["hand_scale"] and row["hand_scale"] > 1]
        result[group] = {"joints": len(selected),
                         "mean_error_px": float(errors.mean()) if errors.size else None,
                         "median_error_px": float(np.median(errors)) if errors.size else None,
                         "p95_error_px": float(np.percentile(errors, 95)) if errors.size else None,
                         "pck_5px": float(np.mean(errors <= 5)) if errors.size else None,
                         "pck_10px": float(np.mean(errors <= 10)) if errors.size else None,
                         "pck_body_scale_005": float(np.mean([row["error"] <= .05 * row["person_scale"]
                                                              for row in selected])) if selected else None,
                         "pck_hand_scale_020": float(np.mean([row["error"] <= .2 * row["hand_scale"]
                                                              for row in hand_rows])) if hand_rows else None,
                         "score_at_least_025": float(np.mean([row["score"] >= .25 for row in selected])) if selected else None}
    return result


def evaluate(args):
    # Heavy dependencies are deliberately not imported by --probe.
    import cv2
    import numpy as np
    import onnxruntime as ort
    from evaluate import (MODELS, HAND_MODEL, CACHE as MODEL_CACHE, contract,
                          download, session, pose_inference, refine_hands)

    model_ids = args.models.split(",")
    if not model_ids or any(name not in MODELS for name in model_ids):
        raise ValueError(f"Use pinned model names: {sorted(MODELS)}")
    endpoint, calibration, calibration_metadata, attempts = choose_endpoint(args)
    indices = list(range(args.start, args.start + args.frames))
    source_cameras = {camera["name"]: camera for camera in calibration["cameras"]
                      if camera["panel"] == 0 and camera["node"] in args.views}
    cameras = {name: camera_contract(camera, np) for name, camera in source_cameras.items()}
    bodies, body_metadata = selected_annotations(data_url(endpoint, args.sequence, "hdPose3d_stage1_coco19.tar"),
                                                  "body", indices, args.annotation_max_bytes, args.timeout)
    hands, hand_metadata = selected_annotations(data_url(endpoint, args.sequence, "hdHand3d.tar"),
                                                "hands", indices, args.annotation_max_bytes, args.timeout)
    first_people = next((sorted(frame["bodies"], key=lambda body: body["id"])
                         for _, frame in sorted(bodies.items()) if frame.get("bodies")), [])
    if not first_people:
        raise ValueError("No body GT exists in the requested interval")
    person_id = args.person if args.person is not None else first_people[0]["id"]
    truth = {index: frame_truth(bodies.get(index, {}), hands.get(index, {}), person_id, np) for index in indices}
    body_gt_frames = sum(bool(item[1][:17].any()) for item in truth.values())
    hand_gt_frames = sum(bool(item[1][91:].any()) for item in truth.values())
    if body_gt_frames < min(30, args.frames) or hand_gt_frames < min(30, args.frames):
        raise ValueError(f"Insufficient consecutive GT in requested interval: body={body_gt_frames}, hands={hand_gt_frames}")
    decoded, videos = {}, []
    for node in args.views:
        decoded[f"00_{node:02d}"], metadata = decode_video(args, endpoint, node)
        videos.append(metadata)
    hand_session = session(download(HAND_MODEL["url"], MODEL_CACHE / "hand.onnx", HAND_MODEL["sha256"]))
    if hand_session.get_inputs()[0].shape != [1, 224, 224, 3]:
        raise ValueError(f"Unexpected hand input contract: {contract(hand_session)}")
    report = {"schema_version": 1, "mode": "real-panoptic-multiview", "git_sha": os.environ.get("GITHUB_SHA"),
              "backend": "CPUExecutionProvider", "runtime_version": ort.__version__,
              "scope": "Actual synchronized two-view video, oracle person identity and body ROI, metric 3D GT",
              "dataset": {"sequence": args.sequence, "endpoint": endpoint, "first_frame": indices[0],
                          "last_frame": indices[-1], "frame_count": len(indices), "fps": FPS,
                          "person_id": person_id, "person_selection": "Explicit id, or smallest id in first nonempty requested body frame",
                          "body_gt_frames": body_gt_frames, "hand_gt_frames": hand_gt_frames,
                          "body_mapping_coco17_to_panoptic19": BODY_MAP,
                          "body_gt_validity": "finite coordinates, reconstruction confidence > 0.1",
                          "hand_gt_validity": [item[2]["hands"] for item in truth.values()],
                          "units": "Panoptic centimeters converted to world meters before product geometry",
                          "calibration_source": calibration_metadata, "endpoint_attempts": attempts,
                          "body_annotations": body_metadata, "hand_annotations": hand_metadata, "videos": videos},
              "hand_model": {**HAND_MODEL, "contract": contract(hand_session)}, "models": {},
              "limitations": ["One short sequence and two views are a PoC measurement, not a population benchmark.",
                              "Person identity and person ROI are supplied from 3D body GT; person detection is not measured.",
                              "GT hand coordinates never determine a hand ROI; predicted palm landmarks do.",
                              "Panoptic 3D annotations are multi-camera reconstructions, not a marker-based motion-capture reference.",
                              "There is no toe or instep GT and no individual toe articulation measurement.",
                              "No GT per-view natural-occlusion label is assumed. Artificial missing observations are reported separately.",
                              "2D runtime uses ONNX Runtime CPU. Geometry and causal completion use the product JavaScript modules.",
                              "CPU inference latency does not establish browser WebGPU throughput.",
                              "Original frames, GT arrays and calibration arrays are kept out of public artifacts."]}
    internal = {"schemaVersion": 1, "sequence": args.sequence, "fps": FPS, "cameras": cameras, "conditions": []}
    public = {"schemaVersion": 1, "sequence": args.sequence, "isTemporalSequence": True,
              "fps": FPS, "cameraIds": list(cameras), "conditions": []}
    for model_id in model_ids:
        spec = MODELS[model_id]
        pose_session = session(download(spec["url"], MODEL_CACHE / f"{model_id}.onnx", spec["sha256"]))
        if pose_session.get_inputs()[0].shape != [1, 3, spec["input"][1], spec["input"][0]]:
            raise ValueError(f"Unexpected pose input contract: {contract(pose_session)}")
        collected = {"original": [], "hand_refined": []}
        all_frames = {"original": [], "hand_refined": []}
        timings, hand_timings, refinement_notes, skipped = [], [], [], []
        for ordinal, index in enumerate(indices):
            xyz_cm, valid, _ = truth[index]
            time_ms = index * 1000 / FPS
            gt = [{"id": int(joint_id), "xyz": (xyz_cm[joint_id] / 100).tolist(), "valid": True}
                  for joint_id in np.flatnonzero(valid)]
            frame_observations = {"original": [], "hand_refined": []}
            for source_id, camera in source_cameras.items():
                image = cv2.imread(str(decoded[source_id][ordinal]), cv2.IMREAD_COLOR)
                if image is None or list(image.shape[:2][::-1]) != camera["resolution"]:
                    raise ValueError(f"Decoded dimensions differ from calibration for {source_id}, frame {index}")
                box = oracle_body_box(xyz_cm, valid, camera, np)
                if box is None:
                    skipped.append({"frame": index, "sourceId": source_id, "reason": "No valid oracle body ROI"})
                    continue
                xy, scores, elapsed, _ = pose_inference(pose_session, image, {"bbox": box}, spec)
                rxy, rscores, hand_elapsed, notes = refine_hands(hand_session, image, xy, scores)
                timings.append(elapsed)
                hand_timings.append(hand_elapsed)
                refinement_notes.extend({"frame": index, "sourceId": source_id, **note} for note in notes)
                for name, coordinates, confidence in [("original", xy, scores), ("hand_refined", rxy, rscores)]:
                    collected[name].extend(pixel_records(coordinates, confidence, xyz_cm, valid, camera, box, np))
                    frame_observations[name].extend({"sourceId": source_id, "timeMs": time_ms, "id": int(joint_id),
                                                     "u": float(point[0]), "v": float(point[1]), "score": float(confidence[joint_id])}
                                                    for joint_id, point in enumerate(coordinates))
            for name in all_frames:
                all_frames[name].append({"index": index, "timeMs": time_ms,
                                        "observations": frame_observations[name], "groundTruth": gt})
            print(json.dumps({"panoptic_model": model_id, "frame": index, "completed": ordinal + 1,
                              "total": len(indices), "observations": len(frame_observations["original"])}), flush=True)
        if not timings:
            raise ValueError("No valid oracle person ROIs could be inferred")
        for name, frames in all_frames.items():
            condition_name = model_id if name == "original" else model_id + "+hand-roi"
            internal["conditions"].append({"name": condition_name, "frames": frames})
            public["conditions"].append({"name": condition_name, "frames": [
                {key: value for key, value in frame.items() if key != "groundTruth"} for frame in frames]})
        report["models"][model_id] = {**spec, "contract": contract(pose_session),
                                      "pixel_metrics": {name: aggregate_pixels(rows, np) for name, rows in collected.items()},
                                      "cpu_pose_ms": {"mean": float(np.mean(timings)), "p95": float(np.percentile(timings, 95))},
                                      "cpu_hand_ms_per_view": float(np.mean(hand_timings)),
                                      "refinement_attempted_hands": len(refinement_notes),
                                      "refinement_accepted_hands": sum(bool(note["accepted"]) for note in refinement_notes),
                                      "refinement_notes": refinement_notes, "skipped_person_views": skipped}
        del pose_session
    CACHE.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    input_path = CACHE / "evaluation-input.json"
    input_path.write_text(json.dumps(internal, allow_nan=False), encoding="utf-8")
    (args.out / "observations.json").write_text(json.dumps(public, allow_nan=False), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    geometry_path = args.out / "geometry.json"
    subprocess.run(["node", str(ROOT / "scripts" / "verify-geometry.mjs"), "--panoptic-input", str(input_path),
                    "--panoptic-out", str(geometry_path)], cwd=ROOT, check=True, timeout=120)
    report["geometry"] = json.loads(geometry_path.read_text(encoding="utf-8"))
    report["ok"] = True
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"panoptic_complete": True, "sequence": args.sequence, "frames": len(indices),
                      "body_gt_frames": body_gt_frames, "hand_gt_frames": hand_gt_frames,
                      "conditions": [condition["name"] for condition in internal["conditions"]]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--views", default="0,8", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--endpoint")
    parser.add_argument("--probe-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--out", type=Path, default=Path("artifacts/panoptic"))
    parser.add_argument("--models", default="rtmw-l")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--person", type=int)
    parser.add_argument("--annotation-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--video-timeout", type=float, default=180)
    args = parser.parse_args()
    if len(args.views) != 2 or len(set(args.views)) != 2 or any(not 0 <= node <= 30 for node in args.views):
        parser.error("--views must name two distinct HD cameras numbered 0..30")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.sequence):
        parser.error("Invalid sequence name")
    if not 30 <= args.frames <= 60 or args.start < 0:
        parser.error("Use 30..60 consecutive frames and a nonnegative first frame")
    if args.probe_bytes < 4096 or args.annotation_max_bytes < 4 * 1024 * 1024:
        parser.error("Download read budgets are too small")
    return probe(args) if args.probe else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
