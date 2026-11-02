"""HanCo tester: bounded download and real hand/multiview evaluation in Actions.

HanCo's 3D reference is a MANO fit, optionally manually inspected. It is not an
independent optical-marker measurement. RGB files already contain a hand crop.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".evaluation-cache" / "hanco"
SOURCE = "https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_tester.zip"
SOURCE_PAGE = "https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html"
SOURCE_CODE = "https://github.com/lmb-freiburg/contra-hand/blob/main/show_dataset.py"
FPS = 5


def tester_archive(timeout=60):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "HanCo_tester.zip"
    maximum = 96 * 1024 * 1024
    if not path.is_file():
        request = urllib.request.Request(SOURCE, headers={"User-Agent": "Wholebody-test-HanCo-research/1.0"})
        temporary = path.with_suffix(".partial")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > maximum:
                        raise ValueError("HanCo tester exceeds the 96 MiB download limit")
                    output.write(chunk)
            if not zipfile.is_zipfile(temporary):
                raise ValueError("HanCo tester response is not a ZIP archive")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return path, {"url": SOURCE, "bytes": path.stat().st_size, "sha256": digest}


def archive_index(archive, requested_sequence=None):
    images = {}
    pattern = re.compile(r"(?:^|/)rgb/(\d{4})/cam(\d)/(\d{8})\.jpg$")
    for name in archive.namelist():
        match = pattern.search(name)
        if match:
            sequence, camera, frame = (int(item) for item in match.groups())
            images[(sequence, camera, frame)] = name
    sequences = sorted({key[0] for key in images})
    if not sequences:
        raise ValueError("No HanCo RGB hand crops found in the tester")
    sequence = requested_sequence if requested_sequence is not None else (110 if 110 in sequences else sequences[0])
    if sequence not in sequences:
        raise ValueError(f"Sequence {sequence} is absent; available: {sequences}")
    metadata_names = [name for name in archive.namelist() if name == "meta.json" or name.endswith("/meta.json")]
    if len(metadata_names) != 1:
        raise ValueError("Expected one HanCo meta.json")
    metadata = json.loads(archive.read(metadata_names[0]))
    frames = sorted({frame for sid, _, frame in images if sid == sequence})
    views = sorted({camera for sid, camera, _ in images if sid == sequence})
    return images, sequence, frames, views, metadata


def find_member(archive, suffix):
    found = [name for name in archive.namelist() if name == suffix or name.endswith("/" + suffix)]
    if len(found) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix}; found {len(found)}")
    info = archive.getinfo(found[0])
    if info.file_size > 4 * 1024 * 1024:
        raise ValueError(f"Unreasonably large selected JSON: {suffix}")
    return found[0]


def metadata_flag(metadata, field, sequence, frame):
    try:
        return bool(metadata[field][sequence][frame])
    except (IndexError, KeyError, TypeError):
        return False


def schema(value, depth=0):
    if isinstance(value, dict):
        return {key: schema(item, depth + 1) for key, item in value.items()} if depth < 3 else {"type": "dict"}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "item": schema(value[0], depth + 1) if value and depth < 4 else None}
    return type(value).__name__


def probe(args):
    report = {"mode": "HanCo-tester-probe", "git_sha": os.environ.get("GITHUB_SHA"), "ok": False,
              "source_page": SOURCE_PAGE, "official_projection_example": SOURCE_CODE,
              "reference_kind": "MANO fit, with separate manually validated flag", "fps": FPS}
    try:
        path, source = tester_archive(args.timeout)
        report["source"] = source
        with zipfile.ZipFile(path) as archive:
            images, sequence, frames, views, metadata = archive_index(archive, args.sequence)
            fits = [frame for frame in frames if metadata_flag(metadata, "has_fit", sequence, frame)]
            validated = [frame for frame in frames if metadata_flag(metadata, "is_valid", sequence, frame)]
            report.update({"sequence": sequence, "frame_count": len(frames), "frames": frames,
                           "views": views, "fit_frames": fits, "manually_validated_frames": validated,
                           "metadata_schema": schema(metadata), "archive_file_count": len(archive.namelist())})
            first = fits[0] if fits else frames[0]
            calibration = json.loads(archive.read(find_member(archive, f"calib/{sequence:04d}/{first:08d}.json")))
            reference = json.loads(archive.read(find_member(archive, f"xyz/{sequence:04d}/{first:08d}.json")))
            report["calibration_schema"] = schema(calibration)
            report["reference_schema"] = schema(reference)
            if len(reference) != 21 or any(len(point) != 3 for point in reference):
                raise ValueError("Expected 21x3 HanCo world coordinates")
            # Scalar size diagnostics help verify documented units; reference
            # coordinate arrays and original images are not redistributed.
            report["reference_native_bone_lengths"] = {f"{a}-{b}": math.dist(reference[a], reference[b])
                                                         for a, b in [(0, 9), (5, 9), (9, 13)]}
            report["ok"] = bool(fits and validated and len(views) >= 2 and len(frames) >= args.frames)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


def camera_contract(calibration, view, np):
    matrix = np.asarray(calibration["M"][view], dtype=float).reshape(4, 4)
    intrinsics = np.asarray(calibration["K"][view], dtype=float).reshape(3, 3)
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    if (not np.isfinite(matrix).all() or not np.isfinite(intrinsics).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or abs(np.linalg.det(rotation) - 1) > 1e-4
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        raise ValueError(f"Unsupported HanCo calibration for camera {view}")
    # Match the official crop-projection example, which uses M and K directly
    # without applying the raw camera's distortion to the provided crop again.
    return {"id": f"cam{view}", "width": 224, "height": 224,
            "fx": float(intrinsics[0, 0]), "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]), "cy": float(intrinsics[1, 2]), "skew": float(intrinsics[0, 1]),
            "distortion": [0, 0, 0, 0, 0], "calibrated": True, "intrinsicsCalibrated": True,
            "worldFromCamera": {"rotation": rotation.T.reshape(-1).tolist(),
                                "translation": (-rotation.T @ translation).tolist()}}


def project_reference(xyz, calibration, view, np):
    matrix = np.asarray(calibration["M"][view], dtype=float).reshape(4, 4)
    intrinsics = np.asarray(calibration["K"][view], dtype=float).reshape(3, 3)
    camera_xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    homogeneous = camera_xyz @ intrinsics.T
    uv = np.full((21, 2), np.nan)
    front = np.isfinite(homogeneous).all(axis=1) & (camera_xyz[:, 2] > 1e-8)
    uv[front] = homogeneous[front, :2] / homogeneous[front, 2, None]
    return uv, front


def hand_inference(model, image, np, cv2):
    """Known hand crop with a fixed, GT-independent rotation search.

    All eight candidates use exactly the hand model's RGB/255 NHWC input. Only
    its image-conditioned presence selects orientation; no reference palm,
    world depth, hand orientation or 3D coordinate enters inference.
    """
    best = None
    milliseconds = 0.0
    for degrees in range(0, 360, 45):
        radians = math.radians(degrees)
        ex = np.array([math.cos(radians), math.sin(radians)])
        ey = np.array([-math.sin(radians), math.cos(radians)])
        origin = np.array([112., 112.]) - 112 * (ex + ey)
        affine = np.array([[ex[0], ey[0], origin[0]], [ex[1], ey[1], origin[1]]], dtype=np.float32)
        crop = cv2.warpAffine(image, affine, (224, 224), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        tensor = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255)
        began = time.perf_counter()
        values = model.run(None, {model.get_inputs()[0].name: tensor})
        milliseconds += (time.perf_counter() - began) * 1000
        output = dict(zip([item.name for item in model.get_outputs()], values))
        landmarks = np.asarray(output["Identity"]).reshape(21, 3)
        presence = float(np.asarray(output["Identity_1"]).reshape(-1)[0])
        if not np.isfinite(landmarks).all() or not math.isfinite(presence):
            raise ValueError("Hand model returned nonfinite output")
        xy = np.array([112., 112.]) + np.outer(landmarks[:, 0] - 112, ex) + np.outer(landmarks[:, 1] - 112, ey)
        candidate = {"xy": xy, "presence": float(np.clip(presence, 0, 1)), "angle_degrees": degrees}
        if best is None or candidate["presence"] > best["presence"]:
            best = candidate
    inside = (best["xy"][:, 0] >= 0) & (best["xy"][:, 0] < 224) & (best["xy"][:, 1] >= 0) & (best["xy"][:, 1] < 224)
    # A hand-level presence is not per-finger visibility. It is recorded as a
    # shared model confidence for geometric weighting, with visibility unknown.
    best["scores"] = np.where(inside & (best["presence"] >= .65), best["presence"], 0)
    best["inference_ms"] = milliseconds
    return best


def observations(prediction, source_id, time_ms):
    return [{"id": 112 + joint, "sourceId": source_id, "timeMs": time_ms,
             "u": float(point[0]), "v": float(point[1]), "score": float(prediction["scores"][joint]),
             "visibility": "unknown", "source": "known-hand-crop-image-inference"}
            for joint, point in enumerate(prediction["xy"])]


def summarize_errors(errors, np):
    values = np.asarray(errors, dtype=float)
    return {"joints": len(errors), "mean_error_px": float(values.mean()) if len(values) else None,
            "median_error_px": float(np.median(values)) if len(values) else None,
            "p95_error_px": float(np.percentile(values, 95)) if len(values) else None,
            "pck_5px": float(np.mean(values <= 5)) if len(values) else None,
            "pck_10px": float(np.mean(values <= 10)) if len(values) else None}


def evaluate(args):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from evaluate import CACHE as MODEL_CACHE, HAND_MODEL, contract, download, session

    path, source = tester_archive(args.timeout)
    hand = session(download(HAND_MODEL["url"], MODEL_CACHE / "hand.onnx", HAND_MODEL["sha256"]))
    if hand.get_inputs()[0].shape != [1, 224, 224, 3]:
        raise ValueError(f"Unexpected MediaPipe hand input contract: {contract(hand)}")
    with zipfile.ZipFile(path) as archive:
        images, sequence, available, views, metadata = archive_index(archive, args.sequence)
        if views != list(range(8)):
            raise ValueError(f"Expected eight synchronized HanCo views, found {views}")
        indices = list(range(args.start, args.start + args.frames))
        if any(frame not in available or any((sequence, view, frame) not in images for view in views) for frame in indices):
            raise ValueError("Requested consecutive frames are not present in all eight views")
        validated = [index for index, frame in enumerate(indices)
                     if metadata_flag(metadata, "has_fit", sequence, frame) and metadata_flag(metadata, "is_valid", sequence, frame)]
        eligible = [index for index in validated if index >= 2
                    and all(metadata_flag(metadata, "has_fit", sequence, indices[index - back]) for back in [1, 2])]
        if not eligible:
            raise ValueError("No manually validated reference frame with at least two preceding frames in the requested interval")
        drop_start = eligible[0]
        mask_indices = set(range(max(0, drop_start - 2), min(len(indices), drop_start + 3)))
        paired_views = [0, 3]
        condition_views = {"two_views": paired_views, "eight_views": views,
                           "two_views_one_image_masked": paired_views, "eight_views_one_image_masked": views}
        frames_by_condition = {name: [] for name in condition_views}
        point_errors = {name: [] for name in condition_views}
        mask_errors = {"unmodified_same_targets": [], "masked_targets": []}
        notes, timings = [], []
        native_bones = []
        rectangle = [67, 34, 157, 190]
        first_cameras = None
        for ordinal, frame in enumerate(indices):
            calibration = json.loads(archive.read(find_member(archive, f"calib/{sequence:04d}/{frame:08d}.json")))
            camera_map = {f"cam{view}": camera_contract(calibration, view, np) for view in views}
            if first_cameras is None:
                first_cameras = camera_map
            has_fit = metadata_flag(metadata, "has_fit", sequence, frame)
            manually_validated = ordinal in validated
            reference = None
            if has_fit:
                reference = np.asarray(json.loads(archive.read(find_member(archive, f"xyz/{sequence:04d}/{frame:08d}.json"))), dtype=float)
                if reference.shape != (21, 3) or not np.isfinite(reference).all():
                    raise ValueError(f"Invalid HanCo reference coordinates at frame {frame}")
                native_bones.append(float(np.linalg.norm(reference[9] - reference[0])))
            gt = [{"id": 112 + joint, "xyz": point.tolist(), "valid": True}
                  for joint, point in enumerate(reference)] if manually_validated else []
            time_ms = frame * 1000 / FPS
            normal_observations, masked_observations = {}, {}
            frame_errors, masked_frame_errors = {}, {}
            for view in views:
                compressed = archive.read(images[(sequence, view, frame)])
                image = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (224, 224, 3):
                    raise ValueError(f"Unexpected HanCo RGB shape at camera {view}, frame {frame}")
                predicted = hand_inference(hand, image, np, cv2)
                source_id = f"cam{view}"
                normal_observations[view] = observations(predicted, source_id, time_ms)
                masked_observations[view] = normal_observations[view]
                timings.append(predicted["inference_ms"])
                note = {"frame": frame, "view": view, "presence": predicted["presence"],
                        "rotation_degrees": predicted["angle_degrees"], "inference_ms": predicted["inference_ms"]}
                corrupted = None
                if view == paired_views[0] and ordinal in mask_indices:
                    masked_image = image.copy()
                    x0, y0, x1, y1 = rectangle
                    masked_image[y0:y1, x0:x1] = 128
                    corrupted = hand_inference(hand, masked_image, np, cv2)
                    masked_observations[view] = observations(corrupted, source_id, time_ms)
                    note["image_mask"] = {"rectangle_xyxy": rectangle, "presence": corrupted["presence"],
                                          "rotation_degrees": corrupted["angle_degrees"]}
                if manually_validated:
                    uv, front = project_reference(reference, calibration, view, np)
                    valid_2d = front & (uv[:, 0] >= 0) & (uv[:, 0] < 224) & (uv[:, 1] >= 0) & (uv[:, 1] < 224)
                    base_error = np.linalg.norm(predicted["xy"] - uv, axis=1)
                    frame_errors[view] = base_error[valid_2d].tolist()
                    masked_frame_errors[view] = frame_errors[view]
                    if corrupted:
                        mask_error = np.linalg.norm(corrupted["xy"] - uv, axis=1)
                        masked_frame_errors[view] = mask_error[valid_2d].tolist()
                        affected = valid_2d & (uv[:, 0] >= rectangle[0]) & (uv[:, 0] < rectangle[2])
                        affected &= (uv[:, 1] >= rectangle[1]) & (uv[:, 1] < rectangle[3])
                        mask_errors["unmodified_same_targets"].extend(base_error[affected].tolist())
                        mask_errors["masked_targets"].extend(mask_error[affected].tolist())
                notes.append(note)
            for name, selected_views in condition_views.items():
                masked = name.endswith("masked")
                chosen = masked_observations if masked else normal_observations
                errors = masked_frame_errors if masked else frame_errors
                frames_by_condition[name].append({"index": frame, "timeMs": time_ms, "cameras": camera_map,
                                                   "observations": [point for view in selected_views for point in chosen[view]],
                                                   "groundTruth": gt, "referenceManuallyValidated": manually_validated})
                point_errors[name].extend(error for view in selected_views for error in errors.get(view, []))
            print(json.dumps({"hanco_frame": frame, "completed": ordinal + 1, "total": len(indices),
                              "manually_validated_reference": manually_validated}), flush=True)
    if not native_bones or not .02 < float(np.median(native_bones)) < .2:
        raise ValueError("Reference sizes disagree with the inherited metre convention; no automatic rescaling is allowed")
    internal = {"schemaVersion": 1, "dataset": "HanCo Tester", "sequence": f"{sequence:04d}", "fps": FPS,
                "referenceKind": "MANO-fit, manually validated frames only", "coordinateUnit": "meter",
                "metricGroups": {"hands": list(range(112, 133))}, "dropJointIds": list(range(112, 133)),
                "dropFrameCount": 1, "dropStartIndex": drop_start,
                "dropScenarioName": "all_hand_views_observation_drop",
                "dropSelection": "First manually validated frame with two preceding fitted frames, selected before inference",
                "provenance": {"source": source, "sourcePage": SOURCE_PAGE,
                               "projectionSource": SOURCE_CODE,
                               "coordinateConversion": "No conversion: native coordinates treated as metres under the inherited FreiHAND/MANO convention.",
                               "unitStatus": "Inference from the shared convention, not a direct HanCo statement of SI units.",
                               "unitSource": "https://github.com/lmb-freiburg/freihand/blob/master/eval.py",
                               "reference": "Only manually inspected MANO fits are scored; no independent optical-marker reference.",
                               "input": "Dataset-provided hand crops with eight fixed rotation candidates, selected by model presence."},
                "scope": "Known-crop hand inference followed by the product triangulation and causal tracker; two-view comparison and eight-view upper-bound diagnostic.",
                "limitations": ["Known hand crop plus eight-direction search does not evaluate the product whole-body-to-hand ROI pipeline.",
                                "Eight-view reconstruction is a diagnostic; the published app accepts two views.",
                                "Training-data overlap is unknown; this does not establish unseen-data generalization.",
                                "Reported millimetres express agreement with MANO fits under the inherited metre convention, not independently measured physical accuracy."],
                "cameras": first_cameras,
                "conditions": [{"name": name, "frames": frames} for name, frames in frames_by_condition.items()]}
    public = {"schemaVersion": 1, "dataset": "HanCo Tester", "sequence": f"{sequence:04d}",
              "fps": FPS, "isTemporalSequence": True,
              "conditions": [{"name": name, "frames": [{key: value for key, value in frame.items()
                                                          if key not in {"groundTruth", "cameras"}}
                                                         for frame in frames]} for name, frames in frames_by_condition.items()]}
    report = {"schema_version": 1, "mode": "real-cropped-hand-multiview", "git_sha": os.environ.get("GITHUB_SHA"),
              "ok": False, "source": source, "source_page": SOURCE_PAGE, "projection_source": SOURCE_CODE,
              "dataset": {"sequence": sequence, "frames": indices, "fps": FPS, "views": views,
                          "paired_views": paired_views, "manually_validated_frames": [indices[index] for index in validated],
                          "reference_kind": "Manually inspected MANO fits; not independent optical-marker ground truth",
                          "crop": "Dataset-provided 224x224 hand crops; no full-image person/hand detection evaluated",
                          "units": "Metres inferred from the shared FreiHAND/MANO convention; no bone-length alignment",
                          "unit_source": "https://github.com/lmb-freiburg/freihand/blob/master/eval.py",
                          "native_wrist_to_middle_mcp_median": float(np.median(native_bones)),
                          "projection": "Frame-specific K times M_world_to_camera; official crop example applies no extra distortion"},
              "model": {**HAND_MODEL, "contract": contract(hand)}, "backend": "CPUExecutionProvider",
              "runtime_version": ort.__version__, "rotation_candidates_degrees": list(range(0, 360, 45)),
              "rotation_selection": "Maximum model hand-presence, independent of reference coordinates",
              "cpu_ms_per_crop_eight_candidates": {"mean": float(np.mean(timings)), "p95": float(np.percentile(timings, 95))},
              "pixel_metrics": {name: summarize_errors(values, np) for name, values in point_errors.items()},
              "one_view_image_mask": {"camera": "cam0", "frames": [indices[index] for index in sorted(mask_indices)],
                                      "rectangle_xyxy": rectangle, "fill_value": 128,
                                      "same_target_pixel_metrics": {name: summarize_errors(values, np) for name, values in mask_errors.items()}},
              "artificial_all_view_gap": {"frame": indices[drop_start], "duration_ms": 1000 / FPS,
                                          "selection": internal["dropSelection"]}, "inference_notes": notes,
              "limitations": ["One tester sequence is not a held-out population benchmark; model training overlap is unknown.",
                              "Only manually validated reference frames count toward errors; all frames supply causal history.",
                              "Eight-way rotation search on known hand crops is a component evaluation, not the whole-body ROI pipeline.",
                              "Eight-view reconstruction is an upper-bound diagnostic; the published app accepts two views.",
                              "MANO-fit agreement is not independently measured physical finger accuracy.",
                              "Presence is a shared hand-level model score, not per-finger visibility ground truth.",
                              "Fixed gray rectangles and omitted observations are artificial corruption, not labeled natural occlusion.",
                              "No body, instep or toe reference is included; browser throughput is not measured.",
                              "Original images, references and calibration arrays are excluded from public artifacts."]}
    args.out.mkdir(parents=True, exist_ok=True)
    input_path = CACHE / "evaluation-input.json"
    input_path.write_text(json.dumps(internal, allow_nan=False), encoding="utf-8")
    (args.out / "observations.json").write_text(json.dumps(public, allow_nan=False), encoding="utf-8")
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    geometry_path = args.out / "geometry.json"
    subprocess.run(["node", str(ROOT / "scripts" / "verify-geometry.mjs"), "--sequence-input", str(input_path),
                    "--sequence-out", str(geometry_path)], cwd=ROOT, check=True, timeout=120)
    report["geometry"] = json.loads(geometry_path.read_text(encoding="utf-8"))
    report["ok"] = True
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"hanco_complete": True, "sequence": sequence, "frames": len(indices),
                      "manually_validated_frames": len(validated), "conditions": list(condition_views)}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "hanco")
    args = parser.parse_args()
    if not 30 <= args.frames <= 60 or args.start < 0:
        parser.error("Use 30..60 consecutive frames and a nonnegative first frame")
    return probe(args) if args.probe else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
