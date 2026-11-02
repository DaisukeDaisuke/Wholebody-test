"""Fetch bounded FreeMoCap/Qualisys trajectory pairs for GitHub Actions.

This evaluates the product's causal 3D tracker, not its image inference. The
released trajectories have already been aligned and postprocessed upstream.
Only provenance belongs in CI artifacts; coordinate inputs stay in the cache.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request
import zipfile


RECORD_ID = 23041223
RECORD_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
SOURCE_COMMIT = "39207e112a9fb79ee71e15f061bd98b6ac4cbdce"
SOURCE_BASE = f"https://raw.githubusercontent.com/freemocap/validation/{SOURCE_COMMIT}"
SOURCE_WEB = f"https://github.com/freemocap/validation/blob/{SOURCE_COMMIT}"
USER_AGENT = "Wholebody-test-FreeMoCap-evaluation/1.0"
MAX_PARQUET_BYTES = 256 * 1024 * 1024
JOINT_MAP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
LIMITATIONS = [
    "No video or image inference is run. The input is published aligned RTMPose 3D data.",
    "Qualisys is an independently captured marker reference, with derived anatomical joint centres; it is not exact joint ground truth.",
    "Upstream alignment uses full-sequence rigid registration and temporal synchronization; trajectories are interpolated and filtered before release.",
    "Only the product tracker is causal in this experiment. The complete upstream processing is not an online causal pipeline.",
    "The Qualisys-to-Qualisys condition measures completion of a reference trajectory after artificial observation loss, not markerless accuracy.",
    "The selected 12 body landmarks contain no finger joints, foot instep, or toe articulation.",
    "A constant confidence of 1 marks an available input coordinate; the archive does not supply calibrated joint confidence for this comparison.",
    "Fixed millimetre-to-metre conversion and one common rigid axis rotation are applied. There is no per-frame fitting or scale normalization.",
]


def read_url(url, maximum=8 * 1024 * 1024, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read(maximum + 1)
        if len(content) > maximum:
            raise ValueError(f"Response exceeds {maximum} bytes: {url}")
        return content


def file_url(info):
    links = info.get("links", {})
    for name in ("download", "content", "self"):
        if name in links:
            return links[name]
    raise ValueError(f"No download link for {info.get('key')}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(info, destination, expected, maximum):
    if int(info.get("size", 0)) > maximum:
        raise ValueError(f"Archive is larger than the {maximum}-byte download budget")
    if destination.exists() and sha256_file(destination) == expected:
        return {"bytes": destination.stat().st_size, "sha256": expected, "cache_hit": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    digest = hashlib.sha256()
    count = 0
    request = urllib.request.Request(file_url(info), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > maximum:
                    raise ValueError("Archive exceeds the configured download budget")
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bytes": count, "sha256": expected, "cache_hit": False}


def parse_manifest(content):
    checksums = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        checksum, filename = line.strip().split(maxsplit=1)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
            raise ValueError("Invalid SHA256SUMS.txt entry")
        checksums[filename.lstrip("*")] = checksum.lower()
    return checksums


def member_for(archive, subject, trial, tracker):
    suffix = f"{subject}/{trial}/{tracker}/aligned_3d_data/freemocap_data_by_frame.parquet"
    members = [entry for entry in archive.infolist()
               if not entry.is_dir() and (entry.filename == suffix or entry.filename.endswith("/" + suffix))]
    if len(members) != 1:
        available = [entry.filename for entry in archive.infolist()
                     if entry.filename.endswith("freemocap_data_by_frame.parquet")]
        raise ValueError(f"Expected one {suffix}; found {len(members)}. Canonical members: {available}")
    member = members[0]
    if member.file_size > MAX_PARQUET_BYTES or ".." in PurePosixPath(member.filename).parts:
        raise ValueError(f"Unsupported archive member: {member.filename}")
    return member


def load_points(archive, member, tracker, wanted_frames):
    import pandas as pd

    content = archive.read(member)
    table = pd.read_parquet(io.BytesIO(content))
    required = {"frame", "keypoint", "x", "y", "z", "model", "trajectory"}
    if not required.issubset(table.columns):
        raise ValueError(f"Unexpected Parquet columns: {list(table.columns)}")
    # Selecting 3d_xyz is deliberate: rigid_3d_xyz is a second, altered trajectory.
    body = table[(table["model"] == f"{tracker}.body") & (table["trajectory"] == "3d_xyz")]
    if body.empty:
        raise ValueError(f"No {tracker}.body / 3d_xyz rows in {member.filename}")
    names = set(body["keypoint"].unique())
    if not set(JOINT_MAP).issubset(names):
        raise ValueError(f"Missing body landmarks: {sorted(set(JOINT_MAP) - names)}")
    body = body[body["keypoint"].isin(JOINT_MAP) & body["frame"].isin(wanted_frames)]
    if body.duplicated(subset=["frame", "keypoint"]).any():
        raise ValueError("Duplicate frame/landmark values would require an undocumented aggregation")
    frames = {index: [] for index in wanted_frames}
    nonfinite = 0
    for row in body.itertuples(index=False):
        values = [float(row.x), float(row.y), float(row.z)]
        if not all(math.isfinite(value) for value in values):
            nonfinite += 1
            continue
        # Source: X lateral, Y anterior, Z up. Proper rotation into Y-up world.
        xyz = [values[0] * 0.001, values[2] * 0.001, -values[1] * 0.001]
        frames[int(row.frame)].append({"id": JOINT_MAP[row.keypoint], "xyz": xyz})
    for points in frames.values():
        points.sort(key=lambda joint: joint["id"])
    metadata = {
        "member": member.filename, "parquet_sha256": hashlib.sha256(content).hexdigest(),
        "parquet_bytes": len(content), "selected_finite_points": sum(map(len, frames.values())),
        "selected_nonfinite_points": nonfinite, "selected_frames": len(frames),
        "trajectory": "3d_xyz", "model": f"{tracker}.body",
    }
    return frames, metadata


def build_inputs(args, report):
    import yaml

    record_bytes = read_url(RECORD_URL)
    record = json.loads(record_bytes)
    files = {item["key"]: item for item in record["files"]}
    if "SHA256SUMS.txt" not in files:
        raise ValueError("Published dataset has no SHA256SUMS.txt")
    manifest_bytes = read_url(file_url(files["SHA256SUMS.txt"]))
    manifest = parse_manifest(manifest_bytes)
    report["record"] = {
        "url": RECORD_URL, "doi": "10.5281/zenodo.23041223",
        "title": record.get("metadata", {}).get("title"),
        "license": record.get("metadata", {}).get("license"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for number in args.subjects:
        subject = f"sub-{number:03d}"
        archive_name = f"freemocap_validation_dataset_v1.0_{subject}.zip"
        if archive_name not in files or archive_name not in manifest:
            raise ValueError(f"Archive or checksum missing: {archive_name}")
        config_path = f"configs/{subject}/{args.trial}.yaml"
        config_bytes = read_url(f"{SOURCE_BASE}/{config_path}")
        config = yaml.safe_load(config_bytes)
        if config["trial_path"] != f"{subject}/{args.trial}":
            raise ValueError("Trial configuration path disagrees with requested series")
        project = config["ProjectConfig"]
        fps = float(project["sampling_rate"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid published sampling rate")
        frame_groups = {}
        for name, condition in project["conditions"].items():
            start, end = map(int, condition["frames"])
            stop = min(end, start + args.frames_per_condition)
            if stop - start < 60:
                raise ValueError(f"Condition {name} contains fewer than 60 requested frames")
            frame_groups[name] = list(range(start, stop))
        wanted = sorted({index for indices in frame_groups.values() for index in indices})
        archive_path = args.cache_dir / archive_name
        archive_info = download_verified(files[archive_name], archive_path, manifest[archive_name], args.max_archive_bytes)
        archive_info.update({"archive": archive_name, "source_url": file_url(files[archive_name]),
                             "config_url": f"{SOURCE_WEB}/{config_path}",
                             "config_sha256": hashlib.sha256(config_bytes).hexdigest()})
        report["archives"].append(archive_info)
        with zipfile.ZipFile(archive_path) as archive:
            observed, observed_info = load_points(archive, member_for(archive, subject, args.trial, "rtmpose"), "rtmpose", wanted)
            reference, reference_info = load_points(archive, member_for(archive, subject, args.trial, "qualisys"), "qualisys", wanted)
        archive_info["trajectories"] = [observed_info, reference_info]
        for name, indices in frame_groups.items():
            sequence = f"{subject}/{args.trial}/{name}"
            conditions = []
            for label, inputs in [("rtmpose-to-qualisys", observed), ("qualisys-to-qualisys", reference)]:
                frames = [{
                    "index": index, "timeMs": index / fps * 1000,
                    "inputJoints": [{**point, "confidence": 1} for point in inputs[index]],
                    "groundTruth": [{**point, "valid": True} for point in reference[index]],
                } for index in indices]
                conditions.append({"name": label, "frames": frames})
            valid_reference_points = sum(len(reference[index]) for index in indices)
            if valid_reference_points < len(indices) * len(JOINT_MAP) * 0.8:
                raise ValueError(f"Less than 80% reference coverage in predetermined condition {sequence}")
            provenance = {
                "doi": "10.5281/zenodo.23041223", "license": "CC-BY-4.0",
                "citation": "FreeMoCap Validation Dataset: Multi-Camera Markerless Motion Capture of Treadmill Gait and Standing Balance. Version 1.0.",
                "dataset_url": "https://doi.org/10.5281/zenodo.23041223",
                "archive_sha256": manifest[archive_name], "archive": archive_name,
                "source_commit": SOURCE_COMMIT, "config_url": f"{SOURCE_WEB}/{config_path}",
                "source_unit": "mm", "coordinate_unit": "m", "unit_scale": 0.001,
                "source_to_output_rotation": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                "unit_evidence": f"{SOURCE_WEB}/plotting/gait/joint_trajectories/joint_trajectories_xyz_plots.py",
                "timebase": "Published per-trial 30 Hz sampling grid; original camera capture timestamps are not included in the coordinate input.",
                "frame_selection": "First contiguous frames of each predefined speed interval; no error-based selection.",
                "frame_start": indices[0], "frame_end_inclusive": indices[-1],
                "joint_map": JOINT_MAP, "upstream_preprocessing_is_causal": False,
                "limitations": LIMITATIONS,
            }
            payload = {
                "schemaVersion": 1, "dataset": "FreeMoCap validation", "sequence": sequence,
                "fps": fps, "coordinateUnit": "m", "inputMode": "3d-points",
                "referenceKind": "aligned-independent-Qualisys", "provenance": provenance,
                "metricGroups": {"body": list(JOINT_MAP.values()), "arms": [7, 8, 9, 10],
                                 "elbows": [7, 8], "wrists": [9, 10]},
                "dropJointIds": [7, 8, 9, 10], "dropFrameCount": 8,
                "dropScenarioName": "both_arms_observation_drop", "conditions": conditions,
            }
            destination = args.out_dir / f"{subject}_{args.trial}_{name}.json"
            destination.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            report["input_files"].append(str(destination))
            report["sequences"].append({"sequence": sequence, "frames_per_condition": len(indices),
                                         "fps": fps, "conditions": [entry["name"] for entry in conditions],
                                         "finite_reference_points": valid_reference_points})
            print(f"Prepared {sequence}: {len(indices)} frames x 2 conditions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="1", help="Comma-separated published subject numbers (1..6)")
    parser.add_argument("--trial", default="task-treadmill_trial-01", choices=["task-treadmill_trial-01", "task-treadmill_trial-02"])
    parser.add_argument("--frames-per-condition", type=int, default=300)
    parser.add_argument("--cache-dir", type=Path, default=Path(".evaluation-cache/freemocap-downloads"))
    parser.add_argument("--out-dir", type=Path, default=Path(".evaluation-cache/freemocap"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/freemocap-fetch.json"))
    parser.add_argument("--max-archive-bytes", type=int, default=600 * 1024 * 1024)
    args = parser.parse_args()
    args.subjects = sorted({int(value.strip()) for value in args.subjects.split(",") if value.strip()})
    if not args.subjects or any(number < 1 or number > 6 for number in args.subjects):
        parser.error("--subjects must select published subject numbers 1..6")
    if args.frames_per_condition < 60 or args.frames_per_condition > 1800:
        parser.error("--frames-per-condition must be between 60 and 1800")
    report = {"schema_version": 1, "ok": False, "dataset": "FreeMoCap validation v1.0",
              "git_sha": os.environ.get("GITHUB_SHA"), "source_commit": SOURCE_COMMIT,
              "input_files": [], "archives": [], "sequences": [], "limitations": LIMITATIONS}
    try:
        build_inputs(args, report)
        report["ok"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"FreeMoCap acquisition failed: {error}", file=sys.stderr, flush=True)
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
