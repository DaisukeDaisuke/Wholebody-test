"""Publish an explicit public-model allowlist from GitHub Actions to Hugging Face.

No directory upload, wildcard, media input, frame, feature cache, or pose trace
is accepted. HF_TOKEN is used only by HfApi; anonymous verification uses urllib.
SDK diagnostics are suppressed so signed redirects and credentials cannot enter
Actions logs. Failure reports contain a fixed stage/code and HTTP status only.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
from urllib.parse import urlsplit


FACE_URL = ("https://raw.githubusercontent.com/cunjian/pytorch_face_landmark/"
            "f575be168a24af6f4807c852173fdfedf6d2c67d/onnx/pfld.onnx")
FACE_SHA256 = "7d7bbd5c6a1d9272e58d9773898284a1905d872eba9a662df9b5f20f1ba6f83e"
FACE_BYTES = 2987921
STAGE = "initialization"


class PublicationError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(root, filename):
    path = root / filename
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise PublicationError("unexpected_manifest_path")
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024:
        raise PublicationError("oversized_manifest")
    return json.loads(data)


def checked_model(root, filename, specification):
    path = root / filename
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise PublicationError("unexpected_model_path")
    expected_hash = specification.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise PublicationError("missing_model_sha256")
    data = path.read_bytes()
    if not 1_000 < len(data) < 64 * 1024 * 1024:
        raise PublicationError("unexpected_model_size")
    if len(data) != specification.get("bytes") or hashlib.sha256(data).hexdigest() != expected_hash:
        raise PublicationError("model_checksum_mismatch")
    return data


def public_download(url, expected_bytes, expected_sha256, keep=False):
    # This code never consults the Hub SDK or attaches an Authorization header.
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vrc-compact-public-check/1.0"})
            digest, count, chunks = hashlib.sha256(), 0, []
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise PublicationError("public_download_status")
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > expected_bytes:
                        raise PublicationError("public_download_size")
                    digest.update(chunk)
                    if keep:
                        chunks.append(chunk)
            if count != expected_bytes or digest.hexdigest() != expected_sha256:
                raise PublicationError("public_download_checksum")
            return b"".join(chunks) if keep else {"bytes": count, "sha256": digest.hexdigest()}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def body_metrics(report):
    if report.get("status") != "completed" or report.get("data", {}).get("private_videos_used") is not False:
        raise PublicationError("body_report_not_public_completed_evaluation")
    fields = ("schema", "status", "run_url", "commit", "hardware", "data", "training", "selection",
              "teacher133", "previous_compact59", "sliced59_before_training", "fast59", "compact59", "acceptance")
    result = {key: copy.deepcopy(report[key]) for key in fields if key in report}
    if "reconstruction_reference" in report:
        result["reconstruction_reference"] = {
            key: value for key, value in report["reconstruction_reference"].items() if key != "per_image"}
    return result


def parts_metrics(report):
    if report.get("provenance", {}).get("privateVideoUsed") is not False:
        raise PublicationError("parts_report_not_public_evaluation")
    result = {"schemaVersion": report.get("schemaVersion", 1), "provenance": report["provenance"]}
    fields = {
        "foot": ("validation", "heldout", "onnxMaxAbsoluteProbabilityError", "inference", "dataset", "limits"),
        "face": ("dataset", "interocularNME", "featureRatioMAE", "inference", "limits"),
    }
    for part, keys in fields.items():
        source = report[part]
        result[part] = {key: copy.deepcopy(source[key]) for key in keys if key in source}
        result[part]["modelId"] = source.get("model", {}).get("id")
    # In particular, do not copy face.cases, faceBox, image identifiers, frames,
    # landmarks, embeddings, videos, or any browser observation traces.
    return result


def public_values(value, token):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"hf_token", "access_token", "authorization", "password", "secret"}:
                raise PublicationError("credential_field_in_public_metadata")
            public_values(item, token)
    elif isinstance(value, list):
        for item in value:
            public_values(item, token)
    elif isinstance(value, str):
        if token in value or re.search(r"\bhf_[A-Za-z0-9]{20,}\b", value):
            raise PublicationError("credential_value_in_public_metadata")
        if value.startswith(("http://", "https://")):
            parsed = urlsplit(value)
            if parsed.query or parsed.username or parsed.password:
                raise PublicationError("credential_capable_url_in_public_metadata")


def fast_is_eligible(fast, report):
    eligible = report.get("acceptance", {}).get("production_eligible")
    if eligible is None:
        eligible = fast.get("production_eligible", report["selection"]["validation_gates_passed"])
    return eligible is True and fast.get("production_eligible", True) is True and not report["data"].get("smoke_test_only", False)


def model_card(fast, standard, fast_report, standard_report, parts_report, repo_id):
    current = fast_report["fast59"]
    previous = fast_report["previous_compact59"]
    teacher = fast_report["teacher133"]
    data = fast_report["data"]
    foot, face = parts_report["foot"], parts_report["face"]
    rows = []
    for label, record, accuracy_key in (("Original 133, selected 59 joints", teacher, "selected59_accuracy"),
                                        ("Previous Compact59", previous, "accuracy"),
                                        (fast["id"], current, "accuracy")):
        rows.append(f"| {label} | {record['parameters']:,} | {record['multiply_accumulates']:,} | "
                    f"{record['latency']['median_ms']:.3f} | "
                    f"{100 * record[accuracy_key]['all']['labelled']['pck_005']:.3f}% |")
    part_rows = []
    for group in ("body", "hands", "feet", "face"):
        before = previous["accuracy"][group]["labelled"]
        after = current["accuracy"][group]["labelled"]
        part_rows.append(f"| {group} | {after['points']} | {100 * before['pck_005']:.3f}% | "
                         f"{100 * after['pck_005']:.3f}% | {after['mean_error_input_px']:.3f} |")
    recommended = "fast" if fast_is_eligible(fast, fast_report) else "standard"
    default_model = fast if recommended == "fast" else standard
    shape = default_model["input"]["shape"]
    training_environment = fast_report.get("training", {}).get("runtime", {}).get(
        "execution_environment", "actions" if fast_report.get("run_url") else "not recorded")
    cpu_environment = current["latency"].get("execution_environment", training_environment)
    run_reference = fast_report.get("run_url") or "No Actions run URL for this training execution; see the Colab bundle receipt."
    source_revision = fast_report.get("commit") or "not recorded"
    return f"""---
language:
- en
- ja
license: other
license_name: mixed-model-and-training-data-terms
license_link: https://huggingface.co/{repo_id}/blob/main/README.md#licenses-and-provenance
tags:
- onnx
- pose-estimation
- vrchat
- knowledge-distillation
---

# VRC Compact59 / Fast59

Native 59-landmark 2D pose students, a foot-view classifier, and a pinned PFLD68
face landmark model. The default body model is **{default_model['id']}**, with RGB
float32 input `{shape}`. Recommended profile: **{recommended}**. A fast candidate
that fails validation or the fixed final-test acceptance gate remains an experiment and
does not replace the standard model by default. Architecture and training
changes are recorded in the manifests and measured results.

## Files and inference contract

| File | Purpose | SHA-256 |
|---|---|---|
| `body/fast59.onnx` | Selected lower-compute body student | `{fast['sha256']}` |
| `body/compact59.onnx` | Previous 192x256 standard model | `{standard['sha256']}` |
| `parts/foot-view.onnx` | sole / dorsum / side on a provided foot crop | See `models.json` |
| `parts/pfld68.onnx` | Pinned upstream PFLD68 face landmark weights | `{FACE_SHA256}` |

Read `models.json` for input names, normalization, crop padding, output names,
keypoint order, class order and checksums. Every downloadable model URL in that
manifest includes an immutable Hub commit SHA. `body` selects the recommended model;
`bodies.fast` and `bodies.standard` make the implementations interchangeable.
The two SimCC axes use a split ratio of 2. Convert the argmax coordinates to the
provided padded person crop using `decodeWH` when present, otherwise the actual
input width and height.

## Measured public-image result

Fast59 uses {data['train_images']} public training images, {data['validation_images']}
validation images and **{data['heldout_images']} final test images**. Image IDs are
disjoint; validation/test also exclude images used in the previous Compact59
experiment. Checkpoint and resolution selection were fixed using validation
before the final test was opened. Validation acceptance passed:
**{fast_report['selection']['validation_gates_passed']}**. Selected checkpoint step:
**{fast_report['selection']['selected_step']}**. A failed gate is not a successful
accuracy-preservation claim. Final production eligibility is **{fast_is_eligible(fast, fast_report)}**.
The final test can reject deployment of the fixed candidate; it cannot choose a
different candidate or checkpoint.

| Model | Parameters | MACs | CPU median ms | Test PCK@0.05 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

All rows above use the same fresh test split in the Fast59 run. CPU timings are
FP32 ONNX Runtime CPUExecutionProvider measurements, exclude decoding/rendering,
and do not measure user-device WebGPU utilization, GPU power, or VRChat latency.
Their execution environment is **{cpu_environment}**. Optional PyTorch CUDA event
timings are separate fields in the evidence and are not CPU or browser timings.
PCK uses the same canonical 192x256 crop for every resolution: 0.05 means 12.8
canonical pixels. Ground-truth person boxes are supplied. This is neither full
COCO AP nor end-to-end person detection accuracy.

| Part | Labeled test points | Previous PCK@0.05 | Fast PCK@0.05 | Fast mean canonical px |
|---|---:|---:|---:|---:|
{chr(10).join(part_rows)}

The foot model's separate public holdout contains {foot['heldout']['examples']}
known-foot crops, with accuracy {100 * foot['heldout']['accuracy']:.2f}%.
FootNet lacks subject identifiers; this does not establish unseen-person or
motion-video accuracy. The classifier does not detect whether an arbitrary ROI
contains a foot.

The face check uses {face['dataset']['faces']} public face crops. Its interocular
normalized landmark error is {face['interocularNME']:.6f}. Blink and mouth values
are geometric ratios, not separately trained state labels; cheek width does not
establish cheek-puff recognition. No private user media is used in these results.

Aggregate evidence is in `evidence/fast59-metrics.json`,
`evidence/compact59-metrics.json`, and `evidence/parts-metrics.json`.
Per-image face cases, boxes, observation traces and feature caches are omitted.

Fast59 training environment: **{training_environment}**.

Training run reference: {run_reference}

Training source revision: `{source_revision}`.

Previous standard-model Actions run: {standard_report['run_url']}

## Licenses and provenance

- Body initialization: [DWPose](https://github.com/IDEA-Research/DWPose) /
  [MMPose](https://github.com/open-mmlab/mmpose), Apache-2.0 source provenance.
  Original checkpoint: {fast['pretrained_checkpoint']['url']} .
- Body and face evaluation annotations: COCO-WholeBody, CC BY-NC 4.0 / research
  and noncommercial terms. Original COCO images retain their source licenses.
  This repository does not redistribute those images or annotations.
- Foot model: TorchVision MobileNetV3-Small pretrained features (BSD-3-Clause)
  and a newly trained classification head. FootNet data are CC BY 4.0:
  https://zenodo.org/records/20457253 . Attribution: Adith Vijay, Akash Prabhune,
  Vinay R Srihari, and Akhila Rayampalli.
- Face model: https://github.com/cunjian/pytorch_face_landmark , pinned source
  {FACE_URL} . The upstream checkpoint redistribution grant has **not been
  verified**. This mirror records that unresolved status and does not assert MIT
  or a blanket permissive license for the face weights.

The components have different model/data terms; `license: other` is intentional.
No user videos, screenshots, crops, video-derived features, pose traces, or
authentication credentials are part of the publication.
"""


def publish():
    global STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-dir", type=Path, default=Path("artifacts/compact59"))
    parser.add_argument("--fast-dir", type=Path, default=Path("artifacts/fast59"))
    parser.add_argument("--pose3d-dir", type=Path, default=Path("artifacts/pose3d"))
    parser.add_argument("--parts-dir", type=Path, default=Path("artifacts/parts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/hf"))
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise PublicationError("actions_only")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PublicationError("missing_HF_TOKEN")
    STAGE = "read_allowlisted_public_artifacts"
    standard = read_json(args.standard_dir, "manifest.json")
    fast = read_json(args.fast_dir, "manifest.json")
    parts = read_json(args.parts_dir, "parts-manifest.json")
    standard_report = body_metrics(read_json(args.standard_dir, "metrics.json"))
    fast_report = body_metrics(read_json(args.fast_dir, "metrics.json"))
    public_parts = parts_metrics(read_json(args.parts_dir, "parts-report.json"))
    if standard.get("file") != "compact59.onnx" or fast.get("file") != "fast59.onnx":
        raise PublicationError("unexpected_body_filename")
    if len(parts.get("models", [])) != 2:
        raise PublicationError("unexpected_parts_inventory")
    by_kind = {item.get("kind"): item for item in parts["models"]}
    if set(by_kind) != {"faceLandmarks", "footView"}:
        raise PublicationError("unexpected_parts_kind")
    foot, face = by_kind["footView"], by_kind["faceLandmarks"]
    if foot.get("filename") != "foot-view.onnx":
        raise PublicationError("unexpected_foot_filename")
    if face.get("url") != FACE_URL or face.get("sha256") != FACE_SHA256 or face.get("bytes") != FACE_BYTES:
        raise PublicationError("unexpected_face_source")
    payloads = {
        "body/fast59.onnx": checked_model(args.fast_dir, "fast59.onnx", fast),
        "body/compact59.onnx": checked_model(args.standard_dir, "compact59.onnx", standard),
        "parts/foot-view.onnx": checked_model(args.parts_dir, "foot-view.onnx", foot),
    }
    pose3d = read_json(args.pose3d_dir, "manifest.json")
    if pose3d.get("sha256") != "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f" or pose3d.get("provenance", {}).get("privateInputsUsed") is not False:
        raise PublicationError("unexpected_pose3d_source")
    payloads["body/pose3d.onnx"] = checked_model(args.pose3d_dir, "pose3d.onnx", pose3d)
    payloads["licenses/POSE3D-APACHE-2.0.txt"] = (args.pose3d_dir / "LICENSE-POSE3D.txt").read_bytes()
    person_detector = read_json(args.pose3d_dir, "person-detector-manifest.json")
    if person_detector.get("sha256") != "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f":
        raise PublicationError("unexpected_person_detector_source")
    payloads["body/person-detector.onnx"] = checked_model(args.pose3d_dir, "person-detector.onnx", person_detector)
    payloads["licenses/PERSON-DETECTOR-APACHE-2.0.txt"] = (args.pose3d_dir / "LICENSE-PERSON-DETECTOR.txt").read_bytes()
    STAGE = "download_pinned_public_face"
    payloads["parts/pfld68.onnx"] = public_download(FACE_URL, FACE_BYTES, FACE_SHA256, keep=True)
    for report in (standard, fast, pose3d, person_detector, parts, standard_report, fast_report, public_parts):
        public_values(report, token)
    payloads["evidence/compact59-metrics.json"] = encode_json(standard_report)
    payloads["evidence/fast59-metrics.json"] = encode_json(fast_report)
    payloads["evidence/parts-metrics.json"] = encode_json(public_parts)
    STAGE = "load_hub_dependency"
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(endpoint="https://huggingface.co", token=token)
    STAGE = "resolve_token_owner"
    owner = api.whoami()["name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", owner):
        raise PublicationError("unexpected_hub_owner")
    repo_id = owner + "/VRC-Compact59-7fe4fac1"
    STAGE = "create_public_repository"
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    information = api.repo_info(repo_id=repo_id, repo_type="model")
    if information.private:
        # create_repo(exist_ok=True) does not alter existing visibility. Never
        # expose unrelated contents by changing an already-private repository.
        raise PublicationError("existing_target_repository_is_private")
    def guarded_commit(payload, message):
        # Preserve other authorized experiment commits when two Actions jobs
        # publish disjoint files concurrently. Every retry reads the new HEAD.
        for attempt in range(3):
            current = api.repo_info(repo_id=repo_id, repo_type="model", revision="main")
            if current.private:
                raise PublicationError("existing_target_repository_is_private")
            try:
                return api.create_commit(
                    repo_id=repo_id, repo_type="model", revision="main", parent_commit=current.sha,
                    operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                                for name, data in payload.items()],
                    commit_message=message, num_threads=2)
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status not in (409, 412) or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    card = model_card(fast, standard, fast_report, standard_report, public_parts, repo_id)
    card += "\n## Learned depth with Compact59 XY\n\n`body/pose3d.onnx` is the pinned 5,557,238-byte OpenCV Zoo MediaPipe Pose model (Apache-2.0), not a model trained in this project. Its standalone XY underperformed Compact59 on the initial 32-image labelled comparison, so current reliable Compact59 XY stays primary at up to15 Hz; learned relative depth is supplied at up to5 Hz with a300 ms expiry. `body/person-detector.onnx` acquires/reacquires an ROI at up to2 Hz and its virtual landmarks never become observed body joints. Relative root travel is fitted from current2D and learned3D; it is not calibrated absolute position. Confidence, driven-limb coverage, partial-body loss and retained motion must all be reported. Smaller rotation p95 alone does not establish tracking quality.\n"
    if token in card:
        raise PublicationError("credential_value_in_model_card")
    payloads["README.md"] = card.encode("utf-8")
    STAGE = "commit_allowlisted_models_and_aggregate_evidence"
    commit = guarded_commit(payloads, "Publish public-data VRC pose models and measured evaluation")
    revision = commit.oid
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise PublicationError("unexpected_model_commit")

    def hub_url(name):
        return f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}"

    STAGE = "verify_anonymous_model_and_evidence_downloads"
    published_files = []
    for name, data in payloads.items():
        expected = hashlib.sha256(data).hexdigest()
        verified = public_download(hub_url(name), len(data), expected)
        published_files.append({"path": name, "url": hub_url(name), **verified})
    standard["url"] = hub_url("body/compact59.onnx")
    pose3d["url"] = hub_url("body/pose3d.onnx")
    person_detector["url"] = hub_url("body/person-detector.onnx")
    standard["evaluation"] = hub_url("evidence/compact59-metrics.json")
    fast["url"] = hub_url("body/fast59.onnx")
    fast["evaluation"] = hub_url("evidence/fast59-metrics.json")
    foot["url"] = hub_url("parts/foot-view.onnx")
    face["upstreamUrl"] = FACE_URL
    face["url"] = hub_url("parts/pfld68.onnx")
    face["filename"] = "pfld68.onnx"
    face["redistributed"] = True
    face["license"] = "Upstream checkpoint redistribution grant unverified; no MIT or blanket permissive license asserted."
    recommended = "fast" if fast_is_eligible(fast, fast_report) else "standard"
    manifest = {"schemaVersion": 2, "body": fast if recommended == "fast" else standard,
                "recommendedBody": recommended, "bodies": {"fast": fast, "standard": standard},
                "pose3d": pose3d, "personDetector": person_detector, "parts": parts, "hub": {"repoId": repo_id, "modelRevision": revision,
                                          "publicAnonymousHashVerified": True}}
    public_values(manifest, token)
    metadata = {"models.json": encode_json(manifest), "configs/fast59.json": encode_json(fast),
                "configs/compact59.json": encode_json(standard), "configs/pose3d.json": encode_json(pose3d), "configs/person-detector.json": encode_json(person_detector), "configs/parts.json": encode_json(parts)}
    STAGE = "commit_immutable_download_manifest"
    manifest_commit = guarded_commit(metadata, "Pin browser model downloads to verified model commit")
    manifest_revision = manifest_commit.oid
    if not re.fullmatch(r"[0-9a-f]{40}", manifest_revision):
        raise PublicationError("unexpected_manifest_commit")
    STAGE = "verify_anonymous_manifest_downloads"
    for name, data in metadata.items():
        url = f"https://huggingface.co/{repo_id}/resolve/{manifest_revision}/{name}"
        verified = public_download(url, len(data), hashlib.sha256(data).hexdigest())
        published_files.append({"path": name, "url": url, **verified})
    publication = {"status": "published", "repoId": repo_id, "repoUrl": f"https://huggingface.co/{repo_id}",
                   "modelRevision": revision, "manifestRevision": manifest_revision,
                   "manifestUrl": f"https://huggingface.co/{repo_id}/resolve/{manifest_revision}/models.json",
                   "publicAnonymousHashVerified": True, "privateInputsIncluded": False,
                   "fileCount": len(published_files), "files": published_files}
    STAGE = "write_local_publication_result"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models.json").write_bytes(metadata["models.json"])
    (args.output / "publication.json").write_bytes(encode_json(publication))
    (args.output / "README.md").write_text(card, encoding="utf-8")
    return {"status": "published", "repoId": repo_id, "modelRevision": revision,
            "manifestRevision": manifest_revision, "fileCount": len(published_files),
            "publicAnonymousHashVerified": True, "privateInputsIncluded": False}


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_DEBUG"] = "0"
    logging.disable(logging.CRITICAL)
    try:
        # Never forward SDK exceptions, signed redirect URLs, or debug output.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = publish()
    except Exception as error:
        code = error.code if isinstance(error, PublicationError) else type(error).__name__
        status = getattr(getattr(error, "response", None), "status_code", None)
        if not isinstance(status, int):
            status = getattr(error, "code", None)
        report = {"status": "failed", "stage": STAGE, "code": code,
                  "httpStatus": status if isinstance(status, int) else None,
                  "detailsSuppressed": "Credentials and signed URLs are never logged."}
        if isinstance(error, ValueError):
            # Return only fixed field names, never the exception text itself.
            message = str(error)
            report["diagnosticTags"] = [field for field in
                                        ("Invalid metadata in README.md", "license_link", "license_name",
                                         "language", "tags", "parent_commit", "commit_message")
                                        if field in message]
        print(json.dumps(report), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(summary))
