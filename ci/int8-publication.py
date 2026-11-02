"""Publish one measured public-data INT8 candidate without changing defaults.

The API token is Actions-only. No directory upload, video, image, private
feature, or per-frame trace is accepted. Anonymous resolve URLs are verified
with urllib, which does not consult Hugging Face credentials.
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

from publish_hf import (PublicationError, checked_model, encode_json, public_download,
                        public_values, read_json)


REPO_ID = "RabbitDaisuke/VRC-Compact59-7fe4fac1"
SPLIT_SHA256 = "2e1cab84af5c1ab7174bc177a5aca79d165aa72696940f6d37f55321fbcd0b36"
STAGE = "initialization"


def guarded_commit(api, operations, message):
    """Add only this experiment's paths; retry a concurrent HEAD advance."""
    from huggingface_hub import CommitOperationAdd
    for attempt in range(3):
        info = api.repo_info(repo_id=REPO_ID, repo_type="model", revision="main")
        if info.private:
            raise PublicationError("target_repository_is_not_public")
        try:
            result = api.create_commit(
                repo_id=REPO_ID, repo_type="model", revision="main", parent_commit=info.sha,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                            for name, data in operations.items()], commit_message=message, num_threads=2)
            if not re.fullmatch(r"[0-9a-f]{40}", result.oid):
                raise PublicationError("unexpected_hub_commit")
            return result.oid
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status not in (409, 412) or attempt == 2:
                raise
            time.sleep(2 ** attempt)


def publish():
    global STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true" or os.environ.get("GITHUB_REPOSITORY") != "DaisukeDaisuke/Wholebody-test":
        raise PublicationError("actual_project_actions_runner_required")
    if os.environ.get("GITHUB_REF_NAME") != "compact-vrc-20261102-7fe4fac1":
        raise PublicationError("isolated_branch_required")
    commit = os.environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PublicationError("source_commit_required")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise PublicationError("missing_HF_TOKEN")
    STAGE = "validate_actual_public_evaluation"
    manifest = read_json(args.input, "manifest.json")
    metrics = read_json(args.input, "metrics.json")
    if metrics.get("schema") != "compact59-static-quantization-v1" or metrics.get("status") != "completed" or metrics.get("commit") != commit:
        raise PublicationError("completed_current_actions_quantization_required")
    data, calibration = metrics.get("data", {}), metrics.get("calibration", {})
    if data.get("private_videos_used") is not False or data.get("smoke_test_only") is not False:
        raise PublicationError("public_full_evaluation_required")
    if [data.get(name) for name in ("calibration_images", "validation_images", "heldout_images")] != [128, 128, 192]:
        raise PublicationError("full_calibration_validation_test_counts_required")
    if data.get("calibration_validation_test_image_overlap") != 0 or calibration.get("training_split_only") is not True or calibration.get("shared_student_split_sha256") != SPLIT_SHA256:
        raise PublicationError("fixed_disjoint_public_split_required")
    if manifest.get("source_commit") != commit or manifest.get("input", {}).get("shape") != [1, 3, 256, 192]:
        raise PublicationError("unexpected_model_contract")
    if manifest.get("decodeWH") != [192, 256] or manifest.get("quantization") != calibration:
        raise PublicationError("manifest_calibration_mismatch")
    acceptance = metrics.get("acceptance", {})
    eligible = acceptance.get("production_eligible") is True
    if manifest.get("production_eligible") is not eligible:
        raise PublicationError("eligibility_mismatch")
    if eligible and (acceptance.get("validation_gates_passed") is not True or acceptance.get("heldout_quality_gates_passed") is not True):
        raise PublicationError("eligible_without_quality_gates")
    if acceptance.get("test_can_only_veto_deployment") is not True or acceptance.get("test_driven_candidate_reselection") is not False:
        raise PublicationError("fixed_selection_test_veto_required")
    model = checked_model(args.input, "compact59-int8.onnx", manifest)
    if metrics["int8"]["sha256"] != manifest["sha256"] or metrics["int8"]["onnx_bytes"] != len(model):
        raise PublicationError("measured_model_differs_from_published_model")
    path = "body/int8-candidate.onnx" if eligible else f"experiments/int8/{commit}/compact59-int8.onnx"
    evidence_prefix = f"evidence/int8/{commit}"
    payloads = {path: model}
    for name in ("manifest.json", "metrics.json", "public-split.json", "selection-policy.json", "selection.json", "validation-candidates.json"):
        value = read_json(args.input, name)
        public_values(value, token)
        payloads[evidence_prefix + "/" + name] = encode_json(value)
    note = ("# Compact59 INT8 candidate\n\n"
            f"Source: `{commit}`. Full public-image calibration/evaluation: 128 train calibration images, "
            "128 validation images, 192 fixed test images. The final test can veto, never reselect.\n\n"
            f"Quality/speed eligibility: **{eligible}**. This is quantization of the existing Compact59 "
            "weights, not recovery of the abandoned Colab run or new supervised training.\n\n"
            "No application default was changed. Actual Node inference, visible-bone coverage, "
            "lost intervals, motion retention and private-video evaluation remain separate adoption gates. "
            "Smaller bone rotation changes alone do not establish tracking quality.\n\n"
            "Quantization preserves MAC count. CPU timings do not measure user-device GPU load. "
            "The model inherits the baseline model and public training-data terms recorded in manifest.json.\n")
    payloads[evidence_prefix + "/README.md"] = note.encode("utf-8")
    STAGE = "publish_only_candidate_and_evidence"
    from huggingface_hub import HfApi
    api = HfApi(endpoint="https://huggingface.co", token=token)
    revision = guarded_commit(api, payloads, "Publish measured Compact59 INT8 candidate without changing application defaults")

    def url(name, revision=revision):
        return f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{name}"

    STAGE = "verify_anonymous_candidate_and_evidence_downloads"
    verified = []
    for name, content in payloads.items():
        receipt = public_download(url(name), len(content), hashlib.sha256(content).hexdigest())
        verified.append({"path": name, "url": url(name), **receipt})
    candidate = copy.deepcopy(manifest)
    candidate.update({"url": url(path), "evaluation": url(evidence_prefix + "/metrics.json"),
                      "file": path.rsplit("/", 1)[-1], "publicAnonymousHashVerified": True,
                      "applicationDefaultChanged": False,
                      "applicationAdoption": "pending_actual_node_and_motion_video_evaluation",
                      "privateVideoValidated": False, "modelRevision": revision})
    public_values(candidate, token)
    config_path = f"configs/int8/{commit}.json"
    config_bytes = encode_json(candidate)
    STAGE = "publish_candidate_download_contract"
    config_revision = guarded_commit(api, {config_path: config_bytes}, "Pin anonymous INT8 candidate downloads to the verified revision")
    config_url = url(config_path, config_revision)
    config_verification = public_download(config_url, len(config_bytes), hashlib.sha256(config_bytes).hexdigest())
    verified.append({"path": config_path, "url": config_url, **config_verification})
    publication = {"schema": "vrc-int8-publication-v1", "status": "published", "repoId": REPO_ID,
                   "sourceCommit": commit, "sourceRun": metrics["run_url"], "modelRevision": revision,
                   "manifestRevision": config_revision, "manifestUrl": config_url, "modelUrl": url(path),
                   "productionEligible": eligible, "applicationDefaultChanged": False,
                   "applicationAdoption": "pending_actual_node_and_motion_video_evaluation",
                   "publicAnonymousHashVerified": True, "privateInputsIncluded": False,
                   "fileCount": len(verified), "files": verified}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_bytes(config_bytes)
    (args.output / "publication.json").write_bytes(encode_json(publication))
    return {key: publication[key] for key in ("status", "repoId", "sourceCommit", "productionEligible",
                                            "modelUrl", "manifestUrl", "publicAnonymousHashVerified",
                                            "applicationDefaultChanged", "fileCount")}


if __name__ == "__main__":
    os.environ.update({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_XET": "1",
                       "HF_HUB_VERBOSITY": "error", "HF_DEBUG": "0"})
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = publish()
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(json.dumps({"status": "failed", "stage": STAGE,
                          "code": error.code if isinstance(error, PublicationError) else type(error).__name__,
                          "httpStatus": status if isinstance(status, int) else None}), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result))
