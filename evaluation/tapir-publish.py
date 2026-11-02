#!/usr/bin/env python3
"""Publish verified public ONNX artifacts; this process never loads a checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REPOSITORY = "RabbitDaisuke/Foot-Point-Tracker-onnx"
FALLBACK_REPOSITORY = "RabbitDaisuke/Wholebody-Refinement-onnx"
FALLBACK_PREFIX = "point-tracking/phase3-4fc7ff60"


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("TAPIR model publication runs in GitHub Actions only.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = next((os.environ.get(key) for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN") if os.environ.get(key)), None)
    if not token:
        raise RuntimeError("The repository needs a Hugging Face write token in Actions secrets.")
    api = HfApi(token=token)
    repository, prefix, selection = REPOSITORY, "", "requested-repository"
    try:
        info = api.model_info(repository)
    except HfHubHTTPError as error:
        if error.response is None or error.response.status_code != 404:
            raise
        repository, prefix, selection = FALLBACK_REPOSITORY, FALLBACK_PREFIX, "requested-repository-authenticated-404"
        info = api.model_info(repository)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    if card.get("license") != "apache-2.0":
        raise RuntimeError("The authorized target repository must declare apache-2.0.")
    manifest_path = args.models / "tapir-models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != REPOSITORY or manifest.get("license") != "apache-2.0" or manifest.get("parityPassed") is not True or manifest.get("codecParityPassed") is not True:
        raise ValueError("Publication requires successful official-model and browser-codec numerical parity.")
    if {model.get("id") for model in manifest.get("models", [])} != {"encoder", "query", "step"} or len(manifest["models"]) != 3:
        raise ValueError("Expected exactly the three reviewed model stages.")
    manifest["requestedRepository"] = REPOSITORY
    manifest["repository"] = repository
    manifest["pathPrefix"] = prefix
    allowed = ["tapir-models.json", "README.md", "LICENSE"]
    for model in manifest["models"]:
        filename = model["filename"]
        if filename != f"causal-bootstapir-{model['id']}-256-q3-fp32.onnx":
            raise ValueError("Unexpected model filename.")
        content = (args.models / filename).read_bytes()
        if len(content) != model["bytes"] or hashlib.sha256(content).hexdigest() != model["sha256"]:
            raise ValueError("Model checksum differs from the reviewed Actions artifact.")
        allowed.append(filename)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Fallback publication is confined to this chat's folder. In particular the
    # existing repository-root README, LICENSE and other models are never sent.
    commit = api.upload_folder(repo_id=repository, folder_path=str(args.models), path_in_repo=prefix or None, allow_patterns=allowed,
                               commit_message="Publish numerically verified causal BootsTAPIR ONNX graphs from Actions")
    for model in manifest["models"]:
        model["revision"] = commit.oid
        model_path = f"{prefix}/{model['filename']}" if prefix else model["filename"]
        model["url"] = f"https://huggingface.co/{repository}/resolve/{commit.oid}/{model_path}"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_name = f"{prefix}/tapir-models.json" if prefix else "tapir-models.json"
    metadata = api.upload_file(repo_id=repository, path_or_fileobj=str(manifest_path), path_in_repo=manifest_name,
                               commit_message="Pin recurrent model URLs to the verified weight revision")
    result = {"status": "published", "repository": repository, "pathPrefix": prefix, "requestedRepository": REPOSITORY,
              "targetSelection": selection, "license": "apache-2.0", "weightsRevision": commit.oid,
              "manifestRevision": metadata.oid, "manifestUrl": f"https://huggingface.co/{repository}/resolve/{metadata.oid}/{manifest_name}",
              "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
              "exportGitCommit": manifest.get("gitCommit"), "exportWorkflowRunId": manifest.get("workflowRunId"), "models": manifest["models"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "published", "repository": repository, "pathPrefix": prefix, "weightsRevision": commit.oid, "manifestRevision": metadata.oid}))


if __name__ == "__main__":
    main()
