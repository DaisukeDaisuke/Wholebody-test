"""Create the fixed-name model-only return ZIP from an explicit allowlist."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


RESULT_NAME = "VRC_Compact59_Colab_Result_7fe4fac1.zip"
DIAGNOSTIC_NAME = "VRC_Compact59_Colab_Diagnostic_7fe4fac1.zip"
FAST_FILES = ("fast59.onnx", "fast59.pth", "manifest.json", "metrics.json", "public-split.json",
              "selection-policy.json", "selection.json", "validation-candidates.json", "training-curve.json")
INT8_FILES = ("compact59-int8.onnx", "manifest.json", "metrics.json", "public-split.json",
              "selection-policy.json", "selection.json", "validation-candidates.json")
RECOVERY_IDS = ("fast59-aligned-192", "fast59-aligned-224")


def package(root, destination):
    output = root / "output"
    status = json.loads((output / "run-status.json").read_text(encoding="utf-8"))
    if status.get("handoffId") != "7fe4fac1" or status.get("privateInputsIncluded") is not False:
        raise ValueError("unexpected_return_identity")
    source_manifest = output / "source-bundle.json"
    if not source_manifest.is_file():
        source_manifest = root / "bundle.json"
    members = [(output / "run-status.json", "run-status.json"), (source_manifest, "source-bundle.json")]
    missing = []
    for directory, filenames in (("fast59", FAST_FILES), ("int8", INT8_FILES)):
        for filename in filenames:
            path = output / directory / filename
            if path.is_file():
                members.append((path, directory + "/" + filename))
            elif status.get("status") == "completed":
                missing.append(directory + "/" + filename)
    failure = output / "fast59" / "failure.json"
    if failure.is_file():
        members.append((failure, "fast59/failure.json"))
    for candidate in RECOVERY_IDS:
        for extension in (".pth", ".onnx", ".json"):
            name = "fast59/recovery/" + candidate + extension
            path = output / name
            if path.is_file():
                members.append((path, name))
    records = []
    total = 0
    for path, name in members:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("return_path_escapes_root")
        data = path.read_bytes()
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("return_file_too_large")
        total += len(data)
        if total > 200 * 1024 * 1024:
            raise ValueError("return_archive_too_large")
        records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    completed = status.get("status") == "completed" and not missing
    name = RESULT_NAME if completed else DIAGNOSTIC_NAME
    model_paths = [record["path"] for record in records if record["path"].endswith(".onnx")]
    recovery_issues = []
    for candidate in RECOVERY_IDS:
        metadata_path = output / "fast59" / "recovery" / (candidate + ".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for expected in metadata.get("files", []):
                name_in_archive = "fast59/recovery/" + expected["path"]
                actual = next((record for record in records if record["path"] == name_in_archive), None)
                if actual is None or actual["sha256"] != expected["sha256"]:
                    recovery_issues.append(name_in_archive)
    receipt = {"schema": "vrc-colab-return-v2", "handoffId": "7fe4fac1",
               "status": "completed" if completed else "failed", "runStatus": status["status"],
               "expectedFileName": name, "sourceCommit": status.get("sourceCommit"),
               "sourceBundleManifestSha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
               "privateInputsIncluded": False, "imagesOrFeatureCachesIncluded": False,
               "nodeActionsEvaluationRequired": True, "hfPublicationPerformed": False,
               "containsOnnx": bool(model_paths), "onnxFiles": model_paths,
               "missingRequiredFiles": missing,
               "recoveryPairHashMismatches": recovery_issues,
               "recoveryModelsAreProductionEligible": False,
               "files": records}
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    temporary = target.with_suffix(".pending.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in members:
            archive.write(path, name)
        archive.writestr("handoff.json", json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps({"status": receipt["status"], "fileName": target.name, "bytes": target.stat().st_size,
                      "containsOnnx": bool(model_paths), "onnxFiles": model_paths,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "privateInputsIncluded": False}), flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, default=Path("/content"))
    arguments = parser.parse_args()
    package(arguments.root.resolve(), arguments.destination.resolve())
