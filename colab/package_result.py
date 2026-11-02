"""Create the fixed-name model-only return ZIP from an explicit allowlist."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


RESULT_NAME = "VRC_Compact59_Colab_Result_7fe4fac1.zip"
FAST_FILES = ("fast59.onnx", "fast59.pth", "manifest.json", "metrics.json", "public-split.json",
              "selection-policy.json", "selection.json", "validation-candidates.json", "training-curve.json")
INT8_FILES = ("compact59-int8.onnx", "manifest.json", "metrics.json", "public-split.json",
              "selection-policy.json", "selection.json", "validation-candidates.json")


def package(root, destination):
    output = root / "output"
    status = json.loads((output / "run-status.json").read_text(encoding="utf-8"))
    if status.get("handoffId") != "7fe4fac1" or status.get("privateInputsIncluded") is not False:
        raise ValueError("unexpected_return_identity")
    members = [(output / "run-status.json", "run-status.json"), (root / "bundle.json", "source-bundle.json")]
    for directory, filenames in (("fast59", FAST_FILES), ("int8", INT8_FILES)):
        for filename in filenames:
            path = output / directory / filename
            if path.is_file():
                members.append((path, directory + "/" + filename))
            elif status.get("status") == "completed":
                raise ValueError("completed_experiment_is_missing_an_allowlisted_output: " + directory + "/" + filename)
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
    receipt = {"schema": "vrc-colab-return-v1", "handoffId": "7fe4fac1", "status": status["status"],
               "expectedFileName": RESULT_NAME, "sourceCommit": status.get("sourceCommit"),
               "sourceBundleManifestSha256": hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest(),
               "privateInputsIncluded": False, "imagesOrFeatureCachesIncluded": False,
               "nodeActionsEvaluationRequired": True, "hfPublicationPerformed": False,
               "files": records}
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / RESULT_NAME
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in members:
            archive.write(path, name)
        archive.writestr("handoff.json", json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": status["status"], "fileName": target.name, "bytes": target.stat().st_size,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "privateInputsIncluded": False}), flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, default=Path("/content"))
    arguments = parser.parse_args()
    package(arguments.root.resolve(), arguments.destination.resolve())
