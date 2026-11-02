"""Actions-only recovery/handoff check using actual short-trained artifacts.

This checks transport and failure semantics, not model accuracy or CUDA. The
trained/exported files must already have been produced by the Actions job.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from package_result import DIAGNOSTIC_NAME, FAST_FILES, INT8_FILES, RESULT_NAME, package
from run import execute, safe_error


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        receipt = json.loads(archive.read("handoff.json"))
        expected = {item["path"] for item in receipt["files"]} | {"handoff.json"}
        if set(archive.namelist()) != expected:
            raise AssertionError("Handoff member list differs")
        for record in receipt["files"]:
            data = archive.read(record["path"])
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise AssertionError("Handoff hash differs")
        if not receipt["containsOnnx"]:
            raise AssertionError("Actual model exports were lost during packaging")
        for name in receipt["onnxFiles"]:
            if archive.getinfo(name).file_size < 100_000:
                raise AssertionError("Expected actual ONNX model bytes")
        return receipt


def main():
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise RuntimeError("This check requires an actual GitHub Actions runner")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vrc-handoff-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        root = Path(temporary)
        (root / "bundle.json").write_text(json.dumps({"handoffId": "7fe4fac1", "privateInputsIncluded": False,
             "sourceCommit": os.environ["GITHUB_SHA"], "files": []}), encoding="utf-8")
        output = root / "output"
        for name, source, allowed in (("fast59", args.fast, FAST_FILES), ("int8", args.int8, INT8_FILES)):
            (output / name).mkdir(parents=True)
            for filename in allowed:
                shutil.copyfile(source / filename, output / name / filename)
        recovery = args.fast / "recovery"
        shutil.copytree(recovery, output / "fast59" / "recovery")
        recovery_records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(recovery.glob("*.json"))]
        if len(recovery_records) != 2 or any(record["completed_steps"] < 1 for record in recovery_records):
            raise AssertionError("Expected both candidates' saved actual gradient updates")
        status = {"handoffId": "7fe4fac1", "privateInputsIncluded": False, "status": "completed",
                  "sourceCommit": os.environ["GITHUB_SHA"]}
        status_path = output / "run-status.json"
        status_path.write_text(json.dumps(status), encoding="utf-8")
        successful = package(root, root / "returned")
        complete = inspect(successful)
        if successful.name != RESULT_NAME or complete["status"] != "completed":
            raise AssertionError("A complete handoff was not identified")

        # Exercise an actual subprocess failure, retaining its concrete reason.
        code, tail = execute([sys.executable, "-c", "raise RuntimeError('intentional_handoff_failure_check')"],
                             root, os.environ.copy())
        if code == 0 or "intentional_handoff_failure_check" not in tail:
            raise AssertionError("Subprocess failure or its cause was swallowed")
        status.update({"status": "failed", "stages": [{"name": "failure_path_exercise", "status": "failed",
                                                       "exitCode": code, "errorTail": tail}]})
        status_path.write_text(json.dumps(status), encoding="utf-8")
        # Simulate interruption before final export while keeping real periodic
        # checkpoint/ONNX files. Independent INT8 output must still be returned.
        (output / "fast59" / "fast59.onnx").unlink()
        diagnostic_path = package(root, root / "returned")
        diagnostic = inspect(diagnostic_path)
        if diagnostic_path.name != DIAGNOSTIC_NAME or diagnostic["status"] != "failed":
            raise AssertionError("Failure was presented as a completed result")
        expected_recovery = {"fast59/recovery/" + record["candidate"] + ".onnx" for record in recovery_records}
        if not expected_recovery.issubset(diagnostic["onnxFiles"]) or "int8/compact59-int8.onnx" not in diagnostic["onnxFiles"]:
            raise AssertionError("Recovery or independent INT8 export was dropped")
        # A completed status alone cannot make a model-less archive successful.
        status["status"] = "completed"
        status_path.write_text(json.dumps(status), encoding="utf-8")
        missing_path = package(root, root / "returned")
        missing = inspect(missing_path)
        if missing_path.name != DIAGNOSTIC_NAME or "fast59/fast59.onnx" not in missing["missingRequiredFiles"]:
            raise AssertionError("Missing required ONNX was not visible")
        redacted = safe_error("CUDA out of memory; Tried to allocate 10.0 GiB; token=example; https://invalid.example/private")
        if "10.0 GiB" not in redacted or "token=example" in redacted or "invalid.example" in redacted:
            raise AssertionError("Useful error detail was lost or a URL/credential was retained")
        # A real Actions CPU is not a Colab runtime. The entrypoint must reject
        # it with a nonzero exit and a failed receipt, never pretend success.
        guard_root = root / "environment-guard"
        guard_root.mkdir()
        shutil.copyfile(root / "bundle.json", guard_root / "bundle.json")
        guard = subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")),
                                "--root", str(guard_root)], check=False, text=True, capture_output=True)
        guard_status = json.loads((guard_root / "output" / "run-status.json").read_text())
        if guard.returncode == 0 or guard_status["status"] != "failed":
            raise AssertionError("The Colab entrypoint hid its environment failure")
        report = {"schema": "vrc-handoff-check-v1", "status": "passed", "sourceCommit": os.environ["GITHUB_SHA"],
                  "execution": "actual short-trained public-data artifact packaging and intentional subprocess failure",
                  "cudaExecutionVerified": False, "modelAccuracyEstablished": False,
                  "completeOnnxFiles": complete["onnxFiles"], "recoveredOnnxFiles": diagnostic["onnxFiles"],
                  "recoveryCompletedSteps": [record["completed_steps"] for record in recovery_records],
                  "failureExitCode": code, "failureDiagnosticName": diagnostic_path.name,
                  "colabEntrypointFailureExitCode": guard.returncode,
                  "missingOnnxCannotProduceSuccess": True, "allArchiveHashesVerified": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
