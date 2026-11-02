"""Run the user-authorized public-image experiment on a Colab CUDA GPU.

No credentials, video paths, Drive mount, GitHub push, or Hub upload are needed.
Model publication and final Node measurements remain in GitHub Actions.
"""

import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


HANDOFF_ID = "7fe4fac1"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_error(message):
    message = re.sub(r"https?://\S+", "[url]", message)
    message = re.sub(r"\b(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+", "[credential]", message)
    message = re.sub(r"(?i)(token|password|secret|authorization)\s*[:=]\s*\S+", r"\1=[redacted]", message)
    message = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s/]+/)*[^\s/]+", "[path]", message)
    return message[-6000:]


def execute(command, root, env):
    """Stream notebook output and preserve a bounded, redacted failure tail."""
    tail = deque(maxlen=24)
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line[-4096:])
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return code, safe_error("".join(tail)) if code else None


def verify_bundle(root):
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("handoffId") != HANDOFF_ID or manifest.get("privateInputsIncluded") is not False:
        raise ValueError("unexpected_bundle_identity")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("bundle_path_escapes_root")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("bundle_checksum_mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run-status.json"
    started = time.time()
    status = {"schema": "vrc-colab-run-v2", "handoffId": HANDOFF_ID, "status": "running",
              "trainingEnvironment": "google-colab", "privateInputsIncluded": False,
              "trainingData": "public COCO-WholeBody still images only", "stages": [],
              "finalNodeAndBrowserRuntimeMeasured": False, "hfPublicationPerformed": False}
    try:
        manifest = verify_bundle(root)
        status["sourceCommit"] = manifest["sourceCommit"]
        status["sourceBundleManifestSha256"] = hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest()
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            config = previous.get("configuration", {})
            same = (config.get("stepsPerCandidate") == args.steps and config.get("batchSize") == args.batch_size
                    and config.get("microBatchSize") == min(args.batch_size, args.micro_batch_size)
                    and previous.get("sourceBundleManifestSha256") == status["sourceBundleManifestSha256"])
            if previous.get("status") == "completed" and same:
                print("完了済みの同じ実験を再利用します。testを再選択して再学習しません。", flush=True)
                return 0
            # Check this before probing CUDA or new batch settings, so an
            # unrelated later failure cannot overwrite the original cause.
            print("既存結果を保持しました。先に返却ZIPをダウンロードしてください。", flush=True)
            return 1
        if not importlib.util.find_spec("google.colab") or not Path("/content").is_dir():
            raise RuntimeError("colab_runtime_required")
        import torch
        import numpy
        import onnx
        import onnxruntime
        if not torch.cuda.is_available():
            raise RuntimeError("select_a_gpu_runtime_before_training")
        if args.steps <= 400:
            raise ValueError("full_model_fine_tuning_required")
        if not 2 <= args.batch_size <= 256 or not 1 <= args.micro_batch_size <= 16:
            raise ValueError("batch_size_must_be_2_to_256_and_micro_batch_size_1_to_16")
        micro = min(args.batch_size, args.micro_batch_size)
        status["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                             "gpuMemoryBytes": torch.cuda.get_device_properties(0).total_memory,
                             "numpy": numpy.__version__, "onnx": onnx.__version__,
                             "onnxruntime": onnxruntime.__version__, "logicalCpus": os.cpu_count()}
        print(json.dumps({"gpuReady": True, **status["runtime"]}), flush=True)
        status["configuration"] = {"stepsPerCandidate": args.steps, "batchSize": args.batch_size,
                                   "microBatchSize": micro,
                                   "gradientAccumulationSteps": math.ceil(args.batch_size / micro),
                                   "trainImages": 768, "validationImages": 128, "testImages": 192,
                                   "seed": 20261105, "device": "cuda"}
        print(json.dumps({"batchPlan": status["configuration"]}), flush=True)
        write_json(output / "source-bundle.json", manifest)
        write_json(status_path, status)
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1", "VRC_SOURCE_COMMIT": manifest["sourceCommit"]})
        common = ["--cache", ".cache/compact59", "--baseline-checkpoint", "baseline/compact59.pth",
                  "--exclude-split", "baseline/public-split.json", "--exclude-split", "excluded-fast-v1/public-split.json",
                  "--execution-environment", "colab", "--train-count", "768", "--val-count", "128",
                  "--test-count", "192", "--threads", "2", "--seed", "20261105"]
        if (root / "excluded-smoke" / "public-split.json").is_file():
            common.extend(["--exclude-split", "excluded-smoke/public-split.json"])
        fast_command = [sys.executable, "training/train_fast.py", "--output", "output/fast59", *common,
                                 "--device", "cuda", "--steps", str(args.steps), "--warmup-steps", "400",
                                 "--batch-size", str(args.batch_size), "--micro-batch-size", str(micro)]
        commands = [("public_split", [*fast_command, "--prepare-split-only"]),
                    ("fast59", fast_command),
                    ("int8", [sys.executable, "training/quantize_stable.py", "--output", "output/int8",
                               "--cache", ".cache/compact59", "--baseline-model", "baseline/compact59.onnx",
                               "--baseline-manifest", "baseline/manifest.json", "--baseline-metrics", "baseline/metrics.json",
                               "--shared-split", "output/fast59/public-split.json", "--execution-environment", "colab",
                               "--device", "cpu", "--calibration-count", "128", "--threads", "2"])]
        for name, command in commands:
            stage = {"name": name, "status": "running"}
            status["stages"].append(stage)
            write_json(status_path, status)
            start = time.monotonic()
            code, error_tail = execute(command, root, env)
            stage.update({"status": "completed" if code == 0 else "failed", "exitCode": code,
                          "elapsedSeconds": round(time.monotonic() - start, 3)})
            if code:
                stage["errorTail"] = error_tail
                failure_path = output / "fast59" / "failure.json"
                if name in ("public_split", "fast59") and failure_path.is_file():
                    stage["failure"] = json.loads(failure_path.read_text(encoding="utf-8"))
            write_json(status_path, status)
            if code and name == "public_split":
                break
            # INT8 consumes the sealed split, not the fast model. A failed
            # GPU subprocess must not suppress this independent experiment.
        status["status"] = "completed" if len(status["stages"]) == 3 and all(
            stage["status"] == "completed" for stage in status["stages"]) else "failed"
        if status["status"] == "failed":
            status["errorCode"] = "one_or_more_experiments_failed"
    except (Exception, KeyboardInterrupt) as error:
        status["status"] = "failed"
        status["errorType"] = type(error).__name__
        message = str(error)
        status["errorMessage"] = safe_error(message)
        if message and all(c.islower() or c == "_" for c in message):
            status["errorCode"] = message
        print(json.dumps({"status": "failed", "errorType": type(error).__name__,
                          "errorCode": status.get("errorCode", "see_colab_cell_output")}), flush=True)
    finally:
        if status.get("status") != "running":
            status["elapsedSeconds"] = round(time.time() - started, 3)
            write_json(status_path, status)
    if status["status"] == "completed":
        print("学習と評価が完了しました。モデルを含む結果ZIPを作成します。", flush=True)
        return 0
    print("学習は完了していません。保存済みモデルと失敗理由を診断ZIPへ回収します。", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
