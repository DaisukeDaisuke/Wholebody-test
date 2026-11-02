"""Run the user-authorized public-image experiment on a Colab CUDA GPU.

No credentials, video paths, Drive mount, GitHub push, or Hub upload are needed.
Model publication and final Node measurements remain in GitHub Actions.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


HANDOFF_ID = "7fe4fac1"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run-status.json"
    started = time.time()
    status = {"schema": "vrc-colab-run-v1", "handoffId": HANDOFF_ID, "status": "running",
              "trainingEnvironment": "google-colab", "privateInputsIncluded": False,
              "trainingData": "public COCO-WholeBody still images only", "stages": [],
              "finalNodeAndBrowserRuntimeMeasured": False, "hfPublicationPerformed": False}
    try:
        manifest = verify_bundle(root)
        status["sourceCommit"] = manifest["sourceCommit"]
        status["sourceBundleManifestSha256"] = hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest()
        if not importlib.util.find_spec("google.colab") or not Path("/content").is_dir():
            raise RuntimeError("colab_runtime_required")
        import torch
        import numpy
        import onnx
        import onnxruntime
        if not torch.cuda.is_available():
            raise RuntimeError("select_a_gpu_runtime_before_training")
        if args.steps <= 400 or args.batch_size < 2:
            raise ValueError("full_model_fine_tuning_required")
        status["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                             "gpuMemoryBytes": torch.cuda.get_device_properties(0).total_memory,
                             "numpy": numpy.__version__, "onnx": onnx.__version__,
                             "onnxruntime": onnxruntime.__version__, "logicalCpus": os.cpu_count()}
        print(json.dumps({"gpuReady": True, **status["runtime"]}), flush=True)
        status["configuration"] = {"stepsPerCandidate": args.steps, "batchSize": args.batch_size,
                                   "trainImages": 768, "validationImages": 128, "testImages": 192,
                                   "seed": 20261105, "device": "cuda"}
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("status") == "completed":
                if previous.get("configuration") != status["configuration"] or previous.get("sourceBundleManifestSha256") != status["sourceBundleManifestSha256"]:
                    raise SystemExit("完了済みの実験と設定が異なります。既存結果を保持したまま停止します。")
                print("完了済みの同じ実験を再利用します。testを再選択して再学習しません。", flush=True)
                return
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
        commands = [("fast59", [sys.executable, "training/train_fast.py", "--output", "output/fast59", *common,
                                 "--device", "cuda", "--steps", str(args.steps), "--warmup-steps", "400",
                                 "--batch-size", str(args.batch_size)]),
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
            code = subprocess.run(command, cwd=root, env=env, check=False).returncode
            stage.update({"status": "completed" if code == 0 else "failed", "exitCode": code,
                          "elapsedSeconds": round(time.monotonic() - start, 3)})
            write_json(status_path, status)
            if code:
                raise RuntimeError("training_subprocess_failed")
        status["status"] = "completed"
    except Exception as error:
        status["status"] = "failed"
        status["errorType"] = type(error).__name__
        # Fixed application errors are useful; do not copy arbitrary URLs,
        # environment variables, library exceptions, or tracebacks into output.
        message = str(error)
        if message and all(c.islower() or c == "_" for c in message):
            status["errorCode"] = message
        print(json.dumps({"status": "failed", "errorType": type(error).__name__,
                          "errorCode": status.get("errorCode", "see_colab_cell_output")}), flush=True)
    finally:
        if status.get("status") != "running":
            status["elapsedSeconds"] = round(time.time() - started, 3)
            write_json(status_path, status)
    print("結果を梱包する次のセルへ進んでください。", flush=True)


if __name__ == "__main__":
    main()
