"""Install only the kit dependencies, without Colab's optional ensurepip.

CUDA PyTorch remains the runtime's installed package. Notebook/kernel packages
are untouched: child processes receive a separate PYTHONPATH directory.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def runtime_environment(root):
    target = root / ".python-packages"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(target), environment.get("PYTHONPATH", "")) if part)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def install(root):
    requirements = root / "colab" / "requirements.txt"
    # The pinned requirements deliberately do not install/replace PyTorch.
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith(("torch", "nvidia-")):
            raise ValueError("The kit must preserve the runtime's CUDA PyTorch")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-warn-script-location", "--upgrade", "--target", str(root / ".python-packages"),
                    "-r", str(requirements)], check=True)
    environment = runtime_environment(root)
    # The same child-process import route is used by training, export and
    # packaging, and can be exercised on an Actions CPU without pretending to
    # validate a Colab GPU allocation.
    subprocess.run([sys.executable, "-c",
                    "import torch,numpy,onnx,onnxruntime,cv2; "
                    "print({'dependency_imports':'passed','torch':torch.__version__,"
                    "'cuda_available':torch.cuda.is_available()})"],
                   env=environment, check=True)
    return environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    install(args.root.resolve())
