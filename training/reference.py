"""Compare the PyTorch reconstruction against an independently exported ONNX.

The reference must use the same published DWPose-t checkpoint. A state-dict
match alone does not establish arithmetic equality, so this runs real public
person crops through both implementations before any student optimization.
"""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

import numpy as np
import onnxruntime as ort
import torch

from data import (ANNOTATION_SHA256, ANNOTATION_URL, TEACHER_SHA256, TEACHER_URL,
                  crop, download_records, fetch, sha256)


def fetch_reference(url, cache, expected_sha256=None):
    cache = Path(cache)
    url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    payload = fetch(url, cache / f"independent-reference-{url_key}.payload", expected_sha256)
    path = cache / "independent-reference.onnx"
    if zipfile.is_zipfile(payload):
        with zipfile.ZipFile(payload) as archive:
            entries = [entry for entry in archive.infolist()
                       if entry.filename.endswith(".onnx") and not entry.is_dir()]
            if len(entries) != 1:
                raise ValueError(f"Expected one ONNX in reference archive, found {len(entries)}")
            if entries[0].file_size > 200_000_000:
                raise ValueError("Unexpectedly large tiny-model reference")
            # Copy this single member into a fixed cache filename. Never extract
            # archive paths or execute anything included alongside the model.
            with archive.open(entries[0]) as source, open(path, "wb") as target:
                shutil.copyfileobj(source, target)
    else:
        shutil.copyfile(payload, path)
    return path


def compare_reference(teacher, reference_path, images, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(reference_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if len(runtime.get_inputs()) != 1:
        raise ValueError("Reference must have one image input")
    input_info = runtime.get_inputs()[0]
    expected_shape = [1, 3, 256, 192]
    if list(input_info.shape)[1:] != expected_shape[1:]:
        raise ValueError(f"Reference input mismatch: {input_info.shape}")
    absolute_error = 0.0
    squared_error = 0.0
    elements = 0
    coordinates = []
    comparisons = []
    teacher.eval()
    with torch.inference_mode():
        for image in images:
            pytorch_outputs = teacher(torch.from_numpy(image))
            raw = runtime.run(None, {input_info.name: image})
            by_shape = {tuple(output.shape): output for output in raw}
            expected_outputs = [(1, 133, 384), (1, 133, 512)]
            if not all(shape in by_shape for shape in expected_outputs):
                raise ValueError(f"Reference output mismatch: {[array.shape for array in raw]}")
            reference_outputs = [by_shape[shape] for shape in expected_outputs]
            sample_error = 0.0
            axes = []
            for prediction, reference in zip(pytorch_outputs, reference_outputs):
                prediction = prediction.numpy()
                difference = prediction.astype(np.float64) - reference.astype(np.float64)
                max_error = float(np.abs(difference).max())
                absolute_error = max(absolute_error, max_error)
                sample_error = max(sample_error, max_error)
                squared_error += float(np.square(difference).sum())
                elements += difference.size
                axes.append((prediction.argmax(-1) - reference.argmax(-1)) / 2)
            error_xy = np.stack(axes, axis=-1)
            norm = np.linalg.norm(error_xy, axis=-1)
            coordinates.extend(norm.ravel().tolist())
            comparisons.append({"maximum_absolute_logit_error": sample_error,
                                "maximum_keypoint_error_input_px": float(norm.max())})
    coordinate_error = np.asarray(coordinates)
    report = {
        "reference_sha256": sha256(reference_path),
        "public_images": len(images),
        "same_rgb_normalized_input": True,
        "maximum_absolute_logit_error": absolute_error,
        "rmse_logits": float(np.sqrt(squared_error / max(1, elements))),
        "mean_keypoint_error_input_px": float(coordinate_error.mean()),
        "p95_keypoint_error_input_px": float(np.percentile(coordinate_error, 95)),
        "maximum_keypoint_error_input_px": float(coordinate_error.max()),
        "per_image": comparisons,
    }
    # Equal weight implementations should differ only by floating-point order.
    # Coordinate maxima can move at near-tied bins, so do not make an isolated
    # argmax tie an error if the actual output distributions agree numerically.
    report["passed"] = absolute_error <= .002 and report["rmse_logits"] <= .0002
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-onnx-url", required=True)
    parser.add_argument("--reference-onnx-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/compact59"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    from model import load_teacher

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.cache.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    selected_ids = list(split["evaluation_annotation_ids"][:8])
    annotation_path = fetch(ANNOTATION_URL, args.cache / "coco_wholebody_val_v1.0.json",
                             ANNOTATION_SHA256)
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_index = {image["id"]: image for image in dataset["images"]}
    annotations = {record["id"]: record for record in dataset["annotations"]
                   if record["id"] in selected_ids}
    records = [{"image": image_index[annotations[index]["image_id"]],
                "annotation": annotations[index]} for index in selected_ids]
    records = download_records(records, len(records), args.cache)
    images = [crop(record)[0][None] for record in records]
    checkpoint = fetch(TEACHER_URL, args.cache / "dw-tt_ucoco.pth", TEACHER_SHA256)
    teacher = load_teacher(checkpoint)
    reference = fetch_reference(args.reference_onnx_url, args.cache,
                                  args.reference_onnx_sha256)
    report = compare_reference(teacher, reference, images, args.threads)
    report["source_url"] = args.reference_onnx_url
    report["evaluation_annotation_ids"] = selected_ids
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.metrics:
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        metrics["reconstruction_reference"] = report
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Reconstructed teacher differs from independent ONNX")


if __name__ == "__main__":
    main()
