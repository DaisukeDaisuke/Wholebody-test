"""Public-data INT8 calibration of the original, higher-quality Compact59.

Run only in Actions or Colab. The learned student's public split is mandatory:
calibration cannot consume its validation/test images. Candidates and their
policy are fixed in source; validation selects once, and test can only veto
deployment. No images, activations, or per-image predictions are exported.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch

from data import ANNOTATION_SHA256, ANNOTATION_URL, download_records, fetch, sha256
from train import benchmark, session, write_json
from train_fast import (accuracy, configure_runtime, image_examples, normalized,
                        public_crop, quality_gates, selection_score)


CANDIDATES = (
    {"id": "compact59-int8-all-conv", "quantize_head": True},
    {"id": "compact59-int8-backbone", "quantize_head": False},
)


class PublicCalibrationReader(CalibrationDataReader):
    def __init__(self, records):
        self.records = records
        self.rewind()

    def get_next(self):
        record = next(self.iterator, None)
        if record is None:
            return None
        pixels, _ = public_crop(record)
        return {"input": normalized(pixels).unsqueeze(0).numpy()}

    def rewind(self):
        self.iterator = iter(self.records)


def shared_records(args):
    split = json.loads(args.shared_split.read_text(encoding="utf-8"))
    if split.get("annotation_sha256") != ANNOTATION_SHA256:
        raise ValueError("Shared split does not use the pinned public COCO-WholeBody annotation")
    if bool(split.get("smoke_test_only", False)) != args.smoke_test:
        raise ValueError("Production and smoke split scopes must not be mixed")
    ids = {name: set(split[name + "_image_ids"]) for name in ("train", "validation", "test")}
    if any(ids[a] & ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Shared train/validation/test images overlap")
    earlier = set(split.get("previous_experiment_excluded_from_validation_and_test", []))
    if not earlier or (ids["validation"] | ids["test"]) & earlier:
        raise ValueError("Shared split does not seal validation/test against earlier experiments")
    minimum = {"train": 8, "validation": 8, "test": 8} if args.smoke_test else {
        "train": 512, "validation": 64, "test": 128}
    if any(len(ids[name]) < count for name, count in minimum.items()):
        raise ValueError("Shared split is too small for the requested execution scope")
    if not args.smoke_test and args.calibration_count < 128:
        raise ValueError("Production quantization requires at least 128 calibration images")
    if args.calibration_count < 8 or args.calibration_count > len(ids["train"]):
        raise ValueError("Calibration count must fit the training split and contain at least 8 images")
    annotation = fetch(ANNOTATION_URL, args.cache / "coco_wholebody_val_v1.0.json", ANNOTATION_SHA256)
    dataset = json.loads(annotation.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in dataset["images"]}
    annotations = {record["id"]: record for record in dataset["annotations"]}

    def records(name):
        image_ids, annotation_ids = split[name + "_image_ids"], split[name + "_annotation_ids"]
        if len(image_ids) != len(annotation_ids) or len(image_ids) != len(ids[name]):
            raise ValueError("Shared split has duplicate images or mismatched annotation IDs")
        result = []
        for image_id, annotation_id in zip(image_ids, annotation_ids):
            record = annotations[annotation_id]
            if record["image_id"] != image_id:
                raise ValueError("Shared annotation does not belong to its public image")
            result.append({"image": images[image_id], "annotation": record})
        return result

    calibration = download_records(records("train"), args.calibration_count, args.cache)
    validation = download_records(records("validation"), len(ids["validation"]), args.cache)
    # Only IDs and public annotation metadata are held here. Test pixels and
    # inference are delayed until selection.json has been written below.
    return calibration, validation, records("test"), split


def infer_records(path, records, threads):
    runtime = session(path, threads)
    xs, ys, labels = [], [], []
    for record in records:
        pixels, points = public_crop(record)
        outputs = runtime.run(None, {"input": normalized(pixels).unsqueeze(0).numpy()})
        if len(outputs) != 2 or outputs[0].shape != (1, 59, 384) or outputs[1].shape != (1, 59, 512):
            raise ValueError("Quantized model no longer satisfies the Compact59 output contract")
        if not all(np.isfinite(output).all() for output in outputs):
            raise ValueError("Quantized model emitted a non-finite real-image output")
        xs.append(torch.from_numpy(outputs[0][0]))
        ys.append(torch.from_numpy(outputs[1][0]))
        labels.append(torch.from_numpy(points))
    return (torch.stack(xs), torch.stack(ys)), torch.stack(labels)


def compare_preprocessing(original, processed, examples, threads):
    reference, observed = session(original, threads), session(processed, threads)
    maximum = 0.
    for image in examples:
        before = reference.run(None, {"input": image})
        after = observed.run(None, {"input": image})
        for left, right in zip(before, after):
            if not np.isfinite(right).all():
                raise ValueError("Quantization preprocessing produced non-finite outputs")
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    if maximum > 1e-4:
        raise ValueError("Quantization preprocessing changed the FP32 model outputs")
    return {"images": len(examples), "maximum_absolute_logit_error": maximum, "passed": True}


def measured_summary(path, records, threads, environment, reference):
    outputs, labels = infer_records(path, records, threads)
    latency = benchmark(path, image_examples(records, 192, 256), threads)
    latency["execution_environment"] = environment
    return {"accuracy": accuracy(outputs, labels), "latency": latency,
            "onnx_bytes": path.stat().st_size, "sha256": sha256(path),
            "input_shape": [1, 3, 256, 192], "decode_wh": [192, 256],
            "parameters": reference["parameters"],
            "multiply_accumulates": reference["multiply_accumulates"],
            "mac_count_is_unchanged_by_quantization": True,
            "gpu_latency_measured": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/int8"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/compact59"))
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--shared-split", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--execution-environment", choices=("actions", "colab", "auto"), default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    args.seed, args.amp = 20261105, False
    _, runtime = configure_runtime(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    candidate_root = args.output / "candidates"
    candidate_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    baseline_metrics = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    if sha256(args.baseline_model) != baseline_manifest["sha256"]:
        raise ValueError("Baseline ONNX differs from its pinned manifest")
    if baseline_manifest["input"]["shape"] != [1, 3, 256, 192]:
        raise ValueError("Quantization requires the original, standard-resolution Compact59")
    if baseline_metrics.get("status") != "completed" or baseline_metrics["data"]["private_videos_used"] is not False:
        raise ValueError("Only the approved public-data baseline may be calibrated")
    reference = baseline_metrics["compact59"]
    calibration, validation, test_pool, shared_split = shared_records(args)
    environment = runtime["execution_environment"]
    baseline_validation = measured_summary(args.baseline_model, validation, args.threads, environment, reference)
    preprocessed = args.cache / "compact59-quantization-preprocessed.onnx"
    # Preserve exported node names and arithmetic. BN was already folded at
    # export; shape inference is sufficient to annotate the fixed-shape graph.
    quant_pre_process(input_model=str(args.baseline_model), output_model_path=str(preprocessed),
                      skip_optimization=True, skip_symbolic_shape=True, skip_onnx_shape=False)
    preprocessed_graph = onnx.load(str(preprocessed))
    preprocessed_graph.ir_version = min(preprocessed_graph.ir_version, 10)
    onnx.checker.check_model(preprocessed_graph)
    onnx.save(preprocessed_graph, str(preprocessed))
    preprocessing_parity = compare_preprocessing(args.baseline_model, preprocessed,
                                                  image_examples(validation, 192, 256), args.threads)
    convolution_names = [node.name for node in preprocessed_graph.graph.node if node.op_type == "Conv"]
    head_names = [name for name in convolution_names if "/head/" in name or name.startswith("head/")]
    if not convolution_names or not all(convolution_names) or not head_names:
        raise ValueError("Cannot safely distinguish named backbone and head convolutions")
    policy = {"candidates_fixed_before_validation": list(CANDIDATES),
              "calibration_data": "shared training split only",
              "allowed_drop_vs_previous_compact59": {"all": .015, "body": .015, "hands": .025, "feet": .025},
              "choice": "minimum CPU median among validation quality/speed-eligible candidates",
              "fallback": "best validation score remains experimental; keep standard deployment",
              "test_used_for_selection": False, "test_can_only_veto_deployment": True,
              "cpu_timing_environment": environment}
    write_json(args.output / "selection-policy.json", policy)
    candidates = []
    for config in CANDIDATES:
        names = convolution_names if config["quantize_head"] else [name for name in convolution_names if name not in head_names]
        output = candidate_root / (config["id"] + ".onnx")
        quantize_static(
            str(preprocessed), str(output), PublicCalibrationReader(calibration),
            quant_format=QuantFormat.QDQ, op_types_to_quantize=["Conv"],
            nodes_to_quantize=names, per_channel=True, reduce_range=False,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax, calibration_providers=["CPUExecutionProvider"],
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
        graph = onnx.load(str(output))
        graph.ir_version = min(graph.ir_version, 10)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(output))
        summary = measured_summary(output, validation, args.threads, environment, reference)
        gates = quality_gates(summary["accuracy"], baseline_validation["accuracy"],
                              policy["allowed_drop_vs_previous_compact59"])
        gates["cpu_median_faster_than_previous_compact59"] = (
            summary["latency"]["median_ms"] < baseline_validation["latency"]["median_ms"])
        gates["production_dataset"] = not args.smoke_test
        record = {"config": config, "quantized_convolutions": len(names),
                  "validation": summary, "validation_gates": gates,
                  "eligible": all(gates.values()), "selection_score": selection_score(summary["accuracy"])}
        candidates.append(record)
        write_json(args.output / "validation-candidates.json", candidates)
        print(json.dumps({"quantized_candidate": record}), flush=True)
    eligible = [record for record in candidates if record["eligible"]]
    chosen = min(eligible, key=lambda record: record["validation"]["latency"]["median_ms"]) if eligible else max(
        candidates, key=lambda record: record["selection_score"])
    selection = {"policy": policy, "selected": chosen["config"], "validation_gates_passed": chosen["eligible"],
                 "test_not_yet_opened_in_quantization": True, "test_driven_reselection": False,
                 "previous_compact59_validation": baseline_validation,
                 "shared_student_split_sha256": sha256(args.shared_split)}
    write_json(args.output / "selection.json", selection)
    print(json.dumps({"quantized_selection_fixed_before_test": selection}), flush=True)
    final_model = args.output / "compact59-int8.onnx"
    shutil.copyfile(candidate_root / (chosen["config"]["id"] + ".onnx"), final_model)
    test = download_records(test_pool, len(test_pool), args.cache)
    baseline_test = measured_summary(args.baseline_model, test, args.threads, environment, reference)
    quantized_test = measured_summary(final_model, test, args.threads, environment, reference)
    test_gates = quality_gates(quantized_test["accuracy"], baseline_test["accuracy"],
                              policy["allowed_drop_vs_previous_compact59"])
    production_eligible = chosen["eligible"] and all(test_gates.values()) and not args.smoke_test
    calibration_metadata = {"algorithm": "static MinMax", "format": "QDQ", "operator_types": ["Conv"],
                            "activation_type": "QUInt8 asymmetric", "weight_type": "QInt8 symmetric per channel",
                            "head_quantized": chosen["config"]["quantize_head"],
                            "quantized_convolutions": chosen["quantized_convolutions"],
                            "images": len(calibration), "training_split_only": True,
                            "baseline_model_sha256": baseline_manifest["sha256"],
                            "shared_student_split_sha256": sha256(args.shared_split),
                            "onnxruntime": ort.__version__, "runtime": runtime}
    acceptance = {"validation_gates_passed": chosen["eligible"], "heldout_quality_gates": test_gates,
                  "heldout_quality_gates_passed": all(test_gates.values()),
                  "test_can_only_veto_deployment": True, "test_driven_candidate_reselection": False,
                  "production_eligible": production_eligible,
                  "recommended_deployment": "int8_cpu" if production_eligible else "standard",
                  "pck005_difference_vs_previous_compact59": quantized_test["accuracy"]["all"]["labelled"]["pck_005"] - baseline_test["accuracy"]["all"]["labelled"]["pck_005"],
                  "median_cpu_latency_ratio": quantized_test["latency"]["median_ms"] / baseline_test["latency"]["median_ms"],
                  "onnx_bytes_ratio": final_model.stat().st_size / args.baseline_model.stat().st_size,
                  "macs_ratio": 1.0, "gpu_load_or_gpu_latency_measured": False,
                  "browser_qdq_support_validated": False,
                  "colab_cpu_results_require_actions_confirmation": environment == "colab"}
    run_url = (f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
               f"{os.environ['GITHUB_RUN_ID']}" if os.environ.get("GITHUB_RUN_ID") else None)
    source_commit = os.environ.get("GITHUB_SHA") or os.environ.get("VRC_SOURCE_COMMIT")
    metrics = {"schema": "compact59-static-quantization-v1", "status": "smoke_completed" if args.smoke_test else "completed",
               "run_url": run_url, "commit": source_commit,
               "hardware": {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
                            "python": platform.python_version(), "onnx": onnx.__version__, **runtime},
               "data": {"calibration_images": len(calibration), "validation_images": len(validation),
                        "heldout_images": len(test), "private_videos_used": False,
                        "calibration_validation_test_image_overlap": 0,
                        "same_holdout_as_gpu_student": True, "smoke_test_only": args.smoke_test,
                        "evaluation_uses_annotation_person_boxes": True,
                        "accuracy_backend": "ONNX Runtime CPUExecutionProvider"},
               "calibration": calibration_metadata, "preprocessing_parity": preprocessing_parity,
               "selection": selection, "validation_candidates": candidates,
               "previous_compact59": baseline_test, "int8": quantized_test,
               "acceptance": acceptance, "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output / "metrics.json", metrics)
    exported_split = copy.deepcopy(shared_split)
    exported_split["calibration_image_ids"] = [record["image"]["id"] for record in calibration]
    exported_split["calibration_annotation_ids"] = [record["annotation"]["id"] for record in calibration]
    exported_split["shared_student_split_sha256"] = sha256(args.shared_split)
    write_json(args.output / "public-split.json", exported_split)
    manifest = copy.deepcopy(baseline_manifest)
    manifest.update({"id": chosen["config"]["id"], "file": final_model.name,
                     "sha256": sha256(final_model), "bytes": final_model.stat().st_size,
                     "decodeWH": [192, 256], "quantization": calibration_metadata,
                     "production_eligible": production_eligible,
                     "recommended_deployment": acceptance["recommended_deployment"],
                     "evaluation": "metrics.json", "source_run": run_url, "source_commit": source_commit})
    manifest["limitations"] = manifest.get("limitations", []) + [
        "Quantization changes numerical precision, not the number of multiply-accumulates",
        "The measured CPU runtime does not establish browser WebGPU or user-device performance",
        "Validation fixes the candidate; the shared final test can only veto deployment",
        "The INT8 QDQ model requires separate actual Node/browser runtime verification"]
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"quantization_completed": True, "acceptance": acceptance,
                      "elapsed_seconds": metrics["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
