"""Actions/Colab lower-resolution transfer experiment with a sealed test.

All inputs are public COCO still images. There is no private-media argument.
Candidate/checkpoint selection uses validation only; the final test is opened
after selection.json is written. Every supplied earlier split, including prior
gradient-update images, is excluded from the new validation and test sets.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F

from data import (ANNOTATION_SHA256, ANNOTATION_URL, MEAN, STD, TEACHER_SHA256,
                  TEACHER_URL, all_points, download_records, fetch, sha256)
from fast_model import CANDIDATES, DECODE_WH, initialize_student, load_baseline, multiply_accumulates
from model import GROUPS, SELECTED, load_teacher
from train import benchmark, distribution_loss, keypoint_names, write_json


def configure_runtime(args):
    """Permit the two requested execution environments, with no local fallback."""
    actions = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    try:
        colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        colab = False
    requested = args.execution_environment
    if requested == "actions" and not actions:
        raise ValueError("The actions execution environment requires a real GitHub Actions runner")
    if requested == "colab" and not colab:
        raise ValueError("The colab execution environment requires the Google Colab runtime")
    if not actions and not colab:
        raise ValueError("Model execution is authorized only in GitHub Actions or Google Colab")
    environment = requested if requested != "auto" else ("actions" if actions else "colab")
    if args.smoke_test and environment != "actions":
        raise ValueError("The non-production smoke path is restricted to GitHub Actions")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    if args.amp and device_name != "cuda":
        raise ValueError("Optional AMP requires CUDA; FP32 is the default")
    if args.threads < 1:
        raise ValueError("A positive CPU thread count is required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device_name == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(device_name)
    runtime = {"execution_environment": environment, "training_device": device_name,
               "training_precision": "mixed_float16" if args.amp else "float32",
               "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
               "gpu_name": torch.cuda.get_device_name(device) if device_name == "cuda" else None,
               "cpu_benchmark_environment": environment + " CPU / ONNX Runtime",
               "deterministic_cudnn": True, "tf32_enabled": False,
               "smoke_test": args.smoke_test}
    print(json.dumps({"runtime": runtime}), flush=True)
    return device, runtime


def quality_gates(measured, reference, tolerances):
    return {group: measured[group]["labelled"]["pck_005"] is not None and
            reference[group]["labelled"]["pck_005"] is not None and
            measured[group]["labelled"]["pck_005"] >=
            reference[group]["labelled"]["pck_005"] - tolerance
            for group, tolerance in tolerances.items()}


def prepare_splits(args):
    annotation_path = fetch(ANNOTATION_URL, args.cache / "coco_wholebody_val_v1.0.json",
                            ANNOTATION_SHA256)
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in dataset["images"]}
    eligible = {}
    for annotation in dataset["annotations"]:
        if annotation.get("iscrowd") or annotation.get("num_keypoints", 0) < 6:
            continue
        _, _, width, height = annotation["bbox"]
        if min(width, height) < 64 or max(width, height) < 128:
            continue
        priority = int(np.count_nonzero(all_points(annotation)[SELECTED, 2]))
        image_id = annotation["image_id"]
        if image_id not in eligible or priority > eligible[image_id][0]:
            eligible[image_id] = (priority, annotation)
    previous_ids = set()
    for split_path in args.exclude_split:
        previous = json.loads(split_path.read_text(encoding="utf-8"))
        for key in ("train_image_ids", "evaluation_image_ids", "validation_image_ids", "test_image_ids",
                    "previous_experiment_excluded_from_validation_and_test"):
            previous_ids.update(previous.get(key, []))
    if not previous_ids:
        raise ValueError("The previous experiment's image IDs are required to seal a fresh test")
    order = sorted(eligible)
    random.Random(args.seed).shuffle(order)
    fresh = [image_id for image_id in order if image_id not in previous_ids]
    val_reserve, test_reserve = args.val_count + 32, args.test_count + 32
    if len(fresh) < val_reserve + test_reserve:
        raise ValueError(f"Only {len(fresh)} eligible images remain outside the previous experiment")
    val_pool = fresh[:val_reserve]
    test_pool = fresh[val_reserve:val_reserve + test_reserve]
    held_out_pool = set(val_pool + test_pool)
    train_pool = [image_id for image_id in order if image_id not in held_out_pool]
    if len(train_pool) < args.train_count:
        raise ValueError(f"Only {len(train_pool)} independent training candidates")

    def records(ids):
        return [{"image": images[image_id], "annotation": eligible[image_id][1]}
                for image_id in ids]

    # Test file downloads are delayed until model selection is complete.
    train = download_records(records(train_pool), args.train_count, args.cache)
    validation = download_records(records(val_pool), args.val_count, args.cache)
    return train, validation, records(test_pool), previous_ids


def public_crop(record, width=192, height=256, jitter=None):
    image = cv2.cvtColor(cv2.imread(record["path"]), cv2.COLOR_BGR2RGB)
    x, y, box_w, box_h = map(float, record["annotation"]["bbox"])
    center = np.array([x + box_w / 2, y + box_h / 2], np.float32)
    extent = np.array([box_w, box_h], np.float32) * 1.25
    if extent[0] > extent[1] * .75:
        extent[1] = extent[0] / .75
    else:
        extent[0] = extent[1] * .75
    if jitter is not None:
        center += np.asarray(jitter[:2], np.float32) * extent
        extent *= jitter[2]
    origin = center - extent / 2
    sx, sy = width / extent[0], height / extent[1]
    matrix = np.array([[sx, 0, -origin[0] * sx], [0, sy, -origin[1] * sy]], np.float32)
    pixels = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # Every model is evaluated in the same canonical 192x256 coordinate frame.
    # This prevents lower input resolution from artificially improving PCK.
    points = all_points(record["annotation"])[SELECTED].copy()
    points[:, :2] = (points[:, :2] - origin) * np.array([192, 256], np.float32) / extent
    valid = ((points[:, 0] >= 0) & (points[:, 0] < 192) &
             (points[:, 1] >= 0) & (points[:, 1] < 256))
    points[~valid, 2] = 0
    return pixels, points


def normalized(pixels):
    image = ((pixels.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(image))


def teacher_cache(teacher, baseline, records, augment, seed):
    generator = np.random.default_rng(seed)
    samples, xs, ys, features, labels, old_xs, old_ys = [], [], [], [], [], [], []
    teacher.eval()
    baseline.eval()
    device = next(teacher.parameters()).device
    with torch.inference_mode():
        for index, record in enumerate(records):
            variants = [None]
            if augment:
                variants.append([generator.uniform(-.04, .04), generator.uniform(-.04, .04),
                                 generator.uniform(.75, 1.20)])
            for jitter in variants:
                pixels, points = public_crop(record, jitter=jitter)
                inputs = normalized(pixels).unsqueeze(0).to(device)
                feature = teacher.backbone(inputs)
                tx, ty = teacher.head(feature)
                samples.append((record, jitter))
                xs.append(tx[0, SELECTED].cpu())
                ys.append(ty[0, SELECTED].cpu())
                features.append(feature[0].to(device="cpu", dtype=torch.float16))
                labels.append(torch.from_numpy(points))
                if not augment:
                    bx, by = baseline(inputs)
                    old_xs.append(bx[0].cpu())
                    old_ys.append(by[0].cpu())
            if (index + 1) % 128 == 0:
                print(json.dumps({"cached_teacher_images": index + 1, "total": len(records),
                                  "training": augment}), flush=True)
    result = {"samples": samples, "teacher_x": torch.stack(xs), "teacher_y": torch.stack(ys),
              "teacher_features": torch.stack(features), "labels": torch.stack(labels)}
    if not augment:
        result["baseline"] = (torch.stack(old_xs), torch.stack(old_ys))
    return result


def student_inputs(cached, config):
    return torch.stack([normalized(public_crop(record, config["width"], config["height"], jitter)[0])
                        for record, jitter in cached["samples"]])


def canonical_coordinates(outputs, width, height):
    x, y = outputs
    points = torch.stack((x.argmax(-1), y.argmax(-1)), dim=-1).float() / 2
    return points * torch.tensor([192 / width, 256 / height])


def accuracy(outputs, labels, width=192, height=256):
    error = torch.linalg.vector_norm(canonical_coordinates(outputs, width, height) - labels[..., :2], dim=-1)
    result = {}
    for name, (begin, end) in {"all": [0, 59], **GROUPS, "hands": [19, 49]}.items():
        result[name] = {}
        for visibility_name, valid in (("labelled", labels[..., 2] > 0),
                                       ("visible", labels[..., 2] == 2)):
            distances = error[:, begin:end][valid[:, begin:end]]
            count = len(distances)
            result[name][visibility_name] = {
                "points": count,
                "mean_error_input_px": float(distances.mean()) if count else None,
                "pck_0025": float((distances <= 6.4).float().mean()) if count else None,
                "pck_005": float((distances <= 12.8).float().mean()) if count else None,
                "pck_010": float((distances <= 25.6).float().mean()) if count else None,
            }
    return result


def selection_score(measured):
    # Face expression is a separate model. Prioritize hands and feet here.
    available = [(weight, measured[group]["labelled"]["pck_005"])
                 for group, weight in (("body", .25), ("hands", .45), ("feet", .30))
                 if measured[group]["labelled"]["pck_005"] is not None]
    return sum(weight * value for weight, value in available) / sum(weight for weight, _ in available)


def predict(model, inputs, batch_size=16):
    xs, ys = [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            x, y = model(inputs[start:start + batch_size].to(device))
            xs.append(x.cpu())
            ys.append(y.cpu())
    return torch.cat(xs), torch.cat(ys)


def gaussian(points, length, axis, scale, sigma):
    bins = torch.arange(length, dtype=torch.float32, device=points.device)[None, None]
    center = points[..., axis, None] * 2 * scale
    return torch.exp(-.5 * ((bins - center) / (sigma * scale)).square())


def train_candidate(model, train, validation, args, config):
    device = next(model.parameters()).device
    inputs = student_inputs(train, config)
    val_inputs = student_inputs(validation, config)
    teacher_x, teacher_y = train["teacher_x"], train["teacher_y"]
    warm_features = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), args.batch_size):
            warm_features.append(model.aligned_features(inputs[start:start + args.batch_size].to(device)).cpu())
    warm_features = torch.cat(warm_features)
    generator = torch.Generator().manual_seed(args.seed + 17)
    curve, validations = [], []
    best_score, best_state, best_step = -math.inf, None, None

    def validate(step):
        nonlocal best_score, best_state, best_step
        measured = accuracy(predict(model, val_inputs, args.batch_size), validation["labels"])
        score = selection_score(measured)
        record = {"step": step, "selection_score": score, "accuracy": measured}
        validations.append(record)
        if score > best_score or (args.smoke_test and step == args.steps):
            best_score, best_step = score, step
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"candidate": config["id"], "validation": record}), flush=True)

    validate(0)
    warm_steps = min(args.warmup_steps, args.steps)
    model.backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_learning_rate,
                                  weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    for step in range(args.steps):
        if step == warm_steps:
            model.backbone.requires_grad_(True)
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.backbone_learning_rate},
                {"params": model.head.parameters(), "lr": args.head_learning_rate * .2},
            ], weight_decay=1e-4)
            del warm_features
        model.train()
        indices = torch.randint(len(inputs), (args.batch_size,), generator=generator)
        labels = train["labels"][indices].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
            if step < warm_steps:
                features = warm_features[indices].to(device)
                feature_loss = torch.zeros((), device=device)
            else:
                features = model.aligned_features(inputs[indices].to(device))
                desired = train["teacher_features"][indices].to(device=device, dtype=torch.float32)
                feature_loss = F.mse_loss(features.float(), desired) / desired.square().mean().clamp_min(.01)
            px, py = model.head(features)
        # Norm/softmax targets and losses stay FP32, including optional AMP runs.
        px, py = px.float(), py.float()
        labelled = (labels[..., 2] > 0).float()
        weights = .2 + .8 * labelled
        kd = (distribution_loss(px, teacher_x[indices].to(device), weights) +
              distribution_loss(py, teacher_y[indices].to(device), weights)) / 2
        gx = gaussian(labels, 384, 0, 1, 4.9)
        gy = gaussian(labels, 512, 1, 1, 5.66)
        ground_truth = (distribution_loss(px, gx, labelled) + distribution_loss(py, gy, labelled)) / 2
        loss = args.kd_weight * kd + (1 - args.kd_weight) * ground_truth + .05 * feature_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite training loss; no result will be approved")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        stage_start = 0 if step < warm_steps else warm_steps
        stage_length = warm_steps if step < warm_steps else args.steps - warm_steps
        decay = .1 + .9 * (1 + math.cos(math.pi * (step - stage_start) / max(1, stage_length))) / 2
        for group_index, group in enumerate(optimizer.param_groups):
            rate = args.head_learning_rate if step < warm_steps else (
                args.backbone_learning_rate if group_index == 0 else args.head_learning_rate * .2)
            group["lr"] = rate * decay
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % 100 == 0:
            record = {"candidate": config["id"], "step": step + 1,
                      "loss": float(loss.detach()), "distillation": float(kd.detach()),
                      "label_loss": float(ground_truth.detach()),
                      "feature_loss": float(feature_loss.detach()),
                      "stage": "head" if step < warm_steps else "full_model"}
            curve.append(record)
            print(json.dumps(record), flush=True)
        if step + 1 == warm_steps or (step + 1) % args.validation_interval == 0 or step + 1 == args.steps:
            validate(step + 1)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    best_validation = next(record for record in validations if record["step"] == best_step)
    return {"config": config, "selected_step": best_step, "selection_score": best_score,
            "validation_accuracy": best_validation["accuracy"], "validation_history": validations,
            "training_curve": curve}


def export_model(model, path, width, height):
    model.cpu().eval()
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, height, width), str(path),
                          input_names=["input"], output_names=["simcc_x", "simcc_y"],
                          opset_version=17, do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(path))
    # These fixed FP32 opset-17 graphs need no IR-11 features. Keep the artifact
    # consumable by the deployed ORT 1.22/Web runtime even on newer Colab ONNX.
    graph.ir_version = min(graph.ir_version, 10)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def export_parity(model, path, images, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    worst = 0.
    worst_bin = 0
    with torch.inference_mode():
        for image in images:
            expected = model(torch.from_numpy(image))
            observed = runtime.run(None, {"input": image})
            for pytorch, exported in zip(expected, observed):
                pytorch = pytorch.numpy()
                if not np.isfinite(pytorch).all() or not np.isfinite(exported).all():
                    raise ValueError("Non-finite output in the real-image export parity check")
                if pytorch.shape != exported.shape:
                    raise ValueError("Export output shape differs from the learned model")
                worst = max(worst, float(np.abs(pytorch - exported).max()))
                worst_bin = max(worst_bin, int(np.abs(pytorch.argmax(-1) - exported.argmax(-1)).max()))
    if worst > .01:
        raise ValueError(f"Student ONNX differs from PyTorch: {worst}")
    return {"images": len(images), "maximum_absolute_logit_error": worst,
            "maximum_argmax_bin_error": worst_bin, "passed": True}


def image_examples(records, width, height):
    return [normalized(public_crop(record, width, height)[0]).unsqueeze(0).numpy()
            for record in records[:8]]


def gpu_forward_benchmark(model, images):
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None
    tensors = [torch.from_numpy(image).to(device) for image in images]
    timings = []
    model.eval()
    with torch.inference_mode():
        for index in range(10):
            model(tensors[index % len(tensors)])
        torch.cuda.synchronize(device)
        for index in range(120):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            model(tensors[index % len(tensors)])
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end))
    return {"provider": "PyTorch CUDA", "device": torch.cuda.get_device_name(device),
            "precision": "float32", "batch_size": 1, "warmup": 10, "iterations": len(timings),
            "median_ms": float(np.median(timings)), "mean_ms": float(np.mean(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "clock": "CUDA events",
            "host_transfer_decode_render_included": False, "gpu_utilization_measured": False,
            "browser_webgpu_measurement": False}


def model_summary(model, path, outputs, labels, width, height, images, threads, environment):
    latency = benchmark(path, images, threads)
    latency["execution_environment"] = environment
    return {"accuracy": accuracy(outputs, labels),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "multiply_accumulates": multiply_accumulates(model, width, height),
            "onnx_bytes": path.stat().st_size,
            "input_shape": [1, 3, height, width],
            "decode_wh": list(DECODE_WH), "latency": latency}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/fast59"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/compact59"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--exclude-split", type=Path, action="append", required=True,
                        help="Repeat for every earlier experiment, including the failed fast-v1 split")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--execution-environment", choices=("auto", "actions", "colab"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Optional CUDA mixed precision; default is FP32")
    parser.add_argument("--smoke-test", action="store_true", help="Actions-only non-production execution check")
    parser.add_argument("--train-count", type=int, default=768)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=192)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--kd-weight", type=float, default=.9)
    args = parser.parse_args()
    if not args.smoke_test and (args.train_count < 512 or args.val_count < 64 or args.test_count < 128):
        raise ValueError("Require at least 512 train, 64 validation and 128 independent test images")
    if args.smoke_test and (min(args.train_count, args.val_count, args.test_count) < 8 or args.steps > 8):
        raise ValueError("Smoke execution requires at least 8 images per split and at most 8 steps")
    if args.steps <= args.warmup_steps or args.warmup_steps < 0 or args.validation_interval < 1 or args.batch_size < 1:
        raise ValueError("Full-model fine-tuning and a positive validation interval are required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    candidate_root = args.output / "candidates"
    candidate_root.mkdir(exist_ok=True)
    device, runtime = configure_runtime(args)
    started = time.perf_counter()
    baseline, baseline_manifest = load_baseline(args.baseline_checkpoint)
    baseline.to(device)
    checkpoint = fetch(TEACHER_URL, args.cache / "dw-tt_ucoco.pth", TEACHER_SHA256)
    teacher = load_teacher(checkpoint).to(device)
    train_records, val_records, test_pool, previous_ids = prepare_splits(args)
    train = teacher_cache(teacher, baseline, train_records, True, args.seed)
    validation = teacher_cache(teacher, baseline, val_records, False, args.seed)
    teacher_val = accuracy((validation["teacher_x"], validation["teacher_y"]), validation["labels"])
    baseline_val = accuracy(validation["baseline"], validation["labels"])
    original_macs = multiply_accumulates(teacher, 192, 256)
    baseline_macs = multiply_accumulates(baseline, 192, 256)
    baseline_path = args.cache / "previous-compact59.onnx"
    export_model(baseline, baseline_path, 192, 256)
    baseline_validation_latency = benchmark(baseline_path, image_examples(val_records, 192, 256), args.threads)
    baseline_validation_latency["execution_environment"] = runtime["execution_environment"]
    baseline.to(device)
    policy = {"selection_data": "validation only", "checkpoint_score":
              "0.25*body_PCK005 + 0.45*hands_PCK005 + 0.30*feet_PCK005",
              "allowed_drop_vs_previous_compact59": {"all": .03, "body": .03, "hands": .04, "feet": .05},
              "mac_reduction_required_vs_previous_compact59": True,
              "cpu_latency_improvement_required": True,
              "choice": "lowest FP32 ORT CPU validation median among quality-eligible candidates; otherwise experimental best validation score",
              "cpu_timing_environment": runtime["execution_environment"],
              "test_used_for_selection": False,
              "smoke_override": "export final updated checkpoint; never production eligible" if args.smoke_test else None}
    write_json(args.output / "selection-policy.json", policy)
    candidates, curves = [], []
    for config in CANDIDATES:
        torch.manual_seed(args.seed)
        model = initialize_student(baseline, config).to(device)
        record = train_candidate(model, train, validation, args, config)
        record["multiply_accumulates"] = multiply_accumulates(model, config["width"], config["height"])
        record["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        candidate_onnx = candidate_root / (config["id"] + ".onnx")
        export_model(model, candidate_onnx, config["width"], config["height"])
        validation_images = image_examples(val_records, config["width"], config["height"])
        record["validation_export_parity"] = export_parity(model, candidate_onnx, validation_images, args.threads)
        record["validation_cpu_latency"] = benchmark(candidate_onnx, validation_images, args.threads)
        record["validation_cpu_latency"]["execution_environment"] = runtime["execution_environment"]
        gates = quality_gates(record["validation_accuracy"], baseline_val,
                              policy["allowed_drop_vs_previous_compact59"])
        gates["macs_below_previous_compact59"] = record["multiply_accumulates"] < baseline_macs
        gates["cpu_median_faster_than_previous_compact59"] = (
            record["validation_cpu_latency"]["median_ms"] < baseline_validation_latency["median_ms"])
        gates["production_dataset"] = not args.smoke_test
        record["validation_gates"] = gates
        record["eligible"] = all(gates.values())
        record["macs_ratio_vs_previous_compact59"] = record["multiply_accumulates"] / baseline_macs
        curves.extend(record.pop("training_curve"))
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "architecture": "physical-grid-aligned-original-head-v2"},
                   candidate_root / (config["id"] + ".pth"))
        candidates.append(record)
        write_json(args.output / "validation-candidates.json", candidates)
        write_json(args.output / "training-curve.json", curves)
        del model
    eligible = [record for record in candidates if record["eligible"]]
    chosen = min(eligible, key=lambda record: record["validation_cpu_latency"]["median_ms"]) if eligible else max(
        candidates, key=lambda record: record["selection_score"])
    selection = {"policy": policy, "selected": chosen["config"], "selected_step": chosen["selected_step"],
                 "validation_gates_passed": chosen["eligible"], "test_not_yet_opened": True,
                 "teacher_validation_accuracy": teacher_val, "previous_compact59_validation_accuracy": baseline_val,
                 "previous_compact59_validation_cpu_latency": baseline_validation_latency,
                 "eligible_before_test": bool(eligible) and not args.smoke_test,
                 "recommended_deployment_before_test": "fast" if eligible and not args.smoke_test else "standard"}
    write_json(args.output / "selection.json", selection)
    print(json.dumps({"selection_fixed_before_test": selection}), flush=True)
    # No candidate or checkpoint changes are permitted after this point.
    config = chosen["config"]
    selected_checkpoint = torch.load(candidate_root / (config["id"] + ".pth"),
                                     map_location="cpu", weights_only=True)
    model = initialize_student(baseline, config).to(device)
    model.load_state_dict(selected_checkpoint["state_dict"], strict=True)
    model.eval()
    test_records = download_records(test_pool, args.test_count, args.cache)
    split_ids = [{record["image"]["id"] for record in records}
                 for records in (train_records, val_records, test_records)]
    if any(split_ids[a] & split_ids[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Three-way image leakage")
    if (split_ids[1] | split_ids[2]) & previous_ids:
        raise ValueError("The previous experiment overlaps validation/test")
    test = teacher_cache(teacher, baseline, test_records, False, args.seed)
    test_inputs = student_inputs(test, config)
    selected_outputs = predict(model, test_inputs, args.batch_size)
    model_path = args.output / "fast59.onnx"
    teacher_path = args.cache / "teacher133.onnx"
    full_images = image_examples(test_records, 192, 256)
    small_images = image_examples(test_records, config["width"], config["height"])
    gpu_timings = {"teacher133": gpu_forward_benchmark(teacher, full_images),
                   "previous_compact59": gpu_forward_benchmark(baseline, full_images),
                   "fast59": gpu_forward_benchmark(model, small_images)}
    export_model(model, model_path, config["width"], config["height"])
    export_model(baseline, baseline_path, 192, 256)
    export_model(teacher, teacher_path, 192, 256)
    teacher_summary = model_summary(teacher, teacher_path, (test["teacher_x"], test["teacher_y"]),
                                    test["labels"], 192, 256, full_images, args.threads, runtime["execution_environment"])
    teacher_summary["selected59_accuracy"] = teacher_summary.pop("accuracy")
    old_summary = model_summary(baseline, baseline_path, test["baseline"], test["labels"],
                                192, 256, full_images, args.threads, runtime["execution_environment"])
    small_summary = model_summary(model, model_path, selected_outputs, test["labels"],
                                  config["width"], config["height"], small_images, args.threads, runtime["execution_environment"])
    teacher_summary["gpu_forward_latency"] = gpu_timings["teacher133"]
    old_summary["gpu_forward_latency"] = gpu_timings["previous_compact59"]
    small_summary["gpu_forward_latency"] = gpu_timings["fast59"]
    small_summary["export_parity"] = export_parity(model, model_path, small_images, args.threads)
    small_summary["selected_validation_step"] = chosen["selected_step"]
    selected_pck = small_summary["accuracy"]["all"]["labelled"]["pck_005"]
    teacher_pck = teacher_summary["selected59_accuracy"]["all"]["labelled"]["pck_005"]
    previous_pck = old_summary["accuracy"]["all"]["labelled"]["pck_005"]
    heldout_gates = quality_gates(small_summary["accuracy"], old_summary["accuracy"],
                                 policy["allowed_drop_vs_previous_compact59"])
    # The sealed test can veto deployment of the fixed model. It cannot select
    # another resolution/checkpoint, and selection.json remains unchanged.
    production_eligible = chosen["eligible"] and all(heldout_gates.values()) and not args.smoke_test
    run_url = (f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
               f"{os.environ['GITHUB_RUN_ID']}" if os.environ.get("GITHUB_RUN_ID") else None)
    training = {"teacher": "DWPose-t 133", "initialization": "exact previous Compact59 head and backbone weights; native lower-resolution backbone with physical stride-32 feature alignment",
                "architecture": "physical-grid-aligned-original-head-v2", "head_weights_resized": False,
                "runtime": runtime,
                "steps_per_candidate": args.steps, "warmup_head_steps": args.warmup_steps,
                "full_model_steps": args.steps - args.warmup_steps, "batch_size": args.batch_size,
                "seed": args.seed, "distillation_weight": args.kd_weight,
                "feature_distillation_weight": .05, "batchnorm_statistics_frozen": True,
                "backbone_learning_rate": args.backbone_learning_rate,
                "head_learning_rate": args.head_learning_rate, "candidates": list(CANDIDATES),
                "selected_checkpoint_step": chosen["selected_step"],
                "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint)}
    metrics = {"schema": "fast-vrc-evaluation-v3", "status": "smoke_completed" if args.smoke_test else "completed", "run_url": run_url,
               "commit": os.environ.get("GITHUB_SHA") or os.environ.get("VRC_SOURCE_COMMIT"),
               "hardware": {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
                            "python": platform.python_version(), "torch": torch.__version__,
                            "onnxruntime": ort.__version__, "onnx": onnx.__version__, **runtime},
               "data": {"train_images": len(train_records), "train_crops": len(train["samples"]),
                        "validation_images": len(val_records), "heldout_images": len(test_records),
                        "heldout_persons": len(test_records), "image_id_overlap": 0,
                        "overlap_with_previous_experiment_in_validation_or_test": 0,
                        "private_videos_used": False, "smoke_test_only": args.smoke_test,
                        "evaluation_uses_annotation_person_boxes": True,
                        "normalization": "All models mapped to the same 192x256 crop; PCK distance / 256",
                        "evaluation_protocol": "fresh image-disjoint train/validation/test subsets of COCO val2017, not COCO AP",
                        "evaluation_used_for_checkpoint_selection": False,
                        "accuracy_backend": "PyTorch " + str(device),
                        "onnx_export_checked_on_public_images": True},
               "training": training, "selection": selection, "validation_candidates": candidates,
               "teacher133": teacher_summary, "previous_compact59": old_summary,
               "fast59": small_summary, "compact59": small_summary,
               "acceptance": {"validation_gates_passed": chosen["eligible"],
                              "heldout_quality_gates": heldout_gates,
                              "heldout_quality_gates_passed": all(heldout_gates.values()),
                              "test_can_only_veto_deployment": True,
                              "test_driven_candidate_reselection": False,
                              "recommended_deployment": "fast" if production_eligible else "standard",
                              "pck005_difference_vs_teacher": selected_pck - teacher_pck,
                              "pck005_difference_vs_previous_compact59": selected_pck - previous_pck,
                              "pck005_within_3_percentage_points": selected_pck >= teacher_pck - .03,
                              "macs_ratio_vs_teacher133": small_summary["multiply_accumulates"] / original_macs,
                              "macs_ratio_vs_previous_compact59": small_summary["multiply_accumulates"] / baseline_macs,
                              "median_latency_ratio": small_summary["latency"]["median_ms"] / teacher_summary["latency"]["median_ms"],
                              "median_latency_ratio_vs_previous_compact59": small_summary["latency"]["median_ms"] / old_summary["latency"]["median_ms"],
                              "gpu_load_or_gpu_latency_measured": gpu_timings["fast59"] is not None,
                              "gpu_utilization_measured": False, "browser_webgpu_latency_measured": False,
                              "colab_cpu_results_require_actions_confirmation": runtime["execution_environment"] == "colab",
                              "production_eligible": production_eligible,
                              "private_video_accuracy_measured": False, "complete_vrc_tracking_validated": False},
               "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output / "metrics.json", metrics)
    split = {"protocol": "fresh image-disjoint train/validation/test; test never used for selection",
             "seed": args.seed, "annotation_url": ANNOTATION_URL, "annotation_sha256": ANNOTATION_SHA256,
             "smoke_test_only": args.smoke_test,
             "excluded_split_sha256": [sha256(path) for path in args.exclude_split],
             "image_url_template": "https://s3.amazonaws.com/images.cocodataset.org/val2017/{image_id:012d}.jpg",
             "previous_experiment_excluded_from_validation_and_test": sorted(previous_ids),
             "images_or_private_video_in_artifact": False}
    for name, records in (("train", train_records), ("validation", val_records), ("test", test_records),
                          ("evaluation", test_records)):
        split[name + "_image_ids"] = [record["image"]["id"] for record in records]
        split[name + "_annotation_ids"] = [record["annotation"]["id"] for record in records]
    write_json(args.output / "public-split.json", split)
    manifest = {"id": config["id"] + "-distilled", "format": "onnx", "file": "fast59.onnx",
                "sha256": sha256(model_path), "bytes": model_path.stat().st_size,
                "decodeWH": list(DECODE_WH),
                "production_eligible": production_eligible,
                "recommended_deployment": "fast" if production_eligible else "standard",
                "input": {"name": "input", "shape": [1, 3, config["height"], config["width"]],
                          "dtype": "float32", "color": "RGB", "mean": MEAN.tolist(), "std": STD.tolist(),
                          "crop": {"source": "person bounding box xywh", "padding": 1.25,
                                   "aspect_width_over_height": .75, "resize": "affine bilinear",
                                   "border": "constant RGB 0", "coordinate_mapping":
                                   "source_xy = (argmax(simcc)/2)/decodeWH*padded_extent + center - padded_extent/2"}},
                "outputs": [{"name": "simcc_x", "shape": [1, 59, 384]},
                            {"name": "simcc_y", "shape": [1, 59, 512]}],
                "simcc_split_ratio": 2, "coco133_indices": SELECTED,
                "score": baseline_manifest["score"], "keypoints": keypoint_names(), "groups": GROUPS,
                "training": training, "evaluation": "metrics.json", "source_run": run_url,
                "source_commit": metrics["commit"],
                "pretrained_checkpoint": {"url": TEACHER_URL, "sha256": TEACHER_SHA256, "license": "Apache-2.0"},
                "training_annotations": baseline_manifest["training_annotations"],
                "limitations": baseline_manifest["limitations"] + [
                    "Validation chooses resolution/checkpoint; the final test cannot change that choice",
                    "Lower spatial resolution can lose small hand and foot detail",
                    "MAC reduction is architecture evidence, not a measured user-GPU utilization reduction",
                    "Colab CPU latency and PyTorch CUDA event timing are separate from browser or Actions runtime performance",
                    "A smoke-test result is non-production and cannot establish accuracy"]}
    write_json(args.output / "manifest.json", manifest)
    torch.save({"state_dict": model.state_dict(), "config": config, "manifest": manifest,
                "architecture": "physical-grid-aligned-original-head-v2"},
               args.output / "fast59.pth")
    print(json.dumps({"completed": True, "selected": config, "acceptance": metrics["acceptance"],
                      "elapsed_seconds": metrics["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
