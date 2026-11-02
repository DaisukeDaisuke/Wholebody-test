"""Train, export and measure Compact59 using only public still images.

Run in GitHub Actions. No argument accepts a user video or private media URL.
The final evaluation split is not used for gradient updates or checkpoint choice.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F

from data import (ANNOTATION_SHA256, ANNOTATION_URL, MEAN, STD, TEACHER_SHA256,
                  TEACHER_URL, crop, fetch, prepare, sha256)
from model import GROUPS, SELECTED, load_teacher, make_student, multiply_accumulates


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def feature_cache(teacher, records, augment, seed):
    generator = np.random.default_rng(seed)
    features, targets_x, targets_y, labels = [], [], [], []
    teacher.eval()
    with torch.inference_mode():
        for index, record in enumerate(records):
            variants = [None]
            if augment:
                variants.append([generator.uniform(-.035, .035),
                                 generator.uniform(-.035, .035),
                                 generator.uniform(.85, 1.15)])
            for jitter in variants:
                image, points, _ = crop(record, jitter)
                tensor = torch.from_numpy(image[None])
                feat = teacher.backbone(tensor)
                x, y = teacher.head(feat)
                features.append(feat[0].clone())
                targets_x.append(x[0, SELECTED].clone())
                targets_y.append(y[0, SELECTED].clone())
                labels.append(torch.from_numpy(points))
            if (index + 1) % 64 == 0:
                print(f"Cached public examples {index + 1}/{len(records)}", flush=True)
    return tuple(torch.stack(values) for values in (features, targets_x, targets_y, labels))


def gaussian_targets(points, length, axis, sigma):
    bins = torch.arange(length, dtype=torch.float32)[None, None]
    centers = points[..., axis, None] * 2
    return torch.exp(-.5 * ((bins - centers) / sigma).square())


def distribution_loss(prediction, logits, weight, beta=10):
    target = F.softmax(logits * beta, dim=-1)
    per_joint = F.kl_div(F.log_softmax(prediction * beta, dim=-1), target,
                         reduction="none").sum(-1)
    return (per_joint * weight).sum() / weight.sum().clamp_min(1)


def train_head(student, cached, args):
    features, teacher_x, teacher_y, labels = cached
    student.backbone.eval()
    student.head.train()
    optimizer = torch.optim.AdamW(student.head.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 1)
    curve = []
    for step in range(args.steps):
        indices = torch.randint(len(features), (args.batch_size,), generator=generator)
        feat, tx, ty, target = (array[indices] for array in cached)
        px, py = student.head(feat)
        labelled = (target[..., 2] > 0).float()
        # The original model is not a calibrated visibility classifier. For KD
        # every point gets a small weight; labels receive full weight.
        kd_weights = .2 + .8 * labelled
        kd = (distribution_loss(px, tx, kd_weights) +
              distribution_loss(py, ty, kd_weights)) / 2
        gx = gaussian_targets(target, 384, 0, 4.9)
        gy = gaussian_targets(target, 512, 1, 5.66)
        gt = (distribution_loss(px, gx, labelled) +
              distribution_loss(py, gy, labelled)) / 2
        loss = args.kd_weight * kd + (1 - args.kd_weight) * gt
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.head.parameters(), 1.0)
        scale = .1 + .9 * (1 + math.cos(math.pi * step / args.steps)) / 2
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step == args.steps - 1:
            item = {"step": step + 1, "loss": float(loss), "distillation": float(kd),
                    "label_loss": float(gt), "learning_rate": optimizer.param_groups[0]["lr"]}
            curve.append(item)
            print(json.dumps(item), flush=True)
    student.head.eval()
    return curve


def predict_head(head, features, batch_size=32):
    xs, ys = [], []
    with torch.inference_mode():
        for begin in range(0, len(features), batch_size):
            x, y = head(features[begin:begin + batch_size])
            xs.append(x)
            ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def coordinates(x, y):
    return torch.stack((x.argmax(-1), y.argmax(-1)), dim=-1).float() / 2


def accuracy(outputs, labels):
    points = coordinates(*outputs)
    error = torch.linalg.vector_norm(points - labels[..., :2], dim=-1)
    groups = {"all": [0, 59], **GROUPS, "hands": [19, 49]}
    result = {}
    for name, (begin, end) in groups.items():
        group = {}
        for visibility_name, visibility in (("labelled", labels[..., 2] > 0),
                                             ("visible", labels[..., 2] == 2)):
            mask = visibility[:, begin:end]
            distances = error[:, begin:end][mask]
            count = len(distances)
            group[visibility_name] = {
                "points": count,
                "mean_error_input_px": float(distances.mean()) if count else None,
                "pck_0025": float((distances <= 256 * .025).float().mean()) if count else None,
                "pck_005": float((distances <= 256 * .05).float().mean()) if count else None,
                "pck_010": float((distances <= 256 * .10).float().mean()) if count else None,
            }
        result[name] = group
    return result


def agreement(student_outputs, teacher_outputs):
    error = torch.linalg.vector_norm(coordinates(*student_outputs) - coordinates(*teacher_outputs), dim=-1)
    return {
        "mean_error_input_px": float(error.mean()),
        "p95_error_input_px": float(torch.quantile(error.flatten(), .95)),
        "within_2_input_px": float((error <= 2).float().mean()),
        "within_5_input_px": float((error <= 5).float().mean()),
        "not_ground_truth_accuracy": True,
    }


def export_model(model, path):
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, 256, 192), str(path),
                          input_names=["input"], output_names=["simcc_x", "simcc_y"],
                          opset_version=17, do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(path))
    graph.doc_string = "CSPNeXt-t backbone and learned Compact59 RTMCC head. RGB ImageNet normalization."
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def benchmark(path, images, threads):
    runtime = session(path, threads)
    input_name = runtime.get_inputs()[0].name
    for i in range(10):
        runtime.run(None, {input_name: images[i % len(images)]})
    timings = []
    for i in range(120):
        start = time.perf_counter()
        runtime.run(None, {input_name: images[i % len(images)]})
        timings.append((time.perf_counter() - start) * 1000)
    return {"provider": "CPUExecutionProvider", "threads": threads,
            "warmup": 10, "iterations": len(timings),
            "median_ms": float(np.median(timings)), "mean_ms": float(np.mean(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "decode_and_render_included": False, "gpu_latency_measured": False}


def export_parity(model, path, images, threads):
    runtime = session(path, threads)
    worst_abs = 0
    worst_coordinate = 0
    with torch.inference_mode():
        for image in images:
            expected = model(torch.from_numpy(image))
            observed = runtime.run(None, {"input": image})
            worst_abs = max(worst_abs, *(float(np.max(np.abs(a.numpy() - b)))
                                         for a, b in zip(expected, observed)))
            observed_xy = coordinates(*(torch.from_numpy(value) for value in observed))
            worst_coordinate = max(worst_coordinate,
                                   float((coordinates(*expected) - observed_xy).abs().max()))
    if worst_abs > .01:
        raise RuntimeError(f"ONNX export differs from PyTorch: {worst_abs}")
    return {"images": len(images), "maximum_absolute_logit_error": worst_abs,
            "maximum_coordinate_axis_error_input_px": worst_coordinate,
            "comparison": "exported ONNX versus same trained PyTorch model"}


def keypoint_names():
    body = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle"]
    feet = [f"{side}_{part}" for side in ("left", "right")
            for part in ("big_toe", "small_toe", "heel")]
    hands = [f"{side}_{finger}_{joint}" for side in ("left", "right")
             for finger in ("thumb", "index", "middle", "ring", "little")
             for joint in ("base", "middle", "tip")]
    face = ["eye_a_upper", "eye_a_lower", "eye_b_upper", "eye_b_lower",
            "mouth_left", "mouth_right", "mouth_inner_upper", "mouth_inner_lower",
            "cheek_contour_a", "cheek_contour_b"]
    return body + feet + hands + face


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/compact59"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/compact59"))
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--kd-weight", type=float, default=.85)
    args = parser.parse_args()
    if args.train_count < 256 or args.eval_count < 64:
        raise ValueError("This experiment requires at least 256 train and 64 independent evaluation images")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    begin = time.perf_counter()
    print("Loading hash-verified public DWPose-t checkpoint", flush=True)
    checkpoint = fetch(TEACHER_URL, args.cache / "dw-tt_ucoco.pth", TEACHER_SHA256)
    teacher = load_teacher(checkpoint)
    student = make_student(teacher)
    initial = copy.deepcopy(student.head)
    train, evaluation = prepare(args.cache, args.train_count, args.eval_count, args.seed)
    write_json(args.output / "public-split.json", {
        "protocol": "custom image-disjoint split of COCO val2017; not official full-val AP",
        "seed": args.seed,
        "train_image_ids": [record["image"]["id"] for record in train],
        "evaluation_image_ids": [record["image"]["id"] for record in evaluation],
        "train_annotation_ids": [record["annotation"]["id"] for record in train],
        "evaluation_annotation_ids": [record["annotation"]["id"] for record in evaluation],
        "annotation_url": ANNOTATION_URL,
        "annotation_sha256": ANNOTATION_SHA256,
        "images_or_private_video_in_artifact": False,
    })
    cached_train = feature_cache(teacher, train, True, args.seed)
    cached_eval = feature_cache(teacher, evaluation, False, args.seed)
    features, teacher_x, teacher_y, labels = cached_eval
    baseline_outputs = (teacher_x, teacher_y)
    initial_outputs = predict_head(initial, features)
    curve = train_head(student, cached_train, args)
    distilled_outputs = predict_head(student.head, features)
    write_json(args.output / "training-curve.json", curve)
    output_path = args.output / "compact59.onnx"
    teacher_path = args.cache / "teacher133.onnx"
    export_model(student, output_path)
    export_model(teacher, teacher_path)
    example_images = [crop(record)[0][None] for record in evaluation[:8]]
    measured_teacher = benchmark(teacher_path, example_images, args.threads)
    measured_student = benchmark(output_path, example_images, args.threads)
    teacher_accuracy = accuracy(baseline_outputs, labels)
    student_accuracy = accuracy(distilled_outputs, labels)
    initial_accuracy = accuracy(initial_outputs, labels)
    baseline_pck = teacher_accuracy["all"]["labelled"]["pck_005"]
    trained_pck = student_accuracy["all"]["labelled"]["pck_005"]
    metrics = {
        "schema": "compact-vrc-evaluation-v1", "status": "completed",
        "run_url": (f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                    f"{os.environ['GITHUB_RUN_ID']}" if os.environ.get("GITHUB_RUN_ID") else None),
        "commit": os.environ.get("GITHUB_SHA"),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "logical_cpus": os.cpu_count(), "torch": torch.__version__,
                     "onnxruntime": ort.__version__},
        "data": {"train_images": len(train), "train_crops": len(cached_train[0]),
                 "heldout_images": len(evaluation), "heldout_persons": len(evaluation),
                 "private_videos_used": False, "image_id_overlap": 0,
                 "evaluation_uses_annotation_person_boxes": True,
                 "evaluation_protocol": "custom COCO-WholeBody subset, not COCO AP",
                 "normalization": "PCK distance / 256, in the 192x256 padded person crop",
                 "evaluation_used_for_checkpoint_selection": False},
        "training": {"steps": args.steps, "batch_size": args.batch_size,
                     "seed": args.seed, "learning_rate": args.learning_rate,
                     "distillation_weight": args.kd_weight,
                     "frozen_pretrained_backbone": True, "trainable_module": "59-token RTMCC head",
                     "teacher": "DWPose-t 133, not RTMW3D-X"},
        "teacher133": {"selected59_accuracy": teacher_accuracy,
                       "parameters": sum(p.numel() for p in teacher.parameters()),
                       "multiply_accumulates": multiply_accumulates(teacher),
                       "onnx_bytes": teacher_path.stat().st_size, "latency": measured_teacher},
        "sliced59_before_training": {"accuracy": initial_accuracy,
                                    "teacher_agreement": agreement(initial_outputs, baseline_outputs)},
        "compact59": {"accuracy": student_accuracy,
                      "teacher_agreement": agreement(distilled_outputs, baseline_outputs),
                      "parameters": sum(p.numel() for p in student.parameters()),
                      "trainable_parameters": sum(p.numel() for p in student.head.parameters()),
                      "multiply_accumulates": multiply_accumulates(student),
                      "onnx_bytes": output_path.stat().st_size,
                      "latency": measured_student,
                      "export_parity": export_parity(student, output_path, example_images, args.threads)},
        "acceptance": {"pck005_difference_vs_teacher": trained_pck - baseline_pck,
                       "pck005_within_3_percentage_points": trained_pck >= baseline_pck - .03,
                       "median_latency_ratio": measured_student["median_ms"] / measured_teacher["median_ms"],
                       "gpu_load_or_gpu_latency_measured": False,
                       "private_video_accuracy_measured": False,
                       "complete_vrc_tracking_validated": False},
        "elapsed_seconds": time.perf_counter() - begin,
    }
    write_json(args.output / "metrics.json", metrics)
    manifest = {
        "id": "compact59-dwpose-t-distilled", "format": "onnx", "file": "compact59.onnx",
        "sha256": sha256(output_path), "bytes": output_path.stat().st_size,
        "input": {"name": "input", "shape": [1, 3, 256, 192], "dtype": "float32",
                  "color": "RGB", "mean": MEAN.tolist(), "std": STD.tolist(),
                  "crop": {"source": "person bounding box xyxy", "padding": 1.25,
                           "aspect_width_over_height": .75, "resize": "affine bilinear",
                           "border": "constant RGB 0", "coordinate_mapping":
                           "source_xy = (argmax(simcc)/2)/[192,256]*padded_extent + center - padded_extent/2"}},
        "outputs": [{"name": "simcc_x", "shape": [1, 59, 384]},
                    {"name": "simcc_y", "shape": [1, 59, 512]}],
        "simcc_split_ratio": 2, "coco133_indices": SELECTED,
        "score": {"formula": "min(max(simcc_x), max(simcc_y))",
                  "calibrated_probability": False,
                  "validity": "Low score means uncertain. A high score is not proof of visibility."},
        "keypoints": keypoint_names(), "groups": GROUPS,
        "training": metrics["training"], "evaluation": "metrics.json",
        "pretrained_checkpoint": {"url": TEACHER_URL, "sha256": TEACHER_SHA256,
                                   "license": "Apache-2.0"},
        "training_annotations": {"source": "COCO-WholeBody", "url": ANNOTATION_URL,
                                 "license": "CC BY-NC 4.0; research and noncommercial use"},
        "limitations": ["2D landmarks; no measured metric depth from a single uncalibrated camera",
                        "Cheek contour is a geometry cue, not a learned cheek-puff label",
                        "Sole versus dorsum needs the separate foot-state model",
                        "Unobserved landmarks are estimates, not measurements",
                        "GitHub Actions CPU benchmark is not a user-GPU measurement"],
    }
    write_json(args.output / "manifest.json", manifest)
    torch.save({"state_dict": student.state_dict(), "manifest": manifest},
               args.output / "compact59.pth")
    print(json.dumps({"completed": True, "output": str(args.output),
                      "acceptance": metrics["acceptance"]}), flush=True)


if __name__ == "__main__":
    main()
