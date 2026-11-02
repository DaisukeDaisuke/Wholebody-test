#!/usr/bin/env python3
"""Train a small public-data foot-view model and evaluate upstream PFLD in Actions.

This program never reads videos. It downloads only the pinned public datasets
below. The upstream PFLD weights remain a remote dependency, not a redistributed
artifact. Foot-view accuracy is conditional on a foot crop, not foot detection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


FOOT_URL = "https://zenodo.org/api/records/20457253/files/FootNet_v1.0.zip/content"
FOOT_MD5 = "d3a756d30e2051115c13f044558754a6"
FOOT_BYTES = 556662306
FOOT_CLASSES = ["sole", "dorsum", "side"]
FOOT_LABELS = ["足裏", "足の甲", "側面"]
FACE_REV = "f575be168a24af6f4807c852173fdfedf6d2c67d"
FACE_URL = f"https://raw.githubusercontent.com/cunjian/pytorch_face_landmark/{FACE_REV}/onnx/pfld.onnx"
FACE_GIT_BLOB = "8f9822861a3d18f60672db1410cf7ff5a28012a9"
DATA_REV = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
DATA_ROOT = f"https://raw.githubusercontent.com/open-mmlab/mmpose/{DATA_REV}/tests/data/coco"
FACE_ANNOTATION_URL = "https://huggingface.co/datasets/msdkhairi/coco2017/resolve/main/coco_wholebody_val_v1.0.json"
FACE_ANNOTATION_SHA256 = "f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEED = 17513


def require_actions():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Training, model inference, and evaluation run in GitHub Actions only.")
    if any(os.environ.get(k) for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("Do not pass publication credentials to model preparation.")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path, max_bytes=600 * 1024 * 1024):
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Wholebody-test-compact-parts/1.0"})
            total = 0
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as dst:
                for chunk in iter(lambda: response.read(1048576), b""):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Public download exceeded its declared size bound.")
                    dst.write(chunk)
            if not total:
                raise ValueError("Empty public download.")
            tmp.replace(path)
            return path
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def options(threads):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    opt.inter_op_num_threads = 1
    return opt


def rgb_tensor(image: Image.Image, size=128, normalize=True):
    rgb = np.array(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    if normalize:
        rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)


def square_crop(image: Image.Image, box, padding=1.15):
    x, y, w, h = map(float, box)
    side = max(w, h) * padding
    if not math.isfinite(side) or side < 2:
        raise ValueError("Invalid part crop.")
    cx, cy = x + w / 2, y + h / 2
    # PIL pads out-of-image coordinates with black, matching the browser canvas.
    bounds = (math.floor(cx - side / 2), math.floor(cy - side / 2),
              math.floor(cx - side / 2) + math.ceil(side), math.floor(cy - side / 2) + math.ceil(side))
    return image.crop(bounds), bounds


def dhash(image):
    a = np.array(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def load_footnet(cache):
    archive = fetch(FOOT_URL, cache / "FootNet_v1.0.zip")
    if archive.stat().st_size != FOOT_BYTES or digest(archive, "md5") != FOOT_MD5:
        raise ValueError("FootNet ZIP does not match the pinned Zenodo file.")
    entries = []
    with zipfile.ZipFile(archive) as z:
        manifest = {r["path"]: r for r in csv.DictReader(io.StringIO(z.read("FootNet_v1.0/manifest.csv").decode("utf-8-sig")))}
        names = sorted(name for name in z.namelist() if "/dataset/dataset_" in name and "/images/" in name
                       and Path(name).suffix.lower() in (".png", ".jpg", ".jpeg"))
        for name in names:
            relative = name.removeprefix("FootNet_v1.0/")
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != manifest[relative]["sha256"]:
                raise ValueError("FootNet image integrity mismatch: " + relative)
            source = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
            mask_name = name.replace("/images/", "/masks/").rsplit(".", 1)[0] + ".png"
            mask_bytes = z.read(mask_name)
            mask_relative = mask_name.removeprefix("FootNet_v1.0/")
            if hashlib.sha256(mask_bytes).hexdigest() != manifest[mask_relative]["sha256"]:
                raise ValueError("FootNet mask integrity mismatch.")
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
            bounds = mask.getbbox()
            if bounds is None:
                raise ValueError("Empty foot mask: " + relative)
            # Published redacted images may have different pixel sizes to masks.
            sx, sy = source.width / mask.width, source.height / mask.height
            x0, y0, x1, y1 = bounds
            box = [x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy]
            crop, _ = square_crop(source, box)
            crop = crop.resize((192, 192), Image.Resampling.BILINEAR)
            subset = name.split("/dataset/")[1].split("/")[0]
            label = 0 if "sole" in subset else 1 if "top" in subset else 2 if "side" in subset else None
            if label is None:
                raise ValueError("Unknown foot-view folder: " + subset)
            entries.append({"id": relative, "subset": subset, "label": label, "image": crop,
                            "sha256": hashlib.sha256(content).hexdigest(), "dhash": dhash(crop)})
    if len(entries) != 191:
        raise ValueError(f"Expected the public 191-image release, found {len(entries)} images.")
    return entries


def split_footnet(entries):
    # Union exact and near-duplicate crops before splitting. This cannot establish
    # subject independence because the public release contains no subject IDs.
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(entries):
        for j in range(i):
            b = entries[j]
            if a["sha256"] == b["sha256"] or (a["dhash"] ^ b["dhash"]).bit_count() <= 3:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(entries)):
        groups[find(i)].append(i)
    by_stratum = defaultdict(list)
    for group in groups.values():
        stratum = "|".join(sorted({entries[i]["subset"] for i in group}))
        by_stratum[stratum].append(group)
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(SEED)
    for stratum in sorted(by_stratum):
        these = by_stratum[stratum]
        rng.shuffle(these)
        n = len(these)
        # Singleton cross-view duplicate groups stay in train.
        ntest = max(1, round(n * 0.2)) if n >= 5 else 0
        nval = max(1, round(n * 0.2)) if n >= 5 else 0
        for k, group in enumerate(these):
            split = "test" if k < ntest else "validation" if k < ntest + nval else "train"
            splits[split].extend(group)
    for split, indices in splits.items():
        if set(entries[i]["label"] for i in indices) != {0, 1, 2}:
            raise ValueError("Foot split lacks a class: " + split)
    return splits, len(groups)


def augment(image, rng, index):
    if index == 0:
        return image
    image = image.rotate((index % 4) * 90 + rng.uniform(-20, 20), resample=Image.Resampling.BILINEAR,
                         fillcolor=(0, 0, 0))
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return image


class FootViewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(576, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, 3))
        self.register_buffer("temperature", torch.tensor(1.0))

    def encode(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, images):
        return torch.softmax(self.head(self.encode(images)) / self.temperature, dim=-1)


def cache_features(model, entries, indices, augmentations):
    features, labels = [], []
    rng = random.Random(SEED + len(indices))
    batch, batch_labels = [], []
    with torch.inference_mode():
        for i in indices:
            entry = entries[i]
            for aug in range(augmentations):
                batch.append(rgb_tensor(augment(entry["image"], rng, aug)))
                batch_labels.append(entry["label"])
                if len(batch) == 32:
                    features.append(model.encode(torch.from_numpy(np.stack(batch))).clone())
                    labels.extend(batch_labels)
                    batch, batch_labels = [], []
        if batch:
            features.append(model.encode(torch.from_numpy(np.stack(batch))).clone())
            labels.extend(batch_labels)
    # Turn inference-mode tensors into ordinary tensors before head autograd.
    return torch.cat(features).clone(), torch.tensor(labels, dtype=torch.long)


def foot_metrics(probabilities, labels, threshold):
    pred = probabilities.argmax(1)
    selected = probabilities.max(1) >= threshold
    confusion = [[int(np.sum((labels == a) & (pred == b))) for b in range(3)] for a in range(3)]
    return {"examples": len(labels), "accuracy": float(np.mean(pred == labels)),
            "classOrder": FOOT_CLASSES, "confusionMatrix": confusion,
            "recall": {FOOT_CLASSES[k]: float(np.mean(pred[labels == k] == k)) for k in range(3)},
            "confidenceThreshold": float(threshold), "accepted": int(selected.sum()),
            "coverage": float(selected.mean()),
            "acceptedAccuracy": float(np.mean(pred[selected] == labels[selected])) if selected.any() else None}


def train_foot(args):
    entries = load_footnet(args.cache)
    splits, group_count = split_footnet(entries)
    split_doc = {"seed": SEED, "totalImages": len(entries), "duplicateGroups": group_count,
                 "unit": "image/near-duplicate group, not identified subject",
                 "method": "seeded 60/20/20 groups within anatomical view folder; dHash distance <=3 kept together",
                 "splits": {split: [{"id": entries[i]["id"], "class": FOOT_CLASSES[entries[i]["label"]],
                                      "sha256": entries[i]["sha256"]} for i in ids] for split, ids in splits.items()}}
    write_json(args.report / "foot-split.json", split_doc)
    print(json.dumps({"footSplit": {k: len(v) for k, v in splits.items()}, "duplicateGroups": group_count}), flush=True)
    model = FootViewNet().eval()
    for p in model.features.parameters():
        p.requires_grad_(False)
    data = {k: cache_features(model, entries, v, 8 if k == "train" else 1) for k, v in splits.items()}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.003, weight_decay=0.02)
    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    generator = torch.Generator().manual_seed(SEED)
    xtrain, ytrain = data["train"]
    xval, yval = data["validation"]
    for epoch in range(args.epochs):
        model.head.train()
        perm = torch.randperm(len(xtrain), generator=generator)
        loss_sum = 0.0
        for batch in perm.split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = model.head(xtrain[batch])
            loss = nn.functional.cross_entropy(logits, ytrain[batch], label_smoothing=0.08)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        model.head.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.cross_entropy(model.head(xval), yval))
        if validation_loss < best:
            best, best_epoch, best_state = validation_loss, epoch, copy.deepcopy(model.head.state_dict())
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            item = {"epoch": epoch + 1, "trainLoss": loss_sum / len(xtrain), "validationLoss": validation_loss}
            history.append(item)
            print(json.dumps(item), flush=True)
    model.head.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model.head(xval)
        temperatures = torch.linspace(0.5, 4.0, 71)
        losses = [float(nn.functional.cross_entropy(val_logits / t, yval)) for t in temperatures]
        model.temperature.copy_(temperatures[int(np.argmin(losses))])
        val_prob = torch.softmax(val_logits / model.temperature, -1).numpy()
        test_prob = torch.softmax(model.head(data["test"][0]) / model.temperature, -1).numpy()
    threshold = 0.7
    for candidate in np.arange(0.4, 0.96, 0.025):
        metrics = foot_metrics(val_prob, yval.numpy(), candidate)
        if metrics["accepted"] >= max(8, len(yval) // 3) and metrics["acceptedAccuracy"] >= 0.85:
            threshold = float(candidate)
            break
    target = args.output / "foot-view.onnx"
    sample = rgb_tensor(entries[splits["test"][0]]["image"])[None]
    torch.onnx.export(model, torch.from_numpy(sample), str(target), input_names=["images"],
                      output_names=["probabilities"], opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), sess_options=options(args.threads), providers=["CPUExecutionProvider"])
    max_error = 0.0
    for i in splits["test"][:8]:
        actual = rgb_tensor(entries[i]["image"])[None]
        with torch.no_grad():
            expected = model(torch.from_numpy(actual)).numpy()
        observed = session.run(None, {"images": actual})[0]
        max_error = max(max_error, float(np.max(np.abs(expected - observed))))
    if max_error > 1e-4:
        raise ValueError(f"Foot PyTorch/ONNX probability mismatch: {max_error}")
    for _ in range(8):
        session.run(None, {"images": sample})
    timings = []
    for _ in range(60):
        start = time.perf_counter()
        session.run(None, {"images": sample})
        timings.append((time.perf_counter() - start) * 1000)
    record = {"id": "foot-view-mobilenetv3-small", "kind": "footView", "filename": target.name,
              "url": "./models/parts/" + target.name, "bytes": target.stat().st_size, "sha256": digest(target),
              "inputName": "images", "outputNames": ["probabilities"], "inputShape": [1, 3, 128, 128],
              "inputWH": [128, 128], "outputShape": [1, 3], "color": "RGB", "scale": 1 / 255,
              "mean": MEAN.tolist(), "std": STD.tolist(), "crop": "square known-foot ROI with 1.15 padding",
              "classOrder": FOOT_CLASSES, "classLabels": FOOT_LABELS, "confidenceThreshold": threshold,
              "temperature": float(model.temperature), "parameters": sum(p.numel() for p in model.parameters()),
              "trainableParameters": sum(p.numel() for p in model.head.parameters()),
              "weightOrigin": "TorchVision IMAGENET1K_V1 MobileNetV3-Small features; new 576→96→3 head trained on FootNet",
              "scope": "view classification of a provided foot crop; no foot-presence or generic object detection",
              "license": "new head and integration under repository license; TorchVision BSD-3-Clause; FootNet CC-BY-4.0 attribution",
              "source": "https://zenodo.org/records/20457253"}
    report = {"model": record, "training": {"epochs": args.epochs, "selectedEpoch": best_epoch + 1,
              "backboneFrozen": True, "augmentationCopiesPerTrainImage": 8, "temperatureSource": "validation only",
              "thresholdSource": "validation only; first threshold with >=85% accepted accuracy and >=1/3 coverage, fallback .7",
              "history": history}, "validation": foot_metrics(val_prob, yval.numpy(), threshold),
              "heldout": foot_metrics(test_prob, data["test"][1].numpy(), threshold),
              "onnxMaxAbsoluteProbabilityError": max_error,
              "inference": {"provider": "CPUExecutionProvider", "threads": args.threads,
                            "meanMs": float(np.mean(timings)), "p50Ms": float(np.median(timings)),
                            "p95Ms": float(np.percentile(timings, 95)), "samples": len(timings)},
              "dataset": {"name": "FootNet v1.0", "doi": "10.5281/zenodo.20457253", "images": len(entries),
                          "license": "CC-BY-4.0", "zipMD5": FOOT_MD5, "zipBytes": FOOT_BYTES,
                          "authors": ["Adith Vijay", "Akash Prabhune", "Vinay R Srihari", "Akhila Rayampalli"],
                          "cropSource": "ground-truth segmentation mask bounding box; raw RGB crop retains background"},
              "limits": ["No subject identifiers; image-group holdout does not prove unseen-person accuracy.",
                         "Clinical still-image domain differs from motion video and extreme foot rotations.",
                         "Accuracy assumes a known foot crop; it does not measure automatic foot localization.",
                         "Three anatomical views do not provide toe articulation, ankle rotation, or depth.",
                         "CPU inference timing is not browser WebGPU timing or GPU utilization."]}
    write_json(args.report / "foot-report.json", report)
    return record, report


def face_features(p):
    def distance(a, b):
        return float(np.linalg.norm(p[a] - p[b]))
    def ear(start):
        return (distance(start + 1, start + 5) + distance(start + 2, start + 4)) / max(2 * distance(start, start + 3), 1e-6)
    interocular = max(float(np.linalg.norm(p[36:42].mean(0) - p[42:48].mean(0))), 1e-6)
    return {"eyeLeftEAR": ear(42), "eyeRightEAR": ear(36),
            "mouthOpenRatio": (distance(61, 67) + distance(62, 66) + distance(63, 65)) / max(3 * distance(60, 64), 1e-6),
            "cheekWidthNormalized": distance(3, 13) / interocular}


def prepare_face(args):
    path = fetch(FACE_URL, args.cache / "pfld-upstream.onnx", 4 * 1024 * 1024)
    content = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
    if blob != FACE_GIT_BLOB:
        raise ValueError("PFLD bytes differ from the reviewed upstream Git blob.")
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), sess_options=options(args.threads), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    if len(inp.shape) != 4 or inp.shape[1:] != [3, 112, 112] or (isinstance(inp.shape[0], int) and inp.shape[0] != 1):
        raise ValueError(f"Unexpected PFLD input: {inp.shape}")
    annotation_path = fetch(FACE_ANNOTATION_URL, args.cache / "coco_wholebody_val_v1.0.json", 160 * 1024 * 1024)
    if digest(annotation_path) != FACE_ANNOTATION_SHA256:
        raise ValueError("COCO-WholeBody face evaluation annotations changed.")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data["images"]}
    annotations = [a for a in data["annotations"] if a.get("face_valid") and min(a.get("face_box", [0, 0, 0, 0])[2:]) >= 64
                   and sum(v > 0 for v in a.get("face_kpts", [])[2::3]) >= 55]
    annotations.sort(key=lambda a: (a["image_id"], a["id"]))
    rows, times, errors = [], [], []
    ratio_errors = defaultdict(list)
    for annotation in annotations[:args.face_examples]:
        info = images[annotation["image_id"]]
        source = fetch(f"https://images.cocodataset.org/val2017/{info['id']:012d}.jpg",
                       args.cache / "face-images" / info["file_name"], 8 * 1024 * 1024)
        image = Image.open(source).convert("RGB")
        crop, bounds = square_crop(image, annotation["face_box"], padding=1.1)
        tensor = rgb_tensor(crop, 112, normalize=False)[None]
        started = time.perf_counter()
        output = session.run(None, {inp.name: tensor})[0]
        times.append((time.perf_counter() - started) * 1000)
        if output.size != 136 or not np.isfinite(output).all():
            raise ValueError("PFLD did not produce 68 finite landmarks.")
        prediction = output.reshape(68, 2) * (bounds[2] - bounds[0]) + np.array(bounds[:2])
        target = np.asarray(annotation["face_kpts"], dtype=np.float32).reshape(68, 3)
        valid = target[:, 2] > 0
        eye_distance = float(np.linalg.norm(target[36, :2] - target[45, :2]))
        if eye_distance < 2 or not valid[36] or not valid[45] or not valid.any():
            continue
        normalized = np.linalg.norm(prediction[valid] - target[valid, :2], axis=1) / eye_distance
        errors.extend(normalized.tolist())
        estimated, reference = face_features(prediction), face_features(target[:, :2])
        required = {"eyeLeftEAR": range(42, 48), "eyeRightEAR": range(36, 42),
                    "mouthOpenRatio": range(60, 68), "cheekWidthNormalized": [3, 13, *range(36, 48)]}
        for name, indices in required.items():
            if all(valid[i] for i in indices):
                ratio_errors[name].append(abs(estimated[name] - reference[name]))
        rows.append({"imageId": info["id"], "annotationId": annotation["id"], "faceBox": annotation["face_box"],
                     "annotatedLandmarks": int(valid.sum()), "interocularNME": float(normalized.mean())})
    if not rows:
        raise ValueError("No annotated public face samples were evaluated.")
    record = {"id": "pfld-face68-upstream", "kind": "faceLandmarks", "url": FACE_URL,
              "upstreamRevision": FACE_REV, "upstreamGitBlob": FACE_GIT_BLOB, "sha256": digest(path),
              "bytes": len(content), "redistributed": False, "inputName": inp.name,
              "inputShape": [1, 3, 112, 112], "declaredInputShape": inp.shape, "inputWH": [112, 112],
              "freeDimensionOverrides": {inp.shape[0]: 1} if isinstance(inp.shape[0], str) else {},
              "outputNames": [v.name for v in session.get_outputs()],
              "keypointCount": 68, "outputLayout": "flat interleaved normalized x,y in the square ROI",
              "color": "RGB", "scale": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1],
              "crop": "square face ROI; max(width,height)*1.1; black outside image",
              "stateOutputs": ["eyeLeftEAR", "eyeRightEAR", "mouthOpenRatio", "cheekWidthNormalized", "noseXY"],
              "cheekSemantics": "2D cheek-width ratio, not learned cheek-puff recognition",
              "license": "Upstream checkpoint redistribution grant unresolved; fetch from its author's pinned URL only.",
              "source": "https://github.com/cunjian/pytorch_face_landmark"}
    report = {"model": record, "dataset": {"name": "COCO-WholeBody v1.0 validation faces", "source": FACE_ANNOTATION_URL,
              "annotationSHA256": digest(annotation_path), "annotationLicense": "CC-BY-NC-4.0; images retain their individual source licenses",
              "faceSelection": "face_valid; face_box minimum side >=64; >=55 labeled points; sorted image/id, first requested count",
              "roiSource": "ground-truth face box; face detection is not evaluated", "images": len({r["imageId"] for r in rows}),
              "faces": len(rows), "landmarks": len(errors)}, "interocularNME": float(np.mean(errors)),
              "featureRatioMAE": {name: {"samples": len(vals), "mae": float(np.mean(vals))} for name, vals in ratio_errors.items()},
              "cases": rows, "inference": {"provider": "CPUExecutionProvider", "threads": args.threads,
              "meanMs": float(np.mean(times[1:] or times)), "p95Ms": float(np.percentile(times[1:] or times, 95)),
              "samples": len(times), "warmupExcluded": len(times) > 1},
              "limits": ["No binary blink, cheek-puff, or mouth-state labels: geometric-ratio error is measured instead.",
                         "This fixed 64-face diagnostic subset is not the model author's 300W benchmark or a representative COCO AP result.",
                         "The source model is a pretrained dependency, not a newly trained face model.",
                         "The upstream ONNX example uses RGB; its separate PyTorch batch example uses BGR.",
                         "No face confidence head; unreliable or unobserved faces must not be treated as measured expressions."]}
    write_json(args.report / "face-report.json", report)
    return record, report


def main():
    require_actions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "foot", "face"], default="all")
    parser.add_argument("--output", type=Path, default=Path("model-package/parts"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/parts"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/parts"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--face-examples", type=int, default=64)
    args = parser.parse_args()
    if min(args.threads, args.epochs, args.face_examples) < 1:
        parser.error("threads, epochs, and face-examples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    models, reports = [], {}
    if args.mode in ("all", "foot"):
        model, report = train_foot(args)
        models.append(model)
        reports["foot"] = report
    if args.mode in ("all", "face"):
        model, report = prepare_face(args)
        models.append(model)
        reports["face"] = report
    provenance = {"gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
                  "python": platform.python_version(), "torch": torch.__version__, "onnxruntime": ort.__version__,
                  "privateVideoUsed": False}
    manifest = {"schemaVersion": 1, "models": models, "provenance": provenance}
    write_json(args.output / "parts-manifest.json", manifest)
    write_json(args.report / "parts-report.json", {"schemaVersion": 1, "provenance": provenance, **reports})
    print(json.dumps({"complete": list(reports), "models": [m["id"] for m in models]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
