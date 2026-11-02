"""Public COCO-WholeBody image-only data preparation; no private-video inputs."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

import cv2
import numpy as np

from model import SELECTED


ANNOTATION_URL = ("https://huggingface.co/datasets/msdkhairi/coco2017/resolve/main/"
                  "coco_wholebody_val_v1.0.json")
ANNOTATION_SHA256 = "f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e"
TEACHER_URL = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-tt_ucoco.pth"
TEACHER_SHA256 = "7097f5af7f100609acffe58eb01734f02ffbfe22794fe029c2ea0a4d68d0f42d"
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, destination, expected_sha256=None, timeout=120, attempts=4):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        if expected_sha256 is None or sha256(destination) == expected_sha256:
            return destination
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "compact-vrc-research/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as source, open(partial, "wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            if expected_sha256 and sha256(partial) != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for {destination.name}")
            partial.replace(destination)
            return destination
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)


def all_points(annotation):
    arrays = [annotation["keypoints"]]
    for field, count in (("foot_kpts", 6), ("face_kpts", 68),
                         ("lefthand_kpts", 21), ("righthand_kpts", 21)):
        arrays.append(annotation.get(field, [0] * (count * 3)))
    return np.asarray(sum(arrays, []), np.float32).reshape(133, 3)


def split_candidates(annotation_file, train_count, eval_count, seed):
    with open(annotation_file, encoding="utf-8") as stream:
        dataset = json.load(stream)
    images = {image["id"]: image for image in dataset["images"]}
    per_image = {}
    for annotation in dataset["annotations"]:
        if annotation.get("iscrowd") or annotation.get("num_keypoints", 0) < 6:
            continue
        x, y, w, h = annotation["bbox"]
        if min(w, h) < 64 or max(w, h) < 128:
            continue
        points = all_points(annotation)
        priority = int(np.count_nonzero(points[SELECTED, 2]))
        current = per_image.get(annotation["image_id"])
        if current is None or priority > current[0]:
            per_image[annotation["image_id"]] = (priority, annotation)
    ids = sorted(per_image)
    random.Random(seed).shuffle(ids)
    if len(ids) < train_count + eval_count + 32:
        raise RuntimeError(f"Not enough independent eligible images: {len(ids)}")
    # Separate candidate pools before download. A failed download cannot migrate
    # an image between train and evaluation, and only one person is used per image.
    eval_pool = ids[:max(eval_count * 2, eval_count + 32)]
    train_pool = ids[len(eval_pool):]

    def records(pool):
        return [{"image": images[image_id], "annotation": per_image[image_id][1]}
                for image_id in pool]

    return records(train_pool), records(eval_pool)


def download_records(records, count, cache):
    image_root = Path(cache) / "public-images"
    image_root.mkdir(parents=True, exist_ok=True)

    def one(record):
        image_id = int(record["image"]["id"])
        path = image_root / f"{image_id:012d}.jpg"
        # The official bucket's path-style HTTPS endpoint has a valid S3 TLS
        # name. The historical custom hostname can return a proxy/TLS 503.
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{image_id:012d}.jpg"
        try:
            fetch(url, path, timeout=20, attempts=2)
            if cv2.imread(str(path)) is None:
                raise ValueError("Invalid public image")
            return {**record, "path": str(path), "public_url": url}
        except Exception as exc:
            print(f"Public image {image_id} unavailable: {type(exc).__name__}", flush=True)
            return None

    accepted = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(records), 64):
            batch = list(pool.map(one, records[start:start + min(64, count - len(accepted) + 8)]))
            accepted.extend(record for record in batch if record is not None)
            if not accepted and not any(batch):
                raise RuntimeError("Public COCO image endpoint unavailable; no samples silently substituted")
            if len(accepted) >= count:
                return accepted[:count]
    raise RuntimeError(f"Only {len(accepted)}/{count} public images downloaded")


def crop(record, jitter=None):
    image = cv2.cvtColor(cv2.imread(record["path"]), cv2.COLOR_BGR2RGB)
    x, y, w, h = map(float, record["annotation"]["bbox"])
    center = np.array([x + w / 2, y + h / 2], np.float32)
    extent = np.array([w, h], np.float32) * 1.25
    if extent[0] > extent[1] * .75:
        extent[1] = extent[0] / .75
    else:
        extent[0] = extent[1] * .75
    if jitter is not None:
        center += np.asarray(jitter[:2], np.float32) * extent
        extent *= jitter[2]
    origin = center - extent / 2
    sx, sy = 192 / extent[0], 256 / extent[1]
    matrix = np.array([[sx, 0, -origin[0] * sx], [0, sy, -origin[1] * sy]], np.float32)
    cropped = cv2.warpAffine(image, matrix, (192, 256), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    points = all_points(record["annotation"])[SELECTED].copy()
    points[:, :2] = (points[:, :2] - origin) * np.array([sx, sy], np.float32)
    valid = ((points[:, 0] >= 0) & (points[:, 0] < 192) &
             (points[:, 1] >= 0) & (points[:, 1] < 256))
    points[~valid, 2] = 0
    normalized = ((cropped.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
    return np.ascontiguousarray(normalized), points, extent


def prepare(cache, train_count, eval_count, seed):
    annotation = fetch(ANNOTATION_URL, Path(cache) / "coco_wholebody_val_v1.0.json",
                       ANNOTATION_SHA256)
    train_pool, eval_pool = split_candidates(annotation, train_count, eval_count, seed)
    print(f"Downloading {train_count} training and {eval_count} held-out public images", flush=True)
    train = download_records(train_pool, train_count, cache)
    evaluation = download_records(eval_pool, eval_count, cache)
    train_ids = {record["image"]["id"] for record in train}
    eval_ids = {record["image"]["id"] for record in evaluation}
    if train_ids & eval_ids:
        raise AssertionError("Training/evaluation image leakage")
    return train, evaluation
