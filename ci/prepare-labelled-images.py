"""Actions-only fixed public holdout subset, no private media accepted."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def fetch(url, path, expected=None):
    if path.exists() and (expected is None or hashlib.sha256(path.read_bytes()).hexdigest() == expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vrc-labelled-regression/1"}), timeout=120) as response:
        data = response.read(300 * 1024 * 1024)
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("public_annotation_checksum_mismatch")
    path.write_bytes(data)


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Run labelled model evaluation in GitHub Actions.")
    split = json.loads(Path("artifacts/fast59/public-split.json").read_text())
    cache = Path(".cache/labelled-pose3d")
    annotation = cache / "annotations.json"
    fetch(split["annotation_url"], annotation, split["annotation_sha256"])
    dataset = json.loads(annotation.read_text())
    annotations = {a["id"]: a for a in dataset["annotations"]}
    images = {i["id"]: i for i in dataset["images"]}
    records = []
    # The first 32 were used to reject standalone 3D XY and design hybrid
    # routing. Fix the remaining 160 before evaluating that routing.
    for image_id, annotation_id in zip(split["test_image_ids"][32:], split["test_annotation_ids"][32:], strict=True):
        image, item = images[image_id], annotations[annotation_id]
        assert item["image_id"] == image_id
        path = cache / f"{image_id:012d}.jpg"
        fetch(split["image_url_template"].format(image_id=image_id), path)
        records.append({"imageId": image_id, "annotationId": annotation_id, "path": str(path),
                        "width": image["width"], "height": image["height"], "bbox": item["bbox"], "keypoints": item["keypoints"]})
    (cache / "input.json").write_text(json.dumps({"records": records, "sourceAnnotationSha256": split["annotation_sha256"]}), encoding="utf-8")
    print(json.dumps({"publicImages": len(records), "privateInputsUsed": False, "selection": "remaining160FixedOldHeldoutIds", "excludedRoutingDevelopmentImages": 32}))


if __name__ == "__main__":
    main()
