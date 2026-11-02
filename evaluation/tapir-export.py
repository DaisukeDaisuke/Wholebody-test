#!/usr/bin/env python3
"""Actions-only pinned causal BootsTAPIR export and public-image numerical parity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import urllib.request
import warnings

CODE_REVISION = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
WEIGHT_REVISION = "5d3fb48e76c5422841e38501514121e251beabb7"
WEIGHT_FILENAME = "causal_bootstapir_checkpoint.pt"
WEIGHT_SHA256 = "87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f"
WEIGHT_BYTES = 218887028
REPOSITORY = "RabbitDaisuke/Foot-Point-Tracker-onnx"
ORT_WEB_VERSION = "1.22.0"
INPUT_WH = [256, 256]
QUERY_COUNT = 3
DATA_REVISION = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
DATA_ROOT = f"https://raw.githubusercontent.com/open-mmlab/mmpose/{DATA_REVISION}/tests/data/coco"


def require_actions():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("TAPIR export and inference run in GitHub Actions only.")
    if any(os.environ.get(k) for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("Do not pass model-publication credentials to the export job.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, limit):
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Wholebody-TAPIR-export/1.0"})
    for attempt in range(3):
        try:
            size = 0
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("Public download exceeds its size limit.")
                    output.write(chunk)
            temporary.replace(path)
            return path
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download_source(cache):
    root = cache / "source"
    files = ["tapnet/torch/__init__.py", "tapnet/torch/tapir_model.py", "tapnet/torch/nets.py", "tapnet/torch/utils.py", "LICENSE"]
    records = []
    for name in files:
        url = f"https://raw.githubusercontent.com/google-deepmind/tapnet/{CODE_REVISION}/{name}"
        path = fetch(url, root / name, 1024 * 1024)
        records.append({"path": name, "url": url, "sha256": sha256(path), "bytes": path.stat().st_size})
    sys.path.insert(0, str(root.resolve()))
    return root, records


def public_fixture(cache, results):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as functional

    annotation = fetch(DATA_ROOT + "/test_coco_wholebody.json", cache / "public-annotation.json", 2 * 1024 * 1024)
    data = json.loads(annotation.read_text(encoding="utf-8"))
    image = next(value for value in data["images"] if value["id"] == 785)
    if image["license"] != 4:
        raise ValueError("Only the reviewed CC BY public image may be used for this fixture.")
    path = fetch(DATA_ROOT + "/" + image["file_name"], cache / image["file_name"], 5 * 1024 * 1024)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError("The public fixture could not be decoded.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
    normalized = functional.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    normalized = (normalized / 255 * 2 - 1).permute(0, 2, 3, 1).unsqueeze(1).contiguous()
    points_xy = np.array([[w * .32, h * .62], [w * .54, h * .50], [w * .69, h * .37]], dtype=np.float32)
    query_yx = points_xy[:, ::-1].copy() * np.array([256 / h, 256 / w], dtype=np.float32)
    frames = []
    source = normalized.numpy()[0, 0]
    for index in range(4):
        affine = np.array([[1, 0, index * 3], [0, 1, index * 2]], dtype=np.float32)
        moved = cv2.warpAffine(source, affine, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if index == 2:
            center = np.rint(query_yx[1, ::-1] + [index * 3, index * 2]).astype(int)
            moved[max(0, center[1] - 14):min(256, center[1] + 14), max(0, center[0] - 14):min(256, center[0] + 14)] = -1
        frames.append(torch.from_numpy(moved.copy()).unsqueeze(0).unsqueeze(0))
    fixture_dir = results / "codec-fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rgb.tofile(fixture_dir / "public-rgb.u8")
    normalized.numpy().astype("<f4").tofile(fixture_dir / "frame-reference.f32")
    attribution = {**image, "licenseDetail": next(item for item in data["licenses"] if item["id"] == 4),
                   "retrievedFrom": DATA_ROOT + "/" + image["file_name"], "sha256": sha256(path), "privateMedia": False}
    fixture_metadata = {"width": w, "height": h, "channels": 3, "pointsXY": points_xy.tolist(), "queryYX": query_yx.tolist(),
                        "imageFile": "public-rgb.u8", "frameFile": "frame-reference.f32", "attribution": attribution}
    return frames, torch.from_numpy(query_yx).unsqueeze(0), fixture_metadata


def tensor_difference(actual, expected):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Nonfinite or incompatible parity tensors: {a.shape}, {b.shape}")
    error = np.abs(a - b)
    return {"maxAbs": float(error.max(initial=0)), "meanAbs": float(error.mean()), "shape": list(a.shape)}


def normalize_webgpu_graph(path):
    """Keep fixed-shape math identical while removing unsupported Max/Size ops."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    graph = onnx.load(str(path))
    constants = {value.name: numpy_helper.to_array(value) for value in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            attribute = next((value for value in node.attribute if value.name == "value"), None)
            if attribute is not None:
                constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
    shapes = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if not value.type.tensor_type.HasField("shape"):
            continue
        dimensions = value.type.tensor_type.shape.dim
        if all(dimension.HasField("dim_value") and dimension.dim_value > 0 for dimension in dimensions):
            shapes[value.name] = [dimension.dim_value for dimension in dimensions]
    nodes, changes = [], []
    for index, node in enumerate(graph.graph.node):
        if node.op_type == "Max" and len(node.input) == 2:
            scalar_index = next((i for i, name in enumerate(node.input) if name in constants and constants[name].size == 1 and np.isfinite(constants[name]).all()), None)
            if scalar_index is not None:
                name = f"webgpu_clip_minimum_{index}"
                minimum = np.asarray(constants[node.input[scalar_index]]).reshape(())
                graph.graph.initializer.append(numpy_helper.from_array(minimum, name=name))
                nodes.append(helper.make_node("Clip", [node.input[1 - scalar_index], name], list(node.output), name=node.name))
                changes.append({"node": node.name, "from": "Max", "to": "Clip", "minimum": float(minimum)})
                continue
        if node.op_type == "Size" and len(node.input) == 1 and node.input[0] in shapes:
            value = np.array(np.prod(shapes[node.input[0]], dtype=np.int64), dtype=np.int64)
            nodes.append(helper.make_node("Constant", [], list(node.output), name=node.name, value=numpy_helper.from_array(value)))
            changes.append({"node": node.name, "from": "Size", "to": "Constant", "inputShape": shapes[node.input[0]], "value": int(value)})
            continue
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    return changes


def inspect_graph(path, supported_ops, threads):
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    counts = Counter(node.op_type for node in graph.graph.node)
    host_shape_ops = {"Constant", "ConstantOfShape", "Identity"}
    unlisted = sorted(set(counts) - supported_ops - host_shape_ops)
    shape_errors = []
    try:
        inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
        shapes = {value.name: len(value.type.tensor_type.shape.dim) for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
        for node in graph.graph.node:
            if node.op_type == "GridSample" and shapes.get(node.input[0]) not in (None, 4):
                shape_errors.append({"node": node.name, "operator": "GridSample", "inputRank": shapes.get(node.input[0])})
    except Exception as error:
        shape_errors.append({"shapeInferenceError": type(error).__name__})
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    model = {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "opset": [{"domain": item.domain, "version": item.version} for item in graph.opset_import],
             "inputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()],
             "outputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_outputs()],
             "operators": dict(sorted(counts.items())), "webgpuStatic": {"runtimeVersion": ORT_WEB_VERSION, "unlistedComputeOps": unlisted,
                 "shapeRestrictions": shape_errors, "hostShapeOps": sorted(set(counts) & host_shape_ops), "hardwareExecuted": False}}
    return model, session


def export_and_compare(args, report):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from torch import nn
    import torch.nn.functional as functional

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    source_root, source_files = download_source(args.cache)
    from tapnet.torch import tapir_model

    weight_url = f"https://huggingface.co/google/tapnet/resolve/{WEIGHT_REVISION}/{WEIGHT_FILENAME}"
    weight = fetch(weight_url, args.cache / WEIGHT_FILENAME, 230 * 1024 * 1024)
    if weight.stat().st_size != WEIGHT_BYTES or sha256(weight) != WEIGHT_SHA256:
        raise ValueError("The checkpoint differs from the reviewed official SHA-256.")
    report["source"] = {"codeRepository": "google-deepmind/tapnet", "codeRevision": CODE_REVISION, "weightRepository": "google/tapnet", "weightRevision": WEIGHT_REVISION,
                        "filename": WEIGHT_FILENAME, "bytes": WEIGHT_BYTES, "sha256": WEIGHT_SHA256, "url": weight_url, "license": "apache-2.0", "sourceFiles": source_files}
    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True).eval()
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    initial = model.construct_initial_causal_state(QUERY_COUNT, 1)
    layout, offset = [], 0
    for iteration, values in enumerate(initial):
        for block in range(12):
            for which in (1, 2):
                key = f"block_{block}_causal_{which}"
                shape, length = list(values[key].shape), values[key].numel()
                layout.append({"iteration": iteration, "key": key, "offset": offset, "length": length, "shape": shape})
                offset += length
    zero_state = torch.zeros(offset, dtype=torch.float32)

    def unpack_state(flat):
        state = [{} for _ in range(4)]
        for item in layout:
            state[item["iteration"]][item["key"]] = flat[item["offset"]:item["offset"] + item["length"]].reshape(item["shape"])
        return state

    def pack_state(state):
        return torch.cat([state[item["iteration"]][item["key"]].reshape(-1) for item in layout])

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = model

        def forward(self, frame):
            grids = self.network.get_feature_grids(frame, False, refinement_resolutions=[(256, 256)])
            return grids.lowres[0], grids.hires[0]

    class Query(nn.Module):
        def forward(self, lowres, hires, query_yx):
            # T=1 and query t=0 make official 5D sampling exactly a 2D sample.
            grid = (query_yx.flip(-1) * (2.0 / 256) - 1).unsqueeze(2)
            def sample(features):
                image = features[:, 0].permute(0, 3, 1, 2)
                return functional.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).transpose(1, 2)
            return sample(lowres), sample(hires)

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = model

        def forward(self, lowres, hires, query_lowres, query_hires, causal_state):
            grids = tapir_model.FeatureGrids((lowres, lowres), (hires, hires), ((256, 256), (256, 256)))
            queries = tapir_model.QueryFeatures((query_lowres, query_lowres), (query_hires, query_hires), ((256, 256), (256, 256)))
            result = self.network.estimate_trajectories((256, 256), False, grids, queries, query_points_in_video=None,
                query_chunk_size=QUERY_COUNT, causal_context=unpack_state(causal_state), get_causal_context=True)
            return result["tracks"][-1], result["occlusion"][-1], result["expected_dist"][-1], pack_state(result["causal_context"])

    frames, query_yx, fixture_meta = public_fixture(args.cache, args.results)
    encoder, query, step = Encoder().eval(), Query().eval(), Step().eval()
    low, high = encoder(frames[0])
    if list(low.shape) != [1, 1, 32, 32, 256] or list(high.shape) != [1, 1, 64, 64, 128]:
        raise ValueError(f"Unexpected official feature shapes: {list(low.shape)}, {list(high.shape)}")
    query_low, query_high = query(low, high, query_yx)
    query_tyx = torch.cat([torch.zeros((1, QUERY_COUNT, 1)), query_yx], dim=-1)
    official_grids = model.get_feature_grids(frames[0], False, refinement_resolutions=[(256, 256)])
    official_queries = model.get_query_features(frames[0], False, query_tyx, feature_grids=official_grids)
    query_equivalence = {"lowres": tensor_difference(query_low.numpy(), official_queries.lowres[0].numpy()),
                         "hires": tensor_difference(query_high.numpy(), official_queries.hires[0].numpy()), "restriction": "Exactly one frame and initial query time 0."}
    if max(query_equivalence["lowres"]["maxAbs"], query_equivalence["hires"]["maxAbs"]) > 2e-6:
        raise ValueError("The 2D query extraction does not match the official 5D initialization.")
    report["queryExtractionEquivalence"] = query_equivalence
    stages = [
        ("encoder", encoder, (frames[0],), ["frame"], ["lowres", "hires"]),
        ("query", query, (low, high, query_yx), ["lowres", "hires", "query_yx"], ["query_lowres", "query_hires"]),
        ("step", step, (low, high, query_low, query_high, zero_state), ["lowres", "hires", "query_lowres", "query_hires", "causal_state"], ["tracks", "occlusion", "expected_dist", "new_causal_state"]),
    ]
    table_url = f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{ORT_WEB_VERSION}/js/web/docs/webgpu-operators.md"
    table = fetch(table_url, args.cache / "webgpu-operators.md", 256 * 1024)
    supported_ops = set(re.findall(r"^\|\s*(\w+)\s*\|", table.read_text(encoding="utf-8"), re.MULTILINE))
    models, sessions = [], {}
    for name, wrapper, inputs, input_names, output_names in stages:
        path = args.output / f"causal-bootstapir-{name}-256-q3-fp32.onnx"
        print(json.dumps({"stage": "export", "model": name}), flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(wrapper, inputs, str(path), input_names=input_names, output_names=output_names,
                              opset_version=17, dynamo=False, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
        normalization = normalize_webgpu_graph(path)
        metadata, session = inspect_graph(path, supported_ops, args.threads)
        metadata.update({"id": name, "exportSeconds": time.perf_counter() - started, "graphNormalization": normalization})
        models.append(metadata)
        sessions[name] = session
    first_features = sessions["encoder"].run(None, {"frame": frames[0].numpy()})
    exported_queries = sessions["query"].run(None, {"lowres": first_features[0], "hires": first_features[1], "query_yx": query_yx.numpy()})
    native_state = unpack_state(zero_state.clone())
    onnx_state = zero_state.numpy().copy()
    comparisons, native_ms, encoder_ms, step_ms = [], [], [], []
    native_first_state = None
    for index, frame in enumerate(frames):
        started = time.perf_counter()
        native_grids = model.get_feature_grids(frame, False, refinement_resolutions=[(256, 256)])
        reference = model.estimate_trajectories((256, 256), False, native_grids, official_queries, query_points_in_video=None,
                        query_chunk_size=QUERY_COUNT, causal_context=native_state, get_causal_context=True)
        native_ms.append((time.perf_counter() - started) * 1000)
        native_state = reference["causal_context"]
        native_outputs = [reference["tracks"][-1].numpy(), reference["occlusion"][-1].numpy(), reference["expected_dist"][-1].numpy(), pack_state(native_state).numpy()]
        started = time.perf_counter()
        features = sessions["encoder"].run(None, {"frame": frame.numpy()})
        encoder_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        output = sessions["step"].run(None, {"lowres": features[0], "hires": features[1], "query_lowres": exported_queries[0], "query_hires": exported_queries[1], "causal_state": onnx_state})
        step_ms.append((time.perf_counter() - started) * 1000)
        onnx_state = output[3]
        difference = {name: tensor_difference(actual, expected) for name, actual, expected in zip(["tracks", "occlusion", "expected_dist", "causal_state"], output, native_outputs)}
        visibility = lambda occ, dist: (1 / (1 + np.exp(np.clip(occ, -80, 80)))) * (1 / (1 + np.exp(np.clip(dist, -80, 80)))) > .5
        mismatches = int(np.count_nonzero(visibility(output[1], output[2]) != visibility(native_outputs[1], native_outputs[2])))
        comparisons.append({"frame": index, "syntheticTranslationXYpx": [index * 3, index * 2], "artificialOccluder": index == 2, "differences": difference, "visibleDecisionMismatches": mismatches})
        if index == 0:
            native_first_state = pack_state(native_state).clone()
            fixture_meta["outputs"] = {"tracks": output[0].reshape(-1).tolist(), "occlusion": output[1].reshape(-1).tolist(), "expected_dist": output[2].reshape(-1).tolist()}
            confidence = ((1 - torch.sigmoid(torch.from_numpy(output[1]))) * (1 - torch.sigmoid(torch.from_numpy(output[2])))).numpy().reshape(-1)
            original_xy = output[0].reshape(QUERY_COUNT, 2) * np.array([fixture_meta["width"] / 256, fixture_meta["height"] / 256])
            fixture_meta["decodedReference"] = [{"x": float(point[0]), "y": float(point[1]), "score": float(confidence[i]), "observable": bool(confidence[i] > .5)} for i, point in enumerate(original_xy)]
        print(json.dumps({"stage": "parity", "frame": index, "maxTrackDifferencePx": difference["tracks"]["maxAbs"], "onnxEncoderMs": encoder_ms[-1], "onnxStepMs": step_ms[-1]}), flush=True)
    # A second sequence keeps the original RGB bytes unchanged. The production
    # browser tracker can replay it without duplicating the export preprocessing.
    repeated_state, repeated_reference = unpack_state(native_first_state), []
    for index in (1, 2):
        reference = model.estimate_trajectories((256, 256), False, official_grids, official_queries, query_points_in_video=None,
                        query_chunk_size=QUERY_COUNT, causal_context=repeated_state, get_causal_context=True)
        repeated_state = reference["causal_context"]
        repeated_reference.append({"frame": index, "timestampMs": index * 1000 / 30,
            "outputs": {"tracks": reference["tracks"][-1].numpy().reshape(-1).tolist(),
                        "occlusion": reference["occlusion"][-1].numpy().reshape(-1).tolist(),
                        "expected_dist": reference["expected_dist"][-1].numpy().reshape(-1).tolist()}})
    fixture_meta["repeatedFrameReference"] = {"kind": "Unchanged public RGB image, official PyTorch causal state carried between calls", "frames": repeated_reference}
    passed = all(c["differences"]["tracks"]["maxAbs"] <= .05 and c["differences"]["occlusion"]["maxAbs"] <= .02
                 and c["differences"]["expected_dist"]["maxAbs"] <= .02 and c["differences"]["causal_state"]["maxAbs"] <= .05 and c["visibleDecisionMismatches"] == 0 for c in comparisons)
    report.update({"status": "measured", "passed": passed, "models": models, "inputWH": INPUT_WH, "queryCount": QUERY_COUNT, "stateShape": [offset], "stateBytes": offset * 4,
                   "parity": {"passed": passed, "frameCount": len(frames), "comparisons": comparisons, "maxTrackDifferencePx": max(c["differences"]["tracks"]["maxAbs"] for c in comparisons),
                              "tolerance": {"tracksPx": .05, "logits": .02, "causalStateMaxAbs": .05, "visibilityMismatches": 0}},
                   "latency": {"device": "CPU", "threads": args.threads, "frameCount": len(frames), "officialTorchTotalMs": native_ms, "onnxEncoderMs": encoder_ms, "onnxStepMs": step_ms,
                               "onnxMedianFrameMs": float(np.median(np.array(encoder_ms) + np.array(step_ms))), "webgpuMeasured": False, "realTimeCameraMeasured": False},
                   "webgpuStatic": {"operatorTable": table_url, "operatorTableSha256": sha256(table), "runtimeVersion": ORT_WEB_VERSION,
                       "unlistedComputeOps": sorted({op for m in models for op in m["webgpuStatic"]["unlistedComputeOps"]}), "hardwareExecuted": False},
                   "fixture": {"kind": "Public CC BY real image with artificial translation and one artificial occluder; numerical conversion parity, not a real-video accuracy benchmark.", "attribution": fixture_meta["attribution"]},
                   "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "onnxruntime": ort.__version__, "opencv": cv2.__version__, "einshape": importlib.metadata.version("einshape")}})
    write_json(args.results / "codec-fixture" / "fixture.json", fixture_meta)
    manifest = {"schemaVersion": 1, "repository": REPOSITORY, "license": "apache-2.0", "modelId": "google/causal-bootstapir", "source": report["source"],
                "inputWH": INPUT_WH, "queryCount": QUERY_COUNT, "stateShape": [offset], "stateType": "float32", "stateLayout": layout,
                "preprocessing": {"input": "RGB uint8", "resize": "float bilinear align_corners=false", "tensor": "float32[1,1,256,256,3]", "range": [-1, 1], "query": "first-frame t=0, model-pixel yx float32[1,3,2]"},
                "models": models, "visibility": {"formula": "(1-sigmoid(occlusion))*(1-sigmoid(expected_dist))", "threshold": .5},
                "parityPassed": passed, "webgpuStatic": report["webgpuStatic"], "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID")}
    write_json(args.output / "tapir-models.json", manifest)
    (args.output / "LICENSE").write_text((source_root / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
    readme = f"""---
license: apache-2.0
tags:
- point-tracking
- onnx
- causal
---
# Causal BootsTAPIR: 256px, three query points
Official Google causal BootsTAPIR weights converted in GitHub Actions into encoder, query and recurrent-step ONNX graphs.
This is point tracking from three supplied initial image coordinates, not semantic foot detection, 3D reconstruction, or independent toe articulation.
Source code: https://github.com/google-deepmind/tapnet/tree/{CODE_REVISION}
Source weights: {weight_url}
Source SHA-256: `{WEIGHT_SHA256}`.
Input: RGB float32 `[1,1,256,256,3]` in `[-1,1]`; first-frame queries `[1,3,2]` in model-pixel `(y,x)` order. Outputs are xy pixel tracks and two uncertainty logits.
Run encoder then query once; run encoder and step for each frame, retaining query features and feeding `new_causal_state` into the next `causal_state`. Start from the zero state defined by `tapir-models.json`.
The first-frame query extraction uses a mathematically equivalent 2D sample because time is fixed at zero. It is checked against the official 5D implementation.
The Actions comparison uses only a public CC BY image with artificial motion/occlusion. It verifies conversion numerics, not real-foot tracking quality or occlusion recovery accuracy.
WebGPU operator coverage is checked statically against ONNX Runtime Web {ORT_WEB_VERSION}; GPU execution, latency and RTX memory use are not measured by the CPU comparison.
No private video, extracted private frame, avatar file, or private pose trace is included.
License: Apache-2.0. Copyright and source notices are retained in LICENSE and the pinned source links.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    if not passed:
        raise ValueError("The ONNX pipeline differs from official causal BootsTAPIR beyond the declared tolerances.")


def main():
    require_actions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tapir-model-package"))
    parser.add_argument("--results", type=Path, default=Path("evaluation/tapir-results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/wholebody-tapir"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    report = {"schemaVersion": 1, "status": "started", "passed": False, "gitCommit": os.environ.get("GITHUB_SHA"), "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
              "privateMediaUsed": False, "footSemanticDetection": False, "depthMeasured": False, "realVideoAccuracyMeasured": False, "webgpuHardwareMeasured": False}
    try:
        export_and_compare(args, report)
    except Exception as error:
        report.update({"status": "failed", "passed": False, "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})
        raise
    finally:
        write_json(args.results / "tapir-export-report.json", report)


if __name__ == "__main__":
    main()
