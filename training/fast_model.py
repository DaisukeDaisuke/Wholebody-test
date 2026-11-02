"""Native low-resolution students; no resize or Gather wrapper at inference.

The checkpoint-compatible original lives in model.py and remains unchanged.
Kernel, spatial MLP and SimCC classifier interpolation happens once at
initialization. Exported students contain only their smaller native operators.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import GAU, Pose, SELECTED, ScaleNorm, TinyBackbone


CANDIDATES = (
    {"id": "fast59-128", "width": 96, "height": 128, "kernel": 3},
    {"id": "fast59-160", "width": 120, "height": 160, "kernel": 5},
    {"id": "fast59-192", "width": 144, "height": 192, "kernel": 5},
)


def feature_shape(width, height):
    # Five stride-2 convolutions use padding=1, hence ceiling division.
    return math.ceil(height / 32), math.ceil(width / 32)


class FastHead(nn.Module):
    def __init__(self, width, height, kernel):
        super().__init__()
        feature_h, feature_w = feature_shape(width, height)
        spatial = feature_h * feature_w
        self.final_layer = nn.Conv2d(384, 59, kernel, padding=kernel // 2)
        self.mlp = nn.Sequential(ScaleNorm(spatial), nn.Linear(spatial, 256, bias=False))
        self.gau = GAU()
        self.cls_x = nn.Linear(256, width * 2, bias=False)
        self.cls_y = nn.Linear(256, height * 2, bias=False)

    def forward(self, features):
        tokens = self.gau(self.mlp(self.final_layer(features).flatten(2)))
        return self.cls_x(tokens), self.cls_y(tokens)


class FastPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.backbone = TinyBackbone()
        self.head = FastHead(config["width"], config["height"], config["kernel"])

    def forward(self, image):
        return self.head(self.backbone(image))

    def train(self, mode=True):
        super().train(mode)
        # A small transfer set must not replace the pretrained BN population
        # statistics. Convolution weights are still fine-tuned in stage two.
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self


def load_baseline(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    manifest = checkpoint.get("manifest", {})
    if manifest.get("coco133_indices") != SELECTED:
        raise ValueError("Baseline does not have the established Compact59 ordering")
    if manifest.get("input", {}).get("shape") != [1, 3, 256, 192]:
        raise ValueError("Baseline must be the original 192x256 Compact59")
    model = Pose(59)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, manifest


def initialize_student(baseline, config):
    student = FastPose(config)
    student.backbone.load_state_dict(baseline.backbone.state_dict(), strict=True)
    original = baseline.head.state_dict()
    state = copy.deepcopy(original)
    kernel = config["kernel"]
    weight = original["final_layer.weight"]
    state["final_layer.weight"] = F.interpolate(
        weight.reshape(-1, 1, 7, 7), size=(kernel, kernel), mode="bilinear",
        align_corners=False).reshape(59, 384, kernel, kernel) * (49 / kernel ** 2)
    feature_h, feature_w = feature_shape(config["width"], config["height"])
    state["mlp.1.weight"] = F.interpolate(
        original["mlp.1.weight"].reshape(256, 1, 8, 6),
        size=(feature_h, feature_w), mode="bilinear", align_corners=False,
    ).reshape(256, feature_h * feature_w) * (48 / (feature_h * feature_w))
    for axis, length in (("x", config["width"] * 2), ("y", config["height"] * 2)):
        weight = original[f"cls_{axis}.weight"]
        state[f"cls_{axis}.weight"] = F.interpolate(
            weight.T.unsqueeze(0), size=length, mode="linear", align_corners=False,
        )[0].T.contiguous()
    student.head.load_state_dict(state, strict=True)
    return student.eval()


def multiply_accumulates(model, width, height):
    total = 0
    hooks = []

    def count(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    try:
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                hooks.append(layer.register_forward_hook(count))
        model.eval()
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 3, height, width))
        points = outputs[0].shape[1]
        total += points * points * (128 + 512)
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)
