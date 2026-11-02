"""Lower-resolution backbones with the trained Compact59 head intact.

All original head weights are copied exactly. Two small constant MatMul
operators align native features to the original physical stride-32 grid.
Image convolutions run at reduced resolution; output stays 384/512 SimCC bins.
"""

import math

import torch
from torch import nn
from model import Head, Pose, SELECTED, TinyBackbone


CANDIDATES = (
    {"id": "fast59-aligned-192", "width": 144, "height": 192, "kernel": 7},
    {"id": "fast59-aligned-224", "width": 168, "height": 224, "kernel": 7},
)
DECODE_WH = (192, 256)


def feature_shape(width, height):
    # Five stride-2 convolutions use padding=1, hence ceiling division.
    return math.ceil(height / 32), math.ceil(width / 32)


def interpolation_matrix(source_length, target_length, coordinate_scale):
    """Sample physical feature centers without a resize half-pixel offset.

    Odd-kernel, symmetrically padded stride-two convolutions place feature
    centers at 32*j in input pixels. Target j samples source index
    j * native_input / original_input. Out-of-range neighbors clamp to edge.
    """
    matrix = torch.zeros(target_length, source_length, dtype=torch.float32)
    for target in range(target_length):
        position = target * coordinate_scale
        lower = math.floor(position)
        fraction = position - lower
        matrix[target, min(source_length - 1, lower)] += 1 - fraction
        matrix[target, min(source_length - 1, lower + 1)] += fraction
    return matrix


class PhysicalFeatureAlignment(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        source_h, source_w = feature_shape(width, height)
        self.source_h, self.source_w = source_h, source_w
        self.register_buffer("horizontal", interpolation_matrix(source_w, 6, width / 192).T.contiguous())
        self.register_buffer("vertical", interpolation_matrix(source_h, 8, height / 256))

    def forward(self, features):
        return torch.matmul(self.vertical, torch.matmul(features, self.horizontal))

    def multiply_accumulates(self):
        return 384 * self.source_h * self.source_w * 6 + 384 * 8 * self.source_h * 6


class FastPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.backbone = TinyBackbone()
        self.alignment = PhysicalFeatureAlignment(config["width"], config["height"])
        self.head = Head(59)

    def aligned_features(self, image):
        return self.alignment(self.backbone(image))

    def forward(self, image):
        return self.head(self.aligned_features(image))

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
    student.head.load_state_dict(baseline.head.state_dict(), strict=True)
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
        device = next(model.parameters()).device
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 3, height, width, device=device))
        points = outputs[0].shape[1]
        total += points * points * (128 + 512)
        if hasattr(model, "alignment"):
            total += model.alignment.multiply_accumulates()
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)
