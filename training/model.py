"""Minimal PyTorch implementation of the pretrained CSPNeXt/RTMCC contract.

Architecture reference: OpenMMLab MMPose and MMDetection (Apache-2.0).
The student physically removes unused output convolution channels and GAU tokens.
It is not a Gather wrapper around the original 133-keypoint network.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


BODY = [0, *range(5, 17)]
FEET = list(range(17, 23))
FINGER_OFFSETS = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
LEFT_HAND = [91 + i for i in FINGER_OFFSETS]
RIGHT_HAND = [112 + i for i in FINGER_OFFSETS]
FACE = [23 + i for i in [37, 41, 44, 46, 48, 54, 62, 66, 2, 14]]
SELECTED = BODY + FEET + LEFT_HAND + RIGHT_HAND + FACE
GROUPS = {"body": [0, 13], "feet": [13, 19], "left_hand": [19, 34],
          "right_hand": [34, 49], "face": [49, 59]}


class ConvBN(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                              groups=groups, bias=False)
        # The original config sets SyncBN without eps, hence the PyTorch default.
        self.bn = nn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)
        self.activate = nn.SiLU()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class Depthwise(nn.Module):
    def __init__(self, channels, kernel=5):
        super().__init__()
        self.depthwise_conv = ConvBN(channels, channels, kernel, groups=channels)
        self.pointwise_conv = ConvBN(channels, channels)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Block(nn.Module):
    def __init__(self, channels, identity):
        super().__init__()
        self.conv1 = ConvBN(channels, channels, 3)
        self.conv2 = Depthwise(channels)
        self.identity = identity

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.identity else y


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * F.hardsigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class CSP(nn.Module):
    def __init__(self, channels, blocks, identity):
        super().__init__()
        half = channels // 2
        self.main_conv = ConvBN(channels, half)
        self.short_conv = ConvBN(channels, half)
        self.final_conv = ConvBN(channels, channels)
        self.blocks = nn.Sequential(*(Block(half, identity) for _ in range(blocks)))
        self.attention = ChannelAttention(channels)

    def forward(self, x):
        z = torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), dim=1)
        return self.final_conv(self.attention(z))


class SPP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBN(channels, channels // 2)
        self.conv2 = ConvBN(channels * 2, channels)
        self.poolings = nn.ModuleList(nn.MaxPool2d(k, 1, k // 2) for k in (5, 9, 13))

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x, *(p(x) for p in self.poolings)], dim=1))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBN(3, 12, 3, 2), ConvBN(12, 12, 3),
                                  ConvBN(12, 24, 3))
        self.stage1 = nn.Sequential(ConvBN(24, 48, 3, 2), CSP(48, 1, True))
        self.stage2 = nn.Sequential(ConvBN(48, 96, 3, 2), CSP(96, 1, True))
        self.stage3 = nn.Sequential(ConvBN(96, 192, 3, 2), CSP(192, 1, True))
        self.stage4 = nn.Sequential(ConvBN(192, 384, 3, 2), SPP(384), CSP(384, 1, False))

    def forward(self, x):
        return self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))


class ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp_min(1e-5) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.scale * x


class GAU(nn.Module):
    def __init__(self):
        super().__init__()
        self.o = nn.Linear(512, 256, bias=False)
        self.uv = nn.Linear(256, 1152, bias=False)
        self.gamma = nn.Parameter(torch.empty(2, 128))
        self.beta = nn.Parameter(torch.empty(2, 128))
        self.ln = ScaleNorm(256)
        self.res_scale = Scale(256)

    def forward(self, x):
        u, v, base = F.silu(self.uv(self.ln(x))).split([512, 512, 128], dim=-1)
        base = base.unsqueeze(2) * self.gamma[None, None] + self.beta
        q, k = base.unbind(2)
        attention = F.relu(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(128)).square()
        return self.res_scale(x) + self.o(u * torch.bmm(attention, v))


class Head(nn.Module):
    def __init__(self, keypoints=133):
        super().__init__()
        self.final_layer = nn.Conv2d(384, keypoints, 7, padding=3)
        self.mlp = nn.Sequential(ScaleNorm(48), nn.Linear(48, 256, bias=False))
        self.gau = GAU()
        self.cls_x = nn.Linear(256, 384, bias=False)
        self.cls_y = nn.Linear(256, 512, bias=False)

    def forward(self, features):
        x = self.gau(self.mlp(self.final_layer(features).flatten(2)))
        return self.cls_x(x), self.cls_y(x)


class Pose(nn.Module):
    def __init__(self, keypoints=133):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = Head(keypoints)

    def forward(self, x):
        return self.head(self.backbone(x))


def load_teacher(path):
    # Trusted, hash-checked author checkpoint contains mmengine HistoryBuffer.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    original = checkpoint.get("state_dict", checkpoint)
    model = Pose()
    expected = model.state_dict()
    prefixes = ("", "module.", "student.", "student.module.", "architecture.")
    best = max(prefixes, key=lambda prefix: sum(prefix + key in original for key in expected))
    state = {key: original[best + key] for key in expected if best + key in original}
    missing = sorted(set(expected) - set(state))
    if missing:
        raise RuntimeError(f"Checkpoint contract mismatch: missing {missing[:15]}; "
                           f"available prefixes={list(original)[:12]}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_student(teacher):
    student = Pose(len(SELECTED))
    student.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)
    state = copy.deepcopy(teacher.head.state_dict())
    state["final_layer.weight"] = state["final_layer.weight"][SELECTED].clone()
    state["final_layer.bias"] = state["final_layer.bias"][SELECTED].clone()
    student.head.load_state_dict(state, strict=True)
    student.backbone.requires_grad_(False)
    student.eval()
    return student


def multiply_accumulates(model):
    total = 0
    handles = []

    def count(module, inputs, outputs):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += outputs.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            total += outputs.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 256, 192))
    k = out[0].shape[1]
    total += k * k * (128 + 512)  # GAU qk and attention-v products.
    for handle in handles:
        handle.remove()
    return int(total)
