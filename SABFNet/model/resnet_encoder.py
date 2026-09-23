"""
Dual-branch ResNet50 encoder for SABFNet.
Adapted from FTransUNet (vit_seg_modeling_resnet_skip.py).

Returns per-stage feature maps for SSAA + final deep features for BF-FVit.
"""

import os
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ Helpers --------------------------------
def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(cin, cout, stride=1):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=False)


# ---------------------- ResNet Pre-Act Bottleneck ----------------------
class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        res = x
        if hasattr(self, "downsample"):
            res = self.gn_proj(self.downsample(x))

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(res + y)

    def load_from(self, weights, n_block, n_unit):
        with torch.no_grad():
            def key(n):
                return os.path.join(n_block, n_unit, n).replace("\\", "/")

            self.conv1.weight.copy_(np2th(weights[key("conv1/kernel")], conv=True))
            self.conv2.weight.copy_(np2th(weights[key("conv2/kernel")], conv=True))
            self.conv3.weight.copy_(np2th(weights[key("conv3/kernel")], conv=True))

            for gn, p in [(self.gn1, "gn1"), (self.gn2, "gn2"), (self.gn3, "gn3")]:
                gn.weight.copy_(np2th(weights[key(f"{p}/scale")]).view(-1))
                gn.bias.copy_(np2th(weights[key(f"{p}/bias")]).view(-1))

            if hasattr(self, "downsample"):
                self.downsample.weight.copy_(np2th(weights[key("conv_proj/kernel")], conv=True))
                self.gn_proj.weight.copy_(np2th(weights[key("gn_proj/scale")]).view(-1))
                self.gn_proj.bias.copy_(np2th(weights[key("gn_proj/bias")]).view(-1))


# --------------------------- Single Branch ResNet ----------------------
class ResNetBranch(nn.Module):
    """
    Single branch of the dual-branch encoder.
    Returns:
        deep: [B, 1024, H/16, W/16]    → BF-FVit input
        skips: list ordered deepest→shallowest
               [block2_feat(512,H/8), block1_feat(256,H/4), root_feat(64,H/2)]
    """
    def __init__(self, block_units, width_factor, in_channels=3):
        super().__init__()
        w = int(64 * width_factor)
        self.width = w

        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(in_channels, w, kernel_size=7, stride=2, padding=3, bias=False)),
            ("gn", nn.GroupNorm(32, w, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(w, w * 4, w))] +
                [(f"unit{i}", PreActBottleneck(w * 4, w * 4, w)) for i in range(2, block_units[0] + 1)]
            ))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(w * 4, w * 8, w * 2, stride=2))] +
                [(f"unit{i}", PreActBottleneck(w * 8, w * 8, w * 2)) for i in range(2, block_units[1] + 1)]
            ))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(w * 8, w * 16, w * 4, stride=2))] +
                [(f"unit{i}", PreActBottleneck(w * 16, w * 16, w * 4)) for i in range(2, block_units[2] + 1)]
            ))),
        ]))

    def forward(self, x):
        B, C, H, W = x.shape
        skips = []

        x = self.root(x)                         # [B, 64, H/2, W/2]
        skips.append(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        for i, blk in enumerate(self.body[:-1]):  # block1, block2
            x = blk(x)
            rh = int(H / 4 / (i + 1))
            rw = int(W / 4 / (i + 1))
            if x.shape[2] != rh:
                f = torch.zeros(B, x.shape[1], rh, rw, device=x.device)
                f[:, :, :x.shape[2], :x.shape[3]] = x
                x = f
            skips.append(x)                       # block1(256,H/4), block2(512,H/8)

        deep = self.body[-1](x)                   # [B, 1024, H/16, W/16]
        return deep, skips[::-1]                   # skips: [block2, block1, root]

    def load_from(self, weights):
        with torch.no_grad():
            self.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            self.root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
            self.root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))

            for bname, blk in self.body.named_children():
                for uname, unit in blk.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)


# --------------------------- Dual Branch Encoder -----------------------
class DualBranchEncoder(nn.Module):
    """
    Dual-branch ResNet50 encoder.
    Returns:
        opt_deep, dsm_deep:    [B,1024,H/16,W/16]  (for BF-FVit)
        stage_pairs:           list of (opt_feat, dsm_feat) at each scale,
                               ordered shallowest→deepest:
                               [(root(64,H/2), root(64,H/2)),
                                (block1(256,H/4), block1(256,H/4)),
                                (block2(512,H/8), block2(512,H/8))]
    """
    def __init__(self, block_units, width_factor, vis_ch=3, dsm_ch=1):
        super().__init__()
        self.opt_branch = ResNetBranch(block_units, width_factor, vis_ch)
        self.dsm_branch = ResNetBranch(block_units, width_factor, dsm_ch)

    def forward(self, opt, dsm):
        opt_deep, opt_skips = self.opt_branch(opt)
        dsm_deep, dsm_skips = self.dsm_branch(dsm)
        # opt_skips / dsm_skips: deepest→shallowest [block2, block1, root]
        # stage_pairs: shallowest→deepest [root, block1, block2]
        stage_pairs = list(zip(reversed(opt_skips), reversed(dsm_skips)))
        return opt_deep, dsm_deep, opt_skips, dsm_skips, stage_pairs

    def load_pretrained(self, path):
        weights = np.load(path)
        self.opt_branch.load_from(weights)
        try:
            self.dsm_branch.load_from(weights)
        except Exception:
            pass   # DSM root conv has different in_channels; skip mismatch
