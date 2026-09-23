"""
Same-Scale Attention Aggregation (SSAA) Module  —  SABFNet §III-B

Pipeline per call:
  1. Fine-Grained Multi-Scale Fusion (FG-MS)
       Split Fin into G groups → 3 parallel convolutions (k=3,5,7) per group
       → Gaussian similarity aggregation (3×3 neighbourhood)
       → Fsim = Concat of all per-group, per-kernel results
  2. Channel Attention on original Fin  →  Fch
  3. Spatial Attention using Fsim as prior  →  Fcs
  4. Output: Fout = Fcs + Fin   (residual)

SSAA is applied independently to each encoder stage.
After SSAA, the fused feature (VIS-side calibrated + DSM contribution)
becomes the skip connection E_s for the corresponding decoder scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------- Fine-Grained Multi-Scale Fusion (FG-MS) ----------------
class GaussianSimilarityAggregation(nn.Module):
    """
    For each pixel (p,q) and its 3×3 neighbourhood, compute:
      S(p,q),(m,n) = exp(-α * ||feat(p,q) - feat(m,n)||² / 2)
    then normalise to weights ω and aggregate.

    Implemented efficiently with unfold (zero-pad borders → keep spatial size).
    """
    def __init__(self, alpha: float = 1.0, nbr: int = 3):
        super().__init__()
        self.alpha = alpha
        self.nbr = nbr      # neighbourhood window size (3 = 3×3)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, C, H, W] → [B, C, H, W]"""
        B, C, H, W = feat.shape
        pad = self.nbr // 2
        # Unfold → [B, C*nbr*nbr, H*W]
        patches = F.unfold(feat, kernel_size=self.nbr, padding=pad)  # [B, C*k², H*W]
        k2 = self.nbr * self.nbr
        patches = patches.view(B, C, k2, H * W)          # [B, C, k², H*W]
        center = feat.view(B, C, 1, H * W)               # [B, C, 1, H*W]

        diff_sq = (patches - center).pow(2).mean(dim=1)   # [B, k², H*W]
        sim = torch.exp(-self.alpha * diff_sq / 2.0)      # [B, k², H*W]
        weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-6)  # normalise

        # Weighted aggregate
        agg = (patches * weights.unsqueeze(1)).sum(dim=2)  # [B, C, H*W]
        return agg.view(B, C, H, W)


class FGMSBlock(nn.Module):
    """
    Fine-Grained Multi-Scale Fusion block for one channel group Zi.
    Applies conv-3×3, conv-5×5, conv-7×7 in parallel and then
    similarity-aggregates each output.
    """
    def __init__(self, group_ch: int, kernels=(3, 5, 7), alpha: float = 1.0):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_ch, group_ch, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(group_ch),
                nn.ReLU(inplace=True),
            )
            for k in kernels
        ])
        self.agg = GaussianSimilarityAggregation(alpha=alpha)
        self.out_ch = group_ch * len(kernels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, group_ch, H, W]  →  [B, group_ch*len(kernels), H, W]"""
        outs = []
        for conv in self.convs:
            u = conv(z)
            outs.append(self.agg(u))
        return torch.cat(outs, dim=1)


class FineGrainedMultiScale(nn.Module):
    """
    Split Fin (C channels) into G equal groups.
    Apply FGMSBlock to each group, then concatenate → Fsim.

    Output channels: C * len(kernels)  (because each group is tripled then concat'd)
    → We project back to C so Fsim has same shape as Fin.
    """
    def __init__(self, channels: int, n_groups: int = 4,
                 kernels=(3, 5, 7), alpha: float = 1.0):
        super().__init__()
        assert channels % n_groups == 0, "channels must be divisible by n_groups"
        self.n_groups = n_groups
        group_ch = channels // n_groups

        self.blocks = nn.ModuleList([
            FGMSBlock(group_ch, kernels, alpha) for _ in range(n_groups)
        ])

        sim_ch = group_ch * len(kernels) * n_groups
        # project Fsim back to original channel dim
        self.proj = nn.Sequential(
            nn.Conv2d(sim_ch, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, fin: torch.Tensor) -> torch.Tensor:
        """fin: [B, C, H, W]  →  Fsim: [B, C, H, W]"""
        groups = fin.chunk(self.n_groups, dim=1)
        outs = [blk(g) for blk, g in zip(self.blocks, groups)]
        fsim = torch.cat(outs, dim=1)       # [B, C*len(kernels), H, W]
        return self.proj(fsim)                 # [B, C, H, W]


# ----------------------- Channel & Spatial Attention ----------------------
class ChannelAttention(nn.Module):
    """Eq. (13): wch = σ(Conv1×1(ReLU(Conv1×1(GAP(Fin)))))"""
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        mid = max(channels // ratio, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fin: torch.Tensor) -> torch.Tensor:
        """Returns weight [B, C, 1, 1]."""
        return self.fc(self.gap(fin))


class SpatialAttention(nn.Module):
    """
    Eq. (14): wspat = σ(Conv7×7(Fpool))
    Uses average+max pooling on Fsim, then large-kernel conv.
    """
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, k, padding=k // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fsim: torch.Tensor) -> torch.Tensor:
        """fsim: [B, C, H, W]  →  weight: [B, 1, H, W]"""
        avg = fsim.mean(dim=1, keepdim=True)
        mx = fsim.max(dim=1, keepdim=True).values
        return self.conv(torch.cat([avg, mx], dim=1))


# ---------------------- SSAA for a single modality ------------------------
class SSAABranch(nn.Module):
    """
    SSAA applied to one modality's feature map.
    Fout = (Fin ⊙ wch ⊙ wspat) + Fin
    """
    def __init__(self, channels: int, n_groups: int = 4,
                 kernels=(3, 5, 7), alpha: float = 1.0, spatial_k: int = 7):
        super().__init__()
        self.fg_ms = FineGrainedMultiScale(channels, n_groups, kernels, alpha)
        self.ch_attn = ChannelAttention(channels)
        self.sp_attn = SpatialAttention(spatial_k)

    def forward(self, fin: torch.Tensor):
        """fin: [B, C, H, W]  →  fout: [B, C, H, W]"""
        fsim = self.fg_ms(fin)                          # [B, C, H, W]
        wch = self.ch_attn(fin)                         # [B, C, 1, 1]
        fch = fin * wch                                 # channel-calibrated
        wsp = self.sp_attn(fsim)                        # [B, 1, H, W]
        fcs = fch * wsp                                 # spatial-calibrated
        return fcs + fin                                # residual


# ------------------- SSAA cross-modal fusion at one scale ----------------
class SSAAFusionBlock(nn.Module):
    """
    SSAA cross-modal fusion for one encoder scale.

    Process:
      1. Apply SSAABranch to opt_feat  → opt_enhanced
      2. Apply SSAABranch to dsm_feat  → dsm_enhanced
      3. Fuse: project opt+dsm to a combined skip feature E_s
    """
    def __init__(self, channels: int, n_groups: int = 4,
                 kernels=(3, 5, 7), alpha: float = 1.0, spatial_k: int = 7):
        super().__init__()
        self.opt_ssaa = SSAABranch(channels, n_groups, kernels, alpha, spatial_k)
        self.dsm_ssaa = SSAABranch(channels, n_groups, kernels, alpha, spatial_k)
        # Fuse both modalities → same channel dim (simple add after projection)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, opt_feat: torch.Tensor,
                dsm_feat: torch.Tensor) -> torch.Tensor:
        """
        Returns fused skip feature E_s: [B, C, H, W]
        """
        oe = self.opt_ssaa(opt_feat)
        de = self.dsm_ssaa(dsm_feat)
        return self.fuse_conv(torch.cat([oe, de], dim=1))


# ----------------------- SSAA stack (all 3 encoder scales) ----------------
class SSAAStack(nn.Module):
    """
    Applies SSAAFusionBlock at each of the n_scales encoder stages.
    Input: stage_pairs = [(opt_0, dsm_0), (opt_1, dsm_1), ...]
           ordered shallowest → deepest
    Returns: list of fused skip features [E_s0, E_s1, E_s2]
    """
    def __init__(self, encoder_channels, n_scales: int = 3,
                 n_groups: int = 4, kernels=(3, 5, 7),
                 alpha: float = 1.0, spatial_k: int = 7):
        super().__init__()
        self.blocks = nn.ModuleList([
            SSAAFusionBlock(encoder_channels[i], n_groups, kernels, alpha, spatial_k)
            for i in range(n_scales)
        ])

    def forward(self, stage_pairs):
        assert len(stage_pairs) == len(self.blocks)
        return [blk(o, d) for blk, (o, d) in zip(self.blocks, stage_pairs)]
