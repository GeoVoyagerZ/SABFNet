"""
Cross-branch Fusion Mamba (CFMamba) Decoder  —  SABFNet §III-D

At each decoding scale, CFMamba receives:
  • F_skip: skip feature from encoder SSAA  (low-level spatial details)
  • F_up  : upsampled feature from previous decoder stage (high-level semantics)

Processing (Eq. 26-29):
  F_m_proj = LN(F_m_in)                               [m ∈ {skip, up}]
  Z_m      = LN(SSM2D(DWConv(Linear(F_m_proj))))
  O_skip   = F_skip_in + Linear(F_up_proj   ⊙ Z_skip)
  O_up     = F_up_in   + Linear(F_skip_proj ⊙ Z_up)
  F_out    = O_skip + O_up

Three CFMamba blocks:
  Stage 1: CFMamba(E_1/8, Up(D_BFVit)) → U_1/8
  Stage 2: CFMamba(E_1/4, Up(U_1/8))   → U_1/4
  Stage 3: CFMamba(E_1/2, Up(U_1/4))   → U_1/2
Then: Up(U_1/2) → 1×1 SegHead → final logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_utils import SSM2D


# ── Channel adapter (match skip and up channel dims) ────────────────────────
class ChannelAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x): return self.conv(x)


# ── CFMamba block ────────────────────────────────────────────────────────────
class CFMambaBlock(nn.Module):
    """
    One CFMamba block operating on two input branches of the same channel dim.
    """

    def __init__(self, channels: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        # Per-branch: LN → Linear → DWConv → SSM2D → LN
        for branch in ['skip', 'up']:
            setattr(self, f'ln_in_{branch}',  nn.LayerNorm(channels))
            setattr(self, f'linear_{branch}',  nn.Linear(channels, channels))
            setattr(self, f'dw_{branch}',
                    nn.Conv2d(channels, channels, d_conv,
                              padding=d_conv // 2, groups=channels, bias=False))
            setattr(self, f'ssm_{branch}',
                    SSM2D(channels, d_state, d_conv, expand))
            setattr(self, f'ln_out_{branch}', nn.LayerNorm(channels))
            # Cross-branch gating linear
            setattr(self, f'gate_lin_{branch}', nn.Linear(channels, channels))

    def _process_branch(self, name: str, feat: torch.Tensor) -> torch.Tensor:
        """
        Returns Z_m: [B, C, H, W] — processed branch representation
        """
        B, C, H, W = feat.shape
        # LN on last dim
        seq = feat.flatten(2).transpose(1, 2)                           # [B, H*W, C]
        ln  = getattr(self, f'ln_in_{name}')
        lin = getattr(self, f'linear_{name}')
        dw  = getattr(self, f'dw_{name}')
        ssm = getattr(self, f'ssm_{name}')
        ln2 = getattr(self, f'ln_out_{name}')

        proj  = lin(ln(seq))                                             # [B, H*W, C]
        proj2d = proj.transpose(1, 2).view(B, C, H, W)
        conv2d = dw(proj2d)                                              # DWConv
        ssm_out = ssm(conv2d)                                            # SSM2D [B,C,H,W]
        out_seq = ssm_out.flatten(2).transpose(1, 2)                    # [B, H*W, C]
        z_m   = ln2(out_seq).transpose(1, 2).view(B, C, H, W)
        return z_m, proj2d   # return proj2d = F_m_proj as a spatial tensor

    def forward(self, f_skip: torch.Tensor,
                f_up: torch.Tensor) -> torch.Tensor:
        """
        f_skip, f_up: [B, C, H, W]
        Returns: F_fusion_out: [B, C, H, W]
        """
        B, C, H, W = f_skip.shape

        z_skip, proj_skip = self._process_branch('skip', f_skip)
        z_up,   proj_up   = self._process_branch('up',   f_up)

        # Cross-branch gating (Eq. 28-29)
        gate_skip = getattr(self, 'gate_lin_skip')
        gate_up   = getattr(self, 'gate_lin_up')

        # proj_up ⊙ Z_skip  (element-wise, both [B,C,H,W])
        gated_skip = proj_up * z_skip
        gated_skip_seq = gated_skip.flatten(2).transpose(1, 2)          # [B,H*W,C]
        o_skip = f_skip + gate_skip(gated_skip_seq).transpose(1, 2).view(B, C, H, W)

        # proj_skip ⊙ Z_up
        gated_up = proj_skip * z_up
        gated_up_seq = gated_up.flatten(2).transpose(1, 2)
        o_up   = f_up + gate_up(gated_up_seq).transpose(1, 2).view(B, C, H, W)

        return o_skip + o_up


# ── Full cascaded decoder ────────────────────────────────────────────────────
class CFMambaDecoder(nn.Module):
    """
    3-stage CFMamba decoder.

    Channel mapping (default for 256-input ResNet50):
      BF-FVit output D: [B, 1024, H/16, W/16]
      Skip channels (deepest→shallowest): [512, 256, 64]
      Decoder channels: (256, 128, 64)

    Each stage:
      1. Upsample D × 2
      2. Adapt channel of upsampled + skip to common dim
      3. CFMamba
    """

    def __init__(self, deep_channels: int = 1024,
                 skip_channels=(512, 256, 64),
                 decoder_channels=(256, 128, 64),
                 n_classes: int = 6,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        assert len(skip_channels) == len(decoder_channels)
        n = len(decoder_channels)

        self.up_projs    = nn.ModuleList()   # project upsampled feature to dec_ch
        self.skip_projs  = nn.ModuleList()   # project skip feature to dec_ch
        self.cf_blocks   = nn.ModuleList()

        in_ch = deep_channels
        for i in range(n):
            sk_ch  = skip_channels[i]
            dec_ch = decoder_channels[i]
            self.up_projs.append(ChannelAdapter(in_ch, dec_ch))
            self.skip_projs.append(ChannelAdapter(sk_ch, dec_ch))
            self.cf_blocks.append(CFMambaBlock(dec_ch, d_state, d_conv, expand))
            in_ch = dec_ch

        # Segmentation head: final upsample ×2 then ×2 (1/2 → full) → head
        # After 3 CFMamba stages we're at 1/2 scale; one more ×2 → full resolution
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], n_classes, 1),
        )

    def forward(self, bffvit_out: torch.Tensor,
                ssaa_skips: list) -> torch.Tensor:
        """
        Args:
            bffvit_out:  [B, 1024, H/16, W/16]
            ssaa_skips:  list of SSAA fused features ordered deepest→shallowest
                         [E_1/8(512), E_1/4(256), E_1/2(64)]
        Returns:
            seg_logits:  [B, n_classes, H, W]  (full resolution)
        """
        x = bffvit_out
        for up_proj, sk_proj, cf_blk, skip in zip(
                self.up_projs, self.skip_projs, self.cf_blocks, ssaa_skips):
            x     = F.interpolate(x, scale_factor=2,
                                  mode='bilinear', align_corners=True)
            x     = up_proj(x)          # channel adapt upsampled
            s     = sk_proj(skip)       # channel adapt skip
            x     = cf_blk(s, x)        # CFMamba(skip, up)

        # Final ×2 upsample (from 1/2 → full) + segmentation head
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=True)
        return self.seg_head(x)
