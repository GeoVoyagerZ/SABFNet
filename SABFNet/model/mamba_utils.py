"""
Mamba / SSM utilities for SABFNet.

Provides:
  MambaSSM  — wraps mamba_ssm.Mamba if available, else a simple recurrent SSM
  SSM2D     — 2-D selective scan (serialises feature map, runs SSM, reshapes)

Install (requires CUDA):
    pip install mamba-ssm causal-conv1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaOp
    _HAS_MAMBA = True
except ImportError:
    _MambaOp = None
    _HAS_MAMBA = False


# ---------------- Simple recurrent SSM (CPU fallback) ----------------
class _SimpleSSM(nn.Module):
    """Minimal S4-style SSM. Not optimised — for debugging / CPU-only envs."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)   # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        residual = x
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        xc = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xc = self.act(xc)

        params = self.x_proj(xc)
        Bs, _, dt = params.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt) + 0.5)
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * Bs.unsqueeze(-2)

        h, ys = torch.zeros(B, x_in.shape[-1], self.d_state, device=x.device), []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * xc[:, t].unsqueeze(-1)
            ys.append((h * Bs[:, t].unsqueeze(-2)).sum(-1))
        y = torch.stack(ys, dim=1) + xc * self.D
        y = y * self.act(z)
        return self.norm(self.out_proj(y) + residual)


# ---------------- MambaSSM wrapper ----------------
class MambaSSM(nn.Module):
    """
    Drop-in 1-D SSM block. Uses mamba_ssm if installed, else _SimpleSSM.
    Input/output: [B, L, d_model]
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if _HAS_MAMBA:
            self.ssm = _MambaOp(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        else:
            self.ssm = _SimpleSSM(d_model, d_state=d_state, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(x)


# ---------------- 2-D selective scan (SSM2D) ----------------
class SSM2D(nn.Module):
    """
    2-D state-space model block for spatial feature maps.
    Serialises [B, C, H, W] → [B, H*W, C], runs MambaSSM, reshapes back.

    For better coverage, we scan in 4 directions and sum:
      row-major, col-major, row-major reversed, col-major reversed.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ssm_row = MambaSSM(d_model, d_state, d_conv, expand)
        self.ssm_col = MambaSSM(d_model, d_state, d_conv, expand)
        self.ssm_rrow = MambaSSM(d_model, d_state, d_conv, expand)
        self.ssm_rcol = MambaSSM(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]  →  [B, C, H, W]"""
        B, C, H, W = x.shape

        # row-major scan
        seq_row = x.flatten(2).transpose(1, 2)              # [B, H*W, C]
        y_row = self.ssm_row(seq_row)

        # col-major scan (transpose H,W before flatten)
        seq_col = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)
        y_col = self.ssm_col(seq_col)
        # un-transpose spatial dims back
        y_col = y_col.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2)
        y_col = y_col.flatten(2).transpose(1, 2)

        # reversed scans
        y_rrow = self.ssm_rrow(seq_row.flip(1)).flip(1)
        y_rcol = self.ssm_rcol(seq_col.flip(1)).flip(1)
        y_rcol = y_rcol.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2)
        y_rcol = y_rcol.flatten(2).transpose(1, 2)

        out = self.norm(y_row + y_col + y_rrow + y_rcol)   # [B, H*W, C]
        return out.transpose(1, 2).view(B, C, H, W)
