"""
BF-FVit: Bidirectional Fusion Vision Transformer  —  SABFNet §III-C

Three-stage structure:
  Stage 1  (N_sa  blocks): Self-attention — intra-modal context enhancement
  Stage 2  (N_ada blocks): AdaMBA         — adaptive cross-modal fusion
  Stage 3  (N_bfm blocks): BFMamba        — bidirectional local-global integration

Equations:
  SA  (Eq.15-16): standard Transformer encoder block
  AdaMBA (Eq.17-19):
      SAx = SA(Zx), CAx = CA(Zx, Zy)
      SAy = SA(Zy), CAy = CA(Zy, Zx)
      Gx = λ_sa_x * SAx + λ_ca_x * CAx  (learnable λ)
      Gy = λ_sa_y * SAy + λ_ca_y * CAy
      Zx = Zx + MLP(LN(Gx + Zx))
  BFMamba (Eq.20-25):
      Ux = DWConv(Linear(Zx_last)), Uy = DWConv(Linear(Zy_last))
      Sfwd = Concat_seq(Flat(Ux), Flat(Uy))
      Sinv = Reverse(Sfwd)
      Y = SSM(Sfwd) + SSM(Sinv)
      Vx, Vy = split Y; scale-gate each; Zfuse = Concat(Vx, Vy)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_utils import MambaSSM


# ── helpers ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, in_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads,
                                          dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L, D]"""
        out, _ = self.attn(x, x, x)
        return self.drop(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads,
                                          dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_src, kv_src):
        """q_src, kv_src: [B, L, D]"""
        out, _ = self.attn(q_src, kv_src, kv_src)
        return self.drop(out)


# ── Encoder (projection 1024 → hidden_size) ─────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Flatten spatial feature map to token sequence + linear projection.
    Optionally adds learnable positional embeddings.
    """
    def __init__(self, in_channels, hidden_size, n_patches):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_size)
        self.pos  = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W]  →  tokens: [B, H*W, hidden_size]"""
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # [B, H*W, C]
        tokens = self.proj(tokens)               # [B, H*W, hidden]
        if tokens.shape[1] == self.pos.shape[1]:
            tokens = tokens + self.pos
        else:
            # interpolate positional embedding if size mismatch
            pos = self.pos.transpose(1, 2).view(1, -1, *[int(math.sqrt(self.pos.shape[1]))] * 2)
            pos = F.interpolate(pos, size=(H, W), mode='bilinear', align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
            tokens = tokens + pos
        return tokens, H, W


# ── Stage 1: Self-Attention Block ───────────────────────────────────────────
class SABlock(nn.Module):
    """Standard Transformer encoder block (Eq. 15)."""

    def __init__(self, hidden, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.sa  = MultiHeadSelfAttention(hidden, heads, dropout)
        self.mlp = MLP(hidden, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SAStage(nn.Module):
    def __init__(self, n_layers, hidden, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SABlock(hidden, heads, mlp_dim, dropout) for _ in range(n_layers)
        ])
        self.layers_y = nn.ModuleList([
            SABlock(hidden, heads, mlp_dim, dropout) for _ in range(n_layers)
        ])

    def forward(self, zx, zy):
        for lx, ly in zip(self.layers, self.layers_y):
            zx = lx(zx)
            zy = ly(zy)
        return zx, zy


# ── Stage 2: AdaMBA (Adaptive Mutual Promotion Attention) ───────────────────
class AdaMBABlock(nn.Module):
    """
    Eq. (17-19):
      SAx = SA(Zx), CAx = CA(Zx, Zy)
      Gx = λ_sa_x * SAx + λ_ca_x * CAx
      Zx ← Zx + MLP(LN(Gx + Zx))
    """

    def __init__(self, hidden, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln_x1 = nn.LayerNorm(hidden)
        self.ln_y1 = nn.LayerNorm(hidden)
        self.ln_x2 = nn.LayerNorm(hidden)
        self.ln_y2 = nn.LayerNorm(hidden)

        self.sa_x  = MultiHeadSelfAttention(hidden, heads, dropout)
        self.sa_y  = MultiHeadSelfAttention(hidden, heads, dropout)
        self.ca_xy = MultiHeadCrossAttention(hidden, heads, dropout)  # q=x, kv=y
        self.ca_yx = MultiHeadCrossAttention(hidden, heads, dropout)  # q=y, kv=x

        # Learnable fusion weights (initialised equally)
        self.lambda_sa_x = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_ca_x = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_sa_y = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_ca_y = nn.Parameter(torch.ones(1) * 0.5)

        self.mlp_x = MLP(hidden, mlp_dim, dropout)
        self.mlp_y = MLP(hidden, mlp_dim, dropout)

    def forward(self, zx, zy):
        zx_n, zy_n = self.ln_x1(zx), self.ln_y1(zy)

        sax = self.sa_x(zx_n)
        cax = self.ca_xy(zx_n, zy_n)
        say = self.sa_y(zy_n)
        cay = self.ca_yx(zy_n, zx_n)

        gx = self.lambda_sa_x * sax + self.lambda_ca_x * cax
        gy = self.lambda_sa_y * say + self.lambda_ca_y * cay

        # Eq (19)
        zx = zx + self.mlp_x(self.ln_x2(gx + zx))
        zy = zy + self.mlp_y(self.ln_y2(gy + zy))
        return zx, zy


class AdaMBAStage(nn.Module):
    def __init__(self, n_layers, hidden, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AdaMBABlock(hidden, heads, mlp_dim, dropout) for _ in range(n_layers)
        ])

    def forward(self, zx, zy):
        for blk in self.layers:
            zx, zy = blk(zx, zy)
        return zx, zy


# ── Stage 3: BFMamba (Bidirectional Fusion Mamba) ───────────────────────────
class BFMambaBlock(nn.Module):
    """
    Eq. (20-25):
      Ux = DWConv(Linear(Zx)), Uy = DWConv(Linear(Zy))
      Sfwd = Concat_seq(Flat(Ux), Flat(Uy))
      Sinv = Reverse(Sfwd)
      Y    = SSM(Sfwd) + SSM_inv(Sinv) [un-flipped]
      Vx, Vy = split(Y); scale-gate with g_m = Scale(Linear(Zx, Zy))
      Zfuse  = Concat(Vx, Vy)
    """

    def __init__(self, hidden, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.linear_x = nn.Linear(hidden, hidden)
        self.linear_y = nn.Linear(hidden, hidden)
        # DWConv on flattened sequence treated as 1-D (channel → d_inner is done inside ssm)
        self.dw_x = nn.Conv1d(hidden, hidden, d_conv, padding=d_conv // 2,
                               groups=hidden, bias=False)
        self.dw_y = nn.Conv1d(hidden, hidden, d_conv, padding=d_conv // 2,
                               groups=hidden, bias=False)

        self.ssm_fwd = MambaSSM(hidden, d_state, d_conv, expand)
        self.ssm_inv = MambaSSM(hidden, d_state, d_conv, expand)

        # Scale gate: input = cat(Zx, Zy) along feature dim → 2*hidden → hidden
        self.gate_x = nn.Linear(hidden * 2, hidden)
        self.gate_y = nn.Linear(hidden * 2, hidden)
        self.norm   = nn.LayerNorm(hidden * 2)

    def forward(self, zx, zy):
        """zx, zy: [B, L, D]  →  zfuse: [B, L, 2D]"""
        B, L, D = zx.shape

        # Eq (20): Linear + DWConv
        ux = self.dw_x(self.linear_x(zx).transpose(1, 2))[:, :, :L].transpose(1, 2)
        uy = self.dw_y(self.linear_y(zy).transpose(1, 2))[:, :, :L].transpose(1, 2)

        # Eq (21): concat along sequence dim
        sfwd = torch.cat([ux, uy], dim=1)          # [B, 2L, D]

        # Eq (22-23): state-space processing
        sinv = sfwd.flip(1)                          # reverse
        y_fwd = self.ssm_fwd(sfwd)                  # [B, 2L, D]
        y_inv = self.ssm_inv(sinv).flip(1)           # [B, 2L, D], un-reverse
        y = y_fwd + y_inv                            # [B, 2L, D]

        # Split back into two halves (Vx, Vy)
        vx, vy = y[:, :L], y[:, L:]                 # each [B, L, D]

        # Eq (24): scale gates
        gate_in = torch.cat([zx, zy], dim=-1)        # [B, L, 2D]
        gx = torch.sigmoid(self.gate_x(gate_in))     # [B, L, D]
        gy = torch.sigmoid(self.gate_y(gate_in))

        vx = gx * vx
        vy = gy * vy

        # Eq (25): concat modalities
        zfuse = torch.cat([vx, vy], dim=-1)          # [B, L, 2D]
        return self.norm(zfuse)


class BFMambaStage(nn.Module):
    """
    N_bfm BFMamba blocks.
    Each block merges the two modalities into a joint [B, L, 2D] representation.
    We project back to [B, L, D] × 2 so subsequent blocks get same-shape inputs.
    """
    def __init__(self, n_layers, hidden, d_state=16, d_conv=4,
                 expand=2, dropout=0.1):
        super().__init__()
        self.layers    = nn.ModuleList([
            BFMambaBlock(hidden, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        # Project fused [2D] back to [D] per modality for residual
        self.proj_back = nn.Linear(hidden * 2, hidden)

    def forward(self, zx, zy):
        """Returns fused token sequence: [B, L, hidden]"""
        for blk in self.layers:
            fused = blk(zx, zy)          # [B, L, 2*hidden]
            # Residual: project fused back and split to update both branches
            proj  = self.proj_back(fused)  # [B, L, hidden]
            zx    = zx + proj
            zy    = zy + proj
        # Return fused representation combining both modalities
        fused = blk(zx, zy)              # last block output [B, L, 2*hidden]
        return self.proj_back(fused)     # [B, L, hidden]


# ── Full BF-FVit ─────────────────────────────────────────────────────────────
class BFFVit(nn.Module):
    """
    Bidirectional Fusion Vision Transformer.

    Input:  opt_deep, dsm_deep  — [B, 1024, H/16, W/16]
    Output: fused               — [B, 1024, H/16, W/16]  (same spatial size)
    """

    def __init__(self, in_channels: int,
                 hidden_size: int, mlp_dim: int, num_heads: int,
                 n_sa: int, n_ada: int, n_bfm: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1, attn_dropout: float = 0.0,
                 patches_grid=(16, 16)):
        super().__init__()
        n_patches = patches_grid[0] * patches_grid[1]

        # Project ResNet feature maps to ViT hidden dim
        self.embed_opt = PatchEmbedding(in_channels, hidden_size, n_patches)
        self.embed_dsm = PatchEmbedding(in_channels, hidden_size, n_patches)

        # Three stages
        self.sa_stage  = SAStage(n_sa, hidden_size, num_heads, mlp_dim, dropout)
        self.ada_stage = AdaMBAStage(n_ada, hidden_size, num_heads, mlp_dim, dropout)
        self.bfm_stage = BFMambaStage(n_bfm, hidden_size, d_state, d_conv, expand, dropout)

        self.norm = nn.LayerNorm(hidden_size)

        # Project hidden_size back to in_channels for skip / decoder compatibility
        self.out_proj = nn.Linear(hidden_size, in_channels)

    def forward(self, opt_deep: torch.Tensor,
                dsm_deep: torch.Tensor) -> torch.Tensor:
        """
        opt_deep, dsm_deep: [B, 1024, H/16, W/16]
        Returns: fused [B, 1024, H/16, W/16]
        """
        B, C, H, W = opt_deep.shape

        # Embed
        zx, _, _ = self.embed_opt(opt_deep)   # [B, H*W, hidden]
        zy, _, _ = self.embed_dsm(dsm_deep)

        # Stage 1: SA (intra-modal)
        zx, zy = self.sa_stage(zx, zy)

        # Stage 2: AdaMBA (cross-modal adaptive)
        zx, zy = self.ada_stage(zx, zy)

        # Stage 3: BFMamba (bidirectional fusion)
        z_fused = self.bfm_stage(zx, zy)        # [B, L, hidden]

        z_fused = self.norm(z_fused)
        out_seq = self.out_proj(z_fused)         # [B, L, C]
        out     = out_seq.transpose(1, 2).view(B, C, H, W)
        return out
