"""
SABFNet — Same-Scale Attention Aggregation and Bidirectional Mamba Fusion
for Multimodal Remote Sensing Semantic Segmentation.

Architecture (see paper Fig. 2):
  ┌─ Dual-branch ResNet50 encoder
  │    ├── root   → 1/2  (64ch)   ──┐
  │    ├── block1 → 1/4  (256ch)  ──┼── SSAA fusion (E_1/2, E_1/4, E_1/8)
  │    ├── block2 → 1/8  (512ch)  ──┘
  │    └── block3 → 1/16 (1024ch) ──→ BF-FVit
  │
  ├─ BF-FVit (Stage 1 SA + Stage 2 AdaMBA + Stage 3 BFMamba)
  │      output: D [B,1024,H/16,W/16]
  │
  └─ CFMamba Decoder (3 stages + seg head)
         Stage1: CFMamba(E_1/8, Up(D))   → U_1/8
         Stage2: CFMamba(E_1/4, Up(U_1/8)) → U_1/4
         Stage3: CFMamba(E_1/2, Up(U_1/4)) → U_1/2
         Final:  Up(U_1/2) → 1×1 conv → segmentation map
"""

import torch
import torch.nn as nn

from .resnet_encoder   import DualBranchEncoder
from .ssaa             import SSAAStack
from .bffvit           import BFFVit
from .cfmamba_decoder  import CFMambaDecoder


class SABFNet(nn.Module):
    """
    Full SABFNet model.

    Args:
        config: ml_collections.ConfigDict from model/configs.py
    """

    def __init__(self, config):
        super().__init__()
        cfg = config

        # ── 1. Dual-branch ResNet50 encoder ──────────────────────────────
        self.encoder = DualBranchEncoder(
            block_units  = cfg.resnet.num_layers,
            width_factor = cfg.resnet.width_factor,
            vis_ch       = cfg.vis_channels,
            dsm_ch       = cfg.dsm_channels,
        )
        enc_ch = cfg.encoder_channels   # [64, 256, 512, 1024]

        # ── 2. SSAA at scales 1/2, 1/4, 1/8 ────────────────────────────
        self.ssaa = SSAAStack(
            encoder_channels = enc_ch,
            n_scales  = cfg.n_skip,       # 3
            n_groups  = cfg.ssaa.n_groups,
            kernels   = list(cfg.ssaa.kernels),
            alpha     = cfg.ssaa.alpha,
            spatial_k = cfg.ssaa.spatial_k,
        )

        # ── 3. BF-FVit (deep feature fusion) ─────────────────────────────
        self.bffvit = BFFVit(
            in_channels = enc_ch[-1],      # 1024
            hidden_size = cfg.bffvit.hidden_size,
            mlp_dim     = cfg.bffvit.mlp_dim,
            num_heads   = cfg.bffvit.num_heads,
            n_sa        = cfg.bffvit.n_sa_layers,
            n_ada       = cfg.bffvit.n_ada_layers,
            n_bfm       = cfg.bffvit.n_bfm_layers,
            d_state     = cfg.bffvit.d_state,
            d_conv      = cfg.bffvit.d_conv,
            expand      = cfg.bffvit.expand,
            dropout     = cfg.bffvit.dropout_rate,
            attn_dropout= cfg.bffvit.attn_dropout,
            patches_grid= tuple(cfg.bffvit.patches_grid),
        )

        # ── 4. CFMamba decoder ────────────────────────────────────────────
        # Skip channels are in deepest→shallowest order: [block2(512), block1(256), root(64)]
        # (Because SSAA fuses opt+dsm → fused channels equal to per-branch channels)
        self.decoder = CFMambaDecoder(
            deep_channels   = enc_ch[-1],
            skip_channels   = tuple(enc_ch[i] for i in range(cfg.n_skip - 1, -1, -1)),
            decoder_channels= tuple(cfg.decoder_channels),
            n_classes       = cfg.n_classes,
            d_state         = cfg.bffvit.d_state,
            d_conv          = cfg.bffvit.d_conv,
            expand          = cfg.bffvit.expand,
        )

        self._pretrained_path = getattr(cfg, 'pretrained_path', None)
        self.n_classes = cfg.n_classes

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, opt: torch.Tensor,
                dsm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opt: [B, 3, H, W]   optical / RGB image
            dsm: [B, 1, H, W]   DSM or TIR image
        Returns:
            seg_logits: [B, n_classes, H, W]
        """
        # 1. Encoder
        opt_deep, dsm_deep, opt_skips, dsm_skips, stage_pairs = \
            self.encoder(opt, dsm)
        # opt_skips / dsm_skips: [block2(512,H/8), block1(256,H/4), root(64,H/2)]
        # stage_pairs: [(root_opt,root_dsm), (b1_opt,b1_dsm), (b2_opt,b2_dsm)]

        # 2. SSAA — fuse at each shallow scale
        ssaa_feats = self.ssaa(stage_pairs)
        # ssaa_feats: [E_1/2(64), E_1/4(256), E_1/8(512)]  shallowest→deepest

        # 3. BF-FVit — deep fusion
        bffvit_out = self.bffvit(opt_deep, dsm_deep)   # [B,1024,H/16,W/16]

        # 4. CFMamba Decoder
        # Decoder expects skips in deepest→shallowest order: [E_1/8, E_1/4, E_1/2]
        ssaa_skips_dec = list(reversed(ssaa_feats))    # [E_1/8, E_1/4, E_1/2]

        seg_logits = self.decoder(bffvit_out, ssaa_skips_dec)
        return seg_logits

    # ─────────────────────────────────────────────────────────────────────
    def load_pretrained(self, path: str = None):
        """Load R50+ViT-B_16 pretrained weights into the encoder."""
        path = path or self._pretrained_path
        if path is None:
            raise ValueError('No pretrained path specified.')
        self.encoder.load_pretrained(path)
        print(f'[SABFNet] Loaded pretrained encoder from: {path}')

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Convenience factory ──────────────────────────────────────────────────────

def build_sabfnet(config_name: str = 'SABFNet-R50',
                  n_classes: int = None,
                  pretrained: bool = False) -> SABFNet:
    from .configs import CONFIGS
    cfg = CONFIGS[config_name]()
    if n_classes is not None:
        cfg.n_classes = n_classes
    model = SABFNet(cfg)
    if pretrained:
        model.load_pretrained()
    return model
