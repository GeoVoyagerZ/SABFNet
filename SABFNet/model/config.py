"""
SABFNet configuration.

Encoder:  Dual-branch ResNet50 (from FTransUNet, R50+ViT-B_16 pretrained)
Scales:   root→1/2(64ch), block1→1/4(256ch), block2→1/8(512ch), block3→1/16(1024ch)
SSAA:     applied at 1/2, 1/4, 1/8 scales
BF-FVit:  operates on 1/16 (1024ch) token sequences
Decoder:  3×CFMamba (1/8→1/4→1/2) + final head
"""

import ml_collections


def get_sabfnet_r50_config():
    config = ml_collections.ConfigDict()

    # -------------------------- Encoder (ResNet50) --------------------------
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)     # R50 block units
    config.resnet.width_factor = 1

    config.vis_channels = 3   # optical RGB
    config.dsm_channels = 1   # DSM / TIR

    # Channel dimensions at each encoder stage: [root, block1, block2, block3]
    config.encoder_channels = [64, 256, 512, 1024]

    # ------------------------------- SSAA -----------------------------------
    config.ssaa = ml_collections.ConfigDict()
    config.ssaa.n_groups = 4        # G=4 channel groups
    config.ssaa.kernels = [3, 5, 7]
    config.ssaa.alpha = 1.0         # Gaussian similarity decay
    config.ssaa.spatial_k = 7       # spatial attention kernel size

    # ----------------------------- BF-FVit ----------------------------------
    config.bffvit = ml_collections.ConfigDict()
    config.bffvit.hidden_size = 768    # ViT-B hidden dim
    config.bffvit.mlp_dim = 3072
    config.bffvit.num_heads = 12
    config.bffvit.n_sa_layers = 2      # self-attention only blocks
    config.bffvit.n_ada_layers = 8     # AdaMBA cross-modal blocks
    config.bffvit.n_bfm_layers = 2     # BFMamba blocks
    config.bffvit.dropout_rate = 0.1
    config.bffvit.attn_dropout = 0.0

    # d_state / d_conv / expand for Mamba inside BFMamba
    config.bffvit.d_state = 16
    config.bffvit.d_conv = 4
    config.bffvit.expand = 2

    # Patch grid for token flattening (16×16 grid for 256×256 input at 1/16 scale)
    config.bffvit.patches_grid = (16, 16)

    # -------------------------- CFMamba Decoder -----------------------------
    config.decoder_channels = (256, 128, 64)  # output dims at each decode stage
    config.n_skip = 3                         # number of SSAA skip connections used

    # ------------------------ Segmentation head -----------------------------
    config.n_classes = 6
    config.activation = 'softmax'

    # ------------------------------- Loss -----------------------------------
    config.lambda_ce = 1.0
    config.lambda_dice = 1.0

    # --------------------------- Pretrained ---------------------------------
    config.pretrained_path = './pretrained/R50+ViT-B_16.npz'

    return config


def get_sabfnet_debug_config():
    """Tiny config for unit testing (CPU, small model)."""
    cfg = get_sabfnet_r50_config()

    cfg.resnet.num_layers = (1, 1, 1)

    cfg.bffvit.hidden_size = 64
    cfg.bffvit.mlp_dim = 128
    cfg.bffvit.num_heads = 2
    cfg.bffvit.n_sa_layers = 1
    cfg.bffvit.n_ada_layers = 1
    cfg.bffvit.n_bfm_layers = 1
    cfg.bffvit.d_state = 4

    cfg.decoder_channels = (64, 32, 16)

    return cfg


CONFIGS = {
    'SABFNet-R50': get_sabfnet_r50_config,
    'SABFNet-Debug': get_sabfnet_debug_config,
}
