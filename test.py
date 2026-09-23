"""
SABFNet Test / Evaluation Script

Usage:
    python test.py --config configs/vaihingen.yaml \
                   --checkpoint checkpoints/vaihingen/best_model.pth

    # Save per-image prediction maps
    python test.py --config configs/vaihingen.yaml \
                   --checkpoint checkpoints/vaihingen/best_model.pth \
                   --save_pred outputs/vaihingen_preds
"""

import os
import argparse
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from model    import build_sabfnet
from losses   import ComboLoss
from datasets import VaihingenDataset, PotsdamDataset, RGBTDataset
from utils    import SegmentationMetrics, get_logger, load_checkpoint


# ── Colour palettes for visualisation ────────────────────────────────────────

VAIHINGEN_PALETTE = [
    [255, 255, 255],   # 0 Impervious surfaces
    [  0,   0, 255],   # 1 Building
    [  0, 255, 255],   # 2 Low vegetation
    [  0, 255,   0],   # 3 Tree
    [255, 255,   0],   # 4 Car
    [255,   0,   0],   # 5 Clutter
]

POTSDAM_PALETTE   = VAIHINGEN_PALETTE

RGBT_PALETTE = [
    [  0,   0,   0],   # 0 Background
    [255,   0,   0],   # 1 Person
    [  0, 255,   0],   # 2 Car
    [  0,   0, 255],   # 3 Bicycle
    [255, 255,   0],   # 4 Other vehicle
    [255,   0, 255],   # 5 Building
    [  0, 255, 255],   # 6 Vegetation
]

PALETTES = {
    'vaihingen': VAIHINGEN_PALETTE,
    'potsdam'  : POTSDAM_PALETTE,
    'rgbt'     : RGBT_PALETTE,
}


def label_to_rgb(label: np.ndarray, palette: list) -> np.ndarray:
    """Convert H×W label map to H×W×3 RGB image."""
    h, w  = label.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, colour in enumerate(palette):
        rgb[label == cls_idx] = colour
    return rgb


def build_dataset(cfg: dict, split: str = 'test'):
    name = cfg['dataset']['name'].lower()
    root = cfg['dataset']['root']
    ps   = cfg['dataset'].get('patch_size', 256)
    st   = cfg['dataset'].get('stride', 128)

    if name == 'vaihingen':
        return VaihingenDataset(root, split=split, patch_size=ps, stride=st,
                                use_eroded=cfg['dataset'].get('use_eroded', False))
    elif name == 'potsdam':
        return PotsdamDataset(root, split=split, patch_size=ps, stride=st,
                              use_eroded=cfg['dataset'].get('use_eroded', False))
    elif name == 'rgbt':
        return RGBTDataset(root, split=split, patch_size=ps)
    else:
        raise ValueError(f'Unknown dataset: {name}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate SABFNet')
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split',      type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--save_pred',  type=str, default=None,
                        help='Directory to save coloured prediction images')
    parser.add_argument('--device',     type=str, default=None,
                        help='Override device (cuda / cpu)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or cfg.get('device', 'cuda')
    device     = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    logger = get_logger('sabfnet_test',
                        log_dir=cfg['training'].get('log_dir', 'logs'))
    logger.info(f'Config    : {args.config}')
    logger.info(f'Checkpoint: {args.checkpoint}')
    logger.info(f'Split     : {args.split}')
    logger.info(f'Device    : {device}')

    # Model
    n_classes = cfg['dataset'].get('n_classes', 6)
    model     = build_sabfnet(cfg['model'].get('config_name', 'SABFNet-R50'),
                              n_classes=n_classes)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model = model.to(device)
    model.eval()
    logger.info(f'Model parameters: {model.n_parameters:,}')

    # Dataset & loader
    ds     = build_dataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    logger.info(f'Evaluation samples: {len(ds)}')

    # Loss (optional — to compute test loss)
    loss_cfg  = cfg.get('loss', {})
    criterion = ComboLoss(
        lambda_ce   =loss_cfg.get('lambda_ce', 1.0),
        lambda_dice =loss_cfg.get('lambda_dice', 1.0),
        ignore_index=loss_cfg.get('ignore_index', 255),
    )

    # Class names
    ds_name = cfg['dataset']['name'].lower()
    from datasets import VAIHINGEN_CLASSES, POTSDAM_CLASSES, RGBT_CLASSES
    cls_map     = {'vaihingen': VAIHINGEN_CLASSES,
                   'potsdam'  : POTSDAM_CLASSES,
                   'rgbt'     : RGBT_CLASSES}
    class_names = cls_map.get(ds_name, None)
    palette     = PALETTES.get(ds_name, VAIHINGEN_PALETTE)

    if args.save_pred:
        os.makedirs(args.save_pred, exist_ok=True)

    metrics    = SegmentationMetrics(n_classes,
                                     ignore_index=loss_cfg.get('ignore_index', 255))
    total_loss = 0.0

    with torch.no_grad():
        for idx, (images, dsm, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            dsm    = dsm.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            logits = model(images, dsm)
            loss, _, _ = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)            # [B, H, W]
            metrics.update(preds, masks)

            # Save coloured prediction
            if args.save_pred:
                pred_np = preds[0].cpu().numpy().astype(np.uint8)
                rgb     = label_to_rgb(pred_np, palette)
                pred_img = Image.fromarray(rgb)
                pred_img.save(os.path.join(args.save_pred, f'pred_{idx:05d}.png'))

    avg_loss = total_loss / len(loader)
    logger.info(f'Test loss: {avg_loss:.4f}')
    results = metrics.print_summary(class_names=class_names, logger=logger)

    # Print final summary line
    logger.info(
        f'\n[FINAL]  OA={results["OA"]:.4f}  '
        f'mF1={results["mF1"]:.4f}  '
        f'mIoU={results["mIoU"]:.4f}'
    )
    return results


if __name__ == '__main__':
    main()
