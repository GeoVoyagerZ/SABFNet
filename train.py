"""
SABFNet Training Script

Usage:
    python train.py --config configs/vaihingen.yaml
    python train.py --config configs/potsdam.yaml
    python train.py --config configs/rgbt.yaml

    # Resume from checkpoint
    python train.py --config configs/vaihingen.yaml --resume checkpoints/vaihingen/epoch_020.pth
"""

import os
import sys
import argparse
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Local imports ─────────────────────────────────────────────────────────────
from model    import build_sabfnet, CONFIGS
from losses   import ComboLoss
from datasets import VaihingenDataset, PotsdamDataset, RGBTDataset
from utils    import (SegmentationMetrics, get_logger, CSVLogger,
                      save_checkpoint, load_checkpoint, save_best_checkpoint)


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, split: str):
    name = cfg['dataset']['name'].lower()
    root = cfg['dataset']['root']
    ps   = cfg['dataset'].get('patch_size', 256)
    st   = cfg['dataset'].get('stride', 128)

    if name == 'vaihingen':
        return VaihingenDataset(root, split=split, patch_size=ps, stride=st,
                                use_eroded=cfg['dataset'].get('use_eroded', True))
    elif name == 'potsdam':
        return PotsdamDataset(root, split=split, patch_size=ps, stride=st,
                              use_eroded=cfg['dataset'].get('use_eroded', True))
    elif name == 'rgbt':
        return RGBTDataset(root, split=split, patch_size=ps,
                           val_ratio=cfg['dataset'].get('val_ratio', 0.3))
    else:
        raise ValueError(f'Unknown dataset: {name}')


def build_model(cfg: dict, device):
    config_name = cfg['model'].get('config_name', 'SABFNet-R50')
    n_classes   = cfg['dataset'].get('n_classes', 6)

    model = build_sabfnet(config_name, n_classes=n_classes)

    if cfg['model'].get('pretrained', False):
        pretrained_path = cfg['model'].get('pretrained_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            model.load_pretrained(pretrained_path)
        else:
            print(f'[WARNING] pretrained_path not found: {pretrained_path}. '
                  'Training from scratch.')

    return model.to(device)


def build_optimizer(cfg: dict, model):
    opt_cfg = cfg['optimizer']
    lr      = opt_cfg.get('lr', 1e-4)
    wd      = opt_cfg.get('weight_decay', 0.01)
    betas   = tuple(opt_cfg.get('betas', [0.9, 0.999]))
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=wd, betas=betas)


def build_scheduler(cfg: dict, optimizer, last_epoch=-1):
    sch_cfg  = cfg['scheduler']
    sch_type = sch_cfg.get('type', 'cosine').lower()
    epochs   = cfg['training']['epochs']

    if sch_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = sch_cfg.get('T_max', epochs),
            eta_min = sch_cfg.get('eta_min', 1e-6),
            last_epoch=last_epoch,
        )
    elif sch_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get('step_size', 20),
            gamma    =sch_cfg.get('gamma', 0.1),
            last_epoch=last_epoch,
        )
    elif sch_type == 'poly':
        total = epochs
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: (1 - e / total) ** sch_cfg.get('power', 0.9),
            last_epoch=last_epoch,
        )
    else:
        return None


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    device, epoch, cfg, logger):
    model.train()
    total_loss = 0.0
    total_ce   = 0.0
    total_dice = 0.0
    log_every  = cfg['training'].get('log_every', 10)
    n_batches  = len(loader)

    for i, (images, dsm, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        dsm    = dsm.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, dsm)                         # [B, C, H, W]

        loss, ce_loss, dice_loss = criterion(logits, masks)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce   += ce_loss.item()
        total_dice += dice_loss.item()

        if (i + 1) % log_every == 0 or i == n_batches - 1:
            avg_loss = total_loss / (i + 1)
            logger.info(
                f'Epoch [{epoch}][{i+1}/{n_batches}] '
                f'Loss: {loss.item():.4f}  CE: {ce_loss.item():.4f}  '
                f'Dice: {dice_loss.item():.4f}  AvgLoss: {avg_loss:.4f}'
            )

    n = len(loader)
    return {'loss': total_loss / n, 'ce': total_ce / n, 'dice': total_dice / n}


# ── Validation epoch ──────────────────────────────────────────────────────────

def validate(model, loader, criterion, device, n_classes,
             ignore_index=255, class_names=None, logger=None):
    model.eval()
    metrics = SegmentationMetrics(n_classes, ignore_index=ignore_index)
    total_loss = 0.0

    with torch.no_grad():
        for images, dsm, masks in loader:
            images = images.to(device, non_blocking=True)
            dsm    = dsm.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            logits = model(images, dsm)
            loss, _, _ = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)

    avg_loss = total_loss / len(loader)
    res      = metrics.compute()
    res['val_loss'] = avg_loss

    if logger:
        metrics.print_summary(class_names=class_names, logger=logger)
        logger.info(f'Val loss: {avg_loss:.4f}  '
                    f'OA: {res["OA"]:.4f}  '
                    f'mF1: {res["mF1"]:.4f}  '
                    f'mIoU: {res["mIoU"]:.4f}')
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train SABFNet')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))

    device = torch.device(cfg.get('device', 'cuda')
                          if torch.cuda.is_available() else 'cpu')

    # Logging
    log_dir = cfg['training'].get('log_dir', 'logs')
    logger  = get_logger('sabfnet_train', log_dir=log_dir)
    logger.info(f'Config: {args.config}')
    logger.info(f'Device: {device}')

    # CSV logger
    csv_fields = ['epoch', 'lr', 'train_loss', 'train_ce', 'train_dice',
                  'val_loss', 'val_oa', 'val_mf1', 'val_miou']
    csv_log = CSVLogger(os.path.join(log_dir, 'metrics.csv'), csv_fields)

    # Datasets & loaders
    dl_cfg   = cfg['dataloader']
    train_ds = build_dataset(cfg, 'train')
    val_ds   = build_dataset(cfg, 'val')

    train_loader = DataLoader(
        train_ds,
        batch_size =dl_cfg.get('batch_size', 1),
        shuffle    =True,
        num_workers=dl_cfg.get('num_workers', 4),
        pin_memory =dl_cfg.get('pin_memory', True),
        drop_last  =True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size =1,
        shuffle    =False,
        num_workers=dl_cfg.get('num_workers', 4),
        pin_memory =dl_cfg.get('pin_memory', True),
    )

    logger.info(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}')

    # Model
    model = build_model(cfg, device)
    logger.info(f'Model parameters: {model.n_parameters:,}')

    # Loss
    loss_cfg  = cfg.get('loss', {})
    criterion = ComboLoss(
        lambda_ce   =loss_cfg.get('lambda_ce', 1.0),
        lambda_dice =loss_cfg.get('lambda_dice', 1.0),
        ignore_index=loss_cfg.get('ignore_index', 255),
    )

    # Optimiser & Scheduler
    optimizer = build_optimizer(cfg, model)
    n_classes = cfg['dataset'].get('n_classes', 6)

    # Class names for display
    ds_name = cfg['dataset']['name'].lower()
    from datasets import VAIHINGEN_CLASSES, POTSDAM_CLASSES, RGBT_CLASSES
    cls_map  = {'vaihingen': VAIHINGEN_CLASSES,
                'potsdam'  : POTSDAM_CLASSES,
                'rgbt'     : RGBT_CLASSES}
    class_names = cls_map.get(ds_name, None)

    start_epoch = 1
    best_miou   = 0.0

    # Resume
    if args.resume:
        info = load_checkpoint(args.resume, model, optimizer, device=str(device))
        start_epoch = info.get('epoch', 0) + 1
        best_miou   = info.get('metrics', {}).get('mIoU', 0.0)
        logger.info(f'Resumed from epoch {start_epoch - 1}, best mIoU={best_miou:.4f}')

    scheduler = build_scheduler(cfg, optimizer, last_epoch=start_epoch - 2)

    total_epochs = cfg['training']['epochs']
    val_every    = cfg['training'].get('val_every', 1)
    save_dir     = cfg['training'].get('save_dir', 'checkpoints')

    logger.info(f'Training for {total_epochs} epochs ...')

    for epoch in range(start_epoch, total_epochs + 1):
        logger.info(f'─── Epoch {epoch}/{total_epochs}  '
                    f'lr={optimizer.param_groups[0]["lr"]:.2e} ───')

        train_res = train_one_epoch(model, train_loader, criterion,
                                    optimizer, device, epoch, cfg, logger)

        if scheduler is not None:
            scheduler.step()

        val_res = {'val_loss': None, 'OA': None, 'mF1': None, 'mIoU': None}
        is_best = False

        if epoch % val_every == 0:
            val_res = validate(model, val_loader, criterion, device,
                               n_classes, class_names=class_names, logger=logger)
            if val_res['mIoU'] > best_miou:
                best_miou = val_res['mIoU']
                is_best   = True
                logger.info(f'★ New best mIoU: {best_miou:.4f}')

        # Save checkpoint
        state = {
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'metrics'  : val_res,
        }
        save_checkpoint(state, save_dir,
                        filename=f'epoch_{epoch:03d}.pth',
                        is_best=is_best)

        # CSV row
        csv_log.write({
            'epoch'      : epoch,
            'lr'         : optimizer.param_groups[0]['lr'],
            'train_loss' : f'{train_res["loss"]:.6f}',
            'train_ce'   : f'{train_res["ce"]:.6f}',
            'train_dice' : f'{train_res["dice"]:.6f}',
            'val_loss'   : f'{val_res.get("val_loss", ""):.6f}'
                            if val_res.get('val_loss') is not None else '',
            'val_oa'     : f'{val_res.get("OA", ""):.6f}'
                            if val_res.get('OA') is not None else '',
            'val_mf1'    : f'{val_res.get("mF1", ""):.6f}'
                            if val_res.get('mF1') is not None else '',
            'val_miou'   : f'{val_res.get("mIoU", ""):.6f}'
                            if val_res.get('mIoU') is not None else '',
        })

    logger.info(f'Training complete. Best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    main()
