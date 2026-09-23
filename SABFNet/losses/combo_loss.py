"""
Loss functions for SABFNet  —  §III-E

L_total = λ_ce * L_ce + λ_dice * L_dice
  λ_ce = λ_dice = 1.0  (paper default)

L_ce:   standard cross-entropy over all pixels
L_dice: 1 - (2 * TP) / (pred_sum + gt_sum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Pixel-wise cross-entropy (Eq. 30).
    Ignores class index `ignore_index` (default 255 for void labels).
    """
    def __init__(self, ignore_index: int = 255, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight,
                                      ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, C, H, W]  (raw logits)
        target: [B, H, W]     (long, class indices)
        """
        return self.ce(pred, target)


class DiceLoss(nn.Module):
    """
    Soft Dice loss (Eq. 31).
    Computed per-class and then averaged over foreground classes.
    """
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, C, H, W]
        target: [B, H, W]
        """
        n_classes = pred.shape[1]
        prob = F.softmax(pred, dim=1)                       # [B, C, H, W]

        # One-hot encode target; ignore void pixels
        valid_mask = (target != self.ignore_index)          # [B, H, W]
        target_clipped = target.clone()
        target_clipped[~valid_mask] = 0                     # avoid OOB index

        one_hot = F.one_hot(target_clipped, n_classes)      # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()       # [B, C, H, W]
        # Zero out void positions
        one_hot = one_hot * valid_mask.unsqueeze(1).float()
        prob    = prob    * valid_mask.unsqueeze(1).float()

        # Per-class Dice
        dims   = (0, 2, 3)
        inter  = (prob * one_hot).sum(dims)                 # [C]
        union  = prob.sum(dims) + one_hot.sum(dims)         # [C]
        dice_c = (2 * inter + self.smooth) / (union + self.smooth)

        return 1.0 - dice_c.mean()


class ComboLoss(nn.Module):
    """
    Combined CE + Dice loss.
    L_total = λ_ce * L_ce + λ_dice * L_dice
    """
    def __init__(self, lambda_ce: float = 1.0,
                 lambda_dice: float = 1.0,
                 ignore_index: int = 255,
                 class_weights=None):
        super().__init__()
        self.ce   = CrossEntropyLoss(ignore_index=ignore_index,
                                     weight=class_weights)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.lambda_ce   = lambda_ce
        self.lambda_dice = lambda_dice

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor):
        """
        Returns: (total_loss, ce_loss, dice_loss)
        """
        ce_loss   = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        total     = self.lambda_ce * ce_loss + self.lambda_dice * dice_loss
        return total, ce_loss, dice_loss
