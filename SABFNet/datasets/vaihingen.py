"""
ISPRS Vaihingen Dataset Loader  —  SABFNet

Directory structure expected:
  root/
    ISPRS_semantic_labeling_Vaihingen/
      top/        *.tif   — IRRG images (3-band)
      dsm/        *.tif   — normalised DSM
      gts_for_participants/  *.tif — ground truth (RGB colour-coded)
      gts_eroded_for_participants/   *.tif  (optional, boundary-eroded GT)

Paper train/val split (area IDs):
  train: 1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37
  val  : 2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38

Classes (6):
  0 = Impervious surfaces  (RGB 255,255,255)
  1 = Building             (RGB 0,0,255)
  2 = Low vegetation       (RGB 0,255,255)
  3 = Tree                 (RGB 0,255,0)
  4 = Car                  (RGB 255,255,0)
  5 = Clutter/background   (RGB 255,0,0)
"""

import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


# ── Colour → class index mapping ─────────────────────────────────────────────

VAIHINGEN_COLOUR_MAP = {
    (255, 255, 255): 0,  # Impervious surfaces
    (  0,   0, 255): 1,  # Building
    (  0, 255, 255): 2,  # Low vegetation
    (  0, 255,   0): 3,  # Tree
    (255, 255,   0): 4,  # Car
    (255,   0,   0): 5,  # Clutter / background
}

VAIHINGEN_CLASSES = [
    'Impervious surfaces',
    'Building',
    'Low vegetation',
    'Tree',
    'Car',
    'Clutter',
]

TRAIN_IDS = [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]
VAL_IDS   = [2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]


def _colour_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an H×W×3 uint8 RGB mask to an H×W int64 label map."""
    h, w = mask_rgb.shape[:2]
    label = np.full((h, w), 255, dtype=np.int64)          # 255 = ignore / void
    for rgb, cls in VAIHINGEN_COLOUR_MAP.items():
        r, g, b = rgb
        match = (mask_rgb[:, :, 0] == r) & \
                (mask_rgb[:, :, 1] == g) & \
                (mask_rgb[:, :, 2] == b)
        label[match] = cls
    return label


class VaihingenDataset(Dataset):
    """
    Vaihingen dataset with random patch extraction during training.

    Args:
        root       : path to dataset root
        split      : 'train' | 'val' | 'test'
        patch_size : crop size (default 256)
        stride     : sliding-window stride for 'test' split
        use_eroded : use boundary-eroded GT (removes 3-px boundary)
        transforms : override default transforms
    """

    def __init__(self, root: str,
                 split: str = 'train',
                 patch_size: int = 256,
                 stride: int = 128,
                 use_eroded: bool = True,
                 transforms=None):
        super().__init__()
        self.root       = root
        self.split      = split
        self.patch_size = patch_size
        self.stride     = stride

        if transforms is not None:
            self.transforms = transforms
        elif split == 'train':
            self.transforms = get_train_transforms(patch_size)
        elif split == 'val':
            self.transforms = get_val_transforms(patch_size)
        else:
            self.transforms = get_test_transforms()

        img_dir  = os.path.join(root, 'ISPRS_semantic_labeling_Vaihingen', 'top')
        dsm_dir  = os.path.join(root, 'ISPRS_semantic_labeling_Vaihingen', 'dsm')
        gt_subdir = ('gts_eroded_for_participants'
                     if use_eroded else 'gts_for_participants')
        gt_dir   = os.path.join(root, 'ISPRS_semantic_labeling_Vaihingen', gt_subdir)

        ids = TRAIN_IDS if split == 'train' else VAL_IDS

        self.samples = []
        for area_id in ids:
            img_pat  = os.path.join(img_dir, f'*{area_id:02d}*.tif')
            imgs     = sorted(glob.glob(img_pat))
            if not imgs:
                img_pat = os.path.join(img_dir, f'*{area_id}*.tif')
                imgs    = sorted(glob.glob(img_pat))
            if not imgs:
                continue

            for img_path in imgs:
                basename = os.path.basename(img_path)
                stem     = os.path.splitext(basename)[0]

                dsm_path = os.path.join(dsm_dir, stem.replace('top_', 'dsm_') + '.tif')
                if not os.path.exists(dsm_path):
                    # try exact same stem
                    dsm_path = os.path.join(dsm_dir, stem + '.tif')
                if not os.path.exists(dsm_path):
                    dsm_candidates = glob.glob(os.path.join(dsm_dir, f'*{area_id:02d}*.tif'))
                    dsm_path = dsm_candidates[0] if dsm_candidates else None

                gt_path = os.path.join(gt_dir, stem.replace('top_', '') + '.tif')
                if not os.path.exists(gt_path):
                    gt_candidates = glob.glob(os.path.join(gt_dir, f'*{area_id:02d}*.tif'))
                    gt_path = gt_candidates[0] if gt_candidates else None

                if dsm_path and gt_path:
                    self.samples.append((img_path, dsm_path, gt_path))

        if not self.samples:
            raise RuntimeError(
                f'[VaihingenDataset] No samples found in {root} for split={split}. '
                'Check directory structure.')

        # For 'test' split build sliding-window tile list
        if split == 'test':
            self.tiles = self._build_test_tiles()

    def _build_test_tiles(self):
        """Pre-compute (sample_idx, y0, x0) sliding-window tiles."""
        tiles = []
        for idx, (img_path, _, _) in enumerate(self.samples):
            img = Image.open(img_path)
            W, H = img.size
            for y0 in range(0, H - self.patch_size + 1, self.stride):
                for x0 in range(0, W - self.patch_size + 1, self.stride):
                    tiles.append((idx, y0, x0))
            img.close()
        return tiles

    def __len__(self):
        if self.split == 'test':
            return len(self.tiles)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == 'test':
            sample_idx, y0, x0 = self.tiles[idx]
            img_path, dsm_path, gt_path = self.samples[sample_idx]
            box = (x0, y0, x0 + self.patch_size, y0 + self.patch_size)
            image = Image.open(img_path).convert('RGB').crop(box)
            dsm   = Image.open(dsm_path)
            if dsm.mode not in ('L', 'F'):
                dsm = dsm.convert('L')
            dsm = dsm.crop(box)
            mask_rgb = np.array(Image.open(gt_path).convert('RGB').crop(box))
            mask = Image.fromarray(_colour_to_label(mask_rgb).astype(np.uint8))
        else:
            img_path, dsm_path, gt_path = self.samples[idx]
            image    = Image.open(img_path).convert('RGB')
            dsm      = Image.open(dsm_path)
            if dsm.mode not in ('L', 'F'):
                dsm = dsm.convert('L')
            mask_rgb = np.array(Image.open(gt_path).convert('RGB'))
            mask     = Image.fromarray(_colour_to_label(mask_rgb).astype(np.uint8))

        image, dsm, mask = self.transforms(image, dsm, mask)
        # mask is now a long tensor
        mask = mask.long()
        return image, dsm, mask
