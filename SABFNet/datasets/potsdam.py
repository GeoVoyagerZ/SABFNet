"""
ISPRS Potsdam Dataset Loader  —  SABFNet

Directory structure expected:
  root/
    Potsdam/
      2_Ortho_RGB/          *_RGB.tif
      1_DSM_normalisation/  *_normalized_lasground.tif  (or similar)
      5_Labels_all/         *_label.tif   (colour-coded)
      5_Labels_all_noBoundary/  (optional eroded GT)

Train / val split (tile IDs, commonly used):
  train: 2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12,
         5_10, 5_11, 5_12, 6_10, 6_11, 6_12, 6_7,  6_8,  6_9,
         7_7,  7_8,  7_9,  7_10, 7_11, 7_12
  val  : 2_13, 2_14, 3_13, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15,
         6_13, 6_14, 6_15, 7_13

Classes (6):
  0 = Impervious surfaces  (255,255,255)
  1 = Building             (0,0,255)
  2 = Low vegetation       (0,255,255)
  3 = Tree                 (0,255,0)
  4 = Car                  (255,255,0)
  5 = Clutter/background   (255,0,0)
"""

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


POTSDAM_COLOUR_MAP = {
    (255, 255, 255): 0,
    (  0,   0, 255): 1,
    (  0, 255, 255): 2,
    (  0, 255,   0): 3,
    (255, 255,   0): 4,
    (255,   0,   0): 5,
}

POTSDAM_CLASSES = [
    'Impervious surfaces',
    'Building',
    'Low vegetation',
    'Tree',
    'Car',
    'Clutter',
]

TRAIN_IDS = [
    '2_10','2_11','2_12',
    '3_10','3_11','3_12',
    '4_10','4_11','4_12',
    '5_10','5_11','5_12',
    '6_7', '6_8', '6_9',
    '6_10','6_11','6_12',
    '7_7', '7_8', '7_9',
    '7_10','7_11','7_12',
]
VAL_IDS = [
    '2_13','2_14',
    '3_13',
    '4_13','4_14','4_15',
    '5_13','5_14','5_15',
    '6_13','6_14','6_15',
    '7_13',
]


def _colour_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    h, w = mask_rgb.shape[:2]
    label = np.full((h, w), 255, dtype=np.int64)
    for rgb, cls in POTSDAM_COLOUR_MAP.items():
        r, g, b = rgb
        match = (mask_rgb[:,:,0]==r)&(mask_rgb[:,:,1]==g)&(mask_rgb[:,:,2]==b)
        label[match] = cls
    return label


class PotsdamDataset(Dataset):
    """
    ISPRS Potsdam dataset.

    Args:
        root       : path to dataset root
        split      : 'train' | 'val' | 'test'
        patch_size : random/centre crop size
        stride     : sliding-window stride for test
        use_eroded : use boundary-eroded GT
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

        potsdam_dir = os.path.join(root, 'Potsdam')
        rgb_dir     = os.path.join(potsdam_dir, '2_Ortho_RGB')
        dsm_dir     = os.path.join(potsdam_dir, '1_DSM_normalisation')
        gt_subdir   = ('5_Labels_all_noBoundary' if use_eroded
                       else '5_Labels_all')
        gt_dir      = os.path.join(potsdam_dir, gt_subdir)

        ids = TRAIN_IDS if split in ('train', 'test') else VAL_IDS

        self.samples = []
        for tile_id in ids:
            rgb_pat = os.path.join(rgb_dir, f'*{tile_id}*_RGB.tif')
            rgbs    = sorted(glob.glob(rgb_pat))
            if not rgbs:
                rgb_pat = os.path.join(rgb_dir, f'*{tile_id}*.tif')
                rgbs    = sorted(glob.glob(rgb_pat))
            if not rgbs:
                continue

            for rgb_path in rgbs:
                # Match DSM file
                dsm_pats = [
                    os.path.join(dsm_dir, f'*{tile_id}*normalized*.tif'),
                    os.path.join(dsm_dir, f'*{tile_id}*.tif'),
                ]
                dsm_path = None
                for pat in dsm_pats:
                    cands = glob.glob(pat)
                    if cands:
                        dsm_path = sorted(cands)[0]
                        break

                # Match GT file
                gt_pats = [
                    os.path.join(gt_dir, f'*{tile_id}*label*.tif'),
                    os.path.join(gt_dir, f'*{tile_id}*.tif'),
                ]
                gt_path = None
                for pat in gt_pats:
                    cands = glob.glob(pat)
                    if cands:
                        gt_path = sorted(cands)[0]
                        break

                if dsm_path and gt_path:
                    self.samples.append((rgb_path, dsm_path, gt_path))

        if not self.samples:
            raise RuntimeError(
                f'[PotsdamDataset] No samples found in {root} for split={split}.')

        if split == 'test':
            self.tiles = self._build_test_tiles()

    def _build_test_tiles(self):
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
        mask = mask.long()
        return image, dsm, mask
