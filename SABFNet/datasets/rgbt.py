"""
RS-RGB-T UAV Dataset Loader  —  SABFNet

Directory structure expected:
  root/
    images/       *.png / *.jpg   — RGB images
    thermal/      *.png / *.jpg   — TIR / thermal images (single channel)
    labels/       *.png           — class index masks (not colour-coded)

OR split files provided directly:
  root/
    train.txt   (one stem per line)
    val.txt
    test.txt

Dataset stats (paper §IV-A):
  Total 4024 RGB+TIR pairs, 7:3 train/val split
  7 semantic classes

Classes (indices 0-6):
  0 = Background
  1 = Person
  2 = Car
  3 = Bicycle
  4 = Other vehicle
  5 = Building
  6 = Vegetation
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


RGBT_CLASSES = [
    'Background',
    'Person',
    'Car',
    'Bicycle',
    'Other vehicle',
    'Building',
    'Vegetation',
]

N_CLASSES = 7


def _find_pair(root, stem, modality_dirs, exts=('.png', '.jpg', '.jpeg', '.tif')):
    """Try to find a file matching stem in a list of candidate directories."""
    for d in modality_dirs:
        for ext in exts:
            path = os.path.join(root, d, stem + ext)
            if os.path.exists(path):
                return path
    return None


class RGBTDataset(Dataset):
    """
    RS-RGB-T dataset for RGB + Thermal segmentation.

    Args:
        root       : dataset root directory
        split      : 'train' | 'val' | 'test'
        patch_size : crop size
        val_ratio  : fraction of data to use as val when no split files exist
        seed       : random seed for auto split
        img_dirs   : list of sub-directories to search for RGB images
        tir_dirs   : list of sub-directories to search for thermal images
        lbl_dirs   : list of sub-directories to search for label masks
        transforms : override default transforms
    """

    def __init__(self, root: str,
                 split: str = 'train',
                 patch_size: int = 256,
                 val_ratio: float = 0.3,
                 seed: int = 42,
                 img_dirs=('images', 'rgb', 'RGB'),
                 tir_dirs=('thermal', 'tir', 'TIR', 'infrared'),
                 lbl_dirs=('labels', 'masks', 'annotations'),
                 transforms=None):
        super().__init__()
        self.root  = root
        self.split = split

        if transforms is not None:
            self.transforms = transforms
        elif split == 'train':
            self.transforms = get_train_transforms(patch_size)
        elif split == 'val':
            self.transforms = get_val_transforms(patch_size)
        else:
            self.transforms = get_test_transforms()

        # 1. Try split list files first
        split_file = os.path.join(root, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                stems = [l.strip() for l in f if l.strip()]
        else:
            # Auto-split based on all stems found in the image directory
            stems = self._collect_stems(root, img_dirs)
            rng = random.Random(seed)
            rng.shuffle(stems)
            n_val = max(1, int(len(stems) * val_ratio))
            if split == 'val':
                stems = stems[:n_val]
            elif split == 'train':
                stems = stems[n_val:]
            else:
                stems = stems  # test = all

        self.samples = []
        for stem in stems:
            img_path = _find_pair(root, stem, img_dirs)
            tir_path = _find_pair(root, stem, tir_dirs)
            lbl_path = _find_pair(root, stem, lbl_dirs)
            if img_path and tir_path and lbl_path:
                self.samples.append((img_path, tir_path, lbl_path))

        if not self.samples:
            raise RuntimeError(
                f'[RGBTDataset] No valid samples found in {root} for split={split}. '
                'Expected sub-dirs: images/, thermal/, labels/ (or train.txt/val.txt).')

    @staticmethod
    def _collect_stems(root, img_dirs):
        stems = []
        for d in img_dirs:
            d_path = os.path.join(root, d)
            if os.path.isdir(d_path):
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                    for p in glob.glob(os.path.join(d_path, ext)):
                        stem = os.path.splitext(os.path.basename(p))[0]
                        if stem not in stems:
                            stems.append(stem)
                if stems:
                    break
        return sorted(stems)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tir_path, lbl_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        tir = Image.open(tir_path)
        if tir.mode not in ('L', 'F'):
            tir = tir.convert('L')

        lbl_arr = np.array(Image.open(lbl_path))
        # If label image is RGB colour-coded, convert (not standard but handle it)
        if lbl_arr.ndim == 3:
            # Assume first channel is index or use custom colour map
            lbl_arr = lbl_arr[:, :, 0]
        mask = Image.fromarray(lbl_arr.astype(np.uint8))

        image, tir, mask = self.transforms(image, tir, mask)
        mask = mask.long()
        return image, tir, mask
