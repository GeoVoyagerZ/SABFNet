"""
Shared data augmentation transforms for SABFNet.

All transforms operate on (image, dsm, mask) triples where:
  image: PIL.Image or np.ndarray (H, W, 3) — optical / RGB
  dsm  : PIL.Image or np.ndarray (H, W)    — DSM or TIR, single channel
  mask : PIL.Image or np.ndarray (H, W)    — long class indices
"""

import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


# ── Base class ────────────────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, dsm, mask):
        for t in self.transforms:
            image, dsm, mask = t(image, dsm, mask)
        return image, dsm, mask


# ── Geometric (applied identically to all three inputs) ──────────────────────

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, dsm, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            dsm   = TF.hflip(dsm)
            mask  = TF.hflip(mask)
        return image, dsm, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, dsm, mask):
        if random.random() < self.p:
            image = TF.vflip(image)
            dsm   = TF.vflip(dsm)
            mask  = TF.vflip(mask)
        return image, dsm, mask


class RandomRotation90:
    """Randomly rotate by 0, 90, 180, or 270 degrees."""
    def __call__(self, image, dsm, mask):
        k = random.randint(0, 3)
        if k > 0:
            angle = 90 * k
            image = TF.rotate(image, angle)
            dsm   = TF.rotate(dsm,   angle)
            mask  = TF.rotate(mask,  angle)
        return image, dsm, mask


class RandomCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(self, image, dsm, mask):
        i, j, h, w = TF.RandomCrop.get_params(image, output_size=self.size)
        image = TF.crop(image, i, j, h, w)
        dsm   = TF.crop(dsm,   i, j, h, w)
        mask  = TF.crop(mask,  i, j, h, w)
        return image, dsm, mask


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, dsm, mask):
        image = TF.center_crop(image, self.size)
        dsm   = TF.center_crop(dsm,   self.size)
        mask  = TF.center_crop(mask,  self.size)
        return image, dsm, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, dsm, mask):
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        dsm   = TF.resize(dsm,   self.size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.size, interpolation=Image.NEAREST)
        return image, dsm, mask


# ── Colour jitter (optical only) ─────────────────────────────────────────────

class ColorJitter:
    """Apply colour jitter to the optical image only."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        import torchvision.transforms as T
        self.jitter = T.ColorJitter(brightness=brightness,
                                    contrast=contrast,
                                    saturation=saturation,
                                    hue=hue)

    def __call__(self, image, dsm, mask):
        image = self.jitter(image)
        return image, dsm, mask


# ── Normalisation & tensor conversion ─────────────────────────────────────────

class ToTensor:
    """
    Convert PIL images to torch tensors.
      image → [3, H, W]  float32, [0, 1]
      dsm   → [1, H, W]  float32, [0, 1]  (if uint8) or raw float
      mask  → [H, W]     int64
    """
    def __call__(self, image, dsm, mask):
        image = TF.to_tensor(image)                          # [3, H, W] float

        dsm_arr = np.array(dsm, dtype=np.float32)
        if dsm_arr.ndim == 2:
            dsm_arr = dsm_arr[np.newaxis]                    # [1, H, W]
        dsm_t = torch.from_numpy(dsm_arr)
        # Normalise uint8 DSM to [0, 1]
        if dsm_arr.max() > 1.0:
            dsm_t = dsm_t / 255.0

        mask_arr = np.array(mask, dtype=np.int64)
        mask_t = torch.from_numpy(mask_arr)                  # [H, W]

        return image, dsm_t, mask_t


class Normalize:
    """
    Normalise optical image with ImageNet stats.
    DSM is normalised to zero mean / unit std using provided stats.
    """
    def __init__(self,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std =(0.229, 0.224, 0.225),
                 dsm_mean=(0.5,),
                 dsm_std =(0.5,)):
        self.img_mean = img_mean
        self.img_std  = img_std
        self.dsm_mean = dsm_mean
        self.dsm_std  = dsm_std

    def __call__(self, image, dsm, mask):
        image = TF.normalize(image, self.img_mean, self.img_std)
        dsm   = TF.normalize(dsm,   self.dsm_mean, self.dsm_std)
        return image, dsm, mask


# ── Convenience factories ─────────────────────────────────────────────────────

def get_train_transforms(patch_size: int = 256):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        RandomCrop(patch_size),
        ColorJitter(),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms(patch_size: int = 256):
    return Compose([
        CenterCrop(patch_size),
        ToTensor(),
        Normalize(),
    ])


def get_test_transforms():
    """No cropping for test — evaluate full tile via sliding window."""
    return Compose([
        ToTensor(),
        Normalize(),
    ])
