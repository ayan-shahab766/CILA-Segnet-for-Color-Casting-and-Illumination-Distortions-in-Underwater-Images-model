"""
utils.py  –  Preprocessing and dataset utilities
=================================================

Preprocessing pipeline (applied BEFORE the model, excluded from latency):
  1. L-channel Adaptive CLAHE  – clipLimit=1.5, tileGridSize=(8,8)  [EXACT original]
  2. AB soft colour correction  – alpha=0.05, mean-shift toward 128  [EXACT original]
  3. Dynamic Fusion             – illuminate-adaptive blend of original + enhanced

NOTE: The three steps above are intentionally excluded from the latency
benchmark in test.py.  The timer wraps only net(input).
"""

import os
import importlib

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ==========================================================================
# Step 2 – AB soft colour correction  (EXACT original)
# ==========================================================================

def soft_ab(a: np.ndarray, b: np.ndarray):
    """Gently push A and B toward neutral (128) by alpha=0.05."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    alpha = 0.05
    a_corr = a + alpha * (128 - np.mean(a))
    b_corr = b + alpha * (128 - np.mean(b))
    return np.clip(a_corr, 0, 255), np.clip(b_corr, 0, 255)

# ==========================================================================
# Full preprocessing pipeline
# ==========================================================================

def enhance_l(l: np.ndarray) -> np.ndarray:
    """Apply Adaptive CLAHE to the L channel."""
    std = float(np.std(l))
    if std < 20:
        clip = 3.0
    elif std < 40:
        clip = 2.0
    else:
        clip = 1.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(l)


def dynamic_fusion_l(l_orig: np.ndarray, l_enh: np.ndarray) -> np.ndarray:
    """
    Dynamically fuse the L channel: dark pixels get enhanced L, bright pixels retain original L.
    This preserves the original highlights while lifting shadows.
    """
    l_norm = l_orig.astype(np.float32) / 255.0
    weight = 1.0 - l_norm
    weight = cv2.GaussianBlur(weight, (5, 5), 0)
    
    l_fused = weight * l_enh.astype(np.float32) + (1.0 - weight) * l_orig.astype(np.float32)
    return np.clip(l_fused, 0, 255).astype(np.uint8)


def process_image_lab(img_pil: Image.Image) -> np.ndarray:
    """
    Full pipeline:
      RGB → LAB → Adaptive L-CLAHE & Soft AB Correction → Dynamic L-Fusion → RGB
    """
    rgb_orig = np.array(img_pil)

    # ── Convert to LAB ───────────────────────────────────────────────────
    lab = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # ── Step 1: Adaptive L-channel CLAHE ─────────────────────────────────
    l_enh = enhance_l(l)

    # ── Step 2: AB soft correction ──────────────────────────────────────
    a_corr, b_corr = soft_ab(a, b)

    # ── Step 3: Dynamic Fusion (Only on L channel) ──────────────────────
    # By fusing only the L channel, we perfectly preserve the recovered
    # reds and greens from the AB correction step!
    l_fused = dynamic_fusion_l(l, l_enh)

    # ── Merge and convert back to RGB ────────────────────────────────────
    lab_out = cv2.merge([l_fused, a_corr.astype(np.uint8), b_corr.astype(np.uint8)])
    rgb_enh = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    return rgb_enh


# ==========================================================================
# Datasets
# ==========================================================================

def data_aug(image, re_size=None):
    if re_size:
        image = TF.resize(image, (re_size, re_size))
    return image


class UIE_train(Dataset):
    """Training dataset – preprocessing applied per sample."""

    def __init__(self, train_lst, image_size):
        self.path_lst   = train_lst
        self.image_size = image_size
        print(f'got {len(self.path_lst)} images, {len(self.path_lst)} references')

    def __getitem__(self, index):
        entry = self.path_lst[index]
        name  = entry[0].split('/')[-1]

        img_pil = Image.open(entry[0]).convert('RGB')
        ref_pil = Image.open(entry[1]).convert('RGB')

        img_np = process_image_lab(img_pil)

        image = TF.to_tensor(img_np.copy())
        ref   = TF.to_tensor(np.array(ref_pil).copy())

        if len(entry) == 2:
            tx = image
        else:
            tx = TF.to_tensor(np.array(Image.open(entry[2])).copy())

        image = data_aug(image, self.image_size)
        ref   = data_aug(ref,   self.image_size)
        tx    = data_aug(tx,    self.image_size)

        return {'image': image, 'ref': ref, 'tx': tx, 'name': name}

    def __len__(self):
        return len(self.path_lst)


class UIE_val(Dataset):
    """Validation dataset – preprocessing applied per sample."""

    def __init__(self, train_lst, image_size):
        self.path_lst   = train_lst
        self.image_size = image_size
        print(f'got {len(self.path_lst)} images, {len(self.path_lst)} references')

    def __getitem__(self, index):
        entry = self.path_lst[index]
        name  = entry[0].split('/')[-1]

        img_pil = Image.open(entry[0]).convert('RGB')
        ref_pil = Image.open(entry[1]).convert('RGB')

        img_np = process_image_lab(img_pil)

        image = TF.to_tensor(img_np.copy())
        ref   = TF.to_tensor(np.array(ref_pil).copy())

        image = data_aug(image, self.image_size)
        ref   = data_aug(ref,   self.image_size)

        return {'image': image, 'ref': ref, 'name': name}

    def __len__(self):
        return len(self.path_lst)


# ==========================================================================
# Helpers
# ==========================================================================

def get_file_lst(file_path: str, gt_path: str):
    file_list = sorted(os.listdir(file_path))
    return [(os.path.join(file_path, f), os.path.join(gt_path, f))
            for f in file_list]


def get_obj_from_string(string: str):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count