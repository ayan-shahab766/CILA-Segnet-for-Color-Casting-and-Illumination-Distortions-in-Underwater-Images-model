"""
predict.py  –  Single-image or folder inference for CCMSRNet (optimized)
=========================================================================
Usage:
    python predict.py \
        --exp    CCMSRNet.yml \
        --ckpt   /path/to/best_val.pth \
        --input  /path/to/raw/images/ \
        --output /path/to/save/results \
        --resize 0

    --resize 0  → keep original resolution
    --resize N  → resize to N×N before inference, then back

Full preprocessing pipeline (L-CLAHE + AB correction + dynamic fusion)
is applied to every input image before the model sees it.
Preprocessing time is NOT counted in timing prints.
"""

import os
import os.path as osp
import random
import time
import argparse

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from utils import get_obj_from_string, process_image_lab


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--exp',    type=str, required=True, help='path to YAML config')
parser.add_argument('--ckpt',   type=str, required=True, help='path to best_val.pth')
parser.add_argument('--input',  type=str, required=True, help='path to input image(s)')
parser.add_argument('--output', type=str, default='./predict_outputs',
                    help='directory to save enhanced images')
parser.add_argument('--resize', type=int, default=0,
                    help='resize input to N×N before inference (0 = no resize)')
parser.add_argument('--cuda_id', type=int, default=0)
args = parser.parse_args()

with open(args.exp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available()
                      else 'cpu')
print(f'Device: {device}')

# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────
out_cc    = osp.join(args.output, 'cc_imgs')     # colour-corrected intermediate
out_final = osp.join(args.output, 'final_imgs')  # final enhanced output
os.makedirs(out_cc,    exist_ok=True)
os.makedirs(out_final, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (no ground truth – inference only)
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class InferenceDataset(Dataset):
    """Loads raw images and applies full preprocessing pipeline."""

    def __init__(self, input_path: str):
        if osp.isfile(input_path):
            self.files = [input_path]
        else:
            self.files = sorted([
                osp.join(input_path, f)
                for f in os.listdir(input_path)
                if osp.splitext(f)[1].lower() in SUPPORTED
            ])
        print(f'Found {len(self.files)} image(s) in: {input_path}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path   = self.files[idx]
        name   = osp.basename(path)
        img_pil = Image.open(path).convert('RGB')

        # Store original size for optional resize-back
        orig_h, orig_w = img_pil.height, img_pil.width

        # Apply full preprocessing (CLAHE + AB + dynamic fusion)
        img_np = process_image_lab(img_pil)
        tensor = TF.to_tensor(img_np.copy())   # (3, H, W)  float32 [0,1]

        return {'image': tensor, 'name': name,
                'orig_h': orig_h, 'orig_w': orig_w}


dataset    = InferenceDataset(args.input)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
net = get_obj_from_string(config['model_file'])(
    3, 3, img_size=config['img_size']).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
net.load_state_dict(ckpt['net'])
net.eval()
print('Model loaded successfully')


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────
print('Starting inference …')
total_ms = 0.0

with torch.no_grad():
    for data in tqdm(dataloader):
        image  = data['image'].to(device, non_blocking=True)
        name   = data['name'][0]
        orig_h = data['orig_h'].item()
        orig_w = data['orig_w'].item()

        # Optional: resize to fixed square before inference
        if args.resize > 0:
            image = F.interpolate(image, size=(args.resize, args.resize),
                                  mode='bilinear', align_corners=True)

        # ── Timed model forward pass (preprocessing excluded) ──────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cc_pred, pred = net(image)
        torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        # ── Restore original resolution if resized ─────────────────────
        if args.resize > 0:
            pred    = F.interpolate(pred,    size=(orig_h, orig_w),
                                    mode='bilinear', align_corners=True)
            cc_pred = F.interpolate(cc_pred, size=(orig_h, orig_w),
                                    mode='bilinear', align_corners=True)

        # ── Save outputs ───────────────────────────────────────────────
        stem = osp.splitext(name)[0]
        torchvision.utils.save_image(cc_pred, f'{out_cc}/{stem}.png')
        torchvision.utils.save_image(pred,    f'{out_final}/{stem}.png')

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
n = len(dataset)
print(f'\nDone. {n} image(s) processed.')
print(f'Average inference time : {total_ms/n:.2f} ms/image  '
      f'(preprocessing excluded)')
print(f'Results saved to       : {args.output}')
print(f'  Colour-corrected     : {out_cc}')
print(f'  Final enhanced       : {out_final}')