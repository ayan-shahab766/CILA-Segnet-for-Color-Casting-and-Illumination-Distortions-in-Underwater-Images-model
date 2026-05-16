"""
test.py  –  Evaluation script for CCMSRNet (optimized)
=======================================================
Usage:
    python test.py \
        --ckpt  /path/to/best_val.pth \
        --input_dir /path/to/test/raw/ \
        --gt_dir    /path/to/test/reference/ \
        --save_dir  ./test_outputs \
        --log_path  /path/to/exp_results/...log/ \
        --img_size  256 \
        --project_path /kaggle/working/CCMSR-Net-main-50-imp

IMPORTANT LATENCY NOTE
----------------------
The latency measurement wraps ONLY the model forward pass.
Preprocessing (L-CLAHE + AB correction + dynamic fusion) is intentionally
excluded from the latency timer, as instructed.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')           # headless (no display needed on Kaggle)
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity  as compare_ssim
from thop import profile

# ── Parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',         type=str, required=True)
parser.add_argument('--input_dir',    type=str, required=True)
parser.add_argument('--gt_dir',       type=str, required=True)
parser.add_argument('--save_dir',     type=str, default='./test_outputs')
parser.add_argument('--log_path',     type=str, required=True)
parser.add_argument('--img_size',     type=int, default=256)
parser.add_argument('--project_path', type=str, required=True)
args = parser.parse_args()

sys.path.insert(0, args.project_path)
from utils import get_obj_from_string, process_image_lab  # preprocessing helpers

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

os.makedirs(args.save_dir, exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────────────
net = get_obj_from_string('models.CCMSRNet.UNet')(
    3, 3, img_size=args.img_size).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
net.load_state_dict(ckpt['net'])
net.eval()
print('Model loaded successfully')


# ============================================================================
# Latency / cost  (preprocessing is EXCLUDED from timing)
# ============================================================================
def compute_model_cost(net, device, img_size=256):
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    # ── FLOPs & params ───────────────────────────────────────────────────
    flops, params = profile(net, inputs=(dummy,), verbose=False)

    # ── Warm-up (10 passes, not timed) ───────────────────────────────────
    with torch.no_grad():
        for _ in range(10):
            net(dummy)

    torch.cuda.synchronize()

    # ── Timed runs (100 passes) — model only, no preprocessing ───────────
    N = 100
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            net(dummy)
            torch.cuda.synchronize()
    end = time.perf_counter()

    latency = (end - start) / N

    print('\n===== MODEL COST (preprocessing excluded) =====')
    print(f'  Params  : {params/1e6:.3f} M')
    print(f'  GFLOPs  : {flops/1e9:.3f} G')
    print(f'  Latency : {latency*1000:.3f} ms  ({latency:.6f} s)')
    print(f'  FPS     : {1/latency:.1f}')
    return flops, params, latency


flops, params, latency = compute_model_cost(net, device, args.img_size)

with open(os.path.join(args.save_dir, 'model_cost.txt'), 'w') as f:
    f.write(f'Params   : {params/1e6:.3f} M\n')
    f.write(f'GFLOPs   : {flops/1e9:.3f} G\n')
    f.write(f'Latency  : {latency*1000:.3f} ms\n')
    f.write(f'FPS      : {1/latency:.1f}\n')
    f.write('NOTE: Latency excludes preprocessing (L-CLAHE + AB + dynamic fusion)\n')


# ============================================================================
# Test loop
# ============================================================================
test_files = sorted(os.listdir(args.input_dir))
print(f'\nTotal test images: {len(test_files)}')

psnr_list, ssim_list = [], []

for fname in tqdm(test_files, desc='Testing'):
    img_path = os.path.join(args.input_dir, fname)
    gt_path  = os.path.join(args.gt_dir,    fname)
    if not os.path.exists(gt_path):
        continue

    # ── Load & preprocess (preprocessing time NOT measured) ──────────────
    img_pil = Image.open(img_path).convert('RGB')
    gt_pil  = Image.open(gt_path).convert('RGB')
    gt_np   = np.array(gt_pil)

    img_np  = process_image_lab(img_pil)          # full pipeline: L-CLAHE+AB+fusion

    # ── To tensor ────────────────────────────────────────────────────────
    inp = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    inp = inp.unsqueeze(0).to(device)

    # ── Model inference (only this is counted toward latency) ────────────
    with torch.no_grad():
        cc, pred = net(inp)

    # ── Back to numpy ─────────────────────────────────────────────────────
    pred_np   = pred.squeeze().permute(1, 2, 0).cpu().numpy()
    pred_np   = np.clip(pred_np, 0, 1)
    pred_uint8 = (pred_np * 255).astype(np.uint8)

    if pred_uint8.shape != gt_np.shape:
        gt_np = cv2.resize(gt_np, (pred_uint8.shape[1], pred_uint8.shape[0]))

    psnr = compare_psnr(gt_np, pred_uint8, data_range=255)
    ssim = compare_ssim(gt_np, pred_uint8, channel_axis=2, data_range=255)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    save_path = os.path.join(args.save_dir, fname)
    cv2.imwrite(save_path, cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR))


# ============================================================================
# Metrics summary
# ============================================================================
psnr_arr = np.array(psnr_list)
ssim_arr = np.array(ssim_list)

print('\n===== FINAL RESULTS =====')
print(f'Images : {len(psnr_arr)}')
print(f'\n--- PSNR ---')
print(f'  Mean : {psnr_arr.mean():.4f}')
print(f'  Std  : {psnr_arr.std():.4f}')
print(f'  Min  : {psnr_arr.min():.4f}')
print(f'  Max  : {psnr_arr.max():.4f}')
print(f'\n--- SSIM ---')
print(f'  Mean : {ssim_arr.mean():.4f}')
print(f'  Std  : {ssim_arr.std():.4f}')
print(f'  Min  : {ssim_arr.min():.4f}')
print(f'  Max  : {ssim_arr.max():.4f}')

with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
    f.write(f'PSNR Mean : {psnr_arr.mean():.4f}\n')
    f.write(f'PSNR Std  : {psnr_arr.std():.4f}\n')
    f.write(f'SSIM Mean : {ssim_arr.mean():.4f}\n')
    f.write(f'SSIM Std  : {ssim_arr.std():.4f}\n')


# ============================================================================
# Loss curves
# ============================================================================
def plot_losses(log_path, save_dir):
    train_loss = np.load(os.path.join(log_path, 'tr_mse_loss.npy'))
    val_loss   = np.load(os.path.join(log_path, 'val_mse_loss.npy'))

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,   label='Validation Loss')
    plt.xlabel('Epoch');  plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend();  plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)
    plt.close()

    print('\n===== LOSS SUMMARY =====')
    print(f'  Final Train Loss : {train_loss[-1]:.5f}')
    print(f'  Final Val Loss   : {val_loss[-1]:.5f}')
    print(f'  Best Val Loss    : {np.min(val_loss):.5f}')


plot_losses(args.log_path, args.save_dir)
print(f'\nAll results saved to: {args.save_dir}')