"""
train.py  –  Training script for CCMSRNet (optimized) on Kaggle T4 GPU
=======================================================================
Usage (Kaggle notebook):
    %cd /kaggle/working/CCMSR-Net-main-50-imp
    !python train.py --exp CCMSRNet.yml
"""

import os
import os.path as osp
import random
import time
import argparse
import importlib

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter, get_obj_from_string, UIE_train, UIE_val, get_file_lst


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id',    type=int,  default=0)
parser.add_argument('--exp',        type=str,  required=True, help='path to YAML config')
parser.add_argument('--debug',      action='store_true')
parser.add_argument('--resize_val', action='store_true')
args = parser.parse_args()

with open(args.exp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 1234
random.seed(SEED);  np.random.seed(SEED)
torch.manual_seed(SEED);  torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True   # speed up fixed-size ops

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
assert torch.cuda.is_available(), 'CUDA is not available'
device = torch.device('cuda:0')

# ---------------------------------------------------------------------------
# Paths  (dataset is read directly from Kaggle input – read-only mount)
# Set env var CCMSR_DATA_ROOT before running to override the default path.
# ---------------------------------------------------------------------------
config['model_name'] = config['model_file'].split('.')[-1]

_DEFAULT_ROOT = '/kaggle/input/datasets/ahmedfrahim/my-ccmsr-dataset/UIEB-dataset-processed'
DATASET_ROOT  = os.environ.get('CCMSR_DATA_ROOT', _DEFAULT_ROOT)

if 'UIEB' in config['name']:
    tr_input_path  = f'{DATASET_ROOT}/train/raw/'
    tr_gt_path     = f'{DATASET_ROOT}/train/reference/'
    val_input_path = f'{DATASET_ROOT}/val/raw/'
    val_gt_path    = f'{DATASET_ROOT}/val/reference/'
    ts_input_path  = f'{DATASET_ROOT}/test/raw/'
    ts_gt_path     = f'{DATASET_ROOT}/test/reference/'

# ---------------------------------------------------------------------------
# Log / snapshot paths
# ---------------------------------------------------------------------------
tag = (f"{config['name']}_{config['model_name']}"
       f"_lr{config['lr']}_wd{config['weight_decay']}"
       f"_epoch{config['num_epochs']}_ccweight{config['cc_weight']}")

if args.debug:
    log_path      = f'./debug/{tag}_log/'
else:
    log_path      = f'./exp_results/{tag}_log/'
snapshot_path = log_path + 'weights/'

os.makedirs(snapshot_path, exist_ok=True)
os.makedirs(log_path,      exist_ok=True)

log_file = open(log_path + 'save_log.txt', 'w')
log_file.write(f'config: {args.exp}\n')
for k, v in config.items():
    log_file.write(f'{k}: {v}\n')

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
print(f'[train] input: {tr_input_path}\n[train] gt   : {tr_gt_path}')
train_lst    = get_file_lst(tr_input_path, tr_gt_path)
train_set    = UIE_train(train_lst, image_size=config['img_size'])
train_loader = DataLoader(train_set,
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=4,          # 4 workers on T4
                          pin_memory=True,
                          persistent_workers=True)

print(f'[val]   input: {val_input_path}\n[val]   gt   : {val_gt_path}')
val_lst = get_file_lst(val_input_path, val_gt_path)
if args.resize_val:
    val_set = UIE_val(val_lst, image_size=config['img_size'])
else:
    val_set = UIE_val(val_lst, image_size=None)
val_loader = DataLoader(val_set,
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        persistent_workers=True)

# ---------------------------------------------------------------------------
# Model, optimiser, scheduler
# ---------------------------------------------------------------------------
net = get_obj_from_string(config['model_file'])(3, 3, img_size=config['img_size']).to(device)

# Use AdamW + cosine annealing warm restarts for better convergence
optimizer = torch.optim.AdamW(net.parameters(),
                              lr=config['lr'],
                              weight_decay=config['weight_decay'])

num_epochs = config['num_epochs']
if config.get('lr_decay', 'None') == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6)
else:
    scheduler = None

criterion = nn.MSELoss()
cc_weight  = config['cc_weight']

# ---------------------------------------------------------------------------
# Optional resume
# ---------------------------------------------------------------------------
ckpt_path   = osp.join(snapshot_path, 'latest.pth')
start_epoch = 0

if osp.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
    if scheduler and 'scheduler' in ckpt and ckpt['scheduler'] is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    print(f'Resumed from epoch {start_epoch}')

# ---------------------------------------------------------------------------
# Tensorboard
# ---------------------------------------------------------------------------
writer = SummaryWriter(f'{log_path[:-1]}_tensorboard')

# ---------------------------------------------------------------------------
# AMP scaler for mixed-precision training (free speed-up on T4)
# ---------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
mse_loss_lst    = []
cc_mse_loss_lst = []
val_mse_loss_lst = []


def train(net, num_epochs):
    best_val_loss    = float('inf')
    patience         = 8           # slightly more generous
    min_delta        = 1e-4
    start_from_epoch = 40
    counter          = 0

    for epoch in range(start_epoch, num_epochs):
        net.train()
        tr_loss_meter = AverageMeter()
        cc_loss_meter = AverageMeter()
        tic = time.time()

        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()
            image = data['image'].to(device, non_blocking=True)
            label = data['ref'].to(device,   non_blocking=True)

            # Mixed-precision forward
            with torch.cuda.amp.autocast():
                cc, pred     = net(image)
                loss_mse     = criterion(pred, label)
                cc_loss_mse  = criterion(cc,   label)
                loss         = loss_mse + cc_weight * cc_loss_mse

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss_meter.update(loss_mse.item(),    len(label))
            cc_loss_meter.update(cc_loss_mse.item(), len(label))

        if scheduler is not None:
            scheduler.step()

        toc = time.time()
        print(f'epoch:{epoch+1}/{num_epochs}, '
              f'mse={tr_loss_meter.avg:.5f}, '
              f'cc_mse={cc_loss_meter.avg:.5f}, '
              f'time={toc-tic:.1f}s, '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        mse_loss_lst.append(tr_loss_meter.avg)
        cc_mse_loss_lst.append(cc_loss_meter.avg)
        np.save(log_path + 'tr_mse_loss.npy',    np.array(mse_loss_lst))
        np.save(log_path + 'tr_cc_mse_loss.npy', np.array(cc_mse_loss_lst))

        if (epoch + 1) % 5 == 0:
            writer.add_scalar('Loss/train',    tr_loss_meter.avg, epoch)
            writer.add_scalar('Loss/cc_train', cc_loss_meter.avg, epoch)

        # ── Validation ────────────────────────────────────────────────
        net.eval()
        val_loss_meter = AverageMeter()
        tic = time.time()
        with torch.no_grad():
            for data in tqdm(val_loader, desc='  Val'):
                image = data['image'].to(device, non_blocking=True)
                label = data['ref'].to(device,   non_blocking=True)
                with torch.cuda.amp.autocast():
                    cc, pred = net(image)
                loss_mse = criterion(pred, label)
                val_loss_meter.update(loss_mse.item())

        toc = time.time()
        print(f'  val_loss={val_loss_meter.avg:.5f}, time={toc-tic:.1f}s')

        if (epoch + 1) % 5 == 0:
            writer.add_scalar('Loss/val', val_loss_meter.avg, epoch)

        val_mse_loss_lst.append(val_loss_meter.avg)
        np.save(log_path + 'val_mse_loss.npy', np.array(val_mse_loss_lst))

        # ── Best model ────────────────────────────────────────────────
        if val_loss_meter.avg < best_val_loss - min_delta:
            best_val_loss = val_loss_meter.avg
            counter = 0
            state = {
                'net':       net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch':     epoch,
            }
            torch.save(state, snapshot_path + 'best_val.pth')
            print(f'  ✓ Validation improved → saved (epoch {epoch+1})')
            log_file.write(f'Improved at epoch {epoch+1}, val_loss={best_val_loss:.6f}\n')

        elif epoch >= start_from_epoch:
            counter += 1
            print(f'  No improvement: {counter}/{patience}')
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                log_file.write(f'Early stopping at epoch {epoch+1}\n')
                best_ckpt = torch.load(snapshot_path + 'best_val.pth')
                net.load_state_dict(best_ckpt['net'])
                break

        # ── Latest checkpoint ─────────────────────────────────────────
        torch.save({
            'net':       net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch':     epoch,
        }, ckpt_path)

    log_file.close()
    writer.close()


train(net, num_epochs)
