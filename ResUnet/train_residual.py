"""
Training Script for Residual-UNet Denoiser (FIXED VALIDATION BUG)

Changes:
1. Batch Size = 8
2. Epochs = 5
3. Input Channel = 1
4. FIXED: Validation loop device transfer error
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- SETUP PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'modified'))

# Import Model
from residual_unet import ResUNet

# --- CEK GPU ---
if torch.cuda.is_available():
    print(f"✅ GPU DITEMUKAN: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device('cuda')
    PIN_MEMORY = True
else:
    print("⚠️  GPU TIDAK DITEMUKAN. Menggunakan CPU (Akan lambat).")
    DEVICE = torch.device('cpu')
    PIN_MEMORY = False


class DenoisingDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, noise_levels=(5, 50)):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.noise_min, self.noise_max = noise_levels
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        if len(self.image_paths) == 0:
            print(f"⚠️  WARNING: Tidak ada gambar di {image_dir}")
        else:
            print(f"✅ Dataset: {len(self.image_paths)} gambar di {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('L')
        except:
            return torch.zeros(1, self.patch_size, self.patch_size), torch.zeros(1, self.patch_size, self.patch_size)

        img = np.array(img).astype(np.float32) / 255.0
        
        # Random Crop
        h, w = img.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img = img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize(
                (self.patch_size, self.patch_size), Image.BILINEAR
            )).astype(np.float32) / 255.0
        
        # Augmentation
        if random.random() > 0.5: img = np.fliplr(img).copy()
        if random.random() > 0.5: img = np.flipud(img).copy()
        
        # Add Noise
        noise_level = random.uniform(self.noise_min, self.noise_max)
        sigma = noise_level / 255.0
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        noisy = np.clip(img + noise, 0, 1)
        
        # 1 Channel Input
        input_tensor = noisy[np.newaxis, ...] 
        target_tensor = img[np.newaxis, ...]
        
        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0: return 100.0
    return 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10 * torch.log10(mse)


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss, total_psnr = 0, 0
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for noisy, clean in pbar:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        
        optimizer.zero_grad()
        denoised = model(noisy)
        loss = F.l1_loss(denoised, clean)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            psnr = compute_psnr(denoised, clean)
        
        total_loss += loss.item()
        total_psnr += psnr.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'psnr': f'{psnr.item():.2f}'})
    
    return total_loss / len(dataloader), total_psnr / len(dataloader)


def validate(model, dataloader):
    model.eval()
    total_loss, total_psnr = 0, 0
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating"):
            # --- FIX: Pindahkan kedua variabel ke device secara terpisah ---
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            denoised = model(noisy)
            total_loss += F.l1_loss(denoised, clean).item()
            total_psnr += compute_psnr(denoised, clean).item()
            
    return total_loss / len(dataloader), total_psnr / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_filtered/train')
    parser.add_argument('--val_dir', type=str, default='data_filtered/test')
    parser.add_argument('--output_dir', type=str, default='weights/residualblocks')
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nInitialize Model (Input Channel: 1)...")
    model = ResUNet(in_nc=1, out_nc=1).to(DEVICE)
    
    train_ds = DenoisingDataset(args.data_dir)
    val_ds = DenoisingDataset(args.val_dir)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=PIN_MEMORY)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_psnr = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t_loss, t_psnr = train_epoch(model, train_loader, optimizer)
        v_loss, v_psnr = validate(model, val_loader)
        scheduler.step()
        
        print(f"  Result: Train PSNR={t_psnr:.2f} | Val PSNR={v_psnr:.2f}")
        
# --- 1. SIMPAN MODEL TIAP EPOCH (Checkpoint) ---
        # Pastikan variabel 'epoch' tersedia dari loop training kamu
        epoch_save_path = os.path.join(args.output_dir, f'residual_denoiser_epochss{epoch}.pth')
        torch.save(model.state_dict(), epoch_save_path)
        print(f"   💾 Saved Checkpoint: Epoch {epoch}")

        # --- 2. SIMPAN MODEL TERBAIK (Best Model) ---
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            # Simpan dengan nama khusus 'best' agar mudah ditemukan nanti
            best_save_path = os.path.join(args.output_dir, 'residual_denoiser_best50.pth')
            torch.save(model.state_dict(), best_save_path)
            print(f"   ✅ Saved Best Model (PSNR: {best_psnr:.2f})")
            

if __name__ == '__main__':
    main()