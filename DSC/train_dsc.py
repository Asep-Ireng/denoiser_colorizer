"""
Training Script for DSC-UNetRes (Reynard's Task)
Architecture: UNetRes with Depthwise Separable Convolutions

Fitur:
- Melatih Full Model (tanpa fase FPN karena struktur backbone berubah)
- Menyimpan checkpoint otomatis pada epoch 5, 10, 15, 20

Usage:
    python train_dsc.py --epochs 20
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

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'modified'))

# Import model DSC Anda
from dsc_unet import UNetResDSC

class DenoisingDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, noise_levels=(5, 75)):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.noise_min, self.noise_max = noise_levels
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('L')
        except:
            return self.__getitem__((idx + 1) % len(self))

        img = np.array(img).astype(np.float32) / 255.0
        
        h, w = img.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img = img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize(
                (self.patch_size, self.patch_size), Image.BILINEAR
            )).astype(np.float32) / 255.0
        
        if random.random() > 0.5: img = np.fliplr(img).copy()
        if random.random() > 0.5: img = np.flipud(img).copy()
        
        noise_level = random.uniform(self.noise_min, self.noise_max)
        sigma = noise_level / 255.0
        
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        noisy = np.clip(img + noise, 0, 1)
        noise_map = np.ones_like(img) * sigma
        
        input_tensor = np.stack([noisy, noise_map], axis=0)
        target_tensor = img[np.newaxis, ...]
        
        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0: return float('inf')
    return 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10 * torch.log10(mse)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)
        
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

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating"):
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)
            loss = F.l1_loss(denoised, clean)
            psnr = compute_psnr(denoised, clean)
            total_loss += loss.item()
            total_psnr += psnr.item()
    return total_loss / len(dataloader), total_psnr / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_filtered/train')
    parser.add_argument('--val_dir', type=str, default='data_filtered/test')
    parser.add_argument('--output_dir', type=str, default='weights/dsc')
    parser.add_argument('--pretrained', type=str, default='weights/denoiser/drunet_gray.pth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20) # Default 20 agar mencakup 5,10,15,20
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Initializing DSC-UNet on {device}...")
    model = UNetResDSC(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    
    # Load pretrained base (jika ada, skip layer yang tidak cocok)
    if os.path.exists(args.pretrained):
        print(f"Loading compatible weights from {args.pretrained}...")
        state_dict = torch.load(args.pretrained, map_location='cpu', weights_only=True)
        model.load_pretrained_base(state_dict)
    
    model.to(device)
    
    train_loader = DataLoader(DenoisingDataset(args.data_dir), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DenoisingDataset(args.val_dir), batch_size=args.batch_size, shuffle=False)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_psnr = 0
    # Epoch target penyimpanan
    save_epochs = [5, 10, 15, 20]
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        loss, psnr = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_psnr = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"  Train: Loss={loss:.4f}, PSNR={psnr:.2f} | Val: Loss={val_loss:.4f}, PSNR={val_psnr:.2f}")
        
        # Save Best Model (selalu simpan jika ada peningkatan)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'dsc_best.pth'))
            print(f"  ✓ Saved Best Model (PSNR: {best_psnr:.2f})")
        
        # Save Specific Epochs (5, 10, 15, 20)
        if epoch in save_epochs:
            filename = f"dsc_epoch_{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(args.output_dir, filename))
            print(f"  💾 Checkpoint saved: {filename}")

if __name__ == '__main__':
    main()