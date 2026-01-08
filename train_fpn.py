"""
Training Script for FPN-Enhanced Denoiser

Fine-tuning strategy:
- Phase 1: Train FPN layers only (freeze encoder/decoder)
- Phase 2: Unfreeze all, train with lower learning rate

Usage:
    python train_fpn.py --data_dir data_filtered/train --epochs 30
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
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'modified'))

from fpn_unet import UNetResFPN


class DenoisingDataset(Dataset):
    """
    Dataset for denoising training.
    Loads grayscale images and adds Gaussian noise on-the-fly.
    """
    
    def __init__(self, image_dir, patch_size=128, noise_levels=(5, 75)):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.noise_min, self.noise_max = noise_levels
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = np.array(img).astype(np.float32) / 255.0
        
        # Random crop if image is larger than patch size
        h, w = img.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img = img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if too small
            img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize(
                (self.patch_size, self.patch_size), Image.BILINEAR
            )).astype(np.float32) / 255.0
        
        # Random augmentation
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
        
        # Generate random noise level
        noise_level = random.uniform(self.noise_min, self.noise_max)
        sigma = noise_level / 255.0
        
        # Add Gaussian noise
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        noisy = np.clip(img + noise, 0, 1)
        
        # Create noise map (constant value matching DRUNet input format)
        noise_map = np.ones_like(img) * sigma
        
        # Stack noisy + noise_map as input (2 channels)
        input_tensor = np.stack([noisy, noise_map], axis=0)
        target_tensor = img[np.newaxis, ...]  # 1 channel
        
        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)


def compute_psnr(pred, target):
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10 * torch.log10(mse)


def train_epoch(model, dataloader, optimizer, device, freeze_backbone=False):
    """Train for one epoch."""
    model.train()
    
    if freeze_backbone:
        # Freeze everything except FPN
        for name, param in model.named_parameters():
            param.requires_grad = 'fpn' in name
    else:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
    
    total_loss = 0
    total_psnr = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (noisy, clean) in enumerate(pbar):
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        denoised = model(noisy)
        
        # L1 Loss
        loss = F.l1_loss(denoised, clean)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(denoised, clean)
        
        total_loss += loss.item()
        total_psnr += psnr.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr.item():.2f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            denoised = model(noisy)
            loss = F.l1_loss(denoised, clean)
            psnr = compute_psnr(denoised, clean)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr


def main():
    parser = argparse.ArgumentParser(description='Train FPN-enhanced denoiser')
    parser.add_argument('--data_dir', type=str, default='data_filtered/train',
                        help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default='data_filtered/test',
                        help='Directory with validation images')
    parser.add_argument('--pretrained', type=str, default='weights/denoiser/drunet_gray.pth',
                        help='Path to pretrained DRUNet weights')
    parser.add_argument('--output_dir', type=str, default='weights/fpn',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--epochs_phase1', type=int, default=10,
                        help='Epochs for Phase 1 (FPN only)')
    parser.add_argument('--epochs_phase2', type=int, default=20,
                        help='Epochs for Phase 2 (all layers)')
    parser.add_argument('--lr_phase1', type=float, default=1e-4)
    parser.add_argument('--lr_phase2', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=0)  # 0 for Windows compatibility
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("\n" + "="*50)
    print("Loading model...")
    print("="*50)
    
    model = UNetResFPN(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    
    if os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location='cpu', weights_only=True)
        model.load_pretrained(state_dict)
    else:
        print("⚠ No pretrained weights found, training from scratch")
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    fpn_params = sum(p.numel() for n, p in model.named_parameters() if 'fpn' in n)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"FPN parameters: {fpn_params:,}")
    
    # Create datasets
    print("\n" + "="*50)
    print("Loading datasets...")
    print("="*50)
    
    train_dataset = DenoisingDataset(args.data_dir, patch_size=args.patch_size)
    val_dataset = DenoisingDataset(args.val_dir, patch_size=args.patch_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ============ PHASE 1: Train FPN only ============
    print("\n" + "="*50)
    print("PHASE 1: Training FPN layers only")
    print("="*50)
    
    # Only optimize FPN parameters
    fpn_params_list = [p for n, p in model.named_parameters() if 'fpn' in n]
    optimizer = Adam(fpn_params_list, lr=args.lr_phase1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_phase1)
    
    best_psnr = 0
    
    # History for plotting
    history = {
        'phase1_train_loss': [], 'phase1_train_psnr': [],
        'phase1_val_loss': [], 'phase1_val_psnr': [],
        'phase2_train_loss': [], 'phase2_train_psnr': [],
        'phase2_val_loss': [], 'phase2_val_psnr': []
    }
    
    for epoch in range(1, args.epochs_phase1 + 1):
        print(f"\nEpoch {epoch}/{args.epochs_phase1}")
        
        train_loss, train_psnr = train_epoch(model, train_loader, optimizer, device, freeze_backbone=True)
        val_loss, val_psnr = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}")
        
        history['phase1_train_loss'].append(train_loss)
        history['phase1_train_psnr'].append(train_psnr)
        history['phase1_val_loss'].append(val_loss)
        history['phase1_val_psnr'].append(val_psnr)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'fpn_phase1_best.pth'))
            print(f"  ✓ Saved best model (PSNR: {best_psnr:.2f})")
    
    # ============ PHASE 2: Fine-tune all ============
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning all layers")
    print("="*50)
    
    optimizer = Adam(model.parameters(), lr=args.lr_phase2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_phase2)
    
    for epoch in range(1, args.epochs_phase2 + 1):
        print(f"\nEpoch {epoch}/{args.epochs_phase2}")
        
        train_loss, train_psnr = train_epoch(model, train_loader, optimizer, device, freeze_backbone=False)
        val_loss, val_psnr = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}")
        
        history['phase2_train_loss'].append(train_loss)
        history['phase2_train_psnr'].append(train_psnr)
        history['phase2_val_loss'].append(val_loss)
        history['phase2_val_psnr'].append(val_psnr)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'fpn_best.pth'))
            print(f"  ✓ Saved best model (PSNR: {best_psnr:.2f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))
    
    print("\n" + "="*50)
    print(f"Training complete! Best PSNR: {best_psnr:.2f}")
    print(f"Model saved to: {args.output_dir}/fpn_best.pth")
    print("="*50)
    
    # --- Save Training Plot ---
    print("\n📈 Saving Training Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Phase 1 Loss
    axes[0, 0].plot(history['phase1_train_loss'], 'b-o', label='Train')
    axes[0, 0].plot(history['phase1_val_loss'], 'r--s', label='Val')
    axes[0, 0].set_title('Phase 1: Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Phase 1 PSNR
    axes[0, 1].plot(history['phase1_train_psnr'], 'b-o', label='Train')
    axes[0, 1].plot(history['phase1_val_psnr'], 'r--s', label='Val')
    axes[0, 1].set_title('Phase 1: PSNR (dB)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Phase 2 Loss
    axes[1, 0].plot(history['phase2_train_loss'], 'b-o', label='Train')
    axes[1, 0].plot(history['phase2_val_loss'], 'r--s', label='Val')
    axes[1, 0].set_title('Phase 2: Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Phase 2 PSNR
    axes[1, 1].plot(history['phase2_train_psnr'], 'b-o', label='Train')
    axes[1, 1].plot(history['phase2_val_psnr'], 'r--s', label='Val')
    axes[1, 1].set_title('Phase 2: PSNR (dB)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle('FPN Training Progress', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, 'training_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Plot saved: {plot_path}")


if __name__ == '__main__':
    main()
