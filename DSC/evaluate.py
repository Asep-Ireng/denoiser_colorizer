"""
Evaluation Script - Compare PSNR across 3 denoiser models

Usage:
    python evaluate.py --num_images 100
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'denoiser'))
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'modified'))

from network_unet import UNetRes
from models.modified.dsc_unet import UNetResDSC

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_psnr(pred, target):
    """Compute PSNR between two numpy arrays (0-1 range)."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0) - 10 * np.log10(mse)


def load_models():
    """Load all models."""
    models = {}
    
    # Baseline
    print("Loading Baseline...")
    baseline = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    baseline_weights = PROJECT_ROOT / 'weights' / 'denoiser' / 'drunet_gray.pth'
    if baseline_weights.exists():
        baseline.load_state_dict(torch.load(baseline_weights, map_location=DEVICE, weights_only=True))
    baseline.to(DEVICE).eval()
    models['Baseline'] = baseline
    
      # DSC Model

    print("Loading DSC Model...")

    dsc_model = UNetResDSC(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')

    dsc_weights = PROJECT_ROOT / 'weights' / 'dsc' / 'dsc_best.pth'

    if dsc_weights.exists():

        dsc_model.load_state_dict(torch.load(dsc_weights, map_location=DEVICE, weights_only=True))

    dsc_model.to(DEVICE).eval()

    models['DSC'] = dsc_model

    return models


@torch.no_grad()
def denoise_image(model, noisy, noise_level):
    """Denoise a single image."""
    h, w = noisy.shape
    
    # Pad to divisible by 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Create input
    noise_map = np.full_like(noisy, noise_level / 255.0)
    input_tensor = np.stack([noisy, noise_map], axis=0)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(DEVICE)
    
    # Denoise
    output = model(input_tensor)
    result = output.squeeze().cpu().numpy()
    
    # Crop
    result = result[:h, :w]
    return np.clip(result, 0, 1)


def evaluate(test_dir, models, num_images=100, noise_levels=[15, 25, 50]):
    """Evaluate all models on test images."""
    
    # Get image paths
    image_paths = list(Path(test_dir).glob('*.jpg'))
    if len(image_paths) > num_images:
        image_paths = random.sample(image_paths, num_images)
    
    print(f"\nEvaluating on {len(image_paths)} images...")
    
    results = {name: {nl: [] for nl in noise_levels} for name in models.keys()}
    
    for img_path in tqdm(image_paths, desc="Evaluating"):
        # Load image
        img = Image.open(img_path).convert('L')
        clean = np.array(img).astype(np.float32) / 255.0
        
        # Resize if too large
        if clean.shape[0] > 256 or clean.shape[1] > 256:
            img = img.resize((256, 256), Image.BILINEAR)
            clean = np.array(img).astype(np.float32) / 255.0
        
        for noise_level in noise_levels:
            # Add noise
            sigma = noise_level / 255.0
            noise = np.random.randn(*clean.shape).astype(np.float32) * sigma
            noisy = np.clip(clean + noise, 0, 1)
            
            for name, model in models.items():
                denoised = denoise_image(model, noisy, noise_level)
                psnr = compute_psnr(denoised, clean)
                results[name][noise_level].append(psnr)
    
    return results


def print_results(results, noise_levels):
    """Print results table."""
    print("\n" + "="*60)
    print("PSNR Results (dB) - Higher is Better")
    print("="*60)
    
    # Header
    header = f"{'Model':<15}"
    for nl in noise_levels:
        header += f" | σ={nl:<4}"
    header += " | Average"
    print(header)
    print("-"*60)
    
    # Data rows
    for name, nl_results in results.items():
        row = f"{name:<15}"
        avg_all = []
        for nl in noise_levels:
            avg = np.mean(nl_results[nl])
            avg_all.append(avg)
            row += f" | {avg:>5.2f}"
        row += f" | {np.mean(avg_all):>5.2f}"
        print(row)
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate denoiser models')
    parser.add_argument('--test_dir', type=str, default='data_filtered/test',
                        help='Directory with test images')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--noise_levels', type=int, nargs='+', default=[15, 25, 50])
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    # Load models
    models = load_models()
    
    # Evaluate
    results = evaluate(args.test_dir, models, args.num_images, args.noise_levels)
    
    # Print results
    print_results(results, args.noise_levels)


if __name__ == '__main__':
    main()
