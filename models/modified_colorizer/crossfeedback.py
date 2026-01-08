# =================================================================
# SCRIPT TRAINING DRUNET (LOCAL VERSION) - CrossFeedback
# =================================================================

import os
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & ARGS ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train DualTaskDRUNet with CrossFeedback')
    parser.add_argument('--data_dir', type=str, default='data_filtered/train',
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_dir', type=str, default='data_filtered/test',
                        help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, default='weights/modified_colorizer',
                        help='Directory to save models and plots')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers for data loading')
    parser.add_argument('--val_interval', type=int, default=5, help='Run validation every N epochs')
    parser.add_argument('--save_name', type=str, default='model_drunet_final.pth',
                        help='Name of the saved model file')
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET CLASS ---
class ColorizationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # Recursively find all images
        self.files = sorted([
            p for p in self.root_dir.rglob('*') 
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])
        print(f"📊 Total Training Images Found: {len(self.files)}")

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        try: 
            img = Image.open(self.files[idx]).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open {self.files[idx]}: {e}")
            img = Image.new('RGB', (256, 256)) # Fallback
        
        # Preprocessing
        tf = transforms.Compose([
            transforms.Resize((256,256)), 
            transforms.ToTensor()
        ])
        rgb = tf(img)
        gray = transforms.Grayscale(1)(rgb)
        
        # Add Noise for Input
        noise = torch.randn_like(gray) * 0.1
        noisy = torch.clamp(gray + noise, 0, 1)
        
        return noisy, gray, rgb

# --- 3. ARCHITECTURE (DualTaskDRUNet) ---
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv(x) + x)

class ControlledCrossFeedback(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(c, c, 3, padding=1)
    def forward(self, d, c_feat): 
        return self.fuse(d + (c_feat.detach() * self.gate(c_feat.detach())))

class DualTaskDRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def layer(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(), ResBlock(o))
        
        self.enc1 = layer(1, 64); self.pool = nn.MaxPool2d(2)
        self.enc2 = layer(64, 128); self.enc3 = layer(128, 256); self.bot = layer(256, 512)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc1 = layer(768, 256); self.dc2 = layer(384, 128); self.dc3 = layer(192, 64)
        self.hc = nn.Conv2d(64, 3, 1)
        
        self.cf1 = ControlledCrossFeedback(256); self.cf2 = ControlledCrossFeedback(128)
        self.dd1 = layer(768, 256); self.dd2 = layer(384, 128); self.dd3 = layer(192, 64)
        self.hd = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); b = self.bot(self.pool(e3))
        c1 = self.dc1(torch.cat([self.up(b), e3], 1)); c2 = self.dc2(torch.cat([self.up(c1), e2], 1)); c3 = self.dc3(torch.cat([self.up(c2), e1], 1))
        d1 = self.cf1(self.dd1(torch.cat([self.up(b), e3], 1)), c1); d2 = self.cf2(self.dd2(torch.cat([self.up(d1), e2], 1)), c2); d3 = self.dd3(torch.cat([self.up(d2), e1], 1))
        return self.hd(d3), self.hc(c3)

def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))

# --- 4. VALIDATION FUNCTION ---
def validate(model, val_loader, loss_fn, device):
    """Run validation and return average loss and PSNR values."""
    model.eval()
    total_loss, total_psnr_d, total_psnr_c, steps = 0, 0, 0, 0
    
    with torch.no_grad():
        for n, g, c in tqdm(val_loader, desc="Validating", unit="batch"):
            n, g, c = n.to(device), g.to(device), c.to(device)
            pd, pc = model(n)
            
            loss = loss_fn(pd, g) + 0.5 * loss_fn(pc, c)
            total_loss += loss.item()
            total_psnr_d += calc_psnr(torch.clamp(pd, 0, 1), g)
            total_psnr_c += calc_psnr(torch.clamp(pc, 0, 1), c)
            steps += 1
    
    avg_loss = total_loss / steps if steps > 0 else 0
    avg_psnr_d = total_psnr_d / steps if steps > 0 else 0
    avg_psnr_c = total_psnr_c / steps if steps > 0 else 0
    
    return avg_loss, avg_psnr_d, avg_psnr_c

# --- 5. MAIN EXECUTION ---
def main():
    args = parse_args()
    
    print(f"🖥️  Running on: {DEVICE}")
    print(f"📂 Data Directory: {args.data_dir}")
    
    # Check paths
    if not os.path.exists(args.data_dir):
        print(f"❌ ERROR: Data directory '{args.data_dir}' not found!")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.save_name)
    plot_path = os.path.join(args.output_dir, 'training_plot.png')

    # Load Dataset
    ds = ColorizationDataset(args.data_dir)
    if len(ds) == 0:
        print("❌ No images found in dataset folder.")
        return

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                    num_workers=args.num_workers, pin_memory=True)
    
    # Load Validation Dataset
    val_ds = ColorizationDataset(args.val_dir)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"📊 Validation Images: {len(val_ds)}")
    
    # Setup Model
    model = DualTaskDRUNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()  # For Mixed Precision
    
    history = {'loss': [], 'psnr_d': [], 'psnr_c': []}

    print(f"\n🚀 START TRAINING ({args.epochs} Epochs)")
    print(f"⚡ Mixed Precision (AMP): Enabled")
    print(f"💾 Saving to: {save_path}")

    try:
        for epoch in range(args.epochs):
            model.train()
            loop = tqdm(dl, unit="batch", desc=f"Ep {epoch+1}/{args.epochs}")
            
            rl, rpd, rpc, steps = 0, 0, 0, 0
            
            for n, g, c in loop:
                n, g, c = n.to(DEVICE), g.to(DEVICE), c.to(DEVICE)
                
                opt.zero_grad()
                
                # Mixed Precision Forward Pass
                with autocast():
                    pd, pc = model(n)
                    loss = loss_fn(pd, g) + 0.5 * loss_fn(pc, c)
                
                # Scaled Backward Pass
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                rl += loss.item()
                rpd += calc_psnr(torch.clamp(pd, 0, 1), g)
                rpc += calc_psnr(torch.clamp(pc, 0, 1), c)
                steps += 1
                
                loop.set_postfix(loss=loss.item())
            
            # Epoch Stats
            avg_loss = rl/steps if steps > 0 else 0
            avg_pd = rpd/steps if steps > 0 else 0
            avg_pc = rpc/steps if steps > 0 else 0
            
            history['loss'].append(avg_loss)
            history['psnr_d'].append(avg_pd)
            history['psnr_c'].append(avg_pc)
            
            print(f"   Done Ep {epoch+1} -> Loss: {avg_loss:.4f} | PSNR D: {avg_pd:.2f} dB | PSNR C: {avg_pc:.2f} dB")
            
            # Validation and checkpoint every val_interval epochs
            if (epoch + 1) % args.val_interval == 0:
                val_loss, val_psnr_d, val_psnr_c = validate(model, val_dl, loss_fn, DEVICE)
                print(f"   📊 Validation -> Loss: {val_loss:.4f} | PSNR D: {val_psnr_d:.2f} dB | PSNR C: {val_psnr_c:.2f} dB")
                torch.save(model.state_dict(), save_path.replace('.pth', f'_ep{epoch+1}.pth'))
                print(f"   💾 Checkpoint saved: ep{epoch+1}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        print("💾 Saving current model state...")

    # Save Final Model
    print(f"\n💾 Saving Final Model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved.")

    # Save Plot
    print("📈 Saving Training Plot...")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history['loss'], 'r-o'); plt.title('Loss')
    plt.subplot(1,2,2); plt.plot(history['psnr_d'], 'b-o', label='Denoise'); plt.plot(history['psnr_c'], 'g--s', label='Color'); plt.legend(); plt.title('PSNR')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"✅ Plot saved: {plot_path}")

if __name__ == "__main__":
    main()
