import os
import sys
import random
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_metric

# IMPORT MODEL
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'denoiser'))
try:
    from unet_attention import AttentionUNetRes
except ImportError:
    # Fallback jika file ada di root folder
    try:
        import unet_attention
        AttentionUNetRes = unet_attention.AttentionUNetRes
    except ImportError:
        print("Error: File 'unet_attention.py' tidak ditemukan.")
        sys.exit(1)

# CHARBONNIER LOSS
class CharbonnierLoss(nn.Module):
    """L1 Loss variant that is differentiable at zero (smoother convergence)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # Formula: sqrt(diff^2 + eps^2)
        loss = torch.sqrt((diff * diff) + (self.eps*self.eps))
        return torch.mean(loss)

# 1. KONFIGURASI
class Config:
    TRAIN_DIR = 'data_filtered/train'
    VAL_DIR   = 'data_filtered/test'
    PRETRAINED_PATH = 'weights/drunet_gray.pth'
    OUTPUT_DIR = 'weights/attention_final_run'
    LOG_CSV = 'training_log.csv'
    
    PATCH_SIZE = 128      
    BATCH_SIZE = 10        
    NUM_WORKERS = 4       
    
    # Phase 1 Params (Warmup)
    EPOCHS_P1 = 10
    LR_P1 = 1e-4
    
    # Phase 2 Params (Fine-tune)
    EPOCHS_P2 = 20        
    LR_P2 = 1e-5          
    
    NOISE_MIN = 5
    NOISE_MAX = 75

# 2. DATASET (DENGAN FIX PADDING)
class DynamicDenoisingDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, mode='train'):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.mode = mode
        self.image_paths = sorted(list(self.root_dir.glob('*.[jJ][pP]*[gG]')) + list(self.root_dir.glob('*.png')))
        
        if len(self.image_paths) == 0:
            print(f"WARNING: Tidak ada gambar di {self.root_dir}")
        else:
            print(f"[{mode.upper()}] Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('L')
        img = np.array(img).astype(np.float32) / 255.0

        h, w = img.shape
        
        # Jika gambar lebih kecil dari patch, lakukan padding refleksi
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        
        if pad_h > 0 or pad_w > 0:
            # Mode 'reflect' menjaga kontinuitas visual tanpa merusak statistik noise
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape
            
        if self.mode == 'train':
            # Random Crop
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img = img[top:top+self.patch_size, left:left+self.patch_size]
            
            # Augmentasi
            if random.random() > 0.5: img = np.flipud(img)
            if random.random() > 0.5: img = np.fliplr(img)
        else:
            # Center Crop untuk validasi
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
            img = img[top:top+self.patch_size, left:left+self.patch_size]

        # Pastikan array contiguous di memori
        img = np.ascontiguousarray(img)

        # Generate Noise
        noise_level = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
        sigma = noise_level / 255.0
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        noisy = np.clip(img + noise, 0, 1)
        
        # Channel 2: Noise Map
        noise_map = np.full_like(noisy, sigma)

        # Stack jadi (2, H, W)
        input_tensor = torch.from_numpy(np.stack([noisy, noise_map], axis=0))
        target_tensor = torch.from_numpy(img[np.newaxis, ...])

        return input_tensor, target_tensor

# UTILS
def calc_ssim(img1, img2):
    # Pindah ke CPU & numpy hanya saat perhitungan
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    return ssim_metric(img1, img2, data_range=1.0)

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_backbone(model, path, device):
    if not os.path.exists(path):
        print(f"Warning: Pretrained path '{path}' not found. Training from scratch.")
        return model
        
    print(f"Loading Backbone: {path}")
    state_dict = torch.load(path, map_location=device)
    if 'model_state' in state_dict: state_dict = state_dict['model_state']
    
    model_dict = model.state_dict()
    # Filter bobot yang bentuknya cocok (shape matching)
    pretrained_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"✓ Loaded {len(pretrained_dict)} layers successfully.")
    return model

# 3. TRAINING ENGINE
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, freeze_mode=False):
    model.train()
    
    # LOGIC FREEZE
    if freeze_mode:
        for name, param in model.named_parameters():
            # UNFREEZE:
            # 1. 'att' : Komponen baru (Attention Gates)
            # 2. 'm_head' : Pintu masuk (Adaptasi karakteristik data/noise kita) 
            # 3. 'm_tail' : Pintu keluar (Adaptasi fitur yang sudah terfilter attention) 
            if any(k in name for k in ['att', 'm_head', 'm_tail']):
                param.requires_grad = True
            else:
                # FREEZE:
                # Bagian tengah (ResBlocks Encoder/Decoder) tetap beku agar ilmunya tidak rusak
                param.requires_grad = False
    else:
        # Phase 2: Semua layer dilatih
        for param in model.parameters():
            param.requires_grad = True
            
    avg_loss = 0
    loop = tqdm(loader, desc="Train", leave=False)
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            preds = model(inputs)
            loss = criterion(preds, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        avg_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return avg_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    avg_loss = 0
    avg_psnr = 0
    avg_ssim = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward & Clamp
            preds = torch.clamp(model(inputs), 0, 1)
            
            loss = criterion(preds, targets)
            avg_loss += loss.item()
            
            avg_psnr += compute_psnr(preds, targets).item()
            avg_ssim += calc_ssim(preds, targets)

    return avg_loss / len(loader), avg_psnr / len(loader), avg_ssim / len(loader)

# 4. MAIN PROGRAM
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True # Optimasi
    print(f"Device: {device}")

    # Dataset
    train_ds = DynamicDenoisingDataset(Config.TRAIN_DIR, Config.PATCH_SIZE, mode='train')
    val_ds = DynamicDenoisingDataset(Config.VAL_DIR, Config.PATCH_SIZE, mode='val')
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model
    # in_nc=2 (Image + Noise Map) -> Sesuai standar DPIR/DRUNet
    model = AttentionUNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4).to(device)
    model = load_backbone(model, Config.PRETRAINED_PATH, device)
    
    print("Using Charbonnier Loss")
    criterion = CharbonnierLoss(eps=1e-3).to(device)
    
    scaler = torch.amp.GradScaler('cuda')

    # Logger
    csv_file = open(Config.LOG_CSV, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Phase', 'Epoch', 'Train_Loss', 'Val_Loss', 'Val_PSNR', 'Val_SSIM', 'Time_Sec', 'LR'])

    print("\n=== START TRAINING ===")
    
    # PHASE 1: WARMUP
    print(f"\n[PHASE 1] Warmup: Attention + Head + Tail ({Config.EPOCHS_P1} Epochs)")
    
    # Init Optimizer hanya untuk parameter yang aktif
    for name, param in model.named_parameters():
        if any(k in name for k in ['att', 'm_head', 'm_tail']):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR_P1)
    best_psnr = 0
    
    for epoch in range(Config.EPOCHS_P1):
        start_t = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, freeze_mode=True)
        val_loss, psnr, ssim = validate(model, val_loader, criterion, device)
        duration = time.time() - start_t
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"P1 Ep{epoch+1:02d} | TL:{loss:.5f} | VL:{val_loss:.5f} | PSNR:{psnr:.2f}dB | {duration:.0f}s")
        writer.writerow(['Phase1', epoch+1, loss, val_loss, psnr, ssim, duration, current_lr])
        csv_file.flush()
        
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), f"{Config.OUTPUT_DIR}/att_best_phase1.pth")

    # PHASE 2: FINE-TUNING
    print(f"\n[PHASE 2] Global Fine-Tuning ({Config.EPOCHS_P2} Epochs)")
    
    # Unfreeze ALL Layers
    for param in model.parameters(): param.requires_grad = True
    
    # Optimizer baru untuk seluruh model dengan LR lebih kecil
    optimizer = optim.Adam(model.parameters(), lr=Config.LR_P2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_P2, eta_min=1e-7)
    
    for epoch in range(Config.EPOCHS_P2):
        start_t = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, freeze_mode=False)
        val_loss, psnr, ssim = validate(model, val_loader, criterion, device)
        duration = time.time() - start_t
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"P2 Ep{epoch+1:02d} | TL:{loss:.5f} | VL:{val_loss:.5f} | PSNR:{psnr:.2f}dB | {duration:.0f}s")
        writer.writerow(['Phase2', epoch+1, loss, val_loss, psnr, ssim, duration, current_lr])
        csv_file.flush()
        
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), f"{Config.OUTPUT_DIR}/att_best_final.pth")
            print(">>> New Best Model Saved!")

    csv_file.close()
    print("Training Selesai.")

if __name__ == "__main__":
    main()