import os

# ==========================================
# KONFIGURASI KONTEN FILE
# ==========================================

# 1. KODE UNTUK utils/dataset.py
code_dataset = r'''import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, mode='train', limit_train=12000, limit_test=3000):
        self.root_dir = root_dir
        self.mode = mode
        
        # Ekstensi gambar yang dicari
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.all_image_paths = []
        
        # Scan folder secara recursive (mencari sampai ke sub-folder)
        print(f"🔍 Scanning gambar di: {root_dir}")
        for ext in extensions:
            found = glob.glob(os.path.join(root_dir, '**', ext), recursive=True)
            self.all_image_paths.extend(found)
        
        # Hapus duplikat & Shuffle
        self.all_image_paths = sorted(list(set(self.all_image_paths)))
        np.random.seed(42) 
        np.random.shuffle(self.all_image_paths)
        
        total_images = len(self.all_image_paths)
        print(f"   Ditemukan {total_images} gambar.")
        
        if total_images == 0:
            print("⚠️ PERINGATAN: Tidak ada gambar ditemukan di folder tersebut!")

        # Split Train/Test
        if mode == 'train':
            end_idx = min(limit_train, total_images)
            self.image_paths = self.all_image_paths[:end_idx]
        elif mode == 'test':
            start_idx = limit_train
            end_idx = min(limit_train + limit_test, total_images)
            self.image_paths = self.all_image_paths[start_idx:end_idx] if start_idx < total_images else []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Ambil nama folder sebagai label
        parent_folder = os.path.basename(os.path.dirname(img_path))
        label_name = parent_folder

        try:
            image_rgb = Image.open(img_path).convert('RGB')
        except:
            image_rgb = Image.new('RGB', (256, 256)) # Dummy hitam jika error

        # Resize & Transform
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        target_rgb = transform(image_rgb)
        target_gray = transforms.Grayscale(num_output_channels=1)(target_rgb)
        
        # Tambah Noise
        noise = torch.randn_like(target_gray) * 0.1 
        input_noisy = torch.clamp(target_gray + noise, 0., 1.)
        
        return {
            'input': input_noisy, 
            'target_gray': target_gray, 
            'target_rgb': target_rgb, 
            'label': label_name
        }
'''

# 2. KODE UNTUK models/unet_crossfield.py
code_model = r'''import torch
import torch.nn as nn

class ControlledCrossFeedback(nn.Module):
    """
    Modul Cross-Field Terkontrol:
    1. Menggunakan .detach() (Stop Gradient) agar error denoise tidak merusak colorizer.
    2. Menggunakan Gating (Sigmoid) untuk memfilter fitur warna.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, denoise_feat, color_feat):
        # Stop Gradient (PENTING)
        color_feat_detached = color_feat.detach()
        
        # Attention Gate
        attention = self.gate(color_feat_detached)
        
        # Injeksi Fitur
        fused = denoise_feat + (color_feat_detached * attention)
        return self.fuse(fused)

class DualTaskUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Shared)
        self.enc1 = self.conv_block(1, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder Colorizer (Helper Task)
        self.up_c1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_c1 = self.conv_block(512 + 256, 256)
        self.up_c2 = nn.Upsample(scale_factor=2)
        self.dec_c2 = self.conv_block(256 + 128, 128)
        self.up_c3 = nn.Upsample(scale_factor=2)
        self.dec_c3 = self.conv_block(128 + 64, 64)
        self.head_color = nn.Conv2d(64, 3, 1) 
        
        # Decoder Denoiser (Main Task) + Cross Feedback
        self.cross1 = ControlledCrossFeedback(256)
        self.cross2 = ControlledCrossFeedback(128)
        
        self.up_d1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_d1 = self.conv_block(512 + 256, 256)
        self.up_d2 = nn.Upsample(scale_factor=2)
        self.dec_d2 = self.conv_block(256 + 128, 128)
        self.up_d3 = nn.Upsample(scale_factor=2)
        self.dec_d3 = self.conv_block(128 + 64, 64)
        self.head_denoise = nn.Conv2d(64, 1, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b = self.bottleneck(p3)
        
        # Colorizer Path
        c_up1 = self.up_c1(b)
        c_feat1 = self.dec_c1(torch.cat([c_up1, e3], dim=1))
        c_up2 = self.up_c2(c_feat1)
        c_feat2 = self.dec_c2(torch.cat([c_up2, e2], dim=1))
        c_up3 = self.up_c3(c_feat2)
        c_feat3 = self.dec_c3(torch.cat([c_up3, e1], dim=1))
        out_color = self.head_color(c_feat3)
        
        # Denoiser Path + Injeksi
        d_up1 = self.up_d1(b)
        d_raw1 = self.dec_d1(torch.cat([d_up1, e3], dim=1))
        
        d_fused1 = self.cross1(d_raw1, c_feat1) # Injeksi 1
        
        d_up2 = self.up_d2(d_fused1)
        d_raw2 = self.dec_d2(torch.cat([d_up2, e2], dim=1))
        
        d_fused2 = self.cross2(d_raw2, c_feat2) # Injeksi 2
        
        d_up3 = self.up_d3(d_fused2)
        d_raw3 = self.dec_d3(torch.cat([d_up3, e1], dim=1))
        out_denoise = self.head_denoise(d_raw3)
        
        return out_denoise, out_color
'''

# 3. KODE UNTUK train_crossfield.py
code_train = r'''import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import modul lokal
from models.unet_crossfield import DualTaskUNet
from utils.dataset import SimpleImageDataset

# --- KONFIGURASI ---
EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! PENTING: GANTI PATH INI DENGAN LOKASI DATASET HASIL EKSTRAK DI LAPTOP ANDA !!!
# Contoh: r"C:\Users\Hp\Downloads\dataset_extracted\data_filtered"
DATA_PATH = r"C:\Users\Hp\Downloads\dataset_extracted\data_filtered" 

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print(f"🚀 Device: {DEVICE}")
    print(f"📂 Dataset: {DATA_PATH}")
    
    # Cek path dulu
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: Folder dataset tidak ditemukan di: {DATA_PATH}")
        print("   Silakan edit file 'train_crossfield.py' dan perbaiki variabel DATA_PATH.")
        return

    # Setup Data
    try:
        # num_workers=0 agar aman di Windows
        train_ds = SimpleImageDataset(DATA_PATH, mode='train')
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print(f"✅ Data Loaded: {len(train_ds)} images")
    except Exception as e:
        print(f"❌ Error Data: {e}")
        return

    # Setup Model
    model = DualTaskUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("🔥 Mulai Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for batch in loop:
            input_img = batch['input'].to(DEVICE)
            target_g = batch['target_gray'].to(DEVICE)
            target_c = batch['target_rgb'].to(DEVICE)
            
            optimizer.zero_grad()
            pred_denoise, pred_color = model(input_img)
            
            # Loss Function: Fokus utama Denoiser + 0.5 Colorizer
            loss_d = criterion(pred_denoise, target_g)
            loss_c = criterion(pred_color, target_c)
            total_loss = loss_d + (0.5 * loss_c)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        # Save Checkpoint
        save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint disimpan: {save_path}")

if __name__ == "__main__":
    main()
'''

# 4. KODE UNTUK evaluate.py
code_eval = r'''import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.unet_crossfield import DualTaskUNet
from utils.dataset import SimpleImageDataset
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! SESUAIKAN PATH DATASET !!!
DATA_PATH = r"C:\Users\Hp\Downloads\dataset_extracted\data_filtered"
MODEL_PATH = "outputs/model_epoch_5.pth" # Load model epoch terakhir

def show_results():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model tidak ditemukan: {MODEL_PATH}. Jalankan training dulu!")
        return

    test_ds = SimpleImageDataset(DATA_PATH, mode='test', limit_test=20)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=0)
    
    model = DualTaskUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    batch = next(iter(test_loader))
    inputs = batch['input'].to(DEVICE)
    pred_d, pred_c = model(inputs)
    
    plt.figure(figsize=(10, 8))
    for i in range(3):
        if i >= len(inputs): break
        plt.subplot(3, 3, i*3 + 1); plt.imshow(inputs[i].cpu().squeeze(), cmap='gray'); plt.title("Input"); plt.axis('off')
        plt.subplot(3, 3, i*3 + 2); plt.imshow(pred_d[i].cpu().detach().squeeze(), cmap='gray'); plt.title("Output Denoise"); plt.axis('off')
        plt.subplot(3, 3, i*3 + 3); plt.imshow(pred_c[i].cpu().detach().permute(1,2,0)); plt.title("Output Color"); plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_results()
'''

# ==========================================
# EKSEKUSI PEMBUATAN FOLDER & FILE
# ==========================================

# 1. Buat Struktur Folder
folders = ["models", "utils", "outputs"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Folder dibuat: {folder}/")

# 2. Tulis File
def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 File dibuat: {path}")

write_file(os.path.join("utils", "dataset.py"), code_dataset)
write_file(os.path.join("models", "unet_crossfield.py"), code_model)
write_file("train_crossfield.py", code_train)
write_file("evaluate.py", code_eval)

print("\n🎉 SETUP SELESAI!")
print("Cara Menjalankan:")
print("1. Buka file 'train_crossfield.py'.")
print("2. Ubah variabel 'DATA_PATH' ke lokasi folder dataset Anda.")
print("3. Jalankan 'python train_crossfield.py' di terminal.")