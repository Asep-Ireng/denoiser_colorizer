import torch
import torch.nn as nn
import os
import sys

# --- 1. SETUP PATH YANG LEBIH KUAT ---
# Ambil lokasi file ini (models/baseline_arch.py)
current_file = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file)          # Folder: .../models
project_root = os.path.dirname(models_dir)          # Folder: .../Project_Anda
weights_dir = os.path.join(project_root, 'weights') # Folder: .../weights

# FUNGSI DARURAT: Tambahkan semua kemungkinan path ke sistem
# Ini mencegah error "ModuleNotFoundError: No module named 'denoiser'"
paths_to_add = [
    project_root,
    models_dir,
    os.path.join(models_dir, 'denoiser'),  # Agar bisa import basicblock dll
    os.path.join(models_dir, 'ddcolor_arch_utils')
]

for p in paths_to_add:
    if p not in sys.path:
        sys.path.append(p)

# --- 2. IMPORT ARSITEKTUR ---
# Kita gunakan Try-Except bertingkat untuk menangani berbagai cara struktur folder
try:
    # Cara 1: Langsung (jika folder models sudah di path)
    from denoiser.network_unet import UNet
    from ddcolor import DDColor
except ImportError:
    try:
        # Cara 2: Pakai prefix models.
        from models.denoiser.network_unet import UNet
        from models.ddcolor import DDColor
    except ImportError as e:
        print(f"❌ CRITICAL IMPORT ERROR: {e}")
        # Definisi Dummy agar tidak crash saat import, tapi akan error saat dipakai
        UNet = None
        DDColor = None

class BaselineCascade(nn.Module):
    def __init__(self):
        super(BaselineCascade, self).__init__()
        
        print(f"🏗️  System: Menginisialisasi Baseline di root: {project_root}")
        
        # Cek apakah import berhasil
        if UNet is None or DDColor is None:
            raise ImportError("Gagal mengimport script model (UNet/DDColor). Cek folder 'models/'!")

        # 1. Setup Denoiser
        self.denoiser = UNet(
            in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, 
            act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'
        )

        # 2. Setup Colorizer
        self.colorizer = DDColor(
            encoder_name='convnext-l', nf=512, num_queries=256, 
            num_scales=3, dec_layers=9, input_size=(256, 256)
        )
        
        # 3. Load Weights
        self._load_weights_automatically()

    def _load_weights_automatically(self):
        file_denoise = "baseline_denoiser.pth"
        file_color   = "baseline_ddcolor.pth"

        path_denoise = os.path.join(weights_dir, file_denoise)
        path_color   = os.path.join(weights_dir, file_color)

        # Cek keberadaan file
        if not os.path.exists(path_denoise):
            raise FileNotFoundError(f"❌ File tidak ditemukan: {path_denoise}")
        
        if not os.path.exists(path_color):
            raise FileNotFoundError(f"❌ File tidak ditemukan: {path_color}")

        # Load Denoiser
        print(f"🔹 Loading: {file_denoise}")
        state = torch.load(path_denoise, map_location='cpu')
        self.denoiser.load_state_dict(
            {k.replace('module.', ''): v for k, v in state.items()}, strict=False
        )

        # Load Colorizer
        print(f"🔹 Loading: {file_color}")
        state = torch.load(path_color, map_location='cpu')
        self.colorizer.load_state_dict(
            {k.replace('module.', ''): v for k, v in state.items()}, strict=False
        )
        print("✅ Semua weights Baseline berhasil dimuat.")

    def forward(self, x):
        if x.shape[1] == 3: 
            x_gray_input = x[:, :1, :, :]
        else:
            x_gray_input = x

        clean_gray = self.denoiser(x_gray_input)
        
        clean_clamped = torch.clamp(clean_gray, 0, 1)
        clean_3ch = clean_clamped.repeat(1, 3, 1, 1)

        color_out = self.colorizer(clean_3ch)
        
        return clean_clamped, color_out