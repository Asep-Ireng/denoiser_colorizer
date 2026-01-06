import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import pandas as pd
import traceback

# --- SETUP PATH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'models'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- IMPORT METRICS ---
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_score
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

# --- MODEL MANAGER ---
class ModelManager:
    def __init__(self):
        self.proposed_model = None
        self.baseline_model = None

    def load_proposed(self, file_obj):
        try:
            from models.drunet_crossfield import DualTaskDRUNet
            self.proposed_model = DualTaskDRUNet().to(DEVICE)
            state = torch.load(file_obj.name, map_location=DEVICE)
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            self.proposed_model.load_state_dict(new_state, strict=False)
            self.proposed_model.eval()
            return "✅ Proposed Model Berhasil Dimuat"
        except Exception as e:
            return f"❌ Error Proposed: {str(e)}"

    def load_baseline(self):
        try:
            from models.baseline_arch import BaselineCascade
            self.baseline_model = BaselineCascade().to(DEVICE)
            self.baseline_model.eval()
            return "✅ Baseline Model Berhasil Dimuat"
        except Exception as e:
            return f"❌ Error Baseline: {str(e)}"

manager = ModelManager()

# --- FUNGSI HITUNG PSNR ---
def get_psnr(model, img_tensor, gt_np, noise_level):
    if model is None: return 0.0
    img_gray = transforms.Grayscale(1)(img_tensor)
    noise = torch.randn_like(img_gray) * (noise_level / 255.0)
    input_noisy = torch.clamp(img_gray + noise, 0, 1)
    
    with torch.no_grad():
        _, pred_color = model(input_noisy)
    
    pred_np = (pred_color.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return psnr_score(gt_np, pred_np, data_range=255)

# --- FUNGSI PROSES UTAMA ---
def process_full(input_img, noise_level):
    if input_img is None: return None, None, None, None
    
    # Pre-processing
    gt_pil = input_img.convert('RGB').resize((256, 256))
    gt_np = np.array(gt_pil)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(gt_pil).unsqueeze(0).to(DEVICE)
    
    # 1. Jalankan Model Aktif untuk Visual
    # Kita asumsikan menampilkan hasil dari Proposed jika ada, jika tidak Baseline
    active_model = manager.proposed_model if manager.proposed_model else manager.baseline_model
    if active_model is None: return None, None, None, None

    img_gray = transforms.Grayscale(1)(img_tensor)
    noise = torch.randn_like(img_gray) * (noise_level / 255.0)
    input_noisy = torch.clamp(img_gray + noise, 0, 1)
    
    with torch.no_grad():
        pred_dn, pred_cl = active_model(input_noisy)

    to_pil = lambda t: transforms.ToPILImage()(t.squeeze().cpu().clamp(0, 1))
    
    # 2. Hitung Benchmark untuk Tabel (Baseline vs Proposed)
    psnr_baseline = get_psnr(manager.baseline_model, img_tensor, gt_np, noise_level) if manager.baseline_model else 0.0
    psnr_proposed = get_psnr(manager.proposed_model, img_tensor, gt_np, noise_level) if manager.proposed_model else 0.0

    # Buat Dataframe sesuai format gambar yang Anda berikan
    data = {
        "Model": ["Baseline", "Controled Cross-Field"],
        "σ=15": [round(psnr_baseline, 2) if noise_level == 15 else "-", round(psnr_proposed, 2) if noise_level == 15 else "-"],
        "σ=25": [round(psnr_baseline, 2) if noise_level == 25 else "-", round(psnr_proposed, 2) if noise_level == 25 else "-"],
        "σ=50": [round(psnr_baseline, 2) if noise_level == 50 else "-", round(psnr_proposed, 2) if noise_level == 50 else "-"],
        "Average": [round(psnr_baseline, 2), round(psnr_proposed, 2)]
    }
    
    return to_pil(input_noisy), to_pil(pred_dn), to_pil(pred_cl), pd.DataFrame(data)

# --- UI GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Image Restoration Benchmark System")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Konfigurasi")
            file_proposed = gr.File(label="Upload Proposed Model (.pth)")
            btn_load_p = gr.Button("📂 LOAD PROPOSED")
            btn_load_b = gr.Button("🏗️ LOAD BASELINE")
            status_txt = gr.Textbox(label="Status")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Visualisasi")
            img_in = gr.Image(type="pil", label="Input Ground Truth")
            noise_sl = gr.Slider(15, 50, 25, step=None, label="Noise Level (σ)")
            run_btn = gr.Button("🚀 RUN PROCESS", variant="primary")

    with gr.Row():
        out_n = gr.Image(label="Noisy")
        out_d = gr.Image(label="Denoised")
        out_c = gr.Image(label="Colorized")

    gr.Markdown("### 📊 Data Benchmark (PSNR)")
    out_table = gr.Dataframe(label="Tabel Evaluasi Kuantitatif")

    # Interaksi
    btn_load_p.click(manager.load_proposed, inputs=file_proposed, outputs=status_txt)
    btn_load_b.click(manager.load_baseline, outputs=status_txt)
    run_btn.click(process_full, inputs=[img_in, noise_sl], outputs=[out_n, out_d, out_c, out_table])

demo.launch()