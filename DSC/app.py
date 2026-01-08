"""
Gradio UI for Image Denoiser & Colorizer
Features: Comparison of Epochs 5, 10, 15, 20 for DSC Model.

Usage:
    python app.py
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import sys

# Add project root and models directory to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'denoiser'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'colorizer'))

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Paths
DENOISER_WEIGHTS = os.path.join(PROJECT_ROOT, "weights/denoiser/drunet_gray.pth")
DSC_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights/dsc") # Folder folder dsc
COLORIZER_WEIGHTS = os.path.join(PROJECT_ROOT, "weights/colorizer/ddcolor_paper.pth")


class ModelManager:
    """Manages loading and inference for denoiser and colorizer models."""
    
    def __init__(self):
        self.denoiser_baseline = None
        self.denoiser_dsc = None
        self.colorizer = None
    
    def load_denoiser_baseline(self):
        """Load the baseline UNetRes denoiser (DRUNet)."""
        try:
            import basicblock as B
            from network_unet import UNetRes
            model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
            if os.path.exists(DENOISER_WEIGHTS):
                state_dict = torch.load(DENOISER_WEIGHTS, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict, strict=True)
                print(f"✓ Loaded baseline denoiser weights")
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to load baseline denoiser: {e}")
            return None
    
    def get_dsc_model_structure(self):
        """Returns the DSC model structure (without weights)."""
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'modified'))
            from models.modified.dsc_unet import UNetResDSC
            model = UNetResDSC(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to initialize DSC architecture: {e}")
            return None

    def load_dsc_weights(self, model, epoch):
        """Load specific epoch weights into the provided model instance."""
        # ---------------------------------------------------------
        # SESUAIKAN FORMAT NAMA FILE DI SINI JIKA PERLU
        # Contoh format: 'model_epoch_5.pth' atau 'checkpoint_5.pth'
        filenames_to_try = [
            f"model_epoch_{epoch}.pth",
            f"checkpoint_epoch_{epoch}.pth",
            f"dsc_epoch_{epoch}.pth",
            f"{epoch}.pth" 
        ]
        # ---------------------------------------------------------

        weight_path = None
        for fname in filenames_to_try:
            path = os.path.join(DSC_WEIGHTS_DIR, fname)
            if os.path.exists(path):
                weight_path = path
                break
        
        if weight_path:
            try:
                state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                return True
            except Exception as e:
                print(f"⚠ Error loading {weight_path}: {e}")
                return False
        else:
            print(f"⚠ Weights for Epoch {epoch} not found in {DSC_WEIGHTS_DIR}")
            return False

    def load_colorizer(self):
        """Load the DDColor colorizer."""
        try:
            from ddcolor import DDColor
            model = DDColor(
                encoder_name='convnext-l',
                decoder_name='MultiScaleColorDecoder',
                input_size=[256, 256],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            )
            if os.path.exists(COLORIZER_WEIGHTS):
                state_dict = torch.load(COLORIZER_WEIGHTS, map_location=DEVICE, weights_only=True)
                if 'params' in state_dict: state_dict = state_dict['params']
                model_state = model.state_dict()
                filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                model.load_state_dict(filtered_state, strict=False)
                print(f"✓ Loaded colorizer")
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to load colorizer: {e}")
            return None
    
    def load_initial_models(self):
        print("\n" + "="*50)
        print("Loading base models...")
        self.denoiser_baseline = self.load_denoiser_baseline()
        # Kita memuat struktur DSC sekali saja, nanti bobotnya di-swap saat runtime
        self.denoiser_dsc = self.get_dsc_model_structure()
        self.colorizer = self.load_colorizer()
        print("="*50 + "\n")


model_manager = ModelManager()


# ============== DENOISER FUNCTIONS ==============

def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + noise
    return np.clip(noisy, 0, 1)

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    return image

@torch.no_grad()
def run_denoiser(noisy_gray: np.ndarray, noise_level: float, model) -> np.ndarray:
    original_h, original_w = noisy_gray.shape
    pad_h = (8 - original_h % 8) % 8
    pad_w = (8 - original_w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        noisy_padded = np.pad(noisy_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        noisy_padded = noisy_gray
    
    h, w = noisy_padded.shape
    noise_map = np.full((h, w), noise_level / 255.0, dtype=np.float32)
    
    input_tensor = np.stack([noisy_padded.astype(np.float32), noise_map], axis=0)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(DEVICE)
    
    output = model(input_tensor)
    result = output.squeeze().cpu().numpy()
    result = result[:original_h, :original_w]
    return np.clip(result, 0, 1)

def process_denoiser(input_image, noise_level, add_noise: bool):
    """
    Process image:
    1. Baseline
    2. Epoch 5, 10, 15, 20
    """
    if input_image is None:
        return [None] * 6 # Return 6 Nones
    
    # Preprocessing
    image = np.array(input_image)
    if image.max() > 1: image = image.astype(np.float32) / 255.0
    gray = to_grayscale(image)
    
    if add_noise: noisy = add_gaussian_noise(gray, noise_level / 255.0)
    else: noisy = gray
    
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    
    # 1. Run Baseline
    if model_manager.denoiser_baseline:
        res_base = run_denoiser(noisy, noise_level, model_manager.denoiser_baseline)
        base_uint8 = (res_base * 255).astype(np.uint8)
    else:
        base_uint8 = noisy_uint8
    
    # 2. Run Epochs 5, 10, 15, 20
    epoch_results = []
    target_epochs = [5, 10, 15, 20]
    
    dsc_model = model_manager.denoiser_dsc
    
    if dsc_model is None:
        # Jika model gagal load, kembalikan gambar noisy utk semua
        epoch_results = [noisy_uint8] * 4
    else:
        for epoch in target_epochs:
            print(f"Running Inference for Epoch {epoch}...")
            # Load weights dinamis
            success = model_manager.load_dsc_weights(dsc_model, epoch)
            
            if success:
                res = run_denoiser(noisy, noise_level, dsc_model)
                res_uint8 = (res * 255).astype(np.uint8)
                epoch_results.append(res_uint8)
            else:
                # Jika weight tidak ketemu, pakai gambar hitam atau noisy dengan text
                epoch_results.append(noisy_uint8) # Fallback ke noisy
    
    # Return: Noisy, Baseline, Ep5, Ep10, Ep15, Ep20
    return [noisy_uint8, base_uint8] + epoch_results


# ============== COLORIZER FUNCTIONS ==============
# (Code Colorizer tetap sama, disingkat untuk menghemat tempat)
@torch.no_grad()
def run_colorizer(img: np.ndarray, model) -> np.ndarray:
    import cv2
    target_size = 256
    height, width = img.shape[:2]
    if img.max() > 1: img = (img / 255.0).astype(np.float32)
    else: img = img.astype(np.float32)
    
    if len(img.shape) == 2: img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    elif img.shape[2] == 1: img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    
    orig_l = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)[:, :, :1]
    img_resized = cv2.resize(img, (target_size, target_size))
    img_l = cv2.cvtColor(img_resized.astype(np.float32), cv2.COLOR_RGB2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
    tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(DEVICE)
    output_ab = model(tensor_gray_rgb).cpu()
    
    output_ab_resized = F.interpolate(output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_rgb = cv2.cvtColor(output_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(output_rgb, 0, 1)

def process_colorizer(input_image):
    if input_image is None: return None
    image = np.array(input_image)
    if model_manager.colorizer: colorized = run_colorizer(image, model_manager.colorizer)
    else: colorized = image.astype(np.float32) / 255.0 if image.max() > 1 else image
    return (colorized * 255).astype(np.uint8)


# ============== UI ==============

def create_ui():
    """Create the Gradio interface with tabs."""
    
    with gr.Blocks(title="Denoiser & Colorizer Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Image Denoiser Training Analysis")
        
        with gr.Tabs():
            # ============== DENOISER TAB ==============
            with gr.TabItem("🔇 Denoiser Epoch Comparison"):
                gr.Markdown("### Compare DSC Model Training Progression (Epoch 5 vs 10 vs 15 vs 20)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        denoise_input = gr.Image(label="📷 Input Image", type="pil", height=250)
                        noise_slider = gr.Slider(5, 75, 25, step=5, label="Noise Level (σ)")
                        add_noise_cb = gr.Checkbox(value=True, label="Add Noise to Input")
                        denoise_btn = gr.Button("🚀 Run Analysis (All Epochs)", variant="primary", size="lg")
                
                gr.Markdown("### 1. Baselines")
                with gr.Row():
                    denoise_noisy = gr.Image(label="Input + Noise", height=250)
                    denoise_baseline = gr.Image(label="Baseline (DRUNet)", height=250)

                gr.Markdown("### 2. DSC Model Training Progression")
                with gr.Row():
                    out_ep5 = gr.Image(label="Epoch 5", height=200)
                    out_ep10 = gr.Image(label="Epoch 10", height=200)
                    out_ep15 = gr.Image(label="Epoch 15", height=200)
                    out_ep20 = gr.Image(label="Epoch 20 (Final)", height=200)
                
                with gr.Row():
                    # Kirim hasil Epoch 20 (terbaik) ke colorizer
                    send_to_colorizer_btn = gr.Button("📤 Send Epoch 20 Result to Colorizer →", variant="secondary")
            
            # ============== COLORIZER TAB ==============
            with gr.TabItem("🎨 Colorizer"):
                gr.Markdown("### Colorize grayscale images")
                with gr.Row():
                    with gr.Column(scale=1):
                        colorize_input = gr.Image(label="📷 Grayscale Input", type="pil", height=250)
                        colorize_btn = gr.Button("🎨 Colorize", variant="primary", size="lg")
                with gr.Row():
                    colorize_output = gr.Image(label="🎨 Colorized Output", height=350)
                with gr.Row():
                    send_to_denoiser_btn = gr.Button("📤 Send Result to Denoiser →", variant="secondary")
        
        # ============== EVENT HANDLERS ==============
        
        # Output urutannya: Noisy, Baseline, Ep5, Ep10, Ep15, Ep20
        denoise_btn.click(
            fn=process_denoiser,
            inputs=[denoise_input, noise_slider, add_noise_cb],
            outputs=[denoise_noisy, denoise_baseline, out_ep5, out_ep10, out_ep15, out_ep20]
        )
        
        colorize_btn.click(
            fn=process_colorizer,
            inputs=[colorize_input],
            outputs=[colorize_output]
        )
        
        # Send buttons (Mengirim Epoch 20 ke Colorizer)
        send_to_colorizer_btn.click(lambda x: x, inputs=[out_ep20], outputs=[colorize_input])
        send_to_denoiser_btn.click(lambda x: x, inputs=[colorize_output], outputs=[denoise_input])
    
    return demo


if __name__ == "__main__":
    print("Initializing Models...")
    model_manager.load_initial_models()
    
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)