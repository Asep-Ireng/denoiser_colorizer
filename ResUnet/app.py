"""
Gradio UI: Tabbed Interface for Restoration Comparison
Tab 1: Denoiser Comparison (Plain vs Residual vs DRUNet)
Tab 2: Colorizer (DDColor)
Flow: Denoise -> Send Best Result to Colorizer

Fix: DRUNet in_nc=2 (Image + Noise Map) handling.

"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2

# --- SETUP PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. Path untuk Plain & Residual (models/modified)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'modified'))

# 2. Path untuk DRUNet (models/denoiser)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'denoiser'))

# --- IMPORT MODELS ---
from residual_unet import ResUNet
from plain_unet import PlainUNet

# Import DRUNet (Handling potential class name differences)
try:
    from network_unet import UNetRes as DRUNet
    HAS_DRUNET_CLASS = True
except ImportError:
    try:
        from network_unet import DRUNet
        HAS_DRUNET_CLASS = True
    except ImportError:
        print("⚠️  Gagal import DRUNet. Menggunakan ResUNet sebagai placeholder.")
        from residual_unet import ResUNet as DRUNet
        HAS_DRUNET_CLASS = False

# Import ModelScope Colorizer
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.outputs import OutputKeys
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False
    print("⚠️  Library 'modelscope' tidak ditemukan. Colorizer mungkin tidak berjalan.")

# Device Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- KONFIGURASI ---
TARGET_EPOCH = 20
COLORIZER_WEIGHTS = os.path.join(PROJECT_ROOT, "weights/colorizer/ddcolor_paper.pth")
BASELINE_WEIGHTS  = os.path.join(PROJECT_ROOT, "weights/denoiser/drunet_gray.pth")


class ModelManager:
    def __init__(self):
        self.plain_model = None
        self.res_model = None
        self.baseline_model = None
        self.colorizer_pipeline = None
    
    def load_single_model(self, model_class, folder_name, file_prefix, epoch):
        """Memuat Plain/Residual Model (Biasanya in_nc=1)."""
        weights_dir = os.path.join(PROJECT_ROOT, "weights", folder_name)
        possible_names = [
            f"{file_prefix}_epoch{epoch}.pth",
            f"{file_prefix}_denoiser_epoch{epoch}.pth",
            f"{file_prefix}_denoiser_{epoch}_epoch.pth"
        ]
        
        path = None
        for name in possible_names:
            p = os.path.join(weights_dir, name)
            if os.path.exists(p):
                path = p
                break
        
        if path:
            try:
                # Plain & ResUNet biasanya in_nc=1
                model = model_class(in_nc=1, out_nc=1)
                state_dict = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE).eval()
                print(f"  ✅ Loaded: {file_prefix.capitalize()} Epoch {epoch}")
                return model
            except Exception as e:
                print(f"  ❌ Error loading {file_prefix}: {e}")
                return None
        else:
            print(f"  ⚠️  Missing file for {file_prefix} Epoch {epoch}")
            return None

    def load_baseline(self):
        """Memuat DRUNet Baseline (Fix: in_nc=2)."""
        if os.path.exists(BASELINE_WEIGHTS):
            try:
                # FIX: DRUNet weights expects 2 channels (Image + Noise Map)
                model = DRUNet(in_nc=2, out_nc=1) 
                
                state_dict = torch.load(BASELINE_WEIGHTS, map_location=DEVICE)
                if 'params' in state_dict: state_dict = state_dict['params']
                
                model.load_state_dict(state_dict, strict=True) # Strict True untuk memastikan struktur pas
                model.to(DEVICE).eval()
                self.baseline_model = model
                print("  ✅ Baseline DRUNet Loaded (in_nc=2)!")
            except Exception as e:
                print(f"  ❌ Failed to load Baseline: {e}")
                print("     (Pastikan arsitektur network_unet.py mendukung in_nc=2)")
        else:
            print(f"  ⚠️  Baseline file not found at: {BASELINE_WEIGHTS}")

    def load_colorizer(self):
        """Memuat DDColor."""
        if not HAS_MODELSCOPE: return None
        try:
            color_pipe = pipeline(
                Tasks.image_colorization, 
                model='damo/cv_ddcolor_image-colorization',
                device='gpu' if torch.cuda.is_available() else 'cpu'
            )
            if os.path.exists(COLORIZER_WEIGHTS):
                local_state = torch.load(COLORIZER_WEIGHTS, map_location=DEVICE)
                new_state = {}
                for k, v in local_state.items():
                    if k.startswith('module.'): k = k[7:]
                    if k.startswith('generator.'): k = k[10:]
                    if k.startswith('params.'): k = k[7:]
                    new_state[k] = v
                color_pipe.model.load_state_dict(new_state, strict=False)
                print("  ✅ Colorizer Ready!")
            return color_pipe
        except Exception as e:
            print(f"  ❌ Failed to load Colorizer: {e}")
            return None

    def load_all(self):
        print("\n=== Loading Models ===")
        self.plain_model = self.load_single_model(PlainUNet, "plain", "plain", TARGET_EPOCH)
        self.res_model = self.load_single_model(ResUNet, "residualblocks", "residual", TARGET_EPOCH)
        self.load_baseline()
        self.colorizer_pipeline = self.load_colorizer()
        print("======================\n")


model_manager = ModelManager()


# ============== INFERENCE FUNCTIONS ==============

def add_noise(img, sigma):
    noise = np.random.randn(*img.shape) * (sigma / 255.0)
    return np.clip(img + noise, 0, 1)

def to_gray(img):
    if len(img.shape) == 3:
        return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return img

@torch.no_grad()
def infer_denoise(model, img_gray, noise_level_sigma=0):
    """
    Inference Denoiser dengan penanganan otomatis untuk in_nc=1 vs in_nc=2.
    """
    if model is None: return img_gray
    
    h, w = img_gray.shape
    # Padding kelipatan 8
    ph = (8 - h%8)%8
    pw = (8 - w%8)%8
    img_pad = np.pad(img_gray, ((0,ph), (0,pw)), 'reflect') if (ph+pw)>0 else img_gray
    
    # Buat Tensor Gambar (1 Channel)
    img_tensor = torch.from_numpy(img_pad).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Cek jumlah input channel yang dibutuhkan model
    # Kita cek layer pertama (biasanya module.head atau m_head)
    try:
        # Mencoba akses parameter layer pertama untuk cek shape
        first_layer_weight = next(model.parameters())
        required_channels = first_layer_weight.shape[1]
    except:
        # Default fallback ke 1 jika gagal deteksi
        required_channels = 1
        
    if required_channels == 2:
        # Jika model butuh 2 channel (DRUNet), kita buat Noise Map
        sigma_map = torch.full((1, 1, img_pad.shape[0], img_pad.shape[1]), noise_level_sigma/255.0).float().to(DEVICE)
        inp = torch.cat([img_tensor, sigma_map], dim=1) # Concat Channel (B, 2, H, W)
    else:
        # Jika model butuh 1 channel (Plain/ResUNet)
        inp = img_tensor

    # Inference
    out = model(inp)
    
    if isinstance(out, (list, tuple)): out = out[0]
    out = out.squeeze().cpu().numpy()
    
    return np.clip(out[:h, :w], 0, 1)

def infer_colorize(pipeline, img_rgb_or_gray):
    if pipeline is None: return None
    try:
        # Pipeline modelscope butuh input BGR uint8 atau RGB
        if len(img_rgb_or_gray.shape) == 2:
             img_bgr = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_GRAY2BGR)
        else:
             img_bgr = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)

        result = pipeline(img_bgr)
        output_img = result[OutputKeys.OUTPUT_IMG] # BGR numpy
        return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Colorize Error: {e}")
        return None

# ============== PROCESS FUNCTIONS FOR UI ==============

def process_denoiser(input_pil, noise_level, add_noise_cb):
    if input_pil is None: return [None] * 4
    
    img_rgb = np.array(input_pil)
    img = img_rgb.astype(np.float32) / 255.0
    gray = to_gray(img)
    
    if add_noise_cb:
        noisy = add_noise(gray, noise_level)
    else:
        noisy = gray

    # Pass noise_level ke fungsi inference agar DRUNet bisa bikin noise map
    
    # 1. Plain (in_nc=1, noise_level diabaikan di infer_denoise)
    plain_out = infer_denoise(model_manager.plain_model, noisy, noise_level)
    
    # 2. Residual (in_nc=1, noise_level diabaikan di infer_denoise)
    res_out = infer_denoise(model_manager.res_model, noisy, noise_level)
    
    # 3. Baseline (in_nc=2, noise_level dipakai untuk map)
    base_out = infer_denoise(model_manager.baseline_model, noisy, noise_level)
    
    # Convert ke uint8 untuk display
    return [
        (noisy * 255).astype(np.uint8),
        (plain_out * 255).astype(np.uint8) if plain_out is not None else None,
        (res_out * 255).astype(np.uint8) if res_out is not None else None,
        (base_out * 255).astype(np.uint8) if base_out is not None else None
    ]

def process_colorizer(input_pil):
    if input_pil is None: return None
    img_np = np.array(input_pil)
    return infer_colorize(model_manager.colorizer_pipeline, img_np)

# ============== UI LAYOUT ==============

def create_ui():
    with gr.Blocks(title="Final Project: Restoration UI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🖼️ Image Restoration: Denoising & Colorization")
        gr.Markdown("Comparing **Plain U-Net**, **Residual U-Net**, and **DRUNet**.")
        
        with gr.Tabs():
            
            # --- TAB 1: DENOISER ---
            with gr.TabItem("🔇 Denoiser Comparison"):
                gr.Markdown("### Step 1: Compare Denoising Models")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        denoise_input = gr.Image(label="Input Image", type="pil", height=250)
                        noise_slider = gr.Slider(0, 100, 25, label="Noise Level (Sigma)")
                        add_noise_cb = gr.Checkbox(value=True, label="Add Noise?")
                        denoise_btn = gr.Button("🚀 Run Comparison", variant="primary")
                    
                    with gr.Column(scale=1):
                        out_noisy = gr.Image(label="Noisy Input", type="numpy", height=250)

                gr.Markdown("---")
                gr.Markdown("### Denoising Results")
                
                with gr.Row():
                    out_plain = gr.Image(label=f"Plain U-Net (Ep {TARGET_EPOCH})", type="numpy")
                    out_res = gr.Image(label=f"Residual U-Net (Ep {TARGET_EPOCH})", type="numpy")
                    out_base = gr.Image(label="Baseline (DRUNet)", type="numpy")
                
                gr.Markdown("### Next Step: Send Result to Colorizer")
                with gr.Row():
                    btn_send_res = gr.Button("📤 Send Residual Result to Colorizer", variant="secondary")
                    btn_send_base = gr.Button("📤 Send DRUNet Result to Colorizer", variant="secondary")

            # --- TAB 2: COLORIZER ---
            with gr.TabItem("🎨 Colorizer"):
                gr.Markdown("### Step 2: Colorize Grayscale Image (DDColor)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        col_input = gr.Image(label="Grayscale Input", type="pil", height=300)
                        col_btn = gr.Button("🎨 Colorize Image", variant="primary")
                    
                    with gr.Column(scale=1):
                        col_output = gr.Image(label="Colorized Output", type="numpy", height=300)

        # --- EVENT HANDLERS ---
        
        denoise_btn.click(
            process_denoiser,
            inputs=[denoise_input, noise_slider, add_noise_cb],
            outputs=[out_noisy, out_plain, out_res, out_base]
        )
        
        col_btn.click(
            process_colorizer,
            inputs=[col_input],
            outputs=[col_output]
        )
        
        # Simple transfer functions
        def send_image(img): return img 
        
        btn_send_res.click(
            send_image, 
            inputs=[out_res], 
            outputs=[col_input]
        )
        
        btn_send_base.click(
            send_image, 
            inputs=[out_base], 
            outputs=[col_input]
        )

    return app

if __name__ == "__main__":
    model_manager.load_all()
    ui = create_ui()
    ui.launch(share=False)