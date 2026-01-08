import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import importlib
import cv2  # pip install opencv-python
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# --- 1. SETUP PROJECT & PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'modified'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'denoiser'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'colorizer'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'DSC'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'ResUnet'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'mod_attention_gate'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'modified_colorizer'))

# --- 2. KONFIGURASI DEVICE ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")

WEIGHTS_PATHS = {
    'baseline':       os.path.join(PROJECT_ROOT, "weights/denoiser/drunet_gray.pth"),
    'dsc':            os.path.join(PROJECT_ROOT, "weights/Dsc/dsc_best.pth"),
    'fpn':            os.path.join(PROJECT_ROOT, "weights/fpn/fpn_best.pth"),
    'fpn_phase1':     os.path.join(PROJECT_ROOT, "weights/fpn/fpn_phase1_best.pth"),
    'attention':      os.path.join(PROJECT_ROOT, "weights/attention_final_run/att_best_final.pth"),
    'plain':          os.path.join(PROJECT_ROOT, "weights/ResUnet/plain_denoiser_epoch20.pth"),
    'residual':       os.path.join(PROJECT_ROOT, "weights/ResUnet/residual_denoiser_epoch20.pth"),
    'drunet_crossfield': os.path.join(PROJECT_ROOT, "weights/modified_colorizer/model_drunet_final.pth"),
    'colorizer':      os.path.join(PROJECT_ROOT, "weights/colorizer/ddcolor_paper.pth"), 
}

# --- 3. MODEL MANAGER ---
# Model configurations
MODEL_CONFIGS = {
    'Baseline':   {'module': 'network_unet', 'class': 'UNetRes', 'weights': WEIGHTS_PATHS['baseline'], 'in_nc': 2},
    'DSC':        {'module': 'dsc_unet', 'class': 'UNetResDSC', 'weights': WEIGHTS_PATHS['dsc'], 'in_nc': 2},
    'FPN':        {'module': 'fpn_unet', 'class': 'UNetResFPN', 'weights': WEIGHTS_PATHS['fpn'], 'in_nc': 2},
    'FPN-P1':     {'module': 'fpn_unet', 'class': 'UNetResFPN', 'weights': WEIGHTS_PATHS['fpn_phase1'], 'in_nc': 2},
    'Attention':  {'module': 'unetres_attention', 'class': 'AttentionUNetRes', 'weights': WEIGHTS_PATHS['attention'], 'in_nc': 2},
    'DRUNet-CF':  {'module': 'crossfeedback', 'class': 'DualTaskDRUNet', 'weights': WEIGHTS_PATHS['drunet_crossfield'], 'in_nc': 1},
    'Residual':   {'module': 'residual_unet', 'class': 'ResUNet', 'weights': WEIGHTS_PATHS['residual'], 'in_nc': 1},
    'Plain':      {'module': 'plain_unet', 'class': 'PlainUNet', 'weights': WEIGHTS_PATHS['plain'], 'in_nc': 1},
}

class ModelManager:
    def __init__(self):
        self.models = {}  # All models loaded at startup
        self.colorizer = None
    
    def _load_single_model(self, model_key):
        """Load a single model."""
        if model_key not in MODEL_CONFIGS:
            print(f"   ❌ Unknown model: {model_key}")
            return
        
        config = MODEL_CONFIGS[model_key]
        print(f"⏳ Loading {model_key}...")
        
        try:
            module = None
            prefixes = [
                "models.modified", "models.denoiser", "models.DSC",
                "models.ResUnet", "models.mod_attention_gate", "models.modified_colorizer",
            ]
            for prefix in prefixes:
                try:
                    module = importlib.import_module(f"{prefix}.{config['module']}")
                    break
                except ImportError:
                    continue
            
            if module is None:
                module = importlib.import_module(config['module'])

            model_class = getattr(module, config['class'])
            kwargs = {'in_nc': config['in_nc'], 'out_nc': 1, 'nc': [64, 128, 256, 512], 'nb': 4, 'act_mode': 'R'}
            
            try:
                model = model_class(**kwargs)
            except TypeError:
                model = model_class()
            
            if os.path.exists(config['weights']):
                raw = torch.load(config['weights'], map_location=DEVICE)
                state_dict = raw.get('params', raw.get('model_state_dict', raw.get('state_dict', raw))) if isinstance(raw, dict) else raw
                model_state = model.state_dict()
                filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                model.load_state_dict(filtered_state, strict=False)
                print(f"   ✓ Weights loaded: {len(filtered_state)} keys")
            else:
                print(f"   [!] Weights missing: {config['weights']}")

            model.to(DEVICE).eval()
            self.models[model_key] = model

        except Exception as e:
            print(f"   ❌ FAIL {model_key}: {e}")

    def load_colorizer(self):
        print("\n=== Initializing Colorizer ===")
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
            path = WEIGHTS_PATHS['colorizer']
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
                if 'params' in state_dict: state_dict = state_dict['params']
                model_state = model.state_dict()
                filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                model.load_state_dict(filtered_state, strict=False)
                model.to(DEVICE).eval()
                self.colorizer = model
                print(f"   ✓ Loaded colorizer")
            else:
                print(f"   ✗ Colorizer weights missing at {path}")
        except Exception as e:
            print(f"   ✗ Failed to load colorizer: {e}")

    def load_all_models(self):
        """Load all denoiser models at startup."""
        print("\n=== Initializing Denoisers ===")
        for model_key in MODEL_CONFIGS.keys():
            self._load_single_model(model_key)
        self.load_colorizer()
        print(f"\n✅ Loaded {len(self.models)} denoisers + colorizer")

model_manager = ModelManager()

# --- 4. HELPER FUNCTIONS ---
def add_noise(image_arr, sigma):
    noise = np.random.randn(*image_arr.shape).astype(np.float32) * (sigma / 255.0)
    return np.clip(image_arr + noise, 0, 1).astype(np.float32)

def get_heatmap_matplotlib(clean, denoised):
    clean = clean.astype(np.float32)
    denoised = denoised.astype(np.float32)
    diff = np.abs(clean - denoised)
    diff_norm = np.clip(diff * 5.0, 0, 1) 
    cmap = plt.get_cmap('magma')
    heatmap_rgba = cmap(diff_norm).astype(np.float32) 
    return (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)

def get_zoom_np(img_arr, zoom_factor=3, crop_size=60):
    h, w = img_arr.shape[:2]
    left, top = (w - crop_size) // 2, (h - crop_size) // 2
    crop = img_arr[top:top+crop_size, left:left+crop_size]
    if crop.dtype != np.uint8: crop = (crop * 255).astype(np.uint8)
    pil_crop = Image.fromarray(crop)
    zoomed = pil_crop.resize((crop_size * zoom_factor, crop_size * zoom_factor), resample=Image.NEAREST)
    return np.array(zoomed)

# --- 5. COLORIZER INFERENCE LOGIC (UPDATED WITH SIZE PARAMETER) ---
@torch.no_grad()
def run_colorizer_inference(img_gray_np, model, input_size=256):
    """
    Args:
        input_size: Resolusi proses internal (256, 512, 1024). 
                    Semakin besar = semakin detail, tapi lebih lambat/berat.
    """
    if model is None: return (img_gray_np * 255).astype(np.uint8)

    # Input Check
    img_uint8 = (img_gray_np * 255).astype(np.uint8)
    
    # Force 3 channel BGR for OpenCV
    if len(img_uint8.shape) == 2:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Extract L Channel
    orig_l = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)[:, :, :1]
    height, width = orig_l.shape[:2]

    # Resize for Model Processing
    img_resized = cv2.resize(img_float, (input_size, input_size))
    img_l_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Lab)[:, :, :1]
    
    # Duplicate L channel to fake Lab input
    img_gray_lab = np.concatenate((img_l_resized, np.zeros_like(img_l_resized), np.zeros_like(img_l_resized)), axis=-1)
    img_gray_rgb_input = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
    tensor_input = torch.from_numpy(img_gray_rgb_input.transpose((2, 0, 1))).float().unsqueeze(0).to(DEVICE)
    
    # Inference
    output_ab = model(tensor_input).cpu()
    
    # Resize AB back to original Size
    output_ab_resized = F.interpolate(output_ab, size=(height, width), mode='bilinear', align_corners=False)[0].float().numpy().transpose(1, 2, 0)
    
    # Combine Original L + Predicted AB
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_rgb = cv2.cvtColor(output_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    
    return np.clip(output_rgb * 255, 0, 255).astype(np.uint8)

# --- 6. CORE PIPELINES ---

# A. Pipeline Comparison (Denoising) - Sequential with memory cleanup
@torch.no_grad()
def run_full_pipeline(input_pil, noise_level):
    if input_pil is None: return [None]*36 + [pd.DataFrame()]
    torch.cuda.empty_cache()

    gray_pil = input_pil.convert('L')
    clean_gray_np = np.array(gray_pil).astype(np.float32) / 255.0
    noisy_gray_np = add_noise(clean_gray_np, noise_level)
    
    # Padding logic for UNet
    h, w = noisy_gray_np.shape
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    noisy_padded = np.pad(noisy_gray_np, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    noise_map = torch.full((1, 1, noisy_padded.shape[0], noisy_padded.shape[1]), noise_level / 255.0).to(DEVICE)
    input_tensor = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    input_with_map = torch.cat([input_tensor, noise_map], dim=1)
    input_single = input_tensor

    labels = ['Noisy Input', 'Baseline', 'DSC', 'FPN', 'FPN-P1', 'Residual', 'Attention', 'Plain', 'DRUNet-CF']
    all_outputs = []
    metrics = []

    for name in labels:
        result_np = None
        if name == 'Noisy Input':
            result_np = noisy_gray_np
        else:
            model = model_manager.models.get(name)
            if model is None: 
                result_np = noisy_gray_np
            else:
                try:
                    # Models with in_nc=1 use single channel input
                    if name in ['Plain', 'DRUNet-CF', 'Residual']: 
                        out = model(input_single)
                    else: 
                        out = model(input_with_map)
                    
                    # Wait for GPU to finish before next model (sequential)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    if isinstance(out, (tuple, list)): out = out[0]
                    res_padded = out.squeeze().cpu().numpy()
                    if len(res_padded.shape) == 3: res_padded = res_padded[0]
                    result_np = np.clip(res_padded[:h, :w], 0, 1)
                except Exception as e:
                    print(f"   ⚠ Inference error {name}: {e}")
                    result_np = noisy_gray_np

        val_psnr = psnr_metric(clean_gray_np, result_np, data_range=1.0)
        metrics.append({"Model": name, "PSNR": round(val_psnr, 2)})
        
        # Colorize (Default Resolution 256 for speed in grid)
        try:
            color_res = run_colorizer_inference(result_np, model_manager.colorizer, input_size=256)
        except Exception:
            gray_uint8 = (result_np * 255).astype(np.uint8)
            color_res = np.stack([gray_uint8]*3, axis=-1)

        all_outputs.append((result_np * 255).astype(np.uint8))
        all_outputs.append(get_heatmap_matplotlib(clean_gray_np, result_np))
        all_outputs.append(get_zoom_np(result_np))
        all_outputs.append(color_res)
        
        # Clear intermediate GPU cache between models
        torch.cuda.empty_cache()

    return all_outputs + [pd.DataFrame(metrics)]

# B. Pipeline Single Image Colorization
@torch.no_grad()
def run_custom_colorization(input_img, resolution):
    if input_img is None: return None
    
    # Convert input to Grayscale normalized float32
    if isinstance(input_img, Image.Image):
        img = np.array(input_img.convert('L'))
    else:
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        
    img_gray_np = img.astype(np.float32) / 255.0
    
    # Run Inference
    return run_colorizer_inference(img_gray_np, model_manager.colorizer, input_size=int(resolution))


# --- 7. UI CONSTRUCTION (TABBED) ---
def create_ui():
    labels = ['Noisy Input', 'Baseline', 'DSC', 'FPN', 'FPN-P1', 'Residual', 'Attention', 'Plain', 'DRUNet-CF']
    
    with gr.Blocks(title="Deep Learning Image Restoration", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Image Restoration & Colorization")
        
        with gr.Tabs():
            # ================= TAB 1: DENOISING & ANALYSIS =================
            with gr.TabItem("Denoising Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(label="Input Clean Image", type="pil", height=250)
                        slider = gr.Slider(5, 75, 25, step=5, label="Noise Level")
                        btn_run = gr.Button("Run Full Analysis", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("### PSNR Performance")
                        tbl = gr.Dataframe(label="Metrics")

                image_components = []
                for lbl in labels:
                    with gr.Row():
                        gr.Markdown(f"### {lbl}")
                    with gr.Row():
                        image_components.append(gr.Image(label="Denoised", height=200))
                        image_components.append(gr.Image(label="Error Heatmap", height=200))
                        image_components.append(gr.Image(label="Zoom", height=200))
                        image_components.append(gr.Image(label="Colorized", height=200))
                    gr.Markdown("---")

                btn_run.click(
                    run_full_pipeline,
                    inputs=[input_img, slider],
                    outputs=image_components + [tbl]
                )

            # ================= TAB 2: DEDICATED COLORIZER =================
            with gr.TabItem("🎨 Dedicated Colorizer"):
                gr.Markdown("### Upload BW/Old Photo to Colorize")
                with gr.Row():
                    with gr.Column():
                        bw_input = gr.Image(label="Input Black & White Image", type="pil", height=400)
                        # Slider resolusi untuk kontrol kualitas vs kecepatan
                        res_slider = gr.Slider(
                            minimum=256, maximum=1024, value=512, step=128, 
                            label="Render Resolution (Higher = More Detail, Slower)"
                        )
                        btn_colorize = gr.Button("🎨 Colorize Now", variant="primary")
                    
                    with gr.Column():
                        color_output = gr.Image(label="Colorized Result", height=500)
                
                btn_colorize.click(
                    run_custom_colorization,
                    inputs=[bw_input, res_slider],
                    outputs=[color_output]
                )

    return demo

if __name__ == "__main__":
    model_manager.load_all_models()
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)