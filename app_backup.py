"""
Gradio UI for Image Denoiser & Colorizer
Two separate tabs with ability to send results between them.

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
FPN_ARCH_WEIGHTS = os.path.join(PROJECT_ROOT, "weights/fpn/fpn_arch_only.pth")
FPN_TRAINED_WEIGHTS = os.path.join(PROJECT_ROOT, "weights/fpn/fpn_best.pth")
COLORIZER_WEIGHTS = os.path.join(PROJECT_ROOT, "weights/colorizer/ddcolor_paper.pth")


class ModelManager:
    """Manages loading and inference for denoiser and colorizer models."""
    
    def __init__(self):
        self.denoiser_baseline = None
        self.denoiser_fpn_arch = None    # FPN with random init
        self.denoiser_fpn_trained = None  # FPN after training
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
                print(f"✓ Loaded denoiser weights")
            
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to load denoiser: {e}")
            return None
    
    def load_denoiser_fpn_arch(self):
        """Load FPN model with random FPN weights (architecture-only benefit)."""
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'modified'))
            from fpn_unet import UNetResFPN
            
            model = UNetResFPN(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
            
            # Load arch-only weights (pretrained backbone + random FPN)
            if os.path.exists(FPN_ARCH_WEIGHTS):
                state_dict = torch.load(FPN_ARCH_WEIGHTS, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"✓ Loaded FPN arch-only weights")
            else:
                # Fall back to loading pretrained baseline
                if os.path.exists(DENOISER_WEIGHTS):
                    state_dict = torch.load(DENOISER_WEIGHTS, map_location=DEVICE, weights_only=True)
                    model.load_pretrained(state_dict)
            
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to load FPN arch: {e}")
            return None
    
    def load_denoiser_fpn_trained(self):
        """Load FPN model with trained weights (if available)."""
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'modified'))
            from fpn_unet import UNetResFPN
            
            model = UNetResFPN(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R')
            
            if os.path.exists(FPN_TRAINED_WEIGHTS):
                state_dict = torch.load(FPN_TRAINED_WEIGHTS, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"✓ Loaded FPN trained weights")
            else:
                print(f"⚠ FPN trained weights not found (not trained yet)")
                return None
            
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to load FPN trained: {e}")
            return None
    
    def load_colorizer(self):
        """Load the DDColor colorizer."""
        try:
            from ddcolor import DDColor
            
            # Match official DDColor inference settings
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
                if 'params' in state_dict:
                    state_dict = state_dict['params']
                
                # Filter out weight normalization params with shape mismatch
                model_state = model.state_dict()
                filtered_state = {}
                for k, v in state_dict.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state[k] = v
                        else:
                            print(f"  Skipping {k}: {v.shape} vs {model_state[k].shape}")
                    else:
                        filtered_state[k] = v  # New key, let it error if wrong
                
                missing, unexpected = model.load_state_dict(filtered_state, strict=False)
                print(f"✓ Loaded colorizer ({len(filtered_state)} params, {len(missing)} missing)")
            else:
                print(f"⚠ Colorizer weights not found")
            
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"✗ Failed to load colorizer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_models(self):
        """Load all models."""
        print("\n" + "="*50)
        print("Loading models...")
        print("="*50)
        
        self.denoiser_baseline = self.load_denoiser_baseline()
        self.denoiser_fpn_arch = self.load_denoiser_fpn_arch()
        self.denoiser_fpn_trained = self.load_denoiser_fpn_trained()
        self.colorizer = self.load_colorizer()
        
        print("="*50 + "\n")


# Global model manager
model_manager = ModelManager()


# ============== DENOISER FUNCTIONS ==============

def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + noise
    return np.clip(noisy, 0, 1)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminance."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    return image


@torch.no_grad()
def run_denoiser(noisy_gray: np.ndarray, noise_level: float, model) -> np.ndarray:
    """Denoise a grayscale image using the given model."""
    original_h, original_w = noisy_gray.shape
    
    # Pad to make dimensions divisible by 8
    pad_h = (8 - original_h % 8) % 8
    pad_w = (8 - original_w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        noisy_padded = np.pad(noisy_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        noisy_padded = noisy_gray
    
    # Create noise level map
    h, w = noisy_padded.shape
    noise_map = np.full((h, w), noise_level / 255.0, dtype=np.float32)
    
    # Stack grayscale and noise map
    input_tensor = np.stack([noisy_padded.astype(np.float32), noise_map], axis=0)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(DEVICE)
    
    # Run model
    output = model(input_tensor)
    result = output.squeeze().cpu().numpy()
    result = result[:original_h, :original_w]
    return np.clip(result, 0, 1)


def process_denoiser(input_image, noise_level, add_noise: bool):
    """Process image through 3 denoisers for comparison."""
    if input_image is None:
        return None, None, None, None, None
    
    # Convert to numpy
    image = np.array(input_image)
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    
    # Convert to grayscale
    gray = to_grayscale(image)
    
    # Add noise if requested
    if add_noise:
        noisy = add_gaussian_noise(gray, noise_level / 255.0)
    else:
        noisy = gray
    
    # Denoise with baseline
    if model_manager.denoiser_baseline is not None:
        denoised_baseline = run_denoiser(noisy, noise_level, model_manager.denoiser_baseline)
    else:
        denoised_baseline = noisy
    
    # Denoise with FPN arch-only
    if model_manager.denoiser_fpn_arch is not None:
        denoised_fpn_arch = run_denoiser(noisy, noise_level, model_manager.denoiser_fpn_arch)
    else:
        denoised_fpn_arch = noisy
    
    # Denoise with FPN trained
    if model_manager.denoiser_fpn_trained is not None:
        denoised_fpn_trained = run_denoiser(noisy, noise_level, model_manager.denoiser_fpn_trained)
    else:
        denoised_fpn_trained = None  # Not trained yet
    
    # Convert to uint8
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    baseline_uint8 = (denoised_baseline * 255).astype(np.uint8)
    fpn_arch_uint8 = (denoised_fpn_arch * 255).astype(np.uint8)
    fpn_trained_uint8 = (denoised_fpn_trained * 255).astype(np.uint8) if denoised_fpn_trained is not None else None
    
    return noisy_uint8, baseline_uint8, fpn_arch_uint8, fpn_trained_uint8


# ============== COLORIZER FUNCTIONS ==============

@torch.no_grad()
def run_colorizer(img: np.ndarray, model) -> np.ndarray:
    """Colorize an image using DDColor - matches official inference code."""
    import cv2
    import torch.nn.functional as F
    
    target_size = 256
    height, width = img.shape[:2]
    
    # Normalize to [0, 1]
    if img.max() > 1:
        img = (img / 255.0).astype(np.float32)
    else:
        img = img.astype(np.float32)
    
    # Handle grayscale input - convert to BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    elif img.shape[2] == 1:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    
    # Get original L channel (from full resolution)
    orig_l = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)[:, :, :1]  # (h, w, 1)
    
    # Resize and convert to grayscale Lab
    img_resized = cv2.resize(img, (target_size, target_size))
    img_l = cv2.cvtColor(img_resized.astype(np.float32), cv2.COLOR_RGB2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
    # Convert to tensor
    tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(DEVICE)
    
    # Run model
    output_ab = model(tensor_gray_rgb).cpu()  # (1, 2, size, size)
    
    # Resize output and concatenate with original L channel
    output_ab_resized = F.interpolate(output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_rgb = cv2.cvtColor(output_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    
    return np.clip(output_rgb, 0, 1)


def process_colorizer(input_image):
    """Process image through colorizer."""
    if input_image is None:
        return None
    
    # Convert to numpy
    image = np.array(input_image)
    
    # Colorize (handles grayscale or RGB)
    if model_manager.colorizer is not None:
        colorized = run_colorizer(image, model_manager.colorizer)
    else:
        # Fallback: return as-is
        colorized = image.astype(np.float32) / 255.0 if image.max() > 1 else image
    
    return (colorized * 255).astype(np.uint8)


# ============== UI ==============

def create_ui():
    """Create the Gradio interface with tabs."""
    
    with gr.Blocks(title="Denoiser & Colorizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ Image Denoiser & Colorizer
        
        Two separate tools - use the buttons to send results between them!
        """)
        
        with gr.Tabs():
            # ============== DENOISER TAB ==============
            with gr.TabItem("🔇 Denoiser"):
                gr.Markdown("### Denoise grayscale images - compare 3 models")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        denoise_input = gr.Image(label="📷 Input Image", type="pil", height=250)
                        noise_slider = gr.Slider(5, 75, 25, step=5, label="Noise Level (σ)")
                        add_noise_cb = gr.Checkbox(value=True, label="Add Noise to Input")
                        denoise_btn = gr.Button("🚀 Denoise", variant="primary", size="lg")
                
                gr.Markdown("### Results (4-way comparison)")
                with gr.Row():
                    denoise_noisy = gr.Image(label="Noisy Input", height=200)
                    denoise_baseline = gr.Image(label="1️⃣ Baseline U-Net", height=200)
                    denoise_fpn_arch = gr.Image(label="2️⃣ FPN (Arch Only)", height=200)
                    denoise_fpn_trained = gr.Image(label="3️⃣ FPN (Trained)", height=200)
                
                with gr.Row():
                    send_to_colorizer_btn = gr.Button("📤 Send Baseline Result to Colorizer →", variant="secondary")
                
                # Sample images
                sample_dir = os.path.join(PROJECT_ROOT, "data_filtered/test")
                if os.path.exists(sample_dir):
                    samples = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)[:6] if f.endswith('.jpg')]
                    if samples:
                        gr.Markdown("### Sample Images")
                        gr.Examples(
                            examples=[[s, 25, True] for s in samples],
                            inputs=[denoise_input, noise_slider, add_noise_cb],
                            outputs=[denoise_noisy, denoise_baseline, denoise_fpn_arch, denoise_fpn_trained],
                            fn=process_denoiser,
                            cache_examples=False
                        )
            
            # ============== COLORIZER TAB ==============
            with gr.TabItem("🎨 Colorizer"):
                gr.Markdown("### Colorize grayscale images using DDColor")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        colorize_input = gr.Image(label="📷 Grayscale Input", type="pil", height=250)
                        colorize_btn = gr.Button("🎨 Colorize", variant="primary", size="lg")
                
                with gr.Row():
                    colorize_output = gr.Image(label="🎨 Colorized Output", height=350)
                
                with gr.Row():
                    send_to_denoiser_btn = gr.Button("📤 Send Result to Denoiser →", variant="secondary")
        
        # ============== EVENT HANDLERS ==============
        
        denoise_btn.click(
            fn=process_denoiser,
            inputs=[denoise_input, noise_slider, add_noise_cb],
            outputs=[denoise_noisy, denoise_baseline, denoise_fpn_arch, denoise_fpn_trained]
        )
        
        colorize_btn.click(
            fn=process_colorizer,
            inputs=[colorize_input],
            outputs=[colorize_output]
        )
        
        # Send between tabs
        def send_image_to_colorizer(baseline_result):
            """Send denoised result to colorizer input."""
            return baseline_result
        
        def send_image_to_denoiser(colorized_result):
            """Send colorized result to denoiser input."""
            return colorized_result
        
        send_to_colorizer_btn.click(
            fn=send_image_to_colorizer,
            inputs=[denoise_baseline],
            outputs=[colorize_input]
        )
        
        send_to_denoiser_btn.click(
            fn=send_image_to_denoiser,
            inputs=[colorize_output],
            outputs=[denoise_input]
        )
        
        gr.Markdown("""
        ---
        ### 📋 Model Status
        | Model | Status |
        |-------|--------|
        | Denoiser (Baseline) | ✅ DRUNet |
        | Denoiser (FPN) | 🔄 Placeholder |
        | Colorizer | ⚠️ DDColor (weight mismatch) |
        """)
    
    return demo


if __name__ == "__main__":
    print("Initializing Denoiser & Colorizer...")
    model_manager.load_models()
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
