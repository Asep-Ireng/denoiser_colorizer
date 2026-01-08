import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import io
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from collections import OrderedDict

# 1. SETUP & MODEL LOADING

try:
    from models.denoiser.network_unet import UNetRes
    from models.denoiser.unet_attention import AttentionUNetRes
except ImportError:
    try:
        from network_unet import UNetRes
        from unet_attention import AttentionUNetRes
    except ImportError:
        print("File model python tidak ditemukan")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print("Loading Models: Drunet & Best Final")
    
    # 1. Model Benchmark: DRUNet
    model_drunet = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R').to(device)
    
    # 2. Model Utama: Attention U-Net Final
    model_final = AttentionUNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R').to(device)
    
    base_path = "weights/" 
    path_drunet = os.path.join(base_path, "drunet_gray.pth")
    path_final = os.path.join(base_path, "att_best_final.pth")

    def clean_state_dict(path):
        if not os.path.exists(path):
            print(f"File {path} tidak ditemukan.")
            return None
        try:
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
            return new_state_dict
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None

    # Load Weights
    w_drunet = clean_state_dict(path_drunet)
    w_final = clean_state_dict(path_final)

    if w_drunet: model_drunet.load_state_dict(w_drunet, strict=True)
    if w_final: model_final.load_state_dict(w_final, strict=False)

    model_drunet.eval()
    model_final.eval()
    
    return model_drunet, model_final

# Load Models Global
m_drunet, m_final = load_models()

# 2. HELPER FUNCTIONS

def get_crop(img_tensor, x_pct, y_pct, crop_size=64):
    _, h, w = img_tensor.shape
    center_x = int((x_pct / 100) * w)
    inv_y_pct = 100 - y_pct
    center_y = int((inv_y_pct / 100) * h)
    
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    
    if x2 - x1 < crop_size: x1 = max(0, w - crop_size)
    if y2 - y1 < crop_size: y1 = max(0, h - crop_size)
    
    cropped = img_tensor[:, y1:y2, x1:x2]
    zoom_factor = 4
    cropped_np = (cropped.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(cropped_np)
    img_pil = img_pil.resize((crop_size * zoom_factor, crop_size * zoom_factor), resample=Image.NEAREST)
    return np.array(img_pil)

def generate_residual_map(pred_tensor, gt_tensor):
    diff = torch.abs(pred_tensor - gt_tensor).squeeze().cpu().numpy()
    diff_norm = np.clip(diff / 0.15, 0, 1) # Sensitivity adjusted
    heatmap = cm.inferno(diff_norm)[:, :, :3] 
    return (heatmap * 255).astype(np.uint8)

def create_histogram(gt, drunet, final):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    bins = 100
    
    ax.hist(gt.flatten(), bins=bins, color='green', alpha=0.3, label='Ground Truth', density=True)
    ax.hist(drunet.flatten(), bins=bins, color='gray', alpha=0.7, label='DRUNet', histtype='step', linestyle='--', density=True)
    ax.hist(final.flatten(), bins=bins, color='red', alpha=1.0, label='Attention Best', histtype='step', linewidth=1.5, density=True)
    
    ax.set_title("Pixel Intensity Distribution Comparison")
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# 3. PROCESS IMAGE

def process_image(image, noise_level, zoom_x, zoom_y):
    global m_drunet, m_final
    
    if image is None: return [None] * 12

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_pil = Image.fromarray(image).convert('L') 
    gt_tensor = transform(img_pil).to(device)
    
    # Noise
    sigma = noise_level / 255.0
    noise = torch.randn_like(gt_tensor) * sigma
    noisy_tensor = torch.clamp(gt_tensor + noise, 0, 1)
    
    # Input Batch
    noise_map = torch.full_like(noisy_tensor, sigma)
    input_batch = torch.cat([noisy_tensor, noise_map], dim=0).unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred_drunet = torch.clamp(m_drunet(input_batch), 0, 1)
        pred_final = torch.clamp(m_final(input_batch), 0, 1)

    # Conversions
    def to_u8(t): return (t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    gt_np = gt_tensor.squeeze().cpu().numpy()
    noisy_np = noisy_tensor.squeeze().cpu().numpy()
    d_np = pred_drunet.squeeze().cpu().numpy()
    f_np = pred_final.squeeze().cpu().numpy()

    # Metrics
    pd_val, sd_val = psnr_metric(gt_np, d_np, data_range=1.0), ssim_metric(gt_np, d_np, data_range=1.0)
    pf_val, sf_val = psnr_metric(gt_np, f_np, data_range=1.0), ssim_metric(gt_np, f_np, data_range=1.0)
    pn_val, sn_val = psnr_metric(gt_np, noisy_np, data_range=1.0), ssim_metric(gt_np, noisy_np, data_range=1.0)

    # DataFrame Logic (Updated)
    df_metrics = pd.DataFrame({
        "Model": ["Noisy Input", "DRUNet (base)", "Attention Modification"],
        "PSNR (dB)": [f"{pn_val:.2f}", f"{pd_val:.2f}", f"{pf_val:.2f}"],
        "SSIM": [f"{sn_val:.4f}", f"{sd_val:.4f}", f"{sf_val:.4f}"],
        "Improvement (dB)": ["-", f"+{pd_val - pn_val:.2f}", f"+{pf_val - pn_val:.2f}"]
    })

    # Visuals
    vis_full = [to_u8(noisy_tensor), to_u8(gt_tensor), to_u8(pred_drunet), to_u8(pred_final)]
    vis_zoom = [
        get_crop(noisy_tensor, zoom_x, zoom_y), get_crop(gt_tensor, zoom_x, zoom_y),
        get_crop(pred_drunet[0], zoom_x, zoom_y), get_crop(pred_final[0], zoom_x, zoom_y)
    ]
    vis_res = [generate_residual_map(pred_drunet.cpu(), gt_tensor.cpu()), generate_residual_map(pred_final.cpu(), gt_tensor.cpu())]
    hist_img = create_histogram(gt_np, d_np, f_np)
    
    return [df_metrics] + vis_full + vis_zoom + vis_res + [hist_img]

# 4. TRAINING PLOTS (DIPISAH)

def get_training_plots():
    csv_path = "training_log_attention.csv"
    if not os.path.exists(csv_path): return None, None, None

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        p1 = df[df['Phase'] == 'Phase1'].copy()
        p2 = df[df['Phase'] == 'Phase2'].copy()
        
        # Merge Epochs
        max_epoch_p1 = p1['Epoch'].max() if not p1.empty else 0
        p2['Global_Epoch'] = p2['Epoch'] + max_epoch_p1
        p1['Global_Epoch'] = p1['Epoch']
        df_full = pd.concat([p1, p2])

        # Plot Config
        def create_plot(y_data, title, ylabel, color, is_loss=False):
            fig = plt.figure(figsize=(8, 5))
            if is_loss:
                plt.plot(df_full['Global_Epoch'], df_full['Train_Loss'], label='Train', color='lightblue', linestyle='--')
                plt.plot(df_full['Global_Epoch'], df_full['Val_Loss'], label='Validation', color=color)
            else:
                plt.plot(df_full['Global_Epoch'], y_data, label=ylabel, color=color)
            
            if max_epoch_p1 > 0:
                plt.axvline(x=max_epoch_p1 + 0.5, color='gray', linestyle=':', label='Phase Switch')
            
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.close(fig)
            return fig

        fig_loss = create_plot(None, "Loss Convergence", "Loss", "blue", is_loss=True)
        fig_psnr = create_plot(df_full['Val_PSNR'], f"PSNR (Max: {df_full['Val_PSNR'].max():.2f} dB)", "PSNR (dB)", "red")
        fig_ssim = create_plot(df_full['Val_SSIM'], "SSIM Metric", "SSIM", "purple")

        return fig_loss, fig_psnr, fig_ssim
    except Exception as e:
        print(f"Plot Error: {e}")
        return None, None, None

init_fig_loss, init_fig_psnr, init_fig_ssim = get_training_plots()

# 5. GRADIO INTERFACE

with gr.Blocks(title="Final Denoising Analysis") as demo:
    gr.Markdown("# Analisis Komparatif Akhir: DRUNet vs Attention U-Net")
    
    with gr.Tabs():
        with gr.TabItem("Comparison Dashboard"):
            with gr.Row():
                # KOLOM KIRI: HANYA INPUT DAN KONTROL
                with gr.Column(scale=1):
                    gr.Markdown("### Input & Kontrol")
                    input_img = gr.Image(label="Upload Ground Truth", type="numpy")
                    noise_slider = gr.Slider(15, 75, value=25, step=1, label="Noise Level (Sigma)")
                    
                    gr.Markdown("### Zoom Navigator")
                    zoom_x = gr.Slider(0, 100, value=50, label="Posisi X (%)")
                    zoom_y = gr.Slider(0, 100, value=50, label="Posisi Y (%)")
                    
                    btn_run = gr.Button("Proses Analisis", variant="primary")

                # KOLOM KANAN: METRIK DAN VISUALISASI
                with gr.Column(scale=3):
                    gr.Markdown("### Statistik Performa")
                    score_table = gr.Dataframe(label="Quantitative Metrics")
                    
                    gr.Markdown("### 1. Perbandingan Visual (Full Image)")
                    with gr.Row():
                        out_noisy = gr.Image(label="Input (Noisy)")
                        out_gt = gr.Image(label="Ground Truth (Target)")
                    with gr.Row():
                        out_drunet = gr.Image(label="Benchmark: DRUNet")
                        out_final = gr.Image(label="Ours: Attention Best")
                    
                    gr.Markdown("### 2. Analisis Detail (Zoom Crop)")
                    with gr.Row():
                        z_noisy = gr.Image(label="Zoom Input")
                        z_gt = gr.Image(label="Zoom Gambar asli")
                    with gr.Row():
                        z_drunet = gr.Image(label="Zoom DRUNet")
                        z_final = gr.Image(label="Zoom Attention")
                    
                    gr.Markdown("### 3. Analisis Error (Heatmap)")
                    gr.Markdown("Warna lebih terang = Error lebih tinggi")
                    with gr.Row():
                        res_drunet = gr.Image(label="DRUNet")
                        res_final = gr.Image(label="Attention Best")
                    
                    gr.Markdown("### 4. Distribusi Intensitas")
                    hist_plot = gr.Image(label="Histogram Pixel Matching")

            # OUTPUT BINDING
            outs = [score_table, out_noisy, out_gt, out_drunet, out_final, 
                    z_noisy, z_gt, z_drunet, z_final, 
                    res_drunet, res_final, hist_plot]

            btn_run.click(process_image, inputs=[input_img, noise_slider, zoom_x, zoom_y], outputs=outs)
            zoom_x.release(process_image, inputs=[input_img, noise_slider, zoom_x, zoom_y], outputs=outs)
            zoom_y.release(process_image, inputs=[input_img, noise_slider, zoom_x, zoom_y], outputs=outs)

        with gr.TabItem("Training Dynamics"):
            gr.Markdown("### Log Training: Phase 1 -> Phase 2")
            with gr.Row():
                plot_loss = gr.Plot(value=init_fig_loss, label="Loss")
                plot_psnr = gr.Plot(value=init_fig_psnr, label="PSNR")
                plot_ssim = gr.Plot(value=init_fig_ssim, label="SSIM")
            gr.Button("Refresh Logs").click(get_training_plots, outputs=[plot_loss, plot_psnr, plot_ssim])

if __name__ == "__main__":
    demo.launch()