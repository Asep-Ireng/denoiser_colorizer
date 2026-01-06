import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import os

def load_img(path):
    """Load image, convert to grayscale, normalize to [0,1]"""
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return np.zeros((256, 256))
    
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float32) / 255.0

def visualize_progression(
    clean_path, 
    noisy_path, 
    baseline_path, 
    dsc_paths_dict, 
    save_path="outputs/comparison_analysis25.png"
):
    """
    dsc_paths_dict: Dictionary {epoch: path}
    """
    # 1. Load Images
    clean = load_img(clean_path)
    noisy = load_img(noisy_path)
    baseline = load_img(baseline_path)
    
    dsc_imgs = {}
    for ep, path in dsc_paths_dict.items():
        dsc_imgs[ep] = load_img(path)

    # 2. Setup Plot Grid
    # Row 1: Images (Clean, Noisy, Baseline, DSC Ep20)
    # Row 2: Difference Maps (vs Clean)
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 10))
    
    # --- ROW 1: REAL IMAGES ---
    # Column 1: GT
    axes[0,0].imshow(clean, cmap='gray')
    axes[0,0].set_title("Ground Truth (Clean)")
    
    # Column 2: Baseline
    axes[0,1].imshow(baseline, cmap='gray')
    axes[0,1].set_title("Baseline (DRUNet)")
    
    # Column 3: DSC Epoch 5 (Awal)
    if 5 in dsc_imgs:
        axes[0,2].imshow(dsc_imgs[5], cmap='gray')
        axes[0,2].set_title("DSC Epoch 5")

    if 10 in dsc_imgs:
        axes[0,3].imshow(dsc_imgs[10], cmap='gray')
        axes[0,3].set_title("DSC Epoch 5")

    if 15 in dsc_imgs:
        axes[0,4].imshow(dsc_imgs[15], cmap='gray')
        axes[0,4].set_title("DSC Epoch 5")

    
        
    # Column 4: DSC Epoch 20 (Akhir)
    if 20 in dsc_imgs:
        axes[0,5].imshow(dsc_imgs[20], cmap='gray')
        axes[0,5].set_title("DSC Epoch 20 (Final)")

    # --- ROW 2: DIFFERENCE MAPS (ERROR) ---
    # Gunakan cmap 'inferno' atau 'hot'. 
    # vmin=0, vmax=0.2 artinya: Error > 0.2 akan sangat terang (putih/kuning)
    
    # Diff Baseline
    diff_base = np.abs(clean - baseline)
    im_b = axes[1,1].imshow(diff_base, cmap='inferno', vmin=0, vmax=0.15)
    axes[1,1].set_title("Diff: Baseline Error")
    plt.colorbar(im_b, ax=axes[1,1], fraction=0.046, pad=0.04)

    # Diff DSC Epoch 5
    if 5 in dsc_imgs:
        diff_ep5 = np.abs(clean - dsc_imgs[5])
        im_5 = axes[1,2].imshow(diff_ep5, cmap='inferno', vmin=0, vmax=0.15)
        axes[1,2].set_title("Diff: DSC Ep 5 Error")
        plt.colorbar(im_5, ax=axes[1,2], fraction=0.046, pad=0.04)

    # Diff DSC Epoch 10
    if 10 in dsc_imgs:
        diff_ep10 = np.abs(clean - dsc_imgs[10])
        im_10 = axes[1,3].imshow(diff_ep10, cmap='inferno', vmin=0, vmax=0.15)
        axes[1,3].set_title("Diff: DSC Ep 10 Error")
        plt.colorbar(im_10, ax=axes[1,3], fraction=0.046, pad=0.04)

    # Diff DSC Epoch 15
    if 15 in dsc_imgs:
        diff_ep15 = np.abs(clean - dsc_imgs[15])
        im_15 = axes[1,4].imshow(diff_ep15, cmap='inferno', vmin=0, vmax=0.15)
        axes[1,4].set_title("Diff: DSC Ep 15 Error")
        plt.colorbar(im_15, ax=axes[1,4], fraction=0.046, pad=0.04)

    # Diff DSC Epoch 20
    if 20 in dsc_imgs:
        diff_ep20 = np.abs(clean - dsc_imgs[20])
        im_20 = axes[1,5].imshow(diff_ep20, cmap='inferno', vmin=0, vmax=0.15)
        axes[1,5].set_title("Diff: DSC Ep 20 Error")
        plt.colorbar(im_20, ax=axes[1,5], fraction=0.046, pad=0.04)
        
    # Diff Noisy (Reference) - Taruh di kolom 1
    diff_noise = np.abs(clean - noisy)
    im_n = axes[1,0].imshow(diff_noise, cmap='inferno', vmin=0, vmax=0.15)
    axes[1,0].set_title("Diff: Input Noise")
    
    # Cleanup axes
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Analysis saved to {save_path}")

# --- CONTOH PENGGUNAAN (Ganti path sesuai file Anda) ---
if __name__ == "__main__":
    # Ganti path ini dengan path gambar hasil generate Anda
    # Misal Anda sudah save gambar hasil test ke folder 'results/'
    visualize_progression(
        clean_path="visual/visual_test75/clean.jpg",      # Ganti
        noisy_path="visual/visual_test25/noisy.webp",      # Ganti
        baseline_path="visual/visual_test25/baseline.webp",             # Ganti
        dsc_paths_dict={
            5: "visual/visual_test25/DSC5.webp",    # Ganti
            10: "visual/visual_test25/DSC10.webp",  # Ganti
            15: "visual/visual_test25/DSC15.webp",  # Ganti
            20: "visual/visual_test25/DSC20.webp"   # Ganti
        }
    )