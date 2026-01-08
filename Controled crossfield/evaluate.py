import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.drunet_crossfield import DualTaskUNet
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
