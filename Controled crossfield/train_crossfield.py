import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import math

# --- IMPORT MODEL DARI FOLDER MODELS (PERBAIKAN DI SINI) ---
from models.drunet_crossfield import DualTaskDRUNet

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
ZIP_FILENAME = "data_filtered.zip"
EXTRACT_FOLDER = "./dataset_temp"
SAVE_MODEL_NAME = "model_final.pth"

print(f"🖥️  Running on: {DEVICE}")

# --- DATA PREPARATION ---
def prepare_data():
    if not os.path.exists(EXTRACT_FOLDER):
        if not os.path.exists(ZIP_FILENAME):
            print(f"❌ File {ZIP_FILENAME} tidak ditemukan!")
            exit()
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)

class AutoDataset(Dataset):
    def __init__(self, root_dir):
        def find_train(path):
            for root, dirs, _ in os.walk(path):
                if 'train' in dirs: return os.path.join(root, 'train')
            return path
        real_path = find_train(root_dir)
        self.files = sorted(glob.glob(os.path.join(real_path, '**', '*.*'), recursive=True))
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try: img = Image.open(self.files[idx]).convert('RGB')
        except: img = Image.new('RGB', (256, 256))
        
        tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
        rgb = tf(img)
        gray = transforms.Grayscale(1)(rgb)
        noise = torch.randn_like(gray) * 0.1
        noisy = torch.clamp(gray + noise, 0, 1)
        return noisy, gray, rgb

# --- MAIN TRAINING LOOP ---
def main():
    prepare_data()
    ds = AutoDataset(EXTRACT_FOLDER)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0) # num_workers=0 agar aman di Windows
    
    # Inisialisasi Model
    model = DualTaskDRUNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    print(f"🚀 Mulai Training {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dl, desc=f"Ep {epoch+1}")
        total_loss = 0
        
        for n, g, c in loop:
            n, g, c = n.to(DEVICE), g.to(DEVICE), c.to(DEVICE)
            opt.zero_grad()
            pd, pc = model(n)
            loss = loss_fn(pd, g) + 0.5 * loss_fn(pc, c)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"✅ Epoch {epoch+1} Selesai. Avg Loss: {total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_NAME)
    print(f"💾 Model disimpan: {SAVE_MODEL_NAME}")

if __name__ == "__main__":
    main()