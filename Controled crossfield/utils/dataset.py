import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, mode='train', limit_train=None, limit_test=None):
        """
        root_dir: Folder utama (misal: .../data_filtered)
        mode: 'train' atau 'test'
        """
        self.root_dir = root_dir
        self.mode = mode
        
        # 1. CEK STRUKTUR FOLDER
        # Coba cari apakah ada subfolder 'train' atau 'test' spesifik
        specific_folder = os.path.join(root_dir, mode) # misal: .../data_filtered/train
        
        if os.path.exists(specific_folder):
            # Jika folder 'train'/'test' ADA, gunakan folder itu saja
            search_dir = specific_folder
            use_split_logic = False # Tidak perlu split manual lagi
            print(f"[{mode.upper()}] Menggunakan folder khusus: {search_dir}")
        else:
            # Jika TIDAK ADA, scan folder induk (mode campuran/berantakan)
            search_dir = root_dir
            use_split_logic = True
            print(f"[{mode.upper()}] Folder spesifik tidak ditemukan, scan total di: {search_dir}")

        # 2. SCAN GAMBAR
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_paths = []
        
        for ext in extensions:
            # recursive=True mencari sampai ke dalam sub-sub folder
            found = glob.glob(os.path.join(search_dir, '**', ext), recursive=True)
            self.image_paths.extend(found)
        
        # Hapus duplikat dan acak
        self.image_paths = sorted(list(set(self.image_paths)))
        np.random.seed(42) 
        np.random.shuffle(self.image_paths)
        
        # 3. LOGIKA SPLIT (Hanya jika folder train/test tidak terpisah)
        if use_split_logic:
            total = len(self.image_paths)
            limit_train_count = 12000 if limit_train is None else limit_train
            limit_test_count = 3000 if limit_test is None else limit_test
            
            if mode == 'train':
                end = min(limit_train_count, total)
                self.image_paths = self.image_paths[:end]
            elif mode == 'test':
                start = limit_train_count
                end = min(limit_train_count + limit_test_count, total)
                self.image_paths = self.image_paths[start:end] if start < total else []

        print(f"   Total gambar dimuat: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Ambil nama folder induk sebagai label
        # Misal: .../train/kucing/gambar.jpg -> Label: kucing
        # Misal: .../train/gambar.jpg -> Label: train (jika tidak ada subfolder hewan)
        parent_folder = os.path.basename(os.path.dirname(img_path))
        label_name = parent_folder

        try:
            image_rgb = Image.open(img_path).convert('RGB')
        except:
            image_rgb = Image.new('RGB', (256, 256))

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        target_rgb = transform(image_rgb)
        target_gray = transforms.Grayscale(num_output_channels=1)(target_rgb)
        
        noise = torch.randn_like(target_gray) * 0.1 
        input_noisy = torch.clamp(target_gray + noise, 0., 1.)
        
        return {
            'input': input_noisy, 
            'target_gray': target_gray, 
            'target_rgb': target_rgb, 
            'label': label_name
        }