import torch
import torch.nn as nn

# --- 1. RESIDUAL BLOCK (Ciri Khas DRUNet) ---
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(c), 
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(c)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x) + x) 

# --- 2. CONTROLLED FEEDBACK (Gate Mechanism) ---
class ControlledCrossFeedback(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(c, c, 3, padding=1)
        
    def forward(self, d, c_feat): 
        # Fitur dari cabang lain difilter dulu oleh 'gate'
        return self.fuse(d + (c_feat.detach() * self.gate(c_feat.detach())))

# --- 3. ARSITEKTUR UTAMA (DualTaskDRUNet) ---
class DualTaskDRUNet(nn.Module): 
    def __init__(self):
        super().__init__()
        
        def layer(i, o): 
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), 
                nn.BatchNorm2d(o), 
                nn.ReLU(), 
                ResBlock(o)
            )
        
        # Encoder
        self.enc1 = layer(1, 64); self.pool = nn.MaxPool2d(2)
        self.enc2 = layer(64, 128); self.enc3 = layer(128, 256); self.bot = layer(256, 512)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Decoder Color
        self.dc1 = layer(768, 256); self.dc2 = layer(384, 128); self.dc3 = layer(192, 64)
        self.hc = nn.Conv2d(64, 3, 1)
        
        # Cross Feedback
        self.cf1 = ControlledCrossFeedback(256); self.cf2 = ControlledCrossFeedback(128)
        
        # Decoder Denoise
        self.dd1 = layer(768, 256); self.dd2 = layer(384, 128); self.dd3 = layer(192, 64)
        self.hd = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); b = self.bot(self.pool(e3))
        
        c1 = self.dc1(torch.cat([self.up(b), e3], 1))
        c2 = self.dc2(torch.cat([self.up(c1), e2], 1))
        c3 = self.dc3(torch.cat([self.up(c2), e1], 1))
        
        d1 = self.cf1(self.dd1(torch.cat([self.up(b), e3], 1)), c1)
        d2 = self.cf2(self.dd2(torch.cat([self.up(d1), e2], 1)), c2)
        d3 = self.dd3(torch.cat([self.up(d2), e1], 1))
        
        return self.hd(d3), self.hc(c3)