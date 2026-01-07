"""
Plain/Standard U-Net (Baseline)
Untuk perbandingan (Ablation Study). Tidak menggunakan Residual Connection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainBlock(nn.Module):
    """
    Standard Block (Tanpa Residual)
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(PlainBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Tidak ada variabel 'residual' disini
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) # Output langsung
        
        return out

class PlainUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1):
        super(PlainUNet, self).__init__()
        
        # Encoder (Menggunakan PlainBlock)
        self.enc1 = PlainBlock(in_nc, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = PlainBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = PlainBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = PlainBlock(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = PlainBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = PlainBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = PlainBlock(128, 64)
        
        self.final = nn.Conv2d(64, out_nc, kernel_size=1)

    def forward(self, x):
        # Logika forward sama persis, hanya beda di blok internalnya
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        d3 = self.up3(b)
        if d3.size() != e3.size(): d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.size() != e2.size(): d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.size() != e1.size(): d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)