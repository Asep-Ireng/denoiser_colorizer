"""
Residual U-Net for Denoising
Standard U-Net architecture replaced with Residual Blocks for better gradient flow.

Architecture:
    - Encoder: Residual Blocks + MaxPool
    - Bottleneck: Deep Residual Block
    - Decoder: TransposeConv + Concat + Residual Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block dengan PRE-ACTIVATION (Sesuai Proposal)
    Structure: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> (+ Input)
    """
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        # Pre-activation 1: BN & ReLU didefinisikan untuk input
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Pre-activation 2
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Shortcut connection (1x1 Projection jika dimensi beda)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1x1 projection pada shortcut
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, x):
        residual = x
        # Flow Pre-activation: BN -> ReLU -> Conv
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # Penjumlahan (Tanpa ReLU di akhir untuk Pre-activation murni)
        out += self.shortcut(residual)
        
        return out


class ResUNet(nn.Module):
    """
    Residual U-Net Architecture.
    
    Args:
        in_nc: Input channels (2 for grayscale + noise map)
        out_nc: Output channels (1 for grayscale)
    """
    
    def __init__(self, in_nc=2, out_nc=1):
        super(ResUNet, self).__init__()
        
        # ============ ENCODER ============
        # Level 1: Input -> 64
        self.enc1 = ResidualBlock(in_nc, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        # Level 2: 64 -> 128
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Level 3: 128 -> 256
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # ============ BOTTLENECK ============
        # Center: 256 -> 512
        self.bottleneck = ResidualBlock(256, 512)
        
        # ============ DECODER ============
        # Level 3 Up: 512 -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256) # 256 (from up) + 256 (from skip) -> 512 input internally handled
        
        # Level 2 Up: 256 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        
        # Level 1 Up: 128 -> 64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        
        # ============ TAIL ============
        self.final = nn.Conv2d(64, out_nc, kernel_size=1)
        
        # Adjust input channels for decoder blocks (because of concatenation)
        # We redefine them to accept double channels (Skip + Up)
        self.dec3 = ResidualBlock(512, 256) # Input 256+256=512, Output 256
        self.dec2 = ResidualBlock(256, 128) # Input 128+128=256, Output 128
        self.dec1 = ResidualBlock(128, 64)  # Input 64+64=128, Output 64

    def forward(self, x):
        """
        Forward pass with auto-resize for safe concatenation.
        """
        # Encoder
        e1 = self.enc1(x)       # [B, 64, H, W]
        p1 = self.pool1(e1)     # [B, 64, H/2, W/2]
        
        e2 = self.enc2(p1)      # [B, 128, H/2, W/2]
        p2 = self.pool2(e2)     # [B, 128, H/4, W/4]
        
        e3 = self.enc3(p2)      # [B, 256, H/4, W/4]
        p3 = self.pool3(e3)     # [B, 256, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(p3) # [B, 512, H/8, W/8]
        
        # Decoder 3
        d3 = self.up3(b)
        # Auto-resize if dimensions don't match (e.g., odd input size)
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((e3, d3), dim=1) # Concat: 256 + 256 = 512
        d3 = self.dec3(d3)              # Reduce to 256
        
        # Decoder 2
        d2 = self.up2(d3)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((e2, d2), dim=1) # Concat: 128 + 128 = 256
        d2 = self.dec2(d2)              # Reduce to 128
        
        # Decoder 1
        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((e1, d1), dim=1) # Concat: 64 + 64 = 128
        d1 = self.dec1(d1)              # Reduce to 64
        
        return self.final(d1)
    
    def load_pretrained(self, state_dict, strict=False):
        """
        Load pretrained weights safely.
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        
        print(f"Loaded pretrained weights:")
        print(f"  - Matched layers: {len(state_dict) - len(unexpected)}")
        if missing:
            print(f"  - Missing layers: {len(missing)}")
        if unexpected:
            print(f"  - Unexpected layers: {len(unexpected)}")
            
        return missing, unexpected


def create_model(pretrained_path=None):
    """
    Factory function to create ResUNet model.
    """
    model = ResUNet(in_nc=2, out_nc=1)
    
    if pretrained_path:
        import torch
        # Load weights on CPU to avoid CUDA errors if GPU not available
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Handle case where weights are inside 'model_state_dict' key
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_pretrained(state_dict)
    
    return model


if __name__ == '__main__':
    # Quick Test
    model = create_model()
    # Test with odd dimensions to check auto-resize
    x = torch.randn(1, 2, 255, 255) 
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")