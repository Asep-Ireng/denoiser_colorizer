"""
FPN-Enhanced UNetRes for Denoising
Adds Feature Pyramid Network with concat-based fusion to the baseline UNetRes.

Based on: DPIR DRUNet architecture
Modified: Added FPN blocks on 2 resolution levels (64, 128 channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from local copies
try:
    from . import basicblock as B
except ImportError:
    import basicblock as B


class FPNConcatBlock(nn.Module):
    """
    FPN block with concat-based fusion.
    
    Flow: lateral 1×1 → upsample top-down → concat → 3×3 fuse
    """
    
    def __init__(self, encoder_ch, top_down_ch, out_ch):
        super().__init__()
        self.lateral = nn.Conv2d(encoder_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(out_ch + top_down_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.lateral.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fuse.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, encoder_feat, top_down_feat):
        """
        Args:
            encoder_feat: Feature from encoder at this level
            top_down_feat: Feature from higher level (lower resolution)
        
        Returns:
            Fused feature at this level
        """
        # Lateral connection (1×1 conv to match channels)
        lateral = self.lateral(encoder_feat)
        
        # Upsample top-down to match spatial size
        top_up = F.interpolate(
            top_down_feat, 
            size=lateral.shape[2:], 
            mode='nearest'
        )
        
        # Concat and fuse with 3×3 conv
        concat = torch.cat([lateral, top_up], dim=1)
        fused = self.fuse(concat)
        
        return fused


class UNetResFPN(nn.Module):
    """
    UNetRes with FPN-enhanced skip connections.
    
    Architecture:
        - Encoder: Same as UNetRes (ResBlocks + downsampling)
        - FPN: 2-level concat-based fusion (64ch, 128ch)
        - Decoder: Same as UNetRes but uses FPN features
    
    Args:
        in_nc: Input channels (2 for grayscale + noise map)
        out_nc: Output channels (1 for grayscale)
        nc: Channel counts at each level [64, 128, 256, 512]
        nb: Number of ResBlocks per level
        act_mode: Activation mode ('R' for ReLU)
    """
    
    def __init__(self, in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, 
                 act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetResFPN, self).__init__()
        
        self.nc = nc
        
        # ============ ENCODER (same as UNetRes) ============
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
        
        # Downsample blocks
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError(f'downsample mode [{downsample_mode}] not found')
        
        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2')
        )
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2')
        )
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2')
        )
        
        # Body
        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        
        # ============ FPN BLOCKS (NEW) ============
        # Level 2: 128 channels, receives from 256 channels
        self.fpn2 = FPNConcatBlock(encoder_ch=nc[1], top_down_ch=nc[2], out_ch=nc[1])
        
        # Level 1: 64 channels, receives from 128 channels  
        self.fpn1 = FPNConcatBlock(encoder_ch=nc[0], top_down_ch=nc[1], out_ch=nc[0])
        
        # ============ DECODER (same as UNetRes) ============
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(f'upsample mode [{upsample_mode}] not found')
        
        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], bias=False, mode='2'),
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], bias=False, mode='2'),
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], bias=False, mode='2'),
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')
    
    def forward(self, x0):
        """
        Forward pass with FPN-enhanced skip connections.
        
        Args:
            x0: Input tensor [B, 2, H, W] (grayscale + noise level map)
        
        Returns:
            Denoised output [B, 1, H, W]
        """
        # ============ ENCODER ============
        x1 = self.m_head(x0)      # [B, 64, H, W]
        x2 = self.m_down1(x1)     # [B, 128, H/2, W/2]
        x3 = self.m_down2(x2)     # [B, 256, H/4, W/4]
        x4 = self.m_down3(x3)     # [B, 512, H/8, W/8]
        
        x = self.m_body(x4)       # [B, 512, H/8, W/8]
        
        # ============ FPN - build enhanced skip features ============
        # FPN fuses encoder features with higher-level context
        p2 = self.fpn2(x2, x3)    # FPN fusion: x2 (128ch) + x3 (256ch) → p2 (128ch)
        p1 = self.fpn1(x1, x2)    # FPN fusion: x1 (64ch) + x2 (128ch) → p1 (64ch)
        
        # ============ DECODER with FPN-enhanced skips ============
        # Level 3: Regular skip (no FPN at this level)
        x = self.m_up3(x + x4)    # [B, 256, H/4, W/4]
        
        # Level 2: Use FPN-enhanced skip instead of raw x3
        x = self.m_up2(x + x3)    # Keep baseline skip for stability
        # Blend in FPN features
        x = x + 1.0 * p2          # Moderate contribution from FPN
        
        # Level 1: Use FPN-enhanced skip 
        x = self.m_up1(x + x2)    # Keep baseline skip
        x = x + 0.6 * p1          # Moderate contribution from FPN
        
        # Tail
        x = self.m_tail(x + x1)
        
        return x
    
    def load_pretrained(self, state_dict, strict=False):
        """
        Load pretrained UNetRes weights.
        FPN layers will remain randomly initialized.
        
        Args:
            state_dict: Pretrained UNetRes state dict
            strict: If False, ignore missing/unexpected keys
        
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Filter out FPN keys from our model for comparison
        model_keys = set(self.state_dict().keys())
        fpn_keys = {k for k in model_keys if 'fpn' in k}
        
        # Load what we can
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        # FPN keys will be in missing - that's expected
        actual_missing = [k for k in missing if k not in fpn_keys]
        
        print(f"Loaded pretrained weights:")
        print(f"  - Matched: {len(state_dict) - len(unexpected)} layers")
        print(f"  - FPN layers (random init): {len(fpn_keys)}")
        if actual_missing:
            print(f"  - Actually missing: {actual_missing}")
        
        return missing, unexpected


def create_fpn_unet(pretrained_path=None):
    """
    Factory function to create UNetResFPN model.
    
    Args:
        pretrained_path: Path to pretrained UNetRes weights (optional)
    
    Returns:
        UNetResFPN model
    """
    model = UNetResFPN(
        in_nc=2,  # grayscale + noise map
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R'
    )
    
    if pretrained_path:
        import torch
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_pretrained(state_dict)
    
    return model


if __name__ == '__main__':
    # Quick test
    model = UNetResFPN(in_nc=2, out_nc=1)
    x = torch.randn(1, 2, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Count FPN parameters
    fpn_params = sum(p.numel() for n, p in model.named_parameters() if 'fpn' in n)
    print(f"FPN parameters: {fpn_params:,}")
