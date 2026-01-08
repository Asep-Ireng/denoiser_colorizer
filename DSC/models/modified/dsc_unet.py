"""
DSC-UNetRes for Denoising (Reynard's Task)
Base Architecture: UNetRes
Modification: Replaced Standard ResBlocks with Depthwise Separable Convolution ResBlocks
"""

import torch
import torch.nn as nn
import models.denoiser.basicblock as B  # Menggunakan basicblock dari base

# =========================================================================
# 1. Definisikan Depthwise Separable Convolution Block
# =========================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Terdiri dari:
    1. Depthwise: Conv2d dengan groups=in_channels (spatial filtering)
    2. Pointwise: Conv2d 1x1 (channel mixing)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_ch, # Kunci dari Depthwise: groups = input channels
            bias=bias
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(
            in_ch, out_ch, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# =========================================================================
# 2. Definisikan Residual Block dengan DSC
# =========================================================================

class DSCResBlock(nn.Module):
    """
    Residual Block yang menggunakan Depthwise Separable Conv alih-alih Conv standar.
    Struktur: x + DSC(ReLU(DSC(x)))
    """
    def __init__(self, in_channels=64, out_channels=64, bias=True, act_mode='R'):
        super().__init__()
        assert in_channels == out_channels, 'Input dan Output channels harus sama untuk ResBlock'

        # Activation
        if act_mode == 'R':
            self.act = nn.ReLU(inplace=True)
        elif act_mode == 'L':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

        # Kita mengganti Conv standar B.conv dengan DepthwiseSeparableConv buatan kita
        self.dsc1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.dsc2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        identity = x
        
        # Layer 1
        out = self.dsc1(x)
        out = self.act(out)
        
        # Layer 2
        out = self.dsc2(out)
        
        # Residual Connection
        return identity + out

# =========================================================================
# 3. Arsitektur Utama (UNetRes dengan DSC)
# =========================================================================

class UNetResDSC(nn.Module):
    """
    Arsitektur Base UNetRes tetapi menggunakan DSCResBlock.
    Tidak ada FPN. Struktur persis seperti base model.
    """
    def __init__(self, in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, 
                 act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetResDSC, self).__init__()

        # Head (Conv standar biasanya tetap digunakan di awal untuk ekstraksi fitur awal yang kaya)
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # Helper untuk downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError(f'downsample mode [{downsample_mode}] not found')

        # Helper untuk upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(f'upsample mode [{upsample_mode}] not found')

        # --- ENCODER ---
        # Perhatikan: Kita mengganti B.ResBlock dengan DSCResBlock
        self.m_down1 = B.sequential(
            *[DSCResBlock(nc[0], nc[0], bias=False, act_mode=act_mode) for _ in range(nb)], 
            downsample_block(nc[0], nc[1], bias=False, mode='2')
        )
        self.m_down2 = B.sequential(
            *[DSCResBlock(nc[1], nc[1], bias=False, act_mode=act_mode) for _ in range(nb)], 
            downsample_block(nc[1], nc[2], bias=False, mode='2')
        )
        self.m_down3 = B.sequential(
            *[DSCResBlock(nc[2], nc[2], bias=False, act_mode=act_mode) for _ in range(nb)], 
            downsample_block(nc[2], nc[3], bias=False, mode='2')
        )

        # --- BODY ---
        self.m_body  = B.sequential(
            *[DSCResBlock(nc[3], nc[3], bias=False, act_mode=act_mode) for _ in range(nb)]
        )

        # --- DECODER ---
        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], bias=False, mode='2'), 
            *[DSCResBlock(nc[2], nc[2], bias=False, act_mode=act_mode) for _ in range(nb)]
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], bias=False, mode='2'), 
            *[DSCResBlock(nc[1], nc[1], bias=False, act_mode=act_mode) for _ in range(nb)]
        )
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], bias=False, mode='2'), 
            *[DSCResBlock(nc[0], nc[0], bias=False, act_mode=act_mode) for _ in range(nb)]
        )

        # Tail
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        # Forward pass standar UNet (tanpa FPN)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        
        x = self.m_body(x4)
        
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        
        x = self.m_tail(x+x1)

        return x

    def load_pretrained_base(self, state_dict):
        """
        Custom loader jika ingin memuat bobot dari DRUNet standar.
        NOTE: Bobot Conv standar TIDAK AKAN COCOK langsung dengan DSC.
        Fungsi ini hanya akan memuat layer yang tidak diubah (seperti head/tail/upsample).
        Layer DSCResBlock akan tetap random init.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # Backward compatibility for serialized parameters
                    param = param.data
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    # print(f"Loaded: {name}")
                else:
                    print(f"Skipped (shape mismatch - expected for DSC layers): {name}")
            else:
                pass
                # print(f"Skipped (missing): {name}")

if __name__ == '__main__':
    # Test Output
    model = UNetResDSC()
    print(f"DSC Model Created. Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compare with standard ResBlock param count to prove reduction
    # Standard 3x3 conv (64->64) params: 64*64*3*3 = 36,864
    # DSC 3x3 conv (64->64) params: (64*1*3*3) + (64*64*1*1) = 576 + 4096 = 4,672
    # Massive reduction!