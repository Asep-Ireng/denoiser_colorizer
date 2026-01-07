import torch
import torch.nn as nn
from collections import OrderedDict

try:
    from . import basicblock as B
except ImportError:
    import basicblock as B

# 1. Attention gates
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Jumlah channel dari Gating Signal (Decoder)
            F_l: Jumlah channel dari Skip Connection (Encoder)
            F_int: Jumlah channel intermediate (biasanya setengah dari F_g)
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Output 0 s.d 1 (Masker)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi # Fitur Encoder dikali Masker

# 2. Arsitektur Attention DRUNet
class AttentionUNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        """
        Args:
            nb: Jumlah Residual Block per level (Default 4 untuk DRUNet)
            nc: List jumlah channel [Level1, Level2, Level3, Level4]
        """
        super(AttentionUNetRes, self).__init__()

        # ENCODER
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # Downsample Block Selection
        if downsample_mode == 'avgpool': downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool': downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv': downsample_block = B.downsample_strideconv
        else: raise NotImplementedError(f'downsample mode [{downsample_mode}] is not found')

        # Level 1 Down (Menggunakan ResBlock)
        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[0], nc[1], bias=False, mode='2')
        )
        # Level 2 Down
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[1], nc[2], bias=False, mode='2')
        )
        # Level 3 Down
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[2], nc[3], bias=False, mode='2')
        )

        # BODY (Bottle Neck)
        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )

        # ATTENTION GATES
        # Gate dipasang untuk memfilter fitur dari encoder sebelum masuk decoder
        self.att3 = AttentionGate(F_g=nc[3], F_l=nc[3], F_int=nc[3]//2) # Untuk Level 3 (512 ch)
        self.att2 = AttentionGate(F_g=nc[2], F_l=nc[2], F_int=nc[2]//2) # Untuk Level 2 (256 ch)
        self.att1 = AttentionGate(F_g=nc[1], F_l=nc[1], F_int=nc[1]//2) # Untuk Level 1 (128 ch)
        self.att0 = AttentionGate(F_g=nc[0], F_l=nc[0], F_int=nc[0]//2) # Untuk Level 0 (64 ch - Tail)

        # DECODER
        # Upsample Block Selection
        if upsample_mode == 'upconv': upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle': upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose': upsample_block = B.upsample_convtranspose
        else: raise NotImplementedError(f'upsample mode [{upsample_mode}] is not found')

        # Level 3 Up
        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], bias=False, mode='2'), 
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        # Level 2 Up
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], bias=False, mode='2'), 
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        # Level 1 Up
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], bias=False, mode='2'), 
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )

        # Output Tail
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        # 1. Encoder Flow (Sama seperti DRUNet Biasa)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        
        # 2. Body Flow
        x = self.m_body(x4)
        
        # 3. Decoder Flow dengan ATTENTION
        
        # Level 3
        # x  : Fitur dari Body (Decoder context)
        # x4 : Fitur dari Encoder
        x4_filtered = self.att3(g=x, x=x4) # Filter x4 dulu
        x = self.m_up3(x + x4_filtered) # Jumlahkan x + x4_filtered
        
        # Level 2
        x3_filtered = self.att2(g=x, x=x3)
        x = self.m_up2(x + x3_filtered)
        
        # Level 1
        x2_filtered = self.att1(g=x, x=x2)
        x = self.m_up1(x + x2_filtered)
        
        # Tail / Output
        x1_filtered = self.att0(g=x, x=x1)
        x = self.m_tail(x + x1_filtered)
        
        # Ambil gambar noisy asli (Channel ke-0 saja)
        noisy_image = x0[:, 0:1, :, :]


        return x + noisy_image

if __name__ == "__main__":
    # Test Code untuk memastikan tidak ada error dimensi
    model = AttentionUNetRes(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4)
    print("Model AttentionUNetRes berhasil dibuat.")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")
    
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")