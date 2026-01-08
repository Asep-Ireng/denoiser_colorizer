import torch
import torch.nn as nn
import numpy as np

# Handle both relative and absolute imports
try:
    import basicblock as B
except ImportError:
    import models.denoiser.basicblock as B

'''
# ====================
# unet
# ====================
'''


class UNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0

        
        return x


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
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


class ResUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='L', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.IMDBlock(nc[3], nc[3], bias=False, mode='C'+act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        x = x[..., :h, :w]

        return x














class UNetResSubP(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetResSubP, self).__init__()
        sf = 2
        self.m_ps_down = B.PixelUnShuffle(sf)
        self.m_ps_up = nn.PixelShuffle(sf)
        self.m_head = B.conv(in_nc*sf*sf, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], mode='C'+act_mode+'C') for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc*sf*sf, bias=False, mode='C')

    def forward(self, x0):
        x0_d = self.m_ps_down(x0)
        x1 = self.m_head(x0_d)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        x = self.m_ps_up(x) + x0

        return x


class UNetPlus(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetPlus, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode[1]))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode[1]))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode[1]))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[2], nc[2], mode='C'+act_mode[1]))
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[1], nc[1], mode='C'+act_mode[1]))
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[0], nc[0], mode='C'+act_mode[1]))

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x

'''
# ====================
# nonlocalunet
# ====================
'''

class NonLocalUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64,128,256,512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(NonLocalUNet, self).__init__()

        down_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')
        up_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')

        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(down_nonlocal, *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], up_nonlocal)
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x

# '''
# # ====================
# # Attention Gate Module
# # ====================
# '''
# class AttentionGate(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionGate, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# '''
# # ====================
# # Attention U-Net
# # ====================
# '''
# class AttentionUNet(nn.Module):
#     def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
#         super(AttentionUNet, self).__init__()

#         # --- ENCODER (Sama dengan UNet) ---
#         self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])
        
#         if downsample_mode == 'avgpool': downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool': downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv': downsample_block = B.downsample_strideconv
#         else: raise NotImplementedError(f'downsample mode [{downsample_mode}] is not found')

#         self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
#         self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
#         self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

#         self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

#         # --- ATTENTION GATES ---
#         self.att3 = AttentionGate(F_g=nc[3], F_l=nc[3], F_int=nc[3]//2)
#         self.att2 = AttentionGate(F_g=nc[2], F_l=nc[2], F_int=nc[2]//2)
#         self.att1 = AttentionGate(F_g=nc[1], F_l=nc[1], F_int=nc[1]//2)
#         self.att0 = AttentionGate(F_g=nc[0], F_l=nc[0], F_int=nc[0]//2)

#         # --- DECODER ---
#         if upsample_mode == 'upconv': upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle': upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose': upsample_block = B.upsample_convtranspose
#         else: raise NotImplementedError(f'upsample mode [{upsample_mode}] is not found')

#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

#         self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

#     def forward(self, x0):
#         x1 = self.m_head(x0)
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
        
#         x = self.m_body(x4)
        
#         # Attention + Upsample
#         x4_att = self.att3(g=x, x=x4)
#         x = self.m_up3(x + x4_att)
        
#         x3_att = self.att2(g=x, x=x3)
#         x = self.m_up2(x + x3_att)
        
#         x2_att = self.att1(g=x, x=x2)
#         x = self.m_up1(x + x2_att)
        
#         x1_att = self.att0(g=x, x=x1)
#         x = self.m_tail(x + x1_att) + x0
#         return x


# if __name__ == '__main__':
#     x = torch.rand(1,3,256,256)
# #    net = UNet(act_mode='BR')
#     net = NonLocalUNet()
#     net.eval()
#     with torch.no_grad():
#         y = net(x)
#     y.size()