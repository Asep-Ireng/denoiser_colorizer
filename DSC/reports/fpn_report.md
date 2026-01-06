# Feature Pyramid Network (FPN) Modification Report

**Author:** Rui Krisna (C14230277)  
**Modification:** Feature Pyramid Network (FPN) for Multi-scale Feature Fusion  
**Status:** ✅ Complete

---

## 1. Introduction

### Objective

Integrate Feature Pyramid Network (FPN) into the DRUNet denoiser to improve detail preservation through multi-scale feature fusion.

### Motivation

The baseline DRUNet uses simple skip connections that add encoder features directly to decoder features. This can lead to:

- Loss of fine details during upsampling
- Insufficient multi-scale context integration
- Over-smoothing of textures (e.g., animal fur)

FPN addresses these issues by fusing features from multiple resolution levels before passing them to the decoder.

---

## 2. Implementation Details

### 2.1 Architecture

**Base Model:** UNetRes (DRUNet) with 4 resolution levels [64, 128, 256, 512 channels]

**FPN Integration:** 2-level concat-based fusion at 64ch and 128ch levels

```
Encoder:  x1(64) → x2(128) → x3(256) → x4(512)
                      ↓         ↓
FPN:              p2 = FPN(x2, x3)
                      ↓
                  p1 = FPN(x1, x2)
                      ↓
Decoder:  Uses p1, p2 as enhanced skip connections
```

### 2.2 FPN Block Design

```python
class FPNConcatBlock(nn.Module):
    def __init__(self, encoder_ch, top_down_ch, out_ch):
        self.lateral = nn.Conv2d(encoder_ch, out_ch, kernel_size=1)
        self.fuse = nn.Conv2d(out_ch + top_down_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, encoder_feat, top_down_feat):
        lateral = self.lateral(encoder_feat)
        top_up = F.interpolate(top_down_feat, size=lateral.shape[2:], mode='nearest')
        concat = torch.cat([lateral, top_up], dim=1)
        return self.fuse(concat)
```

**Key Design Choices:**

- **Concat-based fusion** (not addition) to preserve all information
- **3×3 fusion conv** for spatial coherence after concatenation
- **0.6 blending weight** for FPN features in decoder

### 2.3 Modified Forward Pass

```python
def forward(self, x0):
    # Encoder (same as baseline)
    x1 = self.m_head(x0)       # [B, 64, H, W]
    x2 = self.m_down1(x1)      # [B, 128, H/2, W/2]
    x3 = self.m_down2(x2)      # [B, 256, H/4, W/4]
    x4 = self.m_down3(x3)      # [B, 512, H/8, W/8]
    x = self.m_body(x4)

    # FPN - build enhanced skip features
    p2 = self.fpn2(x2, x3)     # 128ch + 256ch context → 128ch
    p1 = self.fpn1(x1, x2)     # 64ch + 128ch context → 64ch

    # Decoder with FPN-enhanced skips
    x = self.m_up3(x + x4)
    x = self.m_up2(x + x3) + 0.6 * p2   # Blend FPN features
    x = self.m_up1(x + x2) + 0.6 * p1
    x = self.m_tail(x + x1)

    return x
```

### 2.4 Training Strategy

**Two-Phase Fine-Tuning:**

| Phase   | Epochs | Layers Trained             | Learning Rate |
| ------- | ------ | -------------------------- | ------------- |
| Phase 1 | 10     | FPN only (freeze backbone) | 1e-4          |
| Phase 2 | 20     | All layers                 | 1e-5          |

**Loss Function:** L1 Loss (Mean Absolute Error)

**Dataset:** COCO Animals (12,000 train / 3,000 test)

---

## 3. Experimental Results

### 3.1 PSNR Comparison (100 test images)

| Model                  | σ=15      | σ=25      | σ=50      | Average      |
| ---------------------- | --------- | --------- | --------- | ------------ |
| Baseline (DRUNet)      | 32.80     | 30.07     | 25.92     | 29.60 dB     |
| FPN Arch-Only (random) | 19.47     | 18.96     | 17.77     | 18.73 dB     |
| **FPN Trained**        | **32.99** | **30.56** | **27.45** | **30.33 dB** |

### 3.2 Key Findings

1. **FPN Trained outperforms Baseline by +0.73 dB** on average
2. **Best improvement at high noise (σ=50):** +1.53 dB
3. **FPN Architecture-only has low PSNR** but interesting visual properties

### 3.3 Interesting Observation: Architecture-Only Effect

During development, we discovered that the FPN architecture with **random (untrained) weights** produced outputs with more perceived texture/detail, despite having much lower PSNR.

**Explanation:**

- Random FPN weights add high-frequency patterns that look like texture
- Training with L1 loss optimizes for pixel-accurate reconstruction
- L1 loss penalizes these "fake textures" as incorrect
- Trained model learns to minimize error → smoother but higher PSNR

**Insight:** The architecture provides capability, but the loss function determines behavior.

---

## 4. Files Created

| File                            | Description                      |
| ------------------------------- | -------------------------------- |
| `models/modified/fpn_unet.py`   | UNetResFPN model with FPN blocks |
| `models/modified/basicblock.py` | Copied building blocks           |
| `train_fpn.py`                  | Two-phase training script        |
| `evaluate.py`                   | PSNR evaluation script           |
| `weights/fpn/fpn_arch_only.pth` | Architecture-only weights        |
| `weights/fpn/fpn_best.pth`      | Fully trained weights            |

---

## 5. Parameter Count

| Component              | Parameters |
| ---------------------- | ---------- |
| Baseline (UNetRes)     | 32.6M      |
| FPN Blocks (added)     | 573K       |
| **Total (UNetResFPN)** | **33.2M**  |

FPN adds only **1.7%** additional parameters.

---

## 6. Conclusion

The FPN modification successfully improves denoising performance:

- **+0.73 dB average PSNR improvement** over baseline
- **+1.53 dB at high noise levels** (σ=50)
- **Minimal parameter overhead** (+1.7%)

The multi-scale feature fusion enabled by FPN helps the model better preserve structure, especially in challenging high-noise scenarios.

### Future Work

- Explore perceptual loss functions for better texture preservation
- Investigate learnable blending weights for FPN features
- Combine FPN with other team modifications (Attention Gates, etc.)

---

## 7. References

1. Lin, T. Y., et al. "Feature Pyramid Networks for Object Detection." CVPR 2017.
2. Zhang, K., et al. "Plug-and-Play Image Restoration with Deep Denoiser Prior." TPAMI 2021.
3. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
