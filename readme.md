# Denoiser & Colorizer - U-Net Modification Study

> Comparative analysis of U-Net architectural modifications for image denoising and color restoration.

**Team 3 - Deep Learning Project**

| Name                      | NIM       | Task                          | Status  |
| ------------------------- | --------- | ----------------------------- | ------- |
| Rui Krisna                | C14230277 | Feature Pyramid Network (FPN) | ✅ Done |
| Bryan Alexander Limanto   | C14230114 | Residual Blocks               | ✅ Done |
| Satrio Adi Rinekso        | C14230112 | Controlled Cross-Feedback     | ✅ Done |
| Reynard Sebastian Hartono | C14230155 | Depthwise Separable Conv      | ✅ Done |
| Juan Matthew Davidson     | C14230124 | Attention Gates               | ⬜ TODO |

**Advisor:** Liliana, S.T., M.Eng., Ph.D.

---

## 📦 Resources

- **Code:** [GitHub Repository](https://github.com/Asep-Ireng/denoiser_colorizer)
- **Weights:** [Hugging Face Models](https://huggingface.co/Asep-Ireng/Denoiser_Colorizer)
- **Dataset:** COCO Animals (15,000 images filtered from COCO 2017)

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Asep-Ireng/denoiser_colorizer.git
cd denoiser_colorizer
```

### 2. Download Weights

```bash
pip install huggingface_hub
huggingface-cli download Asep-Ireng/Denoiser_Colorizer --local-dir weights/
```

### 3. Install Dependencies

```bash
pip install torch torchvision gradio numpy pillow tqdm timm
```

### 4. Run Gradio App

```bash
python app.py
```

---

## 📊 Results

### FPN Modification (by Rui Krisna)

| Model             | σ=15      | σ=25      | σ=50      | Average   |
| ----------------- | --------- | --------- | --------- | --------- |
| Baseline (DRUNet) | 32.80     | 30.07     | 25.92     | 29.60     |
| FPN Arch-Only     | 19.47     | 18.96     | 17.77     | 18.73     |
| **FPN Trained**   | **32.99** | **30.56** | **27.45** | **30.33** |

**Key Finding:** FPN-enhanced denoiser achieves **+0.73 dB PSNR improvement** over baseline, with best gains at high noise (+1.53 dB at σ=50).


### Residual Blocks Modification (by Bryan Alexander Limanto)


| Model Name        |  σ=15  |  σ=25  |  σ=50  | Average   |
| ----------------- | -------| -------| -------| --------- |
| Plain U-Net       | 30.31  | 28.06  | 24.91  | 27.76     |
| Residual U-Net    | 31.62  | 29.18  | 26.11  | 28.97     |
| DRUNet (Baseline) | 31.86  | 29.16  | 25.12  | 28.71 	   |

**Key Finding:** The Residual U-Net outperforms the DRUNet baseline with **a +0.26 dB average improvement**. Most notably, it demonstrates superior robustness at high noise levels, achieving a significant **+0.99 dB gain at σ=50** compared to the baseline.


### DSC Modification (by Reynard Sebastian Hartono)

* Base Params: 32,638,656
* DSC Params : 4,952,768
* Parameters Reduction  : 84.83%
* Base Inference: 236.17 ms
* DSC Inference : 68.73 ms

| Model             | σ=15      | σ=25      | σ=50      | Average   |
| ----------------- | --------- | --------- | --------- | --------- |
| Baseline          | 32.55     | 29.85     | 25.80     | 29.40     |
| DSC               | 31.88     | 29.47     | 26.46     | 29.27     |

**Key Finding:** The DSC version of the denoiser achieves less PNSR score (with the average being -0.13 dB PSNR reduction). This is expected considering the reduction of parameters (which in turn causes less inference time), which causes this reduction to be insignificant and can be considered as a worthwile trade-off.


###Controled Cross-Field Feedback Modification (by Satrio Adi Rinekso)

Total Parameters:  16,390,404

Inference Speed: 946.49 ms ms Tested on CPU

| Model             | σ=15      | σ=25      | σ=50      | Average   |
| ----------------- | --------- | --------- | --------- | --------- |
| Baseline          | 15.03      |  14.35     | 12.35      | 13.87      |
| Controled Cross-Field               | 23.46     | 23.43     | 16.39     | 21.09     |

**Key Finding:** The Controlled Cross-Field Feedback mechanism implements a dual-task architecture that simultaneously performs image denoising and colorization, a far more complex challenge than the single-task methods used in other modifications. By utilizing semantic information from the colorization task to guide the denoising process, the model achieves a significant average improvement of 21.09 dB (+7.22 dB gain) over the baseline. 

## 🏗️ Architecture

### Base Model

- **Denoiser:** DRUNet (UNetRes) - grayscale denoising
- **Colorizer:** DDColor - transformer-based colorization

### Modifications Being Studied

1. **Residual Blocks** - Improve gradient flow
2. **Attention Gates** - Focus on relevant regions
3. **Cross-Feedback** - Share features between denoiser/colorizer
4. **Feature Pyramid Network (FPN)** ✅ - Multi-scale feature fusion
5. **Depthwise Separable Conv** - Reduce parameters

---

## 📁 Project Structure

```
denoiser_colorizer/
├── app.py                  # Gradio UI
├── train_fpn.py            # FPN training script
├── evaluate.py             # PSNR evaluation
├── models/
│   ├── denoiser/           # Baseline DRUNet
│   ├── colorizer/          # DDColor
│   └── modified/           # FPN-enhanced UNet
├── weights/                # Model weights (download from HF)
└── utils/                  # Dataset utilities
```

---

## 👥 Team Checklist

### For Each Modification:

- [ ] **Create model file** in `models/modified/`
- [ ] **Add training script** (e.g., `train_xxx.py`)
- [ ] **Run evaluation** using `evaluate.py`
- [ ] **Update this README** with results table
- [ ] **Commit and push** to GitHub
- [ ] **Upload weights** to Hugging Face

### How to Add Your Modification:

1. Copy baseline model to `models/modified/`
2. Implement your modification
3. Train using your training script
4. Run `python evaluate.py --num_images 100`
5. Add your results to this README
6. Push code and weights

---

## 📄 References

- **DRUNet:** [DPIR Paper](https://github.com/cszn/DPIR)
- **DDColor:** [DDColor Paper](https://github.com/piddnad/DDColor)
- **COCO Dataset:** [COCO 2017](https://cocodataset.org/)
- **FPN:** [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

---

## 📝 License

MIT License
