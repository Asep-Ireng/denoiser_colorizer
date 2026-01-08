import torch
import time
from models.denoiser.network_unet import UNetRes
from models.modified.dsc_unet import UNetResDSC

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dummy_input = torch.randn(1, 2, 128, 128).to(device) # Batch 1, 128x128

# 1. Load Models
model_base = UNetRes(in_nc=2, out_nc=1).to(device)
model_dsc = UNetResDSC(in_nc=2, out_nc=1).to(device)

# 2. Count Params
params_base = sum(p.numel() for p in model_base.parameters())
params_dsc = sum(p.numel() for p in model_dsc.parameters())

print(f"Base Params: {params_base:,}")
print(f"DSC Params : {params_dsc:,}")
print(f"Reduction  : {100 * (1 - params_dsc/params_base):.2f}%")

# 3. Measure Inference Time
def measure_time(model, input_tensor, runs=100):
    model.eval()
    # Warmup
    for _ in range(10): 
        _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return (end - start) / runs * 1000 # returns ms

time_base = measure_time(model_base, dummy_input)
time_dsc = measure_time(model_dsc, dummy_input)

print(f"Base Inference: {time_base:.2f} ms")
print(f"DSC Inference : {time_dsc:.2f} ms")