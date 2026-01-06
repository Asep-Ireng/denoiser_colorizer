train_loss = [
    0.0353,  # epoch 4
    0.0349,  # epoch 5
    0.0348,  # epoch 6
    0.0345,  # epoch 7
    0.0345,  # epoch 8
    0.0342,  # epoch 9
    0.0340,  # epoch 10
    0.0340,  # epoch 11
    0.0336,  # epoch 12
    0.0335,  # epoch 13
    0.0337,  # epoch 14
    0.0336,  # epoch 15
    0.0334,  # epoch 16
    0.0334,  # epoch 17
    0.0335,  # epoch 18
    0.0333,  # epoch 19
    0.0333   # epoch 20
]

train_psnr = [
    25.53,
    25.63,
    25.62,
    25.71,
    25.70,
    25.77,
    25.82,
    25.79,
    25.89,
    25.89,
    25.84,
    25.84,
    25.93,
    25.91,
    25.89,
    25.94,
    25.94
]

val_loss = [
    0.0353,
    0.0346,
    0.0346,
    0.0344,
    0.0343,
    0.0336,
    0.0337,
    0.0337,
    0.0331,
    0.0333,
    0.0331,
    0.0335,
    0.0336,
    0.0333,
    0.0332,
    0.0333,
    0.0334
]

val_psnr = [
    25.53,
    25.63,
    25.64,
    25.74,
    25.73,
    25.91,
    25.85,
    25.88,
    25.98,
    25.92,
    25.98,
    25.92,
    25.91,
    25.93,
    25.94,
    25.96,
    25.96
]

epochs = list(range(4, 21))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) #

ax1.plot(epochs, train_psnr, label="Train PSNR")
ax1.plot(epochs, val_psnr, label="Val PSNR")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("PSNR (dB)")
ax1.legend()
ax1.grid(True)


ax2.plot(epochs, train_loss, label="Train Loss")
ax2.plot(epochs, val_loss, label="Val Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('psnr_loss_graph.png')

