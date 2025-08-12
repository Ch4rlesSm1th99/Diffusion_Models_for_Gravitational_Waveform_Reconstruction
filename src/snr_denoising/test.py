import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from models import UNet1D, CustomDiffusion
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r"C:\Users\charl\PycharmProjects\snr_denoising\data\safe_data\dataset.h5"
MODEL_PATH = r"C:\Users\charl\PycharmProjects\snr_denoising\model\safe_model\model_diffusion.pth"
BATCH_SIZE = 1
DEVICE = torch.device('cpu')
print(f" DATA_PATH = {DATA_PATH}")
print(f" MODEL_PATH = {MODEL_PATH}")
print(f" BATCH_SIZE = {BATCH_SIZE}\n and device: {DEVICE}")



class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5     = h5py.File(h5_path, 'r')
        self.signals= self.h5['signal']
        self.noisys = self.h5['noisy']
        self.snrs   = self.h5['snr']

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        clean_raw  = torch.from_numpy(self.signals[idx]).unsqueeze(0).float()
        noisy_raw  = torch.from_numpy(self.noisys[idx]).unsqueeze(0).float()
        sigma      = noisy_raw.std() if noisy_raw.std()>0 else torch.tensor(1.0)
        clean_norm = clean_raw / sigma
        return clean_norm, sigma, clean_raw, noisy_raw, float(self.snrs[idx])

    def __del__(self):
        self.h5.close()

ds     = InferenceDataset(DATA_PATH)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

clean_norm, sigma, clean_raw, noisy_raw, sample_snr = next(iter(loader))
clean_norm, sigma, clean_raw, noisy_raw = \
    clean_norm.to(DEVICE), sigma.to(DEVICE), clean_raw.to(DEVICE), noisy_raw.to(DEVICE)

print(f"[STEP 3] clean_norm.shape = {clean_norm.shape}")
print(f"[STEP 3] clean_raw.shape  = {clean_raw.shape}")
print(f"[STEP 3] sigma = {sigma.item():.3e}")
print(f"[STEP 3] clean_norm stats: std={clean_norm.std():.3f}, min={clean_norm.min():.3f}, max={clean_norm.max():.3f}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
print(f"[STEP 2] Checkpoint keys: {list(ckpt.keys())}")

model = UNet1D(
    in_ch=1,
    base_ch=ckpt['args']['base_ch'],
    time_dim=ckpt['args']['time_dim'],
    depth=ckpt['args']['depth']
).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"[STEP 2] Model loaded. #params = {sum(p.numel() for p in model.parameters())}")

diffusion = CustomDiffusion(T=ckpt['args']['T'], device=DEVICE)
print(f"[STEP 2] Diffusion initialized. T = {diffusion.T}")

alpha_bar = diffusion.alpha_bar.cpu().numpy()              # shape: (T,)
snr_ts   = np.sqrt(alpha_bar / (1.0 - alpha_bar))         # SNR(t) = sqrt(alpha_bar / (1-alpha_bar))

# pick the timesteps closest to the SNRs you care about (say 10 and 50):
targets  = [10, 50]
chosen_t = { tgt: int(np.argmin(np.abs(snr_ts - tgt))) for tgt in targets }
print(f"[STEP 3.5] SNR→t mapping: {chosen_t}")

# now loop over those rather than hard‐coded 0, T//2, T-1:
for tgt_snr, t_val in chosen_t.items():
    t = torch.full((BATCH_SIZE,), t_val, dtype=torch.long, device=DEVICE)

    # forward corrupt:
    xt, true_noise  = diffusion.q_sample(clean_norm, t)
    pred_noise      = model(xt, t)

    # one‐step inversion:
    ab               = diffusion.alpha_bar[t_val]
    sqrt_ab          = torch.sqrt(ab)
    sqrt_omb         = torch.sqrt(1 - ab)
    x0_hat_norm      = (xt - sqrt_omb * pred_noise) / sqrt_ab
    x0_hat_raw       = x0_hat_norm * sigma.view(-1,1,1)



    # plot
    plt.figure(figsize=(6,3))
    plt.plot(clean_raw[0,0].cpu(),               label='Clean')
    plt.plot(x0_hat_raw[0,0].detach().cpu(),
             alpha=0.7, label=f'Recon @ SNR≈{tgt_snr}')
    plt.title(f'one‐step recon @ t={t_val} (SNR≈{snr_ts[t_val]:.1f})')
    plt.legend(); plt.tight_layout(); plt.show()



# print raw values (first 10 samples)
raw_vals = clean_raw[0,0,:10].cpu().numpy()
print("Raw clean [first 10]:", raw_vals)

# print normalized values (first 10 samples)
norm_vals = clean_norm[0,0,:10].cpu().numpy()
print("Normalized clean [first 10]:", norm_vals)

# also show min/max/std to confirm ranges
print(f"Raw clean range: min={clean_raw.min().item():.3e}, max={clean_raw.max().item():.3e}, std={clean_raw.std().item():.3e}")
print(f"Norm clean range: min={clean_norm.min().item():.3f}, max={clean_norm.max().item():.3f}, std={clean_norm.std().item():.3f}")

for t_val in [0, diffusion.T // 2, diffusion.T - 1]:
    # create a batch of identical timesteps
    t = torch.full((BATCH_SIZE,), t_val, dtype=torch.long, device=DEVICE)

    # q_sample returns the noisy signal x_t and the true ε
    xt, true_noise = diffusion.q_sample(clean_norm, t)

    # UNet’s prediction of that noise
    pred_noise = model(xt, t)

    # print out their statistics
    print(f"\n[STEP 4] t = {t_val}")
    print(f"  x_t      → min={xt.min():.3e}, max={xt.max():.3e}, std={xt.std():.3e}")
    print(f"  true_noise → min={true_noise.min():.3e}, max={true_noise.max():.3e}, std={true_noise.std():.3e}")
    print(f"  pred_noise → min={pred_noise.min():.3e}, max={pred_noise.max():.3e}, std={pred_noise.std():.3e}")

    # 5) Single‐step reconstruction at t = T//2
    t_mid = diffusion.T // 2
    t = torch.full((BATCH_SIZE,), t_mid, dtype=torch.long, device=DEVICE)

    # get x_t and true noise (we already did this, but let’s grab them again)
    xt, true_noise = diffusion.q_sample(clean_norm, t)

    # network’s noise prediction
    pred_noise = model(xt, t)

    # compute alpha_bar[t]
    alpha_bar = diffusion.alpha_bar[t_mid]  # scalar
    sqrt_ab = torch.sqrt(alpha_bar)  # scalar
    sqrt_omb = torch.sqrt(1 - alpha_bar)  # scalar

    # invert one step:  x0_hat_norm = (xt - sqrt(1-ᾱₜ)·ε̂) / sqrt(ᾱₜ)
    x0_hat_norm = (xt - sqrt_omb * pred_noise) / sqrt_ab

    # un-normalize back to raw scale
    x0_hat = x0_hat_norm * sigma.view(-1, 1, 1)

    rand_snr = float(np.random.uniform(10, 50))
    # find the timestep whose SNR(t) is closest
    t_val = int(np.argmin(np.abs(snr_ts - rand_snr)))

    print(f"[STEP 3.5] Random target SNR={rand_snr:.1f} → t={t_val} (actual SNR≈{snr_ts[t_val]:.1f})")

    # now do your q_sample + one‐step inversion at that t_val
    t = torch.full((BATCH_SIZE,), t_val, dtype=torch.long, device=DEVICE)
    xt, true_noise = diffusion.q_sample(clean_norm, t)
    pred_noise = model(xt, t)
    ab = diffusion.alpha_bar[t_val]
    sqrt_ab = torch.sqrt(ab)
    sqrt_omb = torch.sqrt(1 - ab)
    x0_hat_norm = (xt - sqrt_omb * pred_noise) / sqrt_ab
    x0_hat_raw = x0_hat_norm * sigma.view(-1, 1, 1)

    # plot noisy vs reconstructed
    plt.figure(figsize=(6, 3))
    plt.plot(noisy_raw[0, 0].cpu().numpy(), label='Noisy (raw)')
    plt.plot(x0_hat_raw[0, 0].detach().cpu().numpy(), alpha=0.7, label=f'Recon at SNR≈{rand_snr:.1f}')
    plt.title(f'Noisy vs Recon at t={t_val} (SNR target={rand_snr:.1f})')
    plt.xlabel("Time index")
    plt.ylabel("Strain")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print stats
    print(f"\n[STEP 5] Single‐step recon at t={t_mid}")
    print(f"  x0_hat_norm stats: min={x0_hat_norm.min():.3e}, max={x0_hat_norm.max():.3e}, std={x0_hat_norm.std():.3e}")
    print(f"  x0_hat       stats: min={x0_hat.min():.3e}, max={x0_hat.max():.3e}, std={x0_hat.std():.3e}")

    # plot the clean vs reconstructed
    plt.figure(figsize=(8, 3))
    plt.plot(clean_raw[0, 0].cpu().numpy(), label='Clean raw')
    plt.plot(x0_hat[0, 0].detach().cpu().numpy(), alpha=0.7, label='Recon raw')
    plt.legend()
    plt.title(f'Clean vs Recon at t={t_mid}')
    plt.tight_layout()
    plt.show()