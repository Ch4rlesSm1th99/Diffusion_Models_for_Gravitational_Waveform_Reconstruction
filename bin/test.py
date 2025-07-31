import h5py
import numpy as np
import torch
from pathlib import Path
from model import UNet1D
import matplotlib.pyplot as plt
import sys

'''
Script prints off stats about randomly selected sample as well as metrics averaged over the whole test set. 
'''

DATA_BASE = Path(r'C:\Users\charl\PycharmProjects\snr_denoising\bin')
PATTERN = 'latest_dataset'
MODEL_FILE = Path(r'')
NUM_SAMPLES = 5     # sampels to plot and print stats off of


def find_latest_dataset(base: Path, prefix: str) -> Path:
    if not base.exists() or not base.is_dir():
        print(f"error: cannot find DATA_BASE --> '{base}'.")
        for p in Path('.').iterdir():
            print("  -", p.name)
        sys.exit(1)
    candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not candidates:
        print(f"no folders starting with '{prefix}' under {base}. Contents:")
        for p in base.iterdir(): print("  -", p.name)
        sys.exit(1)
    def key_fn(d):
        name = d.name.replace(prefix, '')
        return int(name) if name.isdigit() else name
    latest = sorted(candidates, key=key_fn)[-1]
    return latest


def compute_metrics_time(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray):
    mse     = np.mean((recon - clean)**2)
    snr_in  = np.sum(clean**2) / (np.sum((noisy - clean)**2) + 1e-12)
    snr_out = np.sum(clean**2) / (np.sum((recon - clean)**2) + 1e-12)
    return mse, snr_in, snr_out


def inspect_dataset(h5_path: Path):
    print(f"\n inspecting dataset: {h5_path}\n")
    with h5py.File(h5_path, 'r') as f:
        for key, val in f.attrs.items():
            print(f"attribute '{key}': {val}")

        t = f['time/t'][:]
        print(f"Time axis: {len(t)} points, span {t.min():.4f} to {t.max():.4f}\n")

        for split in ['train', 'test']:
            clean = f[f'time/clean_{split}'][:]
            noisy = f[f'time/noisy_{split}'][:]
            p_signal = np.mean(clean**2, axis=1)
            p_noise  = np.mean((noisy - clean)**2, axis=1)
            snrs     = p_signal / (p_noise + 1e-12)
            print(f"{split.capitalize()} set: {clean.shape[0]} samples")
            print(f"  Signal power: mean={p_signal.mean():.3e}, std={p_signal.std():.3e}")
            print(f"  Noise power:  mean={p_noise.mean():.3e}, std={p_noise.std():.3e}")
            print(f"  Input SNR (linear): mean={snrs.mean():.3f}, std={snrs.std():.3f}\n")


def run_model_checks(h5_path: Path, model_path: Path, num_samples: int = 5):
    print(f"\n--- Running model inference checks with '{model_path}'\n")
    if not model_path.exists():
        print(f"Error: MODEL_FILE '{model_path}' not found.")
        sys.exit(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = UNet1D(in_ch=1).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # load test data
    with h5py.File(h5_path, 'r') as f:
        noisy_all = f['time/noisy_test'][:]
        clean_all = f['time/clean_test'][:]
        t_axis    = f['time/t'][:]

    for i in range(min(num_samples, noisy_all.shape[0])):
        noisy = torch.from_numpy(noisy_all[i:i+1]).unsqueeze(1).float().to(device)
        clean = torch.from_numpy(clean_all[i:i+1]).unsqueeze(1).float().to(device)
        with torch.no_grad():
            t_idx = torch.zeros(1, dtype=torch.long, device=device)
            pred_noise = model(noisy, t_idx)
            recon = noisy - pred_noise

        clean_np = clean.cpu().numpy().squeeze()
        noisy_np = noisy.cpu().numpy().squeeze()
        recon_np = recon.cpu().numpy().squeeze()

        mse, snr_in, snr_out = compute_metrics_time(clean_np, noisy_np, recon_np)
        print(f"Sample {i+1:2d}: MSE={mse:.3e} , SNR_in={snr_in:6.3f} , SNR_out={snr_out:6.3f}")

        # plot
        plt.figure(figsize=(6,3))
        plt.plot(t_axis, noisy_np, linestyle=':', label='Noisy', alpha=0.6)
        plt.plot(t_axis, clean_np,  label='Clean', linewidth=1)
        plt.plot(t_axis, recon_np,  label='Recon', linewidth=1)
        plt.title(f"Sample {i+1}")
        plt.legend(); plt.tight_layout(); plt.show()


def main():
    dataset_dir = find_latest_dataset(DATA_BASE, PATTERN)
    h5_file = dataset_dir / 'dataset.h5'

    inspect_dataset(h5_file)

    run_model_checks(h5_file, MODEL_FILE, num_samples=NUM_SAMPLES)


if __name__ == '__main__':
    main()
