import argparse
import os
import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models import UNet1D, CustomDiffusion


def prepare_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, 'latest_model')
    if os.path.exists(out_dir):
        for file in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, file))
    else:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


class CleanWaveDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5 = h5py.File(h5_path, 'r')
        self.signals = self.h5['signal']

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        clean_np = self.signals[idx]
        clean = torch.from_numpy(clean_np).unsqueeze(0).float()
        sigma = clean.std()
        if sigma == 0:      # normalise + avoid divide by zero error (dont add small numebr)
            sigma = torch.tensor(1.0)
        clean_norm = clean / sigma
        return clean_norm, sigma

    def __del__(self):
        try:
            self.h5.close()
        except Exception:
            pass

class NoisyWaveDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5 = None
        self.noisy = None
        self.mask = None

    def _ensure_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r', swmr=True)  # read-only
            self.noisy = self.h5['noisy']
            self.mask  = self.h5.get('mask', None)

    def __len__(self):
        if self.noisy is not None:
            return self.noisy.shape[0]
        with h5py.File(self.h5_path, 'r') as f:
            return f['noisy'].shape[0]

    def __getitem__(self, idx):
        self._ensure_open()
        noisy = torch.from_numpy(self.noisy[idx]).unsqueeze(0).float()
        if self.mask is not None:
            mask = torch.from_numpy(self.mask[idx].astype(np.float32)).unsqueeze(0)
            valid = noisy[mask.bool()]
            sigma = valid.std() if valid.numel() > 0 else torch.tensor(1.0)
        else:
            mask = torch.ones_like(noisy)
            s = noisy.std()
            sigma = s if s > 0 else torch.tensor(1.0)
        noisy_norm = noisy / sigma
        return noisy_norm, sigma, mask

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except:
            pass




def train_diffusion(args):
    out_dir = prepare_output_dir(args.model_dir)

    ds = NoisyWaveDataset(args.data)
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=4, pin_memory=True,persistent_workers=True,prefetch_factor=2)

    device = torch.device(args.device)
    model = UNet1D(in_ch=1, base_ch=args.base_ch, time_dim=args.time_dim, depth=args.depth).to(device)
    diffusion = CustomDiffusion(T=args.T, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (noisy_norm, sigma, mask) in enumerate(pbar):
            noisy_norm = noisy_norm.to(device)
            sigma = sigma.to(device)
            mask = mask.to(device)
            bsz = noisy_norm.size(0)

            t = torch.randint(0, args.T, (bsz,), device=device)
            xt, noise = diffusion.q_sample(noisy_norm, t)
            pred_noise = model(xt, t)

            # training diagnostics
            if i == 0:
                print("model device:", next(model.parameters()).device)
                print("batch device:", noisy_norm.device, sigma.device, mask.device)
                print("current cuda:", torch.cuda.current_device() if torch.cuda.is_available() else None)
                def _stats(x):
                    return x.min().item(), x.max().item(), x.mean().item(), x.std().item()

                nn_min, nn_max, nn_mean, nn_std = _stats(noisy_norm)           # input singal to forward diffusion stat
                noise_min, noise_max, noise_mean, noise_std = _stats(noise)         # ground truth epsilon stat
                xt_min, xt_max, xt_mean, xt_std = _stats(xt)       # noised signal at timestep t after forward process stat
                p_min, p_max, p_mean, p_std = _stats(pred_noise)           # pred of epsilon stat

                recon_norm = xt - pred_noise        # reconstruct normalised signal
                r_min, r_max, r_mean, r_std = _stats(recon_norm)    # trend check

                recon_unnorm = recon_norm * sigma.view(-1, 1, 1)        # un-normalize
                ru_min, ru_max, ru_mean, ru_std = _stats(recon_unnorm)

                print(f"[DEBUG] noisy_norm:  min={nn_min:.3e}, max={nn_max:.3e}, mean={nn_mean:.3e}, std={nn_std:.3e}")
                print(
                    f"[DEBUG] noise:        min={noise_min:.3e}, max={noise_max:.3e}, mean={noise_mean:.3e}, std={noise_std:.3e}")
                print(f"[DEBUG] xt:           min={xt_min:.3e}, max={xt_max:.3e}, mean={xt_mean:.3e}, std={xt_std:.3e}")
                print(f"[DEBUG] pred_noise:   min={p_min:.3e}, max={p_max:.3e}, mean={p_mean:.3e}, std={p_std:.3e}")
                print(f"[DEBUG] recon_norm:   min={r_min:.3e}, max={r_max:.3e}, mean={r_mean:.3e}, std={r_std:.3e}")
                print(
                    f"[DEBUG] recon_unnorm: min={ru_min:.3e}, max={ru_max:.3e}, mean={ru_mean:.3e}, std={ru_std:.3e}\n")


                recon_mse_norm = torch.nn.functional.mse_loss(recon_norm, noisy_norm)
                print(f"[CHECK] Epoch {epoch} recon MSE (norm): {recon_mse_norm:.3e}")  # MSE on recon norm

                noisy_flat = noisy_norm.view(bsz, -1)
                recon_flat = recon_norm.view(bsz, -1)
                cos_sim_norm = torch.nn.functional.cosine_similarity(noisy_flat, recon_flat, dim=1).mean()
                print(f"[CHECK] Epoch {epoch} cos-sim (norm): {cos_sim_norm:.3f}")      # cosine simaliry on norm

                true_noise_norm = xt - noisy_norm
                res_noise_norm = true_noise_norm - pred_noise
                snr_imp_norm = true_noise_norm.std() / res_noise_norm.std()
                print(f"[CHECK] Epoch {epoch} SNR improvement (norm): {snr_imp_norm:.2f}x") # SNR improvement

            mse = (pred_noise - noise) ** 2
            denom = mask.sum(dim=[1, 2]).clamp_min(1.0)  # unpadded points per sample
            loss = (mse * mask).sum(dim=[1, 2]) / denom  # masked MSE
            loss = loss.mean()  # average over batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch}/{args.epochs} - Average Loss: {avg_loss:.6f}")

    save_path = os.path.join(out_dir, 'model_diffusion.pth')
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': vars(args),
        'epoch': args.epochs,
    }, save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train diffusion model on LIGO waveforms")
    parser.add_argument('--data',       type=str, required=True,
                        help='Path to HDF5 file with "signal" dataset')
    parser.add_argument('--model_dir',  type=str, default='model',
                        help='Base directory to create latest_model')
    parser.add_argument('--epochs',     type=int,   default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int,   default=16,
                        help='Batch size')
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--T',          type=int,   default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--base_ch',    type=int,   default=64,
                        help='Base channel count for UNet')
    parser.add_argument('--time_dim',   type=int,   default=128,
                        help='Time embedding dimension')
    parser.add_argument('--depth',      type=int,   default=3,
                        help='UNet depth (number of down/up blocks)')
    parser.add_argument('--device',     type=str,   default=None,
                        help='Device (cpu or cuda)')
    args = parser.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    train_diffusion(args)
