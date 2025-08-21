import argparse
import os
import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models import UNet1D, CustomDiffusion
from dataloader import make_dataloader, NoisyWaveDataset

def prepare_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, 'latest_model')
    if os.path.exists(out_dir):
        for file in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, file))
    else:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def train_diffusion(args):
    out_dir = prepare_output_dir(args.model_dir)

    # loader yields: clean_raw, noisy_raw, sigma, mask
    loader = make_dataloader(
        h5_path=args.data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = UNet1D(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim, depth=args.depth).to(device)
    diffusion = CustomDiffusion(T=args.T, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for i, (clean_raw, noisy_raw, sigma, mask) in enumerate(pbar):
            clean_raw = clean_raw.to(device).float()
            noisy_raw = noisy_raw.to(device).float()
            sigma     = sigma.to(device)
            mask      = mask.to(device).float()
            bsz = clean_raw.size(0)

            sigma_ = sigma.view(-1, 1, 1)
            clean_norm = clean_raw / sigma_
            cond_norm  = noisy_raw / sigma_

            t = torch.randint(0, args.T, (bsz,), device=device)
            x_t, eps = diffusion.q_sample(clean_norm, t)

            net_in = torch.cat([x_t, cond_norm], dim=1)
            eps_hat = model(net_in, t)


            mse = (eps_hat - eps) ** 2
            denom = mask.sum(dim=[1, 2]).clamp_min(1.0)    # unpadded points per sample
            loss = (mse * mask).sum(dim=[1, 2]) / denom
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            if i == 0:
                def _stats(x):
                    return x.min().item(), x.max().item(), x.mean().item(), x.std().item()

                ct_min, ct_max, ct_mean, ct_std = _stats(x_t)
                cn_min, cn_max, cn_mean, cn_std = _stats(clean_norm)
                yn_min, yn_max, yn_mean, yn_std = _stats(cond_norm)
                eh_min, eh_max, eh_mean, eh_std = _stats(eps_hat)
                print(f"[DEBUG] clean_norm: min={cn_min:.3e}, max={cn_max:.3e}, mean={cn_mean:.3e}, std={cn_std:.3e}")
                print(f"[DEBUG] cond_norm:  min={yn_min:.3e}, max={yn_max:.3e}, mean={yn_mean:.3e}, std={yn_std:.3e}")
                print(f"[DEBUG] x_t:        min={ct_min:.3e}, max={ct_max:.3e}, mean={ct_mean:.3e}, std={ct_std:.3e}")
                print(f"[DEBUG] eps_hat:    min={eh_min:.3e}, max={eh_max:.3e}, mean={eh_mean:.3e}, std={eh_std:.3e}")

                # x0_hat_norm = (x_t - sqrt(1 - alpha_bar[t])*eps_hat) / sqrt(alpha_bar[t])
                ab = diffusion.alpha_bar[t].view(-1, 1, 1)        # [B,1,1]
                x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
                x0_hat = x0_hat_norm * sigma_

                x0_min, x0_max, x0_mean, x0_std = _stats(x0_hat)
                print(f"[DEBUG] recon_unnorm: min={x0_min:.3e}, max={x0_max:.3e}, mean={x0_mean:.3e}, std={x0_std:.3e}")

        avg_loss = total_loss / sum(len for _, _, _, len in loader.dataset) if hasattr(loader, 'dataset') else total_loss
        print(f"Epoch {epoch}/{args.epochs} - Average Loss: {avg_loss:.6f}")

    save_path = os.path.join(out_dir, 'model_diffusion.pth')
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': {**vars(args), 'conditional': True, 'in_ch': 2, 'conditioning': 'concat_noisy'},
        'epoch': args.epochs,
    }, save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train conditional diffusion denoiser on LIGO waveforms")
    parser.add_argument('--data',       type=str, required=True,
                        help='Path to HDF5 with "signal" and "noisy" datasets (variable-length)')
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
