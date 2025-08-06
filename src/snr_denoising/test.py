import argparse
import os
import h5py
import torch
import matplotlib.pyplot as plt
from snr_denoising.models.model import UNet1D, CustomDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and plot diffusion denoising results")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to HDF5 dataset containing 'times' and 'signal'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained UNet checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (cpu or cuda)")
    parser.add_argument("--T", type=int, default=1000,
                        help="Number of diffusion timesteps (must match training)")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="How many examples to plot")
    return parser.parse_args()


def load_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        times = f['times'][:]
        signals = f['signal'][:]
    return times, signals


def main():
    args = parse_args()
    times, signals = load_data(args.dataset)

    device = torch.device(args.device)
    diffusion = CustomDiffusion(T=args.T, device=device)
    model = UNet1D(in_ch=1, base_ch=64, time_dim=128, depth=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    for idx in range(min(args.num_examples, signals.shape[0])):
        x0 = torch.from_numpy(signals[idx]).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,L)
        t = torch.randint(0, args.T, (1,), device=device)
        xt, _ = diffusion.q_sample(x0, t)

        with torch.no_grad():
            noise_pred = model(xt, t)

        # reconstruct x0
        alpha_bar_t = diffusion.alpha_bar[t].view(1,1,1)
        sqrt_ab = alpha_bar_t.sqrt()
        sqrt_mb = (1 - alpha_bar_t).sqrt()
        x0_pred = (xt - sqrt_mb * noise_pred) / sqrt_ab

        noisy_np    = xt.squeeze().cpu().numpy()
        denoised_np = x0_pred.squeeze().cpu().numpy()
        clean_np    = x0.squeeze().cpu().numpy()

        # Plot: noisy in first panel, clean vs denoised overlay in second
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(times, noisy_np)
        plt.title(f"Noisy Signal at t={t.item()} (Example {idx})")
        plt.ylabel('Strain')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(times, clean_np, label='Clean')
        plt.plot(times, denoised_np, label='Denoised', alpha=0.7)
        plt.title(f"Clean vs Denoised (Example {idx})")
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
