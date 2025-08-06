import argparse
import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from snr_denoising.models.model import UNet1D, CustomDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D diffusion model on LIGO waveforms")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to HDF5 dataset (with /waveform)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--T", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    return parser.parse_args()


class WaveDataset(Dataset):
    """
    PyTorch Dataset for loading clean waveforms from an HDF5 file.
    Expects dataset['waveform'] shaped (N, L).
    """
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, 'r')
        self.data = self.h5['signal'][:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return shape (1, L)
        x = self.data[idx]
        return torch.from_numpy(x).unsqueeze(0).float()


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # dataset & loader
    dataset = WaveDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # model & diffusion
    model = UNet1D(in_ch=1, base_ch=64, time_dim=128, depth=3).to(args.device)
    diffusion = CustomDiffusion(T=args.T, device=args.device)

    # optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x0 in loader:
            x0 = x0.to(args.device)  # shape (B,1,L)

            # sample random timesteps for each sample
            bsz = x0.size(0)
            t = torch.randint(0, diffusion.T, (bsz,), device=args.device)

            # forward diffusion: get noisy x_t and the true noise
            xt, noise = diffusion.q_sample(x0, t)

            # predict noise
            pred = model(xt, t)

            # compute loss
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d}/{args.epochs}, Loss: {avg_loss:.6f}")

        # save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"unet_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
