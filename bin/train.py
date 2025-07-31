import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import UNet1D


def prepare_output_dir(base_dir: str) -> Path:
    """
    overwrite the 'latest_model' directory under base_dir. Unless specified other.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    out_dir = base / "latest_model"
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()
    out_dir.mkdir(exist_ok=True)
    return out_dir


class H5TimeSupervisedDataset(Dataset):
    """
    Loads (noisy, clean, snr) for training. Samples retrieved via SNR bin in most cases.
    """

    def __init__(self, h5_path: str, split="train", normalize=True):
        with h5py.File(h5_path, "r") as f:
            grp = f["time"]
            clean_np = grp[f"clean_{split}"][:]  # [N, L]
            noisy_np = grp[f"noisy_{split}"][:]
        self.clean = torch.from_numpy(clean_np)[:, None, :].float()     # to tensors
        self.noisy = torch.from_numpy(noisy_np)[:, None, :].float()

        # compute snr per sample
        p_signal = np.mean(clean_np ** 2, axis=1)
        p_noise = np.mean((noisy_np - clean_np) ** 2, axis=1)
        self.snr = torch.from_numpy(p_signal / (p_noise + 1e-12)).float()

        if normalize:
            m = self.noisy.abs().max()
            self.noisy /= m
            self.clean /= m
        self.length = self.noisy.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return noisy, clean, and the snr (scalar value)
        return self.noisy[idx], self.clean[idx], self.snr[idx]


def train_supervised(args):
    out_dir = prepare_output_dir(args.model_dir)        # prep output dir
    log_path = out_dir / "train_supervised_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss\n")

    ds = H5TimeSupervisedDataset(f"{args.data}/dataset.h5", split="train")      #load dataset

    snr_array_np = ds.snr.numpy() if args.scheduled_sampling else None      # for scheduled sampling SNR bins

    max_snr = ds.snr.max().item()     # global max SNR for weighting

    device = args.device
    model = UNet1D(in_ch=1).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.MSELoss(reduction='none')       # use element-wise MSE for per-sample loss

    for epoch in range(1, args.epochs + 1):
        # build dataLoader every epoch if scheduled sampling
        if args.scheduled_sampling:
            max_snr_val, min_snr_val = snr_array_np.max(), snr_array_np.min()
            thresh = max_snr_val - (max_snr_val - min_snr_val) * (epoch - 1) / (args.epochs - 1)
            # ensure thresh lies within [min_snr, max_snr]
            thresh = float(min(max(thresh, min_snr_val), max_snr_val))
            keep_idx = np.where(snr_array_np >= thresh)[0]

            if len(keep_idx) == 0:
                # fallback--> use all data
                dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2)
            else:
                sampler = SubsetRandomSampler(keep_idx)
                dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=2)
        else:
            dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2)

        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for noisy, clean, snr in pbar:
            noisy, clean, snr = noisy.to(device), clean.to(device), snr.to(device)
            # predict noise
            target_noise = noisy - clean
            t_idx = torch.zeros(noisy.size(0), dtype=torch.long, device=device)
            pred_noise = model(noisy, t_idx)

            loss_per_sample = criterion(pred_noise, target_noise).mean(dim=[1, 2])  # [B]

            if args.snr_weighted:
                # SNR weighting to loss --> note doesn't work/not optimised at the momemnt (need to find alpha sweet spot if it works)
                eps = 1e-6
                alpha = args.snr_weight_exp
                weights = (max_snr / (snr + eps)) ** alpha
                weights = weights / weights.mean()
                loss = (weights * loss_per_sample).mean()
            else:
                # basic MSE --> works normally just use this
                loss = loss_per_sample.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch}/{args.epochs} – loss={avg_loss:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f}\n")

    # save final model
    torch.save(model.state_dict(), out_dir / "model.pth")
    print(f"Saved weighted-supervised-denoiser to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Supervised denoiser training with and without SNR weighted loss (broken) and optional curriculum sampling"
    )
    p.add_argument(
        "--data", default=r"C:\Users\charl\PycharmProjects\snr_denoising\bin\data",
        help="Folder containing dataset.h5"
    )
    p.add_argument(
        "--model_dir", default=r"C:/Users/charl/PycharmProjects/snr_denoising/model",
        help="Base folder where 'latest_model' is saved"
    )
    p.add_argument(
        "--scheduled_sampling", action="store_true",
        help="Enable curriculum over SNR: start with high-SNR to lower-SNR samples"
    )
    p.add_argument(
        "--snr_weight_exp", type=float, default=1.0,
        help="Exponent alpha for SNR-based loss weighting (alpha>1 tries hard to solve with low SNR)"
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--device", default=None,
        help="torch device"
    )
    p.add_argument(
        "--snr_weighted", action="store_true",
        help="If yes, uses SNR-weighted loss; otherwise uses standard MSE"
    )
    args = p.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_supervised(args)
