import os
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from models import UNet1D, CustomDiffusion



class InferenceDataset(Dataset):
    """
    Reads HDF5 file with datasets:
      - needed: 'signal', 'noisy'
      - optional: 'mask', 'label_m1','label_m2' (or 'mass1','mass2'), 'snr'
    NOTE: Normalisation uses the noisy waveform std.
    """
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, "r")
        self.signal = self.h5["signal"]
        self.noisy = self.h5["noisy"]
        self.mask = self.h5.get("mask", None)

        self.meta = {}
        for k in ["label_m1", "label_m2", "mass1", "mass2", "snr"]:
            if k in self.h5:
                self.meta[k] = self.h5[k][...]

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx):
        clean_raw = torch.from_numpy(self.signal[idx]).float().unsqueeze(0)  # [1, L]
        noisy_raw = torch.from_numpy(self.noisy[idx]).float().unsqueeze(0)   # [1, L]

        if self.mask is not None:
            mask = torch.from_numpy(self.mask[idx].astype(np.float32)).unsqueeze(0)  # [1, L]
            valid = noisy_raw[mask.bool()]
            sigma = valid.std() if valid.numel() > 0 else torch.tensor(1.0)
        else:
            mask = torch.ones_like(noisy_raw)
            s = noisy_raw.std()
            sigma = s if s > 0 else torch.tensor(1.0)

        clean_norm = clean_raw / sigma

        # choose labels: prefer label_m1/label_m2 if present
        if "label_m1" in self.meta and "label_m2" in self.meta:
            m1 = float(self.meta["label_m1"][idx])
            m2 = float(self.meta["label_m2"][idx])
        elif "mass1" in self.meta and "mass2" in self.meta:
            m1 = float(self.meta["mass1"][idx])
            m2 = float(self.meta["mass2"][idx])
        else:
            m1 = m2 = None

        snr_meta = float(self.meta["snr"][idx]) if "snr" in self.meta else None

        return {
            "clean_norm": clean_norm,
            "clean_raw": clean_raw,
            "noisy_raw": noisy_raw,
            "sigma": sigma,
            "mask": mask,
            "m1": m1,
            "m2": m2,
            "snr_meta": snr_meta,
            "idx": int(idx),
        }

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    def __del__(self):
        self.close()



def snr_from_alpha_bar(alpha_bar: torch.Tensor) -> np.ndarray:
    """SNR(t) = sqrt(alpha_bar / (1 - alpha_bar))"""
    ab = alpha_bar.detach().cpu().numpy()
    return np.sqrt(ab / (1.0 - ab))

def t_for_target_snr(diffusion, target_snr: float) -> (int, float):
    snr_ts = snr_from_alpha_bar(diffusion.alpha_bar)
    t = int(np.argmin(np.abs(snr_ts - target_snr)))
    return t, float(snr_ts[t])

def one_step_recon(model, diffusion, clean_norm: torch.Tensor, sigma: torch.Tensor, t_scalar: int) -> torch.Tensor:
    """
      --> x0_hat_norm = (x_t - sqrt(1 - alpha_bar[t]) * eps_hat) / sqrt(alpha_bar[t])
      --> x0_hat_raw  = x0_hat_norm * sigma
    """
    B = clean_norm.shape[0]
    device = clean_norm.device
    t = torch.full((B,), int(t_scalar), dtype=torch.long, device=device)

    x_t, _ = diffusion.q_sample(clean_norm, t)
    eps_hat = model(x_t, t)

    ab = diffusion.alpha_bar[t_scalar]  # scalar tensor on device
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
    x0_hat_raw = x0_hat_norm * sigma.view(-1, 1, 1)
    return x0_hat_raw

def first_last_true(mask_1d: np.ndarray):
    """Returns (start, stop) indices covering all True in 1D boolean mask and None if empty."""
    if mask_1d.ndim != 1:
        mask_1d = mask_1d.reshape(-1)
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])

def even_pick(indices: list, k: int) -> list:
    """Pick up to k indices evenly spaced from a list, preserving order."""
    n = len(indices)
    if n <= k:
        return indices
    picks = np.linspace(0, n - 1, k)
    picks = np.unique(np.floor(picks).astype(int)).tolist()
    while len(picks) < k:
        picks.append(picks[-1])
    return [indices[i] for i in picks[:k]]

def group_indices_by_mass(h5_path: str, unordered: bool = False):
    """
    Return dict {(m1, m2): [idx, ...]} using label_m1/label_m2 if present, else mass1/mass2.
    If unordered=True, (m1,m2) and (m2,m1) treated as the same combo (diagonalise the options).
    """
    with h5py.File(h5_path, "r") as f:
        if "label_m1" in f and "label_m2" in f:
            m1 = np.array(f["label_m1"])
            m2 = np.array(f["label_m2"])
        elif "mass1" in f and "mass2" in f:
            m1 = np.array(f["mass1"])
            m2 = np.array(f["mass2"])
        else:
            return {("all", "all"): list(range(f["signal"].shape[0]))}

    groups = {}
    for i, (a, b) in enumerate(zip(m1, m2)):
        a = float(np.round(a, 6))
        b = float(np.round(b, 6))
        key = tuple(sorted((a, b))) if unordered else (a, b)
        groups.setdefault(key, []).append(int(i))
    return groups


def plot_overlaid(
    tag: str,
    noisy_1d: np.ndarray,
    recon_1d: np.ndarray,
    clean_1d: np.ndarray,
    out_dir: str,
    title: str = "",
    crop_suffix: str = "",
    ext: str = "pdf",
    dpi: int = 150,
    lw_noisy: float = 1.0,
    lw_recon: float = 1.0,
    lw_clean: float = 1.0,
    plot_type: str = "all",   # "all", "clean_recon", "noisy_recon"
    fig_w: float = 12.0,      # inches
    fig_h: float = 3.6,       # inches
    line_scale: float = 1.0,  # all line widths
    font_scale: float = 1.0,  # font sizes
    tight: bool = False,
):
    """Conditionally plot requested traces and save a larger figure if desired."""
    lw_noisy *= line_scale
    lw_recon *= line_scale
    lw_clean *= line_scale

    with plt.rc_context({
        "font.size":        10 * font_scale,
        "axes.titlesize":   12 * font_scale,
        "axes.labelsize":   11 * font_scale,
        "xtick.labelsize":   9 * font_scale,
        "ytick.labelsize":   9 * font_scale,
        "legend.fontsize":   9 * font_scale,
    }):
        plt.figure(figsize=(fig_w, fig_h))  # inches

        if plot_type in ("all", "noisy_recon"):
            plt.plot(noisy_1d, label="Noisy", alpha=0.6, linewidth=lw_noisy)
        if plot_type in ("all", "clean_recon", "noisy_recon"):
            plt.plot(recon_1d, label="Recon", alpha=0.8, linewidth=lw_recon)
        if plot_type in ("all", "clean_recon"):
            plt.plot(clean_1d, label="Clean", alpha=1.0, linewidth=lw_clean)

        if title:
            plt.title(title)
        plt.xlabel("Time index")
        plt.ylabel("Strain")
        plt.legend(frameon=False)
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        fname = f"{tag}{crop_suffix}.{ext}"
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
        plt.savefig(os.path.join(out_dir, fname), **save_kwargs)
        plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot overlays for each mass combination (3 examples per combo by default)."
    )
    ap.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset")
    ap.add_argument("--model", type=str, required=True, help="Path to trained diffusion checkpoint (.pth)")
    ap.add_argument("--outdir", type=str, default="plots_by_mass", help="Where to save figures")

    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    ap.add_argument("--examples-per-combo", type=int, default=3, help="How many examples to plot per (m1,m2) combo")
    ap.add_argument("--target-snr", type=float, default=20.0, help="Target SNR to pick timestep for one-step recon")
    ap.add_argument("--crop-to-mask", action="store_true", help="Crop plots to the valid (unpadded) region")
    ap.add_argument("--crop-margin", type=int, default=0, help="Extra samples to keep on each side when cropping")
    ap.add_argument("--also-uncropped", action="store_true", help="Save an uncropped figure in addition to cropped one")
    ap.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"], help="Output image format")
    ap.add_argument("--dpi", type=int, default=150, help="DPI for raster outputs (png); ignored for vector pdf")

    ap.add_argument("--plot-type", type=str, default="all",
                    choices=["all", "clean_recon", "noisy_recon"],
                    help="Which graphs to plot")

    ap.add_argument("--fig-width", type=float, default=12.0, help="Figure width in inches")
    ap.add_argument("--fig-height", type=float, default=3.6, help="Figure height in inches")
    ap.add_argument("--line-scale", type=float, default=1.0, help="Multiply all line widths")
    ap.add_argument("--font-scale", type=float, default=1.0, help="Multiply font sizes (title/labels/ticks)")
    ap.add_argument("--tight", action="store_true", help="Use bbox_inches='tight' to trim margins")

    ap.add_argument("--unordered-pairs", action="store_true",
                    help="If set, (m1,m2) and (m2,m1) into the same (min,max) pair")

    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device = {device}")

    # group masses
    groups = group_indices_by_mass(args.data, unordered=args.unordered_pairs)
    n_groups = len(groups)
    print(f"[info] found {n_groups} mass combinations "
          f"({'unordered' if args.unordered_pairs else 'ordered'}).")

    # load model --> diffusion
    ckpt = torch.load(args.model, map_location=device)
    model = UNet1D(
        in_ch=1,
        base_ch=ckpt["args"]["base_ch"],
        time_dim=ckpt["args"]["time_dim"],
        depth=ckpt["args"]["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = CustomDiffusion(T=ckpt["args"]["T"], device=device)
    t_pick, snr_at_t = t_for_target_snr(diffusion, args.target_snr)
    print(f"[info] target SNR={args.target_snr:.1f} → using t={t_pick} (actual SNR≈{snr_at_t:.2f})")

    # iterate over allowed combos
    ds = InferenceDataset(args.data)

    def sort_key(k):
        m1, m2 = k
        if isinstance(m1, str):
            return (float("inf"), float("inf"))
        return (m1, m2)

    for combo_idx, (combo, idx_list) in enumerate(sorted(groups.items(), key=lambda kv: sort_key(kv[0]))):
        m1, m2 = combo
        tag_combo = f"m1_{m1}_m2_{m2}" if not isinstance(m1, str) else "all"

        pick_idxs = even_pick(idx_list, args.examples_per_combo)
        print(f"[combo {combo_idx+1}/{n_groups}] {tag_combo}: {len(idx_list)} samples --> plotting {len(pick_idxs)}")

        out_dir_combo = os.path.join(args.outdir, tag_combo)
        os.makedirs(out_dir_combo, exist_ok=True)

        for j, idx in enumerate(pick_idxs):
            item = ds[idx]
            clean_norm = item["clean_norm"].to(device)   #
            clean_raw  = item["clean_raw"].to(device)
            noisy_raw  = item["noisy_raw"].to(device)
            sigma      = item["sigma"].to(device)

            mask_tensor = item["mask"]
            mask_np = mask_tensor.cpu().numpy().astype(bool)
            mask_1d = mask_np[0] if mask_np.ndim == 2 else mask_np

            # reconstruct at chosen t
            x0_hat_raw = one_step_recon(model, diffusion, clean_norm, sigma, t_pick)

            clean_1d = clean_raw[0].detach().cpu().numpy().ravel()
            noisy_1d = noisy_raw[0].detach().cpu().numpy().ravel()
            recon_1d = x0_hat_raw[0].detach().cpu().numpy().ravel()

            # cropping
            crop_suffix = ""
            if args.crop_to_mask and mask_1d is not None:
                start_stop = first_last_true(mask_1d)
                if start_stop is not None:
                    s, e = start_stop
                    s = max(0, s - args.crop_margin)
                    e = min(len(mask_1d) - 1, e + args.crop_margin)
                    clean_c = clean_1d[s: e + 1]
                    noisy_c = noisy_1d[s: e + 1]
                    recon_c = recon_1d[s: e + 1]
                    crop_suffix = "_cropped"
                else:
                    clean_c, noisy_c, recon_c = clean_1d, noisy_1d, recon_1d
            else:
                clean_c, noisy_c, recon_c = clean_1d, noisy_1d, recon_1d

            title = f"(m1={m1}, m2={m2})  idx={idx}  t={t_pick}  SNR≈{snr_at_t:.1f}"
            tag = f"{tag_combo}_ex{j}_idx{idx}"

            plot_overlaid(
                tag=tag,
                noisy_1d=noisy_c,
                recon_1d=recon_c,
                clean_1d=clean_c,
                out_dir=out_dir_combo,
                title=title + (", cropped" if crop_suffix else ""),
                crop_suffix=crop_suffix,
                ext=args.ext,
                dpi=args.dpi,
                plot_type=args.plot_type,
                fig_w=args.fig_width,
                fig_h=args.fig_height,
                line_scale=args.line_scale,
                font_scale=args.font_scale,
                tight=args.tight,
            )

            # also save the uncropped version if needed
            if args.also_uncropped and crop_suffix:
                plot_overlaid(
                    tag=tag,
                    noisy_1d=noisy_1d,
                    recon_1d=recon_1d,
                    clean_1d=clean_1d,
                    out_dir=out_dir_combo,
                    title=title + ", full",
                    crop_suffix="_full",
                    ext=args.ext,
                    dpi=args.dpi,
                    plot_type=args.plot_type,
                    fig_w=args.fig_width,
                    fig_h=args.fig_height,
                    line_scale=args.line_scale,
                    font_scale=args.font_scale,
                    tight=args.tight,
                )

    ds.close()
    print(f"[complete] plots written under: {args.outdir}")


if __name__ == "__main__":
    main()
