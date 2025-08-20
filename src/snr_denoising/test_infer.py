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
    HDF5 expected (new gen.py):
      - required: 'signal', 'noisy', 'times' (seconds-relative; ideally t=0 at merger)
      - optional: 'label_m1','label_m2' (or 'mass1','mass2'), 'snr'
    Normalisation uses the noisy waveform std (per sample).
    """
    def __init__(self, h5_path: str):
        self.h5 = h5py.File(h5_path, "r")
        self.signal = self.h5["signal"]
        self.noisy  = self.h5["noisy"]
        self.times  = self.h5.get("times", None)

        # attrs for fallback time construction
        self.delta_t   = float(self.h5.attrs.get("delta_t", 1.0 / 4096.0))
        self.time_axis = str(self.h5.attrs.get("time_axis", ""))

        self.meta = {}
        for k in ["label_m1", "label_m2", "mass1", "mass2", "snr"]:
            if k in self.h5:
                self.meta[k] = self.h5[k][...]

    def __len__(self):
        return self.signal.shape[0]

    def _times_rel(self, idx: int, L: int) -> np.ndarray:
        """
        Return seconds-relative time with t=0 at the local merger at (max signal).
        Uses stored 'times' if available, but re-centers on the peak unless the file
        explicitly says it already.
        """
        axis = (self.time_axis or "").lower()

        if self.times is not None:
            t = self.times[idx].astype(np.float64)
            if axis in ("seconds-rel-peak", "seconds_rel_peak"):
                return t
            # Otherwise, enforce peak-centering using the signal.
            s = self.signal[idx]
            pk = int(np.argmax(np.abs(s)))
            return t - t[pk]

        # Fallback: build a time vector from delta_t and center on the peak index.
        s = self.signal[idx]
        pk = int(np.argmax(np.abs(s)))
        base = np.arange(L, dtype=np.float64) * self.delta_t
        return base - base[pk]

    def __getitem__(self, idx):
        clean_raw = torch.from_numpy(self.signal[idx]).float().unsqueeze(0)  # [1, L]
        noisy_raw = torch.from_numpy(self.noisy[idx]).float().unsqueeze(0)   # [1, L]
        L = noisy_raw.shape[-1]

        t_rel = self._times_rel(idx, L)

        # per-sample sigma from noisy
        s = noisy_raw.std()
        sigma = s if s > 0 else torch.tensor(1.0)
        clean_norm = clean_raw / sigma

        # labels (prefer label_m1/label_m2)
        if "label_m1" in self.meta and "label_m2" in self.meta:
            m1 = float(self.meta["label_m1"][idx]); m2 = float(self.meta["label_m2"][idx])
        elif "mass1" in self.meta and "mass2" in self.meta:
            m1 = float(self.meta["mass1"][idx]);    m2 = float(self.meta["mass2"][idx])
        else:
            m1 = m2 = None

        snr_meta = float(self.meta["snr"][idx]) if "snr" in self.meta else None

        return {
            "clean_norm": clean_norm,  # [1, L]
            "clean_raw":  clean_raw,   # [1, L]
            "noisy_raw":  noisy_raw,   # [1, L]
            "times_rel":  t_rel,       # (L,)
            "sigma":      sigma,       # scalar tensor
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
    x0_hat_norm = (x_t - sqrt(1 - alpha_bar[t]) * eps_hat) / sqrt(alpha_bar[t])
    x0_hat_raw  = x0_hat_norm * sigma
    """
    B = clean_norm.shape[0]
    device = clean_norm.device
    t = torch.full((B,), int(t_scalar), dtype=torch.long, device=device)
    x_t, _ = diffusion.q_sample(clean_norm, t)
    eps_hat = model(x_t, t)
    ab = diffusion.alpha_bar[t_scalar]
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
    return x0_hat_norm * sigma.view(-1, 1, 1)

def even_pick(indices: list, k: int) -> list:
    """Pick up to k indices evenly spaced from a list, preserving order."""
    n = len(indices)
    if n <= k: return indices
    picks = np.linspace(0, n - 1, k)
    picks = np.unique(np.floor(picks).astype(int)).tolist()
    while len(picks) < k:
        picks.append(picks[-1])
    return [indices[i] for i in picks[:k]]

def group_indices_by_mass(h5_path: str, unordered: bool = False):
    """Return dict {(m1, m2): [idx, ...]} using label_m1/label_m2 if present, else mass1/mass2."""
    with h5py.File(h5_path, "r") as f:
        if "label_m1" in f and "label_m2" in f:
            m1 = np.array(f["label_m1"]); m2 = np.array(f["label_m2"])
        elif "mass1" in f and "mass2" in f:
            m1 = np.array(f["mass1"]);    m2 = np.array(f["mass2"])
        else:
            return {("all", "all"): list(range(f["signal"].shape[0]))}

    groups = {}
    for i, (a, b) in enumerate(zip(m1, m2)):
        a = float(np.round(a, 6)); b = float(np.round(b, 6))
        key = tuple(sorted((a, b))) if unordered else (a, b)
        groups.setdefault(key, []).append(int(i))
    return groups



def plot_overlaid(
    tag: str,
    t_1d: np.ndarray,
    noisy_1d: np.ndarray,
    recon_1d: np.ndarray,
    clean_1d: np.ndarray,
    out_dir: str,
    title: str = "",
    ext: str = "pdf",
    dpi: int = 150,
    lw_noisy: float = 1.0,
    lw_recon: float = 1.0,
    lw_clean: float = 1.0,
    plot_type: str = "all",
    fig_w: float = 12.0,
    fig_h: float = 3.6,
    line_scale: float = 1.0,
    font_scale: float = 1.0,
    tight: bool = False,
):
    lw_noisy *= line_scale; lw_recon *= line_scale; lw_clean *= line_scale
    with plt.rc_context({
        "font.size":        10 * font_scale,
        "axes.titlesize":   12 * font_scale,
        "axes.labelsize":   11 * font_scale,
        "xtick.labelsize":   9 * font_scale,
        "ytick.labelsize":   9 * font_scale,
        "legend.fontsize":   9 * font_scale,
    }):
        plt.figure(figsize=(fig_w, fig_h))
        if plot_type in ("all", "noisy_recon"):
            plt.plot(t_1d, noisy_1d, label="Noisy", alpha=0.6, linewidth=lw_noisy)
        if plot_type in ("all", "clean_recon", "noisy_recon"):
            plt.plot(t_1d, recon_1d, label="Recon", alpha=0.8, linewidth=lw_recon)
        if plot_type in ("all", "clean_recon"):
            plt.plot(t_1d, clean_1d, label="Clean", alpha=1.0, linewidth=lw_clean)

        if title: plt.title(title)
        plt.xlabel("Time (s) — merger at t=0")
        plt.ylabel("Strain")
        plt.legend(frameon=False)
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
        plt.savefig(os.path.join(out_dir, f"{tag}.{ext}"), **save_kwargs)
        plt.close()


@torch.no_grad()
def compute_mae_per_combo(ds: InferenceDataset, model, diffusion, t_pick: int, device, groups, outdir) -> dict:
    """
    Returns dict: {(m1,m2): {"mae": float, "count": int}} and also saves a CSV.
    MAE is averaged across all samples belonging to that (m1, m2) combo.
    """
    combo_stats = {}
    os.makedirs(outdir, exist_ok=True)

    for combo_idx, (combo, idx_list) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
        maes = []
        for idx in idx_list:
            item = ds[idx]
            clean_norm = item["clean_norm"].to(device)
            clean_raw  = item["clean_raw"].to(device)
            sigma      = item["sigma"].to(device)

            # one-step reconstruction
            x0_hat_raw = one_step_recon(model, diffusion, clean_norm, sigma, t_pick)

            # MAE over the 1D waveform; then aggregate over samples
            err = torch.mean(torch.abs(x0_hat_raw - clean_raw)).item()
            maes.append(err)

        combo_stats[combo] = {"mae": float(np.mean(maes) if maes else np.nan),
                              "count": int(len(maes))}
        print(f"[metric] combo {combo_idx+1}/{len(groups)} {combo}: "
              f"n={len(maes)}, MAE={combo_stats[combo]['mae']:.6e}")

    # write CSV mapping
    csv_path = os.path.join(outdir, "combo_mae.csv")
    with open(csv_path, "w") as fh:
        fh.write("index,m1,m2,count,mae\n")
        for i, (combo, stats) in enumerate(sorted(combo_stats.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
            m1, m2 = combo
            fh.write(f"{i},{m1},{m2},{stats['count']},{stats['mae']}\n")
    print(f"[metric] wrote: {csv_path}")

    return combo_stats

def save_mass_grid_heatmap(combo_stats: dict, outdir: str, ext: str = "png", dpi: int = 150, tight: bool = True):
    combos = list(combo_stats.keys())
    if not combos:
        print("[metric] No combos to plot.")
        return

    m1_vals = sorted({m1 for (m1, m2) in combos})       # build axis from combos
    m2_vals = sorted({m2 for (m1, m2) in combos})

    m1_to_col = {v: i for i, v in enumerate(m1_vals)}
    m2_to_row = {v: i for i, v in enumerate(m2_vals)}

    grid = np.full((len(m2_vals), len(m1_vals)), np.nan, dtype=np.float64)

    for (m1, m2), stats in combo_stats.items():
        r = m2_to_row[m2]
        c = m1_to_col[m1]
        grid[r, c] = stats["mae"]

    map_csv = os.path.join(outdir, "matrix_index_map.csv")
    with open(map_csv, "w") as fh:
        fh.write("axis,index,mass\n")
        for i, v in enumerate(m1_vals):
            fh.write(f"x,{i},{v}\n")
        for i, v in enumerate(m2_vals):
            fh.write(f"y,{i},{v}\n")
    print(f"[metric] wrote: {map_csv}")

    masked = np.ma.masked_invalid(grid)     # plot heatmap with NaNs masked
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')

    fig_w = min(20, max(6, 0.45 * len(m1_vals) + 3))
    fig_h = min(20, max(6, 0.45 * len(m2_vals) + 3))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(masked, interpolation="nearest", aspect="auto", cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label("MAE (|recon - clean|)")

    plt.title("Mean Absolute Error by Mass Combo")
    plt.xlabel("m1")
    plt.ylabel("m2")

    plt.xticks(range(len(m1_vals)), [str(v) for v in m1_vals], rotation=45, ha="right")
    plt.yticks(range(len(m2_vals)), [str(v) for v in m2_vals])

    os.makedirs(outdir, exist_ok=True)
    save_kwargs = {"dpi": dpi}
    if tight:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    out_path = os.path.join(outdir, f"mae_mass_grid.{ext}")
    plt.savefig(out_path, **save_kwargs)
    plt.close()
    print(f"[metric] wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot overlays per mass combo with t=0 at merger, or compute MAE matrix.")
    ap.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset")
    ap.add_argument("--model", type=str, required=True, help="Path to trained diffusion checkpoint (.pth)")
    ap.add_argument("--outdir", type=str, default="plots_by_mass", help="Where to save figures")

    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    ap.add_argument("--examples-per-combo", type=int, default=3)
    ap.add_argument("--target-snr", type=float, default=20.0)

    # plotting-only args
    ap.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--plot-type", type=str, default="all", choices=["all", "clean_recon", "noisy_recon"])
    ap.add_argument("--fig-width", type=float, default=12.0)
    ap.add_argument("--fig-height", type=float, default=3.6)
    ap.add_argument("--line-scale", type=float, default=1.0)
    ap.add_argument("--font-scale", type=float, default=1.0)
    ap.add_argument("--tight", action="store_true")
    ap.add_argument("--unordered-pairs", action="store_true")

    # NEW: metric / matrix
    ap.add_argument("--mass-combo-matrix", action="store_true",
                    help="If set, compute MAE per mass combo and save a mass-grid heatmap instead of overlays.")
    ap.add_argument("--matrix-ext", type=str, default="png", choices=["png", "pdf"],
                    help="Image format for the MAE mass-grid heatmap")

    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device = {device}")

    # mass groups (pass unordered=True if (m1,m2) normalized to (min,max))
    groups = group_indices_by_mass(args.data, unordered=args.unordered_pairs)
    print(f"[info] found {len(groups)} mass combinations "
          f"({'unordered' if args.unordered_pairs else 'ordered'}).")

    # load model + diffusion
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
    print(f"[info] target SNR={args.target_snr:.1f} → t={t_pick} (actual SNR≈{snr_at_t:.2f})")

    # dataset
    ds = InferenceDataset(args.data)

    if args.mass_combo_matrix:
        # metric path
        out_metrics = os.path.join(args.outdir, "metrics")
        os.makedirs(out_metrics, exist_ok=True)

        combo_stats = compute_mae_per_combo(
            ds=ds, model=model, diffusion=diffusion, t_pick=t_pick,
            device=device, groups=groups, outdir=out_metrics
        )
        save_mass_grid_heatmap(combo_stats, outdir=out_metrics, ext=args.matrix_ext, dpi=args.dpi, tight=True)
        ds.close()
        print(f"[complete] metrics written under: {out_metrics}")
        return

    def sort_key(key):
        m1, m2 = key
        if isinstance(m1, str):
            return (float("inf"), float("inf"))
        return (m1, m2)

    for combo_idx, (combo, idx_list) in enumerate(sorted(groups.items(), key=lambda kv: sort_key(kv[0]))):
        m1, m2 = combo
        tag_combo = f"m1_{m1}_m2_{m2}" if not isinstance(m1, str) else "all"

        pick_idxs = even_pick(idx_list, args.examples_per_combo)
        print(f"[combo {combo_idx+1}/{len(groups)}] {tag_combo}: {len(idx_list)} samples → plotting {len(pick_idxs)}")

        out_dir_combo = os.path.join(args.outdir, tag_combo)
        os.makedirs(out_dir_combo, exist_ok=True)

        for j, idx in enumerate(pick_idxs):
            item = ds[idx]
            clean_norm = item["clean_norm"].to(device)
            clean_raw  = item["clean_raw"].to(device)
            noisy_raw  = item["noisy_raw"].to(device)
            sigma      = item["sigma"].to(device)
            t_rel      = item["times_rel"]  # numpy (L,)

            # one-step recon
            x0_hat_raw = one_step_recon(model, diffusion, clean_norm, sigma, t_pick)

            clean_1d = clean_raw[0].detach().cpu().numpy().ravel()
            noisy_1d = noisy_raw[0].detach().cpu().numpy().ravel()
            recon_1d = x0_hat_raw[0].detach().cpu().numpy().ravel()

            title = f"(m1={m1}, m2={m2})  idx={idx}  t={t_pick}  SNR≈{snr_at_t:.1f}"
            tag   = f"{tag_combo}_ex{j}_idx{idx}"

            plot_overlaid(
                tag=tag,
                t_1d=t_rel,
                noisy_1d=noisy_1d,
                recon_1d=recon_1d,
                clean_1d=clean_1d,
                out_dir=out_dir_combo,
                title=title,
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
