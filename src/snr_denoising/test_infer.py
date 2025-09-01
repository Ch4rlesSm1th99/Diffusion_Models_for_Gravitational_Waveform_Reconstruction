import os
import re
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from models import UNet1D, CustomDiffusion


def _slugify_title(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


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
        Return seconds-relative time with t=0 at the local merger.
        NOTE: This assumes the requested L matches the array used to locate the peak.
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

        # Fallback: build a time vector from delta_t and center on the peak index (of the signal).
        s = self.signal[idx]
        pk = int(np.argmax(np.abs(s)))
        base = np.arange(L, dtype=np.float64) * self.delta_t
        pk_clamped = min(max(pk, 0), L - 1)
        return base - base[pk_clamped]

    def __getitem__(self, idx):
        clean_raw = torch.from_numpy(self.signal[idx]).float().unsqueeze(0)
        noisy_raw = torch.from_numpy(self.noisy[idx]).float().unsqueeze(0)

        s = noisy_raw.std()     # per-sample sigma from noisy
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
            "clean_norm": clean_norm,  # [1, Lc]
            "clean_raw":  clean_raw,   # [1, Lc]
            "noisy_raw":  noisy_raw,   # [1, Ln]
            "times_rel":  None,        # legacy; not relied upon when lengths differ
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


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Force 1D + 2D tensors to [B,1,L].
      [L]   -> [1,1,L]
      [1,L] -> [1,1,L]
      [B,L] -> [B,1,L]
    """
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    return x


def _peak_idx(y: np.ndarray) -> int:
    return int(np.argmax(np.abs(y)))


def _center_slice(L: int, center: int, out_len: int) -> slice:
    half = out_len // 2
    start = center - half
    end = start + out_len
    if start < 0:
        start = 0
        end = out_len
    if end > L:
        end = L
        start = L - out_len
    start = max(0, start)
    end = min(L, end)
    return slice(start, end)


def align_for_overlay(noisy: np.ndarray, recon: np.ndarray, clean: np.ndarray, delta_t: float):
    """
    Center-crop each array around its own peak so that all three share the same length Lmin.
    Build a symmetric time axis of length Lmin centered at 0.
    Returns noisy_a, recon_a, clean_a, t_axis
    """
    Ln, Lr, Lc = len(noisy), len(recon), len(clean)
    Lmin = min(Ln, Lr, Lc)

    sn = _center_slice(Ln, _peak_idx(noisy), Lmin)
    sr = _center_slice(Lr, _peak_idx(recon), Lmin)
    sc = _center_slice(Lc, _peak_idx(clean), Lmin)

    noisy_a = noisy[sn]
    recon_a = recon[sr]
    clean_a = clean[sc]

    t = np.arange(Lmin, dtype=np.float64) * delta_t
    t = t - t[Lmin // 2]
    return noisy_a, recon_a, clean_a, t


def align_for_metrics(clean: np.ndarray, recon: np.ndarray, delta_t: float):
    """
    Like align_for_overlay but only for (clean, recon).
    """
    Lr, Lc = len(recon), len(clean)
    Lmin = min(Lr, Lc)

    sr = _center_slice(Lr, _peak_idx(recon), Lmin)
    sc = _center_slice(Lc, _peak_idx(clean), Lmin)

    recon_a = recon[sr]
    clean_a = clean[sc]

    t = np.arange(Lmin, dtype=np.float64) * delta_t
    t = t - t[Lmin // 2]
    return clean_a, recon_a, t


# ---------- xcorr-based alignment helpers ----------
def _overlap_slices_for_lag(n_a: int, n_b: int, k: int):
    """
    For arrays a (length n_a) and b (length n_b), with lag k meaning
    'shift b to the RIGHT by k' (k>0), return slices that produce
    overlapping regions a[a0:a1], b[b0:b1] of equal length.
    """
    start = max(0, -k)
    stop  = min(n_a, n_b - k)
    if stop <= start:
        return slice(0, 0), slice(0, 0)
    return slice(start, stop), slice(start + k, stop + k)


def _best_lag_by_xcorr(a: np.ndarray, b: np.ndarray, max_shift: int) -> int:
    """Return integer lag k (k>0 means shift b to the RIGHT) maximizing dot within ±max_shift."""
    if max_shift <= 0:
        # full-length search
        max_shift = min(len(a), len(b)) - 1
    best_k, best_val = 0, -np.inf
    L = min(len(a), len(b))
    a = a[:L]; b = b[:L]
    for k in range(-max_shift, max_shift + 1):
        if k < 0:   # b left
            v = np.dot(a[-k:], b[:L + k])
        elif k > 0: # b right
            v = np.dot(a[:L - k], b[k:])
        else:
            v = np.dot(a, b)
        if v > best_val:
            best_val, best_k = v, k
    return best_k


def align_for_overlay_xcorr(noisy: np.ndarray,
                            recon: np.ndarray,
                            clean: np.ndarray,
                            delta_t: float,
                            max_shift: int = 0):
    """
    Align RECON to CLEAN by cross-correlation, then crop NOISY to the same
    overlapped segment. Build a time axis whose t=0 is the CLEAN peak.
    """
    k = _best_lag_by_xcorr(clean, recon, max_shift)
    s_clean, s_recon = _overlap_slices_for_lag(len(clean), len(recon), k)
    clean_a = clean[s_clean]
    recon_a = recon[s_recon]
    noisy_a = noisy[s_clean]  # share clean's slice

    L = len(clean_a)
    if L == 0:
        # Fallback to peak-centered cropping if overlap is empty
        return align_for_overlay(noisy, recon, clean, delta_t)

    pk_c = int(np.argmax(np.abs(clean_a)))
    t = np.arange(L, dtype=np.float64) * delta_t
    t -= t[pk_c]
    return noisy_a, recon_a, clean_a, t


def align_for_metrics_xcorr(clean: np.ndarray,
                            recon: np.ndarray,
                            delta_t: float,
                            max_shift: int = 0):
    """
    Align RECON to CLEAN by cross-correlation and return (clean_a, recon_a, t).
    """
    k = _best_lag_by_xcorr(clean, recon, max_shift)
    s_clean, s_recon = _overlap_slices_for_lag(len(clean), len(recon), k)
    clean_a = clean[s_clean]
    recon_a = recon[s_recon]

    L = len(clean_a)
    if L == 0:
        return align_for_metrics(clean, recon, delta_t)

    pk_c = int(np.argmax(np.abs(clean_a)))
    t = np.arange(L, dtype=np.float64) * delta_t
    t -= t[pk_c]
    return clean_a, recon_a, t
# ----------------------------------------------------


@torch.no_grad()
def one_step_recon_uncond(model, diffusion, clean_norm: torch.Tensor, sigma: torch.Tensor, t_scalar: int) -> torch.Tensor:
    """
    Unconditional:
      x0_hat_norm = (x_t - sqrt(1 - alpha_bar[t]) * eps_hat) / sqrt(alpha_bar[t])
      x0_hat_raw  = x0_hat_norm * sigma
    """
    clean_norm = _ensure_3d(clean_norm)
    B = clean_norm.shape[0]
    device = clean_norm.device
    t = torch.full((B,), int(t_scalar), dtype=torch.long, device=device)
    x_t, _ = diffusion.q_sample(clean_norm, t)
    eps_hat = model(x_t, t)
    ab = diffusion.alpha_bar[t_scalar]
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
    sigma_311 = sigma.reshape(1, 1, 1).to(device)
    return x0_hat_norm * sigma_311


@torch.no_grad()
def one_step_recon_cond(model, diffusion,
                        clean_norm: torch.Tensor,
                        cond_norm: torch.Tensor,
                        sigma: torch.Tensor,
                        t_scalar: int) -> torch.Tensor:
    """
    Conditional proxy (matches training diagnostics):
      - Diffuse the CLEAN to x_t
      - Condition the net on the NOISY measurement (normalized)
      - Reconstruct x0 via DDPM formula (one-step proxy)
    """
    clean_norm = _ensure_3d(clean_norm)
    cond_norm  = _ensure_3d(cond_norm)

    B = clean_norm.shape[0]
    device = clean_norm.device
    t = torch.full((B,), int(t_scalar), dtype=torch.long, device=device)

    x_t, _ = diffusion.q_sample(clean_norm, t)
    net_in = torch.cat([x_t, cond_norm], dim=1)
    eps_hat = model(net_in, t)
    ab = diffusion.alpha_bar[t_scalar]
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)

    sigma_311 = sigma.reshape(1, 1, 1).to(device)
    return x0_hat_norm * sigma_311


def even_pick(indices: list, k: int) -> list:
    """Pick up to k indices evenly spaced from a list, whilst preserving order."""
    n = len(indices)
    if n <= k:
        return indices
    picks = np.linspace(0, n - 1, k)
    picks = np.unique(np.floor(picks).astype(int)).tolist()
    while len(picks) < k:
        picks.append(picks[-1])
    return [indices[i] for i in picks[:k]]


def group_indices_by_mass(h5_path: str, unordered: bool = False):
    """Return dict using label_m1/label_m2 if present, else mass1/mass2."""
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
    global_title: str = "",
    filename_suffix: str = "",
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
    lw_noisy *= line_scale
    lw_recon *= line_scale
    lw_clean *= line_scale

    # which to plot
    show_noisy = plot_type in ("all", "noisy_recon")
    show_recon = plot_type in ("all", "clean_recon", "noisy_recon")
    show_clean = plot_type in ("all", "clean_recon")

    # if clean is plotted, make recon slightly thicker but behind it
    if show_recon and show_clean:
        lw_recon = max(lw_recon, lw_clean * 1.15)

    with plt.rc_context({
        "font.size":        10 * font_scale,
        "axes.titlesize":   12 * font_scale,
        "axes.labelsize":   11 * font_scale,
        "xtick.labelsize":   9 * font_scale,
        "ytick.labelsize":   9 * font_scale,
        "legend.fontsize":   9 * font_scale,
    }):
        plt.figure(figsize=(fig_w, fig_h))

        if show_noisy:
            plt.plot(t_1d, noisy_1d, label="Noisy", alpha=0.6, linewidth=lw_noisy, zorder=0)
        if show_recon:
            plt.plot(t_1d, recon_1d, label="Recon", alpha=0.8, linewidth=lw_recon, zorder=1)
        if show_clean:
            plt.plot(t_1d, clean_1d, label="Clean", alpha=1.0, linewidth=lw_clean, zorder=2)

        full_title = title
        if global_title:
            full_title = f"{global_title} — {title}" if title else global_title
        if full_title:
            plt.title(full_title)

        plt.xlabel("Time (s) — merger at t=0")
        plt.ylabel("Strain")
        plt.legend(frameon=False)
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        plt.savefig(os.path.join(out_dir, f"{tag}{suffix}.{ext}"), **save_kwargs)
        plt.close()


def _combo_sort_key(combo):
    m1, m2 = combo
    if isinstance(m1, str):
        return (float("inf"), float("inf"))
    return (m1, m2)


@torch.no_grad()
def compute_mae_per_combo(
    ds: InferenceDataset,
    model,
    diffusion,
    t_pick: int,
    device,
    groups,
    outdir,
    metric: str = "mae",
    win_before_ms: float = 0.0,
    win_after_ms: float = 0.0,
    per_combo_max_samples: int = 0,
    conditional: bool = False,
    metric_align: str = "xcorr",
    xcorr_max_shift: int = 0,
) -> dict:
    """
    Returns dict: {(m1,m2): {"score": float, "count": int}} and also saves a CSV.
    metric:
      - "mae"         : mean(|recon - clean|)
      - "nmae_clean"  : MAE / mean(|clean|)
      - "nmae_sigma"  : MAE / sigma         (per-sample noisy std)
    Optional windowing around merger event at t=0 using [ -window_before_ms, +window_after_ms ].
    metric_align: 'peak' (peak-center both) or 'xcorr' (xcorr-align recon to clean).
    xcorr_max_shift: half-window for lag search; 0 => full-length search.
    """
    eps = 1e-12
    use_window = (win_before_ms > 0.0) or (win_after_ms > 0.0)
    lb_s = -win_before_ms / 1000.0
    ub_s =  win_after_ms / 1000.0

    combo_stats = {}
    os.makedirs(outdir, exist_ok=True)

    ordered = sorted(groups.items(), key=lambda kv: _combo_sort_key(kv[0]))
    for combo_idx, (combo, idx_list) in enumerate(ordered):
        if per_combo_max_samples and per_combo_max_samples > 0:
            use_indices = even_pick(idx_list, per_combo_max_samples)
        else:
            use_indices = idx_list

        scores = []
        for idx in use_indices:
            item = ds[idx]
            clean_norm = item["clean_norm"].to(device)
            clean_raw  = item["clean_raw"].to(device)
            noisy_raw  = item["noisy_raw"].to(device)
            sigma      = item["sigma"]

            # one-step reconstruction (conditional vs uncond)
            if conditional:
                cond_norm = (noisy_raw.to(device) / sigma.to(device).view(1, 1))   # [1,L] -> forced to [1,1,L]
                x0_hat_raw = one_step_recon_cond(model, diffusion, clean_norm, cond_norm, sigma.to(device), t_pick)
            else:
                x0_hat_raw = one_step_recon_uncond(model, diffusion, clean_norm, sigma.to(device), t_pick)

            clean_1d = clean_raw[0].detach().cpu().numpy().ravel()
            recon_1d = x0_hat_raw[0].detach().cpu().numpy().ravel()

            # alignment choice for metric
            if metric_align == "xcorr":
                clean_a, recon_a, t_axis = align_for_metrics_xcorr(
                    clean_1d, recon_1d, ds.delta_t, max_shift=xcorr_max_shift
                )
            else:
                clean_a, recon_a, t_axis = align_for_metrics(clean_1d, recon_1d, ds.delta_t)

            if use_window:
                mask = (t_axis >= lb_s) & (t_axis <= ub_s)
            else:
                mask = slice(None)

            mae = float(np.mean(np.abs(recon_a[mask] - clean_a[mask])))

            if metric == "mae":
                score = mae
            elif metric == "nmae_clean":
                denom = float(np.mean(np.abs(clean_a[mask]))) + eps
                score = mae / denom
            elif metric == "nmae_sigma":
                denom = float(sigma.item()) + eps
                score = mae / denom
            else:
                raise ValueError(f"unknown metric: {metric}")

            scores.append(score)

        combo_stats[combo] = {"score": float(np.mean(scores) if scores else np.nan),
                              "count": int(len(scores))}
        print(f"[metric] combo {combo_idx+1}/{len(groups)} {combo}: "
              f"n={len(scores)}, {metric}={combo_stats[combo]['score']:.6e}")

    csv_path = os.path.join(outdir, "combo_scores.csv")
    with open(csv_path, "w") as fh:
        fh.write("index,m1,m2,count,metric,score\n")
        for i, (combo, _idxs) in enumerate(ordered):
            m1, m2 = combo
            fh.write(f"{i},{m1},{m2},{combo_stats[combo]['count']},{metric},{combo_stats[combo]['score']}\n")
    print(f"[metric] wrote: {csv_path}")

    return combo_stats


def _axes_from_h5(h5_path: str):
    """
    Build expected unique m1 and m2 axes (sorted) from the file labels.
    Returns (m1_vals, m2_vals) or (None, None) if labels are missing.
    """
    with h5py.File(h5_path, "r") as f:
        if "label_m1" in f and "label_m2" in f:
            m1_vals = sorted(set(np.array(f["label_m1"]).astype(float).tolist()))
            m2_vals = sorted(set(np.array(f["label_m2"]).astype(float).tolist()))
            return m1_vals, m2_vals
        if "mass1" in f and "mass2" in f:
            m1_vals = sorted(set(np.array(f["mass1"]).astype(float).tolist()))
            m2_vals = sorted(set(np.array(f["mass2"]).astype(float).tolist()))
            return m1_vals, m2_vals
    return None, None


def save_mass_grid_heatmap(
    combo_stats: dict,
    outdir: str,
    ext: str = "png",
    dpi: int = 150,
    tight: bool = True,
    m1_axis_expected=None,
    m2_axis_expected=None,
    title: str = "",
    filename_suffix: str = "",
    cbar_label: str = "MAE",
    basename: str = "metric_mass_grid",
):
    """
    Build a grid with:
      rows  = sorted unique m1 values (use expected axes if provided)
      cols  = sorted unique m2 values (use expected axes if provided)
      cell(m1, m2) = score for that combo (MAE or NMAE).
    Missing combos remain NaN. Writes mapping CSV + missing_pairs.csv log.
    """
    combos = list(combo_stats.keys())
    if not combos:
        print("[metric] No combos to plot.")
        return

    if m1_axis_expected is not None and m2_axis_expected is not None:
        m1_vals = list(m1_axis_expected)
        m2_vals = list(m2_axis_expected)
    else:
        m1_vals = sorted({m1 for (m1, m2) in combos})
        m2_vals = sorted({m2 for (m1, m2) in combos})

    m1_to_row = {v: i for i, v in enumerate(m1_vals)}
    m2_to_col = {v: i for i, v in enumerate(m2_vals)}

    grid = np.full((len(m1_vals), len(m2_vals)), np.nan, dtype=np.float64)
    for (m1, m2), stats in combo_stats.items():
        if (m1 not in m1_to_row) or (m2 not in m2_to_col):
            continue
        r = m1_to_row[m1]
        c = m2_to_col[m2]
        grid[r, c] = stats["score"]

    map_csv = os.path.join(outdir, "matrix_index_map.csv")
    with open(map_csv, "w") as fh:
        fh.write("axis,index,mass\n")
        for i, v in enumerate(m2_vals):
            fh.write(f"x,{i},{v}\n")
        for i, v in enumerate(m1_vals):
            fh.write(f"y,{i},{v}\n")
    print(f"[metric] wrote: {map_csv}")

    # list missing unordered pairs (m2 <= m1)
    expected = {(m1, m2) for m1 in m1_vals for m2 in m2_vals if m2 <= m1}
    present  = set(combo_stats.keys())
    missing  = sorted(list(expected - present), key=lambda t: (t[0], t[1]))
    miss_csv = os.path.join(outdir, "missing_pairs.csv")
    with open(miss_csv, "w") as fh:
        fh.write("m1,m2\n")
        for (m1, m2) in missing:
            fh.write(f"{m1},{m2}\n")
    print(f"[metric] wrote: {miss_csv}  (missing={len(missing)})")

    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')

    fig_w = min(20, max(6, 0.45 * len(m2_vals) + 3))
    fig_h = min(20, max(6, 0.45 * len(m1_vals) + 3))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(masked, interpolation="nearest", aspect="auto", cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    plt.title(title if title else "Error by Mass Combo (rows=m1, cols=m2)")
    plt.xlabel("m2")
    plt.ylabel("m1")

    plt.xticks(range(len(m2_vals)), [str(v) for v in m2_vals], rotation=45, ha="right")
    plt.yticks(range(len(m1_vals)), [str(v) for v in m1_vals])

    os.makedirs(outdir, exist_ok=True)
    save_kwargs = {"dpi": dpi}
    if tight:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = os.path.join(outdir, f"{basename}{suffix}.{ext}")
    plt.savefig(out_path, **save_kwargs)
    plt.close()
    print(f"[metric] wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot overlays per mass combo with t=0 at merger, or compute MAE/NMAE matrix.")
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

    # titles
    ap.add_argument("--title", type=str, default="", help="Optional plot title (also appended to filenames).")

    # metric + matrix args
    ap.add_argument("--mass-combo-matrix", action="store_true",
                    help="If set, compute a per-mass-combo metric and save a mass-grid heatmap instead of overlays.")
    ap.add_argument("--matrix-ext", type=str, default="png", choices=["png", "pdf"],
                    help="Image format for the mass-grid heatmap")

    # metric options
    ap.add_argument("--metric", type=str, default="mae",
                    help="One or more metrics (comma-separated) from: mae, nmae_clean, nmae_sigma.")
    ap.add_argument("--window-before-ms", type=float, default=0.0,
                    help="If >0, restrict metric to times t >= -before_ms from merger (t=0).")
    ap.add_argument("--window-after-ms", type=float, default=0.0,
                    help="If >0, restrict metric to times t <= +after_ms from merger (t=0).")
    ap.add_argument("--per-combo-max-samples", type=int, default=0,
                    help="If >0, cap the number of examples used per (m1,m2) to this many, evenly spaced.")

    # xcorr alignment options (kept, defaults chosen to fix previous misalignment)
    ap.add_argument("--xcorr-window-samp", type=int, default=0,
                    help="Half-window (samples) for cross-corr lag search. 0 = full-length search.")
    ap.add_argument("--overlay-align", choices=["peak", "xcorr"], default="xcorr",
                    help="Alignment for overlays: 'peak' (center each) or 'xcorr' (shift recon to clean).")
    ap.add_argument("--metric-align", choices=["peak", "xcorr"], default="xcorr",
                    help="Alignment for metrics: 'peak' or 'xcorr'.")

    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    title_slug = _slugify_title(args.title)
    print(f"[info] device = {device}")

    metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
    valid = {"mae", "nmae_clean", "nmae_sigma"}
    for m in metrics:
        if m not in valid:
            raise ValueError(f"Unknown metric '{m}'. Choose from {sorted(valid)}")

    # mass groups (pass unordered=True if (m1,m2) normalized to (min,max))
    groups = group_indices_by_mass(args.data, unordered=args.unordered_pairs)
    print(f"[info] found {len(groups)} mass combinations "
          f"({'unordered' if args.unordered_pairs else 'ordered'}).")

    # load model + diffusion
    ckpt = torch.load(args.model, map_location=device)
    in_ch = ckpt.get("args", {}).get("in_ch", 1)
    is_conditional = bool(ckpt.get("args", {}).get("conditional", False) or (in_ch >= 2))

    model = UNet1D(
        in_ch=in_ch,
        base_ch=ckpt["args"]["base_ch"],
        time_dim=ckpt["args"]["time_dim"],
        depth=ckpt["args"]["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = CustomDiffusion(T=ckpt["args"]["T"], device=device)
    t_pick, snr_at_t = t_for_target_snr(diffusion, args.target_snr)
    print(f"[info] target SNR={args.target_snr:.1f} --> t={t_pick} (actual SNR≈{snr_at_t:.2f})")
    print(f"[info] checkpoint in_ch={in_ch} -> {'CONDITIONAL' if is_conditional else 'UNCONDITIONAL'} inference path")

    # dataset
    ds = InferenceDataset(args.data)

    if args.mass_combo_matrix:
        out_metrics = os.path.join(args.outdir, "metrics")
        os.makedirs(out_metrics, exist_ok=True)

        m1_axis, m2_axis = _axes_from_h5(args.data)

        lbl_map = {
            "mae":        "MAE (recon - clean)",
            "nmae_clean": "Normalized MAE (per mean(clean))",
            "nmae_sigma": "Normalized MAE (per noisy σ)",
        }

        win_suffix = ""
        if args.window_before_ms > 0 or args.window_after_ms > 0:
            win_suffix = f"_win_{int(args.window_before_ms)}ms_{int(args.window_after_ms)}ms"

        # run metrics
        for m in metrics:
            combo_stats = compute_mae_per_combo(
                ds=ds, model=model, diffusion=diffusion, t_pick=t_pick,
                device=device, groups=groups, outdir=out_metrics,
                metric=m,
                win_before_ms=args.window_before_ms,
                win_after_ms=args.window_after_ms,
                per_combo_max_samples=args.per_combo_max_samples,
                conditional=is_conditional,
                metric_align=args.metric_align,
                xcorr_max_shift=args.xcorr_window_samp,
            )

            cbar_label = lbl_map.get(m, m)
            if win_suffix:
                cbar_label += f" in [{-args.window_before_ms:.0f}ms, +{args.window_after_ms:.0f}ms]"

            file_suffix = "_".join([s for s in [title_slug, m + win_suffix] if s])

            save_mass_grid_heatmap(
                combo_stats,
                outdir=out_metrics,
                ext=args.matrix_ext,
                dpi=args.dpi,
                tight=True,
                m1_axis_expected=m1_axis,
                m2_axis_expected=m2_axis,
                title=args.title,
                filename_suffix=file_suffix,
                cbar_label=cbar_label,
                basename="metric_mass_grid",
            )

        ds.close()
        print(f"[complete] metrics written under: {out_metrics}")
        return

    # plotting overlays
    def sort_key(combo):
        return _combo_sort_key(combo)

    for combo_idx, (combo, idx_list) in enumerate(sorted(groups.items(), key=lambda kv: sort_key(kv[0]))):
        m1, m2 = combo
        tag_combo = f"m1_{m1}_m2_{m2}" if not isinstance(m1, str) else "all"

        pick_idxs = even_pick(idx_list, args.examples_per_combo)
        print(f"[combo {combo_idx+1}/{len(groups)}] {tag_combo}: {len(idx_list)} samples --> plotting {len(pick_idxs)}")

        out_dir_combo = os.path.join(args.outdir, tag_combo)
        os.makedirs(out_dir_combo, exist_ok=True)

        for j, idx in enumerate(pick_idxs):
            item = ds[idx]
            clean_norm = item["clean_norm"].to(device)
            clean_raw  = item["clean_raw"].to(device)
            noisy_raw  = item["noisy_raw"].to(device)
            sigma      = item["sigma"].to(device)

            # one-step recon (diagnostic proxy)
            if is_conditional:
                cond_norm = (noisy_raw / sigma.view(1, 1))  # [1,Ln]
                x0_hat_raw = one_step_recon_cond(model, diffusion, clean_norm, cond_norm, sigma, t_pick)
            else:
                x0_hat_raw = one_step_recon_uncond(model, diffusion, clean_norm, sigma, t_pick)

            clean_1d = clean_raw[0].detach().cpu().numpy().ravel()
            noisy_1d = noisy_raw[0].detach().cpu().numpy().ravel()
            recon_1d = x0_hat_raw[0].detach().cpu().numpy().ravel()

            # Align/crop to common length + build time axis (choose method)
            if args.overlay_align == "xcorr":
                noisy_a, recon_a, clean_a, t_axis = align_for_overlay_xcorr(
                    noisy_1d, recon_1d, clean_1d, ds.delta_t, max_shift=args.xcorr_window_samp
                )
            else:
                noisy_a, recon_a, clean_a, t_axis = align_for_overlay(noisy_1d, recon_1d, clean_1d, ds.delta_t)

            local_title = (f"(m1={m1}, m2={m2})  idx={idx}  t={t_pick}  "
                           f"SNR≈{snr_at_t:.1f}  [{'COND' if is_conditional else 'UNCOND'}]")
            tag         = f"{tag_combo}_ex{j}_idx{idx}"

            plot_overlaid(
                tag=tag,
                t_1d=t_axis,
                noisy_1d=noisy_a,
                recon_1d=recon_a,
                clean_1d=clean_a,
                out_dir=out_dir_combo,
                title=local_title,
                global_title=args.title,
                filename_suffix=title_slug,
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
