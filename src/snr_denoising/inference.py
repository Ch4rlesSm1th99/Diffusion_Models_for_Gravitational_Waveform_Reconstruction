import os
import argparse
import h5py
import numpy as np
import torch
import json
import math
from typing import Optional, Tuple, List
from models import UNet1D, CustomDiffusion

def _tail_mask(L: int, fs: float, secs: float = 0.8):
    t = np.arange(L) / fs
    return t >= (t.max() - secs)

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a*a).sum() * (b*b).sum()) + 1e-30
    return float(np.dot(a, b) / den)

def _score_last_window(x: np.ndarray, c: np.ndarray, fs: float, secs: float = 0.8):
    L = min(len(x), len(c))
    x = np.asarray(x[:L], dtype=np.float64)
    c = np.asarray(c[:L], dtype=np.float64)
    m = _tail_mask(L, fs, secs)
    mae = float(np.mean(np.abs(x[m] - c[m])))
    r   = _corr(x[m], c[m])
    return {"corr_last": r, "mae_last": mae}

def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    return x

def _mad_std(x: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.4826 * np.median(np.abs(x64 - np.median(x64))) + 1e-24

def _reduce_to_one_channel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.size(1) > 1:
        return x[:, :1, :]
    return x

def _stats(name: str, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    a64 = np.asarray(arr, dtype=np.float64)
    return (f"{name}: shape={a64.shape}, min={a64.min():.3e}, max={a64.max():.3e}, "
            f"mean={a64.mean():.3e}, std={a64.std():.3e})")

def _corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30
    return float((a * b).sum() / den)

# HDF5 / input loaders
def _load_measurement_from_h5(h5_path: str, index: int):
    """
    Returns (y, clean, fs, P_model, (fw, Pw), meta_dict).
    meta_dict keys (if present): 'mass1','mass2','spin1z','spin2z','q','chirp_mass','snr','epoch',...
    """
    meta = {}
    with h5py.File(h5_path, "r") as f:
        y = np.array(f["noisy"][index], dtype=np.float32)
        clean = np.array(f["signal"][index], dtype=np.float32) if "signal" in f else None
        fs = float(f.attrs.get("sampling_rate", 0.0)) or float(1.0 / f.attrs.get("delta_t", 1.0 / 4096.0))

        # optional PSDs
        P_model = None
        if "psd_model" in f:
            P_model = np.array(f["psd_model"][index], dtype=np.float64)
        elif "psd" in f:  # legacy name
            P_model = np.array(f["psd"][index], dtype=np.float64)
        fw = Pw = None
        if ("psd_welch" in f) and ("psd_welch_freqs" in f):
            Pw = np.array(f["psd_welch"][index], dtype=np.float64)
            fw = np.array(f["psd_welch_freqs"][index], dtype=np.float64)

        # metadata (only grab what exists)
        for k in ["mass1","mass2","spin1z","spin2z","q","chirp_mass","snr","epoch",
                  "label_m1","label_m2","label_s1","label_s2"]:
            if k in f:
                try:
                    meta[k] = float(np.array(f[k][index]).reshape(()))
                except Exception:
                    pass
    return y, clean, fs, P_model, (fw, Pw), meta

def _load_measurement_from_npy(npy_path: str, fs: float):
    y = np.load(npy_path).astype(np.float32).ravel()
    return y, None, fs, None, (None, None), {}

# meta stack (labels) builder
def _meta_to_stack(meta: dict, L: int, cond_in_ch: int, M_SCALE: float, Q_SCALE: float) -> Optional[np.ndarray]:
    """
    Build [C_meta, L] from metadata in the fixed order:
      m1, m2, s1, s2, q, chirp_mass
    Spins left as-is; masses & chirp_mass / M_SCALE; q / Q_SCALE.
    """
    C_needed = max(0, cond_in_ch - 1)
    if C_needed <= 0:
        return None

    def _tile(v: float): return np.full((L,), float(v), dtype=np.float32)

    m1 = _tile(meta.get("mass1", 0.0) / max(M_SCALE, 1e-9))
    m2 = _tile(meta.get("mass2", 0.0) / max(M_SCALE, 1e-9))
    s1 = _tile(meta.get("spin1z", 0.0))
    s2 = _tile(meta.get("spin2z", 0.0))
    qv = meta.get("q", 0.0)
    if not np.isfinite(qv): qv = 0.0
    q  = _tile(min(max(qv, 0.0), Q_SCALE) / max(Q_SCALE, 1e-9))
    mc = _tile(meta.get("chirp_mass", 0.0) / max(M_SCALE, 1e-9))

    base = [m1, m2, s1, s2, q, mc]
    arr = np.stack(base[:C_needed], axis=0) if C_needed > 0 else None
    if arr is not None and arr.shape[0] < C_needed:
        pad = np.zeros((C_needed - arr.shape[0], L), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr

# --------- sigma & whitening helpers ----------------------
def _pick_sigma(y: np.ndarray, mode: str, fixed: float) -> float:
    y64 = np.asarray(y, dtype=np.float64)
    if mode == "std":
        s = float(np.std(y64))
    elif mode == "mad":
        s = float(_mad_std(y64))
    elif mode == "fixed":
        s = float(fixed)
    else:
        raise ValueError(f"unknown sigma-mode: {mode}")
    return s

def _whiten_pair_train_like(y: np.ndarray, x: Optional[np.ndarray], fs: float) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    from numpy.fft import rfft, irfft
    L = len(y)
    y64 = y.astype(np.float64) - float(np.mean(y, dtype=np.float64))
    Y = rfft(y64)
    P = np.abs(Y) ** 2
    if P.size > 9:
        kernel = np.ones(9, dtype=np.float64) / 9.0
        P = np.convolve(P, kernel, mode="same")
    P = np.maximum(P, 1e-20)
    y_w = irfft(Y / np.sqrt(P), n=L).astype(np.float32)
    x_w = None
    if x is not None:
        X = rfft((x.astype(np.float64) - float(np.mean(x, dtype=np.float64))))
        x_w = irfft(X / np.sqrt(P), n=L).astype(np.float32)
    return y_w, x_w, P

def _dewhiten_train_like(sig: np.ndarray, P: np.ndarray) -> np.ndarray:
    from numpy.fft import rfft, irfft
    L = len(sig)
    Xw = rfft(sig)
    return irfft(Xw * np.sqrt(P + 1e-12), n=L)

def _whiten_pair_welch(y: np.ndarray, x: Optional[np.ndarray], fs: float) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[np.ndarray,np.ndarray]]:
    from scipy.signal import welch
    from numpy.fft import rfft, irfft, rfftfreq
    L = len(y)
    f, Pxx = welch(y, fs=fs, nperseg=min(4096, L))
    freqs = rfftfreq(L, 1 / fs)
    P = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])
    Y = rfft(y)
    y_w = irfft(Y / np.sqrt(P + 1e-12), n=L).astype(np.float32)
    x_w = None
    if x is not None:
        X = rfft(x)
        x_w = irfft(X / np.sqrt(P + 1e-12), n=L).astype(np.float32)
    return y_w, x_w, (freqs, P)

def _dewhiten_welch(sig: np.ndarray, freqs_P: Tuple[np.ndarray,np.ndarray], fs: float) -> np.ndarray:
    from numpy.fft import rfft, irfft
    freqs, P = freqs_P
    L = len(sig)
    Xw = rfft(sig)
    return irfft(Xw * np.sqrt(P + 1e-12), n=L)

def _interp_psd_for_length(P: np.ndarray, L_src: int, L_tgt: int, fs: float) -> np.ndarray:
    from numpy.fft import rfftfreq
    if L_src == (L_tgt // 2 + 1):
        return P.astype(np.float64)
    f_src = rfftfreq(L_src * 2 - 2, 1.0 / fs)
    f_tgt = rfftfreq(L_tgt, 1.0 / fs)
    return np.interp(f_tgt, f_src, P, left=P[0], right=P[-1]).astype(np.float64)

def _whiten_pair_model(y: np.ndarray, x: Optional[np.ndarray], P_model: np.ndarray, fs: float):
    from numpy.fft import rfft, irfft
    L = len(y)
    P = _interp_psd_for_length(np.asarray(P_model, dtype=np.float64), len(P_model), L, fs)
    Y = rfft(y.astype(np.float64))
    y_w = irfft(Y / np.sqrt(P + 1e-12), n=L).astype(np.float32)
    x_w = None
    if x is not None:
        X = rfft(x.astype(np.float64))
        x_w = irfft(X / np.sqrt(P + 1e-12), n=L).astype(np.float32)
    return y_w, x_w, P

def _dewhiten_model(sig: np.ndarray, P: np.ndarray) -> np.ndarray:
    from numpy.fft import rfft, irfft
    Xw = rfft(sig)
    return irfft(Xw * np.sqrt(P + 1e-12), n=len(sig))
# --------------------------- end sigma + whitening section --------------------------

#  SNR + scheduler helper funcs
def snr_from_alpha_bar(alpha_bar: torch.Tensor) -> np.ndarray:
    ab = alpha_bar.detach().cpu().numpy().clip(1e-12, 1 - 1e-12)
    return np.sqrt(ab / (1.0 - ab))

def t_for_target_snr(diffusion: CustomDiffusion, target_snr: float) -> int:
    snr = snr_from_alpha_bar(diffusion.alpha_bar)
    return int(np.argmin(np.abs(snr - float(target_snr))))

def _build_t_schedule(T: int, steps: int, device: torch.device, start_t: Optional[int]) -> torch.Tensor:
    if start_t is None:
        start_t = T - 1
    start_t = int(max(0, min(start_t, T - 1)))
    steps = int(max(1, min(steps, start_t + 1)))
    ts = torch.linspace(start_t, 0, steps, device=device).round().long()
    ts = torch.unique_consecutive(ts)
    if ts[0].item() != start_t:
        ts = torch.cat([torch.tensor([start_t], device=device), ts])
    if ts[-1].item() != 0:
        ts = torch.cat([ts, torch.tensor([0], device=device)])
    return ts

def _cfg_weight(i: int, N: int, mode: str, wmax: float, center: float, width: float) -> float:
    if N <= 1:
        s = 1.0
    else:
        s = i / (N - 1)
    mode = mode.lower()
    if mode == "const":
        return float(wmax)
    if mode == "tophat":
        lo, hi = center - width * 0.5, center + width * 0.5
        return float(wmax) if (s >= lo and s <= hi) else 1.0
    if mode == "gauss":
        sig = max(width, 1e-9)
        return float(wmax) * math.exp(-0.5 * ((s - center) / sig) ** 2)
    raise ValueError(f"unknown cfg-mode: {mode}")

# xcorr helpers
def _best_lag_by_xcorr(a: np.ndarray, b: np.ndarray, max_shift: int = 0) -> int:
    if max_shift <= 0:
        max_shift = min(len(a), len(b)) - 1
    best_k, best_val = 0, -np.inf
    L = min(len(a), len(b))
    a = a[:L]; b = b[:L]
    for k in range(-max_shift, max_shift + 1):
        if k < 0:
            v = float(np.dot(a[-k:], b[:L + k]))
        elif k > 0:
            v = float(np.dot(a[:L - k], b[k:]))
        else:
            v = float(np.dot(a, b))
        if v > best_val:
            best_val, best_k = v, k
    return best_k

def _align_xcorr(a: np.ndarray, b: np.ndarray, delta_t: float, max_shift: int = 0):
    k = _best_lag_by_xcorr(a, b, max_shift)
    start = max(0, -k)
    stop = min(len(a), len(b) - k)
    if stop <= start:
        L = min(len(a), len(b))
        a_al, b_al = a[:L], b[:L]
    else:
        a_al = a[start:stop]
        b_al = b[start + k:stop + k]
    L = len(a_al)
    t = np.arange(L, dtype=np.float64) * delta_t
    pk = int(np.argmax(np.abs(a_al)))
    t -= t[pk]
    return a_al, b_al, t

def _save_plot(t, y, xhat, clean, outpng, delta_t, sigma_scalar, xcorr_window_samp=0):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    plt.figure(figsize=(12, 3.2))
    if y is not None:
        plt.plot(t, y, label="measurement (noisy)", alpha=0.5, linewidth=1.0)
    plt.plot(t, xhat, label="reconstruction", linewidth=1.4)
    if clean is not None and len(clean) == len(xhat):
        plt.plot(t, clean, label="clean (gt)", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

    if clean is not None and len(clean) == len(xhat):
        clean_a, recon_a, t_a = _align_xcorr(clean, xhat, delta_t, max_shift=xcorr_window_samp)
        plt.figure(figsize=(12, 3.2))
        plt.plot(t_a, recon_a, label="recon (xcorr-aligned)", linewidth=1.4)
        plt.plot(t_a, clean_a, label="clean (gt)", linewidth=1.0)
        plt.xlabel("Time (s) - t=0 at clean peak")
        plt.ylabel("Strain")
        plt.legend(frameon=False)
        plt.tight_layout()
        base, ext = os.path.splitext(outpng)
        outpng_xc = base + "_xcorr" + ext
        plt.savefig(outpng_xc, dpi=150)
        plt.close()

        mask = (t_a >= -0.080) & (t_a <= 0.040)
        mae = float(np.mean(np.abs(recon_a[mask] - clean_a[mask])))
        nmae_clean = mae / (float(np.mean(np.abs(clean_a[mask]))) + 1e-12)
        nmae_sigma = mae / (float(sigma_scalar) + 1e-12)
        print(f"[scores(xcorr)win -80ms,+40ms] MAE={mae:.3e} NMAE_clean={nmae_clean:.3e} NMAE_sigma={nmae_sigma:.3e}")

# proxy (one-step)
@torch.no_grad()
def one_step_proxy_like_test_infer(model, diffusion,
                                   clean_norm: torch.Tensor,
                                   cond_stack: torch.Tensor,
                                   sigma_scalar: float,
                                   target_snr: float, device: torch.device,
                                   in_ch: int, cond_in_ch: int, use_selfcond: bool,
                                   cfg_scale: float, drop_y_only: bool,
                                   cond_scale: float = 1.0,
                                   eps_scale: float = 1.0,
                                   pred_type: str = "eps",
                                   amp: bool = False):
    ab = diffusion.alpha_bar.detach().cpu().numpy()
    snr_ts = np.sqrt(ab / (1 - ab))
    t_pick = int(np.argmin(np.abs(snr_ts - target_snr)))
    t = torch.full((1,), t_pick, dtype=torch.long, device=device)

    x_t, _ = diffusion.q_sample(clean_norm, t)
    x0_sc = torch.zeros_like(x_t) if use_selfcond else None

    y_channel = cond_stack[:, :1, :]
    meta_channels = cond_stack[:, 1:, :] if cond_stack.size(1) > 1 else None
    y_used = cond_scale * y_channel
    cond_used = torch.cat([y_used, meta_channels], dim=1) if meta_channels is not None else y_used

    def _make_in(x_t_, cond_, sc_):
        if use_selfcond:
            return torch.cat([x_t_, cond_, sc_], dim=1)
        else:
            return torch.cat([x_t_, cond_], dim=1)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        net_in_c = _make_in(x_t, cond_used, x0_sc)
        out_c = model(net_in_c, t)
        if cfg_scale != 1.0:
            if drop_y_only and (meta_channels is not None):
                cond_u = torch.cat([torch.zeros_like(y_used), meta_channels], dim=1)
            else:
                cond_u = torch.zeros_like(cond_used)
            net_in_u = _make_in(x_t, cond_u, x0_sc)
            out_u = model(net_in_u, t)
            out = out_u + cfg_scale * (out_c - out_u)
        else:
            out = out_c

    out = _reduce_to_one_channel(out)
    ab_t = diffusion.alpha_bar[t_pick]

    if pred_type == "eps":
        eps_hat = eps_scale * out
        x0_hat_norm = (x_t - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)
    else:
        x0_hat_norm = out

    return x0_hat_norm * torch.tensor(sigma_scalar, device=device).view(1, 1, 1)

# -------------- DDIM sampler -------------------
@torch.no_grad()
def ddim_sample(model, diffusion, cond_stack: torch.Tensor,
                T: int, steps: int, eta: float,
                device: torch.device, length: int, debug: bool,
                start_t: Optional[int], init_mode: str, x0_std_est: float,
                dc_weight: float, cond_scale: float, eps_scale: float, pred_type: str,
                in_ch: int, cond_in_ch: int, use_selfcond: bool, cfg_scale: float,
                cfg_mode: str, cfg_center: float, cfg_width: float, cfg_u_only_thresh: float,
                oracle_init: bool = False, clean_norm_311: Optional[torch.Tensor] = None,
                log_jsonl_path: Optional[str] = None, log_interval: int = 0,
                xcorr_window_samp: int = 0, delta_t: float = 1.0,
                amp: bool = False, drop_y_only: bool = True) -> torch.Tensor:

    if log_jsonl_path:
        os.makedirs(os.path.dirname(log_jsonl_path), exist_ok=True)
        def _log(obj: dict):
            with open(log_jsonl_path, "a") as fh:
                fh.write(json.dumps(obj) + "\n")
    else:
        def _log(_): pass

    y_chan = cond_stack[:, :1, :]
    meta = cond_stack[:, 1:, :] if cond_stack.size(1) > 1 else None

    t_schedule = _build_t_schedule(T=T, steps=steps, device=device, start_t=start_t)
    ab = diffusion.alpha_bar.to(device).clamp(1e-12, 1.0)
    ab_start = ab[int(t_schedule[0].item())]

    # init x_t
    if oracle_init and (clean_norm_311 is not None):
        t0 = int(t_schedule[0].item())
        x_t, _ = diffusion.q_sample(clean_norm_311, torch.tensor([t0], device=device, dtype=torch.long))
        if debug: print(f"[debug] oracle-init enabled (t0={t0})")
    else:
        if init_mode == "noise":
            x_t = torch.randn(1, 1, length, device=device)
        elif init_mode == "scaled-noise":
            std_init = torch.sqrt(ab_start * (x0_std_est ** 2) + (1 - ab_start))
            x_t = std_init * torch.randn(1, 1, length, device=device)
        elif init_mode == "y-blend":
            z = torch.randn(1, 1, length, device=device)
            x_t = torch.sqrt(ab_start) * y_chan + torch.sqrt(1 - ab_start) * z
        else:
            raise ValueError(f"unknown init_mode: {init_mode}")

    x0_sc = torch.zeros_like(x_t) if use_selfcond else None

    if debug:
        print(f"[debug] init_mode={init_mode}, x0_std_est={x0_std_est:.3f}, ab_start={ab_start.item():.6f}")
        print(f"[debug] schedule length={len(t_schedule)}, first={int(t_schedule[0])}, last={int(t_schedule[-1])}")
        print(_stats("x_T (init)", x_t))

    N = len(t_schedule)

    for i in range(N):
        t_now = int(t_schedule[i].item())
        ab_t = ab[t_now]
        ab_prev = ab[int(t_schedule[i + 1].item())] if i + 1 < N else torch.tensor(1.0, device=device)

        # scale *y only*; meta untouched
        y_used = cond_scale * y_chan
        cond_used = torch.cat([y_used, meta], dim=1) if meta is not None else y_used

        # scheduled CFG
        w_t = _cfg_weight(i=i, N=N, mode=cfg_mode, wmax=cfg_scale, center=cfg_center, width=cfg_width)

        def _make_in(x_t_, cond_, sc_):
            return torch.cat([x_t_, cond_, sc_], dim=1) if use_selfcond else torch.cat([x_t_, cond_], dim=1)

        out_c = None; out_u = None; have_u = False
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            if w_t <= cfg_u_only_thresh:
                cond_u = torch.cat([torch.zeros_like(y_used), meta], dim=1) if (drop_y_only and meta is not None) else torch.zeros_like(cond_used)
                net_in_u = _make_in(x_t, cond_u, x0_sc)
                out_u = model(net_in_u, torch.full((1,), t_now, dtype=torch.long, device=device))
                out = out_u; have_u = True
            elif abs(w_t - 1.0) <= 1e-6:
                net_in_c = _make_in(x_t, cond_used, x0_sc)
                out_c = model(net_in_c, torch.full((1,), t_now, dtype=torch.long, device=device))
                out = out_c
            else:
                net_in_c = _make_in(x_t, cond_used, x0_sc)
                out_c = model(net_in_c, torch.full((1,), t_now, dtype=torch.long, device=device))
                cond_u = torch.cat([torch.zeros_like(y_used), meta], dim=1) if (drop_y_only and meta is not None) else torch.zeros_like(cond_used)
                net_in_u = _make_in(x_t, cond_u, x0_sc)
                out_u = model(net_in_u, torch.full((1,), t_now, dtype=torch.long, device=device))
                out = out_u + w_t * (out_c - out_u); have_u = True

        out = _reduce_to_one_channel(out)

        if pred_type == "eps":
            eps_hat = eps_scale * out
            x0_hat_norm = (x_t - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        else:
            x0_hat_norm = out
            eps_hat = (x_t - torch.sqrt(ab_t) * x0_hat_norm) / torch.sqrt(torch.clamp(1 - ab_t, min=1e-12))

        if dc_weight > 0:
            x0_hat_norm = (1 - dc_weight) * x0_hat_norm + dc_weight * y_chan

        if use_selfcond:
            x0_sc = x0_hat_norm.detach()

        # DDIM update
        if t_now == 0:
            x_t = x0_hat_norm
        else:
            sigma_t = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma_t ** 2, min=0.0)) * eps_hat
            noise = sigma_t * torch.randn_like(x_t) if sigma_t.item() > 0 else 0.0
            x_t = torch.sqrt(ab_prev) * x0_hat_norm + dir_xt + noise

        if debug and (i % max(1, N // 5) == 0 or t_now == 0):
            print(_stats(f"x_t (t={t_now})", x_t))
            print(_stats(f"eps_hat (t={t_now})", eps_hat))

        # per-step log
        if log_jsonl_path and ((i % max(1, log_interval) == 0) or (i == N - 1)):
            # quick xcorr diag vs y
            xt_np = x_t.detach().cpu().numpy().reshape(-1)
            y_np  = y_chan.detach().cpu().numpy().reshape(-1)
            if len(xt_np) and len(y_np):
                win = min(len(xt_np) - 1, int(max(1.0, 0.25 / delta_t)))
                k = _best_lag_by_xcorr(xt_np, y_np, max_shift=win)
                if k < 0:
                    a = xt_np[-k:]; b = y_np[:len(xt_np) + k]
                elif k > 0:
                    a = xt_np[:len(xt_np) - k]; b = y_np[k:]
                else:
                    a = xt_np; b = y_np
                corr_lag = _corr_np(a, b)
            else:
                corr_lag = 0.0
            _log({
                "phase":"ddim_step","i":i,"t":t_now,"i_norm":float(0.0 if N<=1 else i/(N-1)),
                "alpha_bar":float(ab_t.item() if isinstance(ab_t, torch.Tensor) else ab_t),
                "cfg_mode":cfg_mode,"cfg_w_t":float(w_t),"cfg_scale":float(cfg_scale),
                "corr_lag":corr_lag
            })

    return x_t  # = x0_hat_norm


def main():
    ap = argparse.ArgumentParser("Conditional deployment sampler (+self-cond & scheduled CFG, y | meta)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-h5", type=str, help="HDF5 with 'noisy' (and optionally 'signal')")
    src.add_argument("--input-npy", type=str, help="1D npy array with the measurement")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str, required=True)

    # sampler controls
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--start-t", type=int, default=None)
    ap.add_argument("--start-snr", type=float, default=None)

    # init controls
    ap.add_argument("--init-mode", choices=["noise", "scaled-noise", "y-blend"], default="noise")
    ap.add_argument("--x0-std-est", type=float, default=0.14)
    ap.add_argument("--dc-weight", type=float, default=0.0)

    # sigma/conditioning
    ap.add_argument("--sigma-mode", choices=["std", "mad", "fixed"], default="std")
    ap.add_argument("--sigma-fixed", type=float, default=1.0)

    # whitening
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--whiten-mode", choices=["train", "welch", "model", "auto"], default="auto",
                    help="train: FFT+moving-average PSD; welch/model: use per-sample PSD if available; "
                         "auto prefers model->welch->train")

    # plotting / debug
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--xcorr-window-samp", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-jsonl", type=str, default=None, help="Write per-step metrics as JSONL")
    ap.add_argument("--log-interval", type=int, default=10, help="Log every N steps")

    # diagnostic proxy
    ap.add_argument("--one-step-proxy", action="store_true")
    ap.add_argument("--target-snr", type=float, default=20.0)

    # knobs
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--pred-type", choices=["eps", "x0"], default="eps")

    # CFG + self-conditioning
    ap.add_argument("--cfg-scale", type=float, default=1.5, help="Peak CFG weight (wmax). 1.0 = plain conditional")
    ap.add_argument("--cfg-mode", choices=["const", "tophat", "gauss"], default="const")
    ap.add_argument("--cfg-center", type=float, default=0.70)
    ap.add_argument("--cfg-width", type=float, default=0.12)
    ap.add_argument("--cfg-u-only-thresh", type=float, default=0.05)
    ap.add_argument("--selfcond-ema", type=float, default=0.9)

    # oracle init
    ap.add_argument("--oracle-init", action="store_true",
                    help="Start from q_sample(clean_norm, start_t) (requires clean in HDF5)")

    ap.add_argument("--run-tag", type=str, default=None)
    ap.add_argument("--score-secs", type=float, default=0.8)

    # AMP + EMA use
    ap.add_argument("--use-ema", action="store_true", help="If checkpoint has EMA, use it (recommended)")
    ap.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    ap.set_defaults(use_ema=True)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision inference on CUDA")

    args = ap.parse_args()

    # seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # load measurement (+ optional PSDs + meta)
    if args.input_h5:
        y_raw, clean_raw, fs, P_model_in, (fw_in, Pw_in), meta_dict = _load_measurement_from_h5(args.input_h5, args.index)
    else:
        y_raw, clean_raw, fs, P_model_in, (fw_in, Pw_in), meta_dict = _load_measurement_from_npy(args.input_npy, args.fs)

    L = len(y_raw)
    t = np.arange(L) / fs
    if args.debug:
        print(_stats("y_raw", y_raw))
        if clean_raw is not None:
            print(_stats("clean_raw", clean_raw))
        print(f"[debug] fs={fs}, L={L}, dt={1.0 / fs:.6e}")

    # load model + diffusion + scales
    ckpt = torch.load(args.model, map_location=device)
    ck_args = ckpt.get("args", {})
    in_ch = ck_args.get("in_ch", 3)
    cond_in_ch = ck_args.get("cond_in_ch", 1)
    base_ch = ck_args.get("base_ch", 64)
    time_dim = ck_args.get("time_dim", 128)
    depth = ck_args.get("depth", 3)
    T = ck_args.get("T", 1000)
    drop_y_only = bool(ck_args.get("dropout_y_only", True))
    use_selfcond = (in_ch == (1 + cond_in_ch + 1))

    # dataset-adaptive meta scales (from training), with safe fallbacks
    meta_scale = ck_args.get("meta_scale", {"M": 80.0, "q": 10.0})
    M_SCALE = float(meta_scale.get("M", 80.0))
    Q_SCALE = float(meta_scale.get("q", 10.0))

    model = UNet1D(
        in_ch=in_ch,
        base_ch=base_ch,
        time_dim=time_dim,
        depth=depth,
        t_embed_max_time=max(0, T - 1),
        cond_in_ch=cond_in_ch,
        use_selfcond=use_selfcond,
    ).to(device)

    state_loaded = False
    if args.use_ema and ("model_ema_state" in ckpt):
        try:
            model.load_state_dict(ckpt["model_ema_state"], strict=True)
            state_loaded = True
            print("[info] loaded EMA weights")
        except Exception as e:
            print(f"[warn] EMA load failed ({e}); falling back to raw weights]")
    if not state_loaded:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("[info] loaded raw weights")

    model.eval()
    diffusion = CustomDiffusion(T=T, device=device)

    # ---- whitening (model -> welch -> train for AUTO) ----
    P_train = None; freqs_P = None; P_model_used = None
    whiten_kind_used = "raw"
    if args.whiten:
        mode = args.whiten_mode
        if mode == "auto":
            if P_model_in is not None:
                y_for_cond, clean_for_cond, P_model_used = _whiten_pair_model(y_raw, clean_raw, P_model_in, fs)
                whiten_kind_used = "model"
            elif (fw_in is not None) and (Pw_in is not None):
                from numpy.fft import rfft, irfft, rfftfreq
                f_tgt = rfftfreq(L, 1.0 / fs)
                P = np.interp(f_tgt, fw_in, Pw_in, left=Pw_in[0], right=Pw_in[-1])
                Y = np.fft.rfft(y_raw.astype(np.float64))
                y_for_cond = np.fft.irfft(Y/np.sqrt(P+1e-12), n=L).astype(np.float32)
                if clean_raw is not None:
                    X = np.fft.rfft(clean_raw.astype(np.float64))
                    clean_for_cond = np.fft.irfft(X/np.sqrt(P+1e-12), n=L).astype(np.float32)
                else:
                    clean_for_cond = None
                freqs_P = (f_tgt, P)
                whiten_kind_used = "welch"
            else:
                y_for_cond, clean_for_cond, P_train = _whiten_pair_train_like(y_raw, clean_raw, fs)
                whiten_kind_used = "train"
        elif mode == "model":
            if P_model_in is None:
                y_for_cond, clean_for_cond, P_train = _whiten_pair_train_like(y_raw, clean_raw, fs)
                whiten_kind_used = "train"
            else:
                y_for_cond, clean_for_cond, P_model_used = _whiten_pair_model(y_raw, clean_raw, P_model_in, fs)
                whiten_kind_used = "model"
        elif mode == "welch":
            y_for_cond, clean_for_cond, freqs_P = _whiten_pair_welch(y_raw, clean_raw, fs)
            whiten_kind_used = "welch"
        else:  # 'train'
            y_for_cond, clean_for_cond, P_train = _whiten_pair_train_like(y_raw, clean_raw, fs)
            whiten_kind_used = "train"
        if args.debug:
            print(_stats(f"y_white({whiten_kind_used})", y_for_cond))
            if clean_for_cond is not None:
                print(_stats(f"clean_white({whiten_kind_used})", clean_for_cond))
    else:
        y_for_cond = y_raw
        clean_for_cond = clean_raw
        whiten_kind_used = "raw"

    # sigma in same domain as conditioning
    sigma = _pick_sigma(y_for_cond, args.sigma_mode, args.sigma_fixed)

    # fallback sigma if degenerate
    fallback = {"train": 2.914e-12, "welch": 2.914e-16, "model": 2.914e-16, "raw": 2.914e-12}
    try:
        with open(os.path.join(os.path.dirname(args.model), "fallback_sigma.json"), "r") as fh:
            fb = json.load(fh)
            for k in list(fallback.keys()):
                if k in fb and "median" in fb[k]:
                    fallback[k] = float(fb[k]["median"])
    except Exception:
        pass
    if (not np.isfinite(sigma)) or (sigma < 1e-20):
        sigma = fallback.get(whiten_kind_used, fallback["train"])
        print(f"[warn] sigma invalid/too small; using fallback={sigma:.3e} (mode={whiten_kind_used})")

    if args.debug:
        print(f"[debug] whitening resolved to: {whiten_kind_used}")
        print(f"[debug] sigma_mode={args.sigma_mode}, sigma={sigma:.3e}")
        print(f"[debug] SNR(y_for_cond)~mean(|y|)/sigma -> {np.mean(np.abs(y_for_cond)) / sigma:.3e}")

    # normalized y; meta uses dataset-adaptive fixed scales (not sigma)
    y_norm = (y_for_cond / sigma).astype(np.float32)
    y_norm_311 = torch.from_numpy(y_norm).to(device).view(1, 1, -1)
    clean_norm_311 = None
    if clean_for_cond is not None:
        clean_norm_311 = torch.from_numpy((clean_for_cond / sigma).astype(np.float32)).to(device).view(1, 1, -1)
    if args.debug:
        print(_stats("y_norm", y_norm))

    # build conditional stack
    if cond_in_ch <= 1:
        cond_stack = y_norm_311
    else:
        meta_arr = None
        if args.input_h5:
            try:
                meta_arr = _meta_to_stack(meta_dict, L=L, cond_in_ch=cond_in_ch, M_SCALE=M_SCALE, Q_SCALE=Q_SCALE)
            except Exception as e:
                print(f"[warn] meta stack build failed: {e}")
        if meta_arr is None:
            meta_arr = np.zeros((cond_in_ch - 1, L), dtype=np.float32)
        meta_311 = torch.from_numpy(meta_arr).to(device).unsqueeze(0)   # [1, C_meta, L]
        cond_stack = torch.cat([y_norm_311, meta_311], dim=1)

    # choose start_t
    start_t = t_for_target_snr(diffusion, args.start_snr) if (args.start_snr is not None) else args.start_t
    start_t_eff = (T - 1) if (start_t is None) else int(start_t)
    start_snr_eff = snr_from_alpha_bar(diffusion.alpha_bar)[start_t_eff]

    def _mk_tag(mode):
        auto = (f"{mode}_t{start_t_eff}_snr{start_snr_eff:.1f}"
                f"_steps{args.steps}_eta{args.eta}_cfg{args.cfg_scale}"
                f"_cfgmode-{args.cfg_mode}_ctr{args.cfg_center}_w{args.cfg_width}"
                f"_init-{args.init_mode}_pred-{args.pred_type}"
                f"_dc{args.dc_weight}_cond{args.cond_scale}_eps{args.eps_scale}"
                f"_{'white' if args.whiten else 'raw'}_{args.whiten_mode}_{args.sigma_mode}").replace('.', 'p')
        return (args.run_tag or auto)

    tag_iter = _mk_tag("iter")
    tag_proxy = _mk_tag("proxy")

    # estimate clean_norm std for scaled init
    x0_std_est = args.x0_std_est
    if clean_for_cond is not None:
        x0_std_est = float(np.std((clean_for_cond / sigma).astype(np.float32)))
    if args.debug:
        print(f"[debug] x0_std_est (used) = {x0_std_est:.6f}")

    print(f"[info] device={device} steps={args.steps}/{T} eta={args.eta} "
          f"sigma_mode={args.sigma_mode} whiten={'on' if args.whiten else 'off'} mode={args.whiten_mode} seed={args.seed}")
    print(f"[info] start_t={start_t_eff} (SNR≈{start_snr_eff:.2f}) init_mode={args.init_mode} "
          f"pred_type={args.pred_type} in_ch={in_ch} cond_in_ch={cond_in_ch} selfcond={use_selfcond} AMP={args.amp}")
    print(f"[info] CFG: mode={args.cfg_mode} wmax={args.cfg_scale} center={args.cfg_center} "
          f"width={args.cfg_width} u_only_thresh={args.cfg_u_only_thresh} drop_y_only={drop_y_only}")

    # iterative sampling (DDIM): **noising→denoising** schedule
    x0_hat_norm = ddim_sample(
        model=model,
        diffusion=diffusion,
        cond_stack=cond_stack,
        T=T,
        steps=args.steps,
        eta=args.eta,
        device=device,
        length=L,
        debug=args.debug,
        start_t=start_t,
        init_mode=args.init_mode,
        x0_std_est=x0_std_est,
        dc_weight=args.dc_weight,
        cond_scale=args.cond_scale,
        eps_scale=args.eps_scale,
        pred_type=args.pred_type,
        in_ch=in_ch,
        cond_in_ch=cond_in_ch,
        use_selfcond=use_selfcond,
        cfg_scale=args.cfg_scale,
        cfg_mode=args.cfg_mode,
        cfg_center=args.cfg_center,
        cfg_width=args.cfg_width,
        cfg_u_only_thresh=args.cfg_u_only_thresh,
        oracle_init=args.oracle_init,
        clean_norm_311=clean_norm_311,
        log_jsonl_path=args.log_jsonl,
        log_interval=args.log_interval,
        xcorr_window_samp=args.xcorr_window_samp,
        delta_t=1.0 / float(fs),
        amp=args.amp,
        drop_y_only=drop_y_only,
    )

    x0_hat_raw = (x0_hat_norm * torch.tensor(sigma, device=device).view(1, 1, 1)).detach().cpu().numpy().ravel()
    if args.debug:
        print(_stats("x0_hat_raw (iterative, whitened-domain)", x0_hat_raw))

    # de-whiten if used (mirror whitening path)
    if args.whiten:
        if whiten_kind_used == "train":
            x0_hat_raw = _dewhiten_train_like(x0_hat_raw, P_train)
        elif whiten_kind_used == "welch":
            x0_hat_raw = _dewhiten_welch(x0_hat_raw, freqs_P, fs)
        elif whiten_kind_used == "model":
            x0_hat_raw = _dewhiten_model(x0_hat_raw, P_model_used)

    # save tagged
    os.makedirs(args.outdir, exist_ok=True)
    recon_path  = os.path.join(args.outdir, f"reconstruction_{tag_iter}.npy")
    meas_path   = os.path.join(args.outdir, f"measurement_{tag_iter}.npy")
    overlay_path= os.path.join(args.outdir, f"overlay_{tag_iter}.png")
    np.save(recon_path, x0_hat_raw)
    np.save(meas_path, y_raw)

    if args.plot:
        _save_plot(t, y_raw, x0_hat_raw,
                   clean_raw if (clean_raw is not None and len(clean_raw) == L) else None,
                   overlay_path, delta_t=1.0 / fs, sigma_scalar=sigma, xcorr_window_samp=args.xcorr_window_samp)

    # metrics on tail
    metrics = {
        "tag": tag_iter, "fs": fs, "score_secs": args.score_secs,
        "whiten": bool(args.whiten), "whiten_mode": args.whiten_mode,
        "whiten_kind_used": whiten_kind_used,
        "sigma_mode": args.sigma_mode, "cfg_scale": args.cfg_scale, "cfg_mode": args.cfg_mode,
        "cfg_center": args.cfg_center, "cfg_width": args.cfg_width, "cond_scale": args.cond_scale,
        "steps": args.steps, "start_t": int(start_t_eff),
        "cond_in_ch": int(cond_in_ch), "drop_y_only": bool(drop_y_only)
    }

    if (clean_raw is not None) and (len(clean_raw) == L):
        m_strain = _score_last_window(x0_hat_raw, clean_raw, fs, secs=args.score_secs)
        metrics["strain"] = m_strain

    if args.whiten and (clean_for_cond is not None):
        x0_hat_norm_np = x0_hat_norm.detach().cpu().numpy().ravel().astype(np.float64)
        x0_hat_white = x0_hat_norm_np * float(sigma)
        m_white = _score_last_window(x0_hat_white, clean_for_cond, fs, secs=args.score_secs)
        metrics["whitened"] = m_white

    metrics_path = os.path.join(args.outdir, f"metrics_{tag_iter}.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[done] wrote metrics: {metrics_path}")

    # proxy (optional)
    if args.one_step_proxy and (clean_for_cond is not None):
        x0_hat_proxy = one_step_proxy_like_test_infer(
            model=model, diffusion=diffusion,
            clean_norm=clean_norm_311, cond_stack=cond_stack,
            sigma_scalar=sigma, target_snr=args.target_snr, device=device,
            in_ch=in_ch, cond_in_ch=cond_in_ch, use_selfcond=use_selfcond,
            cfg_scale=args.cfg_scale, drop_y_only=drop_y_only,
            cond_scale=args.cond_scale, eps_scale=args.eps_scale, pred_type=args.pred_type,
            amp=args.amp
        ).detach().cpu().numpy().ravel()

        x0_hat_proxy_raw = x0_hat_proxy
        if args.whiten:
            if whiten_kind_used == "train":
                x0_hat_proxy_raw = _dewhiten_train_like(x0_hat_proxy_raw, P_train)
            elif whiten_kind_used == "welch":
                x0_hat_proxy_raw = _dewhiten_welch(x0_hat_proxy_raw, freqs_P, fs)
            elif whiten_kind_used == "model":
                x0_hat_proxy_raw = _dewhiten_model(x0_hat_proxy_raw, P_model_used)

        recon_proxy_path = os.path.join(args.outdir, f"reconstruction_{tag_proxy}.npy")
        meas_proxy_path  = os.path.join(args.outdir, f"measurement_{tag_proxy}.npy")
        overlay_proxy_path = os.path.join(args.outdir, f"overlay_{tag_proxy}.png")
        np.save(recon_proxy_path, x0_hat_proxy_raw)
        np.save(meas_proxy_path, y_raw)
        if args.plot:
            _save_plot(t, y_raw, x0_hat_proxy_raw, clean_raw, overlay_proxy_path,
                       delta_t=1.0 / fs, sigma_scalar=sigma, xcorr_window_samp=args.xcorr_window_samp)

        print(f"[done] wrote proxy reconstruction: {recon_proxy_path}")
        print(f"[done] wrote proxy measurement:    {meas_proxy_path}")

    print(f"[done] wrote iterative reconstruction: {recon_path}")
    print(f"[done] wrote measurement:              {meas_path}")
    if args.plot:
        print(f"[done] wrote iterative overlays:       {overlay_path}")

if __name__ == "__main__":
    main()
