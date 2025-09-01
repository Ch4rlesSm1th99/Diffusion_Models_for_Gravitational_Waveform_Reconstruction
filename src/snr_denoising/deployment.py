import os
import argparse
import h5py
import numpy as np
import torch
import json
import math
from typing import Optional
from models import UNet1D, CustomDiffusion

def _tail_mask(L: int, fs: float, secs: float = 0.8):
    t = np.arange(L)/fs
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

def _load_measurement_from_h5(h5_path: str, index: int):
    with h5py.File(h5_path, "r") as f:
        y = np.array(f["noisy"][index], dtype=np.float32)
        clean = np.array(f["signal"][index], dtype=np.float32) if "signal" in f else None
        fs = float(f.attrs.get("sampling_rate", 0.0)) or float(1.0 / f.attrs.get("delta_t", 1.0 / 4096.0))
    return y, clean, fs

def _load_measurement_from_npy(npy_path: str, fs: float):
    y = np.load(npy_path).astype(np.float32).ravel()
    return y, None, fs

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
    return s if s > 0 else 1.0

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
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30
    return float((a * b).sum() / den)

# ---------- SNR + scheduler helpers ----------
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

# Guidance schedule (time-dependent CFG)
def _cfg_weight(i: int, N: int, mode: str, wmax: float, center: float, width: float) -> float:
    """Return per-step guidance weight w_t.
    s = i/(N-1): 0 at start (very noisy), 1 at end (clean).
    - const:  return wmax
    - tophat: return wmax in [center - width/2, center + width/2], else 1.0
    - gauss:  return wmax * exp(-0.5 * ((s-center)/width)^2)
    """
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

# ---------- xcorr helpers ----------
def _best_lag_by_xcorr(a: np.ndarray, b: np.ndarray, max_shift: int = 0) -> int:
    if max_shift <= 0:
        max_shift = min(len(a), len(b)) - 1
    best_k, best_val = 0, -np.inf
    L = min(len(a), len(b))
    a = a[:L]
    b = b[:L]
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
        with open(base + "_xcorr_scores.txt", "w") as fh:
            fh.write(f"MAE,{mae}\nNMAE_clean,{nmae_clean}\nNMAE_sigma,{nmae_sigma}\n")

# ---------- proxy ONE-STEP ----------
@torch.no_grad()
def one_step_proxy_like_test_infer(model, diffusion, clean_norm: torch.Tensor,
                                   noisy_norm: torch.Tensor, sigma_scalar: float,
                                   target_snr: float, device: torch.device,
                                   in_ch: int, cfg_scale: float,
                                   cond_scale: float = 1.0,
                                   eps_scale: float = 1.0,
                                   pred_type: str = "eps"):
    ab = diffusion.alpha_bar.detach().cpu().numpy()
    snr_ts = np.sqrt(ab / (1 - ab))
    t_pick = int(np.argmin(np.abs(snr_ts - target_snr)))
    t = torch.full((1,), t_pick, dtype=torch.long, device=device)

    x_t, _ = diffusion.q_sample(clean_norm, t)  # diffuse the clean (proxy)

    # self-cond starts at zeros
    x0_sc = torch.zeros_like(x_t) if in_ch >= 3 else None

    def _make_in(x_t_, y_, sc_):
        if in_ch >= 3:
            return torch.cat([x_t_, y_, sc_], dim=1)
        elif in_ch == 2:
            return torch.cat([x_t_, y_], dim=1)
        else:
            return x_t_

    y_used = cond_scale * noisy_norm
    net_in_c = _make_in(x_t, y_used, x0_sc)
    out_c = model(net_in_c, t)

    if cfg_scale != 1.0:
        net_in_u = _make_in(x_t, torch.zeros_like(y_used), x0_sc)
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

# ---------- DDIM SAMPLER WITH SELF-COND AND SCHEDULED CFG ----------
@torch.no_grad()
def ddim_sample(model, diffusion, y_norm_311: torch.Tensor, T: int, steps: int, eta: float,
                device: torch.device, length: int, debug: bool,
                start_t: Optional[int], init_mode: str, x0_std_est: float,
                dc_weight: float, cond_scale: float, eps_scale: float, pred_type: str,
                in_ch: int, cfg_scale: float, selfcond_ema: float,
                cfg_mode: str, cfg_center: float, cfg_width: float, cfg_u_only_thresh: float,
                oracle_init: bool = False, clean_norm_311: Optional[torch.Tensor] = None,
                log_jsonl_path: Optional[str] = None, log_interval: int = 0,
                xcorr_window_samp: int = 0, delta_t: float = 1.0) -> torch.Tensor:

    def _log(obj: dict):
        if not log_jsonl_path:
            return
        with open(log_jsonl_path, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    t_schedule = _build_t_schedule(T=T, steps=steps, device=device, start_t=start_t)
    ab = diffusion.alpha_bar.to(device).clamp(1e-12, 1.0)
    ab_start = ab[int(t_schedule[0].item())]

    # init x_t here
    if oracle_init and (clean_norm_311 is not None):
        t0 = int(t_schedule[0].item())
        x_t, _ = diffusion.q_sample(clean_norm_311, torch.tensor([t0], device=device))
        if debug:
            print(f"[debug] oracle-init enabled (t0={t0})")
    else:
        if init_mode == "noise":
            x_t = torch.randn(1, 1, length, device=device)
        elif init_mode == "scaled-noise":
            std_init = torch.sqrt(ab_start * (x0_std_est ** 2) + (1 - ab_start))
            x_t = std_init * torch.randn(1, 1, length, device=device)
        elif init_mode == "y-blend":
            z = torch.randn(1, 1, length, device=device)
            x_t = torch.sqrt(ab_start) * y_norm_311 + torch.sqrt(1 - ab_start) * z
        else:
            raise ValueError(f"unknown init_mode: {init_mode}")

    # self-conditioning buffer
    x0_sc = torch.zeros_like(x_t) if in_ch >= 3 else None

    if debug:
        print(f"[debug] init_mode={init_mode}, x0_std_est={x0_std_est:.3f}, ab_start={ab_start.item():.6f}")
        print(f"[debug] schedule length={len(t_schedule)}, first={int(t_schedule[0])}, last={int(t_schedule[-1])}")
        print(_stats("x_T (init)", x_t))

    N = len(t_schedule)

    for i in range(N):
        t_now = int(t_schedule[i].item())
        ab_t = ab[t_now]
        ab_prev = ab[int(t_schedule[i + 1].item())] if i + 1 < N else torch.tensor(1.0, device=device)

        def _make_in(x_t_, y_, sc_):
            if in_ch >= 3:
                return torch.cat([x_t_, y_, sc_], dim=1)
            elif in_ch == 2:
                return torch.cat([x_t_, y_], dim=1)
            else:
                return x_t_

        y_used = cond_scale * y_norm_311

        # ------- scheduled CFG weight -------
        w_t = _cfg_weight(i=i, N=N, mode=cfg_mode, wmax=cfg_scale, center=cfg_center, width=cfg_width)
        # Decide which passes to run
        out_c = None
        out_u = None
        have_u = False

        # unconditional only (cheap) if guidance is tiny
        if w_t <= cfg_u_only_thresh:
            net_in_u = _make_in(x_t, torch.zeros_like(y_used), x0_sc)
            out_u = model(net_in_u, torch.full((1,), t_now, dtype=torch.long, device=device))
            out = out_u
            have_u = True
        # conditional only if effectively w==1
        elif abs(w_t - 1.0) <= 1e-6:
            net_in_c = _make_in(x_t, y_used, x0_sc)
            out_c = model(net_in_c, torch.full((1,), t_now, dtype=torch.long, device=device))
            out = out_c
        else:
            # need both passes
            net_in_c = _make_in(x_t, y_used, x0_sc)
            out_c = model(net_in_c, torch.full((1,), t_now, dtype=torch.long, device=device))
            net_in_u = _make_in(x_t, torch.zeros_like(y_used), x0_sc)
            out_u = model(net_in_u, torch.full((1,), t_now, dtype=torch.long, device=device))
            out = out_u + w_t * (out_c - out_u)
            have_u = True
        # -----------------------------------

        out = _reduce_to_one_channel(out)

        if pred_type == "eps":
            eps_hat = eps_scale * out
            x0_hat_norm = (x_t - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        else:
            x0_hat_norm = out
            eps_hat = (x_t - torch.sqrt(ab_t) * x0_hat_norm) / torch.sqrt(torch.clamp(1 - ab_t, min=1e-12))

        if dc_weight > 0:
            x0_hat_norm = (1 - dc_weight) * x0_hat_norm + dc_weight * y_norm_311

        if in_ch >= 3:
            if selfcond_ema <= 0:
                x0_sc = x0_hat_norm.detach()
            else:
                x0_sc = (selfcond_ema * x0_sc + (1 - selfcond_ema) * x0_hat_norm).detach()

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

        # Lightweight per-step log
        do_log = (log_interval and (i % log_interval == 0)) or (log_interval and i == N - 1)
        if do_log or debug:
            # if we didn't run uncond but want cond-delta, do quick uncond eval
            if (out_c is not None) and (not have_u) and log_jsonl_path:
                net_in_u2 = _make_in(x_t, torch.zeros_like(y_used), x0_sc)
                out_u2 = model(net_in_u2, torch.full((1,), t_now, dtype=torch.long, device=device))
            else:
                out_u2 = out_u if have_u else None

            cond_delta_rms = None
            if (out_c is not None) and (out_u2 is not None):
                d = (out_c - out_u2).reshape(-1)
                cond_delta_rms = float(torch.linalg.norm(d) / (d.numel() ** 0.5))

            # lag-aware corr to y (normalized domain)
            xt_np = x_t.detach().cpu().numpy().reshape(-1)
            y_np = y_norm_311.detach().cpu().numpy().reshape(-1)
            if len(xt_np) and len(y_np):
                win = xcorr_window_samp if xcorr_window_samp > 0 else min(len(xt_np) - 1, int(max(1.0, 0.25 / delta_t)))
                k = _best_lag_by_xcorr(xt_np, y_np, max_shift=win)
                if k < 0:
                    a = xt_np[-k:]
                    b = y_np[:len(xt_np) + k]
                elif k > 0:
                    a = xt_np[:len(xt_np) - k]
                    b = y_np[k:]
                else:
                    a = xt_np
                    b = y_np
                corr_lag = _corr_np(a, b)
            else:
                k = 0
                corr_lag = 0.0

            _log({
                "phase": "ddim_step",
                "i": i, "t": t_now,
                "i_norm": float(0.0 if N <= 1 else i/(N-1)),
                "alpha_bar": float(ab_t.item() if isinstance(ab_t, torch.Tensor) else ab_t),
                "cfg_mode": cfg_mode,
                "cfg_w_t": float(w_t),
                "cfg_scale": float(cfg_scale),
                "norm_pred_c": float(torch.linalg.norm(out_c).item()) if out_c is not None else None,
                "norm_pred_u": float(torch.linalg.norm(out_u2).item()) if out_u2 is not None else None,
                "cond_delta_rms": cond_delta_rms,
                "lag_best_samples": int(k),
                "lag_best_ms": float(k * delta_t * 1000.0),
                "corr_lag": corr_lag,
            })

    return x_t  # = x0_hat_norm

def main():
    ap = argparse.ArgumentParser("Conditional deployment sampler (+self-cond & scheduled CFG)")
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
    ap.add_argument("--whiten", action="store_true")

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

    # CFG + self-conditioning at inference
    ap.add_argument("--cfg-scale", type=float, default=1.5, help="Peak CFG weight (wmax). 1.0 = plain conditional")
    ap.add_argument("--cfg-mode", choices=["const", "tophat", "gauss"], default="const",
                    help="Time-dependent guidance schedule")
    ap.add_argument("--cfg-center", type=float, default=0.70,
                    help="Center of schedule in normalized step s∈[0,1]; 0=noisiest, 1=cleanest")
    ap.add_argument("--cfg-width", type=float, default=0.12,
                    help="For tophat: full width; for gauss: sigma; both in s-units")
    ap.add_argument("--cfg-u-only-thresh", type=float, default=0.05,
                    help="If w_t <= this, run unconditional-only (skip cond pass)")
    ap.add_argument("--selfcond-ema", type=float, default=0.9, help="EMA for self-conditioning buffer; 0=last only")

    # start from q_sample(clean, start_t)
    ap.add_argument("--oracle-init", action="store_true",
                    help="Start from q_sample(clean_norm, start_t) instead of noise/y-blend (requires clean in HDF5)")

    ap.add_argument("--run-tag", type=str, default=None,
                    help="Titles for filenames, if none an automatic tag is generated from settings")

    ap.add_argument("--score-secs", type=float, default=0.8,
                    help="length (seconds) of tail window for corr/MAE scoring")

    args = ap.parse_args()

    # seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # load measurement
    if args.input_h5:
        y_raw, clean_raw, fs = _load_measurement_from_h5(args.input_h5, args.index)
    else:
        y_raw, clean_raw, fs = _load_measurement_from_npy(args.input_npy, args.fs)

    L = len(y_raw)
    t = np.arange(L) / fs
    if args.debug:
        print(_stats("y_raw", y_raw))
        if clean_raw is not None:
            print(_stats("clean_raw", clean_raw))
        print(f"[debug] fs={fs}, L={L}, dt={1.0 / fs:.6e}")

    # optional whitening (needs to match training)
    if args.whiten:
        from scipy.signal import welch
        from numpy.fft import rfft, irfft, rfftfreq
        f, Pxx = welch(y_raw, fs=fs, nperseg=min(4096, L))
        freqs = rfftfreq(L, 1 / fs)
        P = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])

        Y = rfft(y_raw)
        y_for_cond = irfft(Y / np.sqrt(P + 1e-12), n=L).astype(np.float32)
        if args.debug:
            print(_stats("y_white", y_for_cond))

        if clean_raw is not None:
            C = rfft(clean_raw)
            clean_for_cond = irfft(C / np.sqrt(P + 1e-12), n=L).astype(np.float32)
            if args.debug:
                print(_stats("clean_white", clean_for_cond))
        else:
            clean_for_cond = None
    else:
        y_for_cond = y_raw
        clean_for_cond = clean_raw

    # sigma from SAME domain as conditioning
    sigma = _pick_sigma(y_for_cond, args.sigma_mode, args.sigma_fixed)
    if args.debug:
        print(f"[debug] sigma_mode={args.sigma_mode}, sigma={sigma:.3e}")
        print(f"[debug] SNR(y_raw)~mean(|y|)/sigma -> {np.mean(np.abs(y_raw)) / sigma:.3e} (rough quick check)")

    # normalized conditioning (match training domain)
    y_norm = (y_for_cond / sigma).astype(np.float32)
    y_norm_311 = torch.from_numpy(y_norm).to(device).view(1, 1, -1)
    clean_norm_311 = None
    if clean_for_cond is not None:
        clean_norm_311 = torch.from_numpy((clean_for_cond / sigma).astype(np.float32)).to(device).view(1, 1, -1)

    if args.debug:
        print(_stats("y_norm", y_norm))

    # load model + diffusion
    ckpt = torch.load(args.model, map_location=device)
    in_ch = ckpt.get("args", {}).get("in_ch", 1)
    model = UNet1D(
        in_ch=in_ch,
        base_ch=ckpt["args"]["base_ch"],
        time_dim=ckpt["args"]["time_dim"],
        depth=ckpt["args"]["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.in_ch = in_ch
    diffusion = CustomDiffusion(T=ckpt["args"]["T"], device=device)

    # choose start_t
    start_t = t_for_target_snr(diffusion, args.start_snr) if (args.start_snr is not None) else args.start_t
    start_t_eff = (ckpt["args"]["T"] - 1) if (start_t is None) else int(start_t)
    start_snr_eff = snr_from_alpha_bar(diffusion.alpha_bar)[start_t_eff]

    def _mk_tag(mode):
        auto = (f"{mode}_t{start_t_eff}_snr{start_snr_eff:.1f}"
                f"_steps{args.steps}_eta{args.eta}_cfg{args.cfg_scale}"
                f"_cfgmode-{args.cfg_mode}_ctr{args.cfg_center}_w{args.cfg_width}"
                f"_init-{args.init_mode}_pred-{args.pred_type}"
                f"_dc{args.dc_weight}_cond{args.cond_scale}_eps{args.eps_scale}"
                f"_{'white' if args.whiten else 'raw'}_{args.sigma_mode}").replace('.', 'p')
        return (args.run_tag or auto)

    tag_iter = _mk_tag("iter")
    tag_proxy = _mk_tag("proxy")

    # estimate clean_norm std for scaled init
    x0_std_est = args.x0_std_est
    if clean_for_cond is not None:
        x0_std_est = float(np.std((clean_for_cond / sigma).astype(np.float32)))
    if args.debug:
        print(f"[debug] x0_std_est (used) = {x0_std_est:.6f}")

    print(f"[info] device={device} steps={args.steps}/{ckpt['args']['T']} eta={args.eta} "
          f"sigma_mode={args.sigma_mode} whiten={'on' if args.whiten else 'off'} seed={args.seed}")
    print(f"[info] start_t={start_t_eff} (SNR≈{start_snr_eff:.2f}) init_mode={args.init_mode} "
          f"pred_type={args.pred_type} in_ch={in_ch} selfcond_ema={args.selfcond_ema}")
    print(f"[info] CFG: mode={args.cfg_mode} wmax={args.cfg_scale} center={args.cfg_center} "
          f"width={args.cfg_width} u_only_thresh={args.cfg_u_only_thresh}")

    # iterative sampling (DDIM)
    x0_hat_norm = ddim_sample(
        model=model,
        diffusion=diffusion,
        y_norm_311=y_norm_311,
        T=ckpt["args"]["T"],
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
        cfg_scale=args.cfg_scale,
        selfcond_ema=args.selfcond_ema,
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
    )

    x0_hat_raw = (x0_hat_norm * torch.tensor(sigma, device=device).view(1, 1, 1)).detach().cpu().numpy().ravel()
    if args.debug:
        print(_stats("x0_hat_raw (iterative)", x0_hat_raw))

    # de-whiten if used
    if args.whiten:
        from scipy.signal import welch
        from numpy.fft import rfft, irfft, rfftfreq
        f, Pxx = welch(y_raw, fs=fs, nperseg=min(4096, L))
        freqs = rfftfreq(L, 1 / fs)
        P = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])
        Xw = rfft(x0_hat_raw)
        x0_hat_raw = irfft(Xw * np.sqrt(P + 1e-12), n=L)
        if args.debug:
            print(_stats("x0_hat_raw (de-whitened)", x0_hat_raw))

    # save tagged
    os.makedirs(args.outdir, exist_ok=True)

    recon_path = os.path.join(args.outdir, f"reconstruction_{tag_iter}.npy")
    meas_path = os.path.join(args.outdir, f"measurement_{tag_iter}.npy")
    overlay_path = os.path.join(args.outdir, f"overlay_{tag_iter}.png")

    np.save(recon_path, x0_hat_raw)
    np.save(meas_path, y_raw)

    # plots
    if args.plot:
        _save_plot(
            t, y_raw, x0_hat_raw,
            clean_raw if (clean_raw is not None and len(clean_raw) == L) else None,
            overlay_path,
            delta_t=1.0 / fs, sigma_scalar=sigma, xcorr_window_samp=args.xcorr_window_samp
        )

    # ---- metrics: last `score-secs` on tail ----
    metrics = {
        "tag": tag_iter,
        "fs": fs,
        "score_secs": args.score_secs,
        "whiten": bool(args.whiten),
        "sigma_mode": args.sigma_mode,
        "cfg_scale": args.cfg_scale,
        "cfg_mode": args.cfg_mode,
        "cfg_center": args.cfg_center,
        "cfg_width": args.cfg_width,
        "cond_scale": args.cond_scale,
        "steps": args.steps,
        "start_t": int(start_t_eff),
    }

    # strain-domain (always available if we have clean_raw)
    if (clean_raw is not None) and (len(clean_raw) == L):
        m_strain = _score_last_window(x0_hat_raw, clean_raw, fs, secs=args.score_secs)
        metrics["strain"] = m_strain

    # whitened-domain (only if whitening was used and we had clean_for_cond)
    if args.whiten and (clean_for_cond is not None):
        x0_hat_norm_np = x0_hat_norm.detach().cpu().numpy().ravel().astype(np.float64)
        x0_hat_white = x0_hat_norm_np * float(sigma)  # un-normalize but keep whitened domain
        m_white = _score_last_window(x0_hat_white, clean_for_cond, fs, secs=args.score_secs)
        metrics["whitened"] = m_white

    metrics_path = os.path.join(args.outdir, f"metrics_{tag_iter}.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[done] wrote metrics:                {metrics_path}")
    # -------------------------------------------

    # proxy save if tagged
    if args.one_step_proxy and (clean_for_cond is not None):
        clean_norm = torch.from_numpy((clean_for_cond / sigma).astype(np.float32)).to(device).view(1, 1, -1)

        x0_hat_proxy = one_step_proxy_like_test_infer(
            model=model, diffusion=diffusion,
            clean_norm=clean_norm, noisy_norm=y_norm_311,
            sigma_scalar=sigma, target_snr=args.target_snr, device=device,
            in_ch=in_ch, cfg_scale=args.cfg_scale,
            cond_scale=args.cond_scale, eps_scale=args.eps_scale, pred_type=args.pred_type
        ).detach().cpu().numpy().ravel()

        # de-whiten proxy to original strain domain (to match clean_raw)
        x0_hat_proxy_raw = x0_hat_proxy
        if args.whiten:
            from scipy.signal import welch
            from numpy.fft import rfft, irfft, rfftfreq
            f, Pxx = welch(y_raw, fs=fs, nperseg=min(4096, L))
            freqs = rfftfreq(L, 1 / fs)
            P = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])
            Xw = rfft(x0_hat_proxy_raw)
            x0_hat_proxy_raw = irfft(Xw * np.sqrt(P + 1e-12), n=L)

        recon_proxy_path = os.path.join(args.outdir, f"reconstruction_{tag_proxy}.npy")
        meas_proxy_path = os.path.join(args.outdir, f"measurement_{tag_proxy}.npy")
        overlay_proxy_path = os.path.join(args.outdir, f"overlay_{tag_proxy}.png")

        np.save(recon_proxy_path, x0_hat_proxy_raw)
        np.save(meas_proxy_path, y_raw)

        if args.plot:
            _save_plot(
                t, y_raw, x0_hat_proxy_raw, clean_raw,
                overlay_proxy_path,
                delta_t=1.0 / fs, sigma_scalar=sigma, xcorr_window_samp=args.xcorr_window_samp
            )

    print(f"[done] wrote iterative reconstruction: {recon_path}")
    print(f"[done] wrote measurement:              {meas_path}")
    if args.plot:
        print(f"[done] wrote iterative overlays:       {overlay_path} (+ _xcorr*.png & _xcorr_scores.txt)")

    if args.one_step_proxy and (clean_for_cond is not None):
        print(f"[done] wrote proxy reconstruction:     {recon_proxy_path}")
        print(f"[done] wrote proxy measurement:        {meas_proxy_path}")
        if args.plot:
            print(f"[done] wrote proxy overlays:          {overlay_proxy_path} (+ _xcorr*.png & _xcorr_scores.txt)")

if __name__ == "__main__":
    main()
