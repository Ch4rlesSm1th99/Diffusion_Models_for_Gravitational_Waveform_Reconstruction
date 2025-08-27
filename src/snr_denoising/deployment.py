# deployment.py
import os
import argparse
import h5py
import numpy as np
import torch
from typing import Optional
from models import UNet1D, CustomDiffusion


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1: return x.view(1, 1, -1)
    if x.ndim == 2: return x.unsqueeze(1)
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

# SNR + schduler helpers
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

# xcorr helpers
def _best_lag_by_xcorr(a: np.ndarray, b: np.ndarray, max_shift: int = 0) -> int:
    if max_shift <= 0:
        max_shift = min(len(a), len(b)) - 1
    best_k, best_val = 0, -np.inf
    L = min(len(a), len(b))
    a = a[:L]; b = b[:L]
    for k in range(-max_shift, max_shift + 1):
        if k < 0: v = float(np.dot(a[-k:], b[:L + k]))
        elif k > 0: v = float(np.dot(a[:L - k], b[k:]))
        else: v = float(np.dot(a, b))
        if v > best_val: best_val, best_k = v, k
    return best_k

def _align_xcorr(a: np.ndarray, b: np.ndarray, delta_t: float, max_shift: int = 0):
    k = _best_lag_by_xcorr(a, b, max_shift)
    start = max(0, -k); stop = min(len(a), len(b) - k)
    if stop <= start:
        L = min(len(a), len(b)); a_al, b_al = a[:L], b[:L]
    else:
        a_al = a[start:stop]; b_al = b[start + k:stop + k]
    L = len(a_al)
    t = np.arange(L, dtype=np.float64) * delta_t
    pk = int(np.argmax(np.abs(a_al))); t -= t[pk]
    return a_al, b_al, t

def _save_plot(t, y, xhat, clean, outpng, delta_t, sigma_scalar, xcorr_window_samp=0):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(outpng), exist_ok=True)

    plt.figure(figsize=(12, 3.2))
    if y is not None: plt.plot(t, y, label="measurement (noisy)", alpha=0.5, linewidth=1.0)
    plt.plot(t, xhat, label="reconstruction", linewidth=1.4)
    if clean is not None and len(clean) == len(xhat): plt.plot(t, clean, label="clean (gt)", linewidth=1.0)
    plt.xlabel("Time (s)"); plt.ylabel("Strain"); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(outpng, dpi=150); plt.close()

    if clean is not None and len(clean) == len(xhat):
        clean_a, recon_a, t_a = _align_xcorr(clean, xhat, delta_t, max_shift=xcorr_window_samp)
        plt.figure(figsize=(12, 3.2))
        plt.plot(t_a, recon_a, label="recon (xcorr-aligned)", linewidth=1.4)
        plt.plot(t_a, clean_a, label="clean (gt)", linewidth=1.0)
        plt.xlabel("Time (s) - t=0 at clean peak"); plt.ylabel("Strain"); plt.legend(frameon=False); plt.tight_layout()
        base, ext = os.path.splitext(outpng); outpng_xc = base + "_xcorr" + ext
        plt.savefig(outpng_xc, dpi=150); plt.close()

        mask = (t_a >= -0.080) & (t_a <= 0.040)
        mae = float(np.mean(np.abs(recon_a[mask] - clean_a[mask])))
        nmae_clean = mae / (float(np.mean(np.abs(clean_a[mask]))) + 1e-12)
        nmae_sigma = mae / (float(sigma_scalar) + 1e-12)
        print(f"[scores(xcorr)win -80ms,+40ms] MAE={mae:.3e} NMAE_clean={nmae_clean:.3e} NMAE_sigma={nmae_sigma:.3e}")
        with open(base + "_xcorr_scores.txt", "w") as fh:
            fh.write(f"MAE,{mae}\nNMAE_clean,{nmae_clean}\nNMAE_sigma,{nmae_sigma}\n")

# proxy ONE-STEP
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

    return x0_hat_norm * torch.tensor(sigma_scalar, device=device).view(1,1,1)

# DDIM SAMPLER WITH SELF-COND AND CFG

@torch.no_grad()
def ddim_sample(model, diffusion, y_norm_311: torch.Tensor, T: int, steps: int, eta: float,
                device: torch.device, length: int, debug: bool,
                start_t: Optional[int], init_mode: str, x0_std_est: float,
                dc_weight: float, cond_scale: float, eps_scale: float, pred_type: str,
                in_ch: int, cfg_scale: float, selfcond_ema: float,
                oracle_init: bool = False, clean_norm_311: Optional[torch.Tensor] = None) -> torch.Tensor:
    t_schedule = _build_t_schedule(T=T, steps=steps, device=device, start_t=start_t)
    ab = diffusion.alpha_bar.to(device).clamp(1e-12, 1.0)
    ab_start = ab[int(t_schedule[0].item())]

    # init x_t here
    if oracle_init and (clean_norm_311 is not None):
        # start exactly from q_sample(clean, t0)
        t0 = int(t_schedule[0].item())
        x_t, _ = diffusion.q_sample(clean_norm_311, torch.tensor([t0], device=device))
        # diffusion.q_sample returns [B,1,L]; we only need the single sample
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

    for i in range(len(t_schedule)):
        t_now = int(t_schedule[i].item())
        ab_t = ab[t_now]
        ab_prev = ab[int(t_schedule[i + 1].item())] if i + 1 < len(t_schedule) else torch.tensor(1.0, device=device)

        def _make_in(x_t_, y_, sc_):
            if in_ch >= 3:
                return torch.cat([x_t_, y_, sc_], dim=1)
            elif in_ch == 2:
                return torch.cat([x_t_, y_], dim=1)
            else:
                return x_t_

        y_used = cond_scale * y_norm_311

        # conditional + (optional) unconditional for CFG
        net_in_c = _make_in(x_t, y_used, x0_sc)
        out_c = model(net_in_c, torch.full((1,), t_now, dtype=torch.long, device=device))

        if cfg_scale != 1.0:
            net_in_u = _make_in(x_t, torch.zeros_like(y_used), x0_sc)
            out_u = model(net_in_u, torch.full((1,), t_now, dtype=torch.long, device=device))
            out = out_u + cfg_scale * (out_c - out_u)
        else:
            out = out_c

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
            dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma_t**2, min=0.0)) * eps_hat
            noise = sigma_t * torch.randn_like(x_t) if sigma_t.item() > 0 else 0.0
            x_t = torch.sqrt(ab_prev) * x0_hat_norm + dir_xt + noise

        if debug and (i % max(1, len(t_schedule)//5) == 0 or t_now == 0):
            print(_stats(f"x_t (t={t_now})", x_t))
            print(_stats(f"eps_hat (t={t_now})", eps_hat))

    return x_t  # = x0_hat_norm


def main():
    ap = argparse.ArgumentParser("Conditional deployment sampler (+self-cond & CFG)")
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
    ap.add_argument("--init-mode", choices=["noise","scaled-noise","y-blend"], default="noise")
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

    # diagnostic proxy
    ap.add_argument("--one-step-proxy", action="store_true")
    ap.add_argument("--target-snr", type=float, default=20.0)

    # knobs
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--pred-type", choices=["eps","x0"], default="eps")

    # CFG + self-conditioning at inference
    ap.add_argument("--cfg-scale", type=float, default=1.5, help=">1 enables CFG; 1.0 disables")
    ap.add_argument("--selfcond-ema", type=float, default=0.9, help="EMA for self-conditioning buffer; 0=last only")

    # start from q_sample(clean, start_t)
    ap.add_argument("--oracle-init", action="store_true",
                    help="Start from q_sample(clean_norm, start_t) instead of noise/y-blend (requires clean in HDF5)")

    ap.add_argument("--run-tag", type=str, default=None,
                    help="Titles for filenames, if none an automatic tag is generated from settings")

    args = ap.parse_args()

    # seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # load measurement
    if args.input_h5:
        y_raw, clean_raw, fs = _load_measurement_from_h5(args.input_h5, args.index)
    else:
        y_raw, clean_raw, fs = _load_measurement_from_npy(args.input_npy, args.fs)

    L = len(y_raw); t = np.arange(L) / fs
    if args.debug:
        print(_stats("y_raw", y_raw))
        if clean_raw is not None: print(_stats("clean_raw", clean_raw))
        print(f"[debug] fs={fs}, L={L}, dt={1.0/fs:.6e}")

    # optional whitening (needs to match training)
    if args.whiten:
        from scipy.signal import welch
        from numpy.fft import rfft, irfft, rfftfreq
        f, Pxx = welch(y_raw, fs=fs, nperseg=min(4096, L))
        freqs = rfftfreq(L, 1 / fs)
        P = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])

        Y = rfft(y_raw); y_for_cond = irfft(Y / np.sqrt(P + 1e-12), n=L).astype(np.float32)
        if args.debug: print(_stats("y_white", y_for_cond))

        if clean_raw is not None:
            C = rfft(clean_raw); clean_for_cond = irfft(C / np.sqrt(P + 1e-12), n=L).astype(np.float32)
            if args.debug: print(_stats("clean_white", clean_for_cond))
        else:
            clean_for_cond = None
    else:
        y_for_cond = y_raw
        clean_for_cond = clean_raw

    # sigma from SAME domain as conditioning
    sigma = _pick_sigma(y_for_cond, args.sigma_mode, args.sigma_fixed)
    if args.debug:
        print(f"[debug] sigma_mode={args.sigma_mode}, sigma={sigma:.3e}")
        print(f"[debug] SNR(y_raw)~mean(|y|)/sigma -> {np.mean(np.abs(y_raw))/sigma:.3e} (rough quick check)")

    # normalized conditioning (match training domain)
    y_norm = (y_for_cond / sigma).astype(np.float32)
    y_norm_311 = torch.from_numpy(y_norm).to(device).view(1,1,-1)
    clean_norm_311 = None
    if clean_for_cond is not None:
        clean_norm_311 = torch.from_numpy((clean_for_cond / sigma).astype(np.float32)).to(device).view(1, 1, -1)

    if args.debug: print(_stats("y_norm", y_norm))

    # load model + diffusion
    ckpt = torch.load(args.model, map_location=device)
    in_ch = ckpt.get("args", {}).get("in_ch", 1)
    model = UNet1D(
        in_ch=in_ch,
        base_ch=ckpt["args"]["base_ch"],
        time_dim=ckpt["args"]["time_dim"],
        depth=ckpt["args"]["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    model.in_ch = in_ch
    diffusion = CustomDiffusion(T=ckpt["args"]["T"], device=device)

    # choose start_t
    start_t = t_for_target_snr(diffusion, args.start_snr) if (args.start_snr is not None) else args.start_t
    start_t_eff = (ckpt["args"]["T"] - 1) if (start_t is None) else int(start_t)
    start_snr_eff = snr_from_alpha_bar(diffusion.alpha_bar)[start_t_eff]

    def _mk_tag(mode):
        auto = (f"{mode}_t{start_t_eff}_snr{start_snr_eff:.1f}"
                f"_steps{args.steps}_eta{args.eta}_cfg{args.cfg_scale}"
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
          f"dc_weight={args.dc_weight} pred_type={args.pred_type} in_ch={in_ch} cfg_scale={args.cfg_scale} "
          f"selfcond_ema={args.selfcond_ema}")


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
        oracle_init=args.oracle_init,
        clean_norm_311=clean_norm_311
    )

    x0_hat_raw = (x0_hat_norm * torch.tensor(sigma, device=device).view(1,1,1)).detach().cpu().numpy().ravel()
    if args.debug: print(_stats("x0_hat_raw (iterative)", x0_hat_raw))

    # de-whiten if used
    if args.whiten:
        from scipy.signal import welch
        from numpy.fft import rfft, irfft, rfftfreq
        f, Pxx = welch(y_raw, fs=fs, nperseg=min(4096, L))
        freqs = rfftfreq(L, 1 / fs)
        P = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])
        Xw = rfft(x0_hat_raw)
        x0_hat_raw = irfft(Xw * np.sqrt(P + 1e-12), n=L)
        if args.debug: print(_stats("x0_hat_raw (de-whitened)", x0_hat_raw))

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
        # save the measurement
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
