import os, json, argparse, random, warnings
import numpy as np
import torch
import inference as inf

"""
Evaluate a (fixed-data–trained) model across the *discrete* (m1, m2) values
present in a dataset and produce per-(m1, m2) performance heatmaps + tables.

- Axes are *discrete* and derived from the unique mass values in the file.
- Bin edges are midpoints between adjacent mass values; each cell = one mass value.
- If there are many unique masses, axis tick labels are thinned automatically.
"""


def _objective(m_strain, m_white):
    r_s = m_strain.get("corr_last", 0.0) if m_strain else 0.0
    r_w = m_white.get("corr_last", 0.0) if m_white else 0.0
    nmae_sigma = m_strain.get("nmae_sigma", 0.0) if m_strain else 0.0
    return float(r_s + 0.5 * r_w - 0.1 * nmae_sigma)

def _window_indices_full(L, fs): return 0, L

def _window_indices_tail(L, fs, secs):
    W = int(max(1, secs * fs)); return max(0, L - W), L

def _window_indices_merger(clean_strain, fs, left_s, right_s):
    pk = int(np.argmax(np.abs(clean_strain)))
    left = int(max(0, pk - left_s * fs))
    right = int(min(len(clean_strain), pk + right_s * fs))
    return int(left), int(right)

def _apply_alignment(a, b, fs, mode="none", max_shift_s=0.02):
    if mode == "none":
        return a, b
    if mode == "peak":
        pa = int(np.argmax(np.abs(a))); pb = int(np.argmax(np.abs(b)))
        k = pb - pa
        if k > 0:  a_al, b_al = a[:len(a)-k], b[k:]
        elif k < 0: a_al, b_al = a[-k:], b[:len(b)+k]
        else: a_al, b_al = a, b
        L = min(len(a_al), len(b_al)); return a_al[:L], b_al[:L]
    max_shift = int(max(1, max_shift_s * fs))
    try: k = int(inf._best_lag_by_xcorr(a, b, max_shift=max_shift))
    except Exception: k = 0
    if k > 0:  a_al, b_al = a[k:], b[:len(b)-k]
    elif k < 0: a_al, b_al = a[:len(a)+k], b[-k:]
    else: a_al, b_al = a, b
    L = min(len(a_al), len(b_al)); return a_al[:L], b_al[:L]

def _prep_sample_from_h5(h5_path, idx, cond_in_ch, whiten_mode, sigma_mode, sigma_fixed, device,
                         M_SCALE, Q_SCALE):
    y_raw, clean_raw, fs, P_model_in, (fw_in, Pw_in), meta = inf._load_measurement_from_h5(h5_path, idx)
    L = len(y_raw)
    whiten_kind_used = "raw"; P_train = None; freqs_P = None; P_model_used = None
    if whiten_mode:
        mode = whiten_mode
        if mode == "auto":
            if (fw_in is not None) and (Pw_in is not None):
                from numpy.fft import rfftfreq
                f_tgt = rfftfreq(L, 1.0 / fs)
                P = np.interp(f_tgt, fw_in, Pw_in, left=Pw_in[0], right=Pw_in[-1])
                Y = np.fft.rfft(y_raw.astype(np.float64))
                y_for_cond = np.fft.irfft(Y/np.sqrt(P+1e-12), n=L).astype(np.float32)
                if clean_raw is not None:
                    X = np.fft.rfft(clean_raw.astype(np.float64))
                    clean_for_cond = np.fft.irfft(X/np.sqrt(P+1e-12), n=L).astype(np.float32)
                else:
                    clean_for_cond = None
                freqs_P = (f_tgt, P); whiten_kind_used = "welch"
            elif P_model_in is not None:
                y_for_cond, clean_for_cond, P_model_used = inf._whiten_pair_model(y_raw, clean_raw, P_model_in, fs)
                whiten_kind_used = "model"
            else:
                y_for_cond, clean_for_cond, P_train = inf._whiten_pair_train_like(y_raw, clean_raw, fs)
                whiten_kind_used = "train"
        elif mode == "welch":
            y_for_cond, clean_for_cond, freqs_P = inf._whiten_pair_welch(y_raw, clean_raw, fs)
            whiten_kind_used = "welch"
        elif mode == "model":
            if P_model_in is None:
                y_for_cond, clean_for_cond, P_train = inf._whiten_pair_train_like(y_raw, clean_raw, fs)
                whiten_kind_used = "train"
            else:
                y_for_cond, clean_for_cond, P_model_used = inf._whiten_pair_model(y_raw, clean_raw, P_model_in, fs)
                whiten_kind_used = "model"
        else:
            y_for_cond, clean_for_cond, P_train = inf._whiten_pair_train_like(y_raw, clean_raw, fs)
            whiten_kind_used = "train"
    else:
        y_for_cond = y_raw; clean_for_cond = clean_raw; whiten_kind_used = "raw"

    sigma = inf._pick_sigma(y_for_cond, sigma_mode, sigma_fixed)
    fallback = {"train": 2.914e-12, "welch": 2.914e-16, "model": 2.914e-16, "raw": 2.914e-12}
    if (not np.isfinite(sigma)) or (sigma < 1e-20):
        sigma = fallback.get(whiten_kind_used, fallback["train"])

    y_norm = (y_for_cond / sigma).astype(np.float32)
    y_norm_311 = torch.from_numpy(y_norm).to(device).view(1, 1, -1)
    clean_norm_311 = None
    if clean_for_cond is not None:
        clean_norm_311 = torch.from_numpy((clean_for_cond / sigma).astype(np.float32)).to(device).view(1, 1, -1)

    if cond_in_ch <= 1:
        cond_stack = y_norm_311
    else:
        meta_arr = inf._meta_to_stack(meta, L=len(y_raw), cond_in_ch=cond_in_ch,
                                      M_SCALE=M_SCALE, Q_SCALE=Q_SCALE)
        if meta_arr is None:
            meta_arr = np.zeros((cond_in_ch - 1, len(y_raw)), dtype=np.float32)
        meta_311 = torch.from_numpy(meta_arr).to(device).unsqueeze(0)
        cond_stack = torch.cat([y_norm_311, meta_311], dim=1)

    return dict(
        y_raw=y_raw, clean_raw=clean_raw, y_for_cond=y_for_cond, clean_for_cond=clean_for_cond,
        y_norm_311=y_norm_311, clean_norm_311=clean_norm_311, cond_stack=cond_stack,
        sigma=float(sigma), fs=float(fs), L=len(y_raw),
        whiten_kind_used=whiten_kind_used, P_train=P_train, freqs_P=freqs_P, P_model_used=P_model_used
    )

def _dewhiten_back(x0_hat_white, prep):
    wk = prep["whiten_kind_used"]
    if wk == "train":  return inf._dewhiten_train_like(x0_hat_white, prep["P_train"])
    if wk == "welch":  return inf._dewhiten_welch(x0_hat_white, prep["freqs_P"], prep["fs"])
    if wk == "model":  return inf._dewhiten_model(x0_hat_white, prep["P_model_used"])
    return x0_hat_white

def _parse_sweep_best(sweep_dir):
    out = {}
    if not sweep_dir or not os.path.isdir(sweep_dir): return out
    bcmd = os.path.join(sweep_dir, "best_cmd.txt")
    if os.path.exists(bcmd):
        line = open(bcmd, "r").read().strip()
        toks = line.split()
        def _get(flag, cast=str, default=None):
            if flag in toks:
                i = toks.index(flag)
                if i+1 < len(toks):
                    try: return cast(toks[i+1])
                    except Exception: return default
            return default
        return dict(
            steps=_get("--steps", int, 300), eta=_get("--eta", float, 0.0),
            start_snr=_get("--start-snr", float, None), start_t=_get("--start-t", int, None),
            init_mode=_get("--init-mode", str, "scaled-noise"),
            cfg_scale=_get("--cfg-scale", float, 1.0), cfg_mode=_get("--cfg-mode", str, "const"),
            cfg_center=_get("--cfg-center", float, 0.70), cfg_width=_get("--cfg-width", float, 0.12),
            dc_weight=_get("--dc-weight", float, 0.0),
        )
    fr = os.path.join(sweep_dir, "final_results.json")
    if os.path.exists(fr):
        try:
            J = json.load(open(fr, "r")); J.sort(key=lambda z:z.get("J_refine_mean", -1e9), reverse=True)
            b = J[0]
            return dict(steps=300, eta=b.get("eta",0.0), start_snr=float(b.get("start_snr",1.0)),
                        init_mode=b.get("init_mode","scaled-noise"), cfg_scale=float(b.get("cfg_scale",1.0)),
                        cfg_mode=b.get("cfg_mode","const"), cfg_center=float(b.get("cfg_center",0.70)),
                        cfg_width=float(b.get("cfg_width",0.12)), dc_weight=float(b.get("dc_weight",0.0)))
        except Exception: pass
    ct = os.path.join(sweep_dir, "coarse_top.json")
    if os.path.exists(ct):
        try:
            J = json.load(open(ct,"r")); J.sort(key=lambda z:z.get("J_coarse",-1e9), reverse=True)
            b = J[0]
            return dict(steps=300, eta=b.get("eta",0.0), start_snr=float(b.get("start_snr",1.0)),
                        init_mode=b.get("init_mode","scaled-noise"), cfg_scale=float(b.get("cfg_scale",1.0)),
                        cfg_mode=b.get("cfg_mode","const"), cfg_center=float(b.get("cfg_center",0.70)),
                        cfg_width=float(b.get("cfg_width",0.12)), dc_weight=float(b.get("dc_weight",0.0)))
        except Exception: pass
    return out

# discrete grid utils

def _unique_sorted(vals, rdec=6):
    u = np.unique(np.round(np.asarray(vals, dtype=np.float64), rdec))
    u.sort()
    return u

def _midpoint_edges_from_values(vals, rdec=6):
    u = _unique_sorted(vals, rdec=rdec)
    if len(u) < 2:
        step = 1.0
        edges = np.array([u[0]-0.5, u[0]+0.5], dtype=np.float64)
    else:
        diffs = np.diff(u)
        step = float(np.median(diffs))
        mids = 0.5*(u[:-1] + u[1:])
        edges = np.concatenate([[u[0]-step/2], mids, [u[-1]+step/2]]).astype(np.float64)
    centers = u.astype(np.float64)
    return edges, centers

def _choose_tick_values(centers, max_labels):
    n = len(centers)
    if n <= max_labels:
        idx = np.arange(n)
    else:
        step = int(np.ceil(n / max_labels))
        idx = np.arange(0, n, step)
        if idx[-1] != n-1:
            idx = np.append(idx, n-1)
    return centers[idx]

# --------------- main ------------- #

def main():
    ap = argparse.ArgumentParser("Evaluate model across (m1,m2) grid with discrete axes from dataset")
    ap.add_argument("--input-h5", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")

    # load knobs / overrides
    ap.add_argument("--from-sweep", dest="from_sweep", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--start-snr", dest="start_snr", type=float, default=None)
    ap.add_argument("--start-t", dest="start_t", type=int, default=None)
    ap.add_argument("--init-mode", dest="init_mode",
                    choices=["noise","scaled-noise","y-blend"], default=None)
    ap.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=None)
    ap.add_argument("--cfg-mode", dest="cfg_mode", choices=["const","gauss","tophat"], default=None)
    ap.add_argument("--cfg-center", dest="cfg_center", type=float, default=None)
    ap.add_argument("--cfg-width", dest="cfg_width", type=float, default=None)
    ap.add_argument("--dc-weight", dest="dc_weight", type=float, default=None)

    # conditioning domain
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--whiten-mode", default="auto", choices=["auto","train","welch","model"])
    ap.add_argument("--sigma-mode", default="std", choices=["std","mad","fixed"])
    ap.add_argument("--sigma-fixed", type=float, default=1.0)

    # dataset pairing
    ap.add_argument("--unordered", action="store_true",
                    help="Sort (m1,m2) within each sample so m1>=m2 (lower triangle in data).")

    # scoring window + alignment
    ap.add_argument("--win", choices=["full","tail","merger"], default="full")
    ap.add_argument("--tail-secs", type=float, default=0.8)
    ap.add_argument("--left", type=float, default=0.08)
    ap.add_argument("--right", type=float, default=0.04)
    ap.add_argument("--align", choices=["none","xcorr","peak"], default="none")
    ap.add_argument("--align-max-shift-s", type=float, default=0.02)

    # MAE normalization (CSV reporting)
    ap.add_argument("--mae-norm", choices=["none","sigma","clean"], default="none")

    # which metrics to plot
    ap.add_argument("--metrics", choices=["mae","corr","mae+corr"], default="mae")

    # plot customization
    ap.add_argument("--xlabel", default="M1 (Solar Mass Units)")
    ap.add_argument("--ylabel", default="M2 (Solar Mass Units)")
    ap.add_argument("--title-corr", dest="title_corr", default="Mean correlation (strain)")
    ap.add_argument("--title-mae",  dest="title_mae",  default="Mean MAE (strain)")
    ap.add_argument("--annot", choices=["count","value","none"], default="count")
    ap.add_argument("--annot-fmt", dest="annot_fmt", default=".2f")
    ap.add_argument("--annot-div", dest="annot_div", type=float, default=None,
                    help="If set and --annot value, divide annotation by this (e.g., 1e-22)")
    ap.add_argument("--tick-fmt", default=".0f",
                    help="Format for tick labels (e.g., .0f)")
    ap.add_argument("--max-ticks-x", type=int, default=10,
                    help="Max x tick labels to draw (auto-thins if exceeded)")
    ap.add_argument("--max-ticks-y", type=int, default=10,
                    help="Max y tick labels to draw (auto-thins if exceeded)")
    ap.add_argument("--gridlines", action="store_true", help="Draw gridlines at bin edges")
    ap.add_argument("--gridline-alpha", type=float, default=0.35, help="Gridline transparency")
    ap.add_argument("--gridline-width", type=float, default=0.6, help="Gridline linewidth (pt)")

    # misc
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--min-per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- load model & diffusion ------ #
    ckpt = torch.load(args.model, map_location=device)
    ck = ckpt.get("args", {})
    in_ch = ck.get("in_ch", 3)
    cond_in_ch = ck.get("cond_in_ch", 1)
    base_ch = ck.get("base_ch", 64)
    time_dim = ck.get("time_dim", 128)
    depth = ck.get("depth", 3)
    T = ck.get("T", 1000)
    drop_y_only = bool(ck.get("dropout_y_only", True))
    use_selfcond = (in_ch == (1 + cond_in_ch + 1))
    meta_scale = ck.get("meta_scale", {"M": 80.0, "q": 10.0})
    M_SCALE = float(meta_scale.get("M", 80.0))
    Q_SCALE = float(meta_scale.get("q", 10.0))

    model = inf.UNet1D(
        in_ch=in_ch, base_ch=base_ch, time_dim=time_dim, depth=depth,
        t_embed_max_time=max(0, T-1), cond_in_ch=cond_in_ch, use_selfcond=use_selfcond
    ).to(device)
    if ("model_ema_state" in ckpt):
        try:
            model.load_state_dict(ckpt["model_ema_state"], strict=True)
            print("[info] using EMA weights")
        except Exception as e:
            print("[warn] EMA load failed, using raw:", e)
            model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    diffusion = inf.CustomDiffusion(T=T, device=device)

    # -- resolve inference knobs ---------- #
    knobs = _parse_sweep_best(args.from_sweep) if args.from_sweep else {}
    def _override(k, v):
        if v is not None: knobs[k] = v
    _override("steps", args.steps); _override("eta", args.eta)
    _override("start_snr", args.start_snr); _override("start_t", args.start_t)
    _override("init_mode", args.init_mode)
    _override("cfg_scale", args.cfg_scale); _override("cfg_mode", args.cfg_mode)
    _override("cfg_center", args.cfg_center); _override("cfg_width", args.cfg_width)
    _override("dc_weight", args.dc_weight)

    knobs.setdefault("steps", 300); knobs.setdefault("eta", 0.0)
    knobs.setdefault("init_mode", "scaled-noise")
    knobs.setdefault("cfg_scale", 1.0); knobs.setdefault("cfg_mode", "const")
    knobs.setdefault("cfg_center", 0.70); knobs.setdefault("cfg_width", 0.12)
    knobs.setdefault("dc_weight", 0.0)
    if knobs.get("start_t") is None and knobs.get("start_snr") is not None:
        knobs["start_t"] = inf.t_for_target_snr(diffusion, float(knobs["start_snr"]))
    knobs.setdefault("start_t", max(0, T-1))
    print("[knobs]", json.dumps(knobs, indent=2))

    # --- read mass labels + make *discrete* grid bins ------- #
    import h5py
    with h5py.File(args.input_h5, "r") as f:
        if ("label_m1" in f) and ("label_m2" in f):
            m1_all = np.array(f["label_m1"][:], dtype=np.float32)
            m2_all = np.array(f["label_m2"][:], dtype=np.float32)
        else:
            m1_all = np.array(f["mass1"][:], dtype=np.float32)
            m2_all = np.array(f["mass2"][:], dtype=np.float32)
        q_all  = np.array(f["q"][:], dtype=np.float32) if "q" in f else None
        mc_all = np.array(f["chirp_mass"][:], dtype=np.float32) if "chirp_mass" in f else None
        N = m1_all.shape[0]

    if args.unordered:
        M_arr = np.maximum(m1_all, m2_all); m_arr = np.minimum(m1_all, m2_all)
        default_x_label = args.xlabel or "M1 (Solar Mass Units)"
        default_y_label = args.ylabel or "M2 (Solar Mass Units)"
    else:
        M_arr = m1_all; m_arr = m2_all
        default_x_label = args.xlabel or "M1 (Solar Mass Units)"
        default_y_label = args.ylabel or "M2 (Solar Mass Units)"

    # build edges/centers from unique values
    edges1, x_centers = _midpoint_edges_from_values(M_arr, rdec=6)
    edges2, y_centers = _midpoint_edges_from_values(m_arr, rdec=6)
    G1, G2 = len(x_centers), len(y_centers)

    # prep bins (index lists)
    bins = [[[] for _ in range(G2)] for __ in range(G1)]
    for idx in range(N):
        Mv = float(M_arr[idx]); mv = float(m_arr[idx])
        i1 = int(np.searchsorted(edges1, Mv, side="right") - 1); i1 = int(np.clip(i1, 0, G1-1))
        i2 = int(np.searchsorted(edges2, mv, side="right") - 1); i2 = int(np.clip(i2, 0, G2-1))
        bins[i1][i2].append(idx)

    rng = np.random.default_rng(args.seed)

    # ----- per-index inference -> metrics ---- #
    per_index = []

    def eval_index(idx):
        prep = _prep_sample_from_h5(
            args.input_h5, idx, cond_in_ch,
            args.whiten_mode if args.whiten else None,
            args.sigma_mode, args.sigma_fixed, device,
            M_SCALE, Q_SCALE
        )
        start_t = int(knobs["start_t"])
        x0_hat_norm = inf.ddim_sample(
            model=model, diffusion=diffusion, cond_stack=prep["cond_stack"],
            T=T, steps=int(knobs["steps"]), eta=float(knobs["eta"]),
            device=device, length=prep["L"], debug=False, start_t=start_t,
            init_mode=str(knobs["init_mode"]),
            x0_std_est=0.14, dc_weight=float(knobs["dc_weight"]),
            cond_scale=1.0, eps_scale=1.0, pred_type="eps",
            in_ch=in_ch, cond_in_ch=cond_in_ch, use_selfcond=use_selfcond,
            cfg_scale=float(knobs["cfg_scale"]), cfg_mode=str(knobs["cfg_mode"]),
            cfg_center=float(knobs["cfg_center"]), cfg_width=float(knobs["cfg_width"]),
            cfg_u_only_thresh=0.05, oracle_init=False, clean_norm_311=prep["clean_norm_311"],
            log_jsonl_path=None, log_interval=0,
            xcorr_window_samp=0, delta_t=1.0/float(prep["fs"]),
            amp=args.amp, drop_y_only=drop_y_only
        )
        x0_hat_white = (x0_hat_norm * torch.tensor(prep["sigma"], device=device).view(1,1,1)).detach().cpu().numpy().ravel()
        x0_hat_strain = _dewhiten_back(x0_hat_white, prep) if args.whiten else x0_hat_white

        fs = prep["fs"]
        a = x0_hat_strain.copy()
        b = (prep["clean_raw"].copy()
             if prep["clean_raw"] is not None and len(prep["clean_raw"]) == prep["L"] else None)
        if b is None:
            return dict(idx=int(idx), m1=float(m1_all[idx]), m2=float(m2_all[idx]),
                        q=float(q_all[idx]) if q_all is not None else float("nan"),
                        chirp_mass=float(mc_all[idx]) if mc_all is not None else float("nan"),
                        corr_last=float("nan"), mae_last=float("nan"),
                        nmae_sigma=float("nan"), J=float("nan"))
        if args.win == "full":   s, e = _window_indices_full(len(a), fs)
        elif args.win == "tail": s, e = _window_indices_tail(len(a), fs, args.tail_secs)
        else:                    s, e = _window_indices_merger(b, fs, args.left, args.right)

        a_w, b_w = a[s:e], b[s:e]
        a_w, b_w = _apply_alignment(a_w, b_w, fs, mode=args.align, max_shift_s=args.align_max_shift_s)

        mae = float(np.mean(np.abs(a_w - b_w)))
        nmae_sigma = mae / (prep["sigma"] + 1e-12)
        mean_abs_clean = float(np.mean(np.abs(b_w))) + 1e-12
        nmae_clean = mae / mean_abs_clean

        J = _objective({"corr_last": 0.0, "nmae_sigma": nmae_sigma, "mae_last": mae}, None)

        return dict(
            idx=int(idx),
            m1=float(m1_all[idx]), m2=float(m2_all[idx]),
            q=float(q_all[idx]) if q_all is not None else float("nan"),
            chirp_mass=float(mc_all[idx]) if mc_all is not None else float("nan"),
            corr_last=float("nan"),
            mae_last=mae,
            nmae_sigma=nmae_sigma,
            nmae_clean=nmae_clean,
            J=float(J)
        )

    print("[grid] evaluating per-cell...")
    for i1 in range(G1):
        for i2 in range(G2):
            idxs = bins[i1][i2]
            if len(idxs) < args.min_per_cell: continue
            take = min(len(idxs), max(1, int(args.per_cell)))
            pick = idxs if len(idxs) <= take else rng.choice(idxs, size=take, replace=False)
            pick = list(map(int, pick))
            for idx in pick:
                try:
                    res = eval_index(idx)
                    res["bin_i1"] = int(i1); res["bin_i2"] = int(i2)
                    per_index.append(res)
                except Exception as e:
                    warnings.warn(f"[skip] idx={idx} failed: {e}")

    if not per_index:
        raise RuntimeError("No per-index results. Check dataset paths / settings.")

    # ------ aggregate per bin----- #
    import pandas as pd
    df = pd.DataFrame(per_index)
    df.to_csv(os.path.join(args.outdir, "per_index_metrics.csv"), index=False)

    agg_dict = dict(
        count=("idx","count"),
        mae_mean=("mae_last","mean"),
        mae_median=("mae_last","median"),
        mae_std=("mae_last","std"),
        nmae_sigma_mean=("nmae_sigma","mean"),
        nmae_clean_mean=("nmae_clean","mean"),
        J_mean=("J","mean"),
    )
    if args.metrics in ("corr","mae+corr"):
        agg_dict.update(dict(
            corr_mean=("corr_last","mean"),
            corr_median=("corr_last","median"),
            corr_std=("corr_last","std"),
        ))

    agg = df.groupby(["bin_i1","bin_i2"]).agg(**agg_dict).reset_index()
    agg.to_csv(os.path.join(args.outdir, "metrics_grid.csv"), index=False)
    with open(os.path.join(args.outdir, "metrics_grid.json"), "w") as fh:
        json.dump(agg.to_dict(orient="records"), fh, indent=2)

    #  heatmap arrays
    mae_grid  = np.full((G1, G2), np.nan, dtype=np.float32)
    cnt_grid  = np.zeros((G1, G2), dtype=np.int32)
    corr_grid = np.full((G1, G2), np.nan, dtype=np.float32) if args.metrics in ("corr","mae+corr") else None

    for _, r in agg.iterrows():
        i1, i2 = int(r["bin_i1"]), int(r["bin_i2"])
        mae_grid[i1, i2]  = float(r["mae_mean"])
        cnt_grid[i1, i2]  = int(r["count"])
        if corr_grid is not None and "corr_mean" in r:
            try: corr_grid[i1, i2] = float(r["corr_mean"])
            except Exception: pass

    #  plotting
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150})

    x_lab = default_x_label
    y_lab = default_y_label

    # choose (thinned) tick locations/labels
    xticks = _choose_tick_values(x_centers, max_labels=max(2, args.max_ticks_x))
    yticks = _choose_tick_values(y_centers, max_labels=max(2, args.max_ticks_y))

    def _plot_heat(Z, title, fname, cmap="viridis", vmin=None, vmax=None):
        Zm = np.ma.masked_invalid(Z.T)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(
            Zm, origin="lower", aspect="auto",
            extent=[edges1[0], edges1[-1], edges2[0], edges2[-1]],
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.set_xlabel(x_lab); ax.set_ylabel(y_lab)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9)

        # ticks at the actual mass values (possibly thinned)
        ax.set_xticks(xticks); ax.set_yticks(yticks)
        ax.set_xticklabels([format(v, args.tick_fmt) for v in xticks])
        ax.set_yticklabels([format(v, args.tick_fmt) for v in yticks])

        # gridlines at bin edges (optional)
        if args.gridlines:
            for e in edges1:
                ax.axvline(e, color="w", lw=args.gridline_width, alpha=args.gridline_alpha)
            for e in edges2:
                ax.axhline(e, color="w", lw=args.gridline_width, alpha=args.gridline_alpha)

        # annotations
        for I1 in range(G1):
            for I2 in range(G2):
                c = cnt_grid[I1, I2]
                if c <= 0: continue
                if args.annot == "count":
                    txt = str(int(c))
                elif args.annot == "value":
                    val = Z[I1, I2]
                    if not np.isfinite(val): continue
                    if args.annot_div and args.annot_div != 0:
                        val = val / args.annot_div
                    txt = format(val, args.annot_fmt)
                else:
                    continue
                ax.text(x_centers[I1], y_centers[I2], txt,
                        ha="center", va="center", fontsize=7, color="w", alpha=0.85)

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, fname), bbox_inches="tight")
        plt.close(fig)

    # MAE heatmap
    vals = mae_grid[np.isfinite(mae_grid)]
    lo = float(np.percentile(vals, 5)) if vals.size else None
    hi = float(np.percentile(vals, 95)) if vals.size else None
    _plot_heat(mae_grid, args.title_mae, "heatmap_mae.png",
               cmap="viridis", vmin=lo, vmax=hi)

    # corr heatmap (optional)
    if corr_grid is not None and np.isfinite(corr_grid).any():
        _plot_heat(corr_grid, args.title_corr, "heatmap_corr.png",
                   cmap="magma", vmin=0.0, vmax=1.0)

    # ------- summary --------
    def _safe_mean(a):
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    macro = dict(
        mae_macro_mean=_safe_mean(mae_grid),
        cells_with_data=int(np.isfinite(mae_grid).sum()),
        total_cells=int(G1 * G2),
        per_cell=args.per_cell,
        unique_m1=len(x_centers),
        unique_m2=len(y_centers),
        xticks_used=[float(x) for x in xticks],
        yticks_used=[float(y) for y in yticks],
    )
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(macro, fh, indent=2)
    print("[done] wrote:", args.outdir)
    print(json.dumps(macro, indent=2))

if __name__ == "__main__":
    main()
