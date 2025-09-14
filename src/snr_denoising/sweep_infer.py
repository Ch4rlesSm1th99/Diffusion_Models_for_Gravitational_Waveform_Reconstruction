import os, json, math, argparse, random
import numpy as np
import torch


import inference as inf

def _objective(m_strain, m_white):
    """Higher is better. Combine corr_last (strain + whitened) and penalize normalized MAE."""
    r_s = m_strain.get("corr_last", 0.0) if m_strain else 0.0
    r_w = m_white.get("corr_last", 0.0) if m_white else 0.0
    nmae_sigma = m_strain.get("nmae_sigma", 0.0) if m_strain else 0.0
    return float(r_s + 0.5 * r_w - 0.1 * nmae_sigma)

def _prep_sample_from_h5(h5_path, idx, cond_in_ch, whiten_mode, sigma_mode, sigma_fixed,
                         M_SCALE, Q_SCALE, device, debug=False):
    # load y/clean/psd/meta
    y_raw, clean_raw, fs, P_model_in, (fw_in, Pw_in), meta = inf._load_measurement_from_h5(h5_path, idx)
    L = len(y_raw)

    # ---- whitening (mirror inference.py) -------
    whiten_kind_used = "raw"
    P_train = None; freqs_P = None; P_model_used = None

    if whiten_mode:
        mode = whiten_mode
        if mode == "auto":
            if (fw_in is not None) and (Pw_in is not None):
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
        y_for_cond = y_raw
        clean_for_cond = clean_raw
        whiten_kind_used = "raw"

    # sigma in conditioning domain
    sigma = inf._pick_sigma(y_for_cond, sigma_mode, sigma_fixed)
    fallback = {"train": 2.914e-12, "welch": 2.914e-16, "model": 2.914e-16, "raw": 2.914e-12}
    if (not np.isfinite(sigma)) or (sigma < 1e-20):
        sigma = fallback.get(whiten_kind_used, fallback["train"])

    # y normalized (meta is NOT normalized remember scaling is applied in loader)
    y_norm = (y_for_cond / sigma).astype(np.float32)
    y_norm_311 = torch.from_numpy(y_norm).to(device).view(1, 1, -1)
    clean_norm_311 = None
    if clean_for_cond is not None:
        clean_norm_311 = torch.from_numpy((clean_for_cond / sigma).astype(np.float32)).to(device).view(1, 1, -1)

    #  build cond_stack to match checkpoint
    if cond_in_ch <= 1:
        cond_stack = y_norm_311
    else:
        meta_arr = inf._meta_to_stack(meta, L=L, cond_in_ch=cond_in_ch, M_SCALE=M_SCALE, Q_SCALE=Q_SCALE)
        if meta_arr is None:
            meta_arr = np.zeros((cond_in_ch - 1, L), dtype=np.float32)
        meta_311 = torch.from_numpy(meta_arr).to(device).unsqueeze(0)     # [1, C_meta, L]
        cond_stack = torch.cat([y_norm_311, meta_311], dim=1)         # [1, cond_in_ch, L]

    return dict(
        y_raw=y_raw, clean_raw=clean_raw, y_for_cond=y_for_cond, clean_for_cond=clean_for_cond,
        y_norm_311=y_norm_311, clean_norm_311=clean_norm_311, cond_stack=cond_stack,
        sigma=float(sigma), fs=float(fs), L=len(y_raw),
        whiten_kind_used=whiten_kind_used, P_train=P_train, freqs_P=freqs_P, P_model_used=P_model_used
    )

def _dewhiten_back(x0_hat_white, prep):
    wk = prep["whiten_kind_used"]
    if wk == "train":
        return inf._dewhiten_train_like(x0_hat_white, prep["P_train"])
    elif wk == "welch":
        return inf._dewhiten_welch(x0_hat_white, prep["freqs_P"], prep["fs"])
    elif wk == "model":
        return inf._dewhiten_model(x0_hat_white, prep["P_model_used"])
    return x0_hat_white

def main():
    ap = argparse.ArgumentParser("Sweep inference hyperparameters")
    ap.add_argument("--input-h5", required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=[0,1,2,3,4,5,6,7])
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default=None)

    # match training
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--whiten-mode", default="auto", choices=["auto","train","welch","model"])
    ap.add_argument("--sigma-mode", default="std", choices=["std","mad","fixed"])
    ap.add_argument("--sigma-fixed", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    # -------- grid mode ----
    ap.add_argument("--grid", action="store_true", help="Enable grid search instead of random sweep")
    ap.add_argument("--grid-snr", type=float, nargs="*", default=[0.9, 1.2, 1.6, 2.2],
                    help="Start SNR values for grid mode")
    ap.add_argument("--grid-cfg", type=float, nargs="*", default=[1.5, 1.7, 1.9, 2.1],
                    help="CFG scales for grid mode")
    ap.add_argument("--grid-init", nargs="*", default=["y-blend","scaled-noise"],
                    help="Init modes for grid mode")
    ap.add_argument("--grid-dc", type=float, nargs="*", default=[0.0, 0.05],
                    help="DC weights for grid mode")
    ap.add_argument("--grid-eta", type=float, nargs="*", default=[0.0],
                    help="DDIM eta values for grid mode")
    ap.add_argument("--grid-steps", type=int, default=200, help="Steps per sample in grid mode")

    # ------ random sweep (coarse --> refine) --------
    ap.add_argument("--n-coarse", type=int, default=64)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--steps-coarse", type=int, default=100)
    ap.add_argument("--steps-refine", type=int, default=200)
    ap.add_argument("--seeds-refine", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # ranges for random sweep
    ap.add_argument("--start-snr-min", type=float, default=0.8)
    ap.add_argument("--start-snr-max", type=float, default=3.0)
    ap.add_argument("--cfg-min", type=float, default=1.0)
    ap.add_argument("--cfg-max", type=float, default=3.0)
    ap.add_argument("--cfg-mode", default="auto", choices=["auto","const","gauss"])
    ap.add_argument("--cfg-center-min", type=float, default=0.55)
    ap.add_argument("--cfg-center-max", type=float, default=0.80)
    ap.add_argument("--cfg-width-min", type=float, default=0.08)
    ap.add_argument("--cfg-width-max", type=float, default=0.18)
    ap.add_argument("--dc-choices", type=float, nargs="*", default=[0.0,0.05,0.10,0.15])
    ap.add_argument("--init-choices", nargs="*", default=["y-blend","scaled-noise"])
    ap.add_argument("--eta-choices", type=float, nargs="*", default=[0.0])

    args = ap.parse_args()

    # seeds + device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)

    # ---- load model once
    ckpt = torch.load(args.model, map_location=device)
    ck = ckpt.get("args", {})
    in_ch = ck.get("in_ch", 3)
    cond_in_ch = ck.get("cond_in_ch", 1)           # how many conditional channels the model expects
    base_ch = ck.get("base_ch", 64)
    time_dim = ck.get("time_dim", 128)
    depth = ck.get("depth", 3)
    T = ck.get("T", 1000)
    drop_y_only = bool(ck.get("dropout_y_only", True))
    use_selfcond = (in_ch == (1 + cond_in_ch + 1))
    meta_scale = ck.get("meta_scale", {"M": 80.0, "q": 10.0})
    M_SCALE = float(meta_scale.get("M", 80.0))
    Q_SCALE = float(meta_scale.get("q", 10.0))

    model = inf.UNet1D(in_ch=in_ch, base_ch=base_ch, time_dim=time_dim, depth=depth,
                       t_embed_max_time=max(0, T-1),
                       cond_in_ch=cond_in_ch, use_selfcond=use_selfcond).to(device)
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

    # ---- prepare data once (using cond_in_ch to build cond_stack) -------
    preps = []
    for idx in args.indices:
        p = _prep_sample_from_h5(args.input_h5, idx, cond_in_ch,
                                 args.whiten_mode if args.whiten else None,
                                 args.sigma_mode, args.sigma_fixed,
                                 M_SCALE, Q_SCALE, device)
        preps.append(p)

    # helper: evaluate one combo (averaged over indices)
    def eval_combo(combo, steps, seed=None):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        scores = []
        for p in preps:
            start_t = inf.t_for_target_snr(diffusion, combo["start_snr"])
            x0_hat_norm = inf.ddim_sample(
                model=model, diffusion=diffusion, cond_stack=p["cond_stack"],
                T=T, steps=steps, eta=combo["eta"], device=device, length=p["L"],
                debug=False, start_t=start_t, init_mode=combo["init_mode"],
                x0_std_est=0.14,
                dc_weight=combo["dc_weight"], cond_scale=1.0, eps_scale=1.0, pred_type="eps",
                in_ch=in_ch, cond_in_ch=cond_in_ch, use_selfcond=use_selfcond,
                cfg_scale=combo["cfg_scale"], cfg_mode=combo["cfg_mode"],
                cfg_center=combo["cfg_center"], cfg_width=combo["cfg_width"],
                cfg_u_only_thresh=0.05,
                oracle_init=False, clean_norm_311=p["clean_norm_311"],
                log_jsonl_path=None, log_interval=0,
                xcorr_window_samp=0, delta_t=1.0/float(p["fs"]), amp=args.amp,
                drop_y_only=drop_y_only
            )
            # to whitened domain then dewhiten to strain
            x0_hat_white = (x0_hat_norm * torch.tensor(p["sigma"], device=device).view(1,1,1)).detach().cpu().numpy().ravel()
            x0_hat_strain = _dewhiten_back(x0_hat_white, p) if args.whiten_mode else x0_hat_white

            # metrics over last 0.8 s (merger for msot samples)
            m_strain = None; m_white = None
            if p["clean_raw"] is not None and len(p["clean_raw"]) == p["L"]:
                m_strain = inf._score_last_window(x0_hat_strain, p["clean_raw"], p["fs"], secs=0.8)
                # add normalized MAE wrt sigma
                L = min(len(x0_hat_strain), len(p["clean_raw"]))
                w = int(p["fs"] * 0.8)
                a = x0_hat_strain[L-w:]; b = p["clean_raw"][L-w:]
                mae = float(np.mean(np.abs(a-b)))
                m_strain["nmae_sigma"] = mae / (p["sigma"] + 1e-12)
            if p["clean_for_cond"] is not None:
                m_white = inf._score_last_window(x0_hat_white, p["clean_for_cond"], p["fs"], secs=0.8)

            scores.append((_objective(m_strain, m_white), m_strain, m_white))

        J = float(np.mean([s[0] for s in scores])) if scores else -1e9
        return J, scores

    # ------ GRID SEARCH ----------
    if args.grid:
        grid = []
        for snr in args.grid_snr:
            for cfg in args.grid_cfg:
                for init in args.grid_init:
                    for dc in args.grid_dc:
                        for et in args.grid_eta:
                            combo = dict(start_snr=float(snr), cfg_scale=float(cfg),
                                         cfg_mode=("gauss" if init=="y-blend" else "const"),  # good defaults
                                         cfg_center=0.70, cfg_width=0.12,
                                         dc_weight=float(dc), init_mode=init, eta=float(et))
                            J, _ = eval_combo(combo, steps=args.grid_steps)
                            grid.append({**combo, "J": J})
        grid = sorted(grid, key=lambda z: z["J"], reverse=True)
        with open(os.path.join(args.outdir, "grid_results.json"), "w") as fh:
            json.dump(grid, fh, indent=2)
        best = grid[0]
        start_t = inf.t_for_target_snr(diffusion, best["start_snr"])
        cmd = [
            "python", "inference.py",
            "--input-h5", args.input_h5, "--index", str(args.indices[0]),
            "--model", args.model, "--outdir", os.path.join(args.outdir, "best"),
            "--steps", str(args.grid_steps),
            "--eta", f"{best['eta']:.2f}",
            "--start-snr", f"{best['start_snr']:.3f}",
            "--init-mode", best["init_mode"],
            "--cfg-scale", f"{best['cfg_scale']:.2f}",
            "--cfg-mode", best["cfg_mode"],
            "--cfg-center", f"{best['cfg_center']:.2f}",
            "--cfg-width", f"{best['cfg_width']:.2f}",
            "--dc-weight", f"{best['dc_weight']:.2f}",
            "--sigma-mode", args.sigma_mode,
        ]
        if args.whiten: cmd += ["--whiten", "--whiten-mode", args.whiten_mode]
        if args.amp: cmd += ["--amp"]
        with open(os.path.join(args.outdir, "best_cmd.txt"), "w") as fh:
            fh.write(" ".join(cmd) + "\n")
        print("\n[grid] Best J =", best["J"])
        print("[grid] approx start_t =", start_t)
        print("[grid] suggested command:\n  " + " ".join(cmd))
        return

    # ---- RANDOM SWEEP (coarse --> refine) ----
    print(f"[sweep] Stage A: {args.n_coarse} combos at {args.steps_coarse} steps on {len(preps)} samples")
    def sample_combo():
        cfg_mode = args.cfg_mode
        if cfg_mode == "auto": cfg_mode = "gauss" if (random.random() < 0.7) else "const"
        return dict(
            start_snr = 10 ** np.random.uniform(math.log10(args.start_snr_min), math.log10(args.start_snr_max)),
            cfg_scale = np.random.uniform(args.cfg_min, args.cfg_max),
            cfg_mode  = cfg_mode,
            cfg_center= np.random.uniform(args.cfg_center_min, args.cfg_center_max),
            cfg_width = np.random.uniform(args.cfg_width_min, args.cfg_width_max),
            dc_weight = float(random.choice(args.dc_choices)),
            init_mode = random.choice(args.init_choices),
            eta       = float(random.choice(args.eta_choices)),
        )

    coarse = []
    for i in range(args.n_coarse):
        c = sample_combo()
        J, _ = eval_combo(c, steps=args.steps_coarse)
        coarse.append({**c, "J_coarse": J})
    coarse = sorted(coarse, key=lambda z: z["J_coarse"], reverse=True)
    top = coarse[:args.topk]
    with open(os.path.join(args.outdir, "coarse_top.json"), "w") as fh:
        json.dump(top, fh, indent=2)

    print(f"[sweep] Stage B: top {args.topk} --> refine at {args.steps_refine} steps, seeds={args.seeds_refine}")
    finals = []
    for c in top:
        JJ = []
        for s in range(args.seeds_refine):
            J, _ = eval_combo(c, steps=args.steps_refine, seed=args.seed + s)
            JJ.append(J)
        finals.append({**c, "J_refine_mean": float(np.mean(JJ)), "J_refine_std": float(np.std(JJ))})
    finals = sorted(finals, key=lambda z: z["J_refine_mean"], reverse=True)
    with open(os.path.join(args.outdir, "final_results.json"), "w") as fh:
        json.dump(finals, fh, indent=2)

    best = finals[0]
    start_t = inf.t_for_target_snr(diffusion, best["start_snr"])
    cmd = [
        "python", "inference.py",
        "--input-h5", args.input_h5, "--index", str(args.indices[0]),
        "--model", args.model, "--outdir", os.path.join(args.outdir, "best"),
        "--steps", str(args.steps_refine),
        "--eta", f"{best['eta']:.2f}",
        "--start-snr", f"{best['start_snr']:.3f}",
        "--init-mode", best["init_mode"],
        "--cfg-scale", f"{best['cfg_scale']:.2f}",
        "--cfg-mode", best["cfg_mode"],
        "--cfg-center", f"{best['cfg_center']:.2f}",
        "--cfg-width", f"{best['cfg_width']:.2f}",
        "--dc-weight", f"{best['dc_weight']:.2f}",
        "--sigma-mode", args.sigma_mode,
    ]
    if args.whiten: cmd += ["--whiten", "--whiten-mode", args.whiten_mode]
    if args.amp: cmd += ["--amp"]
    with open(os.path.join(args.outdir, "best_cmd.txt"), "w") as fh:
        fh.write(" ".join(cmd) + "\n")

    print("\n[best] Refine mean J =", best["J_refine_mean"], "std=", best["J_refine_std"])
    print("[best] approx start_t =", start_t)
    print("[best] suggested command:\n  " + " ".join(cmd))

if __name__ == "__main__":
    main()
