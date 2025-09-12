import argparse
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from models import UNet1D, CustomDiffusion
from dataloader import make_dataloader

def prepare_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, 'latest_model')
    if os.path.exists(out_dir):
        for file in os.listdir(out_dir):
            try:
                os.remove(os.path.join(out_dir, file))
            except Exception:
                pass
    else:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir

def set_seed(seed: int):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def _predict_x0_norm(model, diffusion, x_t, cond_stack, t):
    """
    One-step x0 prediction from x_t using the full conditional stack (y + meta...).
    """
    t = t.long()
    zeros_sc = torch.zeros_like(x_t)
    net_in = torch.cat([x_t, cond_stack, zeros_sc], dim=1)
    eps_hat = model(net_in, t)
    ab = diffusion.alpha_bar[t].view(-1, 1, 1)
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
    return x0_hat_norm.detach()

def _element_loss(eps_hat, eps, mask, loss_type: str, huber_beta: float):
    if loss_type == "huber":
        el = F.smooth_l1_loss(eps_hat, eps, reduction="none", beta=huber_beta)
    else:
        el = (eps_hat - eps) ** 2  # mse
    return el * mask

def _corr_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    den = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt() + 1e-12)
    return (a * b).sum() / den

def _log_jsonl(path: str, obj: dict):
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")

# EMA helpers
@torch.no_grad()
def update_ema(ema_model, model, decay: float):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for k in ema_params.keys():
        ema_params[k].data.mul_(decay).add_(model_params[k].data, alpha=(1.0 - decay))
    # buffers
    for (eb, mb) in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(mb)

# LR schedule --> linear warmup then cosine
def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.1):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_scale + 0.5 * (1 - min_lr_scale) * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def _resolve_h5(path: str) -> str:
    if os.path.isdir(path):
        cands = [os.path.join(path, f) for f in os.listdir(path)
                 if f.lower().endswith(('.h5', '.hdf5'))]
        cands.sort(key=os.path.getmtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"No .h5/.hdf5 in directory: {path}")
        return cands[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 path not found: {path}")
    return path

def _compute_meta_scale(h5_file: str) -> dict:
    """
    Compute dataset-adaptive scales for labels; scaling by 95th percentile by default.
    """
    scale = {"M": 80.0, "q": 10.0}  # some reasonable defaults
    try:
        import h5py
        with h5py.File(h5_file, "r") as f:
            def p95(name):
                if name in f:
                    arr = np.array(f[name][:], dtype=np.float64)
                    if arr.size:
                        return float(np.nanpercentile(arr, 95))
                return None

            m1_p = p95("mass1"); m2_p = p95("mass2"); mc_p = p95("chirp_mass")
            q_p  = p95("q")

            Ms = [x for x in [m1_p, m2_p, mc_p] if (x is not None and np.isfinite(x) and x > 0)]
            if Ms:
                scale["M"] = float(max(Ms))   # one mass scale for m1, m2
            if (q_p is not None) and np.isfinite(q_p) and q_p > 0:
                scale["q"] = float(q_p)
    except Exception as e:
        print(f"[train] meta_scale computation failed; using defaults {scale} ({e})")
    return scale

# batch matcher (safe for enabling t_multi wen stratified t-sampling)
def _match_batch(a: torch.Tensor, target_bsz: int) -> torch.Tensor:
    """
    Return 'a' with batch dimension == target_bsz.
    If a.shape[0] divides target_bsz, repeat_interleave; otherwise trim.
    """
    if a.shape[0] == target_bsz or a.shape[0] == 0:
        return a
    if target_bsz % a.shape[0] == 0:
        rep = target_bsz // a.shape[0]
        return a.repeat_interleave(rep, dim=0)
    rep = (target_bsz + a.shape[0] - 1) // a.shape[0]
    return a.repeat_interleave(rep, dim=0)[:target_bsz]

# ------------- stratified timestep sampler -----------
def _sample_timesteps_stratified(bsz: int, t_min: int, t_max: int, device, bins: int = 0) -> torch.Tensor:
    """
    Return 'bsz' timesteps roughly covering [t_min, t_max] uniformly by strata.
    bins: number of buckets; if 0, use bsz.
    """
    b = int(bins) if bins and bins > 0 else int(bsz)
    b = max(1, min(b, bsz))
    edges = torch.linspace(t_min, t_max + 1, b + 1, device=device).long()
    q, r = divmod(bsz, b)
    counts = [q + 1 if i < r else q for i in range(b)]
    ts = []
    for i in range(b):
        lo = int(edges[i].item())
        hi = int(edges[i + 1].item()) - 1
        if hi < lo:
            hi = lo
        if counts[i] > 0:
            ts.append(torch.randint(lo, hi + 1, (counts[i],), device=device))
    t = torch.cat(ts, dim=0) if ts else torch.randint(t_min, t_max + 1, (bsz,), device=device)
    if t.numel() > bsz:
        t = t[:bsz]
    elif t.numel() < bsz:
        pad = torch.randint(t_min, t_max + 1, (bsz - t.numel(),), device=device)
        t = torch.cat([t, pad], dim=0)
    perm = torch.randperm(bsz, device=device)
    return t[perm].long()

def train_diffusion(args):
    set_seed(args.seed)
    out_dir = prepare_output_dir(args.model_dir)

    # compute label scaling BEFORE creating the loader
    h5_path = _resolve_h5(args.data)
    meta_scale = _compute_meta_scale(h5_path)
    print(f"[train] meta_scale (dataset-adaptive): {meta_scale}")

    # create loade ---> pass mass_scale for m1/m2 scaling
    loader: DataLoader = make_dataloader(
        h5_path=args.data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        whiten=args.whiten,
        whiten_mode=args.whiten_mode,
        sigma_mode=args.sigma_mode,
        sigma_fixed=args.sigma_fixed,
        include_metadata=True,
        mass_scale=float(meta_scale.get("M", 80.0)),
    )

    print(f"[train] Dataset size: {len(loader.dataset)}")
    print(f"[train] Batches/epoch: {len(loader)} , device={args.device} , AMP={args.amp}")
    try:
        ds = loader.dataset
        if hasattr(ds, "_ensure_open"):
            ds._ensure_open()
        h5p = getattr(ds, "h5_path", None)
        fs  = getattr(ds, "fs", None)
        h5  = getattr(ds, "h5", None)
        print(f"[train] HDF5 path: {h5p}")
        print(f"[train] sampling_rate (fs): {fs}")
        print(f"[train] training whitening: {ds.whiten} , whiten_mode(req)={ds.whiten_mode} , sigma_mode={ds.sigma_mode}")
        if h5 is not None:
            ha = h5.attrs
            print(f"[train] H5 attrs -> time_axis={ha.get('time_axis','?')} padding={ha.get('padding','?')}")
            print(f"[train] PSD saved? {bool(ha.get('psd_saved', False))} , model_kind={ha.get('psd_model_kind','')}")
    except Exception as e:
        print(f"[train] dataset banner failed: {e}")

    device = torch.device(args.device)

    # peek one batch to infer meta channels (unless passed 0's)
    try:
        _peek = next(iter(loader))
    except StopIteration:
        raise RuntimeError("Empty dataset")

    # dataloader may return 4 or 5 tensors (clean, noisy, sigma, mask [, meta_stack])
    if isinstance(_peek, (list, tuple)) and len(_peek) == 5:
        _, _, _, _, meta_peek = _peek
        C_meta = int(meta_peek.shape[1])
        meta_enabled = True
    else:
        C_meta = 0
        meta_enabled = False

    # conditional channels = 1 (y) + C_meta (metadata)
    cond_in_ch = 1 + C_meta
    # total input channels = x_t (1) + cond (cond_in_ch) + selfcond (1)
    use_selfcond = True
    in_ch = 1 + cond_in_ch + (1 if use_selfcond else 0)

    print(f"[train] meta_enabled={meta_enabled} , C_meta={C_meta} , cond_in_ch={cond_in_ch} , in_ch={in_ch}")

    # pass t_embed_max_time=T-1 to match normalized time embedding
    model = UNet1D(
        in_ch=in_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        depth=args.depth,
        t_embed_max_time=max(0, args.T - 1),
        cond_in_ch=cond_in_ch,
        use_selfcond=use_selfcond,
    ).to(device)

    diffusion = CustomDiffusion(T=args.T, device=device)

    # load initial weights if requested (EMA preferred)
    if getattr(args, 'init_from', None):
        ckpt = torch.load(args.init_from, map_location=device)
        state = ckpt.get('model_ema_state', ckpt.get('model_state'))
        model.load_state_dict(state, strict=True)
        print(f"[init] loaded weights from {args.init_from} (EMA={'model_ema_state' in ckpt})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    scheduler = None
    if args.warmup_steps > 0 or args.cosine_decay:
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
            min_lr_scale=args.min_lr_scale,
        )

    # EMA
    ema_model = None
    if args.ema:
        ema_model = deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # AMP
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    run_start = time.time()
    global_step = 0

    # control: run deep first-batch probe only on epoch 1 (and optionally every N epochs)
    def _do_debug_this_epoch(epoch_idx: int) -> bool:
        if not args.debug_first:
            return False
        if epoch_idx == 1:
            return True
        if args.debug_first_every and (epoch_idx % args.debug_first_every == 0):
            return True
        return False

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss_weighted = 0.0
        sum_weight = 0
        batch_losses = []
        skipped_batches = 0

        # epoch-level knobs
        t_min_epoch = int(max(0, min(args.T - 1, int(args.t_min_frac * args.T))))
        p_uncond_eff = 0.0 if epoch <= args.force_cond_epochs else args.p_uncond
        p_selfcond_eff = 0.0 if epoch <= args.force_cond_epochs else args.p_selfcond
        print(f"[train] Epoch {epoch}/{args.epochs} :: p_uncond_eff={p_uncond_eff:.2f} "
              f"p_selfcond_eff={p_selfcond_eff:.2f} , t_min={t_min_epoch}")

        pbar = tqdm(
            loader, total=len(loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch", dynamic_ncols=True, leave=True, mininterval=0.1,
        )

        printed_debug_this_epoch = False

        for i, batch in enumerate(pbar):
            # unpack batch; support 4-return (legacy) and 5-return (with meta)
            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                clean_raw, noisy_raw, sigma, mask, meta_stack = batch
                meta_stack = meta_stack.to(device, non_blocking=True).float()
            else:
                clean_raw, noisy_raw, sigma, mask = batch
                meta_stack = None

            clean_raw = clean_raw.to(device, non_blocking=True).float()
            noisy_raw = noisy_raw.to(device, non_blocking=True).float()
            sigma = sigma.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).float()
            bsz, _, L = clean_raw.shape

            # normalize time-series channels by sigma (match whitening domain)
            sigma_ = sigma.view(-1, 1, 1)
            clean_norm = clean_raw / sigma_
            y_norm = noisy_raw / sigma_

            # metadata channels scaled in dataloader
            if meta_stack is not None:
                if meta_stack.size(-1) != y_norm.size(-1):
                    meta_stack = F.interpolate(meta_stack, size=y_norm.size(-1),
                                               mode="linear", align_corners=False)
                cond_stack = torch.cat([y_norm, meta_stack], dim=1)  # [B, 1+C_meta, L]
            else:
                cond_stack = y_norm  # [B, 1, L]

            # optional clamping of normalized signals (keeps meta untouched)
            if args.clamp_inputs > 0:
                clean_norm = clean_norm.clamp(-args.clamp_inputs, args.clamp_inputs)
                y_norm = y_norm.clamp(-args.clamp_inputs, args.clamp_inputs)

            # ------------ timestep sampling (stratified + optional multi-repeats) ----------
            t_min = t_min_epoch
            t_max = args.T - 1

            K = max(1, int(args.t_multi))
            if K > 1:
                clean_norm = clean_norm.repeat_interleave(K, dim=0)
                y_norm     = y_norm.repeat_interleave(K, dim=0)
                mask       = mask.repeat_interleave(K, dim=0)
                sigma      = sigma.repeat_interleave(K, dim=0)
                if meta_stack is not None:
                    meta_stack = meta_stack.repeat_interleave(K, dim=0)
                if meta_stack is not None:
                    cond_stack = torch.cat([y_norm, meta_stack], dim=1)
                else:
                    cond_stack = y_norm

            bsz_eff = clean_norm.size(0)

            if args.t_cover == 'strat':
                t = _sample_timesteps_stratified(bsz_eff, t_min, t_max, device, bins=args.t_bins)
            else:
                t = torch.randint(t_min, args.T, (bsz_eff,), device=device, dtype=torch.long)

            with torch.amp.autocast('cuda', enabled=args.amp):
                x_t, eps = diffusion.q_sample(clean_norm, t)

                if args.clamp_inputs > 0:
                    x_t = x_t.clamp(-args.clamp_inputs, args.clamp_inputs)

                # CFG dropout (y-only if requested) -- uses batch size
                if p_uncond_eff > 0.0:
                    drop = (torch.rand(x_t.size(0), 1, 1, device=device) < p_uncond_eff).float()
                    if (meta_stack is not None) and args.dropout_y_only:
                        y_used = y_norm * (1.0 - drop)
                        if meta_stack.size(-1) != y_used.size(-1):
                            meta_used = F.interpolate(meta_stack, size=y_used.size(-1),
                                                      mode="linear", align_corners=False)
                        else:
                            meta_used = meta_stack
                        cond_used = torch.cat([y_used, meta_used], dim=1)
                    else:
                        cond_used = cond_stack * (1.0 - drop)
                else:
                    cond_used = cond_stack

                # self-conditioning (can be forced off)
                if p_selfcond_eff > 0.0 and torch.rand((), device=device) < p_selfcond_eff:
                    with torch.no_grad():
                        x0_sc = _predict_x0_norm(model, diffusion, x_t, cond_used, t)
                else:
                    x0_sc = torch.zeros_like(x_t)

                net_in = torch.cat([x_t, cond_used, x0_sc], dim=1)
                eps_hat = model(net_in, t)

                # base element loss
                el = _element_loss(eps_hat, eps, mask, args.loss, args.huber_beta)

                # timestep weighting
                if args.loss_weight_power != 0.0:
                    ab = diffusion.alpha_bar[t].view(-1, 1, 1)
                    w = (1.0 - ab).pow(args.loss_weight_power)
                    el = el * w

                denom = mask.sum(dim=[1, 2]).clamp_min(1.0)
                per_sample = el.sum(dim=[1, 2]) / denom
                loss = per_sample.mean()

            # remove any bad behaving batches
            if not torch.isfinite(loss):
                pbar.write("[warn] non-finite loss - batch skipped")
                skipped_batches += 1
                continue
            if args.skip_bad_batches and loss.item() > args.skip_loss_threshold:
                with torch.no_grad():
                    smin = sigma.min().item(); smean = sigma.mean().item(); smax = sigma.max().item()
                    yn_max = y_norm.abs().max().item()
                pbar.write(f"[warn] loss {loss.item():.3e} > {args.skip_loss_threshold} "
                           f"skip (sigma[min/mean/max]={smin:.3e}/{smean:.3e}/{smax:.3e}, "
                           f"(y_norm)_max={yn_max:.3e})")
                skipped_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # grad clip (unscale first if AMP)
            grad_norm = None
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item())

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            # EMA update
            if ema_model is not None:
                update_ema(ema_model, model, args.ema_decay)

            # stats
            loss_val = float(loss.item())
            batch_losses.append(loss_val)
            sum_loss_weighted += loss_val * bsz_eff
            sum_weight += bsz_eff
            running_avg = sum_loss_weighted / max(1, sum_weight)
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.6f}", avg=f"{running_avg:.6f}", lr=f"{lr_now:.2e}")

            # JSONL (the per-batch log)
            _log_jsonl(args.log_jsonl, {
                "phase": "train_batch",
                "step": global_step,
                "epoch": epoch,
                "batch": i,
                "t_min": int(t.min().item()),
                "t_mean": float(t.float().mean().item()),
                "t_max": int(t.max().item()),
                "loss": loss_val,
                "grad_norm": grad_norm,
                "lr": lr_now,
            })
            global_step += 1

            # optional conditioning probe at selected timesteps
            if args.probe_cond and (i % max(1, args.probe_interval) == 0):
                with torch.cuda.amp.autocast(enabled=args.amp):
                    c0 = (clean_norm[0:1]).detach()
                    y0 = (y_norm[0:1]).detach()
                    zeros_y = torch.zeros_like(y0)
                    zeros_sc = torch.zeros_like(c0)

                    if meta_stack is not None:
                        m0 = (meta_stack[0:1]).detach()
                        cond_on  = torch.cat([y0, m0], dim=1)
                        cond_off = torch.cat([zeros_y, m0], dim=1)  # drop y only
                    else:
                        cond_on  = y0
                        cond_off = zeros_y

                    for t_pick in args.probe_t:
                        t_probe = torch.tensor([max(0, min(args.T - 1, int(t_pick)))],
                                               device=device, dtype=torch.long)
                        x_t_p, eps_p = diffusion.q_sample(c0, t_probe)
                        net_on  = torch.cat([x_t_p, cond_on,  zeros_sc], dim=1)
                        net_off = torch.cat([x_t_p, cond_off, zeros_sc], dim=1)
                        eps_on = model(net_on,  t_probe)
                        eps_off= model(net_off, t_probe)
                        mse_on = float(F.mse_loss(eps_on, eps_p).item())
                        mse_off= float(F.mse_loss(eps_off, eps_p).item())
                        corr_on= float(_corr_torch(eps_on, eps_p).item())
                        corr_off=float(_corr_torch(eps_off, eps_p).item())
                        delta = eps_on - eps_off
                        delta_rms = float(torch.linalg.norm(delta.reshape(-1)) / (delta.numel() ** 0.5))
                        _log_jsonl(args.log_jsonl, {
                            "phase": "probe",
                            "epoch": epoch,
                            "batch": i,
                            "t": int(t_pick),
                            "mse_on": mse_on,
                            "mse_off": mse_off,
                            "corr_on": corr_on,
                            "corr_off": corr_off,
                            "cond_delta_rms": delta_rms,
                        })

            # deep first-batch debug (metrics + quick npy dump)
            if (i == 0) and (not printed_debug_this_epoch) and _do_debug_this_epoch(epoch):
                printed_debug_this_epoch = True

                def _stats(x: torch.Tensor):
                    return x.min().item(), x.max().item(), x.mean().item(), x.std().item()

                cn_min, cn_max, cn_mean, cn_std = _stats(clean_norm)
                yn_min, yn_max, yn_mean, yn_std = _stats(y_norm)
                ct_min, ct_max, ct_mean, ct_std = _stats(x_t)
                eh_min, eh_max, eh_mean, eh_std = _stats(eps_hat)

                print(f"[DEBUG] sigma[min/mean/max]={sigma.min().item():.3e}/{sigma.mean().item():.3e}/{sigma.max().item():.3e}")
                print(f"[DEBUG] clean_norm: min={cn_min:.3e}, max={cn_max:.3e}, mean={cn_mean:.3e}, std={cn_std:.3e}")
                print(f"[DEBUG] y_norm:     min={yn_min:.3e}, max={yn_max:.3e}, mean={yn_mean:.3e}, std={yn_std:.3e}")
                if meta_stack is not None:
                    ms_min, ms_max, ms_mean, ms_std = meta_stack.min().item(), meta_stack.max().item(), meta_stack.mean().item(), meta_stack.std().item()
                    print(f"[DEBUG] meta_stack: min={ms_min:.3e}, max={ms_max:.3e}, mean={ms_mean:.3e}, std={ms_std:.3e}")
                print(f"[DEBUG] x_t:        min={ct_min:.3e}, max={ct_max:.3e}, mean={ct_mean:.3e}, std={ct_std:.3e}")
                print(f"[DEBUG] eps_hat:    min={eh_min:.3e}, max={eh_max:.3e}, mean={eh_mean:.3e}, std={eh_std:.3e}")

                ab = diffusion.alpha_bar[t].view(-1, 1, 1)
                x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
                x0_hat = x0_hat_norm * sigma.view(-1, 1, 1)

                def _corr_masked(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
                    a1 = a[m > 0.5].reshape(-1)
                    b1 = b[m > 0.5].reshape(-1)
                    if a1.numel() < 2 or b1.numel() < 2:
                        return float('nan')
                    a1 = a1 - a1.mean()
                    b1 = b1 - b1.mean()
                    den = (a1.pow(2).sum().sqrt() * b1.pow(2).sum().sqrt() + 1e-12)
                    return float((a1 * b1).sum() / den)

                m0 = (mask[0:1]).float()  # [1,1,L] valid=1, pad=0

                # masked correlations
                corr_x0n = _corr_masked(x0_hat_norm[0:1], clean_norm[0:1], m0)
                # ensure raw clean matches effective batch for MSE/corr
                clean_raw_eff = _match_batch(clean_raw, x0_hat.size(0))
                corr_x0w = _corr_masked(x0_hat[0:1], clean_raw_eff[0:1], m0)

                # masked MSEs
                mse_x0n = (((x0_hat_norm - clean_norm) ** 2) * m0).sum() / m0.sum().clamp_min(1.0)
                mse_x0w = (((x0_hat - clean_raw_eff) ** 2) * m0).sum() / m0.sum().clamp_min(1.0)

                valid_frac = float(m0.mean().item())
                print(f"[DEBUG] valid_frac={valid_frac:.3f} , "
                      f"corr_masked(x0_norm, clean_norm)={corr_x0n:.3f} "
                      f"corr_masked(x0, clean)={corr_x0w:.3f} , "
                      f"MSE_masked_norm={mse_x0n.item():.3e} , MSE_masked_white={mse_x0w.item():.3e}")

                if epoch == 1:
                    try:
                        dbg_dir = os.path.join(out_dir, "debug_batch0")
                        os.makedirs(dbg_dir, exist_ok=True)
                        np.save(os.path.join(dbg_dir, "clean_raw.npy"), clean_raw[0].detach().cpu().numpy().ravel())
                        np.save(os.path.join(dbg_dir, "y_norm.npy"), y_norm[0].detach().cpu().numpy().ravel())
                        np.save(os.path.join(dbg_dir, "x0_hat.npy"), x0_hat[0].detach().cpu().numpy().ravel())
                        print(f"[DEBUG] wrote debug npys -> {dbg_dir}")
                    except Exception as e:
                        print("[DEBUG] npy-dump skipped:", e)

        avg_loss_per_sample = sum_loss_weighted / max(1, sum_weight)
        avg_loss_per_batch = float(np.mean(batch_losses)) if batch_losses else float("nan")
        med_loss_per_batch = float(np.median(batch_losses)) if batch_losses else float("nan")
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Avg(per-sample)={avg_loss_per_sample:.6f} "
            f"Mean(per-batch)={avg_loss_per_batch:.6f} "
            f"Median(per-batch)={med_loss_per_batch:.6f}"
        )
        _log_jsonl(args.log_jsonl, {
            "phase": "epoch_end",
            "epoch": epoch,
            "avg_per_sample": avg_loss_per_sample,
            "mean_per_batch": avg_loss_per_batch,
            "median_per_batch": med_loss_per_batch,
            "skipped_batches": skipped_batches,
            "elapsed_s": float(time.time() - run_start),
        })

    # checkpoints (raw + EMA if enabled)
    save_path = os.path.join(out_dir, 'model_diffusion.pth')
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': {
            **vars(args),
            'conditional': True,
            'in_ch': in_ch,
            'cond_in_ch': cond_in_ch,
            'meta_enabled': meta_enabled,
            'meta_channels': C_meta,
            'conditioning': ('concat[y + meta]+selfcond' if meta_enabled else 'concat[y]+selfcond'),
            'whiten': args.whiten,
            'whiten_mode': args.whiten_mode,
            'sigma_mode': args.sigma_mode,
            'dropout_y_only': bool(args.dropout_y_only),
            'meta_scale': meta_scale,  # <--- save for inference reuse
        },
        'epoch': args.epochs,
    }
    if ema_model is not None:
        payload['model_ema_state'] = ema_model.state_dict()
    torch.save(payload, save_path)
    print(f"Saved model to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train conditional diffusion denoiser on LIGO waveforms (y + optional metadata)")
    parser.add_argument('--data',       type=str, required=True)
    parser.add_argument('--model_dir',  type=str, default='model')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--T',          type=int,   default=1000)
    parser.add_argument('--base_ch',    type=int,   default=64)
    parser.add_argument('--time_dim',   type=int,   default=128)
    parser.add_argument('--depth',      type=int,   default=3)
    parser.add_argument('--device',     type=str,   default=None)
    parser.add_argument('--num_workers',type=int,   default=4)
    parser.add_argument('--seed',       type=int,   default=42)

    # guidance & self-conditioning
    parser.add_argument('--p_uncond',   type=float, default=0.2)
    parser.add_argument('--p_selfcond', type=float, default=0.5)

    # training schedule tweaks
    parser.add_argument('--t_min_frac', type=float, default=0.5,
                        help='sample t uniformly from [t_min_frac*T, T-1]')
    parser.add_argument('--force_cond_epochs', type=int, default=0,
                        help='for first N epochs, set p_uncond=0 and p_selfcond=0')

    # timestep coverage controls
    parser.add_argument('--t_cover', choices=['rand','strat'], default='rand',
                        help='How to draw timesteps per batch: rand=old behavior, strat=cover [t_min..T-1] uniformly within each batch')
    parser.add_argument('--t_bins', type=int, default=0,
                        help='Number of strata for --t_cover strat (0=use batch size)')
    parser.add_argument('--t_multi', type=int, default=1,
                        help='Repeat each batch item K times with different t (increases effective batch size by K)')

    # robustness knobs
    parser.add_argument('--loss', choices=['mse','huber'], default='huber')
    parser.add_argument('--huber_beta', type=float, default=0.5, help='Huber transition point')
    parser.add_argument('--clip_grad',  type=float, default=1.0, help='0 to disable')
    parser.add_argument('--clamp_inputs', type=float, default=10.0, help='0 to disable clamping of normalized inputs')
    parser.add_argument('--skip_bad_batches', action='store_true', default=True)
    parser.add_argument('--skip_loss_threshold', type=float, default=50.0)

    # logging / probes
    parser.add_argument('--debug_first', action='store_true',
                        help='Print deep stats for the first batch (epoch 1 only by default).')
    parser.add_argument('--debug_first_every', type=int, default=0,
                        help='If >0, also run the deep first-batch probe every N epochs (rate-limited).')
    parser.add_argument('--log-jsonl', type=str, default=None,
                        help='Append per-batch/probe metrics to this JSONL file')
    parser.add_argument('--probe-cond', action='store_true',
                        help='Enable conditioning probe (cond on vs off) during training (JSONL only)')
    parser.add_argument('--probe-t', type=int, nargs='*',
                        default=[24, 50, 200, 500, 800],
                        help='Timesteps to probe at if --probe-cond is set')
    parser.add_argument('--probe-interval', type=int, default=50,
                        help='Run the probe every N batches')

    # AMP + EMA + schedule
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--ema', action='store_true', help='Enable EMA of weights ')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--cosine_decay', action='store_true', help='Enable cosine decay after warmup')
    parser.add_argument('--min_lr_scale', type=float, default=0.1, help='Final LR as fraction of base LR for cosine')

    # timestep loss weighting: weight --> (1 - alpha_bar[t])^p
    parser.add_argument('--loss_weight_power', type=float, default=0.0,
                        help='0 disables; >0 emphasizes noisier steps')

    parser.add_argument('--whiten', action='store_true')
    parser.add_argument('--whiten_mode', choices=['train', 'model', 'welch', 'auto'], default='auto',
                        help='How to whiten training data: use saved per-sample PSDs if available.')
    parser.add_argument('--sigma_mode', choices=['std', 'mad', 'fixed'], default='std')
    parser.add_argument('--sigma_fixed', type=float, default=1.0)

    parser.add_argument('--init-from', type=str, default=None,
                        help='Path to checkpoint (.pth) to init model weights from (EMA used if present).')

    # metadata conditioning behaviors
    parser.add_argument('--dropout_y_only', action='store_true', default=True,
                        help='If set (default), CFG dropout zeros only the y channel and keeps metadata on.')

    args = parser.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_diffusion(args)
