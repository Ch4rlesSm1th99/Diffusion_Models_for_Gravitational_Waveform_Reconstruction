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
def _predict_x0_norm(model, diffusion, x_t, y_cond, t):
    t = t.long()
    zeros_sc = torch.zeros_like(x_t)
    net_in = torch.cat([x_t, y_cond, zeros_sc], dim=1)
    eps_hat = model(net_in, t)
    ab = diffusion.alpha_bar[t].view(-1, 1, 1)
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
    return x0_hat_norm.detach()

def _element_loss(eps_hat, eps, mask, loss_type: str, huber_beta: float):
    if loss_type == "huber":
        el = F.smooth_l1_loss(eps_hat, eps, reduction="none", beta=huber_beta)
    else:
        el = (eps_hat - eps) ** 2       # mse
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
        # cosine from 1.0 down to min_lr_scale
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_scale + 0.5 * (1 - min_lr_scale) * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_diffusion(args):
    set_seed(args.seed)
    out_dir = prepare_output_dir(args.model_dir)

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
    in_ch = 3

    # pass t_embed_max_time=T-1 to match normalized time embedding
    model = UNet1D(
        in_ch=in_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        depth=args.depth,
        t_embed_max_time=max(0, args.T - 1),
    ).to(device)

    diffusion = CustomDiffusion(T=args.T, device=device)

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

        for i, (clean_raw, noisy_raw, sigma, mask) in enumerate(pbar):
            clean_raw = clean_raw.to(device, non_blocking=True).float()
            noisy_raw = noisy_raw.to(device, non_blocking=True).float()
            sigma = sigma.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).float()
            bsz, _, L = clean_raw.shape

            sigma_ = sigma.view(-1, 1, 1)
            clean_norm = clean_raw / sigma_
            cond_norm = noisy_raw / sigma_

            # optional clamping
            if args.clamp_inputs > 0:
                clean_norm = clean_norm.clamp(-args.clamp_inputs, args.clamp_inputs)
                cond_norm = cond_norm.clamp(-args.clamp_inputs, args.clamp_inputs)

            # biased timestep sampling: t in [t_min, T-1]
            t_min = t_min_epoch
            t = torch.randint(t_min, args.T, (bsz,), device=device, dtype=torch.long)

            with torch.amp.autocast('cuda', enabled=args.amp):
                x_t, eps = diffusion.q_sample(clean_norm, t)

                if args.clamp_inputs > 0:
                    x_t = x_t.clamp(-args.clamp_inputs, args.clamp_inputs)

                # CFG dropout (can be forced off)
                if p_uncond_eff > 0.0:
                    drop = (torch.rand(bsz, 1, 1, device=device) < p_uncond_eff).float()
                    cond_used = cond_norm * (1.0 - drop)
                else:
                    cond_used = cond_norm

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

                # optional timestep weighting (emphasize certain noise region)
                if args.loss_weight_power != 0.0:
                    ab = diffusion.alpha_bar[t].view(-1, 1, 1)
                    # weight --> (1 - alpha_bar)^p ; p>0 emphasizes noisier steps
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
                    cn_max = cond_norm.abs().max().item()
                pbar.write(f"[warn] loss {loss.item():.3e} > {args.skip_loss_threshold} "
                           f"skip (sigma[min/mean/max]={smin:.3e}/{smean:.3e}/{smax:.3e}, "
                           f"(cond_norm)_max={cn_max:.3e})")
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
            sum_loss_weighted += loss_val * bsz
            sum_weight += bsz
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
                    y0 = (cond_norm[0:1]).detach()
                    zeros_y = torch.zeros_like(y0)
                    zeros_sc = torch.zeros_like(c0)
                    for t_pick in args.probe_t:
                        t_probe = torch.tensor([max(0, min(args.T - 1, int(t_pick)))],
                                               device=device, dtype=torch.long)
                        x_t_p, eps_p = diffusion.q_sample(c0, t_probe)
                        net_on = torch.cat([x_t_p, y0, zeros_sc], dim=1)
                        net_off = torch.cat([x_t_p, zeros_y, zeros_sc], dim=1)
                        eps_on = model(net_on, t_probe)
                        eps_off = model(net_off, t_probe)
                        mse_on = float(F.mse_loss(eps_on, eps_p).item())
                        mse_off = float(F.mse_loss(eps_off, eps_p).item())
                        corr_on = float(_corr_torch(eps_on, eps_p).item())
                        corr_off = float(_corr_torch(eps_off, eps_p).item())
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

            # ---- rate limited batch probe: once on epoch 1, optionally every N epochs
            if (i == 0) and (not printed_debug_this_epoch) and _do_debug_this_epoch(epoch):
                printed_debug_this_epoch = True

                def _stats(x: torch.Tensor):
                    return x.min().item(), x.max().item(), x.mean().item(), x.std().item()

                cn_min, cn_max, cn_mean, cn_std = _stats(clean_norm)
                yn_min, yn_max, yn_mean, yn_std = _stats(cond_norm)
                ct_min, ct_max, ct_mean, ct_std = _stats(x_t)
                eh_min, eh_max, eh_mean, eh_std = _stats(eps_hat)

                print(f"[DEBUG] sigma[min/mean/max]={sigma.min().item():.3e}/{sigma.mean().item():.3e}/{sigma.max().item():.3e}")
                print(f"[DEBUG] clean_norm: min={cn_min:.3e}, max={cn_max:.3e}, mean={cn_mean:.3e}, std={cn_std:.3e}")
                print(f"[DEBUG] cond_norm:  min={yn_min:.3e}, max={yn_max:.3e}, mean={yn_mean:.3e}, std={yn_std:.3e}")
                print(f"[DEBUG] x_t:        min={ct_min:.3e}, max={ct_max:.3e}, mean={ct_mean:.3e}, std={ct_std:.3e}")
                print(f"[DEBUG] eps_hat:    min={eh_min:.3e}, max={eh_max:.3e}, mean={eh_mean:.3e}, std={eh_std:.3e}")

                ab = diffusion.alpha_bar[t].view(-1, 1, 1)
                x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
                x0_hat = x0_hat_norm * sigma.view(-1, 1, 1)

                x0_min, x0_max, x0_mean, x0_std = _stats(x0_hat)
                print(f"[DEBUG] recon_unnorm: min={x0_min:.3e}, max={x0_max:.3e}, mean={x0_mean:.3e}, std={x0_std:.3e}")

                def _corr_masked(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
                    # a,b: [1,1,L], m: [1,1,L] with 1s on valid region
                    a1 = a[m > 0.5].reshape(-1)
                    b1 = b[m > 0.5].reshape(-1)
                    if a1.numel() < 2 or b1.numel() < 2:
                        return float('nan')
                    a1 = a1 - a1.mean()
                    b1 = b1 - b1.mean()
                    den = (a1.pow(2).sum().sqrt() * b1.pow(2).sum().sqrt() + 1e-12)
                    return float((a1 * b1).sum() / den)

                m0 = (mask[0:1]).float()  # [1,1,L] valid=1, pad=0

                # correlations in normalized (whitened, sigma-normalized) domain
                corr_x0n = _corr_masked(x0_hat_norm[0:1], clean_norm[0:1], m0)
                # correlations in whitened (but unnormalized-by-sigma) domain
                corr_x0w = _corr_masked(x0_hat[0:1], clean_raw[0:1], m0)

                # masked MSE (both domains)
                mse_x0n = (((x0_hat_norm - clean_norm) ** 2) * m0).sum() / m0.sum().clamp_min(1.0)
                mse_x0w = (((x0_hat - clean_raw) ** 2) * m0).sum() / m0.sum().clamp_min(1.0)

                valid_frac = float(m0.mean().item())
                print(f"[DEBUG] valid_frac={valid_frac:.3f} , "
                      f"corr_masked(x0_norm, clean_norm)={corr_x0n:.3f} "
                      f"corr_masked(x0, clean)={corr_x0w:.3f} , "
                      f"MSE_masked_norm={mse_x0n.item():.3e} , MSE_masked_white={mse_x0w.item():.3e}")

                # quick correlations on sample 0
                try:
                    corr_eps = _corr_torch(eps_hat[0], eps[0]).item()
                    print(f"[DEBUG] corr(eps_hat, eps)={corr_eps:.3f}")
                except Exception:
                    pass

                # tiny npy dump (first epoch only)
                if epoch == 1:
                    try:
                        dbg_dir = os.path.join(out_dir, "debug_batch0")
                        os.makedirs(dbg_dir, exist_ok=True)
                        np.save(os.path.join(dbg_dir, "clean_raw.npy"), clean_raw[0].detach().cpu().numpy().ravel())
                        np.save(os.path.join(dbg_dir, "cond_norm.npy"), cond_norm[0].detach().cpu().numpy().ravel())
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
            'conditioning': 'concat_noisy+selfcond',
            'whiten': args.whiten,
            'whiten_mode': args.whiten_mode,
            'sigma_mode': args.sigma_mode,
        },
        'epoch': args.epochs,
    }
    if ema_model is not None:
        payload['model_ema_state'] = ema_model.state_dict()
    torch.save(payload, save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train conditional diffusion denoiser on LIGO waveforms")
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

    # timestep loss weighting: weight --> (1 - alpha_bar[t])^p i.e --> scheduler
    parser.add_argument('--loss_weight_power', type=float, default=0.0,
                        help='0 disables; >0 emphasizes noisier steps')

    parser.add_argument('--whiten', action='store_true')
    parser.add_argument('--whiten_mode', choices=['train', 'model', 'welch', 'auto'], default='auto',
                        help='How to whiten training data: use saved per-sample PSDs if available.')
    parser.add_argument('--sigma_mode', choices=['std', 'mad', 'fixed'], default='std')
    parser.add_argument('--sigma_fixed', type=float, default=1.0)

    args = parser.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_diffusion(args)
