import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet1D, CustomDiffusion
from dataloader import make_dataloader

def prepare_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, 'latest_model')
    if os.path.exists(out_dir):
        for file in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, file))
    else:
        os.makedirs(out_dir, exist_ok=True)
    return out_dir

@torch.no_grad()
def _predict_x0_norm(model, diffusion, x_t, y_cond, t):
    zeros_sc = torch.zeros_like(x_t)
    net_in = torch.cat([x_t, y_cond, zeros_sc], dim=1)
    eps_hat = model(net_in, t)
    ab = diffusion.alpha_bar[t].view(-1, 1, 1)
    x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
    return x0_hat_norm.detach()

def _element_loss(eps_hat, eps, mask, loss_type: str, huber_beta: float):
    if loss_type == "huber":
        el = F.smooth_l1_loss(eps_hat, eps, reduction="none", beta=huber_beta)
    else:  # mse
        el = (eps_hat - eps) ** 2
    return el * mask

def train_diffusion(args):
    out_dir = prepare_output_dir(args.model_dir)

    loader: DataLoader = make_dataloader(
        h5_path=args.data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        whiten=args.whiten,
        sigma_mode=args.sigma_mode,
        sigma_fixed=args.sigma_fixed,
    )

    print(f"[train] Dataset size: {len(loader.dataset)}")
    print(f"[train] Batches per epoch: {len(loader)}")

    device = torch.device(args.device)
    in_ch = 3
    model = UNet1D(in_ch=in_ch, base_ch=args.base_ch, time_dim=args.time_dim, depth=args.depth).to(device)
    diffusion = CustomDiffusion(T=args.T, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss_weighted = 0.0
        sum_weight = 0
        batch_losses = []

        pbar = tqdm(
            loader, total=len(loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch", dynamic_ncols=True, leave=True, mininterval=0.1,
        )

        # guidance/self-cond (can be forced off for early epochs)
        p_uncond_eff   = 0.0 if epoch <= args.force_cond_epochs else args.p_uncond
        p_selfcond_eff = 0.0 if epoch <= args.force_cond_epochs else args.p_selfcond

        for i, (clean_raw, noisy_raw, sigma, mask) in enumerate(pbar):
            clean_raw = clean_raw.to(device).float()
            noisy_raw = noisy_raw.to(device).float()
            sigma     = sigma.to(device)
            mask      = mask.to(device).float()
            bsz, _, L = clean_raw.shape

            sigma_ = sigma.view(-1, 1, 1)
            clean_norm = clean_raw / sigma_
            cond_norm  = noisy_raw / sigma_

            # optional clamping to prevent rare huge values
            if args.clamp_inputs > 0:
                clean_norm = clean_norm.clamp(-args.clamp_inputs, args.clamp_inputs)
                cond_norm  = cond_norm.clamp (-args.clamp_inputs, args.clamp_inputs)

            # biased timestep sampling: t in [t_min, T-1]
            t_min = int(max(0, min(args.T - 1, int(args.t_min_frac * args.T))))
            t = torch.randint(t_min, args.T, (bsz,), device=device)


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
            use_sc = (torch.rand((), device=device) < p_selfcond_eff).item() if p_selfcond_eff > 0.0 else False
            if use_sc:
                with torch.no_grad():
                    x0_sc = _predict_x0_norm(model, diffusion, x_t, cond_used, t)
            else:
                x0_sc = torch.zeros_like(x_t)

            net_in = torch.cat([x_t, cond_used, x0_sc], dim=1)
            eps_hat = model(net_in, t)

            # weighted loss
            el = _element_loss(eps_hat, eps, mask, args.loss, args.huber_beta)
            denom = mask.sum(dim=[1, 2]).clamp_min(1.0)
            per_sample = el.sum(dim=[1, 2]) / denom
            loss = per_sample.mean()

            # guardrails: skip bad batches with explosive loss
            if not torch.isfinite(loss):
                pbar.write("[warn] non-finite loss — batch skipped")
                continue
            if args.skip_bad_batches and loss.item() > args.skip_loss_threshold:
                # batch stats for debug outliers
                with torch.no_grad():
                    smin = sigma.min().item(); smean = sigma.mean().item(); smax = sigma.max().item()
                    cn_max = cond_norm.abs().max().item()
                pbar.write(f"[warn] loss {loss.item():.3e} > {args.skip_loss_threshold} — "
                           f"skip (sigma[min/mean/max]={smin:.3e}/{smean:.3e}/{smax:.3e}, "
                           f"|cond_norm|_max={cn_max:.3e})")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            # stats
            loss_val = float(loss.item())
            batch_losses.append(loss_val)
            sum_loss_weighted += loss_val * bsz
            sum_weight += bsz
            running_avg = sum_loss_weighted / max(1, sum_weight)
            pbar.set_postfix(loss=f"{loss_val:.6f}", avg=f"{running_avg:.6f}")

            if i == 0 and args.debug_first:
                def _stats(x):
                    return x.min().item(), x.max().item(), x.mean().item(), x.std().item()
                ct_min, ct_max, ct_mean, ct_std = _stats(x_t)
                cn_min, cn_max, cn_mean, cn_std = _stats(clean_norm)
                yn_min, yn_max, yn_mean, yn_std = _stats(cond_norm)
                eh_min, eh_max, eh_mean, eh_std = _stats(eps_hat)
                print(f"[DEBUG] clean_norm: min={cn_min:.3e}, max={cn_max:.3e}, mean={cn_mean:.3e}, std={cn_std:.3e}")
                print(f"[DEBUG] cond_norm:  min={yn_min:.3e}, max={yn_max:.3e}, mean={yn_mean:.3e}, std={yn_std:.3e}")
                print(f"[DEBUG] x_t:        min={ct_min:.3e}, max={ct_max:.3e}, mean={ct_mean:.3e}, std={ct_std:.3e}")
                print(f"[DEBUG] eps_hat:    min={eh_min:.3e}, max={eh_max:.3e}, mean={eh_mean:.3e}, std={eh_std:.3e}")
                ab = diffusion.alpha_bar[t].view(-1, 1, 1)
                x0_hat_norm = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
                x0_hat = x0_hat_norm * sigma.view(-1,1,1)
                x0_min, x0_max, x0_mean, x0_std = _stats(x0_hat)
                print(f"[DEBUG] recon_unnorm: min={x0_min:.3e}, max={x0_max:.3e}, mean={x0_mean:.3e}, std={x0_std:.3e}")

        avg_loss_per_sample = sum_loss_weighted / max(1, sum_weight)
        avg_loss_per_batch  = float(np.mean(batch_losses)) if batch_losses else float("nan")
        med_loss_per_batch  = float(np.median(batch_losses)) if batch_losses else float("nan")
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Avg(per-sample)={avg_loss_per_sample:.6f} | "
            f"Mean(per-batch)={avg_loss_per_batch:.6f} | "
            f"Median(per-batch)={med_loss_per_batch:.6f}"
        )

    save_path = os.path.join(out_dir, 'model_diffusion.pth')
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': {
            **vars(args),
            'conditional': True,
            'in_ch': in_ch,
            'conditioning': 'concat_noisy+selfcond',
            'whiten': args.whiten,
            'sigma_mode': args.sigma_mode,
        },
        'epoch': args.epochs,
    }, save_path)
    print(f"Saved model to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train conditional diffusion denoiser on LIGO waveforms")
    parser.add_argument('--data',       type=str, required=True)
    parser.add_argument('--model_dir',  type=str, default='model')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--T',          type=int,   default=1000)
    parser.add_argument('--base_ch',    type=int,   default=64)
    parser.add_argument('--time_dim',   type=int,   default=128)
    parser.add_argument('--depth',      type=int,   default=3)
    parser.add_argument('--device',     type=str,   default=None)
    parser.add_argument('--num_workers',type=int,   default=4)

    # preprocessing to match deployment
    parser.add_argument('--whiten', action='store_true')
    parser.add_argument('--sigma_mode', choices=['std','mad','fixed'], default='std')
    parser.add_argument('--sigma_fixed', type=float, default=1.0)

    # guidance & self-conditioning
    parser.add_argument('--p_uncond',   type=float, default=0.2)
    parser.add_argument('--p_selfcond', type=float, default=0.5)

    # NEW: training schedule tweaks
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

    parser.add_argument('--debug_first', action='store_true')

    args = parser.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_diffusion(args)
