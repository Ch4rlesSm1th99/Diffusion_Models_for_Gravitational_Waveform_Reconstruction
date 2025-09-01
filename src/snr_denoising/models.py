# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- Time + Schedule -------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * -(math.log(10000) / max(half - 1, 1))
        )
        x = t[:, None].float() * freqs[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=t.device)], dim=1)
        return emb

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float32)
    alphas_cum = torch.cos(((t / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(min=0.0, max=0.999)

class CustomDiffusion:
    def __init__(self, T: int = 1000, device: str = "cpu"):
        self.device = device
        self.T = T
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        a_bar = self.alpha_bar.sqrt()[t].view(-1, 1, 1)
        m_bar = (1 - self.alpha_bar).sqrt()[t].view(-1, 1, 1)
        eps = torch.randn_like(x0)
        x_t = a_bar * x0 + m_bar * eps
        return x_t, eps

# ---------------------------- UNet1D ------------------------------
class UNet1D(nn.Module):
    """
    1D UNet with:
      • FiLM-style time conditioning at every stage
      • NEW: multi-scale conditioning from y injected at every stage (additive bias)
    Inputs: x ∈ R^{B×C_in×L}, where C_in ∈ {1,2,3}
            convention: channel 0 = x_t, channel 1 = y (if present), channel 2 = self-cond (if present)
    Output: eps_hat ∈ R^{B×1×L}
    """
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        time_dim: int = 128,
        depth: int = 3,
        kernel: int = 3,
    ):
        super().__init__()
        self.in_ch = in_ch

        # time embedding -> [B, base_ch]
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, base_ch),
            nn.ReLU(),
        )

        # Encoder tower
        self.encoders = nn.ModuleList()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        in_c = in_ch
        for out_c in chs:
            self.encoders.append(self._conv_block(in_c, out_c, kernel))
            in_c = out_c

        # Bottleneck
        self.mid = self._conv_block(in_c, in_c, kernel)

        # Decoder tower
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = chs[-1]
        for skip_ch in reversed(chs):
            self.ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.decoders.append(self._conv_block(prev_ch + skip_ch, skip_ch, kernel))
            prev_ch = skip_ch

        # Final head
        self.final = nn.Conv1d(prev_ch + in_ch, 1, kernel, padding=kernel // 2)

        # ----- FiLM-style time projections -----
        self.tproj_enc = nn.ModuleList([nn.Linear(base_ch, c) for c in chs])
        self.tproj_mid = nn.Linear(base_ch, chs[-1])
        self.tproj_dec = nn.ModuleList([nn.Linear(base_ch, c) for c in reversed(chs)])

        # ----- NEW: multi-scale conditioning from y (additive bias) -----
        # if in_ch < 2 (no y provided), these layers will exist but fed zeros
        self.cond_enc = nn.ModuleList([nn.Conv1d(1, c, kernel_size=1) for c in chs])
        self.cond_mid = nn.Conv1d(1, chs[-1], kernel_size=1)
        self.cond_dec = nn.ModuleList([nn.Conv1d(1, c, kernel_size=1) for c in reversed(chs)])

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
        pad = k // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _add_time(h: torch.Tensor, t_emb: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        bias = proj(t_emb)[:, :, None]
        return h + bias

    def _y_or_zeros(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_ch >= 2:
            return x[:, 1:2, :]  # assume channel 1 is y
        else:
            return torch.zeros(x.size(0), 1, x.size(-1), device=x.device, dtype=x.dtype)

    def _cond_bias(self, y: torch.Tensor, L: int, layer: nn.Conv1d) -> torch.Tensor:
        # adaptively pool y to match current length then 1×1 conv to channels
        yL = F.adaptive_avg_pool1d(y, L)
        return layer(yL)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, L0 = x.shape
        t_emb = self.time_mlp(t)

        y = self._y_or_zeros(x)

        # Encode
        skips = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)  # [B, C_i, L]
            # inject y bias then time bias
            h = h + self._cond_bias(y, h.size(-1), self.cond_enc[i])
            h = self._add_time(h, t_emb, self.tproj_enc[i])
            skips.append(h)
            h = F.avg_pool1d(h, 2, 2)

        # Mid
        h = self.mid(h)
        h = h + self._cond_bias(y, h.size(-1), self.cond_mid)
        h = self._add_time(h, t_emb, self.tproj_mid)

        # Decode
        for i, (up, dec, skip) in enumerate(zip(self.ups, self.decoders, reversed(skips))):
            h = up(h)
            if h.size(-1) != skip.size(-1):
                diff = skip.size(-1) - h.size(-1)
                h = F.pad(h, (0, diff)) if diff > 0 else h[..., :skip.size(-1)]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
            # inject y bias then time bias
            h = h + self._cond_bias(y, h.size(-1), self.cond_dec[i])
            h = self._add_time(h, t_emb, self.tproj_dec[i])

        # Final skip with original input
        if h.size(-1) != x.size(-1):
            diff = x.size(-1) - h.size(-1)
            h = F.pad(h, (0, diff)) if diff > 0 else h[..., :x.size(-1)]
        out = self.final(torch.cat([h, x], dim=1))
        return out
