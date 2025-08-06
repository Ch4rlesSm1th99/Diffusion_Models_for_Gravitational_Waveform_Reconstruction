import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    '''
    time embeddings using sin + cos encoding --> encodes the timestep 't' into a higher dim feature vector for training
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2            # gen sin embeddings
        emb = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * -(math.log(10000) / (half - 1))           # shape = [B,half]
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:       # if odd dim --> pad with a zero column
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=t.device)], dim=1)
        return emb


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    '''
    introduce thsi to replace typical linear beta scheduler. Idea is that initial and final additions
    of noise are much smoother rather than having nosie additions in the forward process that are equally
    spaced across 'T' diffusion timesteps. --> from Nichol & Dhariwal (2021)
    '''
    steps = T + 1
    t = torch.linspace(0, T, steps)
    alphas_cum = torch.cos(((t / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(min=0.0, max=0.999)

class CustomDiffusion:
    '''
    basic forward diffsuion class - only includes the forward process, no decoding
    uses specified schduler, at the moment the beta scheduler.
    '''
    def __init__(self, T: int = 1000, device: str = 'cpu'):
        self.device = device
        self.T = T
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        '''
        forwad step --> return noise sample 'xt' and the added noise contributions
        '''
        a_bar = self.alpha_bar.sqrt()[t].view(-1,1,1)
        m_bar = (1 - self.alpha_bar).sqrt()[t].view(-1,1,1)
        noise = torch.randn_like(x0)
        xt = a_bar * x0 + m_bar * noise
        return xt, noise


class UNet1D(nn.Module):
    '''
    1D_UNET for time-series data, trrained with the addition of the tiem embeddings.
    Overview:
        - Time embedding -> small MLP net
        -Encoder: downsamples/ extracts feature representations of the noisy data
        -Bottleneck: latent space - note latent space can be skipped via skip connections
            -->helps preserve high level features
        - decoder: upsampling logit extraction
            -->(ntoe skip connections link encoder layers to decoder layers directly)
                ie they don't have to pass via the latent space and are jsut directly concatenated with the features
                of the corresponding decoder level.
    '''
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        time_dim: int = 128,
        depth: int = 3,
        kernel: int = 3,
    ):
        super().__init__()
        # time-embedding MLP
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, base_ch),
            nn.ReLU(),
        )

        # encoder
        self.encoders = nn.ModuleList()
        chs = [base_ch * (2**i) for i in range(depth)]  # shape [64,128,256]
        in_c = in_ch
        for out_c in chs:
            self.encoders.append(self._conv_block(in_c, out_c, kernel))
            in_c = out_c

        # bottleneck
        self.mid = self._conv_block(in_c, in_c, kernel)

        # decoder --> upsample, mirroring encoder with skip connect
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        prev_ch = chs[-1]  # start at deepest channels
        for skip_ch in reversed(chs):
            # upsample
            self.ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            # conv after concatenation of prev feature map + skip
            self.dec_convs.append(self._conv_block(prev_ch + skip_ch, skip_ch, kernel))
            prev_ch = skip_ch

        # final conv: combine last feature map with original input
        self.final = nn.Conv1d(prev_ch + in_ch, in_ch, kernel, padding=kernel//2)

    def _conv_block(self, in_ch: int, out_ch: int, k: int) -> nn.Sequential:
        pad = k // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t embedding
        t_emb = self.time_mlp(t)[:, :, None]

        # encode
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = F.avg_pool1d(h, 2, 2)

        # bottleneck
        h = self.mid(h)

        # decode
        for up, dec_conv, skip in zip(self.ups, self.dec_convs, reversed(skips)):
            h = up(h)
            # fix length mismatch if any
            if h.size(-1) != skip.size(-1):
                diff = skip.size(-1) - h.size(-1)
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    h = h[..., :skip.size(-1)]
            # concat and conv
            h = torch.cat([h, skip], dim=1)
            h = dec_conv(h)

        # final skip with original input
        if h.size(-1) != x.size(-1):
            diff = x.size(-1) - h.size(-1)
            if diff > 0:
                h = F.pad(h, (0, diff))
            else:
                h = h[..., :x.size(-1)]
        out = self.final(torch.cat([h, x], dim=1))
        return out
