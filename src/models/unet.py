"""The denoiser: a conditional UNet that predicts epsilon.

Architecture follows the DDPM UNet
  Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020
with the SR3 conditioning scheme
  Saharia et al., "Image Super-Resolution via Iterative Refinement",
  IEEE TPAMI 45(4), 2023:
the bicubic upsample of the low-resolution observation is concatenated to the
noisy target along the channel axis, so the network sees 6 input channels and
emits 3.

That concatenation is the whole of the conditioning. There is no separate
encoder and no cross-attention; the conditioning signal is present at full
resolution at every layer because it entered at the stem. It is also why the
bicubic operator has to be the same one used to build the training data: the
model learns to trust the structure that operator produces.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet", "SinusoidalPosEmb"]


class SinusoidalPosEmb(nn.Module):
    """Transformer-style timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _norm(c: int) -> nn.GroupNorm:
    # 32 groups is the DDPM default; fall back for narrow layers so the group
    # count always divides the channel count.
    return nn.GroupNorm(min(32, c) if c % min(32, c) == 0 else 1, c)


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _norm(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_c)
        self.norm2 = _norm(out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        # Zero-init the last conv so each block starts as the identity. This
        # is what keeps deep diffusion UNets stable in the first few thousand
        # steps; without it the early loss curve is erratic.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = _norm(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Conditional epsilon-prediction UNet.

    Parameters
    ----------
    image_size : HR patch size during training. Only used to decide at which
        spatial resolutions attention is applied.
    in_ch : 6 = 3 noisy target + 3 conditioning channels.
    base, mults, n_res : width and depth. The defaults give ~30M parameters,
        which trains to a usable model on one GPU within the project's budget.
    attn_res : spatial sizes at which to apply self-attention.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_ch: int = 6,
        out_ch: int = 3,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 4, 8),
        n_res: int = 2,
        attn_res: tuple[int, ...] = (16,),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_ch = in_ch

        emb_dim = base * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        # ---- encoder ----
        self.down = nn.ModuleList()
        skip_channels = [base]
        cur, res = base, image_size
        for level, mult in enumerate(mults):
            for _ in range(n_res):
                block = nn.ModuleList([ResBlock(cur, base * mult, emb_dim, dropout)])
                cur = base * mult
                if res in attn_res:
                    block.append(AttnBlock(cur))
                self.down.append(block)
                skip_channels.append(cur)
            if level != len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(cur)]))
                skip_channels.append(cur)
                res //= 2

        # ---- middle ----
        self.mid1 = ResBlock(cur, cur, emb_dim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, emb_dim, dropout)

        # ---- decoder ----
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for i in range(n_res + 1):
                block = nn.ModuleList(
                    [ResBlock(cur + skip_channels.pop(), base * mult, emb_dim, dropout)]
                )
                cur = base * mult
                if res in attn_res:
                    block.append(AttnBlock(cur))
                if level != 0 and i == n_res:
                    block.append(Upsample(cur))
                    res *= 2
                self.up.append(block)

        assert not skip_channels, f"skip bookkeeping off by {len(skip_channels)}"

        self.out = nn.Sequential(_norm(cur), nn.SiLU(), nn.Conv2d(cur, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[2].weight)
        nn.init.zeros_(self.out[2].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) noisy target. t: (B,) timesteps. cond: (B,3,H,W) upsample."""
        if cond.shape != x.shape:
            raise ValueError(
                f"cond {tuple(cond.shape)} must match x {tuple(x.shape)}; the "
                "conditioning image is the bicubic upsample at HR size, not the LR image"
            )
        emb = self.time_embed(t)
        h = self.conv_in(torch.cat([x, cond], dim=1))

        hs = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for block in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)

        return self.out(h)
