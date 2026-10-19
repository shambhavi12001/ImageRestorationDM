"""EDSR, the distortion-oriented baseline.

  Lim, Son, Kim, Nah, Lee, "Enhanced Deep Residual Networks for Single Image
  Super-Resolution", CVPRW 2017.

This is a faithful reimplementation of the official architecture, written so
the official pretrained state dict loads into it key-for-key. We do NOT train
EDSR: it is downloaded, verified against its published Set5 number, and then
used unchanged as the anchor at the distortion end of the trade-off.

Two details that differ from the diffusion model and are easy to get wrong:

  * EDSR consumes the LOW-RESOLUTION image directly and upsamples internally
    with pixel shuffle. It does not take our bicubic upsample. Feeding it
    `sample.up` instead of `sample.lr` produces a 16x image and nonsense
    numbers.
  * It works in [0, 255], not [0, 1] or [-1, 1], and begins by subtracting the
    DIV2K RGB mean. The mean-shift layers carry fixed, non-learned weights.

Official weights (x4, the configuration whose Set5 PSNR is 32.46 dB):
    https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["EDSR", "load_edsr", "EDSRRunner", "PUBLISHED_PSNR"]


# Published EDSR x4 PSNR (Y channel, 4-px border crop), CVPRW 2017 Table 2.
PUBLISHED_PSNR = {
    "set5": 32.46,
    "set14": 28.80,
    "bsd100": 27.71,
    "urban100": 26.64,
}

# DIV2K training-set RGB mean, as used by the official implementation.
DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)


def _conv(in_c: int, out_c: int, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=True)


class MeanShift(nn.Conv2d):
    """Fixed 1x1 conv that adds or removes the dataset RGB mean."""

    def __init__(self, rgb_range: int = 255, sign: int = -1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.ones(3)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(DIV2K_RGB_MEAN) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """conv-ReLU-conv with a scaled residual.

    `res_scale` is 0.1 in the large model. It is not cosmetic: without it the
    256-feature network diverges early in training (Lim et al., Sec. 3.1).
    """

    def __init__(self, n_feats: int, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            _conv(n_feats, n_feats), nn.ReLU(inplace=True), _conv(n_feats, n_feats)
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x).mul(self.res_scale)


class Upsampler(nn.Sequential):
    """Pixel-shuffle upsampling. x4 is two x2 stages, which is what the
    official checkpoint's tail layout assumes."""

    def __init__(self, scale: int, n_feats: int):
        layers: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:                      # power of two
            for _ in range(int(math.log2(scale))):
                layers += [_conv(n_feats, 4 * n_feats), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [_conv(n_feats, 9 * n_feats), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"unsupported scale {scale}")
        super().__init__(*layers)


class EDSR(nn.Module):
    """Defaults are the large x4 model whose Set5 PSNR is 32.46 dB."""

    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 32,
        n_feats: int = 256,
        res_scale: float = 0.1,
        rgb_range: int = 255,
    ):
        super().__init__()
        self.scale = scale
        self.rgb_range = rgb_range

        self.sub_mean = MeanShift(rgb_range, sign=-1)
        self.add_mean = MeanShift(rgb_range, sign=+1)

        self.head = nn.Sequential(_conv(3, n_feats))

        body: list[nn.Module] = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body.append(_conv(n_feats, n_feats))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(Upsampler(scale, n_feats), _conv(n_feats, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, h, w) float in [0, 255]  ->  (B, 3, 4h, 4w) in [0, 255]."""
        x = self.sub_mean(x)
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        return self.add_mean(x)


# --------------------------------------------------------------------------


def load_edsr(weights: str | Path, scale: int = 4, device: str | None = None,
              n_resblocks: int = 32, n_feats: int = 256, res_scale: float = 0.1) -> EDSR:
    """Build the model and load the official checkpoint strictly.

    `strict=True` is deliberate. A silently partial load would leave random
    weights in some layers and produce a plausible-looking image a couple of dB
    below the published number, which is exactly the failure this baseline
    exists to rule out.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSR(scale=scale, n_resblocks=n_resblocks, n_feats=n_feats, res_scale=res_scale)

    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class EDSRRunner:
    """Numpy in, numpy out, with tiled inference for large images.

    Urban100 at x4 produces outputs over 1000 px on a side with 256 feature
    maps, which can exhaust memory in one pass. `chop` splits the input into
    four overlapping quadrants and reassembles, as the official repository
    does; the overlap means seams are never visible in the output.
    """

    def __init__(self, model: EDSR, device: str | None = None, chop_threshold: int = 160_000):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.scale = model.scale
        self.chop_threshold = chop_threshold

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h * w <= self.chop_threshold:
            return self.model(x)

        s, shave = self.scale, 10
        half_h, half_w = h // 2, w // 2
        pad_h, pad_w = half_h + shave, half_w + shave

        quads = [
            x[..., 0:pad_h, 0:pad_w],
            x[..., 0:pad_h, w - pad_w : w],
            x[..., h - pad_h : h, 0:pad_w],
            x[..., h - pad_h : h, w - pad_w : w],
        ]
        outs = [self._forward(q) for q in quads]

        H, W = h * s, w * s
        hh, ww = half_h * s, half_w * s
        ph, pw = pad_h * s, pad_w * s

        out = x.new_zeros((x.shape[0], 3, H, W))
        out[..., 0:hh, 0:ww] = outs[0][..., 0:hh, 0:ww]
        out[..., 0:hh, ww:W] = outs[1][..., 0:hh, pw - (W - ww) : pw]
        out[..., hh:H, 0:ww] = outs[2][..., ph - (H - hh) : ph, 0:ww]
        out[..., hh:H, ww:W] = outs[3][..., ph - (H - hh) : ph, pw - (W - ww) : pw]
        return out

    @torch.no_grad()
    def __call__(self, lr: np.ndarray) -> np.ndarray:
        """lr: (h, w, 3) uint8  ->  (4h, 4w, 3) uint8."""
        x = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1))[None]
        x = x.to(self.device)
        y = self._forward(x)
        y = y.clamp(0, 255).round()[0].cpu().numpy().transpose(1, 2, 0)
        return y.astype(np.uint8)
