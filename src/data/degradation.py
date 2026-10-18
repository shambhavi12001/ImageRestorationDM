"""Bicubic degradation, MATLAB `imresize`-compatible.

Why a hand-written imresize instead of PIL / OpenCV / torch.nn.functional.interpolate:

Every published super-resolution PSNR number (EDSR, SR3, RRDB, SwinIR, ...) is
measured on low-resolution images produced by MATLAB's `imresize(im, 1/4,
'bicubic')`. PIL's BICUBIC and OpenCV's INTER_CUBIC use a different kernel
support and no antialiasing prefilter when downsampling, and they disagree with
MATLAB by roughly 0.5-1.0 dB on the same image. If we generate LR images one way
and compare against numbers computed another way, the comparison is meaningless.

So this file is the single source of truth for resizing in the whole project.
`configs/*.yaml` records `degradation.impl: matlab_bicubic` and it must never be
swapped for anything else mid-project.

Reference for the algorithm:
  MATLAB R2017b `imresize.m` (The MathWorks), bicubic kernel with a = -0.5 and
  antialiasing enabled, which is the "bicubic" of
  Keys, R. "Cubic convolution interpolation for digital image processing."
  IEEE Trans. ASSP 29(6), 1981.

Standard SR protocol this implements:
  Timofte et al., "NTIRE 2017 Challenge on Single Image Super-Resolution:
  Dataset and Study", CVPRW 2017  (the DIV2K bicubic track)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "imresize",
    "bicubic_down",
    "bicubic_up",
    "degrade",
    "modcrop",
]


# --------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------


def _cubic(x: np.ndarray) -> np.ndarray:
    """MATLAB bicubic kernel, a = -0.5. Support is [-2, 2]."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax

    inner = (1.5 * ax3 - 2.5 * ax2 + 1.0) * (ax <= 1.0)
    outer = (-0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0) * ((ax > 1.0) & (ax <= 2.0))
    return inner + outer


_KERNEL_WIDTH = 4.0  # support of _cubic


# --------------------------------------------------------------------------
# weight / index computation (one axis at a time)
# --------------------------------------------------------------------------


def _contributions(in_length: int, out_length: int, scale: float):
    """Return (weights, indices) for resampling one axis.

    weights : (out_length, P) float64
    indices : (out_length, P) int   -- already clipped/mirrored into range
    """
    if scale < 1.0:
        # Downsampling: widen the kernel so it acts as an antialiasing
        # prefilter. This is the step PIL and OpenCV skip.
        kernel = lambda x: scale * _cubic(scale * x)  # noqa: E731
        kernel_width = _KERNEL_WIDTH / scale
    else:
        kernel = _cubic
        kernel_width = _KERNEL_WIDTH

    # MATLAB works in 1-based continuous coordinates. Output pixel x maps to
    # input coordinate u. The 0.5 * (1 - 1/scale) term is the pixel-centre
    # alignment; dropping it is the classic half-pixel-shift bug.
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)

    left = np.floor(u - kernel_width / 2.0)
    P = int(np.ceil(kernel_width)) + 2

    indices = left[:, None] + np.arange(P, dtype=np.float64)[None, :]
    weights = kernel(u[:, None] - indices)

    norm = weights.sum(axis=1, keepdims=True)
    weights = weights / norm

    # Mirror out-of-range indices (MATLAB's symmetric padding), then convert
    # from 1-based to 0-based.
    aux = np.concatenate([np.arange(in_length), np.arange(in_length)[::-1]])
    indices = aux[np.mod(indices.astype(np.int64) - 1, aux.size)]

    # Drop columns whose weight is zero for every output pixel.
    keep = np.any(weights != 0, axis=0)
    return np.ascontiguousarray(weights[:, keep]), np.ascontiguousarray(indices[:, keep])


def _resize_along_axis(img: np.ndarray, weights: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    """Apply precomputed 1-D weights along `axis` of an (H, W, C) float array."""
    if axis == 0:
        gathered = img[indices, :, :]            # (out_H, P, W, C)
        return np.einsum("opwc,op->owc", gathered, weights)
    gathered = img[:, indices, :]                # (H, out_W, P, C)
    return np.einsum("howc,ow->hoc", gathered, weights)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def imresize(img: np.ndarray, scale: float | None = None, out_shape: tuple[int, int] | None = None) -> np.ndarray:
    """MATLAB-compatible bicubic resize.

    Parameters
    ----------
    img : (H, W) or (H, W, C) array. uint8 in [0, 255] or float in [0, 1].
    scale : multiplicative factor, e.g. 0.25 to downsample 4x.
    out_shape : (H, W) target, used instead of `scale` when given.

    Returns
    -------
    Array of the same dtype and channel layout as the input. uint8 input is
    rounded and clipped back to uint8, matching MATLAB's behaviour.
    """
    if (scale is None) == (out_shape is None):
        raise ValueError("pass exactly one of scale= or out_shape=")

    in_dtype = img.dtype
    squeeze = img.ndim == 2
    x = img[:, :, None] if squeeze else img
    x = x.astype(np.float64, copy=False)

    in_h, in_w = x.shape[:2]
    if out_shape is None:
        out_h = int(np.ceil(in_h * scale))
        out_w = int(np.ceil(in_w * scale))
        scale_h = scale_w = float(scale)
    else:
        out_h, out_w = int(out_shape[0]), int(out_shape[1])
        scale_h = out_h / in_h
        scale_w = out_w / in_w

    wh, ih = _contributions(in_h, out_h, scale_h)
    ww, iw = _contributions(in_w, out_w, scale_w)

    # MATLAB resizes the smaller-scale dimension first; order matters only at
    # the rounding level, but we follow it so numbers reproduce exactly.
    if scale_h <= scale_w:
        x = _resize_along_axis(x, wh, ih, axis=0)
        x = _resize_along_axis(x, ww, iw, axis=1)
    else:
        x = _resize_along_axis(x, ww, iw, axis=1)
        x = _resize_along_axis(x, wh, ih, axis=0)

    if squeeze:
        x = x[:, :, 0]

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        return np.clip(np.round(x), info.min, info.max).astype(in_dtype)
    return x.astype(in_dtype, copy=False)


def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    """Crop H and W down to multiples of `scale`.

    Required before degrading: a 1356-pixel-wide image is not divisible by 4,
    so down-then-up would not return to the original size and PSNR would be
    computed against a misaligned reference.
    """
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


def bicubic_down(hr: np.ndarray, scale: int = 4) -> np.ndarray:
    """HR -> LR. Antialiased bicubic decimation, the NTIRE 2017 bicubic track."""
    return imresize(hr, scale=1.0 / scale)


def bicubic_up(lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """LR -> HR-sized. This is both the bicubic baseline and the conditioning
    input the diffusion model is given (SR3 concatenates it channel-wise)."""
    return imresize(lr, scale=float(scale))


def degrade(hr: np.ndarray, scale: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The full forward operator used everywhere in the project.

    Returns
    -------
    hr_c : HR after modcrop -- the reference every metric is computed against
    lr   : the low-resolution observation y
    up   : bicubic upsample of lr, at HR size
    """
    hr_c = modcrop(hr, scale)
    lr = bicubic_down(hr_c, scale)
    up = bicubic_up(lr, scale)
    return hr_c, lr, up
