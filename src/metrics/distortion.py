"""Distortion metrics: PSNR, SSIM, and LR consistency.

Protocol choices here are not free parameters. They are fixed to match the
numbers published in the SR literature so that our baselines can be checked
against those numbers (the T4 validation gate):

  * PSNR and SSIM on the Y (luma) channel of YCbCr, not RGB.
  * `scale` pixels cropped from every border before measuring.

Both conventions come from the benchmark protocol used by
  Lim et al., "Enhanced Deep Residual Networks for Single Image
  Super-Resolution", CVPRW 2017 (EDSR),
following
  Timofte et al., NTIRE 2017 Challenge on Single Image Super-Resolution,
  CVPRW 2017,
and every paper we compare against reports numbers this way. Measuring on RGB
instead changes PSNR by roughly 1 dB, which is larger than the effects we are
trying to detect.

SSIM follows
  Wang et al., "Image Quality Assessment: From Error Visibility to Structural
  Similarity", IEEE TIP 13(4), 2004.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rgb_to_y", "psnr", "ssim", "lr_consistency"]


# --------------------------------------------------------------------------
# colour conversion
# --------------------------------------------------------------------------

# ITU-R BT.601 coefficients in the "studio swing" form MATLAB's rgb2ycbcr
# uses. They are applied to RGB in [0, 1] and produce Y in [16, 235]:
#   Y = 16 + 65.481 R + 128.553 G + 24.966 B
# The coefficients sum to 219, which is the 235 - 16 studio range. Do not also
# divide them by 255 -- that collapses Y into a ~1-unit band around 16 and
# inflates every PSNR by about 17 dB.
_Y_COEFF = np.array([65.481, 128.553, 24.966], dtype=np.float64)
_Y_OFFSET = 16.0


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """RGB -> Y (luma), returned on the [16, 235] studio scale as float64.

    Accepts uint8 [0, 255] or float [0, 1]. Single-channel input is treated as
    already being luma and is only rescaled to [0, 255], which is what the SR
    benchmark scripts do; note this is the full 0-255 range rather than the
    studio range, so never mix grayscale and RGB images in one average.
    """
    x = img.astype(np.float64)
    if np.issubdtype(img.dtype, np.integer):
        x = x / 255.0
    # x is now in [0, 1]
    if x.ndim == 2:
        return x * 255.0
    if x.shape[2] == 1:
        return x[:, :, 0] * 255.0
    return x[:, :, :3] @ _Y_COEFF + _Y_OFFSET


def _prep(pred: np.ndarray, gt: np.ndarray, border: int):
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"shape mismatch: pred {pred.shape[:2]} vs gt {gt.shape[:2]}")
    y_pred = rgb_to_y(pred)
    y_gt = rgb_to_y(gt)
    if border > 0:
        y_pred = y_pred[border:-border, border:-border]
        y_gt = y_gt[border:-border, border:-border]
    return y_pred, y_gt


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def psnr(pred: np.ndarray, gt: np.ndarray, border: int = 4) -> float:
    """PSNR in dB on the Y channel, with `border` pixels cropped.

    `border` defaults to 4 = the upscaling factor, per the EDSR benchmark
    protocol. Pass border=0 only for diagnostics, never for reported numbers.
    """
    y_pred, y_gt = _prep(pred, gt, border)
    mse = float(np.mean((y_pred - y_gt) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _ssim_single(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on one 2-D channel in [0, 255].

    Gaussian window sigma=1.5, 11x11, C1/C2 from Wang et al. 2004. The window
    is applied with 'valid' support (no padding), which is what the reference
    MATLAB implementation does.
    """
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    # separable 11x11 Gaussian, sigma = 1.5
    r = 5
    coords = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g /= g.sum()

    def blur(x: np.ndarray) -> np.ndarray:
        # 'valid' separable convolution via sliding windows
        w = np.lib.stride_tricks.sliding_window_view(x, (2 * r + 1, 2 * r + 1))
        return np.einsum("ijkl,k,l->ij", w, g, g)

    mu_a = blur(a)
    mu_b = blur(b)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sig_a2 = blur(a * a) - mu_a2
    sig_b2 = blur(b * b) - mu_b2
    sig_ab = blur(a * b) - mu_ab

    num = (2.0 * mu_ab + C1) * (2.0 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


def ssim(pred: np.ndarray, gt: np.ndarray, border: int = 4) -> float:
    """SSIM on the Y channel, same preprocessing as `psnr`."""
    y_pred, y_gt = _prep(pred, gt, border)
    return _ssim_single(y_pred, y_gt)


def lr_consistency(pred: np.ndarray, lr: np.ndarray, scale: int = 4) -> float:
    """Measurement fidelity: PSNR between H(pred) and the observation y.

    H is the same antialiased bicubic decimation used to create the data, so
    this is a direct estimate of || H x_hat - y ||, reported in dB so it reads
    on the same scale as PSNR. It is the one faithfulness number that needs no
    ground-truth HR image, which means it is also computable on real
    low-resolution inputs where no reference exists.

    This is the data-consistency term that
      Wang, Yu & Zhang, "Zero-Shot Image Restoration Using Denoising Diffusion
      Null-Space Model" (DDNM), ICLR 2023
    drives to zero by construction; measuring it tells us how far an
    unconstrained sampler drifts from the measurement.
    """
    from src.data.degradation import bicubic_down

    pred_down = bicubic_down(pred, scale)
    if pred_down.shape[:2] != lr.shape[:2]:
        raise ValueError(
            f"downsampled pred {pred_down.shape[:2]} != lr {lr.shape[:2]}; "
            "was the HR modcropped before degrading?"
        )
    # No border crop: the LR image is small and the border is a real part of
    # the measurement.
    y_a = rgb_to_y(pred_down)
    y_b = rgb_to_y(lr)
    mse = float(np.mean((y_a - y_b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))
