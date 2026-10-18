"""Perceptual metric: LPIPS.

  Zhang, Isola, Efros, Shechtman, Wang, "The Unreasonable Effectiveness of
  Deep Features as a Perceptual Metric", CVPR 2018.

We use the AlexNet backbone with the learned linear calibration (`net='alex'`,
the `lpips` reference package), which is the configuration SR3 reports:
  Saharia, Ho, Chan, Salimans, Fleet, Norouzi, "Image Super-Resolution via
  Iterative Refinement", IEEE TPAMI 45(4), 2023.

torch is imported lazily so that the distortion metrics and the T3 validation
gate run in a plain numpy environment. Only this module needs a GPU box.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lpips_score", "LPIPS"]

_MODEL = None
_DEVICE = None


class LPIPS:
    """Thin wrapper that loads the network once and keeps it resident.

    Batch your calls through a single instance; constructing one per image
    reloads AlexNet every time and dominates the sweep runtime.
    """

    def __init__(self, net: str = "alex", device: str | None = None):
        import torch
        import lpips as lpips_pkg

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net_name = net
        self.model = lpips_pkg.LPIPS(net=net).to(self.device).eval()

    def _to_tensor(self, img: np.ndarray):
        """(H, W, C) uint8 or float[0,1]  ->  (1, 3, H, W) float in [-1, 1]."""
        x = img.astype(np.float32)
        if np.issubdtype(img.dtype, np.integer):
            x = x / 255.0
        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        x = x[:, :, :3]
        x = np.clip(x, 0.0, 1.0) * 2.0 - 1.0          # LPIPS expects [-1, 1]
        t = self.torch.from_numpy(x.transpose(2, 0, 1))[None]
        return t.to(self.device)

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if pred.shape[:2] != gt.shape[:2]:
            raise ValueError(f"shape mismatch: {pred.shape[:2]} vs {gt.shape[:2]}")
        with self.torch.no_grad():
            d = self.model(self._to_tensor(pred), self._to_tensor(gt))
        return float(d.reshape(-1)[0].item())


def lpips_score(pred: np.ndarray, gt: np.ndarray, net: str = "alex") -> float:
    """Convenience single-shot call. Reuses one global model instance.

    Note: LPIPS is computed on the full RGB image with no border crop, which is
    the convention in the LPIPS paper and in SR3. This deliberately differs
    from the Y-channel/border-crop protocol used for PSNR and SSIM; mixing the
    two would make neither comparable to published numbers.
    """
    global _MODEL
    if _MODEL is None or _MODEL.net_name != net:
        _MODEL = LPIPS(net=net)
    return _MODEL(pred, gt)
