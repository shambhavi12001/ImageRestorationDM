"""The six metrics of the study.

Three axes, measured at every generative strength:

  perceptual quality   lpips_score
  distortion           psnr, ssim
  faithfulness         lr_consistency   (needs no ground truth)
  hallucination        cross_seed_std   (needs no ground truth)
  compute              inference_time

`lpips_score` is the only one that imports torch, and it does so lazily, so
everything else runs in a numpy-only environment.
"""

from src.metrics.distortion import psnr, ssim, lr_consistency, rgb_to_y
from src.metrics.hallucination import cross_seed_std, cross_seed_map, inference_time

__all__ = [
    "psnr",
    "ssim",
    "lr_consistency",
    "rgb_to_y",
    "cross_seed_std",
    "cross_seed_map",
    "inference_time",
    "lpips_score",
]


def lpips_score(pred, gt, net: str = "alex") -> float:
    """Lazy re-export so `import src.metrics` does not require torch."""
    from src.metrics.perceptual import lpips_score as _f

    return _f(pred, gt, net=net)
