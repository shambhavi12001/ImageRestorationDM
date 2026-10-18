"""Hallucination measured as posterior uncertainty, plus timing.

The idea we are importing: a conditional generative restoration model samples
from an approximation of p(x | y). Where the measurement y constrains the
image, every sample agrees; where it does not, the model invents, and the
samples disagree. So the spread across samples drawn with different noise
seeds, with y held fixed, is an estimate of the posterior uncertainty, and
that uncertainty is what hallucination *is* rather than a proxy for it.

  Cohen, Kligvasser, Rivlin, Freedman, "Looks Too Good To Be True: An
  Information-Theoretic Analysis of Hallucinations in Generative Restoration
  Models", NeurIPS 2024.

They formalise hallucination through the posterior and validate on BSD100 at
4x bicubic, which is one of our two test sets under the same degradation, so
the estimator transfers directly.

The practical consequence for this project: `cross_seed_std` is the only metric
that needs no ground truth. It measures the model's own uncertainty, so it
separates "wrong because blurry" from "wrong because invented" -- two failures
that PSNR scores identically.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

__all__ = ["cross_seed_std", "cross_seed_map", "inference_time"]


def _stack(preds: Sequence[np.ndarray]) -> np.ndarray:
    if len(preds) < 2:
        raise ValueError(
            f"cross-seed variance needs >= 2 samples, got {len(preds)}. "
            "A single sample carries no uncertainty information."
        )
    shapes = {p.shape for p in preds}
    if len(shapes) != 1:
        raise ValueError(f"all samples must share a shape, got {shapes}")

    arr = np.stack([p.astype(np.float64) for p in preds], axis=0)
    # Normalise to [0, 1] so the number is comparable across dtypes and
    # readable as a fraction of dynamic range.
    if np.issubdtype(preds[0].dtype, np.integer):
        arr = arr / 255.0
    return arr


def cross_seed_map(preds: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel standard deviation across seeds.

    Returns an (H, W) map in units of fraction-of-dynamic-range. Useful as a
    figure: it localises *where* the model invented, which is the qualitative
    claim the paper makes quantitative.
    """
    arr = _stack(preds)
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)        # collapse colour before measuring spread
    return arr.std(axis=0, ddof=1)


def cross_seed_std(preds: Sequence[np.ndarray]) -> float:
    """Scalar hallucination estimate: mean of the per-pixel std map.

    ddof=1 because we are estimating the posterior std from a handful of
    samples, not describing the sample set. With 5 seeds the difference is
    about 12 percent, which matters when the effect size is small.
    """
    return float(cross_seed_map(preds).mean())


def inference_time(fn: Callable[[], object], n: int = 20, warmup: int = 3, sync: Callable[[], None] | None = None) -> dict:
    """Median wall-clock seconds per call.

    Median rather than mean because the first few GPU calls and any background
    allocation produce a long right tail that a mean would track. Pass
    `sync=torch.cuda.synchronize` on GPU, otherwise the timer stops before the
    kernels finish and the numbers are fiction.

    Timing is a reported result here, not bookkeeping: the strength knob also
    controls the number of sampler steps, so faithfulness and compute move
    together and the trade-off has a third axis.
    """
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append(time.perf_counter() - t0)

    s = np.asarray(samples, dtype=np.float64)
    return {
        "median_s": float(np.median(s)),
        "mean_s": float(s.mean()),
        "std_s": float(s.std(ddof=1)) if n > 1 else 0.0,
        "min_s": float(s.min()),
        "n": int(n),
    }
