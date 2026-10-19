"""The degradation operator H and its pseudo-inverse, as torch tensors.

`src/data/degradation.py` is the single source of truth for resizing, but it
is numpy and runs on the CPU. The sampler needs the same operator inside the
denoising loop, on the GPU, once per step, for DDNM back-projection
(Wang, Yu & Zhang, ICLR 2023).

Resampling is a *linear* operator, so rather than reimplement bicubic in torch
and hope the two agree, this module takes the weights and indices that the
numpy implementation already computes and assembles them into explicit
matrices. H is then one matrix multiply per axis. The result is identical to
the numpy path to floating-point precision -- which `scripts/selftest.py`
checks -- and costs microseconds on GPU.

    down:  y = Mh_down @ x @ Mw_down^T
    up:    x = Mh_up   @ y @ Mw_up^T      (bicubic upsample, the H-dagger we use)
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.degradation import _contributions

__all__ = ["resize_matrix", "BicubicOperator"]


def resize_matrix(in_length: int, out_length: int, scale: float,
                  dtype: torch.dtype = torch.float32,
                  device: str | torch.device = "cpu") -> torch.Tensor:
    """Dense (out_length, in_length) resampling matrix for one axis."""
    weights, indices = _contributions(in_length, out_length, scale)
    M = np.zeros((out_length, in_length), dtype=np.float64)
    rows = np.repeat(np.arange(out_length), indices.shape[1])
    np.add.at(M, (rows, indices.ravel()), weights.ravel())
    return torch.tensor(M, dtype=dtype, device=device)


class BicubicOperator:
    """H and H-dagger for one fixed image size.

    Built once per image size and reused across all sampler steps and seeds.
    Sizes are fixed within a sweep over one dataset only when the images are
    the same shape, so the cache is keyed by shape.
    """

    def __init__(self, hr_h: int, hr_w: int, scale: int = 4,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32):
        if hr_h % scale or hr_w % scale:
            raise ValueError(f"HR size {hr_h}x{hr_w} not divisible by scale {scale}")
        self.scale = scale
        self.hr_h, self.hr_w = hr_h, hr_w
        self.lr_h, self.lr_w = hr_h // scale, hr_w // scale
        self.device, self.dtype = device, dtype

        inv = 1.0 / scale
        self.Dh = resize_matrix(hr_h, self.lr_h, inv, dtype, device)
        self.Dw = resize_matrix(hr_w, self.lr_w, inv, dtype, device)
        self.Uh = resize_matrix(self.lr_h, hr_h, float(scale), dtype, device)
        self.Uw = resize_matrix(self.lr_w, hr_w, float(scale), dtype, device)

        # Moore-Penrose pseudo-inverse of the DOWN matrices, per axis.
        #
        # Bicubic upsampling is NOT the pseudo-inverse of bicubic
        # downsampling: H H_up != I, so back-projecting with H_up leaves
        # roughly 40 percent of the measurement residual behind and the
        # correction is not a projection at all. Measured here: one
        # application cut ||Hx - y|| by only ~57 percent.
        #
        # The operator is separable, H x = Dh x Dw^T, so the pseudo-inverse
        # factorises too, and (Dh Dh+) (x) (Dw Dw+)^T = x gives H H+ = I
        # exactly. The matrices are (lr, hr) with lr small, so this is cheap
        # and is computed once per image size.
        self.Dh_pinv = torch.linalg.pinv(self.Dh.double()).to(dtype)
        self.Dw_pinv = torch.linalg.pinv(self.Dw.double()).to(dtype)

    def down(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,C,H/s,W/s). The forward measurement operator H."""
        x = torch.einsum("oh,bchw->bcow", self.Dh, x)
        return torch.einsum("vw,bchw->bchv", self.Dw, x)

    def up(self, y: torch.Tensor) -> torch.Tensor:
        """(B,C,h,w) -> (B,C,h*s,w*s). The pseudo-inverse H-dagger."""
        y = torch.einsum("oh,bchw->bcow", self.Uh, y)
        return torch.einsum("vw,bchw->bchv", self.Uw, y)

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        """(B,C,h,w) -> (B,C,h*s,w*s) via the true pseudo-inverse H-dagger.

        Distinct from `up`: this is the least-squares inverse of the
        measurement operator, satisfying H(pinv(y)) = y, not a bicubic
        upsample. It looks blockier than `up` because it is not trying to be
        a good-looking image -- it is the minimum-norm solution.
        """
        y = torch.einsum("oh,bchw->bcow", self.Dh_pinv, y)
        return torch.einsum("vw,bchw->bchv", self.Dw_pinv, y)

    def back_project(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """DDNM range--null space correction.

            x0 <- x0 - H_dagger (H x0 - y)

        Replaces the component of x0 that H can see with the measurement,
        leaving the null-space component -- everything the measurement cannot
        determine, which is exactly where the model's generative freedom
        lives -- untouched. Applying this at every sampler step is what makes
        the H3 comparison a change of constraint rather than a change of model.

        Uses the true pseudo-inverse, so this is an exact projection: one
        application drives ||H x0 - y|| to numerical zero. With bicubic
        upsampling in place of H-dagger it would not be, and "with data
        consistency" would silently mean "with partial data consistency".
        """
        return x0 - self.pinv(self.down(x0) - y)


_CACHE: dict[tuple, BicubicOperator] = {}


def get_operator(hr_h: int, hr_w: int, scale: int, device, dtype=torch.float32) -> BicubicOperator:
    """Cached constructor. Matrices depend only on sizes, so rebuilding them
    per step would be pure waste in a 9,570-sample sweep."""
    key = (hr_h, hr_w, scale, str(device), str(dtype))
    if key not in _CACHE:
        _CACHE[key] = BicubicOperator(hr_h, hr_w, scale, device, dtype)
    return _CACHE[key]
