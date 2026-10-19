"""The diffusion process: noise schedule, forward corruption, training loss.

  Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020
  Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models",
  ICML 2021  (the cosine schedule)

This module owns `alphas_cumprod`. Everything else in the project reads it:
the sampler needs it to step, and the strength knob needs alpha_bar[t0] to
decide how much noise to inject when it enters the chain partway. Keeping one
authoritative copy here is what stops training and sampling from silently
drifting apart.

Convention throughout: images are float tensors in [-1, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["make_beta_schedule", "GaussianDiffusion", "to_model_space", "to_image_space"]


# --------------------------------------------------------------------------
# image <-> model space
# --------------------------------------------------------------------------


def to_model_space(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] or float [0,1]  ->  float32 [-1,1], channels-first."""
    x = img.astype(np.float32)
    if img.dtype == np.uint8:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0) * 2.0 - 1.0
    return x.transpose(2, 0, 1)


def to_image_space(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """float [-1,1] channels-first  ->  uint8 [0,255] (H,W,3)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 4:
        x = x[0]
    x = (np.clip(x, -1.0, 1.0) + 1.0) * 127.5
    return np.round(x).astype(np.uint8).transpose(1, 2, 0)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def make_beta_schedule(kind: str = "linear", T: int = 1000) -> np.ndarray:
    """Return betas, shape (T,).

    "linear" is the DDPM schedule, 1e-4 to 0.02 scaled for T=1000. It is the
    default because the strength knob's mapping s -> t0 is easiest to reason
    about on a schedule whose noise level rises roughly evenly; the cosine
    schedule spends many more steps at low noise, which would make equal
    steps in s correspond to very unequal steps in actual corruption.

    "cosine" is Nichol & Dhariwal's, kept available for an ablation.
    """
    if kind == "linear":
        scale = 1000.0 / T
        return np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64)

    if kind == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64) / T
        f = np.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_cumprod = f / f[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return np.clip(betas, 0.0, 0.999)

    raise ValueError(f"unknown schedule {kind!r}; use 'linear' or 'cosine'")


# --------------------------------------------------------------------------


@dataclass
class DiffusionConfig:
    T: int = 1000
    schedule: str = "linear"


class GaussianDiffusion:
    """Forward process and training objective.

    Sampling lives in src/sampling/, not here, so that the training definition
    and the several sampling variants cannot drift.
    """

    def __init__(self, T: int = 1000, schedule: str = "linear", device: str = "cpu"):
        self.T = T
        self.schedule = schedule
        self.device = device

        betas = make_beta_schedule(schedule, T)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.betas = t(betas)
        self.alphas = t(alphas)
        self.alphas_cumprod = t(alphas_cumprod)
        self.sqrt_alphas_cumprod = t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = t(np.sqrt(1.0 - alphas_cumprod))

    def to(self, device: str) -> "GaussianDiffusion":
        self.device = device
        for name in ("betas", "alphas", "alphas_cumprod",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ---------------------------------------------------------------- forward

    @staticmethod
    def _gather(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """arr[t] reshaped to broadcast against a batch of images."""
        out = arr.gather(0, t)
        return out.reshape(-1, *([1] * (ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Corrupt x0 to timestep t in one shot.

            x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps

        This closed form is also exactly what the strength knob uses to enter
        the chain at t0 without simulating the intervening steps.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self._gather(self.sqrt_alphas_cumprod, t, x0.ndim)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
        return a * x0 + b * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                            eps: torch.Tensor) -> torch.Tensor:
        """Invert q_sample for the model's epsilon estimate."""
        a = self._gather(self.sqrt_alphas_cumprod, t, x_t.ndim)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        return (x_t - b * eps) / a

    # ----------------------------------------------------------------- loss

    def loss(self, model, x0: torch.Tensor, cond: torch.Tensor,
             t: torch.Tensor | None = None, noise: torch.Tensor | None = None,
             loss_type: str = "l1") -> torch.Tensor:
        """Simple epsilon-prediction objective.

        L1 by default rather than L2: SR3 reports it gives slightly sharper
        results for super-resolution, and it is less sensitive to the rare
        very-high-noise samples that dominate an L2 batch mean.
        """
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (b,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        eps_hat = model(x_t, t, cond)

        if loss_type == "l1":
            return (eps_hat - noise).abs().mean()
        if loss_type == "l2":
            return (eps_hat - noise).pow(2).mean()
        raise ValueError(f"unknown loss_type {loss_type!r}")

    # ------------------------------------------------------------- sanity

    def describe(self) -> dict:
        """Numbers worth recording with every run and eyeballing once.

        `snr_at` reports the signal-to-noise ratio at a few depths. If
        alpha_bar at T-1 is not close to zero, the forward process never
        reaches noise and samples started from pure noise will be wrong.
        """
        ab = self.alphas_cumprod
        marks = {f"t={i}": float(ab[i]) for i in
                 (0, self.T // 4, self.T // 2, 3 * self.T // 4, self.T - 1)}
        return {
            "T": self.T,
            "schedule": self.schedule,
            "alpha_bar": marks,
            "alpha_bar_final": float(ab[-1]),
            "reaches_noise": bool(ab[-1] < 1e-3),
        }
