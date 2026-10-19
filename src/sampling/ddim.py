"""DDIM sampling, with optional DDNM data consistency.

  Song, Meng, Ermon, "Denoising Diffusion Implicit Models", ICLR 2021
  Wang, Yu, Zhang, "Zero-Shot Image Restoration Using Denoising Diffusion
  Null-Space Model" (DDNM), ICLR 2023

The update, from timestep t to the next timestep t' in the chosen subsequence:

    x0_hat = (x_t - sqrt(1 - ab_t) eps) / sqrt(ab_t)
    sigma  = eta sqrt((1 - ab_t') / (1 - ab_t)) sqrt(1 - ab_t / ab_t')
    x_t'   = sqrt(ab_t') x0_hat + sqrt(1 - ab_t' - sigma^2) eps + sigma z

eta = 0 makes the trajectory deterministic given the initial noise, which is
what this project wants: with eta = 0 the only stochasticity is the seed, so
cross-seed variance measures the posterior spread rather than accumulated
sampler noise. eta is still exposed because an eta sweep is the obvious
robustness check if a reviewer asks whether the hallucination trend is an
artifact of the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.data.torch_degradation import get_operator

__all__ = ["ddim_timesteps", "ddim_sample", "SamplerConfig"]


@dataclass
class SamplerConfig:
    eta: float = 0.0
    back_projection: bool = False
    clip_x0: bool = True          # images live in [-1, 1]; clamping x0_hat is
                                  # standard and stops early steps diverging
    scale: int = 4


def ddim_timesteps(t0: int, spacing: int) -> list[int]:
    """Descending timestep subsequence from t0 down to 0.

    Spacing is held FIXED rather than dividing a constant number of steps
    across whatever range t0 covers. That choice is load-bearing for this
    project: it means a lower strength genuinely performs fewer network
    evaluations, so sampling cost falls with strength and can be reported as
    a result, instead of every strength costing the same.

    t0 = 0 yields an empty list: no denoising to do.
    """
    if t0 <= 0:
        return []
    if spacing < 1:
        raise ValueError(f"spacing must be >= 1, got {spacing}")
    return list(range(0, t0, spacing))[::-1]


@torch.no_grad()
def ddim_sample(
    model,
    diffusion,
    x_t: torch.Tensor,
    cond: torch.Tensor,
    timesteps: list[int],
    cfg: SamplerConfig | None = None,
    y: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Denoise x_t to x_0 along `timesteps`.

    Parameters
    ----------
    x_t : (B,3,H,W) starting point, already noised to timesteps[0].
    cond : (B,3,H,W) bicubic upsample of the observation -- the conditioning.
    y : (B,3,h,w) the low-resolution observation. Required when
        cfg.back_projection is on; unused otherwise.

    Returns
    -------
    (x0, nfe) -- the sample and the number of network evaluations used.
    """
    cfg = cfg or SamplerConfig()
    if not timesteps:
        return x_t, 0

    if cfg.back_projection:
        if y is None:
            raise ValueError("back_projection requires the observation y")
        op = get_operator(x_t.shape[-2], x_t.shape[-1], cfg.scale,
                          x_t.device, x_t.dtype)

    ab = diffusion.alphas_cumprod
    x = x_t
    nfe = 0

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

        ab_t = ab[t]
        # t_prev = -1 means the final jump to a clean image, where alpha_bar = 1.
        ab_prev = ab[t_prev] if t_prev >= 0 else torch.ones_like(ab_t)

        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model(x, t_batch, cond)
        nfe += 1

        x0 = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if cfg.clip_x0:
            x0 = x0.clamp(-1.0, 1.0)

        if cfg.back_projection:
            # Enforce consistency on the estimate of the clean image, not on
            # the noisy iterate: x_t is a mixture of signal and noise, and H
            # applied to it would not be comparable with y.
            x0 = op.back_project(x0, y)
            if cfg.clip_x0:
                x0 = x0.clamp(-1.0, 1.0)

        if t_prev < 0:
            x = x0
            break

        sigma = cfg.eta * (
            ((1.0 - ab_prev) / (1.0 - ab_t)).sqrt() * (1.0 - ab_t / ab_prev).sqrt()
        )
        # Re-derive eps from the (possibly corrected) x0 so the direction term
        # stays consistent with it; with back-projection off this is exactly
        # the eps the model returned.
        eps_eff = (x - ab_t.sqrt() * x0) / (1.0 - ab_t).sqrt()

        dir_term = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps_eff
        x = ab_prev.sqrt() * x0 + dir_term

        if cfg.eta > 0:
            z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x + sigma * z

    return x, nfe
