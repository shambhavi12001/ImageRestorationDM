"""The generative strength knob. This is the project's contribution.

No published repository provides this for super-resolution. SDEdit
(Meng et al., ICLR 2022) introduced partial forward diffusion as a
realism--faithfulness dial for stroke-based editing; SR3
(Saharia et al., TPAMI 2023) samples conditional diffusion super-resolution
at one fixed strength. This module is the two put together, instrumented so
the trade-off can be measured as a continuous function of a single scalar.

How it works. Instead of starting from pure noise, start from the bicubic
upsample of the observation, noised only partway up the chain:

    t0  = floor(s * T)
    x   = sqrt(ab[t0]) * up + sqrt(1 - ab[t0]) * eps
    x0  = DDIM(x, from t0 down to 0, spacing fixed at T / n_steps_at_full)

s = 0  returns the bicubic upsample unchanged -- no network call at all.
s = 1  is standard full-chain sampling.
In between, the model is given less room to depart from the observation, and
correspondingly fewer steps in which to do so.

Two invariants the sweep depends on, both asserted in the self-test:

  * s = 0 reproduces bicubic EXACTLY, so the strength axis is anchored at a
    known point rather than at an approximation of one.
  * the seed is an explicit argument and is recorded with every sample.
    Cross-seed variance is a measurement here, not a nuisance, so a sample
    whose seed was not recorded is unusable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.diffusion import to_image_space, to_model_space
from src.sampling.ddim import SamplerConfig, ddim_sample, ddim_timesteps

__all__ = ["StrengthConfig", "strength_to_t0", "sample_at_strength", "SampleResult"]


@dataclass
class StrengthConfig:
    T: int = 1000
    n_steps_at_full: int = 50     # DDIM steps when s = 1; spacing = T / this
    eta: float = 0.0
    back_projection: bool = False
    clip_x0: bool = True
    scale: int = 4

    @property
    def spacing(self) -> int:
        if self.T % self.n_steps_at_full:
            raise ValueError(
                f"T={self.T} is not divisible by n_steps_at_full="
                f"{self.n_steps_at_full}; uneven spacing would make the step "
                f"count a nonlinear function of strength"
            )
        return self.T // self.n_steps_at_full


@dataclass
class SampleResult:
    image: np.ndarray        # (H, W, 3) uint8
    strength: float
    seed: int
    t0: int
    nfe: int                 # network evaluations actually performed
    back_projection: bool


def strength_to_t0(s: float, T: int) -> int:
    """s in [0, 1] -> entry timestep. Clamped, so callers cannot step off
    the end of the schedule with s slightly over 1 from floating point."""
    if not (0.0 <= s <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {s}")
    return int(min(int(s * T), T - 1)) if s > 0 else 0


@torch.no_grad()
def sample_at_strength(
    model,
    diffusion,
    up: np.ndarray,
    lr: np.ndarray,
    s: float,
    seed: int,
    cfg: StrengthConfig | None = None,
    device: str | torch.device = "cpu",
) -> SampleResult:
    """One sample at generative strength `s` with an explicit seed.

    Parameters
    ----------
    up : (H, W, 3) uint8 bicubic upsample -- both the conditioning and, at
        partial strength, the thing that gets noised.
    lr : (h, w, 3) uint8 observation. Used only when back-projection is on.
    """
    cfg = cfg or StrengthConfig()
    t0 = strength_to_t0(s, cfg.T)

    # s = 0: the bicubic estimate itself. Returned without touching the model,
    # so this endpoint is exact rather than a 1-step approximation.
    if t0 == 0:
        return SampleResult(image=up.copy(), strength=s, seed=seed, t0=0,
                            nfe=0, back_projection=cfg.back_projection)

    cond = torch.from_numpy(to_model_space(up))[None].to(device)
    y = torch.from_numpy(to_model_space(lr))[None].to(device)

    # Seed an explicit generator rather than the global RNG: the sweep runs
    # many samples per process, and global state would make a resumed run
    # produce different noise from a fresh one.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(cond.shape, generator=gen, dtype=cond.dtype).to(device)

    ab_t0 = diffusion.alphas_cumprod[t0]
    x = ab_t0.sqrt() * cond + (1.0 - ab_t0).sqrt() * eps

    timesteps = ddim_timesteps(t0, cfg.spacing)
    sampler_cfg = SamplerConfig(eta=cfg.eta, back_projection=cfg.back_projection,
                                clip_x0=cfg.clip_x0, scale=cfg.scale)

    gen_dev = torch.Generator(device=device).manual_seed(int(seed) + 1)
    x0, nfe = ddim_sample(model, diffusion, x, cond, timesteps,
                          cfg=sampler_cfg, y=y, generator=gen_dev)

    return SampleResult(
        image=to_image_space(x0),
        strength=s,
        seed=seed,
        t0=t0,
        nfe=nfe,
        back_projection=cfg.back_projection,
    )
