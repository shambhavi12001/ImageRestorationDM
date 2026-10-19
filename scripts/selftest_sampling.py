"""Correctness checks for the sampler, the strength knob and back-projection.

Run from the repo root:  python scripts/selftest_sampling.py

Needs torch; runs on CPU in well under a minute. Separate from
`scripts/selftest.py` so that the degradation and metric gate stays runnable
in a plain numpy environment.

These use an UNTRAINED network on purpose. Every property checked here is a
property of the sampling machinery, not of the weights: step counts, seed
reproducibility, the s=0 endpoint, and whether back-projection actually
projects. A trained model would hide a bug in any of them behind plausible
output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skimage import data

from src.data.degradation import degrade, imresize
from src.data.torch_degradation import BicubicOperator
from src.metrics import lr_consistency
from src.models.diffusion import GaussianDiffusion, to_model_space
from src.models.unet import UNet
from src.sampling.ddim import ddim_timesteps
from src.sampling.strength import StrengthConfig, sample_at_strength, strength_to_t0

results: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((bool(cond), name, detail))


def test_operator() -> None:
    hr = data.astronaut()[:, :, :3][:256, :256]
    hrc, lr, _up = degrade(hr, 4)
    op = BicubicOperator(256, 256, 4)

    x = torch.from_numpy(hrc.astype(np.float32).transpose(2, 0, 1))[None]
    got = op.down(x)[0].numpy().transpose(1, 2, 0)
    want = imresize(hrc.astype(np.float64), scale=0.25)
    check("torch down matches numpy imresize", np.abs(got - want).max() < 2e-3,
          f"max |d| {np.abs(got - want).max():.2e}")

    y = torch.from_numpy(lr.astype(np.float32).transpose(2, 0, 1))[None]
    got = op.up(y)[0].numpy().transpose(1, 2, 0)
    want = imresize(lr.astype(np.float64), scale=4.0)
    check("torch up matches numpy imresize", np.abs(got - want).max() < 2e-3,
          f"max |d| {np.abs(got - want).max():.2e}")


def test_back_projection() -> None:
    hr = data.astronaut()[:, :, :3][:256, :256]
    _hrc, lr, up = degrade(hr, 4)
    op = BicubicOperator(256, 256, 4)

    ym = torch.from_numpy(to_model_space(lr))[None]
    xm = torch.from_numpy(to_model_space(up))[None]

    # The pseudo-inverse must be a real right inverse. Bicubic upsampling is
    # NOT: using it here leaves ~40% of the residual and back-projection stops
    # being a projection.
    err = (op.down(op.pinv(ym)) - ym).abs().max().item()
    check("H(pinv(y)) == y", err < 1e-4, f"max err {err:.2e}")

    before = (op.down(xm) - ym).abs().mean().item()
    after = (op.down(op.back_project(xm, ym)) - ym).abs().mean().item()
    check("back_project drives ||Hx-y|| to zero", after < 1e-5,
          f"{before:.2e} -> {after:.2e}")

    g = torch.Generator().manual_seed(0)
    noisy = xm + 0.3 * torch.randn(xm.shape, generator=g)
    after_n = (op.down(op.back_project(noisy, ym)) - ym).abs().mean().item()
    check("...on a corrupted image too", after_n < 1e-5, f"{after_n:.2e}")

    # It must be a projection, not a blur: whatever the measurement cannot
    # see has to survive, or back-projection would just erase the model's
    # contribution and H3 would be trivially true.
    d = (op.back_project(noisy, ym) - op.back_project(xm, ym)).abs().mean().item()
    check("null-space detail survives", d > 1e-3, f"mean |d| {d:.4f}")


def test_schedule() -> None:
    cfg = StrengthConfig(T=1000, n_steps_at_full=50)
    steps = [len(ddim_timesteps(strength_to_t0(s, 1000), cfg.spacing))
             for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    check("step count is linear in strength", steps == [0, 10, 20, 30, 40, 50], f"{steps}")

    ok = all(
        ddim_timesteps(t, 20) == sorted(ddim_timesteps(t, 20), reverse=True)
        and ddim_timesteps(t, 20)[-1] == 0
        for t in (1, 19, 20, 21, 999)
    )
    check("schedule descends and always ends at 0", ok)

    try:
        strength_to_t0(1.5, 1000)
        check("rejects strength > 1", False, "did not raise")
    except ValueError:
        check("rejects strength > 1", True)

    try:
        StrengthConfig(T=1000, n_steps_at_full=37).spacing
        check("rejects indivisible step spacing", False, "did not raise")
    except ValueError:
        check("rejects indivisible step spacing", True)


def test_strength_knob() -> None:
    torch.manual_seed(0)
    net = UNet(image_size=64, base=16, mults=(1, 2), n_res=1, attn_res=()).eval()
    diff = GaussianDiffusion(T=1000)
    cfg = StrengthConfig(T=1000, n_steps_at_full=50)

    hr = data.astronaut()[:, :, :3][:64, :64]
    _hrc, lr, up = degrade(hr, 4)

    def S(s, seed, c=cfg):
        return sample_at_strength(net, diff, up, lr, s=s, seed=seed, cfg=c)

    r0 = S(0.0, 0)
    check("s=0 returns bicubic exactly", np.array_equal(r0.image, up), f"nfe {r0.nfe}")

    a, b, c = S(0.4, 7), S(0.4, 7), S(0.4, 8)
    check("same seed reproduces bit-for-bit", np.array_equal(a.image, b.image))
    check("different seed gives a different sample", not np.array_equal(a.image, c.image),
          f"mean |d| {np.abs(a.image.astype(float) - c.image.astype(float)).mean():.1f}")

    nfe = [S(s, 0).nfe for s in (0.2, 0.6, 1.0)]
    check("nfe recorded and rises with strength", nfe == [10, 30, 50], f"{nfe}")

    cfg_bp = StrengthConfig(T=1000, n_steps_at_full=50, back_projection=True)
    c_no = lr_consistency(S(0.6, 3).image, lr, 4)
    c_bp = lr_consistency(S(0.6, 3, cfg_bp).image, lr, 4)
    check("back-projection raises LR consistency", c_bp > c_no,
          f"{c_no:.1f} -> {c_bp:.1f} dB")


def main() -> int:
    test_operator()
    test_back_projection()
    test_schedule()
    test_strength_knob()

    print()
    width = max(len(n) for _, n, _ in results) + 2
    for passed, name, detail in results:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {name:<{width}} {detail}")

    n_fail = sum(1 for p, _, _ in results if not p)
    print(f"\n  {len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
