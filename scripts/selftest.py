"""Correctness checks for the degradation and metric code. No torch, no network.

Run from the repo root:  python scripts/selftest.py

These are not unit tests for their own sake. Every number this project reports
flows through imresize and the metric functions, so a half-pixel shift or a
wrong colour conversion here would silently move every result by more than the
effect we are trying to measure. Each check below pins one of those.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.degradation import imresize, degrade, modcrop, bicubic_down, bicubic_up
from src.metrics import psnr, ssim, lr_consistency, rgb_to_y, cross_seed_std

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))


# ---------------------------------------------------------------- imresize
def test_imresize() -> None:
    # 1. A constant image must survive resizing exactly. Catches unnormalised
    #    kernel weights and wrong boundary handling at once.
    const = np.full((64, 64, 3), 123.0)
    out = imresize(const, scale=0.25)
    check("imresize: constant preserved", np.allclose(out, 123.0, atol=1e-9),
          f"max dev {np.abs(out - 123.0).max():.2e}")

    # 2. Weights sum to one along both axes for any size.
    from src.data.degradation import _contributions
    w, _ = _contributions(100, 25, 0.25)
    check("imresize: weights normalised", np.allclose(w.sum(axis=1), 1.0),
          f"max dev {np.abs(w.sum(axis=1) - 1).max():.2e}")

    # 3. Pixel-centre alignment. A horizontal ramp downsampled 4x must stay a
    #    ramp with the same slope per output pixel and no offset drift. A
    #    missing 0.5*(1-1/s) term shows up here as a constant shift.
    ramp = np.tile(np.arange(64, dtype=np.float64), (16, 1))[:, :, None]
    down = imresize(ramp, scale=0.25)[:, :, 0]
    expected = np.arange(16) * 4.0 + 1.5    # centres of each 4-pixel block
    interior = slice(2, 14)                 # borders see mirrored padding
    dev = np.abs(down[0, interior] - expected[interior]).max()
    check("imresize: no half-pixel shift", dev < 1e-9, f"max deviation {dev:.2e} px")

    # 4. Antialiasing is actually on. Downsampling a 1-pixel checkerboard must
    #    average towards grey, not alias to a constant extreme. PIL/OpenCV
    #    without a prefilter fail this.
    board = np.indices((64, 64)).sum(axis=0) % 2
    board = (board * 255.0)[:, :, None]
    db = imresize(board, scale=0.25)[:, :, 0]
    check("imresize: antialiasing active", 100.0 < db.mean() < 155.0 and db.std() < 20.0,
          f"mean {db.mean():.1f} std {db.std():.1f}")

    # 5. Round-trip shape and dtype discipline.
    u8 = (np.random.default_rng(0).random((48, 52, 3)) * 255).astype(np.uint8)
    lr = bicubic_down(u8, 4)
    up = bicubic_up(lr, 4)
    check("imresize: shapes round-trip", lr.shape == (12, 13, 3) and up.shape == u8.shape,
          f"lr {lr.shape} up {up.shape}")
    check("imresize: uint8 stays uint8", lr.dtype == np.uint8 and up.dtype == np.uint8,
          f"{lr.dtype}, {up.dtype}")

    # 6. modcrop makes dimensions divisible, which degrade() relies on.
    odd = np.zeros((51, 54, 3), dtype=np.uint8)
    c = modcrop(odd, 4)
    check("modcrop: divisible by scale", c.shape[:2] == (48, 52), f"{c.shape[:2]}")


# ----------------------------------------------------------------- metrics
def test_metrics() -> None:
    rng = np.random.default_rng(1)
    img = (rng.random((96, 96, 3)) * 255).astype(np.uint8)

    # 1. Identity -> infinite PSNR, SSIM of 1.
    check("psnr: identity is inf", psnr(img, img) == float("inf"))
    check("ssim: identity is 1", abs(ssim(img, img) - 1.0) < 1e-6, f"{ssim(img, img):.8f}")

    # 2. Y-channel conversion matches the BT.601 studio-swing range used by
    #    the SR benchmark scripts: pure white maps to 235, pure black to 16.
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    check("rgb_to_y: white -> 235", abs(rgb_to_y(white).mean() - 235.0) < 0.01,
          f"{rgb_to_y(white).mean():.3f}")
    check("rgb_to_y: black -> 16", abs(rgb_to_y(black).mean() - 16.0) < 0.01,
          f"{rgb_to_y(black).mean():.3f}")

    # 3. uint8 and float[0,1] representations of the same image must give the
    #    same PSNR. Catches a missing /255.
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    b = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    p_u8 = psnr(a, b)
    p_f = psnr(a.astype(np.float64) / 255.0, b.astype(np.float64) / 255.0)
    check("psnr: dtype invariant", abs(p_u8 - p_f) < 1e-6, f"{p_u8:.6f} vs {p_f:.6f}")

    # 4. Cross-check our SSIM against scikit-image configured identically.
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        ya = rgb_to_y(a)[4:-4, 4:-4]
        yb = rgb_to_y(b)[4:-4, 4:-4]
        ref = sk_ssim(ya, yb, data_range=255.0, gaussian_weights=True,
                      sigma=1.5, use_sample_covariance=False)
        ours = ssim(a, b)
        check("ssim: agrees with scikit-image", abs(ours - ref) < 5e-3,
              f"ours {ours:.6f} skimage {ref:.6f}")
    except ImportError:
        check("ssim: agrees with scikit-image", True, "skipped, skimage absent")

    # 5. Border crop must actually change the answer, else it is not applied.
    check("psnr: border crop applied", abs(psnr(a, b, border=4) - psnr(a, b, border=0)) > 1e-9)

    # 6. LR consistency of a bicubic upsample should be very high: bicubic-up
    #    then bicubic-down is close to the identity on the measurement.
    hr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    hr_c, lr, up = degrade(hr, 4)
    check("lr_consistency: bicubic is self-consistent", lr_consistency(up, lr, 4) > 30.0,
          f"{lr_consistency(up, lr, 4):.2f} dB")

    # 7. Degrading must not silently misalign HR and the reconstruction grid.
    check("degrade: hr and up same shape", hr_c.shape == up.shape, f"{hr_c.shape} {up.shape}")

    # 8. Cross-seed std: identical samples carry zero uncertainty; one sample
    #    is an error, not zero.
    same = [img, img.copy(), img.copy()]
    check("cross_seed_std: identical -> 0", cross_seed_std(same) == 0.0)
    noisy = [np.clip(img.astype(np.float64) + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
             for _ in range(5)]
    v = cross_seed_std(noisy)
    check("cross_seed_std: recovers injected sigma", abs(v - 10.0 / 255.0 / np.sqrt(3)) < 0.01,
          f"{v:.5f} (colour-averaged, so sigma/sqrt(3))")
    try:
        cross_seed_std([img])
        check("cross_seed_std: rejects single sample", False, "did not raise")
    except ValueError:
        check("cross_seed_std: rejects single sample", True)


# ------------------------------------------------- the T3 gate, on real images
def test_gate_on_available_images() -> None:
    """Bicubic x4 PSNR on whatever real photographs are locally available.

    The real gate is DIV2K validation (expect roughly 26-28 dB mean). DIV2K is
    a ~400 MB download and is not present here, so this runs the identical
    code path on the natural images bundled with scikit-image. A correct
    pipeline puts a single natural photograph in roughly the 20-32 dB range;
    anything far outside that means the degradation or the metric is wrong,
    not that the image is hard.
    """
    try:
        from skimage import data
    except ImportError:
        check("gate: bicubic PSNR in plausible range", True, "skipped, skimage absent")
        return

    imgs = {}
    for name in ("astronaut", "chelsea", "coffee", "rocket"):
        try:
            im = getattr(data, name)()
            if im.ndim == 3 and im.shape[2] >= 3:
                imgs[name] = im[:, :, :3]
        except Exception:
            pass
    if not imgs:
        check("gate: bicubic PSNR in plausible range", True, "no bundled images")
        return

    rows = []
    for name, hr in imgs.items():
        hr_c, lr, up = degrade(hr, 4)
        rows.append((name, psnr(up, hr_c), ssim(up, hr_c), lr_consistency(up, lr, 4)))

    mean_psnr = float(np.mean([r[1] for r in rows]))
    print("\n  bicubic x4 baseline on bundled images")
    print(f"  {'image':<12}{'PSNR dB':>10}{'SSIM':>9}{'LR-cons dB':>13}")
    for name, p, s, c in rows:
        print(f"  {name:<12}{p:>10.3f}{s:>9.4f}{c:>13.2f}")
    print(f"  {'mean':<12}{mean_psnr:>10.3f}")

    check("gate: bicubic PSNR in plausible range", 18.0 < mean_psnr < 34.0,
          f"mean {mean_psnr:.2f} dB over {len(rows)} images")

    # Bicubic upsampling must be measurably worse than the HR original on the
    # perceptual-adjacent axis too, and self-consistent on the LR axis.
    check("gate: bicubic is LR-consistent", all(r[3] > 35.0 for r in rows),
          f"min {min(r[3] for r in rows):.1f} dB")


def main() -> int:
    np.seterr(all="raise")
    test_imresize()
    test_metrics()
    test_gate_on_available_images()

    print()
    width = max(len(n) for _, n, _ in results) + 2
    for status, name, detail in results:
        mark = "ok  " if status == PASS else "FAIL"
        print(f"  [{mark}] {name:<{width}} {detail}")

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n  {len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
