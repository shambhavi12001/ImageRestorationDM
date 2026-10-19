"""Gate 1: prove the pipeline reproduces published bicubic numbers.

Run from the repo root:  python scripts/check_data.py

This is the first real gate. It needs no model and no GPU -- only numpy, PIL
and the datasets on disk. If bicubic does not reproduce, nothing downstream
can be trusted, because every later number is measured with the same
degradation operator and the same metric code.

Reference bicubic x4 values (Y channel, 4-px border crop) as reported across
the super-resolution literature, e.g. Table 2 of
  Lim et al., "Enhanced Deep Residual Networks for Single Image
  Super-Resolution", CVPRW 2017.
Matching these to within a few hundredths of a dB is strong evidence that our
imresize agrees with the MATLAB one every published number assumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import SRDataset
from src.metrics import psnr, ssim, lr_consistency

# dataset -> (published bicubic x4 PSNR, published bicubic x4 SSIM)
REFERENCE = {
    "set5": (28.42, 0.8104),
    "set14": (26.00, 0.7027),
    "bsd100": (25.96, 0.6675),
    "urban100": (23.14, 0.6577),
}

# DIV2K validation has no single canonical bicubic number in the literature;
# the proposal's gate is the range the pipeline should land in.
DIV2K_VAL_RANGE = (26.0, 28.0)

TOL_PSNR = 0.15   # dB; a correct imresize lands well inside this
TOL_SSIM = 0.01


def evaluate(ds: SRDataset) -> dict:
    ps, ss, cs = [], [], []
    for s in ds:
        ps.append(psnr(s.up, s.hr))
        ss.append(ssim(s.up, s.hr))
        cs.append(lr_consistency(s.up, s.lr, s.scale))
    return {
        "n": len(ds),
        "psnr": float(np.mean(ps)),
        "ssim": float(np.mean(ss)),
        "lr_consistency": float(np.mean(cs)),
        "manifest": ds.manifest(),
    }


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    failures: list[str] = []
    missing: list[str] = []

    print(f"\nbicubic x4 baseline, HR from {root}/\n")
    print(f"  {'dataset':<14}{'n':>5}{'PSNR':>9}{'ref':>9}{'d':>8}"
          f"{'SSIM':>9}{'ref':>9}{'LR-cons':>10}")
    print("  " + "-" * 73)

    for name, (ref_psnr, ref_ssim) in REFERENCE.items():
        try:
            ds = SRDataset(root, name)
        except FileNotFoundError as e:
            missing.append(f"{name}: {e}")
            continue
        r = evaluate(ds)
        d = r["psnr"] - ref_psnr
        flag = " " if abs(d) <= TOL_PSNR else "*"
        print(f"  {name:<14}{r['n']:>5}{r['psnr']:>9.2f}{ref_psnr:>9.2f}"
              f"{d:>+8.2f}{flag}{r['ssim']:>8.4f}{ref_ssim:>9.4f}"
              f"{r['lr_consistency']:>10.1f}")
        if abs(d) > TOL_PSNR:
            failures.append(f"{name}: PSNR {r['psnr']:.2f} vs published {ref_psnr:.2f}")
        if abs(r["ssim"] - ref_ssim) > TOL_SSIM:
            failures.append(f"{name}: SSIM {r['ssim']:.4f} vs published {ref_ssim:.4f}")

    # DIV2K validation: range check, not an exact match.
    try:
        ds = SRDataset(root, "div2k_val")
        r = evaluate(ds)
        lo, hi = DIV2K_VAL_RANGE
        ok = lo <= r["psnr"] <= hi
        print(f"  {'div2k_val':<14}{r['n']:>5}{r['psnr']:>9.2f}"
              f"{'[26-28]':>9}{'':>8}{' ' if ok else '*'}"
              f"{r['ssim']:>8.4f}{'':>9}{r['lr_consistency']:>10.1f}")
        if not ok:
            failures.append(f"div2k_val: PSNR {r['psnr']:.2f} outside {DIV2K_VAL_RANGE}")
    except FileNotFoundError as e:
        missing.append(f"div2k_val: {e}")

    try:
        n_train = len(SRDataset(root, "div2k_train"))
        print(f"\n  div2k_train: {n_train} images"
              + ("" if n_train == 800 else f"   <-- expected 800"))
        if n_train != 800:
            failures.append(f"div2k_train has {n_train} images, expected 800")
    except FileNotFoundError as e:
        missing.append(f"div2k_train: {e}")

    print()
    for m in missing:
        print(f"  [missing] {m}")
    for f in failures:
        print(f"  [FAIL]    {f}")

    if missing and not failures:
        print("\n  Some datasets are not downloaded yet. "
              "Run: bash scripts/download_data.sh")
        return 2
    if failures:
        print("\n  Gate 1 FAILED. Do not train until bicubic reproduces: a "
              "mismatch here means the degradation operator disagrees with "
              "the one every published number assumes.")
        return 1

    print("\n  Gate 1 passed: bicubic reproduces published numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
