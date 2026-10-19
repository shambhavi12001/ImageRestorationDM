"""Gate 2: prove our evaluation reproduces EDSR's published numbers.

Run on the GPU box, from the repo root:

    curl -L -o checkpoints/edsr_x4.pt \
      https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt
    python scripts/check_edsr.py checkpoints/edsr_x4.pt

Gate 1 (scripts/check_data.py) proves the degradation operator is right.
This proves the whole evaluation path is right: a real model, our LR images,
our metric code, compared against a number someone else published.

If this passes, a later diffusion result that looks surprising is a fact about
the diffusion model rather than a bug in the harness. That is the entire
reason to spend a day on a baseline we are not even studying.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import SRDataset
from src.metrics import psnr, ssim, lr_consistency
from src.models.edsr import PUBLISHED_PSNR, EDSRRunner, load_edsr

TOL_DB = 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="path to edsr_x4-*.pt")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="*", default=list(PUBLISHED_PSNR))
    ap.add_argument("--out", default="results/edsr_baseline.json")
    ap.add_argument("--lpips", action="store_true",
                    help="also compute LPIPS (slower; not needed for the gate)")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  device: {device}")
    if device == "cpu":
        print("  warning: CPU inference on the large EDSR is slow "
              "(minutes per image on Urban100)")

    model = load_edsr(args.weights, scale=4, device=device)
    run = EDSRRunner(model, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {n_params/1e6:.1f}M parameters from {args.weights}\n")

    print(f"  {'dataset':<12}{'n':>4}{'PSNR':>9}{'published':>11}{'delta':>8}"
          f"{'SSIM':>9}{'LR-cons':>10}{'s/img':>8}")
    print("  " + "-" * 71)

    report: dict[str, dict] = {}
    failures: list[str] = []

    for name in args.datasets:
        try:
            ds = SRDataset(args.data_root, name)
        except FileNotFoundError as e:
            print(f"  {name:<12} [missing] {e}")
            continue

        ps, ss, cs, lp = [], [], [], []
        t0 = time.perf_counter()
        for s in ds:
            sr = run(s.lr)
            if sr.shape != s.hr.shape:
                raise RuntimeError(
                    f"{s.name}: EDSR returned {sr.shape}, expected {s.hr.shape}. "
                    "Was the LR image passed, rather than the bicubic upsample?"
                )
            ps.append(psnr(sr, s.hr))
            ss.append(ssim(sr, s.hr))
            cs.append(lr_consistency(sr, s.lr, s.scale))
            if args.lpips:
                from src.metrics import lpips_score
                lp.append(lpips_score(sr, s.hr))
        dt = (time.perf_counter() - t0) / len(ds)

        mean_psnr = float(np.mean(ps))
        ref = PUBLISHED_PSNR[name]
        delta = mean_psnr - ref
        flag = " " if abs(delta) <= TOL_DB else "*"

        print(f"  {name:<12}{len(ds):>4}{mean_psnr:>9.2f}{ref:>11.2f}"
              f"{delta:>+8.2f}{flag}{np.mean(ss):>8.4f}{np.mean(cs):>10.1f}{dt:>8.1f}")

        report[name] = {
            "n": len(ds),
            "psnr": mean_psnr,
            "psnr_published": ref,
            "psnr_delta": delta,
            "ssim": float(np.mean(ss)),
            "lr_consistency": float(np.mean(cs)),
            "lpips": float(np.mean(lp)) if lp else None,
            "seconds_per_image": dt,
            "manifest": ds.manifest(),
        }
        if abs(delta) > TOL_DB:
            failures.append(f"{name}: {mean_psnr:.2f} vs published {ref:.2f} "
                            f"({delta:+.2f} dB)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": "EDSR x4 (Lim et al., CVPRW 2017)", "weights": args.weights,
         "device": device, "results": report}, indent=2))
    print(f"\n  wrote {out}")

    if failures:
        print("\n  Gate 2 FAILED:")
        for f in failures:
            print(f"    {f}")
        print("\n  A gap of more than ~0.1 dB usually means one of: the LR "
              "images came from the archive instead of our degradation; the "
              "metric ran on RGB rather than Y; the border crop is missing; "
              "or the checkpoint is the baseline model (16 blocks, 64 feats) "
              "rather than the large one.")
        return 1

    if not report:
        print("\n  No datasets evaluated. Run scripts/download_data.sh first.")
        return 2

    print("\n  Gate 2 passed: the evaluation path reproduces published EDSR "
          "numbers. The harness can be trusted for the diffusion sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
