"""Run the strength sweep.

    python scripts/run_sweep.py --checkpoint checkpoints/latest.pt

Resumable: rerun the same command after any interruption and it skips the
groups already on disk.

Shardable, for a Kubernetes indexed Job: each pod reads its own
JOB_COMPLETION_INDEX and takes a disjoint slice.

    python scripts/run_sweep.py --checkpoint ... \
        --shard-index $JOB_COMPLETION_INDEX --num-shards 8

The back-projection arm is the same command with --back-projection; it writes
to a separate subtree so the two curves can never be mixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.eval.sweep import SweepConfig, run_sweep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a training checkpoint")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--results", default="results/sweep")
    ap.add_argument("--datasets", nargs="*", default=["bsd100", "urban100"])
    ap.add_argument("--strengths", nargs="*", type=float,
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--back-projection", action="store_true")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--n-steps-at-full", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N images per dataset (pilot runs)")
    ap.add_argument("--save-images", type=int, default=0,
                    help="keep PNGs for the first N groups, for figures")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--shard-index", type=int,
                    default=int(os.environ.get("JOB_COMPLETION_INDEX", 0)))
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    import torch

    from src.models.diffusion import GaussianDiffusion
    from src.models.unet import UNet

    cfg_yaml = yaml.safe_load(Path(args.config).read_text())
    data_root = cfg_yaml["data"]["root"]
    scale = cfg_yaml["degradation"]["scale"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)

    T = ck.get("diffusion", {}).get("T", 1000)
    schedule = ck.get("diffusion", {}).get("schedule", "linear")
    base = ck.get("args", {}).get("base", 64)
    patch = ck.get("args", {}).get("patch", 128)

    model = UNet(image_size=patch, base=base).to(device)
    # Sample from the EMA weights. Raw weights give visibly noisier output,
    # which would inflate the very cross-seed variance this study measures.
    state = ck.get("ema") or ck["model"]
    model.load_state_dict(state)
    model.eval()

    diffusion = GaussianDiffusion(T=T, schedule=schedule, device=device)

    cfg = SweepConfig(
        datasets=tuple(args.datasets),
        strengths=tuple(args.strengths),
        seeds=tuple(args.seeds),
        scale=scale,
        back_projection=args.back_projection,
        T=T,
        n_steps_at_full=args.n_steps_at_full,
        eta=args.eta,
        limit=args.limit,
        save_images=args.save_images,
    )

    print(f"\n  device        {device}")
    print(f"  checkpoint    {args.checkpoint} (step {ck.get('step', '?')}, "
          f"{'EMA' if ck.get('ema') else 'raw'} weights)")
    print(f"  arm           {'back-projection' if args.back_projection else 'plain'}")
    print(f"  strengths     {list(cfg.strengths)}")
    print(f"  seeds         {list(cfg.seeds)}")
    print(f"  datasets      {list(cfg.datasets)}")
    print(f"  shard         {args.shard_index}/{args.num_shards}")
    print(f"  config hash   {cfg.hash()}\n")

    summary = run_sweep(
        model, diffusion, cfg, data_root, args.results,
        device=device, checkpoint=args.checkpoint,
        shard_index=args.shard_index, num_shards=args.num_shards,
        use_lpips=not args.no_lpips,
    )

    print("\n" + json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
