"""Train the one conditional diffusion model this project uses.

    python scripts/train.py --config configs/base.yaml

Everything else in the study is deterministic (bicubic) or pretrained (EDSR),
so this is the only training run. The back-projection variant in T10 reuses
these same weights -- it changes sampling, not the model.

Resumable by design: --resume picks up optimizer, EMA and step count, because
a six-week schedule cannot afford to lose a multi-day run to a dropped SSH
session.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import SRDataset, random_crop_pair
from src.metrics import psnr
from src.models.diffusion import GaussianDiffusion, to_image_space, to_model_space
from src.models.unet import UNet


class EMA:
    """Exponential moving average of the weights.

    Samples are drawn from the EMA copy, not the live weights. This is not
    optional polish: diffusion sample quality from raw weights is visibly
    worse and noticeably noisier run to run, which would add variance to
    exactly the cross-seed measurement this project treats as signal.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def patch_batch(ds: SRDataset, idx: np.ndarray, patch: int, scale: int,
                rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    hrs, ups = [], []
    for i in idx:
        hr_raw = ds.paths[i]
        from src.data.datasets import load_image
        hr, up = random_crop_pair(load_image(hr_raw), scale, patch, rng)
        hrs.append(to_model_space(hr))
        ups.append(to_model_space(up))
    return (torch.from_numpy(np.stack(hrs)), torch.from_numpy(np.stack(ups)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128, help="HR patch size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--keep-last", type=int, default=5,
                    help="how many numbered checkpoints to retain (0 = all). "
                         "Each is ~1 GB (model + EMA + Adam state), so an "
                         "unpruned 300k-step run writes ~60 GB and will fill "
                         "a shared PVC.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16"],
                    help="bfloat16 autocast on CUDA. Roughly 3-4x faster than "
                         "fp32 on a 4090 because it reaches the tensor cores. "
                         "bf16 rather than fp16: it has the same exponent "
                         "range as fp32, so the epsilon-prediction loss cannot "
                         "underflow and no GradScaler is needed.")
    ap.add_argument("--overfit", type=int, default=0,
                    help="train on N images only -- the overfit-10 gate")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    scale = cfg["degradation"]["scale"]
    data_root = cfg["data"]["root"]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = SRDataset(data_root, "div2k_train", scale=scale,
                      limit=args.overfit or None)
    if device == "cuda":
        # TF32 for the fp32 path too; harmless at this precision and free.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True      # fixed shapes, so autotune pays

    print(f"\n  device {device} | train {len(train)} images | patch {args.patch} "
          f"| batch {args.batch_size} | amp {args.amp if device == 'cuda' else 'off (cpu)'}")

    model = UNet(image_size=args.patch, base=args.base).to(device)
    diff = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    print(f"  model {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  schedule: {json.dumps(diff.describe()['alpha_bar'])}")
    if not diff.describe()["reaches_noise"]:
        raise SystemExit("  schedule never reaches noise; sampling from "
                         "N(0,I) would be invalid")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, args.ema_decay)
    start = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
        print(f"  resumed from {args.resume} at step {start}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    running: list[float] = []
    t_start = time.perf_counter()

    for step in range(start, args.steps):
        idx = rng.integers(0, len(train), args.batch_size)
        x0, cond = patch_batch(train, idx, args.patch, scale, rng)
        x0, cond = x0.to(device), cond.to(device)

        use_amp = args.amp == "bf16" and device == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = diff.loss(model, x0, cond, loss_type="l1")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        running.append(loss.item())

        if (step + 1) % args.log_every == 0:
            mean = float(np.mean(running[-args.log_every:]))
            rate = (step + 1 - start) / (time.perf_counter() - t_start)
            eta_h = (args.steps - step - 1) / max(rate, 1e-9) / 3600
            print(f"  step {step+1:>7d}  loss {mean:.4f}  "
                  f"{rate:.2f} it/s  eta {eta_h:.1f}h")
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step + 1, "loss": mean,
                                    "it_per_s": rate}) + "\n")

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            ck = {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "diffusion": {"T": args.T, "schedule": args.schedule},
                "train_manifest": train.manifest(),
            }
            path = out / f"step_{step+1:07d}.pt"
            torch.save(ck, path)
            # Write latest.pt atomically. A pod evicted mid-write would
            # otherwise leave a truncated file, and the restarted job would
            # resume from a corrupt checkpoint instead of the last good one.
            tmp = out / "latest.pt.tmp"
            torch.save(ck, tmp)
            tmp.replace(out / "latest.pt")
            print(f"    saved {path}")

            if args.keep_last > 0:
                numbered = sorted(out.glob("step_*.pt"))
                for old in numbered[: -args.keep_last]:
                    old.unlink()

    print(f"\n  done in {(time.perf_counter()-t_start)/3600:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
