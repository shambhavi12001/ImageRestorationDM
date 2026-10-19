"""The sweep: every (image, strength) evaluated at every seed.

Three properties the six-week schedule depends on.

**The unit of work is (image, strength), not (image, strength, seed).**
Cross-seed standard deviation is one of the six metrics, and it is defined
across the seeds for a fixed image and strength. Computing it means having
all N seed outputs in hand at once. Doing a group together lets us measure
the spread and then discard the images, instead of writing 6,000 PNGs to
disk and reducing them later.

**Resumable at that granularity.** One JSON file per group, named from its
coordinates, never overwritten. A run that dies -- an eviction, a full disk,
a laptop lid -- is restarted with the same command and skips what exists.
There is no run state outside the results directory.

**Shardable.** The work list is sorted deterministically, so shard k of n
takes every nth entry starting at k and no two shards collide. That maps
directly onto a Kubernetes indexed Job: each pod gets one GPU and its own
JOB_COMPLETION_INDEX, and eight pods turn a day into about three hours. The
sweep has no communication between shards, so a failed shard is retried
alone.

Every row records the git commit, the config hash and the checkpoint it came
from. A results directory that outlives memory of how it was produced is
worth very little.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.data.datasets import SRDataset, Sample
from src.metrics import cross_seed_std, lr_consistency, psnr, ssim
from src.sampling.strength import StrengthConfig, sample_at_strength

__all__ = ["SweepConfig", "run_sweep", "work_list", "group_path"]


# --------------------------------------------------------------------------


@dataclass
class SweepConfig:
    datasets: tuple[str, ...] = ("bsd100", "urban100")
    strengths: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    scale: int = 4
    back_projection: bool = False
    T: int = 1000
    n_steps_at_full: int = 50
    eta: float = 0.0
    limit: int | None = None          # first N images per dataset, for a pilot
    save_images: int = 0              # keep the first N groups' PNGs for figures

    def hash(self) -> str:
        """Short fingerprint of the protocol. Changing any field changes it,
        so rows from two different protocols can never be silently averaged."""
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        sha = out.stdout.strip() or "unknown"
        return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except Exception:
        return "unknown"


def _provenance(cfg: SweepConfig, checkpoint: str, device: str) -> dict:
    return {
        "git_commit": _git_commit(),
        "config_hash": cfg.hash(),
        "checkpoint": checkpoint,
        "device": device,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------
# work list and sharding
# --------------------------------------------------------------------------


def work_list(cfg: SweepConfig, data_root: str) -> list[tuple[str, int, float]]:
    """Every (dataset, image index, strength), in a deterministic order.

    Sorted so that shard k of n is reproducible across processes and machines:
    two pods computing the list independently must agree on it exactly, or
    they will duplicate some groups and skip others.
    """
    items: list[tuple[str, int, float]] = []
    for name in sorted(cfg.datasets):
        ds = SRDataset(data_root, name, scale=cfg.scale, limit=cfg.limit)
        for i in range(len(ds)):
            for s in sorted(cfg.strengths):
                items.append((name, i, float(s)))
    return items


def shard(items: Sequence, index: int, total: int) -> list:
    if total < 1 or not (0 <= index < total):
        raise ValueError(f"bad shard {index}/{total}")
    return list(items[index::total])


def group_path(results_dir: Path, dataset: str, image: str, strength: float,
               back_projection: bool) -> Path:
    tag = "bp" if back_projection else "plain"
    # Strength in the filename to 2 dp: s=0.2 and s=0.20 must not produce two
    # different files for the same group.
    return results_dir / tag / dataset / f"{image}__s{strength:.2f}.json"


# --------------------------------------------------------------------------
# one group
# --------------------------------------------------------------------------


def evaluate_group(
    model,
    diffusion,
    sample: Sample,
    strength: float,
    cfg: SweepConfig,
    device: str,
    lpips_fn=None,
    save_dir: Path | None = None,
) -> dict:
    """All seeds for one (image, strength). Returns the row to write."""
    scfg = StrengthConfig(
        T=cfg.T,
        n_steps_at_full=cfg.n_steps_at_full,
        eta=cfg.eta,
        back_projection=cfg.back_projection,
        scale=cfg.scale,
    )

    # s = 0 is deterministic: it returns the bicubic upsample with no network
    # call, so every seed gives the identical image. Running five of them
    # would waste nothing but time and report a cross-seed std of exactly 0
    # either way -- but doing one and saying so is honest about what happened.
    seeds = (cfg.seeds[0],) if strength == 0.0 else cfg.seeds

    images: list[np.ndarray] = []
    per_seed: list[dict] = []
    nfe = 0

    for seed in seeds:
        t0 = time.perf_counter()
        res = sample_at_strength(
            model, diffusion, sample.up, sample.lr, s=strength, seed=seed,
            cfg=scfg, device=device,
        )
        elapsed = time.perf_counter() - t0
        nfe = res.nfe

        row = {
            "seed": int(seed),
            "psnr": psnr(res.image, sample.hr),
            "ssim": ssim(res.image, sample.hr),
            "lr_consistency": lr_consistency(res.image, sample.lr, cfg.scale),
            "seconds": elapsed,
        }
        if lpips_fn is not None:
            row["lpips"] = float(lpips_fn(res.image, sample.hr))
        per_seed.append(row)
        images.append(res.image)

        if save_dir is not None:
            from PIL import Image
            save_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(res.image).save(
                save_dir / f"{sample.name}__s{strength:.2f}__seed{seed}.png"
            )

    keys = [k for k in per_seed[0] if k != "seed"]
    means = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}

    return {
        "dataset": None,              # filled in by run_sweep
        "image": sample.name,
        "strength": strength,
        "t0": int(strength * cfg.T) if strength > 0 else 0,
        "nfe": nfe,
        "back_projection": cfg.back_projection,
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "mean": means,
        # One sample carries no uncertainty; report null rather than a
        # fabricated zero so the analysis can tell the two apart.
        "cross_seed_std": cross_seed_std(images) if len(images) > 1 else None,
        "hr_shape": list(sample.hr.shape),
    }


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


def run_sweep(
    model,
    diffusion,
    cfg: SweepConfig,
    data_root: str,
    results_dir: str | Path,
    device: str = "cpu",
    checkpoint: str = "",
    shard_index: int = 0,
    num_shards: int = 1,
    use_lpips: bool = True,
    progress: bool = True,
) -> dict:
    """Run (this shard of) the sweep. Returns a summary of what it did."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    items = shard(work_list(cfg, data_root), shard_index, num_shards)
    prov = _provenance(cfg, checkpoint, device)

    lpips_fn = None
    if use_lpips:
        from src.metrics.perceptual import LPIPS
        lpips_fn = LPIPS(net="alex", device=device)

    # Dataset objects are reused across items so images are not reopened for
    # every strength.
    cache: dict[str, SRDataset] = {}

    done = skipped = failed = 0
    t_start = time.perf_counter()

    for n, (ds_name, idx, strength) in enumerate(items, 1):
        if ds_name not in cache:
            cache[ds_name] = SRDataset(data_root, ds_name, scale=cfg.scale,
                                       limit=cfg.limit)
        ds = cache[ds_name]
        sample = ds[idx]

        out = group_path(results_dir, ds_name, sample.name, strength,
                         cfg.back_projection)
        if out.exists():
            skipped += 1
            continue

        save_dir = None
        if cfg.save_images and done < cfg.save_images:
            save_dir = results_dir / "images" / ds_name

        try:
            row = evaluate_group(model, diffusion, sample, strength, cfg,
                                 device, lpips_fn, save_dir)
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {ds_name}/{sample.name} s={strength}: "
                  f"{type(e).__name__}: {e}")
            continue

        row["dataset"] = ds_name
        row["provenance"] = prov

        # Write to a temp file and rename. A pod evicted mid-write would
        # otherwise leave a truncated JSON that the resumed run treats as
        # done, silently dropping that group from the results.
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, indent=2))
        tmp.replace(out)
        done += 1

        if progress and (done % 10 == 0 or n == len(items)):
            rate = done / max(time.perf_counter() - t_start, 1e-9)
            left = (len(items) - n) / max(rate, 1e-9) / 3600
            print(f"  [{n}/{len(items)}] done {done} skipped {skipped} "
                  f"failed {failed}  {rate*3600:.0f} groups/h  eta {left:.1f}h")

    return {
        "shard": f"{shard_index}/{num_shards}",
        "assigned": len(items),
        "completed": done,
        "skipped_existing": skipped,
        "failed": failed,
        "seconds": time.perf_counter() - t_start,
        "provenance": prov,
    }


def load_results(results_dir: str | Path, back_projection: bool | None = None) -> list[dict]:
    """Every row written so far, for the analysis in T9."""
    results_dir = Path(results_dir)
    tags = ["plain", "bp"] if back_projection is None else \
           (["bp"] if back_projection else ["plain"])
    rows: list[dict] = []
    for tag in tags:
        for p in sorted((results_dir / tag).rglob("*.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                print(f"  [warn] unreadable, skipping: {p}")
    return rows
