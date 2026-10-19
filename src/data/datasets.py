"""Dataset loading. HR images on disk, LR generated at load time.

The one rule this file enforces: LR images are ALWAYS produced by
`src.data.degradation.degrade`, never read from disk. The benchmark archives
ship pre-made LR images produced with a different resizer, and mixing those
with our own would put a train/test mismatch underneath every result.

Datasets and their role in the study:

  DIV2K train (800)   training                 Agustsson & Timofte, CVPRW 2017
  DIV2K val   (100)   checkpoint selection     same
  Set5        (5)     reproduce published PSNR Bevilacqua et al., BMVC 2012
  Set14       (14)    reproduce published PSNR Zeyde et al., Curves&Surfaces 2010
  BSD100      (100)   stochastic texture       Martin et al., ICCV 2001
  Urban100    (100)   regular structure        Huang et al., CVPR 2015

BSD100 and Urban100 are the contrast pair the result lives in; the first two
benchmark sets exist only to prove the pipeline is correct before we make any
claim with it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from src.data.degradation import degrade

__all__ = ["Sample", "SRDataset", "load_image", "random_crop_pair", "DATASET_DIRS"]


# Directory name on disk for each logical dataset name.
DATASET_DIRS = {
    "div2k_train": "DIV2K_train_HR",
    "div2k_val": "DIV2K_valid_HR",
    "set5": "Set5",
    "set14": "Set14",
    "bsd100": "BSD100",
    "urban100": "Urban100",
}

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------


@dataclass
class Sample:
    """One evaluation item.

    hr : (H, W, 3) uint8, already modcropped to a multiple of `scale`
    lr : (H/s, W/s, 3) uint8, our own degradation
    up : (H, W, 3) uint8, bicubic upsample of lr -- the conditioning input
    """

    name: str
    hr: np.ndarray
    lr: np.ndarray
    up: np.ndarray
    scale: int = 4
    path: str = ""

    def __post_init__(self) -> None:
        if self.hr.shape != self.up.shape:
            raise ValueError(f"{self.name}: hr {self.hr.shape} != up {self.up.shape}")
        h, w = self.hr.shape[:2]
        if (h % self.scale) or (w % self.scale):
            raise ValueError(f"{self.name}: hr {h}x{w} not divisible by {self.scale}")


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as (H, W, 3) uint8 RGB.

    Grayscale is expanded to three channels so the shape contract downstream
    holds; a palette or RGBA image is converted, dropping alpha.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


# --------------------------------------------------------------------------


class SRDataset:
    """HR images from one directory, degraded on access.

    Not a torch Dataset: it has no torch dependency, so metric and sweep code
    can use it in a plain numpy environment. The training loop wraps it.
    """

    def __init__(self, root: str | Path, name: str, scale: int = 4, limit: int | None = None):
        self.name = name
        self.scale = scale
        self.root = Path(root) / DATASET_DIRS.get(name, name)

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.name}: no directory at {self.root}. "
                f"Run scripts/download_data.sh, or point data.root at wherever "
                f"you keep the datasets."
            )

        self.paths: list[Path] = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _EXTS
        )
        # Guard against the pre-made LR images that ship inside some archives.
        self.paths = [p for p in self.paths if not _looks_like_shipped_lr(p)]

        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"{self.name}: no images found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Sample:
        path = self.paths[i]
        hr_raw = load_image(path)
        hr, lr, up = degrade(hr_raw, self.scale)
        return Sample(name=path.stem, hr=hr, lr=lr, up=up, scale=self.scale, path=str(path))

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def manifest(self) -> dict:
        """Content fingerprint, recorded alongside results.

        Hashes the file list and each file's size. If someone swaps a dataset
        or a download is truncated, the hash in an old results file will not
        match and the discrepancy surfaces immediately instead of showing up
        as an unexplained shift in the numbers.
        """
        h = hashlib.sha256()
        for p in self.paths:
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
        return {
            "dataset": self.name,
            "root": str(self.root),
            "n_images": len(self.paths),
            "scale": self.scale,
            "fingerprint": h.hexdigest()[:16],
        }


def _looks_like_shipped_lr(path: Path) -> bool:
    """True for the pre-generated LR images bundled with some benchmarks.

    Those were made with a different imresize. We generate our own, so these
    must never be picked up as if they were HR images.
    """
    parts = {s.lower() for s in path.parts}
    markers = {"lr_bicubic", "lrbicubic", "lr", "x2", "x3", "x4", "x8"}
    if parts & markers:
        return True
    stem = path.stem.lower()
    return stem.endswith(("x2", "x3", "x4", "x8", "_lr"))


# --------------------------------------------------------------------------
# training patches
# --------------------------------------------------------------------------


def random_crop_pair(
    hr: np.ndarray,
    scale: int,
    patch: int,
    rng: np.random.Generator,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One (hr_patch, up_patch) training pair.

    Crops in HR space first, then degrades the crop. Cropping after degrading
    would let the bicubic kernel see context the model never gets at test
    time, which inflates training performance.

    `patch` is the HR patch size and must be a multiple of `scale`.
    Augmentation is the standard flip/rotate set used by EDSR
    (Lim et al., CVPRW 2017): it is label-preserving for this task because
    bicubic degradation commutes with the dihedral group.
    """
    if patch % scale:
        raise ValueError(f"patch {patch} must be a multiple of scale {scale}")
    h, w = hr.shape[:2]
    if h < patch or w < patch:
        raise ValueError(f"image {h}x{w} smaller than patch {patch}")

    y = int(rng.integers(0, h - patch + 1))
    x = int(rng.integers(0, w - patch + 1))
    crop = hr[y : y + patch, x : x + patch]

    if augment:
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        if rng.random() < 0.5:
            crop = crop[::-1, :]
        if rng.random() < 0.5:
            crop = crop.transpose(1, 0, 2)
        crop = np.ascontiguousarray(crop)

    hr_c, _lr, up = degrade(crop, scale)
    return hr_c, up
