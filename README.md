# faithful-sr

Measuring the perception–faithfulness trade-off in conditional diffusion
super-resolution as a continuous function of generative strength, and testing
whether a data-consistency constraint shifts that trade-off.

Every component traces to a published source; see `proposal/refs.bib`.

## Status

| Task | What | State |
|------|------|-------|
| T1 | Repo skeleton, config, environment | done |
| T2 | Degradation operator (`src/data/degradation.py`) | done, validated |
| T3 | Six metrics (`src/metrics/`) | done, validated |
| T4 | EDSR baseline, pretrained | next |
| T5 | SR3 conditional diffusion model | |
| T6 | DDIM sampler with `eta` | |
| T7 | Strength knob (the contribution) | |
| T8 | Resumable sweep runner | |
| T9 | Statistics and plots | |
| T10 | DDNM back-projection variant | |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch only needed from T4 onward
python scripts/selftest.py          # 21 correctness checks, no torch, no network
```

`scripts/selftest.py` should print `21/21 checks passed`. It is the gate for
everything downstream: if the degradation operator or the metrics are wrong,
every later number is wrong by more than the effects we are trying to detect.

## The two things that must never change

**One bicubic implementation.** `src/data/degradation.py` is a MATLAB
`imresize`-compatible bicubic with the antialiasing prefilter. Every published
SR number is measured against LR images made this way. PIL's `BICUBIC` and
OpenCV's `INTER_CUBIC` omit the prefilter when shrinking and disagree by
roughly 0.5–1.0 dB. The config records `degradation.impl: matlab_bicubic`; do
not swap it after the first training run.

**Generate LR ourselves.** The benchmark archives ship pre-made LR images
produced with a different resizer. Using them would create a train/test
mismatch. `data.use_shipped_lr: false` in the config.

## Metric protocol

| Metric | Preprocessing | Source |
|--------|---------------|--------|
| PSNR, SSIM | Y channel of YCbCr (BT.601 studio range), 4-px border crop | EDSR benchmark protocol, Lim et al. CVPRW 2017 |
| LPIPS | full RGB, no crop, AlexNet backbone, inputs in [-1, 1] | Zhang et al. CVPR 2018 |
| LR consistency | PSNR between bicubic↓4(pred) and the observation | DDNM data-consistency term, Wang et al. ICLR 2023 |
| Cross-seed std | per-pixel std over N seeds, ddof=1, then mean | Cohen et al. NeurIPS 2024 |
| Inference time | median of 20 after 3 warmup, with CUDA sync | — |

PSNR/SSIM use the Y channel and LPIPS uses RGB deliberately. Mixing the
conventions would make neither comparable to published numbers.

Only `lpips_score` imports torch, and it does so lazily, so the degradation
code, the distortion metrics and the self-test run in a plain numpy
environment.

## Data

```
data/
  DIV2K_train_HR/     800 images   training
  DIV2K_valid_HR/     100 images   checkpoint selection
  Set5/ Set14/                     verification against published PSNR
  BSD100/                          stochastic texture
  Urban100/                        regular man-made structure
```

HR images only. LR is generated at load time by `src.data.degradation.degrade`.

## Proposal

`proposal/` builds the two-page NeurIPS-format proposal.

```bash
cd proposal && latexmk -pdf main.tex
```

**Before submitting:** download
<https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip> and overwrite
`proposal/neurips_2025.sty` with the official file from the archive. The copy
in this repo is a layout-matched stand-in used only to verify the page limit —
`main.tex` uses only the documented interface, so no edits are needed when you
swap it.

The body occupies exactly two pages; references begin on page 3.
