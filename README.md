# faithful-sr

Measuring the perception–faithfulness trade-off in conditional diffusion
super-resolution as a continuous function of **generative strength**, and
testing whether an explicit data-consistency constraint shifts that trade-off.

Every component traces to a published source. See [`proposal/refs.bib`](proposal/refs.bib).

---

## Status

Code is complete through T7 and T10. **Nothing is trained yet** — the training
run is the critical path.

| | Task | State | Verified by |
|---|---|---|---|
| T1 | Repo skeleton, frozen config | done | — |
| T2 | Degradation operator (MATLAB-compatible bicubic) | done | `selftest.py` |
| T3 | Six metrics | done | `selftest.py` |
| T4 | Dataset loaders + EDSR baseline | done | gates blocked on data |
| T5 | SR3 diffusion model + training script | done | overfit smoke test |
| T6 | DDIM sampler with `eta` | done | `selftest_sampling.py` |
| T7 | **Strength knob — the contribution** | done | `selftest_sampling.py` |
| T8 | Resumable sweep runner | **next** | |
| T9 | Statistics and plots | todo | |
| T10 | DDNM back-projection | done | `selftest_sampling.py` |

**36/36 correctness checks passing.**

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/selftest.py            # 21 checks, numpy only, no network
python scripts/selftest_sampling.py   # 15 checks, needs torch, CPU is fine
```

Both must be green before anything else. If the degradation operator or the
metrics are wrong, every later number is wrong by more than the effects this
study is trying to detect.

## On the GPU box

```bash
bash scripts/download_data.sh                        # ~5 GB

python scripts/check_data.py                         # Gate 1
python scripts/check_edsr.py checkpoints/edsr_x4.pt  # Gate 2
python scripts/train.py --overfit 10 --steps 3000    # Gate 3
python scripts/train.py                              # the real run
```

EDSR weights: `https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt`

`train.py` prints iterations/sec and a live ETA every 100 steps, so you know
within a minute of starting whether the schedule survives. It resumes with
`--resume checkpoints/latest.pt`.

---

## Validation gates

They run in order. Each exists so that a later surprising result is a fact
about the model rather than a bug in the harness.

| Gate | Target | Command | State |
|---|---|---|---|
| 0 | 36/36 checks | `selftest.py`, `selftest_sampling.py` | **passing** |
| 1 | Set5 bicubic 28.42 dB; DIV2K val 26–28 dB | `check_data.py` | blocked on data |
| 2 | Set5 EDSR 32.46 dB | `check_edsr.py` | blocked on data + weights |
| 3 | model overfits 10 images | `train.py --overfit 10` | not started |
| 4 | `s=1` matches full-chain sampling | after training | not started |

---

## The two rules that must never change

**One bicubic implementation.** `src/data/degradation.py` is a MATLAB
`imresize`-compatible bicubic *with* the antialiasing prefilter. Every
published SR number is measured against LR images made this way. PIL's
`BICUBIC` and OpenCV's `INTER_CUBIC` omit the prefilter when shrinking and
disagree by 0.5–1.0 dB. The config records `degradation.impl: matlab_bicubic`;
do not swap it after the first training run.

`src/data/torch_degradation.py` is the same operator as explicit matrices,
built from the same weights, so the sampler's `H` cannot drift from the data
pipeline's. It agrees to 5e-5.

**Generate LR ourselves.** The benchmark archives ship pre-made LR images made
with a different resizer; using them puts a train/test mismatch under every
result. `data.use_shipped_lr: false`, and the loader filters them out by path.

---

## Layout

```
configs/base.yaml              the protocol, frozen
src/data/degradation.py        MATLAB-compatible bicubic (numpy)
src/data/torch_degradation.py  the same operator as matrices, + pseudo-inverse
src/data/datasets.py           HR on disk, LR generated at load time
src/metrics/                   psnr, ssim, lr_consistency, lpips, cross-seed std
src/models/unet.py             conditional eps-prediction UNet
src/models/diffusion.py        schedule, q_sample, training objective
src/models/edsr.py             pretrained baseline wrapper
src/sampling/ddim.py           DDIM with eta + optional back-projection
src/sampling/strength.py       the strength knob
scripts/                       gates, training, self-tests
proposal/                      two-page NeurIPS-format proposal
```

---

## The strength knob

`src/sampling/strength.py` is the contribution. No published repository
provides this for super-resolution: SDEdit introduced partial forward
diffusion for *editing*, SR3 samples at one fixed strength.

```
t0 = floor(s * T)
x  = sqrt(ab[t0]) * up + sqrt(1 - ab[t0]) * eps
x0 = DDIM(x, t0 -> 0, spacing fixed at T / n_steps_at_full)
```

Step spacing is held **fixed** rather than dividing a constant number of steps
across whatever range `t0` covers. So a lower strength genuinely performs
fewer network evaluations, and compute becomes a reported third axis rather
than overhead.

Verified invariants:

- `s = 0` returns the bicubic upsample **bit-exactly**, with zero network calls
- NFE is linear in `s` — 10 / 30 / 50 at `s` = 0.2 / 0.6 / 1.0
- a fixed seed reproduces a sample bit-for-bit (cross-seed variance is a
  measurement here, so the seed is an explicit argument and is recorded)

## Back-projection

DDNM's correction is `x0 <- x0 - H†(H x0 - y)`. **`H†` is the true
Moore-Penrose pseudo-inverse, not bicubic upsampling.**

Bicubic-up is *not* a right inverse of bicubic-down (`H H_up != I`): with it,
one application cut the measurement residual by only 57% and was not a
projection at all. That would quietly make "with data consistency" mean "with
partial data consistency" and render H3 untestable. The operator is separable,
so the real pseudo-inverse factors per axis and is cheap.

Measured: `||Hx - y||` goes from 1.67e-02 to **3.95e-08** in one application,
while null-space detail survives — it constrains what the measurement sees
without erasing what the model invented.

---

## Metric protocol

| Metric | Preprocessing | Source |
|---|---|---|
| PSNR, SSIM | Y of YCbCr (BT.601 studio range), 4-px border crop | EDSR protocol, Lim et al. CVPRW 2017 |
| LPIPS | full RGB, no crop, AlexNet, inputs in [-1,1] | Zhang et al. CVPR 2018 |
| LR consistency | PSNR between bicubic↓4(pred) and the observation | DDNM, Wang et al. ICLR 2023 |
| Cross-seed std | per-pixel std over N seeds (ddof=1), then mean | Cohen et al. NeurIPS 2024 |
| Inference time | median of 20 after 3 warmup, CUDA-synced | — |

PSNR/SSIM use Y and LPIPS uses RGB **deliberately**. Mixing the conventions
would make neither comparable to published numbers.

Only `lpips_score` imports torch, and lazily — so the degradation code, the
distortion metrics and `selftest.py` run in a plain numpy environment.

---

## Data

```
data/
  DIV2K_train_HR/    800   training
  DIV2K_valid_HR/    100   checkpoint selection
  Set5/ Set14/             reproduce published numbers
  BSD100/            100   stochastic texture
  Urban100/          100   regular man-made structure
```

HR only. BSD100 and Urban100 are the contrast pair the result lives in:
invented stochastic texture is perceptually cheap, invented regular structure
is not.

---

## Proposal

```bash
cd proposal && latexmk -pdf main.tex
```

Body is exactly two pages; references begin on page 3.

**Before submitting:** download
<https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip> and overwrite
`proposal/neurips_2025.sty` with the official file. The copy here is a
layout-matched stand-in used to verify the page limit; `main.tex` uses only
the documented interface, so no edits are needed when you swap it.
