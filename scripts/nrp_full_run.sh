#!/usr/bin/env bash
# Setup, gates, and training, in one pass. Runs INSIDE the pod.
#
# Idempotent at every stage, because this is the entrypoint of a Job that
# WILL be restarted: NRP is shared and preemptible. A restart re-runs this
# script from the top, finds the data already downloaded and the gates
# already passed, and resumes training from the last checkpoint on the PVC.
#
# Gates block training by construction: `set -e` means a failed gate exits
# non-zero and training never starts. That is the point of having them.

set -euo pipefail

WORK="${WORK:-/work}"
REPO_DIR="${REPO_DIR:-${WORK}/faithful-sr}"
DATA="${DATA:-${WORK}/data}"
VENV="${VENV:-${WORK}/venv}"
CKPT="${CKPT:-${WORK}/checkpoints}"
STEPS="${STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH="${PATCH:-128}"
SKIP_GATES="${SKIP_GATES:-0}"

cd "${REPO_DIR}"
mkdir -p "${CKPT}" "${DATA}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "0. environment"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)  host $(hostname)"
if [ ! -d "${VENV}" ]; then
  python -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scikit-image pillow pyyaml tqdm pandas scipy matplotlib lpips

# The 5090 is Blackwell (sm_120). A torch built for CUDA < 12.8 has no
# kernels for it and does NOT fail at import -- it fails at the first real
# matmul. One small matmul here turns a late, confusing death into an
# immediate, explicit one.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible.")
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
print(f"  GPU        {name}  (sm_{cap[0]}{cap[1]})")
print(f"  torch      {torch.__version__}  built for CUDA {torch.version.cuda}")
try:
    a = torch.randn(512, 512, device="cuda"); (a @ a).sum().item()
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(f"FATAL: this torch cannot run kernels on {name}.\n  {type(e).__name__}: {e}\n"
             f"  Use a CUDA >= 12.8 image, or pin the job to 4090 only.")
print("  matmul     ok")
PY

say "1. datasets"
if [ ! -d "${DATA}/DIV2K_train_HR" ]; then
  bash scripts/download_data.sh "${DATA}"
else
  echo "  already present ($(find "${DATA}/DIV2K_train_HR" -name '*.png' | wc -l | tr -d ' ') train images)"
fi

say "2. EDSR baseline weights"
if [ ! -f "${CKPT}/edsr_x4.pt" ]; then
  curl -L --fail --progress-bar -o "${CKPT}/edsr_x4.pt" \
    "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt"
else
  echo "  already present"
fi

say "3. point the config at the volume"
python - <<PY
import pathlib, yaml
p = pathlib.Path("configs/base.yaml")
c = yaml.safe_load(p.read_text()); c["data"]["root"] = "${DATA}"
p.write_text(yaml.safe_dump(c, sort_keys=False)); print("  data.root -> ${DATA}")
PY

if [ "${SKIP_GATES}" = "1" ]; then
  say "4-6. gates SKIPPED (SKIP_GATES=1)"
  echo "  Only do this on a restart you have already validated."
else
  # A marker file, so a restarted job does not re-run gates that passed. The
  # gates are cheap next to training, but re-downloading nothing and
  # re-proving the same facts wastes 20 minutes on every eviction.
  if [ -f "${CKPT}/.gates_passed" ]; then
    say "4-6. gates already passed on this volume, skipping"
  else
    say "4. gate 0: code correctness"
    python scripts/selftest.py
    python scripts/selftest_sampling.py

    say "5. gate 1: bicubic reproduces published numbers"
    python scripts/check_data.py "${DATA}"

    say "6. gate 2: EDSR reproduces published Set5"
    python scripts/check_edsr.py "${CKPT}/edsr_x4.pt" --data-root "${DATA}"

    say "7. gate 3: model overfits ten images"
    python scripts/train.py --overfit 10 --steps 3000 --ckpt-every 3000 \
      --out "${CKPT}/overfit" --keep-last 1

    touch "${CKPT}/.gates_passed"
    echo "  all gates passed; marker written"
  fi
fi

say "8. training"
RESUME=""
if [ -f "${CKPT}/latest.pt" ]; then
  RESUME="--resume ${CKPT}/latest.pt"
  echo "  resuming from ${CKPT}/latest.pt"
else
  echo "  starting fresh"
fi
echo "  steps ${STEPS} | batch ${BATCH_SIZE} | patch ${PATCH} | amp bf16"
echo "  watch the first 100 steps: it prints it/s and a live ETA."
echo

exec python scripts/train.py \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --patch "${PATCH}" \
  --out "${CKPT}" \
  --keep-last 5 \
  --amp bf16 \
  ${RESUME}
