#!/usr/bin/env bash
# Entrypoint for the training Job on NRP. Auto-resumes after an eviction.
#
# NRP is shared, preemptible infrastructure and this run takes days, so
# assume the pod will be killed at least once. The Job restarts it, this
# script notices the checkpoint on the PVC, and training continues from the
# last saved step instead of from zero.
#
# It also fails FAST and LOUDLY on the two things that otherwise waste hours:
# a GPU architecture the installed torch has no kernels for, and a missing
# dataset.

set -euo pipefail

STEPS="${STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH="${PATCH:-128}"
CKPT_DIR="${CKPT_DIR:-/work/checkpoints}"
DATA_ROOT="${DATA_ROOT:-/work/data}"
KEEP_LAST="${KEEP_LAST:-5}"

echo "=============================================================="
echo " faithful-sr training"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)  host $(hostname)"
echo "=============================================================="

# --- 1. the GPU actually works with this torch build -------------------
# A 5090 is Blackwell (sm_120). A torch built for CUDA < 12.8 has no kernels
# for it and does not fail at import -- it fails at the first real matmul,
# which on a long run means discovering it well after the job looked healthy.
# One small matmul here settles it in a second.
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible. Check the GPU resource request.")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  GPU            {name}")
print(f"  capability     sm_{cap[0]}{cap[1]}")
print(f"  torch          {torch.__version__}")
print(f"  built for      {torch.version.cuda}")
print(f"  arch list      {torch.cuda.get_arch_list()}")

try:
    a = torch.randn(512, 512, device="cuda")
    (a @ a).sum().item()
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(
        f"FATAL: this torch build cannot run kernels on {name} (sm_{cap[0]}{cap[1]}).\n"
        f"  {type(e).__name__}: {e}\n"
        f"  Use an image with CUDA >= 12.8 for Blackwell (5090), or pin the "
        f"job's nodeAffinity to 4090 only."
    )
print("  matmul         ok")
PY

# --- 2. the data is there ----------------------------------------------
if [ ! -d "${DATA_ROOT}/DIV2K_train_HR" ]; then
  echo "FATAL: ${DATA_ROOT}/DIV2K_train_HR not found."
  echo "Run scripts/nrp_bootstrap.sh in the interactive pod first."
  exit 1
fi
N_TRAIN=$(find "${DATA_ROOT}/DIV2K_train_HR" -name '*.png' | wc -l)
echo "  train images   ${N_TRAIN}  (expected 800)"

# --- 3. resume if a checkpoint exists ----------------------------------
mkdir -p "${CKPT_DIR}"
RESUME_ARG=""
if [ -f "${CKPT_DIR}/latest.pt" ]; then
  RESUME_ARG="--resume ${CKPT_DIR}/latest.pt"
  echo "  resuming from  ${CKPT_DIR}/latest.pt"
else
  echo "  starting fresh (no checkpoint on the volume)"
fi

echo "--------------------------------------------------------------"

exec python scripts/train.py \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --patch "${PATCH}" \
  --out "${CKPT_DIR}" \
  --keep-last "${KEEP_LAST}" \
  ${RESUME_ARG}
