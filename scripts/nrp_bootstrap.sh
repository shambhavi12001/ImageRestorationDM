#!/usr/bin/env bash
# One-time setup inside the interactive pod. Run from /work.
#
#   kubectl exec -it faithful-sr-shell -- bash
#   cd /work && bash /work/faithful-sr/scripts/nrp_bootstrap.sh
#
# (First time, clone the repo yourself: the script can't bootstrap itself.
#    cd /work && git clone <your repo url> faithful-sr )
#
# Everything this writes goes on the PVC, so the training Job and every
# sweep pod see it. Nothing is installed into the container image, which is
# thrown away when the pod dies -- that is why pip installs go to a venv on
# the volume rather than to site-packages.

set -euo pipefail

WORK="${WORK:-/work}"
REPO="${WORK}/faithful-sr"
DATA="${WORK}/data"
VENV="${WORK}/venv"

echo "==> 1. python environment on the volume"
if [ ! -d "${VENV}" ]; then
  # --system-site-packages keeps the image's CUDA-matched torch instead of
  # pulling a second multi-GB copy that may be built for the wrong CUDA.
  python -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet \
  numpy scikit-image pillow pyyaml tqdm pandas scipy matplotlib lpips
echo "    ok: $(python -c 'import torch; print("torch", torch.__version__)')"

echo "==> 2. datasets"
mkdir -p "${DATA}"
cd "${REPO}"
if [ ! -d "${DATA}/DIV2K_train_HR" ]; then
  bash scripts/download_data.sh "${DATA}"
else
  echo "    already present, skipping download"
fi

echo "==> 3. EDSR baseline weights"
mkdir -p "${WORK}/checkpoints"
if [ ! -f "${WORK}/checkpoints/edsr_x4.pt" ]; then
  curl -L --fail --progress-bar \
    -o "${WORK}/checkpoints/edsr_x4.pt" \
    "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt"
else
  echo "    already present"
fi

echo "==> 4. point the config at the volume"
python - <<PY
import pathlib, yaml
p = pathlib.Path("configs/base.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["data"]["root"] = "${DATA}"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"    data.root -> ${DATA}")
PY

echo "==> 5. gate 0: code correctness (no GPU, no data)"
python scripts/selftest.py
python scripts/selftest_sampling.py

echo "==> 6. gate 1: bicubic reproduces published numbers"
python scripts/check_data.py "${DATA}"

echo "==> 7. gate 2: EDSR reproduces published Set5"
python scripts/check_edsr.py "${WORK}/checkpoints/edsr_x4.pt" --data-root "${DATA}"

echo
echo "=============================================================="
echo " Gates 0-2 cleared. Next, the overfit gate (about an hour):"
echo
echo "   python scripts/train.py --overfit 10 --steps 3000 \\"
echo "       --out ${WORK}/checkpoints/overfit"
echo
echo " Then delete this pod and launch the real run as a Job:"
echo "   kubectl delete pod faithful-sr-shell"
echo "   kubectl apply -f k8s/job-train.yaml"
echo "=============================================================="
