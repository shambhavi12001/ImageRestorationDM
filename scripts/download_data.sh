#!/usr/bin/env bash
# Fetch the HR datasets. Run from the repo root:  bash scripts/download_data.sh
#
# Only HR images are used. The LR images bundled inside some of these archives
# were produced with a different resizer and are deliberately ignored by
# src/data/datasets.py -- we generate LR ourselves so train and test share one
# degradation operator.
#
# Total download is roughly 5 GB, dominated by DIV2K train.

set -euo pipefail

ROOT="${1:-data}"
mkdir -p "$ROOT"
cd "$ROOT"

fetch() {  # fetch <url> <zipname> <expected-dir>
  local url="$1" zip="$2" dir="$3"
  if [ -d "$dir" ]; then
    echo "  [skip] $dir already present"
    return
  fi
  echo "  [get ] $dir"
  curl -L --fail --progress-bar -o "$zip" "$url"
  unzip -q -o "$zip"
  rm -f "$zip"
}

echo "DIV2K (Agustsson & Timofte, CVPRW 2017) -- training and validation"
fetch "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" \
      "DIV2K_train_HR.zip" "DIV2K_train_HR"
fetch "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" \
      "DIV2K_valid_HR.zip" "DIV2K_valid_HR"

# Set5, Set14, BSD100 and Urban100 have no single canonical host. The mirror
# below is the one the EDSR and RCAN repositories point at; if it has moved,
# any copy works as long as it is the HR originals.
echo
echo "Benchmarks (Set5, Set14, BSD100, Urban100)"
fetch "https://cv.snu.ac.kr/research/EDSR/benchmark.tar" \
      "benchmark.tar" "benchmark"

# Flatten the benchmark layout into the directory names datasets.py expects.
if [ -d benchmark ]; then
  for name in Set5 Set14 B100 Urban100; do
    src="benchmark/$name/HR"
    [ -d "$src" ] || continue
    out="$name"; [ "$name" = "B100" ] && out="BSD100"
    mkdir -p "$out"
    cp -n "$src"/*.png "$out"/ 2>/dev/null || true
  done
fi

echo
echo "Done. Verify with:"
echo "  python scripts/check_data.py"
