#!/bin/bash
# Run all single-task baselines sequentially. Logs per job under logs/.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
MODELS="unet_r34 unetpp_r34 deeplabv3p_r34 segformer_b1"

echo "===== BENCHMARK START $(date) ====="
for m in $MODELS; do
  echo ">>> AATTCT $m"
  $PY train.py --task aattct --model $m --epochs 30 --bs 8 --workers 6 \
      > ../logs/aattct_$m.log 2>&1
  grep "DONE" ../logs/aattct_$m.log || echo "  (check log: aattct_$m)"
done
for m in $MODELS; do
  echo ">>> THIGH $m"
  $PY train.py --task thigh --model $m --epochs 80 --bs 8 --workers 6 \
      > ../logs/thigh_$m.log 2>&1
  grep "DONE" ../logs/thigh_$m.log || echo "  (check log: thigh_$m)"
done
echo "===== BENCHMARK DONE $(date) ====="
