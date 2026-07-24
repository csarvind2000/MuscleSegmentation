#!/bin/bash
# Complete Table 5: 2.5D (k=0/1/2) for DeepLabV3+, SegFormer, U-Net+HRNet-W18,
# same harness/split as the existing U-Net / U-Net++ rows.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
for m in deeplabv3p_r34 segformer_b1 unet_hrnet18; do
  for k in 0 1 2; do
    echo ">>> $m k=$k"
    $PY train_25d.py --model $m --k $k --epochs 80 > ../logs/thigh_25d_k${k}_${m}.log 2>&1
    grep -h "fgDice=" ../logs/thigh_25d_k${k}_${m}.log || echo "  (check log $m k$k)"
  done
done
echo "2.5D extra sweep done."
