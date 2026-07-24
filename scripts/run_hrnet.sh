#!/bin/bash
# R2: add a strong high-resolution backbone (U-Net + HRNet-W18) to the architecture
# comparison across all three full-data tasks, matching the existing baselines' epochs.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
M=unet_hrnet18
echo ">>> AATTCT $M"
$PY train.py --task aattct --model $M --epochs 30 --bs 8 --workers 6 > ../logs/aattct_$M.log 2>&1
grep -h "fg_mean\|DONE\|test_dice" ../logs/aattct_$M.log | tail -1 || echo "  (check aattct log)"
echo ">>> THIGH $M"
$PY train.py --task thigh --model $M --epochs 80 --bs 8 --workers 6 > ../logs/thigh_$M.log 2>&1
echo ">>> CALF $M"
$PY train.py --task calf --model $M --epochs 80 --bs 8 --workers 6 > ../logs/calf_$M.log 2>&1
echo "HRNet architecture runs done."
