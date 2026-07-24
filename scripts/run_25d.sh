#!/bin/bash
# R1.2/R1.4: 2.5D vs 2D thigh baseline on the identical harness/split/eval.
# k=0 is the 2D control; k=1 (3-slice) and k=2 (5-slice) add inter-slice context.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
for m in unet_r34 unetpp_r34; do
  for k in 0 1 2; do
    echo ">>> $m k=$k"
    $PY train_25d.py --model $m --k $k --epochs 80 > ../logs/thigh_25d_k${k}_${m}.log 2>&1
    grep -h "fgDice=" ../logs/thigh_25d_k${k}_${m}.log || echo "  (check log $m k$k)"
  done
done
echo "2.5D sweep done."
