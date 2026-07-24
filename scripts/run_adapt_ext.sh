#!/bin/bash
# R1.1 option 2: source-model fine-tuning vs ImageNet-from-scratch on the identical
# 4-channel/primary-label/10-muscle pipeline. Same fixed test set, N and seeds.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
for init in primary imagenet; do
  for n in 4 8 46; do
    for s in 0 1 2; do
      echo ">>> $init n=$n seed=$s"
      $PY adapt_external.py --init $init --n $n --seed $s --epochs 60 \
          > ../logs/adapt_${init}_n${n}_s${s}.log 2>&1
      grep -h "dice=" ../logs/adapt_${init}_n${n}_s${s}.log || echo "  (check log $init n$n s$s)"
    done
  done
done
echo "adaptation sweep done."
