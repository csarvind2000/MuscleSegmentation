#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
# wait for the 2.5D-extra sweep to finish
until grep -q "2.5D extra sweep done" ../logs/run_25d_extra.log 2>/dev/null; do sleep 30; done
echo "2.5D-extra done -> multitask seeds $(date)"
for s in 0 1 2; do
  echo ">>> mtl seed $s"
  $PY train_multitask.py --seed $s --epochs 60 > ../logs/mtl_s${s}.log 2>&1
  grep -h "DONE" ../logs/mtl_s${s}.log || echo "  (check mtl s$s)"
done
echo "MTL SEEDS DONE $(date)"
