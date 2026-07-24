#!/bin/bash
# Option A: FF as physics supervision. Same CV protocol as dixon4/dixon5 baselines.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
for n in 2 4 8; do
  for r in 0 1 2; do
    for f in 0 1 2 3 4; do
      $PY cross_val.py --model method:ffsup --n $n --fold $f --repeat $r --epochs 70 \
          > ../logs/cv_ffsup_n${n}_f${f}_r${r}.log 2>&1
    done
    echo "done ffsup n=$n repeat=$r"
  done
done
echo "FFSUP CV SWEEP DONE"
