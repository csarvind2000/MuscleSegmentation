#!/bin/bash
# Fair balanced-weight test of FF-as-supervision (Option A), N=2,4 only.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
# wait for Option B to finish (which itself waits for Option A)
until grep -q "OPTION B DONE" ../logs/run_option_b.log 2>/dev/null; do sleep 60; done
echo "Option B done -> ffsup w=10 test $(date)"
for n in 2 4; do
  for r in 0 1 2; do
    for f in 0 1 2 3 4; do
      $PY cross_val.py --model method:ffsup --ff_weight 10 --n $n --fold $f --repeat $r --epochs 70 \
          > ../logs/cv_ffsupw10_n${n}_f${f}_r${r}.log 2>&1
    done
  done
  echo "done ffsup w10 n=$n"
done
echo "FFSUP W10 DONE $(date)"
