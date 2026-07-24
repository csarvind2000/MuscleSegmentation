#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
echo "===== REVIEW2 START $(date) ====="
# CV at N=2 and N=4 for the DixonFuse claim (very-low-data regime)
for nn in 2 4; do for m in method:dixon4 method:dixonfuse; do for r in 0 1 2; do for f in 0 1 2 3 4; do
  tag="cv_thigh_${m/:/-}_n${nn}_f${f}_r${r}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY cross_val.py --model $m --region thigh --n $nn --fold $f --repeat $r --epochs 70 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1 || echo "  (check ${tag})"
done; done; done; done
# gate-only ablation (fixed)
for n in 2 4 8 18; do for s in 0 1 2; do
  tag="method_gateonly_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_method.py --variant gateonly --n $n --seed $s --epochs 80 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep DONE ../logs/${tag}.log || echo "  (check ${tag})"
done; done
echo "===== REVIEW2 DONE $(date) ====="
