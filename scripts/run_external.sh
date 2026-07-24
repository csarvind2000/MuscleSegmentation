#!/bin/bash
set -u
cd "$(dirname "$0")"; PY=../.venv/bin/python
echo "===== EXTERNAL START $(date) ====="
# Pretraining source (Water,Fat,FF) x N x seed
for init in random imagenet ct; do for n in 4 8 46; do for s in 0 1 2; do
  tag="ext_wff_${init}_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_external.py --init $init --channels wff --n $n --seed $s --epochs 70 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep "dice=" ../logs/${tag}.log | tail -1 || echo "  (check ${tag})"
done; done; done
# Fat-fraction effect: Water,Fat baseline (imagenet) x N x seed
for n in 4 8 46; do for s in 0 1 2; do
  tag="ext_wf_imagenet_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_external.py --init imagenet --channels wf --n $n --seed $s --epochs 70 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep "dice=" ../logs/${tag}.log | tail -1 || echo "  (check ${tag})"
done; done
echo "===== EXTERNAL DONE $(date) ====="
