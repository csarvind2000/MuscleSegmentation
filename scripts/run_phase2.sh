#!/bin/bash
# Phase 2: cross-modality transfer / data-efficiency study + multi-task model.
# Requires checkpoints/aattct_unet_r34.pt (CT-pretrained encoder) to exist.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "===== PHASE2 START $(date) ====="

# --- data-efficiency curve: N subjects x init x seed ---
for n in 2 4 8 18; do
  for init in random imagenet ct; do
    for s in 0 1 2; do
      tag="lowdata_${init}_n${n}_s${s}"
      if [ -f "../results/${tag}.json" ]; then echo "skip $tag"; continue; fi
      echo ">>> $tag"
      $PY train_lowdata.py --n $n --init $init --seed $s --epochs 60 --bs 8 --workers 6 \
          > ../logs/${tag}.log 2>&1
      grep "DONE" ../logs/${tag}.log || echo "  (check log: ${tag})"
    done
  done
done

# --- multi-task shared encoder ---
echo ">>> multitask"
$PY train_multitask.py --epochs 60 --bs 8 --workers 6 > ../logs/mtl.log 2>&1
grep "DONE" ../logs/mtl.log || echo "  (check log: mtl)"

echo "===== PHASE2 DONE $(date) ====="
