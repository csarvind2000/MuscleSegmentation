#!/bin/bash
# Calf-muscle replication: benchmark + pretraining-source transfer + DixonFuse.
# Mirrors the thigh experiments to test whether the findings generalise.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
echo "===== CALF START $(date) ====="

# --- benchmark: 4 architectures, full data ---
for m in unet_r34 unetpp_r34 deeplabv3p_r34 segformer_b1; do
  [ -f "../results/calf_$m.json" ] && { echo "skip calf_$m"; continue; }
  echo ">>> calf benchmark $m"
  $PY train.py --task calf --model $m --epochs 80 --bs 8 --workers 6 > ../logs/calf_$m.log 2>&1
  grep "DONE" ../logs/calf_$m.log || echo "  (check calf_$m)"
done

# --- pretraining source: init x N x seed ---
for n in 2 4 8 18; do for init in random imagenet ct; do for s in 0 1 2; do
  tag="lowdata_calf_${init}_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_lowdata.py --region calf --n $n --init $init --seed $s --epochs 60 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep "DONE" ../logs/${tag}.log || echo "  (check ${tag})"
done; done; done

# --- DixonFuse ablation (n=18) ---
for v in wateronly dixon4 dixon5 dixonfuse; do for s in 0 1 2; do
  tag="method_calf_${v}_n18_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_method.py --region calf --variant $v --n 18 --seed $s --epochs 80 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep "DONE" ../logs/${tag}.log || echo "  (check ${tag})"
done; done

# --- DixonFuse data-efficiency (dixon4/dixon5/dixonfuse at n=2,4,8) ---
for n in 2 4 8; do for v in dixon4 dixon5 dixonfuse; do for s in 0 1 2; do
  tag="method_calf_${v}_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_method.py --region calf --variant $v --n $n --seed $s --epochs 80 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep "DONE" ../logs/${tag}.log || echo "  (check ${tag})"
done; done; done

echo "===== CALF DONE $(date) ====="
