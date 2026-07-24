#!/bin/bash
# Reviewer-response experiments. Runs after the calf suite. Priority order.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
echo "===== REVIEW-EXP START $(date) ====="

# 1) SSL encoder pretraining (needed before ssl downstream)
if [ ! -f ../checkpoints/ssl_encoder_thigh.pt ]; then
  echo ">>> ssl pretrain thigh"
  $PY pretrain_ssl.py --region thigh --epochs 150 > ../logs/ssl_pretrain.log 2>&1
  grep -E "saved|Error" ../logs/ssl_pretrain.log | tail -1
fi

# 2) Gate-only ablation (isolates the gate from the fat-fraction channel)
for n in 2 4 8 18; do for s in 0 1 2; do
  tag="method_gateonly_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_method.py --variant gateonly --n $n --seed $s --epochs 80 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep DONE ../logs/${tag}.log || echo "  (check ${tag})"
done; done

# 3) Multi-task normalisation study (BN collapse reproduction + GN/IN)
for norm in shared_bn groupnorm instancenorm; do for s in 0 1 2; do
  tag="mtlnorm_${norm}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_mtl_norm.py --norm $norm --seed $s --epochs 60 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep -E "ct=|check" ../logs/${tag}.log | tail -1 || echo "  (check ${tag})"
done; done

# 4) SSL downstream data-efficiency curve
for n in 2 4 8 18; do for s in 0 1 2; do
  tag="lowdata_ssl_n${n}_s${s}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY train_lowdata.py --region thigh --init ssl --n $n --seed $s --epochs 60 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep DONE ../logs/${tag}.log || echo "  (check ${tag})"
done; done

# 5) Repeated k-fold cross-validation (core claims, N=8 low-data regime)
for m in method:dixon4 method:dixonfuse; do for r in 0 1 2; do for f in 0 1 2 3 4; do
  tag="cv_thigh_${m/:/-}_n8_f${f}_r${r}"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY cross_val.py --model $m --region thigh --n 8 --fold $f --repeat $r --epochs 70 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1 || echo "  (check ${tag})"
done; done; done
for m in init:random init:imagenet init:ct; do for f in 0 1 2 3 4; do
  tag="cv_thigh_${m/:/-}_n8_f${f}_r0"
  [ -f "../results/${tag}.json" ] && { echo "skip $tag"; continue; }
  echo ">>> $tag"
  $PY cross_val.py --model $m --region thigh --n 8 --fold $f --repeat 0 --epochs 70 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1 || echo "  (check ${tag})"
done; done

echo "===== REVIEW-EXP DONE $(date) ====="
