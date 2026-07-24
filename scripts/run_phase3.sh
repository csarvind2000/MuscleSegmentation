#!/bin/bash
# Phase 3: DixonFuse method — ablation (full data) + method-vs-baseline data-efficiency.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "===== PHASE3 START $(date) ====="

run () {  # variant n seed
  tag="method_$1_n$2_s$3"
  if [ -f "../results/${tag}.json" ]; then echo "skip $tag"; return; fi
  echo ">>> $tag"
  $PY train_method.py --variant $1 --n $2 --seed $3 --epochs 80 --bs 8 --workers 6 \
      > ../logs/${tag}.log 2>&1
  grep "DONE" ../logs/${tag}.log || echo "  (check log: ${tag})"
}

# --- ablation on full training data (n=18), 3 seeds ---
for v in wateronly dixon4 dixon5 dixonfuse; do
  for s in 0 1 2; do run $v 18 $s; done
done

# --- data-efficiency: full method vs Dixon baseline across N subjects ---
for n in 2 4 8; do
  for v in dixon4 dixonfuse; do
    for s in 0 1 2; do run $v $n $s; done
  done
done

echo "===== PHASE3 DONE $(date) ====="
