#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
echo "===== PHASE3B (FF isolation) START $(date) ====="
run () {
  tag="method_$1_n$2_s$3"
  if [ -f "../results/${tag}.json" ]; then echo "skip $tag"; return; fi
  echo ">>> $tag"
  $PY train_method.py --variant $1 --n $2 --seed $3 --epochs 80 --bs 8 --workers 6 > ../logs/${tag}.log 2>&1
  grep "DONE" ../logs/${tag}.log || echo "  (check log: ${tag})"
}
for n in 2 4 8; do for s in 0 1 2; do run dixon5 $n $s; done; done
echo "===== PHASE3B DONE $(date) ====="
