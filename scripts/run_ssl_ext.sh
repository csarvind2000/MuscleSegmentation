#!/bin/bash
# R1.1: few-shot adaptation with SSL-on-target init, matching the existing
# imagenet/ct/random protocol (same fixed test set, same N and seeds).
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
for n in 4 8 46; do
  for s in 0 1 2; do
    echo ">>> ssl wff n=$n seed=$s"
    $PY train_external.py --init ssl --channels wff --n $n --seed $s \
        > ../logs/ext_ssl_wff_n${n}_s${s}.log 2>&1
    grep -h "dice=" ../logs/ext_ssl_wff_n${n}_s${s}.log || echo "  (check log n$n s$s)"
  done
done
echo "SSL few-shot sweep done."
