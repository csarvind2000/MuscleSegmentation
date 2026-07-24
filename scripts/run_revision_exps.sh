#!/bin/bash
# Revision experiments (reviewer items 4 & 5), run sequentially on one GPU.
set -e
cd /media/ranjhaa-local/volume23/sarcopenia/seg_benchmark/scripts
PY=../.venv/bin/python
LOG=../results/revision_exps.log
echo "=== START $(date) ===" > $LOG

# --- 1. dixon5 (fat-fraction only, no gate) cross-validation: the recommended config ---
for N in 2 4 8; do
  for F in 0 1 2 3 4; do
    for R in 0 1 2; do
      OUT=../results/cv_thigh_method-dixon5_n${N}_f${F}_r${R}.json
      [ -f "$OUT" ] && continue
      $PY cross_val.py --model method:dixon5 --n $N --fold $F --repeat $R >> $LOG 2>&1
    done
  done
  echo "=== dixon5 N=$N done $(date) ===" >> $LOG
done

# --- 2. out-of-fold volumetrics over all 25 subjects ---
$PY make_volumetrics_oof.py >> $LOG 2>&1
echo "=== OOF volumetrics done $(date) ===" >> $LOG

# --- 3. gateonly (gate without fat-fraction) cross-validation: isolates the gate ---
for N in 2 4 8; do
  for F in 0 1 2 3 4; do
    for R in 0 1 2; do
      OUT=../results/cv_thigh_method-gateonly_n${N}_f${F}_r${R}.json
      [ -f "$OUT" ] && continue
      $PY cross_val.py --model method:gateonly --n $N --fold $F --repeat $R >> $LOG 2>&1
    done
  done
  echo "=== gateonly N=$N done $(date) ===" >> $LOG
done
echo "=== ALL DONE $(date) ===" >> $LOG
