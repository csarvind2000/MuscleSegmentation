#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
# wait for Option A (ffsup CV) to finish to avoid GPU contention
until grep -q "FFSUP CV SWEEP DONE" ../logs/run_ffsup_cv.log 2>/dev/null; do sleep 60; done
echo "Option A done -> Option B (FF-anchored zero-shot) $(date)"
$PY option_b_ffa.py --epochs 80 > ../logs/option_b.log 2>&1
grep -h "FF-anchored zero-shot" ../logs/option_b.log || echo "(check option_b.log)"
echo "OPTION B DONE $(date)"
