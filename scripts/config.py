"""Central configuration for the body-composition segmentation benchmark."""
import os

DATA_ROOT = "/media/ranjhaa-local/volume23/sarcopenia/datasets/opensource_dataset"
PROJ = "/media/ranjhaa-local/volume23/sarcopenia/seg_benchmark"

AATTCT_ROOT = os.path.join(DATA_ROOT, "AATTCT-IDS")
THIGH_ROOT = os.path.join(DATA_ROOT, "Thigh", "MRI_data")

CACHE = os.path.join(PROJ, "data_cache")
RESULTS = os.path.join(PROJ, "results")
CKPT = os.path.join(PROJ, "checkpoints")
FIG = os.path.join(PROJ, "figures")
LOGS = os.path.join(PROJ, "logs")
for d in (CACHE, RESULTS, CKPT, FIG, LOGS):
    os.makedirs(d, exist_ok=True)

SEED = 42

# --- AATTCT (abdominal CT adipose) ---
# 3-class: 0 background, 1 subcutaneous (SAT), 2 visceral (VAT)
AATTCT_CLASSES = ["background", "SAT", "VAT"]
AATTCT_NUM_CLASSES = 3
IMG_SIZE = 512

# --- Thigh MRI muscle ---
# filled after auditing label ranges in prep_thigh.py
