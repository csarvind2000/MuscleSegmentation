# Label-Efficient Individual Muscle Segmentation from Dixon MRI

A multi-dataset benchmark of **pretraining** and **fat-fraction guidance** for segmenting
individual thigh and calf muscles (and abdominal adipose compartments) under label
scarcity. This repository holds the code, experiment scripts, aggregated results, and
figure-generation pipeline accompanying the study.

> **Status:** manuscript under peer review at *Journal of Imaging Informatics in Medicine*.
> The manuscript sources and journal submission files are **not** included in this
> repository. Trained model weights will be released upon acceptance.

---

## What the study asks

When labelled subjects are few, how should you segment individual muscles on Dixon MRI?
Four design choices are compared like-for-like with repeated, subject-level
cross-validation:

1. **Network architecture** — U-Net, U-Net++, DeepLabV3+, SegFormer.
2. **Encoder pretraining** — random vs. ImageNet vs. CT-dataset pretraining.
3. **Fat-fraction guidance** — adding a Dixon-derived fat-fraction input channel
   (and a contrast-reweighting gate).
4. **Cross-modality sharing** — one encoder shared between the CT and MRI tasks.

## Datasets (all public, not redistributed here)

| Dataset | Modality | Subjects | Role |
|---|---|---|---|
| AATTCT-IDS | Abdominal CT | 300 | Adipose (SAT/VAT) labels; CT-side pretraining source |
| Lower-limb Dixon MRI | MRI (Dixon) | 25 | 13 thigh + 9 calf muscles (main target) |
| Multi-ethnic thigh MRI (Han-Chinese cohort) | MRI (Dixon) | 66 | External test set |

Download each dataset from its original source and point `DATA_ROOT` in
[scripts/config.py](scripts/config.py) at your local copy.

## Repository layout

```
scripts/      all code: data prep, training, cross-validation, analysis, figures
results/      aggregated per-experiment metrics (JSON) — inputs to the figures
figures/      generated figure images
.gitignore
README.md
```

Not tracked (see `.gitignore`): `checkpoints/` (model weights), `data_cache/`
(preprocessed arrays), `logs/`, and the local `.venv/`.

### Key scripts

- `config.py` — central paths and dataset constants
- `datasets.py`, `prep_*.py` — dataset loading and preprocessing
- `models.py`, `models_method.py` — model definitions (via
  [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch))
- `train*.py` — training entry points (single-task, low-data, multitask, external)
- `cross_val.py` — repeated 5-fold, subject-level cross-validation
- `pretrain_ssl.py` — self-supervised pretraining
- `aggregate*.py`, `analyze_*.py` — collate results into `results/`
- `make_*.py` — figure generation
- `run_*.sh` — batch drivers for each experiment phase

## Setup

Python 3.10. There is no pinned requirements file yet; the core stack is:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision segmentation-models-pytorch \
            numpy scipy scikit-learn pandas matplotlib nibabel pydicom tqdm
```

## Running

Edit paths in `scripts/config.py`, then, from `scripts/`:

```bash
# Single-task baselines across all four architectures (CT then thigh MRI)
./run_all.sh

# Repeated subject-level cross-validation (pretraining / fat-fraction factors)
python cross_val.py            # see --help for task, label budget, and seeds

# External-cohort evaluation
./run_external.sh
```

Batch drivers (`run_phase2.sh`, `run_phase3.sh`, `run_review.sh`, …) reproduce the
individual experiment phases; each writes per-job logs to `logs/` and metrics to
`results/`. Regenerate figures with `python make_figures.py` (and the other `make_*.py`).

## Citation

Citation details will be added once the paper is published. Until then, please contact
the authors before reusing results.
