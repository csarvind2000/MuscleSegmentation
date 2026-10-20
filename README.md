# Annotation-Efficient Individual Muscle Segmentation from Dixon MRI

[![Paper](https://img.shields.io/badge/Paper-Springer%20Nature-blue)](https://link.springer.com/article/10.1007/s10278-026-02364-6)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs10278--026--02364--6-blue)](https://doi.org/10.1007/s10278-026-02364-6)
[![Published](https://img.shields.io/badge/Published-2026-green)](https://link.springer.com/article/10.1007/s10278-026-02364-6)

**A Multi-Dataset Benchmark of Pretraining and Fat-Fraction Guidance**

This repository contains the research code, experimental scripts, aggregated results, and figure-generation pipeline accompanying our published study.

**Publication:** Journal of Imaging Informatics in Medicine (Springer Nature)

**Published:** 19 October 2026

**Authors:** Arvind Channarayapatna Srinivasa and Shamshekhar S. Patil

**Published Article:**  
https://link.springer.com/article/10.1007/s10278-026-02364-6

**DOI:**  
https://doi.org/10.1007/s10278-026-02364-6

---

## Overview

Accurate segmentation of individual thigh and calf muscles from Dixon MRI is important for quantitative body composition analysis, sarcopenia assessment, and muscle health research.

However, manually annotating individual muscles across MRI volumes is time-consuming, and annotated medical imaging datasets are often limited.

This study investigates **annotation-efficient deep learning for multi-class muscle segmentation**.

We explore three key questions:

1. How can we achieve accurate muscle segmentation with only a few annotated MRI subjects?
2. Which pretraining and imaging-informed strategies improve performance under limited annotations?
3. How many annotated subjects from an external MRI cohort are needed to improve model generalisation?

We systematically evaluated network architectures, encoder pretraining, Dixon-derived fat-fraction guidance, cross-modality learning, and external cohort adaptation.

The experiments provide practical recommendations for developing reliable segmentation models when labelled medical imaging data are scarce.

---

## Research Questions

The study investigates five major aspects of annotation-efficient segmentation.

### 1. Network Architecture

How does network architecture affect individual-muscle segmentation performance?

Five 2D architectures were evaluated:

- U-Net
- U-Net++
- DeepLabV3+
- SegFormer (MiT-B1)
- U-Net with HRNet-W18 encoder

A 2.5D model was also evaluated.

### 2. Encoder Pretraining

Can pretrained encoders improve segmentation accuracy when only a few annotated subjects are available?

We compared:

- Random initialisation
- ImageNet pretraining
- Body-composition CT pretraining
- Self-supervised pretraining

### 3. Fat-Fraction Guidance

Can Dixon-derived fat-fraction information improve segmentation under severe annotation scarcity?

We evaluated:

- Water-only input
- Four-channel Dixon MRI
- Dixon MRI with fat-fraction guidance
- Contrast attention gating
- Fat-fraction guidance with contrast gating

### 4. Cross-Modality Learning

Can CT and MRI segmentation tasks share a common encoder while maintaining accuracy?

We investigated a multi-task architecture with:

- Shared encoder
- CT-specific segmentation decoder
- MRI-specific segmentation decoder

### 5. External Generalisation

How well does a segmentation model trained on one MRI cohort perform on an independent cohort?

We evaluated:

- Zero-shot external testing
- Label-free domain adaptation
- Few-shot fine-tuning
- Cross-cohort segmentation performance

---

## Datasets

All three datasets used in this study are publicly available research datasets.

The datasets are not redistributed through this repository.

| Dataset | Modality | Subjects | Purpose |
|---|---|---:|---|
| Lower-limb Dixon MRI | Dixon MRI | 25 | Individual thigh and calf muscle segmentation |
| HuashanMyo Han-Chinese cohort | Thigh MRI | 66 | External testing and domain adaptation |
| AATTCT-IDS | Abdominal CT | 300 | Adipose tissue segmentation and cross-modality learning |

### Lower-Limb Dixon MRI

The primary dataset contains:

- 25 subjects
- 13 individually annotated thigh muscles
- 9 individually annotated calf muscles
- Water, fat, in-phase, and opposed-phase MRI images

This dataset was used to investigate annotation-efficient individual-muscle segmentation.

### External Thigh MRI Dataset

An independent cohort containing 66 subjects was used to evaluate:

- Cross-cohort generalisation
- Zero-shot segmentation
- Domain adaptation
- Few-shot fine-tuning

### Abdominal CT Dataset

The AATTCT-IDS dataset contains 300 subjects with annotations for:

- Subcutaneous adipose tissue (SAT)
- Visceral adipose tissue (VAT)

This dataset was used for adipose segmentation and cross-modality learning experiments.

Dataset references and access information are available in the published manuscript.

---

## Experimental Design

Our experiments investigated the effect of different model configurations under limited annotation availability.

### Architecture Benchmarking

Five segmentation architectures were trained and evaluated under comparable experimental settings.

The objective was to determine whether more complex architectures consistently improve individual-muscle segmentation accuracy.

### Pretraining Experiments

We investigated the impact of encoder initialisation under different annotation budgets.

The primary experiments compared training with:

- 2 annotated subjects
- 4 annotated subjects
- 8 annotated subjects
- 18 annotated subjects

Repeated subject-level cross-validation was used to assess the robustness of the principal comparisons.

### Fat-Fraction Guidance

A fat-fraction proxy was calculated from the Dixon water and fat images:

FF = F / (F + W + ε)

The resulting fat-fraction map was incorporated as an additional model input.

This provided imaging-derived information to support muscle boundary identification under limited annotation availability.

The proxy should not be interpreted as a fully corrected proton-density fat-fraction measurement.

### Cross-Modality Learning

A shared CT-MRI encoder was evaluated to investigate whether parameter sharing could improve computational efficiency while maintaining segmentation accuracy.

### External Cohort Adaptation

The externally evaluated models were tested under different annotation budgets.

Experiments included:

- Zero-shot evaluation without target-cohort annotations
- Fine-tuning with 4 annotated subjects
- Fine-tuning with 8 annotated subjects
- Fine-tuning with 46 annotated subjects

Additional experiments investigated label-free domain adaptation.

---

## Main Research Findings

### 1. Individual-Muscle Segmentation Accuracy

The best models achieved:

| Task | Performance |
|---|---|
| Thigh-muscle segmentation | Dice: 0.925 ± 0.010 |
| 2.5D thigh-muscle segmentation | Dice: 0.928 |
| CT adipose tissue segmentation | Dice: 0.947 ± 0.017 |
| Thigh-muscle volume agreement | ICC(2,1): 0.998 |

These results demonstrate strong segmentation and volumetric agreement on the evaluated datasets.

### 2. Learning with Limited Annotations

ImageNet-pretrained encoders improved segmentation performance, particularly when labelled MRI data were scarce.

The results demonstrate the importance of appropriate encoder initialisation when developing segmentation models with limited annotated subjects.

### 3. Fat-Fraction Guidance

Dixon-derived fat-fraction guidance significantly improved segmentation when only 2 or 4 annotated subjects were available.

However, the improvement became negligible at larger evaluated annotation budgets.

The additional contrast-attention gate did not provide consistent improvements over fat-fraction guidance alone.

### 4. External Generalisation and Few-Shot Adaptation

A key finding was the substantial performance reduction when a model trained on one MRI cohort was directly evaluated on an independent cohort.

Few-shot fine-tuning substantially improved performance.

| Annotated External Subjects | Dice Score |
|---:|---:|
| 0 (Zero-shot) | 0.11 |
| 4 | 0.61 |
| 8 | 0.76 |
| 46 | 0.88 |

These findings demonstrate that a small amount of target-domain annotation can substantially improve external segmentation performance.

Four subjects represented the smallest evaluated external fine-tuning budget, not a universal minimum annotation requirement.

### 5. Efficient Cross-Modality Learning

A shared CT-MRI encoder maintained segmentation accuracy while reducing model parameters by approximately 44%.

This demonstrates the potential of parameter-efficient multi-task learning across imaging modalities.

---

## Research and Clinical Relevance

### Impact on Research

The study provides practical guidance for developing reliable AI segmentation models when annotated medical imaging datasets are limited.

Potential research benefits include:

- Reducing dependence on extensive manual annotations
- Supporting annotation-efficient deep learning
- Improving cross-cohort model adaptation
- Enabling reproducible segmentation benchmarking
- Supporting quantitative muscle imaging research

### Impact on Clinical Practice

Automated individual-muscle segmentation provides a foundation for quantitative muscle assessment.

Potential applications include:

- Sarcopenia assessment
- Individual-muscle volume quantification
- Muscle health monitoring
- Body composition analysis
- Longitudinal muscle assessment

These applications require further external and clinical validation before routine clinical use.

---

## Repository Structure

```text
MuscleSegmentation/
│
├── scripts/
│   ├── config.py
│   ├── datasets.py
│   ├── models.py
│   ├── models_method.py
│   ├── cross_val.py
│   ├── pretrain_ssl.py
│   ├── train*.py
│   ├── prep_*.py
│   ├── aggregate*.py
│   ├── analyze_*.py
│   ├── make_*.py
│   └── run_*.sh
│
├── results/
│   └── Aggregated experimental results
│
├── figures/
│   └── Generated figures
│
├── .gitignore
│
└── README.md
```

### Key Scripts

| Script | Description |
|---|---|
| `config.py` | Dataset paths and experimental configuration |
| `datasets.py` | Dataset loading |
| `prep_*.py` | Data preprocessing |
| `models.py` | Segmentation architectures |
| `models_method.py` | Method-specific models |
| `train*.py` | Model training |
| `cross_val.py` | Repeated subject-level cross-validation |
| `pretrain_ssl.py` | Self-supervised pretraining |
| `aggregate*.py` | Experimental result aggregation |
| `analyze_*.py` | Statistical analysis |
| `make_*.py` | Figure generation |
| `run_*.sh` | Experimental execution scripts |

---

## Installation

### Requirements

- Python 3.10
- PyTorch
- torchvision
- segmentation-models-pytorch
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib
- nibabel
- pydicom
- tqdm

Create a Python environment:

```bash
python3.10 -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision segmentation-models-pytorch \
    numpy scipy scikit-learn pandas matplotlib nibabel pydicom tqdm
```

GPU acceleration requires a compatible PyTorch and CUDA installation.

---

## Dataset Preparation

Download the public datasets from their original sources.

Update dataset paths in:

```python
scripts/config.py
```

Configure the appropriate local dataset directories before running the experiments.

Datasets must be used in accordance with their original access conditions and licensing requirements.

---

## Running Experiments

Navigate to the scripts directory:

```bash
cd scripts
```

### Run Architecture Benchmarking

```bash
./run_all.sh
```

### Run Cross-Validation

```bash
python cross_val.py --help
```

Run the relevant experiment using the configured annotation budget and training settings.

```bash
python cross_val.py
```

### Run External Validation

```bash
./run_external.sh
```

### Run Additional Experiments

```bash
./run_phase2.sh
./run_phase3.sh
./run_review.sh
```

### Generate Figures

```bash
python make_figures.py
```

Other figure-generation scripts are available using the `make_*.py` naming convention.

---

## Reproducibility

The study uses subject-level evaluation to reduce the risk of data leakage.

Key experimental considerations include:

- Subject-level train/test separation
- Repeated five-fold cross-validation
- Multiple random seeds
- Consistent annotation budgets
- Independent external cohort testing
- Quantitative segmentation evaluation
- Volumetric agreement analysis

The principal cross-validation experiments use three repetitions of five-fold subject-level cross-validation.

The repository includes experimental code and aggregated results to support reproducibility.

For complete experimental protocols, statistical methods, and limitations, refer to the published manuscript and supplementary material.

---

## Citation

If you use this repository, methodology, code, or experimental results in your research, please cite our published paper:

**Srinivasa, A. C., & Patil, S. S. (2026).**

Annotation-Efficient Individual Muscle Segmentation from Dixon MRI: A Multi-Dataset Benchmark of Pretraining and Fat-Fraction Guidance.

*Journal of Imaging Informatics in Medicine.*

https://doi.org/10.1007/s10278-026-02364-6

### BibTeX

```bibtex
@article{Srinivasa2026AnnotationEfficient,
  author = {Srinivasa, Arvind Channarayapatna and Patil, Shamshekhar S.},
  title = {Annotation-Efficient Individual Muscle Segmentation from Dixon MRI: A Multi-Dataset Benchmark of Pretraining and Fat-Fraction Guidance},
  journal = {Journal of Imaging Informatics in Medicine},
  year = {2026},
  doi = {10.1007/s10278-026-02364-6},
  url = {https://doi.org/10.1007/s10278-026-02364-6}
}
```

---

## Publication

**Title:** Annotation-Efficient Individual Muscle Segmentation from Dixon MRI: A Multi-Dataset Benchmark of Pretraining and Fat-Fraction Guidance

**Authors:** Arvind Channarayapatna Srinivasa and Shamshekhar S. Patil

**Journal:** Journal of Imaging Informatics in Medicine

**Publisher:** Springer Nature

**Publication Year:** 2026

**DOI:** 10.1007/s10278-026-02364-6

**Published Manuscript:**

https://link.springer.com/article/10.1007/s10278-026-02364-6

---

## Rights and Usage

The manuscript is published in the *Journal of Imaging Informatics in Medicine* under the applicable publisher terms.

The datasets used in this study are publicly available from their original sources and are not redistributed through this repository.

Users should follow the respective dataset and software licensing requirements.