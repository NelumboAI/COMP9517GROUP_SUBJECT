# COMP9517 Group Project — Fine-Grained Species Classification

[![Course](https://img.shields.io/badge/Course-COMP9517-blue)]()
[![Term](https://img.shields.io/badge/Term-2026%20T2-lightgrey)]()
[![Dataset](https://img.shields.io/badge/Dataset-iNaturalist--2021-green)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![License](https://img.shields.io/badge/License-Academic%20Use%20Only-red)]()

## Overview

This repository contains the implementation, experiments, and report materials for the COMP9517 (Computer Vision) group project at UNSW Sydney, Term 2 2026. The project addresses **fine-grained visual classification** of living organisms — plants, animals, and fungi — using a subset of the [iNaturalist-2021](https://github.com/visipedia/inat_comp/tree/master/2021) dataset.

Fine-grained species recognition is a core challenge in biodiversity monitoring and citizen science: many species are visually near-identical, differing only in subtle morphological cues, while images of the same species vary widely in pose, lighting, background, and scale. This project develops and rigorously compares multiple computer vision pipelines — from classical handcrafted-feature methods to deep convolutional architectures — to tackle this problem at scale.

### Objectives

- Build and evaluate **at least two structurally different classification pipelines**: a traditional handcrafted-feature approach (e.g., SIFT / HOG / LBP with a classical classifier) and a deep learning approach (CNN trained from scratch and via transfer learning).
- Benchmark methods using top-1 / top-5 accuracy, macro-averaged precision/recall/F1, and confusion-matrix-based error analysis.
- Investigate model interpretability using **Grad-CAM** to verify that predictions are driven by the organism itself rather than background context.
- Conduct in-depth ablation and robustness studies to understand failure modes and generalisation behaviour under real-world conditions (e.g., noise, blur, compression).

## Dataset

- **Source:** [iNaturalist-2021 "Mini" split](https://github.com/visipedia/inat_comp/tree/master/2021) (`train_mini`, `val`)
- **Subset used:** ≥ 500 randomly sampled species, with 40 training / 10 validation images per class, and 10 held-out test images per class (drawn from the official validation set)
- **Data integrity:** Training, validation, and test splits are kept strictly separate throughout all experiments to prevent data leakage.

> Exact class lists, sampling seeds, and per-class image counts used in this project are documented in [`data/README.md`](data/README.md) for full reproducibility.

## Methods

| Category | Approach | Status |
|---|---|---|
| Traditional | Bag-of-Visual-Words (SIFT) + SVM | 🔲 |
| Traditional | HOG / LBP + Random Forest | 🔲 |
| Deep Learning | CNN trained from scratch | 🔲 |
| Deep Learning | CNN with ImageNet-pretrained weights (transfer learning) | 🔲 |
| Explainability | Grad-CAM analysis | 🔲 |

*(Update the status column as components are completed — 🔲 Planned · 🚧 In Progress · ✅ Done)*

## Repository Structure

```
COMP9517GROUP_SUBJECT/
├── data/                  # Dataset scripts, class splits, and metadata (raw images not included)
├── notebooks/             # Exploratory analysis and prototyping notebooks
├── src/
│   ├── traditional/       # Handcrafted-feature pipelines (SIFT, HOG, LBP, SVM, RF)
│   ├── deep_learning/     # CNN architectures, training, and fine-tuning scripts
│   ├── evaluation/        # Metrics, confusion matrix, macro-F1 computation
│   └── explainability/    # Grad-CAM and visualisation utilities
├── configs/                # Experiment configuration files
├── results/                 # Logs, metrics, figures (checkpoints and images excluded)
├── report/                # LaTeX (CVPR format) source for the written report
└── README.md
```

## Getting Started

### Requirements

```bash
python >= 3.10
pip install -r requirements.txt
```

### Setup

```bash
git clone <repo-url>
cd COMP9517GROUP_SUBJECT
pip install -r requirements.txt
```

### Downloading the Data

Refer to [`data/README.md`](data/README.md) for instructions on downloading the iNaturalist-2021 mini subset and generating the project's fixed 500-species split.

### Running Experiments

```bash
# Traditional pipeline
python src/traditional/train.py --config configs/bovw_svm.yaml

# Deep learning pipeline
python src/deep_learning/train.py --config configs/resnet_pretrained.yaml
```

## Results

Quantitative results, training curves, and confusion matrices will be added to [`results/`](results/) as experiments are completed, with a summary reported in the final written report and video presentation.

## Team

| Name | Contribution Area |
|---|---|
| Anrui Geng | |
| Haochen Han | |
| Sergio Insuasti | |
| Zhirong Mai | |
| Zhiyang Liu | Evaluation metrics, dataset splits, Grad-CAM explainability, report (LaTeX/CVPR) integration |

## References

- Van Horn et al., *Benchmarking Representation Learning for Natural World Image Collections*, CVPR 2021.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019.

## License

This repository is submitted as coursework for COMP9517 (UNSW Sydney) and is intended for academic evaluation purposes only.
