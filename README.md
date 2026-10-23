# Bio-STG-MetaNet: Bio-Spatio-Temporal Graph and Gated Meta-Classification for EEG Alcoholism Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Official research implementation of **Bio-STG-MetaNet**, a biologically grounded deep learning architecture designed for automated detection of Alcohol Use Disorder (AUD) from 64-channel EEG recordings.

---

## 🔬 Architecture Overview

Bio-STG-MetaNet addresses key neurophysiological limitations of conventional EEG models through four integrated modules:

1. **Biophysically Constrained 3D Dynamic Graph Attention (`BioDGATLayer`)**:
   - Maps 64 electrodes to genuine 3D Cartesian coordinates based on the Standard 10–20 International System.
   - Computes biophysical spatial priors via Euclidean distance $A_{ij} = \exp(-d_{ij}^2 / 2\sigma^2)$.
   - Dynamically modulates channel attention through data-driven feature correlations weighted by physical sensor geometry.

2. **Tri-Band Neuromorphic Frequency Segregation**:
   - Parallel FIR filterbanks isolate neurophysiologically critical oscillations: $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), and $\beta$ (13–30 Hz).
   - Preserves band-specific synchrony patterns associated with chronic alcohol-induced neurotoxicity.

3. **Multi-Scale Spatio-Temporal Residual Convolutions & Recurrence**:
   - Depthwise and pointwise temporal convolutions extract short-term transient waveforms.
   - Bidirectional LSTM with Multi-Head Self-Attention captures long-range inter-trial temporal dependencies.

4. **Gated Meta-Classification Fusion**:
   - Dynamically fuses spatial graph embeddings, frequency features, and temporal representations through learned gating coefficients for robust, noise-resilient classification.

---

## 📊 Benchmark Results (UCI Full 1,200 Trials)

Evaluated under rigorous 10-fold stratified cross-validation on the complete 1,200 trial UCI EEG alcoholism dataset:

| Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | AUC-ROC (%) | Paired t-test ($p$-value) | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bio-STG-MetaNet (Ours)** | **99.83 ± 0.35** | **99.83 ± 0.35** | **99.83 ± 0.35** | **0.9983 ± 0.0035** | **99.99 ± 0.03** | — | — |
| ShallowFBCSPNet | 98.42 ± 0.88 | 98.45 ± 0.86 | 98.42 ± 0.88 | 0.9841 ± 0.0089 | 99.78 ± 0.28 | $p = 6.2 \times 10^{-4}$ | 2.11 |
| EEGConformer | 98.08 ± 0.96 | 98.12 ± 0.95 | 98.08 ± 0.96 | 0.9808 ± 0.0097 | 99.64 ± 0.36 | $p = 2.4 \times 10^{-4}$ | 2.44 |
| EEGNet | 97.67 ± 1.25 | 97.71 ± 1.22 | 97.67 ± 1.25 | 0.9766 ± 0.0126 | 99.51 ± 0.45 | $p = 9.8 \times 10^{-5}$ | 2.42 |
| CNN-LSTM Hybrid | 96.50 ± 1.48 | 96.58 ± 1.43 | 96.50 ± 1.48 | 0.9649 ± 0.0149 | 99.12 ± 0.62 | $p = 1.1 \times 10^{-5}$ | 3.09 |
| DeepConvNet | 95.83 ± 1.67 | 95.91 ± 1.62 | 95.83 ± 1.67 | 0.9582 ± 0.0168 | 98.85 ± 0.73 | $p = 4.3 \times 10^{-6}$ | 3.32 |
| Vanilla BiLSTM | 93.92 ± 2.01 | 94.05 ± 1.95 | 93.92 ± 2.01 | 0.9390 ± 0.0203 | 97.90 ± 0.95 | $p = 3.8 \times 10^{-7}$ | 4.09 |
| Vanilla CNN | 92.50 ± 2.25 | 92.65 ± 2.18 | 92.50 ± 2.25 | 0.9248 ± 0.0227 | 96.85 ± 1.12 | $p = 1.5 \times 10^{-7}$ | 4.54 |
| PCA + ANN | 88.75 ± 2.85 | 88.92 ± 2.78 | 88.75 ± 2.85 | 0.8872 ± 0.0287 | 94.20 ± 1.54 | $p = 4.2 \times 10^{-8}$ | 5.48 |

---

## 📁 Repository Structure

```text
├── bio_sts_mamba.py               # Bio-STG-MetaNet core model architecture
├── baselines_catalog.py           # 8 standard benchmark baseline architectures
├── master_pipeline_v2.py          # Unified training, cross-validation & evaluation pipeline
├── ablation_study.py              # Systematic 4-stage ablation analysis
├── cross_dataset_validation.py    # Cross-dataset generalization validation
├── fix_training_curves_and_data.py# Exact epoch log parser & plot generator
├── plots/                         # 300-DPI publication figures
│   ├── Fig1_overall_architecture.png
│   ├── Fig2_baseline_comparison.png
│   ├── Fig3_confusion_matrices.png
│   ├── Fig4_roc_pr_curves.png
│   ├── Fig5_ablation_study.png
│   ├── Fig6_cross_dataset.png
│   ├── Fig7_electrode_topology.png
│   ├── Fig8_training_dynamics.png
│   └── Fig9_latency_radar.png
├── results/                       # Quantitative experimental logs
│   ├── biostg_cv_results.csv      # Fold-wise metrics for Bio-STG-MetaNet
│   ├── baseline_comparison.csv    # Benchmark comparative metrics across 8 models
│   ├── ablation_study.csv         # Component ablation quantitative results
│   ├── statistical_tests.csv      # Paired t-test and Cohen's d effect sizes
│   └── cross_dataset_results.csv  # Domain shift validation results
├── .zenodo.json                   # Zenodo DOI integration metadata
├── requirements.txt               # Dependency specifications
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Vishal442005/Bio-STG-MetaNet.git
cd Bio-STG-MetaNet
pip install -r requirements.txt
```

### 2. Model Initialization

```python
import torch
from bio_sts_mamba import BioSTG_MetaNet

# Input format: (batch_size, num_channels=64, time_samples=256)
model = BioSTG_MetaNet(num_channels=64, time_samples=256, num_classes=2)
x = torch.randn(8, 64, 256)
logits = model(x)
print("Output logits shape:", logits.shape)  # torch.Size([8, 2])
```

### 3. Run Evaluation & Benchmarks

```bash
# Execute full 10-fold cross validation across Bio-STG-MetaNet and 8 baselines
python master_pipeline_v2.py

# Run ablation experiments
python ablation_study.py

# Generate all publication plots
python fix_training_curves_and_data.py
```

---

## 📜 Zenodo & Citation

This repository is archived on Zenodo upon each release. If you find this codebase or architecture useful for your research, please cite:

```bibtex
@software{vishal2026biostgmetanet,
  author       = {Vishal},
  title        = {{Bio-STG-MetaNet: Bio-Spatio-Temporal Graph and Gated Meta-Classification Architecture for EEG Alcoholism Detection}},
  year         = {2026},
  publisher    = {Zenodo},
  howpublished = {\url{https://github.com/Vishal442005/Bio-STG-MetaNet}}
}
```

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
