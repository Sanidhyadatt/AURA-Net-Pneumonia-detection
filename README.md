# 🫁 AURA-Net: Attention-driven, Uncertainty-aware Radiological Assessment Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanidhyadatt/AURA-Net-Pneumonia-detection/blob/main/Pneumonia_detection.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AURA-Net** is a deep-learning multi-task architecture for automated Chest X-Ray radiological assessment. Built upon an attention-augmented DenseNet121 backbone, it extends conventional binary pneumonia classification by performing multi-task etiology estimation, severity index regression, and Monte Carlo dropout uncertainty triage for radiologist decision support.

---

## ✨ Key Innovations & Features

- **🔀 Multi-Task Learning Architecture**:
  - **Pneumonia Classification**: Binary diagnostic head (Normal vs. Pneumonia).
  - **Etiology Identification**: Multi-class classification (Normal / Bacterial / Viral).
  - **Severity Index Regression**: Continuous score reflecting ROI opacification density.

- **🎯 CBAM Spatial & Channel Attention**:
  - Custom **Convolutional Block Attention Module (CBAM)** injected across scale boundaries of DenseNet121.
  - Channel attention highlights diagnostically relevant feature channels, while spatial attention suppresses high-contrast non-pathological structures (ribs, clavicles) to focus on focal pulmonary opacities.

- **📊 Epistemic Uncertainty & Auto-Triage (MC Dropout)**:
  - Test-Time Monte Carlo (MC) Dropout estimates model epistemic variance across predictions.
  - Automatically flags high-uncertainty or ambiguous scans for urgent human radiologist review.

- **⚖️ Hard-Example Mining with Focal Loss**:
  - Custom Binary and Multi-class Focal Loss functions dynamically down-weight easy background examples and address class imbalance.

---

## 🏗️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Input Chest X-Ray      │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │  DenseNet121 + CBAM      │
                          │  (Spatial & Channel)     │
                          └────────────┬─────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
┌──────────▼───────────┐   ┌───────────▼───────────┐   ┌───────────▼───────────┐
│   Pneumonia Head     │   │     Etiology Head     │   │     Severity Head     │
│ (Binary Classification)  │ (Bacterial/Viral/Normal)  │ (Continuous Regression) │
└──────────────────────┘   └───────────────────────┘   └───────────────────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │    MC Dropout Inference  │
                          │  (Uncertainty & Triage) │
                          └──────────────────────────┘
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.8+
- PyTorch >= 1.12
- `torchvision`, `Pillow`, `scikit-learn`, `matplotlib`

```bash
pip install torch torchvision pillow scikit-learn matplotlib
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/Sanidhyadatt/AURA-Net-Pneumonia-detection.git
   cd AURA-Net-Pneumonia-detection
   ```
2. Open `Pneumonia_detection.ipynb` in Google Colab or Jupyter Notebook:
   - Make sure GPU acceleration is enabled (e.g. NVIDIA T4 / V100).
   - Execute all cells to run end-to-end verification, data loading, training, and clinical triage generation.

---

## 📁 Repository Structure

```
.
├── Pneumonia_detection.ipynb   # Complete PyTorch pipeline & multi-task architecture
└── README.md                   # Project documentation
```

---

## 📜 References & Acknowledgments

- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **CBAM Paper**: Woo et al., *"CBAM: Convolutional Block Attention Module"*, ECCV 2018.
- **MC Dropout**: Gal & Ghahramani, *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*, ICML 2016.
