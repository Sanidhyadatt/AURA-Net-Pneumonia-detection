# AURA-Net

### Attention-driven, Uncertainty-aware Radiological Assessment Network

AURA-Net is a multi-task deep learning framework for **chest X-ray analysis**, designed to go beyond simple pneumonia detection by jointly learning **pneumonia presence** and **pneumonia etiology** (bacterial vs. viral).

The project explores how a shared deep learning representation can support multiple radiological classification tasks while providing probability-based predictions and uncertainty-aware analysis.

> **Note:** The current dataset does not contain ground-truth severity annotations. Therefore, this implementation does **not claim clinically validated severity prediction**.

---

## Overview

Chest X-ray pneumonia classification is commonly treated as a binary problem:

> **Normal vs. Pneumonia**

AURA-Net extends this into a multi-task setting:

```text
                    Chest X-ray
                         │
                         ▼
                ┌─────────────────┐
                │   CNN Backbone   │
                │    ResNet-18     │
                └────────┬────────┘
                         │
                  Shared Features
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
       Pneumonia Detection    Etiology Classification
          Normal / Pneumonia    Bacterial / Viral
```

The shared representation allows the model to learn features useful for both tasks rather than training two completely independent classifiers.

---

## Key Features

* **Multi-task learning** for pneumonia detection and etiology classification
* **Transfer learning** using a pretrained ResNet-18 backbone
* **Binary Focal Loss** for pneumonia detection
* **Cross-Entropy Loss** for bacterial vs. viral classification
* **Masked etiology loss**, so normal X-rays do not contribute to the bacterial/viral task
* **AdamW optimization**
* **Cosine Annealing learning-rate scheduling**
* GPU/CUDA support
* Validation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion matrices
  * Classification reports

---

## Dataset

AURA-Net was developed using the Kaggle **Chest X-Ray Images (Pneumonia)** dataset.

The dataset used in this implementation contains:

| Class               |    Images |
| ------------------- | --------: |
| Normal              |     1,341 |
| Bacterial Pneumonia |     2,530 |
| Viral Pneumonia     |     1,345 |
| **Total**           | **5,216** |

### Tasks

#### Task 1 — Pneumonia Detection

```text
Normal → 0
Pneumonia → 1
```

Both bacterial and viral pneumonia are treated as pneumonia-positive cases.

#### Task 2 — Etiology Classification

Only pneumonia-positive images are considered:

```text
Bacterial → 0
Viral → 1
```

Normal images are excluded from the etiology loss.

---

## Model Architecture

### 1. ResNet-18 Backbone

A pretrained **ResNet-18** network is used as the shared feature extractor.

The pretrained backbone provides transferable visual representations while allowing the model to specialize in chest X-ray patterns during training.

### 2. Shared Feature Layer

The extracted features are passed through a shared 512-dimensional representation.

```text
ResNet-18
    ↓
Shared Feature Layer
    ↓
 ┌───────────────┬────────────────┐
 │               │                │
 ▼               ▼
Pneumonia      Etiology
Head           Head
```

### 3. Pneumonia Head

A binary classification head predicts whether pneumonia is present.

The output is converted to a probability using the sigmoid function:

```text
P(pneumonia) = sigmoid(logit)
```

### 4. Etiology Head

A two-class classification head predicts:

```text
Bacterial
    vs.
Viral
```

This head is trained only using pneumonia-positive samples.

---

## Loss Function

AURA-Net uses a weighted multi-task objective:

```text
Total Loss =
    Pneumonia Loss
    + 0.5 × Etiology Loss
```

### Pneumonia Loss

Binary Focal Loss is used to focus learning on harder examples:

```text
L_pneumonia = Binary Focal Loss
```

### Etiology Loss

Cross-Entropy Loss is applied only to pneumonia-positive samples:

```text
L_etiology =
    Cross Entropy
    for pneumonia samples only
```

The final objective is:

```text
L_total = L_pneumonia + 0.5 L_etiology
```

This prevents normal X-rays from being incorrectly treated as bacterial or viral cases during etiology training.

---

## Training Configuration

| Parameter                | Value                     |
| ------------------------ | ------------------------- |
| Backbone                 | ResNet-18                 |
| Input Size               | 224 × 224                 |
| Batch Size               | 32                        |
| Train / Validation Split | 85 / 15                   |
| Optimizer                | AdamW                     |
| Learning Rate            | 1 × 10⁻⁴                  |
| Scheduler                | Cosine Annealing          |
| Epochs                   | 10                        |
| Pneumonia Loss           | Binary Focal Loss         |
| Etiology Loss            | Cross-Entropy             |
| Etiology Weight          | 0.5                       |
| Device                   | CUDA / GPU when available |
| Random Seed              | 42                        |

The best model checkpoint is selected according to validation loss.

---

# Results

Evaluation was performed on the **15% validation split containing 783 images**.

## Pneumonia Detection

| Metric        |      Score |
| ------------- | ---------: |
| **Accuracy**  | **98.08%** |
| **Precision** | **99.63%** |
| **Recall**    | **97.67%** |
| **F1-score**  | **98.64%** |
| **ROC-AUC**   | **99.80%** |

### Confusion Matrix

```text
                    Predicted
                 Normal  Pneumonia
Actual Normal       224       2
       Pneumonia     13     544
```

The model correctly classified **768 out of 783 validation images** for pneumonia detection.

---

## Bacterial vs. Viral Classification

The etiology task was evaluated on **557 pneumonia-positive validation images**.

| Metric        |      Score |
| ------------- | ---------: |
| **Accuracy**  | **77.74%** |
| **Precision** | **71.94%** |
| **Recall**    | **54.05%** |
| **F1-score**  | **61.73%** |
| **ROC-AUC**   | **84.04%** |

### Confusion Matrix

```text
                    Predicted
                 Bacterial  Viral
Actual Bacterial     333      39
       Viral          85     100
```

The results show that distinguishing bacterial and viral pneumonia is substantially more difficult than detecting pneumonia itself, particularly for viral-case recall.

---

# Classification Reports

### Pneumonia Detection

```text
              precision    recall  f1-score   support

Normal           0.95      0.99      0.97       226
Pneumonia        1.00      0.98      0.99       557

accuracy                              0.98       783
macro avg        0.97      0.98      0.98       783
weighted avg     0.98      0.98      0.98       783
```

### Bacterial vs. Viral

```text
              precision    recall  f1-score   support

Bacterial        0.80      0.90      0.84       372
Viral            0.72      0.54      0.62       185

accuracy                              0.78       557
macro avg        0.76      0.72      0.73       557
weighted avg     0.77      0.78      0.77       557
```

---

# Why Multi-Task Learning?

Instead of learning pneumonia detection and etiology independently, AURA-Net shares visual representations between the two tasks.

The model learns:

```text
Chest X-ray
     │
     ▼
Shared visual representation
     │
     ├──────────────► Pneumonia detection
     │
     └──────────────► Bacterial / Viral classification
```

This allows common radiological patterns to contribute to multiple predictions while maintaining separate task-specific outputs.

---

# Evaluation

AURA-Net uses several complementary metrics.

### Accuracy

Measures the overall proportion of correctly classified samples.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were successfully identified.

### F1-score

Provides a combined measure of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Confusion Matrix

Provides a detailed view of correct and incorrect predictions for each class.

---

# Project Structure

```text
AURA-Net/
│
├── dataset/
│   ├── Normal/
│   ├── Pneumonia_bacteria/
│   └── Pneumonia_virus/
│
├── models/
│   └── aura_net_best_model.pth
│
├── notebooks/
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   └── evaluation.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Sanidhyadatt/AURA-Net.git
cd AURA-Net
```

Install the required dependencies:

```bash
pip install torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn
```

For CUDA-enabled systems, install the appropriate PyTorch build for your CUDA version.

---

# Dataset Setup

Download the Chest X-Ray Images (Pneumonia) dataset and organize it as:

```text
xraychest/
├── Normal/
├── Pneumonia_bacteria/
└── Pneumonia_virus/
```

The implementation parses the class information from the directory structure.

---

# Training

Run the training pipeline:

```bash
python train.py
```

The training process:

1. Loads and preprocesses the X-ray images.
2. Creates an 85/15 training-validation split.
3. Initializes the pretrained ResNet-18 backbone.
4. Trains the pneumonia and etiology heads jointly.
5. Tracks validation loss.
6. Saves the best-performing checkpoint.

Example:

```text
Epoch 1
Train Loss: 0.2720
Val Loss:   0.2479

Epoch 2
Train Loss: 0.2012
Val Loss:   0.2456
```

The best checkpoint is saved as:

```text
aura_net_best_model.pth
```

---

# Evaluation

Run the evaluation pipeline:

```bash
python evaluate.py
```

The evaluation script generates:

* Pneumonia accuracy
* Pneumonia precision
* Pneumonia recall
* Pneumonia F1-score
* Pneumonia ROC-AUC
* Bacterial/viral accuracy
* Bacterial/viral precision
* Bacterial/viral recall
* Bacterial/viral F1-score
* Bacterial/viral ROC-AUC
* Confusion matrices
* Classification reports
* ROC curves

---

# Limitations

This project is a research/academic implementation and should **not be used as a clinical diagnostic system**.

### 1. Validation rather than independent testing

The reported metrics were obtained on a held-out validation split rather than an independent external test cohort.

### 2. Dataset size and composition

The model was evaluated on a single public chest X-ray dataset. Performance on other hospitals, scanners, populations, or acquisition protocols may differ.

### 3. Etiology classification

The bacterial-vs-viral task has considerably lower performance than pneumonia detection, particularly for viral recall.

### 4. No ground-truth severity labels

The dataset used in this project does not provide radiologist-annotated severity labels.

Therefore, **AURA-Net does not currently claim clinically validated severity prediction**.

### 5. Patient-level separation

For stronger medical-model evaluation, future experiments should ensure patient-level separation between training and validation/test data wherever patient identifiers are available.

---

# Future Work

Potential extensions include:

* CBAM or other attention mechanisms
* DenseNet / EfficientNet backbone comparison
* Monte Carlo Dropout for epistemic uncertainty
* Calibration and reliability analysis
* Patient-level data splitting
* External test-set evaluation
* Grad-CAM visual explanations
* Radiologist-annotated severity labels
* Lung and opacity segmentation
* Explainable AI dashboards
* Model calibration and threshold optimization
* Larger and more diverse chest X-ray datasets

---

# Ethical & Clinical Note

AURA-Net is intended for **research and educational purposes**.

The predictions generated by the model should not be interpreted as medical diagnoses or used as a substitute for evaluation by a qualified medical professional.

Particular caution is required when interpreting bacterial-vs-viral predictions because the current results demonstrate substantially lower viral recall than pneumonia detection performance.

---

# Technologies

```text
Python
PyTorch
TorchVision
Scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
CUDA
```

---

# Model Summary

```text
Input
  │
  ▼
224 × 224 Chest X-ray
  │
  ▼
Pretrained ResNet-18
  │
  ▼
Shared 512-D Representation
  │
  ├─────────────────────┐
  ▼                     ▼
Pneumonia Head       Etiology Head
  │                     │
  ▼                     ▼
Normal / Pneumonia   Bacterial / Viral
```

### Validation Performance

```text
Pneumonia Detection
────────────────────
Accuracy     98.08%
Precision    99.63%
Recall       97.67%
F1-score     98.64%
ROC-AUC      99.80%


Bacterial vs Viral
───────────────────
Accuracy     77.74%
Precision    71.94%
Recall       54.05%
F1-score     61.73%
ROC-AUC      84.04%
```

---

## Author

**Sanidhya Datt**  
B.Tech — Artificial Intelligence & Machine Learning  
NMAM Institute of Technology, Nitte  

GitHub: `Sanidhyadatt`

---

## Disclaimer

This repository is an academic/research project demonstrating multi-task deep learning for chest X-ray classification. It is not a clinically validated medical device and must not be used for diagnosis or treatment decisions.
