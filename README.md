# Hybrid Deep Learning Framework for Arrhythmia Detection Using CNN-Attention Model

## Project Overview

Cardiac arrhythmias are abnormalities in the heart's electrical rhythm that can affect normal cardiac function. Early and accurate identification of different arrhythmia patterns through Electrocardiogram (ECG) signals can support cardiac monitoring and healthcare research.

This project presents a **Hybrid Deep Learning Framework combining a 1D Convolutional Neural Network (CNN) with a Multi-Head Self-Attention mechanism** for multiclass arrhythmia classification using the MIT-BIH Arrhythmia Database.

The proposed framework extracts important morphological features from ECG signals using CNN layers and captures relevant relationships between signal features through an attention mechanism.

The model classifies ECG heartbeats into five AAMI categories: Normal (N), Supraventricular Ectopic (S), Ventricular Ectopic (V), Fusion (F), and Unknown (Q).

## Project Objectives

- Develop a hybrid CNN-Attention deep learning architecture for ECG signal classification.
- Perform ECG signal preprocessing and R-peak-based heartbeat segmentation.
- Classify ECG signals into five AAMI heartbeat categories.
- Apply class-weighted categorical cross-entropy to address class imbalance.
- Evaluate model performance using multiple classification metrics.
- Explore the application of deep learning in automated cardiac arrhythmia analysis.

## Dataset

**Dataset:** MIT-BIH Arrhythmia Database

**Source:** PhysioNet

https://physionet.org/content/mitdb/1.0.0/

The MIT-BIH Arrhythmia Database contains annotated ECG recordings used for research and development in cardiac arrhythmia detection.

### AAMI Classification Categories

| Class | Description |
|---|---|
| N | Normal Beats |
| S | Supraventricular Ectopic Beats |
| V | Ventricular Ectopic Beats |
| F | Fusion Beats |
| Q | Unknown / Other Beats |

### Dataset Distribution

| Class | Number of Samples |
|---|---:|
| N | 90,363 |
| S | 11,398 |
| V | 7,235 |
| F | 2,781 |
| Q | To be verified from final dataset |
| **Total** | **111,777 + Q samples** |

## Methodology

### 1. ECG Signal Preprocessing

- Applied a 0.5–40 Hz bandpass filter to ECG signals.
- Performed signal normalization.
- Used R-peak detection and heartbeat segmentation.
- Prepared ECG heartbeat segments for model training.

### 2. Feature Extraction

A one-dimensional Convolutional Neural Network (1D-CNN) was implemented to extract meaningful morphological features from ECG heartbeat signals.

### 3. Attention Mechanism

A lightweight Multi-Head Self-Attention mechanism with four attention heads was integrated into the architecture to capture relationships between extracted ECG features.

### 4. Model Architecture

The proposed hybrid architecture consists of:

- 1D Convolutional Layers
- Batch Normalization
- Max Pooling
- Multi-Head Self-Attention (4 Heads)
- Fully Connected Dense Layers
- ReLU Activation
- Dropout Regularization
- Softmax Output Layer

### 5. Model Training

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Class-Weighted Categorical Cross-Entropy
- Random Seed: 42
- Framework: PyTorch
- Execution Environment: Google Colab
- Hardware: NVIDIA Tesla T4 GPU (16 GB)

### Technologies & Tools Used

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Neural Network Architecture:** 1D Convolutional Neural Network (CNN), Multi-Head Self-Attention
- **Signal Processing:** SciPy, Digital Bandpass Filtering, R-Peak Detection, ECG Signal Segmentation
- **Data Manipulation:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning Utilities:** Scikit-learn
- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
- **Data Preprocessing:** Signal Normalization, Data Cleaning, Feature Preparation
- **Model Training:** Adam Optimizer, Class-Weighted Categorical Cross-Entropy, Dropout Regularization
- **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, Matthews Correlation Coefficient (MCC), Cohen's Kappa
- **Development Environments:** Google Colab, Jupyter Notebook, Visual Studio Code (VS Code)
- **Hardware Acceleration:** CUDA, NVIDIA Tesla T4 GPU (16 GB VRAM)
- **Version Control & Collaboration:** Git, GitHub
- **Research & Documentation:** IEEE Conference Research, Technical Documentation

## Model Performance

The proposed CNN-Attention model achieved the following evaluation results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 96.18% |
| Weighted F1-Score | 96.50% |
| Macro F1-Score | 84.04% |
| Matthews Correlation Coefficient | 0.8849 |
| Cohen's Kappa | 0.8822 |

### Class-wise ROC-AUC Scores

| AAMI Class | ROC-AUC |
|---|---:|
| N | 0.995 |
| S | 0.987 |
| V | 0.998 |
| F | 0.985 |
| Q | 0.999 |

## Project Contributions

- **ECG Data Processing:** Implemented ECG signal preprocessing, including bandpass filtering, normalization, and R-peak-based heartbeat segmentation.
- **Feature Preparation:** Prepared ECG heartbeat segments for multiclass classification.
- **Deep Learning Architecture:** Developed a hybrid 1D-CNN and Multi-Head Self-Attention architecture.
- **Class Imbalance Handling:** Implemented class-weighted categorical cross-entropy to address differences in heartbeat class distributions.
- **Model Training:** Conducted model training and experimentation using PyTorch and Google Colab with GPU acceleration.
- **Performance Evaluation:** Evaluated the model using accuracy, weighted F1-score, macro F1-score, MCC, Cohen's Kappa, and class-wise ROC-AUC.
- **Inference Demonstration:** Developed a web-based demonstration that accepts ECG input, displays the waveform, and presents the predicted arrhythmia class.

## Research Publication & Conference Presentation

**Research Paper Title:**

Hybrid Deep Learning Framework for Arrhythmia Detection Using CNN-Attention Model on MIT-BIH ECG Dataset.

- **Conference:** IEEE AMSCEHT 2026
- **Presentation Type:** Oral / Virtual Presentation
- **Paper ID:** 410
- **Presentation Dates:** 11–13 September 2026
- **Venue:** OP Jindal University, Raigarh, India

## Applications

- ECG signal classification research.
- Automated cardiac rhythm analysis.
- Deep learning applications in healthcare.
- Research and experimentation in biomedical signal processing.
- Development of ECG-based AI inference demonstrations.

## Future Improvements

- Explore improved generalization across independent ECG datasets.
- Investigate performance on diverse patient populations.
- Optimize model inference for resource-constrained environments.
- Explore explainability techniques for ECG classification.
- Further evaluate robustness against noisy ECG signals.

## Disclaimer

This project is intended for academic research and educational purposes. The model and web demonstration are experimental and have not been clinically validated. Predictions should not be used for medical diagnosis, treatment decisions, or emergency care.
