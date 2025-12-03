# 🫁 Multimodal Deep Learning Framework for Early COPD Prediction & Severity Stage Classification

## 📌 Overview  
This project implements a **dual Deep Learning–based system** designed for early diagnosis and accurate staging of **Chronic Obstructive Pulmonary Disease (COPD)**.  
Unlike traditional approaches that rely only on spirometry or binary classification of lung sounds, this framework **combines two complementary data sources**:

- 📊 **Structured spirometry & demographic data (FEV1, FVC, Age, BMI, Smoking History)**  
- 🎧 **Respiratory audio signals converted into MFCC features**

By integrating both modalities, the system achieves **98.19% accuracy** and **AUC = 0.99**, significantly outperforming single-modality baselines :contentReference[oaicite:0]{index=0}.

---

## 🎯 Key Features  
- Dual-model system combining:
  - **MLP** for COPD stage prediction  
  - **CNN–LSTM** for COPD vs Non-COPD lung sound classification  
- MFCC-based feature extraction for respiratory sounds  
- Automated preprocessing pipelines  
- Flask-based lightweight web deployment  
- Real-time, non-invasive respiratory assessment  
- High accuracy and clinically meaningful predictions

---

## 🧬 System Architecture  
```mermaid
flowchart TD
A[Input Data] --> B[Spirometry Data Preprocessing]
A --> C[Lung Sound Audio Preprocessing]
B --> D[MLP Model - Stage Prediction]
C --> E[CNN-LSTM Model - Sound Classification]
D --> F[Prediction Fusion]
E --> F
F --> G[Final COPD Diagnosis & Stage Output]



📂 Dataset Details
1️⃣ Structured Tabular Data (NHANES)

Contains spirometry and demographic attributes: FEV1, FVC, Age, Sex, BMI, Smoking History.

Training Samples: 5,615

Testing Samples: 1,404


6a792677-8b32-4d72-8b45-d60d8b1…

2️⃣ Respiratory Audio Dataset (Kaggle Respiratory Sound Database)

Labeled .wav files of:

COPD

Asthma

Pneumonia

Bronchial Disorder

Healthy

Training Files: 1,937

Testing Files: 485


6a792677-8b32-4d72-8b45-d60d8b1…

🛠️ Preprocessing Steps
Tabular Data (NHANES)

Missing value handling

Feature scaling (z-score normalization)

Label encoding of categorical attributes

Train/test split


6a792677-8b32-4d72-8b45-d60d8b1…

Audio Data

Noise reduction (100–2000 Hz band-pass filtering)

Segmentation into uniform frames

MFCC extraction

Augmentation (pitch shift, time-stretching, background noise)

Normalization (0–1 scaling)


6a792677-8b32-4d72-8b45-d60d8b1…

🧠 Model Architectures
1️⃣ MLP (Multilayer Perceptron) — COPD Stage Prediction

Uses spirometry + demographic features to classify:

Mild

Moderate

Severe

Confusion Matrix (from paper):


6a792677-8b32-4d72-8b45-d60d8b1…

2️⃣ CNN–LSTM — Lung Sound COPD Classification

CNN extracts spectral features from MFCCs

LSTM learns temporal breathing cycles

Output: COPD vs Non-COPD

Performance Highlights:

Accuracy: 98.19%

Precision: 98%

Recall: 97.8%

F1-Score: 97.9%

AUC: 1.00


6a792677-8b32-4d72-8b45-d60d8b1…

📈 Results
🔹 CNN–LSTM Performance

Extremely high classification accuracy (98.19%)

Near-perfect AUC of 1.00

Strong generalization with low overfitting


6a792677-8b32-4d72-8b45-d60d8b1…

🔹 MLP Stage Prediction

High accuracy for Mild, Moderate, and Severe stages

Stable training and validation curves


6a792677-8b32-4d72-8b45-d60d8b1…

🔹 Combined Framework

Integrates predictions into a unified COPD risk profile

Suitable for real-time clinical or remote monitoring

Deployed via Flask lightweight server


6a792677-8b32-4d72-8b45-d60d8b1…
