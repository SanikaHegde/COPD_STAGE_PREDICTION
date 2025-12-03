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


