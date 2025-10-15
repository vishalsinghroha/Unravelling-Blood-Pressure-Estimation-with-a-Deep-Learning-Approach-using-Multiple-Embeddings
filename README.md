# Blood Pressure Estimation with Deep Learning and Multiple Embeddings

This repository provides an implementation of deep learning frameworks for **cuffless blood pressure (BP) estimation** using **pulse arrival time (PAT)** features and **multiple embedding strategies**. The implementation includes Euclidean and Manhattan distance-based representations and attention-driven convolutional models.

---

## Overview

The project introduces multiple neural network architectures designed to estimate systolic (SBP) and diastolic (DBP) blood pressure from PAT sequences.  
This implementation focuses on the **PTT-PPG** dataset, which provides synchronized ECG and PPG recordings with reference blood pressure values.

The core architectures include:

1. **AttentiveConvRegNet** — CNN with Channel and Spatial Attention (CBAM)
2. **ConvolutionalPoolingRegNet** — Standard CNN with pooling layers
3. **NeuralRegressionNet** — Fully connected regression baseline

---

## Dataset

The **PTT-PPG dataset** from PhysioNet was used for this implementation. It contains synchronized ECG and PPG recordings with corresponding BP measurements for multiple subjects.

Dataset link:  
🔗 [https://www.physionet.org/content/pulse-transit-time-ppg/1.0.0/](https://www.physionet.org/content/pulse-transit-time-ppg/1.0.0/)

Please ensure you comply with the PhysioNet Data Use Agreement before downloading.

---

## Key Features

- Multi-embedding similarity matrices using **Euclidean (EUC)** and **Manhattan (MAN)** distance formulations  
- Combined and concatenated feature embeddings (**EUC + MAN**)  
- **Attention-based architecture (CBAM)** with channel and spatial attention modules  
- **Bootstrap-based statistical evaluation (95% CI)** for robust analysis  
- **Bland-Altman plots** and **cumulative error percentage** evaluation for clinical interpretability  
- **Edge-efficient design** — AttentiveConvRegNet weights remain under **2 MB** for EUC/MAN, suitable for mobile or wearable devices

---

## Environment

This implementation was tested and verified on the following configuration:

- **Device**: Apple MacBook Pro (M3 Pro, 36 GB Unified Memory)  
- **Operating System**: macOS 15 Sonoma  
- **Python Version**: 3.10.14  
- **TensorFlow Version**: 2.16.2  
- **TensorFlow-Metal Version**: 1.2.0  

Install dependencies using:
```bash
pip install -r requirements.txt
