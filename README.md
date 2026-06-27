# PCB Defect Inspection Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688)
![ONNX](https://img.shields.io/badge/ONNX-Optimized-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Production-ready AI-powered PCB Defect Detection System for Automated Optical Inspection**

Designed for semiconductor and electronics manufacturing environments using **YOLOv8**, **ONNX Runtime**, **FastAPI**, **Streamlit**, and a **Knowledge Retrieval Assistant**.

</div>

---

# Demo

> **Working Demo Video**

https://github.com/user-attachments/assets/c8a2de29-561a-457b-8493-0a5b0eb772f4

---

# System Architecture

<p align="center">
<img src="assets/PCBDEFECTSYSTEM.png" width="100%">
</p>

The platform provides an end-to-end inspection pipeline covering:

- Dataset engineering
- Training pipeline
- Model evaluation
- Model optimization
- ONNX deployment
- FastAPI inference service
- Streamlit dashboard
- Knowledge Retrieval Assistant
- Production monitoring

---

# Project Highlights

## End-to-End ML Engineering Pipeline

✔ Dataset validation

✔ Annotation conversion

✔ Automated train/validation/test split

✔ Offline augmentation

✔ YOLOv8 transfer learning

✔ Evaluation framework

✔ Failure analysis

✔ ONNX optimization

✔ FastAPI deployment

✔ Streamlit dashboard

✔ REST API

✔ Docker support

✔ Knowledge Assistant (RAG)

---

# Features

## Data Engineering

- DeepPCB dataset preprocessing
- Annotation conversion (YOLO format)
- Dataset quality validation
- Duplicate detection
- Statistical reporting
- Offline data augmentation
- Automated dataset splitting

---

## Model Training

- YOLOv8n transfer learning
- Configurable hyperparameters
- Early stopping
- Training callbacks
- Experiment tracking
- Multi-device support

---

## Evaluation

- mAP
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Precision-Recall Curve
- Failure Analysis

---

## Inference

Supports

- Image inference
- Batch inference
- Folder inference
- Video inference
- Camera inference

Unified inference engine using

```
PCBDetector
```

Supports

- PyTorch
- ONNX Runtime

---

## Optimization

- ONNX Export
- Duplicate suppression
- Latency benchmarking
- Memory benchmarking
- Quantization framework

---

## REST API

Built using FastAPI.

Available endpoints include

```
GET /health

GET /metadata

POST /predict

POST /predict/batch

POST /predict/url

POST /predict/annotated

GET /docs
```

Swagger documentation is automatically generated.

---

## Streamlit Dashboard

Interactive dashboard containing

- Model Card
- Evaluation Dashboard
- Benchmark Results
- Knowledge Assistant
- Detection Demo

---

# Repository Structure

```text
PCB-DEFECT-INSPECTION-PLATFORM/

├── app/
│   ├── Streamlit Dashboard
│   └── Pages
│
├── configs/
│
├── data/
│   ├── knowledge_base/
│   └── metadata/
│
├── docs/
│
├── notebooks/
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── deployment/
│   ├── evaluation/
│   ├── inference/
│   ├── optimization/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── rag/
│   ├── training/
│   ├── video/
│   └── utils/
│
├── tests/
│
└── assets/
```

---

# Dataset

Dataset

DeepPCB

Contains six PCB defect classes

| Class |
|---------|
| Open Circuit |
| Short Circuit |
| Mouse Bite |
| Spur |
| Spurious Copper |
| Pin Hole |

Pipeline

Raw Dataset

↓

Validation

↓

Annotation Conversion

↓

Augmentation

↓

Train / Validation / Test Split

↓

Training

---

# Model Performance

## Evaluation Results

<p align="center">
<img src="assets/evaluation_results.png" width="90%">
</p>

### Performance

| Metric | Score |
|---------|--------|
| mAP@50 | **98.8%** |
| Precision | **98%+** |
| Recall | **98%+** |
| F1 Score | **97.5%** |

---

# Deployment Benchmark

<p align="center">
<img src="assets/benchmark_results.png" width="90%">
</p>

| Model | Device | FPS | Latency |
|---------|---------|---------|----------|
| PyTorch | Apple MPS | **57.8 FPS** | **17.30 ms** |
| ONNX Runtime | CPU | **54.8 FPS** | **18.26 ms** |

---

# Technologies Used

### Computer Vision

- OpenCV
- YOLOv8
- Albumentations

### Deep Learning

- PyTorch
- ONNX Runtime

### API

- FastAPI
- Uvicorn

### Dashboard

- Streamlit

### Deployment

- Docker
- Docker Compose

### Utilities

- NumPy
- Pandas
- Matplotlib

---

# Installation

Clone repository

```bash
git clone https://github.com/solo938/PCB-DEFECT-INSPECTION-PLATFORM.git

cd PCB-DEFECT-INSPECTION-PLATFORM
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Train

```bash
python -m src.training.train
```

---

# Evaluate

```bash
python -m src.evaluation.report
```

---

# Run API

```bash
python -m src.api.app \
--weights runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt \
--device mps
```

Swagger

```
http://localhost:8000/docs
```

---

# Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# Testing

```bash
pytest
```

---

# Future Improvements

- TensorRT deployment
- INT8 Quantization
- Active Learning pipeline
- Multi-camera inspection
- Real-time production monitoring
- Defect tracking across frames
- Industrial PLC integration
- MLOps with MLflow
- Kubernetes deployment

---

# Engineering Skills Demonstrated

- Computer Vision
- Object Detection
- Transfer Learning
- Dataset Engineering
- Model Evaluation
- Error Analysis
- Model Optimization
- ONNX Deployment
- REST API Development
- Production Inference
- Docker
- Streamlit
- Software Engineering
- Testing
- Documentation

---

# License

MIT License

---

# Author

**Sahariar Hasan**

Machine Learning Engineer | Computer Vision | Deep Learning | GenAI

GitHub

https://github.com/solo938

LinkedIn

https://www.linkedin.com/in/sahariar-hasan-b81885194
