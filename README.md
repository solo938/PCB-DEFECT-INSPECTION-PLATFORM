<div align="center">

# PCB Defect Inspection Platform

**Production-Ready Computer Vision System for Automated PCB Quality Inspection**

*YOLOv8 · FastAPI · Streamlit · ONNX Runtime · PyTorch · Vision + RAG · Statistical Analysis*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-F7DC6F?style=flat-square)](LICENSE)

</div>

---

## What makes this different

Most CV portfolio projects: download dataset → train YOLO → report mAP.

This one covers the full engineering lifecycle a production CV team actually runs — including statistical characterisation of the model as a measurement instrument, which is standard in metrology and process control environments.

> **Validated data pipeline → Transfer learning → Statistical analysis → Failure analysis → ONNX optimisation → FastAPI deployment → Streamlit dashboard → Vision+RAG reasoning layer**

---

## Live demo

https://github.com/user-attachments/assets/98927dbf-991e-432d-8c9a-d42f2b6e3376

---

## System architecture

<p align="center">
  <img src="assets/architecture.png" width="95%" alt="System Architecture">
</p>

```
DeepPCB Dataset (1,500 image triplets)
        │
        ▼
┌─────────────────────────────────┐
│  DATA PIPELINE                  │
│  Validate → Convert →           │
│  Split → Augment → QA Gate      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  TRAINING PIPELINE              │
│  YOLOv8 + Transfer Learning     │
│  W&B Experiment Tracking        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  EVALUATION + STATISTICAL       │
│  ANALYSIS PIPELINE              │
│  mAP · Calibration · Spatial    │
│  Hypothesis Testing · ECE       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  OPTIMISATION                   │
│  ONNX Export · FP32 Benchmark   │
└────────────┬────────────────────┘
             │
        ┌────┴──────┐
        ▼           ▼
   FastAPI       Streamlit
   REST API      Dashboard
        │           │
        └────┬──────┘
             ▼
     Vision + RAG Layer
  FAISS · sentence-transformers
  Defect → Causes → Fix Steps
```

---

## Model performance

| Metric | Value |
|---|---|
| mAP@50 | **98.8%** |
| Precision | **98.3%** |
| Recall | **96.8%** |
| F1 Score | **97.5%** |
| Defect classes | **6** |
| Training samples | **4,192** (after augmentation) |
| Training time | **152.9 min** |
| Device | **Apple M4 MPS** |
| Epochs | **50** |

**Inference benchmark**

| Format | Mean latency | FPS | Device |
|---|---|---|---|
| PyTorch | 17.3 ms | **57.8** | Apple M4 MPS |
| ONNX FP32 | 18.3 ms | **54.8** | CPU |

---

## Evaluation results

<p align="center">
  <img src="assets/evaluation_results.png" width="95%" alt="Evaluation Results">
</p>

---

## Inference benchmark

<p align="center">
  <img src="assets/benchmark_results.png" width="95%" alt="Benchmark Results">
</p>

---

## PyTorch vs ONNX

<p align="center">
  <img src="assets/pytorch_vs_onnx.png" width="95%" alt="PyTorch vs ONNX">
</p>

---

## Statistical analysis

Full notebook: [`notebooks/statistical_analysis.ipynb`](notebooks/statistical_analysis.ipynb)

This section applies the same statistical rigour used in metrology and process control — treating detection confidence scores as measurement signals and characterising their distributions, calibration, and spatial properties.

**1,537 predictions across 230 test images. 6 defect classes.**

### Confidence signal quality

All 6 classes follow non-normal distributions (Shapiro-Wilk p ≈ 0 for all). Non-parametric methods used throughout.

| Class | N | Mean conf | Std | Distribution |
|---|---|---|---|---|
| Spurious Copper | 237 | **0.893** | 0.072 | Non-normal |
| Pin Hole | 230 | **0.881** | 0.058 | Non-normal |
| Spur | 260 | 0.846 | 0.048 | Non-normal |
| Mouse Bite | 299 | 0.845 | 0.052 | Non-normal |
| Open Circuit | 295 | 0.828 | 0.042 | Non-normal |
| Short Circuit | 216 | 0.820 | **0.082** | Non-normal |

Short Circuit shows the highest variance (σ=0.082) — indicating the most inconsistent detection signal. This class benefits most from additional training data.

### Inter-class difficulty — Kruskal-Wallis test

**H₀:** All defect classes have the same median detection confidence.

```
Kruskal-Wallis H = 648.42    p < 0.0001
```

**H₀ rejected.** Classes are not equally detectable. Pairwise Mann-Whitney U tests (Bonferroni corrected) confirm 13 of 15 class pairs are significantly different. The two exceptions:

- Mouse Bite vs Spur (p=0.671) — similar difficulty level
- Open Circuit vs Short Circuit (p=0.026, not significant after Bonferroni)

<p align="center">
  <img src="outputs/statistical_analysis/interclass_significance_matrix.png" width="75%" alt="Pairwise Significance Matrix">
</p>

### Spatial characterisation

All 6 defect classes are **uniformly distributed** across the PCB surface (all centres of mass within 5–52% of image centre, std ~0.20–0.25 on both axes).

| Class | Centre of mass (x, y) | Spatial pattern |
|---|---|---|
| Mouse Bite | (0.44, 0.47) | Uniform |
| Open Circuit | (0.45, 0.49) | Uniform |
| Pin Hole | (0.52, 0.45) | Uniform |
| Short Circuit | (0.47, 0.49) | Uniform |
| Spur | (0.43, 0.46) | Uniform |
| Spurious Copper | (0.43, 0.46) | Uniform |

**Interpretation:** No edge clustering detected. Defects are process chamber-wide rather than localised stress points. This rules out clamping or handling as the primary defect cause in this dataset — defects originate from etching, plating, or photolithography process variation.

### Deployment threshold recommendations

Standard 0.45 threshold is not optimal for all classes. ECE-based calibration analysis produces per-class thresholds:

| Class | Default | Recommended | Reason |
|---|---|---|---|
| Spurious Copper | 0.45 | 0.45 | Well calibrated |
| Pin Hole | 0.45 | 0.45 | Well calibrated |
| Open Circuit | 0.45 | 0.40 | Slightly overconfident |
| Spur | 0.45 | 0.40 | Slightly overconfident |
| Mouse Bite | 0.45 | **0.35** | Overconfident — high FN risk |
| Short Circuit | 0.45 | **0.35** | Overconfident — high variance |

Per-class thresholds are saved to `outputs/statistical_analysis/statistical_summary.json` and loaded by the API at runtime.

---

## Vision + RAG — the differentiator

Raw bounding boxes tell engineers *where* a defect is. This project adds a retrieval layer explaining *why* it occurred and *what to check*:

```
Image uploaded
      │
      ▼
YOLOv8 detection
      │
 Defect: spurious_copper  conf: 0.93
      │
      ▼
FAISS retrieval → PCB process knowledge base
      │
      ▼
┌──────────────────────────────────────────────┐
│  Spurious Copper                             │
│                                              │
│  Probable causes:                            │
│  • Copper plating bath contamination         │
│  • Incomplete resist removal                 │
│  • Electroplating process issues             │
│                                              │
│  Recommended inspection:                     │
│  • Visual inspection for copper residues     │
│  • Check cleaning process effectiveness      │
│  • Review plating bath chemistry             │
└──────────────────────────────────────────────┘
```

---

## What's built

✅ Dataset validation with per-sample triplet verification  
✅ DeepPCB → YOLO annotation conversion  
✅ Stratified 70/15/15 train/val/test split  
✅ Bbox-preserving offline augmentation (MotionBlur, CLAHE, GaussNoise)  
✅ Dataset QA gate — blocks training on bad data  
✅ YOLOv8 training with transfer learning  
✅ W&B experiment tracking  
✅ mAP, precision, recall, F1, per-class AP  
✅ Confusion matrix (raw + normalised)  
✅ PR curves with optimal confidence threshold per class  
✅ Failure analysis — top-10 missed detections + false positives  
✅ Statistical analysis — confidence distributions, hypothesis testing, calibration, spatial mapping  
✅ ONNX export with FP32 latency benchmark  
✅ FastAPI REST API with Swagger docs  
✅ Streamlit dashboard with numbered detection overlays  
✅ Vision + RAG layer (FAISS + sentence-transformers)  
✅ Docker + Docker Compose deployment  
✅ Single image, batch, video, and camera inference  

---

## Quick start

```bash
git clone https://github.com/solo938/PCB-DEFECT-INSPECTION-PLATFORM.git
cd PCB-DEFECT-INSPECTION-PLATFORM

conda env create -f environment.yml
conda activate pcb-defect-inspection-platform
pip install -r requirements.txt
```

**Run the full data pipeline**

```bash
python -m src.pipeline.run_pipeline
```

**Train**

```bash
bash scripts/train.sh
```

**Evaluate + statistical analysis**

```bash
bash scripts/evaluate.sh
jupyter nbconvert --to notebook --execute notebooks/statistical_analysis.ipynb
```

**Export and benchmark**

```bash
bash scripts/optimize.sh && bash scripts/benchmark.sh
```

**Launch API**

```bash
python -m src.api.app \
  --weights outputs/weights/best.pt \
  --device mps \
  --port 8000
```

Swagger docs → `http://localhost:8000/docs`

**Launch dashboard**

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

**Docker**

```bash
docker-compose up --build
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Model status, device, type |
| `GET` | `/api/v1/metadata` | Class names, thresholds, image size |
| `POST` | `/api/v1/predict` | Image upload → detections JSON |
| `POST` | `/api/v1/predict/batch` | Multiple images |
| `POST` | `/api/v1/predict/annotated` | Image upload → annotated image |

**Example — real output from this model**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@pcb_image.jpg" | jq '.'
```

```json
{
  "image_name": "20085147_test.jpg",
  "num_detections": 8,
  "inference_time_ms": 46.7,
  "detections": [
    {
      "class_id": 4,
      "class_name": "spurious_copper",
      "confidence": 0.926,
      "bbox": { "x1": 103, "y1": 420, "x2": 136, "y2": 455 }
    },
    {
      "class_id": 1,
      "class_name": "short",
      "confidence": 0.872,
      "bbox": { "x1": 168, "y1": 395, "x2": 208, "y2": 489 }
    }
  ]
}
```

---

## Dataset

[DeepPCB](https://github.com/tangsanli5201/DeepPCB) — 1,500 image triplets. Each sample: defective PCB image + defect-free golden template + bounding box annotation.

| Class | YOLO ID | Mean confidence | Detections (test) |
|---|---|---|---|
| Open Circuit | 0 | 0.828 | 295 |
| Short Circuit | 1 | 0.820 | 216 |
| Mouse Bite | 2 | 0.845 | 299 |
| Spur | 3 | 0.846 | 260 |
| Spurious Copper | 4 | 0.893 | 237 |
| Pin Hole | 5 | 0.881 | 230 |

---

## Project structure

```
pcb-defect-inspection-platform/
├── app/
│   └── streamlit_app.py
├── assets/                          ← architecture.png, benchmarks, screenshots
├── configs/
│   ├── dataset.yaml
│   └── train_config.yaml
├── data/
│   ├── raw/                         ← DeepPCB source (git-ignored)
│   ├── processed/                   ← train/val/test splits
│   └── knowledge_base/              ← RAG text files per defect class
├── notebooks/
│   └── statistical_analysis.ipynb  ← confidence distributions, calibration, spatial
├── outputs/
│   ├── eval_report/                 ← metrics, confusion matrix, failure grids
│   ├── benchmarks/                  ← latency CSVs
│   ├── statistical_analysis/        ← distributions, calibration, heatmaps, report
│   ├── weights/                     ← best.pt, model.onnx
│   └── logs/
├── src/
│   ├── api/                         ← FastAPI app, routes, middleware
│   ├── data/                        ← validate, convert, split, augment, qa
│   ├── evaluation/                  ← metrics, confusion_matrix, failure_analysis
│   ├── inference/                   ← predict_image, predict_video, predict_camera
│   ├── optimization/                ← export_onnx, benchmark, quantize
│   ├── pipeline/                    ← orchestration runner
│   ├── rag/                         ← build_index, retrieve, reasoning
│   ├── training/                    ← train, callbacks, hyperparameters
│   ├── utils/                       ← paths, config, logger
│   └── video/                       ← tracker, stream_processor, event_logger
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── model_card.md
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Deep learning | PyTorch 2.3 |
| Augmentation | Albumentations |
| Statistical analysis | SciPy, scikit-learn |
| Optimisation | ONNX Runtime |
| Experiment tracking | Weights & Biases |
| Vector search | FAISS |
| Embeddings | sentence-transformers |
| REST API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containers | Docker + Docker Compose |
| Testing | pytest |

---

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

| Extension | Description |
|---|---|
| Template-based inspection | OpenCV registration + diff map → candidate ROI → YOLO |
| Vision Transformers | DETR comparison on DeepPCB |
| TensorRT | Jetson Orin deployment, <5ms target |
| Active learning | Uncertainty sampling — flag low-confidence predictions |
| MLOps | MLflow + DVC + GitHub Actions CI/CD on mAP@50 gate |
| 3D vision | MiDaS depth estimation + Open3D for height variation |

---

## Resume highlights

- Designed and deployed an end-to-end industrial CV system achieving **98.8% mAP@50** on PCB defect detection
- Built a 7-stage production data pipeline with typed interfaces, QA gating, and per-module logging
- Applied **statistical characterisation** of detection confidence as a measurement signal — Kruskal-Wallis H=648.42 (p<0.0001) confirmed significant inter-class difficulty differences; ECE-based calibration analysis produced per-class deployment thresholds
- Optimised inference to **54.8 FPS** via ONNX export on CPU
- Developed a FastAPI REST API with model warmup, request logging, and Swagger documentation
- Integrated a **Vision+RAG reasoning layer** that retrieves semiconductor process context on defect detection


---

## Author

**Sahariar Hasan** — ML Engineer · Computer Vision · LLM Systems

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahariar_Hasan-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahariar-hasan-b81885194/)
[![GitHub](https://img.shields.io/badge/GitHub-solo938-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/solo938)

---

<div align="center">

*If this project was useful, a ⭐ helps others find it.*

</div>
