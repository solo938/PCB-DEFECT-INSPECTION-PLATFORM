# src/inference/__init__.py
"""
Inference module for PCB defect detection.

Core components:
- PCBDetector: Main inference engine
- Visualization utilities
- Batch processing
- CLI interfaces
"""

from src.inference.predictor import PCBDetector
from src.inference.visualize import draw_detections, save_annotated_image
from src.inference.batch import run_batch_inference
from src.inference.schemas import (
    Detection,
    BBox,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
)

__all__ = [
    "PCBDetector",
    "draw_detections",
    "save_annotated_image",
    "run_batch_inference",
    "Detection",
    "BBox",
    "PredictionResponse",
    "BatchPredictionResponse",
    "HealthResponse",
    "MetadataResponse",
]