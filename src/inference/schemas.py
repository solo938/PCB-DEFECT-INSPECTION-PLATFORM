# src/inference/schemas.py
"""
Pydantic schemas for inference input/output validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BBox(BaseModel):
    """Bounding box in xyxy format."""
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    image_name: str
    timestamp: str
    num_detections: int
    inference_time_ms: float
    detections: List[Detection]


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction."""
    total_images: int
    total_detections: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_type: str
    timestamp: str


class MetadataResponse(BaseModel):
    """Model metadata response."""
    model_path: str
    model_type: str
    device: str
    num_classes: int
    class_names: Dict[int, str]
    image_size: int
    confidence_threshold: float
    iou_threshold: float