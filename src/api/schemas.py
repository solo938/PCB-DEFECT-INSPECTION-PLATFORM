# src/api/schemas.py
"""
Pydantic schemas for API request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BBox(BaseModel):
    """Bounding box in xyxy format."""
    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")


class Detection(BaseModel):
    """Single detection result."""
    class_id: int = Field(..., description="Class ID (0-indexed)")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BBox = Field(..., description="Bounding box coordinates")


class PredictionResponse(BaseModel):
    """Response from single image prediction."""
    image_name: str = Field(..., description="Original image filename")
    timestamp: str = Field(..., description="ISO timestamp")
    num_detections: int = Field(..., description="Number of detections found")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    detections: List[Detection] = Field(default_factory=list, description="List of detections")


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction."""
    total_images: int = Field(..., description="Total images processed")
    total_detections: int = Field(..., description="Total detections found")
    results: List[PredictionResponse] = Field(..., description="Per-image results")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    model_type: str = Field(..., description="Model format (pytorch/onnx)")
    timestamp: str = Field(..., description="ISO timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_path: str = Field(..., description="Path to model weights")
    model_type: str = Field(..., description="Model format")
    device: str = Field(..., description="Device for inference")
    num_classes: int = Field(..., description="Number of classes")
    class_names: Dict[int, str] = Field(..., description="Class ID to name mapping")
    image_size: int = Field(..., description="Input image size")
    confidence_threshold: float = Field(..., description="Confidence threshold")
    iou_threshold: float = Field(..., description="IoU threshold for NMS")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="ISO timestamp")