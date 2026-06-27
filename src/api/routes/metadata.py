# src/api/routes/metadata.py
"""
Model metadata endpoints.
"""

from fastapi import APIRouter, Depends

from src.api.schemas import ModelInfoResponse
from src.api.dependencies import get_detector
from src.inference.predictor import PCBDetector

router = APIRouter(tags=["metadata"])


@router.get("/api/v1/metadata", response_model=ModelInfoResponse)
async def get_metadata(detector: PCBDetector = Depends(get_detector)):
    """
    Get model metadata including class names and configuration.
    """
    meta = detector.get_metadata()
    return ModelInfoResponse(
        model_path=meta["model_path"],
        model_type=meta["model_type"],
        device=meta["device"],
        num_classes=meta["num_classes"],
        class_names=meta["class_names"],
        image_size=meta["image_size"],
        confidence_threshold=meta["confidence_threshold"],
        iou_threshold=meta["iou_threshold"],
    )


@router.get("/api/v1/classes")
async def get_classes(detector: PCBDetector = Depends(get_detector)):
    """
    Get class names mapping.
    """
    return {
        "num_classes": len(detector.class_names),
        "classes": detector.class_names,
    }