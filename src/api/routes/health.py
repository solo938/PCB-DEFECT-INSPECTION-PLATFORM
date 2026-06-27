# src/api/routes/health.py
"""
Health check endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse
from src.api.dependencies import get_detector
from src.inference.predictor import PCBDetector

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check(detector: PCBDetector = Depends(get_detector)):
    """
    Health check endpoint.
    Returns service status and model information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=detector._is_loaded,
        device=detector.device,
        model_type=detector.model_type,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/api/v1/health/ready")
async def readiness_check(detector: PCBDetector = Depends(get_detector)):
    """
    Readiness probe for Kubernetes.
    """
    return {
        "status": "ready",
        "model_loaded": detector._is_loaded,
    }


@router.get("/api/v1/health/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes.
    """
    return {"status": "alive"}