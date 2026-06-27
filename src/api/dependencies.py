# src/api/dependencies.py
"""
Dependency injection for FastAPI.
"""

from typing import Optional
from fastapi import HTTPException

from src.inference.predictor import PCBDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global detector instance
_detector: Optional[PCBDetector] = None


def get_detector() -> PCBDetector:
    """
    Dependency injection for detector.
    Returns the global detector instance.
    """
    global _detector
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return _detector


def set_detector(detector: PCBDetector) -> None:
    """
    Set the global detector instance.
    Called during application startup.
    """
    global _detector
    _detector = detector
    logger.info(f"Detector set: {detector.weights_path} ({detector.model_type})")