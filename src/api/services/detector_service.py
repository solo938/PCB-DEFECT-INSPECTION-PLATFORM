# src/api/services/detector_service.py
"""
Detector service wrapper with warmup and background tasks.
"""

import numpy as np

from src.inference.predictor import PCBDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def warmup_detector(detector: PCBDetector) -> None:
    """
    Warm up the detector with dummy inference.
    This ensures the first real inference is not slowed by initialization.
    """
    logger.info("Warming up detector...")
    dummy = np.zeros((detector.img_size, detector.img_size, 3), dtype=np.uint8)
    detector.predict(dummy)
    logger.info("Detector warmed up successfully")