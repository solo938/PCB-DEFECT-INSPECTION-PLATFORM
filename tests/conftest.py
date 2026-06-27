# tests/conftest.py
import pytest
from pathlib import Path
import cv2
import numpy as np

from src.inference.predictor import PCBDetector
from src.api.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def sample_image_path():
    """Path to a sample test image."""
    path = Path("data/processed/test/images/20085147_test.jpg")
    if not path.exists():
        # Create a dummy image if real one doesn't exist
        img = np.zeros((480, 480, 3), dtype=np.uint8)
        cv2.imwrite("test_dummy.jpg", img)
        return Path("test_dummy.jpg")
    return path


@pytest.fixture
def detector():
    """PCBDetector fixture using test weights."""
    # Use a small test model or mock for speed
    from src.inference.predictor import PCBDetector
    weights = "runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt"
    if not Path(weights).exists():
        pytest.skip("Model weights not found")
    return PCBDetector(weights_path=weights, device="cpu", conf_threshold=0.45)


@pytest.fixture
def test_client():
    """FastAPI test client."""
    app = create_app(
        weights_path="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        device="cpu",
    )
    return TestClient(app)