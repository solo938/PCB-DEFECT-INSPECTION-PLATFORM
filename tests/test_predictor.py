# tests/test_predictor.py
import numpy as np
from pathlib import Path


class TestPCBDetector:
    def test_initialization(self, detector):
        assert detector._is_loaded is True
        assert detector.model_type == "pytorch"

    def test_predict_image(self, detector, sample_image_path):
        result = detector.predict(sample_image_path)
        assert "num_detections" in result
        assert "detections" in result
        assert "inference_time_ms" in result

    def test_predict_numpy(self, detector):
        dummy = np.zeros((480, 480, 3), dtype=np.uint8)
        result = detector.predict(dummy)
        assert "num_detections" in result

    def test_metadata(self, detector):
        meta = detector.get_metadata()
        assert meta["num_classes"] == 6
        assert "class_names" in meta