# tests/test_api.py
import pytest
from pathlib import Path


class TestHealth:
    def test_health_endpoint(self, test_client):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_metadata_endpoint(self, test_client):
        response = test_client.get("/api/v1/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "num_classes" in data
        assert data["num_classes"] == 6


class TestPrediction:
    def test_predict_image(self, test_client, sample_image_path):
        with open(sample_image_path, "rb") as f:
            response = test_client.post(
                "/api/v1/predict",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        data = response.json()
        assert "num_detections" in data
        assert "detections" in data

    def test_predict_invalid_file(self, test_client):
        response = test_client.post(
            "/api/v1/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_url_invalid(self, test_client):
        response = test_client.post(
            "/api/v1/predict/url?url=https://example.com/invalid.jpg"
        )
        assert response.status_code == 400