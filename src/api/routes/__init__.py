# src/api/routes/__init__.py
"""
API routes module.
"""

from src.api.routes.health import router as health_router
from src.api.routes.metadata import router as metadata_router
from src.api.routes.predict import router as predict_router

__all__ = [
    "health_router",
    "metadata_router",
    "predict_router",
]