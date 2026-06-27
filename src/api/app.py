# src/api/app.py
"""
FastAPI application for PCB defect detection.
"""

import argparse
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.inference.predictor import PCBDetector
from src.api.dependencies import set_detector
from src.api.middleware import RequestLoggingMiddleware, add_cors_middleware
from src.api.services.detector_service import warmup_detector
from src.api.routes.health import router as health_router
from src.api.routes.metadata import router as metadata_router
from src.api.routes.predict import router as predict_router
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads model on startup and warms it up.
    """
    logger.info("Starting up...")
    
    # Get detector from app state
    detector = app.state.detector
    
    # Warm up the detector
    warmup_detector(detector)
    
    yield
    
    logger.info("Shutting down...")


def create_app(
    weights_path: str = "runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.3,
    device: str = "mps",
    img_size: int = 480,
) -> FastAPI:
    """
    Create FastAPI application with detector.
    
    Args:
        weights_path: Path to model weights
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: Device for inference
        img_size: Image size for inference
    
    Returns:
        FastAPI app
    """
    # Initialize detector
    logger.info(f"Initializing detector with weights: {weights_path}")
    detector = PCBDetector(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
        img_size=img_size,
    )
    set_detector(detector)
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="PCB Defect Detection API",
        description="""
        YOLOv8-based PCB defect detection service.
        
        ## Features
        - Upload images for defect detection
        - URL-based image detection
        - Batch image processing
        - Annotated image download
        - Model metadata and health checks
        
        ## Defect Classes
        - missing_hole
        - mouse_bite
        - open_circuit
        - short
        - spur
        - copper
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Store detector in app state for lifespan
    app.state.detector = detector
    
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    app = add_cors_middleware(app)
    
    # Register routes
    app.include_router(health_router)
    app.include_router(metadata_router)
    app.include_router(predict_router)
    
    @app.get("/")
    async def root():
        return {
            "service": "PCB Defect Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "metadata": "/api/v1/metadata",
        }
    
    return app


def main() -> None:
    """Run the API server."""
    parser = argparse.ArgumentParser(description="PCB Defect Detection API")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to model weights (.pt or .onnx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="Image size for inference",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create app
    app = create_app(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        img_size=args.imgsz,
    )
    
    # Run server
    import uvicorn
    logger.info(f"Starting API on {args.host}:{args.port}")
    logger.info(f"Docs available at: http://localhost:{args.port}/docs")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()