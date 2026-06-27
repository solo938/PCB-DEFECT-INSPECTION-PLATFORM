# src/api/routes/predict.py
"""
Prediction endpoints with image upload, URL, and batch support.
"""

import io
import time
import asyncio
from typing import List
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import aiohttp
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import Response

from src.api.schemas import PredictionResponse, BatchPredictionResponse, Detection, BBox, ErrorResponse
from src.api.dependencies import get_detector
from src.inference.predictor import PCBDetector
from src.inference.visualize import draw_detections
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["predict"])


async def download_image_from_url(url: str) -> np.ndarray:
    """Download image from URL and return as RGB numpy array."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {response.status}")
            
            content = await response.read()
            nparr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to decode image from URL")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@router.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    detector: PCBDetector = Depends(get_detector),
):
    """
    Detect defects in a single uploaded image.
    
    Returns:
        PredictionResponse with detections and metadata
    """
    start_time = time.perf_counter()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image, got: {file.content_type}"
        )
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.predict(image_rgb)
        
        detections = [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=BBox(x1=d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3]),
            )
            for d in result["detections"]
        ]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            image_name=file.filename,
            timestamp=result["timestamp"],
            num_detections=result["num_detections"],
            inference_time_ms=elapsed_ms,
            detections=detections,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/api/v1/predict/url", response_model=PredictionResponse)
async def predict_image_url(
    url: str,
    detector: PCBDetector = Depends(get_detector),
):
    """
    Detect defects in an image from a URL.
    
    Returns:
        PredictionResponse with detections and metadata
    """
    start_time = time.perf_counter()
    
    try:
        image_rgb = await download_image_from_url(url)
        result = detector.predict(image_rgb)
        
        detections = [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=BBox(x1=d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3]),
            )
            for d in result["detections"]
        ]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            image_name=url.split("/")[-1],
            timestamp=result["timestamp"],
            num_detections=result["num_detections"],
            inference_time_ms=elapsed_ms,
            detections=detections,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    detector: PCBDetector = Depends(get_detector),
):
    """
    Detect defects in multiple uploaded images.
    
    Returns:
        BatchPredictionResponse with results for each image
    """
    results = []
    total_detections = 0
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.predict(image_rgb)
            
            detections = [
                Detection(
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=BBox(x1=d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3]),
                )
                for d in result["detections"]
            ]
            
            total_detections += len(detections)
            
            results.append(PredictionResponse(
                image_name=file.filename,
                timestamp=result["timestamp"],
                num_detections=result["num_detections"],
                inference_time_ms=result["inference_time_ms"],
                detections=detections,
            ))
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            continue
    
    return BatchPredictionResponse(
        total_images=len(results),
        total_detections=total_detections,
        results=results,
    )


@router.post("/api/v1/predict/annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    detector: PCBDetector = Depends(get_detector),
):
    """
    Detect defects and return an annotated image with bounding boxes.
    
    Returns:
        JPEG image with bounding boxes drawn
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image, got: {file.content_type}"
        )
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.predict(image_rgb)
        
        annotated = draw_detections(image_rgb, result["detections"])
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        _, img_encoded = cv2.imencode(".jpg", annotated_bgr)
        
        return Response(
            content=img_encoded.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Detections": str(result["num_detections"]),
                "X-Inference-Time": f"{result['inference_time_ms']:.2f}ms",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Annotated prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")