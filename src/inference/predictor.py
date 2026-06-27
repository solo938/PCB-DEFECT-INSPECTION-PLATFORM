# src/inference/predictor.py
"""
Core inference engine for PCB defect detection.
Single source of truth for all inference operations.
"""

import time
import json
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.paths import OUTPUTS_DIR
from src.utils.config import YOLO_CLASS_ID_TO_NAME, DEFAULT_CONF_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PCBDetector:
    """
    Core inference engine for PCB defect detection.
    
    This is the single source of truth for all inference.
    CLI, API, and batch inference all use this class.
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.3,
        device: str = "mps",
        img_size: int = 480,
    ):
        """
        Initialize the detector.
        
        Args:
            weights_path: Path to model weights (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS (default: 0.3 for aggressive suppression)
            device: Device to run inference on
            img_size: Image size for inference
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.class_names = YOLO_CLASS_ID_TO_NAME
        
        # Determine model type
        self.model_type = "onnx" if self.weights_path.suffix == ".onnx" else "pytorch"
        
        # Load model
        self.model = self._load_model()
        self._is_loaded = True
        
        logger.info(f"Detector initialized:")
        logger.info(f"  Model: {self.weights_path} ({self.model_type})")
        logger.info(f"  Device: {device}")
        logger.info(f"  Image size: {img_size}")
        logger.info(f"  Conf threshold: {conf_threshold}")
        logger.info(f"  IoU threshold: {iou_threshold}")
    
    def _load_model(self):
        """Load the model from weights path."""
        if self.model_type == "onnx":
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ort.InferenceSession(str(self.weights_path), providers=providers)
        else:
            return YOLO(str(self.weights_path))
    
    def predict(self, image: Union[str, Path, np.ndarray]) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: Image path or numpy array (RGB)
        
        Returns:
            Dict with detections and metadata
        """
        start_time = time.perf_counter()
        
        # Get image name
        if isinstance(image, (str, Path)):
            image_name = Path(image).name
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_name = "unknown"
            img = image
        
        # Run inference
        if self.model_type == "onnx":
            detections = self._predict_onnx(img)
        else:
            detections = self._predict_pytorch(img)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "image_name": image_name,
            "timestamp": datetime.now().isoformat(),
            "num_detections": len(detections),
            "inference_time_ms": elapsed_ms,
            "detections": detections,
        }
    
    def _predict_pytorch(self, image: np.ndarray) -> List[Dict]:
        """Run inference using PyTorch model."""
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        return self._parse_pytorch_results(results[0])
    
    def _predict_onnx(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference using ONNX model.
        Handles multiple output formats:
        - End2End: (1, 10, 4725) where 10 = [x1, y1, x2, y2, conf, class_id, ...]
        - Standard: (num_detections, 6) where 6 = [x1, y1, x2, y2, conf, class_id]
        - Raw: (1, 84, 8400) or (84, 8400) where 84 = [cx, cy, w, h, scores...]
        """
        h, w = image.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: img_batch})
        
        # Parse outputs based on format
        output = outputs[0]
        
        # Debug: log output shape
        logger.debug(f"ONNX output shape: {output.shape}")
        
        # Try different parsing strategies based on shape
        # End2End format: (1, 10, N) where N = number of anchors
        if len(output.shape) == 3 and output.shape[0] == 1 and output.shape[1] == 10:
            logger.debug("Detected End2End format (1, 10, N)")
            return self._parse_onnx_end2end(output[0], w, h)
        # End2End format: (10, N) without batch dimension
        elif len(output.shape) == 2 and output.shape[0] == 10:
            logger.debug("Detected End2End format (10, N)")
            return self._parse_onnx_end2end(output, w, h)
        # End2End format: (N, 10) transposed
        elif len(output.shape) == 2 and output.shape[1] == 10:
            logger.debug("Detected End2End format (N, 10)")
            return self._parse_onnx_end2end(output.T, w, h)
        # Raw YOLO output: (1, 84, 8400)
        elif len(output.shape) == 3 and output.shape[0] == 1 and output.shape[1] == 84:
            logger.debug("Detected Raw format (1, 84, 8400)")
            return self._parse_onnx_raw(output, w, h)
        # Raw YOLO output: (84, 8400)
        elif len(output.shape) == 2 and output.shape[0] == 84:
            logger.debug("Detected Raw format (84, 8400)")
            return self._parse_onnx_raw(np.expand_dims(output, 0), w, h)
        # Standard output: (1, num_detections, 6)
        elif len(output.shape) == 3 and output.shape[2] == 6:
            logger.debug("Detected Standard format (1, N, 6)")
            return self._parse_onnx_standard(output[0], w, h)
        # Standard output: (num_detections, 6)
        elif len(output.shape) == 2 and output.shape[1] == 6:
            logger.debug("Detected Standard format (N, 6)")
            return self._parse_onnx_standard(output, w, h)
        else:
            # Try generic parsing
            logger.warning(f"Unknown ONNX output shape: {output.shape}, trying generic parse")
            return self._parse_onnx_generic(output, w, h)
    
    def _parse_onnx_end2end(self, output: np.ndarray, img_w: int, img_h: int) -> List[Dict]:
        """
        Parse ONNX output from YOLO with end2end/NMS.
        Shape: (10, N) where:
        - Rows: [x1, y1, x2, y2, conf, class_id, ...]
        - Columns: detections (N = number of anchors, e.g., 4725)
        """
        logger.debug(f"Parsing End2End output with shape: {output.shape}")
        
        # Transpose to (N, 10)
        output = output.T
        
        if len(output) == 0:
            return []
        
        # Filter by confidence
        confs = output[:, 4]
        mask = confs > self.conf_threshold
        
        if not np.any(mask):
            logger.debug("No detections above confidence threshold")
            return []
        
        filtered = output[mask]
        logger.debug(f"Filtered {len(filtered)} detections above threshold")
        
        # Sort by confidence descending
        sorted_idx = np.argsort(filtered[:, 4])[::-1]
        filtered = filtered[sorted_idx]
        
        # Determine scaling
        scale_x = img_w / self.img_size
        scale_y = img_h / self.img_size
        
        # Build raw detections list
        raw_detections = []
        for det in filtered:
            x1 = det[0] * scale_x
            y1 = det[1] * scale_y
            x2 = det[2] * scale_x
            y2 = det[3] * scale_y
            conf = det[4]
            class_id = int(det[5]) if len(det) > 5 else 0
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h))
            y2 = max(0, min(y2, img_h))
            
            raw_detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": class_id,
            })
        
        logger.debug(f"Built {len(raw_detections)} raw detections")
        
        # Apply aggressive NMS
        nms_detections = self._apply_aggressive_nms(raw_detections, self.iou_threshold)
        logger.debug(f"After NMS: {len(nms_detections)} detections")
        
        # Format results
        detections = []
        for det in nms_detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = self.class_names.get(det["class_id"], f"class_{det['class_id']}")
            
            detections.append({
                "class_id": det["class_id"],
                "class_name": class_name,
                "confidence": det["confidence"],
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })
        
        return detections
    
    def _apply_aggressive_nms(self, detections: List[Dict], iou_threshold: float = 0.25) -> List[Dict]:
        """
        Apply aggressive Non-Maximum Suppression to remove duplicate detections.
        Uses a very low IoU threshold and also checks centroid distance.
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence descending
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        kept = []
        
        for det in detections:
            keep = True
            for kept_det in kept:
                # Compute IoU
                iou = self._compute_iou(det["bbox"], kept_det["bbox"])
                # Also compute centroid distance
                cx1 = (det["bbox"][0] + det["bbox"][2]) / 2
                cy1 = (det["bbox"][1] + det["bbox"][3]) / 2
                cx2 = (kept_det["bbox"][0] + kept_det["bbox"][2]) / 2
                cy2 = (kept_det["bbox"][1] + kept_det["bbox"][3]) / 2
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                # If IoU > threshold OR centroids are very close (< 5px), suppress
                if iou > iou_threshold or dist < 5.0:
                    keep = False
                    break
            if keep:
                kept.append(det)
        
        # Keep only top 2 detections per class to avoid duplicates
        class_counts = {}
        final_kept = []
        for det in kept:
            class_id = det["class_id"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            if class_counts[class_id] <= 2:  # Keep max 2 per class
                final_kept.append(det)
        
        return final_kept
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two boxes in [x1, y1, x2, y2] format.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _parse_onnx_standard(self, output: np.ndarray, img_w: int, img_h: int) -> List[Dict]:
        """
        Parse standard YOLO ONNX output: (num_detections, 6).
        6 = [x1, y1, x2, y2, confidence, class_id]
        """
        detections = []
        
        if len(output) == 0:
            return detections
        
        # Filter by confidence
        mask = output[:, 4] > self.conf_threshold
        filtered = output[mask]
        
        if len(filtered) == 0:
            return detections
        
        # Sort by confidence descending
        sorted_idx = np.argsort(filtered[:, 4])[::-1]
        filtered = filtered[sorted_idx]
        
        # Determine scaling
        scale_x = img_w / self.img_size
        scale_y = img_h / self.img_size
        
        for det in filtered:
            x1, y1, x2, y2, conf, class_id = det[:6]
            
            # Scale back to original image size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h))
            y2 = max(0, min(y2, img_h))
            
            class_id = int(class_id)
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
            })
        
        return detections
    
    def _parse_onnx_raw(self, output: np.ndarray, img_w: int, img_h: int) -> List[Dict]:
        """
        Parse raw YOLO ONNX output: (1, 84, 8400)
        Where 84 = [cx, cy, w, h, class_scores...]
        """
        detections = []
        
        # Output shape: (1, 84, 8400) -> (8400, 84)
        if len(output.shape) == 3:
            output = output[0]  # (84, 8400)
        
        # Transpose to (8400, 84)
        output = output.T  # (8400, 84)
        
        # Extract boxes (cx, cy, w, h) and scores
        boxes = output[:, :4]  # (8400, 4)
        scores = output[:, 4:]  # (8400, 80)
        
        # Apply sigmoid to scores
        scores = 1 / (1 + np.exp(-scores))
        
        # Get max scores and class ids
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter by confidence
        mask = max_scores > self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]
        
        if len(filtered_boxes) == 0:
            return detections
        
        # Sort by confidence descending
        sorted_idx = np.argsort(filtered_scores)[::-1]
        filtered_boxes = filtered_boxes[sorted_idx]
        filtered_scores = filtered_scores[sorted_idx]
        filtered_classes = filtered_classes[sorted_idx]
        
        for box, conf, class_id in zip(filtered_boxes, filtered_scores, filtered_classes):
            # Convert cx, cy, w, h to x1, y1, x2, y2
            cx, cy, w, h = box
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h))
            y2 = max(0, min(y2, img_h))
            
            class_id = int(class_id)
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
            })
        
        return detections
    
    def _parse_onnx_generic(self, output: np.ndarray, img_w: int, img_h: int) -> List[Dict]:
        """
        Generic parser for unknown ONNX output formats.
        Tries to infer the format and parse accordingly.
        """
        detections = []
        
        # Flatten if needed
        if len(output.shape) == 3 and output.shape[0] == 1:
            output = output[0]
        
        # Try to detect format
        if len(output.shape) == 2:
            # If second dimension is 6, treat as standard format
            if output.shape[1] == 6:
                return self._parse_onnx_standard(output, img_w, img_h)
            # If first dimension is 6, transpose and parse
            elif output.shape[0] == 6:
                return self._parse_onnx_standard(output.T, img_w, img_h)
            # If second dimension is 84, treat as raw format
            elif output.shape[1] == 84:
                return self._parse_onnx_raw(np.expand_dims(output.T, 0), img_w, img_h)
            # If first dimension is 84, transpose and parse raw
            elif output.shape[0] == 84:
                return self._parse_onnx_raw(np.expand_dims(output, 0), img_w, img_h)
            # If shape is (N, 10), try end2end parser
            elif output.shape[1] == 10:
                return self._parse_onnx_end2end(output.T, img_w, img_h)
            elif output.shape[0] == 10:
                return self._parse_onnx_end2end(output, img_w, img_h)
        
        # If we can't detect the format, try a simple approach
        logger.warning(f"Could not parse ONNX output with shape: {output.shape}")
        return detections
    
    def _parse_pytorch_results(self, result) -> List[Dict]:
        """Parse PyTorch YOLO results."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })
        
        return detections
    
    def predict_path(self, image_path: Union[str, Path]) -> Dict:
        """Predict from image path."""
        return self.predict(image_path)
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """Predict multiple images."""
        results = []
        for path in image_paths:
            result = self.predict(path)
            results.append(result)
        return results
    
    def predict_folder(
        self,
        folder_path: Union[str, Path],
        extensions: List[str] = [".jpg", ".jpeg", ".png"],
    ) -> List[Dict]:
        """Predict all images in a folder."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder.glob(f"*{ext}"))
        image_paths = sorted(image_paths)
        
        if not image_paths:
            logger.warning(f"No images found in {folder_path}")
            return []
        
        logger.info(f"Found {len(image_paths)} images in {folder_path}")
        return self.predict_batch(image_paths)
    
    def get_metadata(self) -> Dict:
        """Get model metadata."""
        return {
            "model_path": str(self.weights_path),
            "model_type": self.model_type,
            "device": self.device,
            "image_size": self.img_size,
            "confidence_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "is_loaded": self._is_loaded,
        }