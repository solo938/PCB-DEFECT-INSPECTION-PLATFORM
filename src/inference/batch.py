# src/inference/batch.py
"""
Batch inference for processing large datasets.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import cv2
from tqdm import tqdm

from src.inference.predictor import PCBDetector
from src.inference.visualize import draw_detections
from src.utils.paths import OUTPUTS_DIR
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def run_batch_inference(
    detector: PCBDetector,
    source_dir: Path,
    output_dir: Path,
    save_images: bool = True,
    save_json: bool = True,
    image_extensions: List[str] = [".jpg", ".jpeg", ".png"],
) -> Dict:
    """
    Run batch inference on a folder of images.
    
    Args:
        detector: PCBDetector instance
        source_dir: Directory containing images
        output_dir: Directory to save results
        save_images: Save annotated images
        save_json: Save predictions as JSON
        image_extensions: Valid image extensions
    
    Returns:
        Summary statistics
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(source_dir.glob(f"*{ext}"))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        logger.warning(f"No images found in {source_dir}")
        return {"total_images": 0, "total_detections": 0, "results": []}
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Create output subdirectories
    annotated_dir = output_dir / "annotated"
    json_dir = output_dir / "json"
    
    if save_images:
        annotated_dir.mkdir(exist_ok=True)
    if save_json:
        json_dir.mkdir(exist_ok=True)
    
    results = []
    total_detections = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        result = detector.predict(image)
        result["image_path"] = str(img_path)
        results.append(result)
        total_detections += result["num_detections"]
        
        # Save annotated image
        if save_images and result["num_detections"] > 0:
            annotated = draw_detections(image, result["detections"])
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            save_path = annotated_dir / f"{img_path.stem}_annotated{img_path.suffix}"
            cv2.imwrite(str(save_path), annotated_bgr)
        
        # Save JSON
        if save_json:
            json_path = json_dir / f"{img_path.stem}.json"
            with json_path.open("w") as f:
                json.dump(result, f, indent=2, default=str)
    
    # Save summary
    summary = {
        "total_images": len(results),
        "total_detections": total_detections,
        "avg_detections": total_detections / len(results) if results else 0,
        "avg_inference_ms": sum(r["inference_time_ms"] for r in results) / len(results) if results else 0,
        "results": results,
    }
    
    # Save master JSON
    master_json = output_dir / "batch_predictions.json"
    with master_json.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Batch inference complete: {len(results)} images, {total_detections} detections")
    logger.info(f"Results saved to {output_dir}")
    
    return summary


def main() -> None:
    """CLI entry point for batch inference."""
    parser = argparse.ArgumentParser(description="Batch inference on folder")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/batch_inference",
        help="Output directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold",
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
        help="Image size",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving annotated images",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving JSON predictions",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Initialize detector
    detector = PCBDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        img_size=args.imgsz,
    )
    
    # Run batch inference
    run_batch_inference(
        detector,
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        save_images=not args.no_save_images,
        save_json=not args.no_save_json,
    )


if __name__ == "__main__":
    main()