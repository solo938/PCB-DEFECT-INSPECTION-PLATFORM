# src/evaluation/core.py
"""
Core evaluation module. Runs model.val() once and saves all raw results.
All other evaluation modules read from these saved files.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import time
import numpy as np

from ultralytics import YOLO

from src.utils.paths import OUTPUTS_DIR, DATASET_YAML_PATH
from src.utils.config import NUM_CLASSES, YOLO_CLASS_ID_TO_NAME
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Output directories
EVAL_REPORT_DIR = OUTPUTS_DIR / "eval_report"
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def run_evaluation(
    weights_path: str,
    dataset_yaml: Path = DATASET_YAML_PATH,
    split: str = "test",
    img_size: int = 480,
    conf: float = 0.001,
    iou: float = 0.45,
    batch: int = 1,
    device: str = "mps",
) -> Dict:
    """
    Run YOLO evaluation and save all raw results.
    
    Args:
        weights_path: Path to best.pt
        dataset_yaml: Path to dataset.yaml
        split: 'train', 'val', or 'test'
        img_size: Image size for evaluation
        conf: Confidence threshold
        iou: IoU threshold for NMS
        batch: Batch size for inference
        device: Device to run on
    
    Returns:
        Dict with evaluation results
    """
    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    logger.info(f"Evaluating on {split} set...")
    
    # Run validation
    results = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        batch=batch,
        device=device,
        plots=False,
        save_json=False,
        save_txt=False,
    )
    
    # ─────────────────────────────────────────────
    # Safely extract metrics - handle API differences
    # ─────────────────────────────────────────────
    
    # Debug: print available attributes
    logger.info(f"Results box attributes: {[a for a in dir(results.box) if not a.startswith('_')]}")
    
    # Overall metrics - safe extraction
    map_value = float(results.box.map) if hasattr(results.box, 'map') else 0.0
    map50_value = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
    map75_value = float(results.box.map75) if hasattr(results.box, 'map75') else 0.0
    
    # Precision and recall - handle array vs scalar
    if hasattr(results.box, 'mp'):
        precision = float(results.box.mp)
    elif hasattr(results.box, 'p') and results.box.p is not None:
        if isinstance(results.box.p, (list, np.ndarray)):
            precision = float(np.mean(results.box.p))
        else:
            precision = float(results.box.p)
    else:
        precision = 0.0
    
    if hasattr(results.box, 'mr'):
        recall = float(results.box.mr)
    elif hasattr(results.box, 'r') and results.box.r is not None:
        if isinstance(results.box.r, (list, np.ndarray)):
            recall = float(np.mean(results.box.r))
        else:
            recall = float(results.box.r)
    else:
        recall = 0.0
    
    # Calculate F1 from precision and recall
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Per-class metrics
    maps = []
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        maps = [float(x) for x in results.box.maps]
    
    # Get class indices
    ap_class_index = []
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        ap_class_index = [int(x) for x in results.box.ap_class_index]
    
    # Build per-class metrics
    per_class = {}
    class_names = list(YOLO_CLASS_ID_TO_NAME.values())
    
    for i, name in enumerate(class_names):
        per_class[name] = {
            "ap": maps[i] if i < len(maps) else 0.0,
            "ap50": 0.0,  # Not reliably exposed in recent versions
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    # Build metrics dict
    metrics = {
        "map": map_value,
        "map50": map50_value,
        "map75": map75_value,
        "maps": maps,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap_class_index": ap_class_index,
        "per_class": per_class,
        "speed": results.speed if hasattr(results, 'speed') else {},
    }
    
    # Extract confusion matrix if available
    confusion_matrix = None
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        if hasattr(results.confusion_matrix, 'matrix'):
            confusion_matrix = results.confusion_matrix.matrix.tolist()
    
    # Build full result dict
    full_result = {
        "weights_path": str(weights_path),
        "dataset_yaml": str(dataset_yaml),
        "split": split,
        "img_size": img_size,
        "conf": conf,
        "iou": iou,
        "device": device,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "speed": results.speed if hasattr(results, 'speed') else {},
    }
    
    # Save results
    save_json(full_result, EVAL_REPORT_DIR / "evaluation_results.json")
    save_json(metrics, EVAL_REPORT_DIR / "metrics.json")
    
    logger.info(f"Results saved to {EVAL_REPORT_DIR}")
    logger.info(f"mAP@50:    {map50_value:.4f}")
    logger.info(f"mAP@50-95: {map_value:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1:        {f1:.4f}")
    
    return full_result


def load_evaluation_results() -> Dict:
    """Load previously saved evaluation results."""
    with open(EVAL_REPORT_DIR / "evaluation_results.json", "r") as f:
        return json.load(f)


def print_results_table(metrics: Dict) -> None:
    """Print evaluation results as a formatted table."""
    print("\n" + "=" * 60)
    print("PCB DEFECT DETECTION — EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@50:      {metrics.get('map50', 0):.4f}")
    print(f"mAP@50-95:   {metrics.get('map', 0):.4f}")
    print(f"Precision:   {metrics.get('precision', 0):.4f}")
    print(f"Recall:      {metrics.get('recall', 0):.4f}")
    print(f"F1:          {metrics.get('f1', 0):.4f}")
    print("=" * 60)
    
    # Per-class table
    print("\nPer-class results:")
    print(f"{'Class':<18} {'AP':<10}")
    print("-" * 30)
    
    per_class = metrics.get('per_class', {})
    for name, stats in sorted(per_class.items()):
        ap = stats.get('ap', 0)
        print(f"{name:<18} {ap:<10.4f}")
    
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, 
                       default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML_PATH))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    
    results = run_evaluation(
        weights_path=args.weights,
        dataset_yaml=Path(args.data),
        split=args.split,
        img_size=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
    )
    
    print_results_table(results["metrics"])


if __name__ == "__main__":
    main()