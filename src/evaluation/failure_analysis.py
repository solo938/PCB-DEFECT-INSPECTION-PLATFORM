# src/evaluation/failure_analysis.py
"""
Failure analysis: finds worst-performing images and visualizes them.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict

from ultralytics import YOLO

from src.utils.paths import PROCESSED_DIR
from src.utils.config import YOLO_CLASS_ID_TO_NAME
from src.utils.logger import get_logger
from src.evaluation.utils import EVAL_REPORT_DIR, compute_iou, get_class_name

logger = get_logger(__name__)


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Match predictions to ground truth boxes.
    
    Args:
        predictions: List of dicts with 'bbox', 'class_id', 'confidence'
        ground_truth: List of dicts with 'bbox', 'class_id'
        iou_threshold: IoU threshold for positive match
    
    Returns:
        Dict with matched counts and unmatched items
    """
    matched_gt = set()
    matched_pred = set()
    true_positives = []
    false_positives = []
    false_negatives = []
    
    for gt_idx, gt in enumerate(ground_truth):
        best_iou = 0
        best_pred_idx = -1
        
        for pred_idx, pred in enumerate(predictions):
            if pred_idx in matched_pred:
                continue
            
            iou = compute_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        
        if best_iou >= iou_threshold and best_pred_idx >= 0:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
            true_positives.append({
                "gt": gt,
                "pred": predictions[best_pred_idx],
                "iou": best_iou,
            })
        else:
            false_negatives.append({
                "gt": gt,
                "best_iou": best_iou,
                "best_pred": predictions[best_pred_idx] if best_pred_idx >= 0 else None,
            })
    
    # Remaining predictions are false positives
    for pred_idx, pred in enumerate(predictions):
        if pred_idx not in matched_pred:
            false_positives.append(pred)
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "n_gt": len(ground_truth),
        "n_pred": len(predictions),
        "n_tp": len(true_positives),
        "n_fp": len(false_positives),
        "n_fn": len(false_negatives),
    }


def analyze_test_set(
    weights_path: str,
    test_images_dir: Path,
    test_labels_dir: Path,
    conf: float = 0.25,
    iou: float = 0.45,
) -> List[Dict]:
    """
    Run inference on test set and analyze results.
    
    Returns:
        List of result dicts per image
    """
    model = YOLO(weights_path)
    
    results_list = []
    image_files = list(test_images_dir.glob("*_test.jpg"))
    
    for img_path in image_files:
        stem = img_path.stem.replace("_test", "")
        label_path = test_labels_dir / f"{stem}.txt"
        
        # Run inference
        results = model(img_path, conf=conf, iou=iou)
        predictions = results[0].boxes
        
        # Format predictions
        pred_list = []
        for box in predictions:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_list.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
            })
        
        # Load ground truth
        gt_list = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w/2) * 640
                        y1 = (cy - h/2) * 640
                        x2 = (cx + w/2) * 640
                        y2 = (cy + h/2) * 640
                        gt_list.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_id": class_id,
                        })
        
        # Match
        match_result = match_predictions_to_ground_truth(pred_list, gt_list, iou_threshold=0.5)
        
        results_list.append({
            "image_path": img_path,
            "stem": stem,
            "label_path": label_path,
            "predictions": pred_list,
            "ground_truth": gt_list,
            "match": match_result,
            "missed_detections": match_result["n_fn"],
            "false_positives": match_result["n_fp"],
            "confidences": [p["confidence"] for p in pred_list],
            "max_confidence": max([p["confidence"] for p in pred_list]) if pred_list else 0,
        })
    
    return results_list


def find_worst_failures(results: List[Dict], n: int = 10) -> Dict:
    """Find the worst failure cases."""
    # Sort by missed detections
    by_missed = sorted(results, key=lambda x: x["missed_detections"], reverse=True)[:n]
    
    # Sort by false positives
    by_fp = sorted(results, key=lambda x: x["false_positives"], reverse=True)[:n]
    
    # Sort by lowest confidence true positives (most uncertain correct detections)
    by_low_conf = sorted(
        [r for r in results if r["max_confidence"] > 0],
        key=lambda x: x["max_confidence"],
    )[:n]
    
    return {
        "worst_missed": by_missed,
        "worst_fp": by_fp,
        "lowest_confidence": by_low_conf,
    }


def visualize_failures(
    failure_cases: List[Dict],
    output_dir: Path,
    label: str = "missed",
    max_cases: int = 10,
) -> None:
    """Visualize failure cases as a grid."""
    n = min(len(failure_cases), max_cases)
    cols = 2
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(15, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    for idx, case in enumerate(failure_cases[:n]):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Load image
        img = cv2.imread(str(case["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw ground truth (green)
        for gt in case["ground_truth"]:
            x1, y1, x2, y2 = gt["bbox"]
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="green", linewidth=2
            ))
            ax.text(x1, y1 - 5, f"GT", color="green", fontsize=8)
        
        # Draw predictions (red)
        for pred in case["predictions"]:
            x1, y1, x2, y2 = pred["bbox"]
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", linewidth=2
            ))
            ax.text(x1, y2 + 5, f"{pred['confidence']:.2f}", color="red", fontsize=8)
        
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{case['stem']}\nGT: {len(case['ground_truth'])} | "
                    f"Pred: {len(case['predictions'])} | "
                    f"Missed: {case['missed_detections']} | "
                    f"FP: {case['false_positives']}")
    
    plt.tight_layout()
    output_path = output_dir / f"top10_{label}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Failure visualization saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                       default="outputs/training/pcb_defect_yolov8/weights/best.pt")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()
    
    # Paths
    test_images_dir = PROCESSED_DIR / "test" / "images"
    test_labels_dir = PROCESSED_DIR / "test" / "labels"
    
    # Analyze test set
    logger.info("Analyzing test set...")
    results = analyze_test_set(
        args.weights,
        test_images_dir,
        test_labels_dir,
        conf=args.conf,
        iou=args.iou,
    )
    
    # Find worst failures
    failures = find_worst_failures(results, n=args.n)
    
    # Visualize
    visualize_failures(failures["worst_missed"], EVAL_REPORT_DIR, "missed", args.n)
    visualize_failures(failures["worst_fp"], EVAL_REPORT_DIR, "false_positives", args.n)
    
    # Save results
    from src.evaluation.utils import save_json
    save_json({
        "total_images": len(results),
        "total_gt": sum(r["match"]["n_gt"] for r in results),
        "total_pred": sum(r["match"]["n_pred"] for r in results),
        "total_tp": sum(r["match"]["n_tp"] for r in results),
        "total_fp": sum(r["match"]["n_fp"] for r in results),
        "total_fn": sum(r["match"]["n_fn"] for r in results),
    }, EVAL_REPORT_DIR / "failure_summary.json")
    
    logger.info("Failure analysis complete!")


if __name__ == "__main__":
    main()