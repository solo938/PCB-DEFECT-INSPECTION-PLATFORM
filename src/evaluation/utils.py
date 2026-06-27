# src/evaluation/utils.py
"""
Shared utilities for evaluation modules.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.utils.paths import OUTPUTS_DIR
from src.utils.config import CLASS_ID_TO_NAME, YOLO_CLASS_ID_TO_NAME

# Output directories
EVAL_REPORT_DIR = OUTPUTS_DIR / "eval_report"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"

# Ensure directories exist
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)


def get_class_name(class_id: int) -> str:
    """Get class name from ID using YOLO mapping."""
    return YOLO_CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}")


def get_class_id(class_name: str) -> int:
    """Get class ID from name."""
    for idx, name in YOLO_CLASS_ID_TO_NAME.items():
        if name == class_name:
            return idx
    return -1


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Save data as pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """Load pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
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


def format_metrics_table(metrics: Dict) -> str:
    """Format metrics as a text table."""
    lines = []
    lines.append("=" * 60)
    lines.append("PCB DEFECT DETECTION — EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"Overall mAP@50:      {metrics.get('map50', 0):.4f}")
    lines.append(f"Overall mAP@50-95:   {metrics.get('map', 0):.4f}")
    lines.append(f"Overall Precision:   {metrics.get('precision', 0):.4f}")
    lines.append(f"Overall Recall:      {metrics.get('recall', 0):.4f}")
    lines.append(f"Overall F1:          {metrics.get('f1', 0):.4f}")
    lines.append("")
    lines.append("Per-class results:")
    lines.append(f"  {'Class':<18} {'AP@50':<10} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    lines.append("-" * 60)
    
    per_class = metrics.get('per_class', {})
    for name, stats in per_class.items():
        lines.append(
            f"  {name:<18} {stats.get('ap50', 0):<10.4f} "
            f"{stats.get('precision', 0):<12.4f} "
            f"{stats.get('recall', 0):<12.4f} "
            f"{stats.get('f1', 0):<10.4f}"
        )
    
    lines.append("=" * 60)
    return "\n".join(lines)