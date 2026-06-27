# src/evaluation/metrics.py
"""
Metrics extraction and display. Reads from evaluation_results.json.
"""

import argparse
from pathlib import Path
from typing import Dict

from src.evaluation.utils import EVAL_REPORT_DIR, load_json, save_json, format_metrics_table
from src.utils.logger import get_logger
from src.utils.config import YOLO_CLASS_ID_TO_NAME

logger = get_logger(__name__)


def extract_metrics_from_results(results: Dict) -> Dict:
    """Extract clean metrics from evaluation results."""
    metrics = results.get("metrics", {})
    
    # Ensure per_class is properly formatted
    per_class = {}
    class_names = list(YOLO_CLASS_ID_TO_NAME.values())
    
    for name in class_names:
        if name in metrics.get("per_class", {}):
            per_class[name] = metrics["per_class"][name]
        else:
            per_class[name] = {"ap": 0, "ap50": 0, "precision": 0, "recall": 0, "f1": 0}
    
    return {
        "map50": metrics.get("map50", 0),
        "map": metrics.get("map", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1": metrics.get("f1", 0),
        "per_class": per_class,
        "speed": results.get("speed", {}),
    }


def save_metrics_summary(metrics: Dict) -> None:
    """Save metrics summary as text."""
    table = format_metrics_table(metrics)
    
    # Save as text
    with open(EVAL_REPORT_DIR / "metrics_summary.txt", "w") as f:
        f.write(table)
    
    # Save as JSON
    save_json(metrics, EVAL_REPORT_DIR / "metrics_clean.json")
    
    logger.info(f"Metrics summary saved to {EVAL_REPORT_DIR}/metrics_summary.txt")


def print_metrics_table(metrics: Dict) -> None:
    """Print metrics table to console."""
    print(format_metrics_table(metrics))


def main() -> None:
    # Load results
    results = load_json(EVAL_REPORT_DIR / "evaluation_results.json")
    
    # Extract metrics
    metrics = extract_metrics_from_results(results)
    
    # Save summary
    save_metrics_summary(metrics)
    
    # Print to console
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()