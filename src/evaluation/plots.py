# src/evaluation/plots.py
"""
Plot generation for evaluation: confusion matrix, PR curves, calibration.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from src.evaluation.utils import EVAL_REPORT_DIR, load_json, save_pickle, load_pickle
from src.utils.config import YOLO_CLASS_ID_TO_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_evaluation_results() -> Dict:
    """Load evaluation results (prefer pickle for full object)."""
    results_path = EVAL_REPORT_DIR / "evaluation_results.pkl"
    if results_path.exists():
        return load_pickle(results_path)
    return load_json(EVAL_REPORT_DIR / "evaluation_results.json")


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Plot confusion matrix."""
    if normalize:
        # Row-normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / (row_sums + 1e-8)
        suffix = "normalised"
        title = "Confusion Matrix — Normalised (Recall)"
    else:
        suffix = "raw"
        title = "Confusion Matrix — Raw Counts"
    
    plt.figure(figsize=(10, 8))
    
    # Add background class
    labels = class_names + ["Background"]
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_pr_curves(pr_data: Dict, output_path: Path) -> None:
    """Plot PR curves per class and combined."""
    class_names = list(YOLO_CLASS_ID_TO_NAME.values())
    
    # Per-class subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for idx, (ax, name) in enumerate(zip(axes, class_names)):
        if name in pr_data:
            data = pr_data[name]
            ax.plot(data["recall"], data["precision"], color=colors[idx])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{name}\nAP: {data.get('ap', 0):.3f}")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig(output_path / "pr_curves_per_class.png", dpi=150)
    plt.close()
    
    # Combined PR curves
    plt.figure(figsize=(10, 8))
    for idx, name in enumerate(class_names):
        if name in pr_data:
            data = pr_data[name]
            plt.plot(
                data["recall"],
                data["precision"],
                label=f"{name} (AP: {data.get('ap', 0):.3f})",
                color=colors[idx],
            )
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Classes")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "pr_curves_combined.png", dpi=150)
    plt.close()
    
    logger.info(f"PR curves saved to {output_path}")


def plot_calibration_curve(predictions: list, output_path: Path) -> None:
    """Plot calibration curve (confidence vs accuracy)."""
    if not predictions:
        return
    
    # Extract confidences and correctness
    confidences = []
    accuracies = []
    
    # Group by confidence bins
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins)
    
    for bin_idx in range(1, len(bins)):
        mask = bin_indices == bin_idx
        if not mask.any():
            continue
        confidences.append(confidences[mask].mean())
        accuracies.append(accuracies[mask].mean())
    
    plt.figure(figsize=(8, 8))
    plt.plot(confidences, accuracies, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration", color="gray")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "calibration_curve.png", dpi=150)
    plt.close()
    
    logger.info(f"Calibration curve saved to {output_path}")


def main() -> None:
    """Generate all plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                       default="outputs/training/pcb_defect_yolov8/weights/best.pt")
    args = parser.parse_args()
    
    # Load results
    results = load_evaluation_results()
    
    class_names = list(YOLO_CLASS_ID_TO_NAME.values())
    
    # Confusion matrix
    if results.get("confusion_matrix") is not None:
        matrix = np.array(results["confusion_matrix"])
        
        # Raw counts
        plot_confusion_matrix(
            matrix,
            class_names,
            EVAL_REPORT_DIR / "confusion_matrix_raw.png",
            normalize=False,
        )
        
        # Normalised
        plot_confusion_matrix(
            matrix,
            class_names,
            EVAL_REPORT_DIR / "confusion_matrix_normalised.png",
            normalize=True,
        )
    
    # PR curves - need to extract from evaluation results
    # This requires access to the full results object
    if EVAL_REPORT_DIR / "evaluation_results.pkl".exists():
        results_obj = load_pickle(EVAL_REPORT_DIR / "evaluation_results.pkl")
        # Extract PR data from results_obj
        # ...


if __name__ == "__main__":
    main()