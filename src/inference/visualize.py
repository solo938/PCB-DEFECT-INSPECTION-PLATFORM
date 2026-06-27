# src/inference/visualize.py
"""
Visualization utilities for inference results.
"""

from typing import List, Dict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import YOLO_CLASS_ID_TO_NAME


# Color map for classes
CLASS_COLORS = {
    0: (0, 255, 0),      # missing_hole - green
    1: (255, 0, 0),      # mouse_bite - blue
    2: (0, 255, 255),    # open_circuit - cyan
    3: (255, 0, 255),    # short - magenta
    4: (255, 165, 0),    # spur - orange
    5: (0, 0, 255),      # copper - red
}


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Dict[int, str] = YOLO_CLASS_ID_TO_NAME,
    colors: Dict[int, tuple] = CLASS_COLORS,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: RGB image as numpy array
        detections: List of detection dicts
        class_names: Mapping of class_id to name
        colors: Mapping of class_id to color
        show_confidence: Whether to show confidence scores
    
    Returns:
        Annotated image
    """
    img_copy = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        conf = det["confidence"]
        name = class_names.get(class_id, f"class_{class_id}")
        
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{name}" if not show_confidence else f"{name}: {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Background for text
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_h - 10),
            (x1 + text_w, y1),
            color,
            -1,
        )
        
        # Text
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    
    return img_copy


def save_annotated_image(
    image: np.ndarray,
    detections: List[Dict],
    output_path: Path,
    class_names: Dict[int, str] = YOLO_CLASS_ID_TO_NAME,
) -> None:
    """
    Save annotated image to file.
    """
    annotated = draw_detections(image, detections, class_names)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), annotated_bgr)


def create_detection_grid(
    results: List[Dict],
    images: List[np.ndarray],
    cols: int = 4,
    figsize: tuple = (20, 15),
) -> plt.Figure:
    """
    Create a grid of annotated images.
    
    Args:
        results: List of prediction results
        images: List of original images
        cols: Number of columns in grid
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n = len(results)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for idx, (result, img) in enumerate(zip(results, images)):
        annotated = draw_detections(img, result["detections"])
        axes[idx].imshow(annotated)
        axes[idx].axis("off")
        axes[idx].set_title(
            f"{result['image_name']}\n"
            f"{result['num_detections']} detections, "
            f"{result['inference_time_ms']:.1f}ms"
        )
    
    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    return fig