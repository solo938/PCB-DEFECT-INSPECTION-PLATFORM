# src/data/augment.py
"""
Offline data augmentation for DeepPCB dataset.

Generates augmented copies of training images while preserving bounding boxes
and template images. Uses Albumentations for augmentations.

Output:
    data/processed/train/ (overwrites/adds to existing)
    ├── images/
    │   ├── 00041011_test.jpg      # Original
    │   ├── 00041011_test_aug0.jpg # Augmented copy 1
    │   ├── 00041011_test_aug1.jpg # Augmented copy 2
    │   └── ...
    ├── labels/
    │   ├── 00041011.txt           # Original labels
    │   ├── 00041011_aug0.txt      # Augmented labels 1
    │   ├── 00041011_aug1.txt      # Augmented labels 2
    │   └── ...
    └── templates/
        ├── 00041011_temp.jpg      # Original template
        ├── 00041011_temp_aug0.jpg # Copied template 1
        ├── 00041011_temp_aug1.jpg # Copied template 2
        └── ...

Usage:
    python -m src.data.augment

Exits with code 0 if successful, 1 if errors occur.
"""

# ─────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────

from pathlib import Path
import shutil
import json
import sys
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

# ─────────────────────────────────────────────
# Third Party
# ─────────────────────────────────────────────

import cv2
import numpy as np
import albumentations as A

# ─────────────────────────────────────────────
# src.utils imports
# ─────────────────────────────────────────────

from src.utils import paths
from src.utils import config
from src.utils import logger

# ─────────────────────────────────────────────
# Paths from utils.paths
# ─────────────────────────────────────────────

PROCESSED_DIR = paths.PROCESSED_DIR
TRAIN_IMAGES_DIR = PROCESSED_DIR / "train" / "images"
TRAIN_LABELS_DIR = PROCESSED_DIR / "train" / "labels"
TRAIN_TEMPLATES_DIR = PROCESSED_DIR / "train" / "templates"
LOGS_DIR = paths.LOGS_DIR
OUTPUTS_DIR = paths.OUTPUTS_DIR
AUGMENTATION_REPORT_PATH = OUTPUTS_DIR / "reports" / "augmentation_report.json"

# ─────────────────────────────────────────────
# Constants from utils.config
# ─────────────────────────────────────────────

AUG_COPIES_PER_IMAGE = config.AUG_COPIES_PER_IMAGE
AUG_FLIP_PROB = config.AUG_FLIP_PROB
AUG_ROTATE_PROB = config.AUG_ROTATE_PROB
AUG_BRIGHTNESS_PROB = config.AUG_BRIGHTNESS_PROB
AUG_NOISE_PROB = config.AUG_NOISE_PROB
AUG_MOTION_BLUR_PROB = config.AUG_MOTION_BLUR_PROB
AUG_CLAHE_PROB = config.AUG_CLAHE_PROB
TEST_SUFFIX = config.TEST_SUFFIX
TEMP_SUFFIX = config.TEMP_SUFFIX
ANNOTATION_SUFFIX = config.ANNOTATION_SUFFIX

# ─────────────────────────────────────────────
# Patterns for matching augmentation files
# ─────────────────────────────────────────────

# Matches: stem_aug0, stem_aug1, stem_aug2, etc.
AUG_PATTERN = re.compile(r".*_aug\d+$")

# ─────────────────────────────────────────────
# Disable OpenCV threads
# ─────────────────────────────────────────────

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# ─────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.setup_logging(log_file=LOGS_DIR / "augment.log")
log = logger.get_logger(__name__)


# ─────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────

@dataclass
class Sample:
    """Represents one training sample."""
    stem: str
    image_path: Path
    label_path: Path
    template_path: Path


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def is_augmented_filename(stem: str) -> bool:
    """
    Check if a filename stem corresponds to an augmented file.
    
    Matches patterns like:
        - 00041011_aug0
        - 00041011_aug1
        - 00041011_aug2
    
    Does NOT match:
        - 00041011 (original)
        - real_augmented_board (valid original filename)
    
    Args:
        stem: Filename stem (without extension)
    
    Returns:
        True if the filename is an augmentation, False otherwise
    """
    return bool(AUG_PATTERN.match(stem))


def remove_old_augmentations() -> None:
    """
    Remove previously generated augmentation files.
    Uses regex to safely match only augmentation files.
    """
    log.info("Removing old augmentation files...")
    
    removed_count = 0
    for directory in (TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TRAIN_TEMPLATES_DIR):
        if not directory.exists():
            continue
        
        for file in directory.iterdir():
            if is_augmented_filename(file.stem):
                file.unlink()
                removed_count += 1
    
    if removed_count > 0:
        log.info(f"Removed {removed_count} old augmentation files")
    else:
        log.info("No old augmentation files found")


def build_manifest() -> List[Sample]:
    """
    Scan train/images and build a manifest of training samples.
    Only includes original images (not augmented copies).
    
    Returns:
        List of Sample objects
    """
    log.info("Loading training set...")
    
    samples = []
    image_files = list(TRAIN_IMAGES_DIR.glob(f"*{TEST_SUFFIX}"))
    
    # Filter out augmented files using regex
    original_images = [
        p for p in image_files
        if not is_augmented_filename(p.stem.replace("_test", ""))
    ]
    
    log.info(f"Found {len(original_images)} original images (excluding augmentations)")
    
    for img_path in original_images:
        stem = img_path.stem.replace("_test", "")
        
        label_path = TRAIN_LABELS_DIR / f"{stem}{ANNOTATION_SUFFIX}"
        template_path = TRAIN_TEMPLATES_DIR / f"{stem}{TEMP_SUFFIX}"
        
        # Check required files exist
        if not label_path.exists():
            log.warning(f"Label missing for {stem}: {label_path}")
            continue
        if not template_path.exists():
            log.warning(f"Template missing for {stem}: {template_path}")
            continue
        
        samples.append(Sample(
            stem=stem,
            image_path=img_path,
            label_path=label_path,
            template_path=template_path
        ))
    
    log.info(f"Found {len(samples)} original samples")
    return samples


def load_yolo_labels(label_path: Path) -> Tuple[np.ndarray, List[int]]:
    """
    Load YOLO labels from a text file.
    
    Handles both integer and float class IDs (0 or 0.0).
    Validates coordinate ranges.
    
    Args:
        label_path: Path to YOLO label file
    
    Returns:
        Tuple of (boxes, class_labels)
        boxes: numpy array of shape (N, 4) with normalized YOLO format
        class_labels: list of class IDs (as integers)
    """
    boxes = []
    class_labels = []
    
    with label_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                log.warning(
                    f"Malformed label in {label_path}:{line_num} "
                    f"(expected 5 parts, got {len(parts)})"
                )
                continue
            
            try:
                # Handle both '0' and '0.0' for class ID
                class_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as e:
                log.warning(f"Invalid label in {label_path}:{line_num} - {e}")
                continue
            
            # Validate coordinate ranges
            if not (0.0 <= cx <= 1.0):
                log.warning(f"cx={cx} out of range [0,1] in {label_path}:{line_num}")
                continue
            if not (0.0 <= cy <= 1.0):
                log.warning(f"cy={cy} out of range [0,1] in {label_path}:{line_num}")
                continue
            if not (0.0 < w <= 1.0):
                log.warning(f"w={w} out of range (0,1] in {label_path}:{line_num}")
                continue
            if not (0.0 < h <= 1.0):
                log.warning(f"h={h} out of range (0,1] in {label_path}:{line_num}")
                continue
            
            boxes.append([cx, cy, w, h])
            class_labels.append(class_id)
    
    return np.array(boxes, dtype=np.float32), class_labels


def clamp_boxes(boxes: np.ndarray) -> np.ndarray:
    """
    Clamp YOLO bbox coordinates to [0, 1] range.
    
    Fixes floating-point precision errors where coordinates
    may be slightly outside the valid range (e.g., -4e-7 or 1.0000002).
    
    Args:
        boxes: numpy array of shape (N, 4) with normalized YOLO format
    
    Returns:
        Clamped boxes array
    """
    return np.clip(boxes, 0.0, 1.0)


def save_yolo_labels(label_path: Path, boxes: np.ndarray, class_labels: List[int]) -> None:
    """
    Save YOLO labels to a text file.
    
    Args:
        label_path: Path to output YOLO label file
        boxes: numpy array of shape (N, 4) with normalized YOLO format
        class_labels: list of class IDs (integers)
    """
    # Clamp boxes to [0, 1] range to fix floating point issues
    boxes = clamp_boxes(boxes)
    
    with label_path.open("w", encoding="utf-8") as f:
        for box, class_id in zip(boxes, class_labels):
            cx, cy, w, h = box
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image as RGB numpy array.
    
    Args:
        image_path: Path to image file
    
    Returns:
        RGB image as numpy array
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image(image_path: Path, image_rgb: np.ndarray) -> None:
    """
    Save RGB image as JPG.
    
    Args:
        image_path: Path to output image file
        image_rgb: RGB image as numpy array
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), image_bgr)


def atomic_write_files(
    image_path: Path,
    image: np.ndarray,
    label_path: Path,
    boxes: np.ndarray,
    class_labels: List[int],
    template_src: Path,
    template_dst: Path
) -> bool:
    """
    Write all files atomically. If any operation fails, clean up.
    
    Args:
        image_path: Path to output image
        image: RGB image array
        label_path: Path to output label
        boxes: YOLO boxes
        class_labels: Class IDs
        template_src: Source template path
        template_dst: Destination template path
    
    Returns:
        True if all files were written successfully, False otherwise
    """
    # Track created files for cleanup on failure
    created_files = []
    
    try:
        # Save image
        save_image(image_path, image)
        created_files.append(image_path)
        
        # Save labels
        save_yolo_labels(label_path, boxes, class_labels)
        created_files.append(label_path)
        
        # Copy template
        shutil.copy2(template_src, template_dst)
        created_files.append(template_dst)
        
        return True
        
    except Exception as e:
        # Clean up any partially written files
        for file_path in created_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
        raise e


# ─────────────────────────────────────────────
# Build Albumentations Pipeline
# ─────────────────────────────────────────────

def build_transform() -> A.Compose:
    """
    Build Albumentations augmentation pipeline.
    
    Returns:
        A.Compose object with all augmentations
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=AUG_FLIP_PROB),
        A.RandomRotate90(p=AUG_ROTATE_PROB),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=True,
            p=AUG_BRIGHTNESS_PROB
        ),
        
        # Noise augmentations
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            p=AUG_NOISE_PROB
        ),
        
        # Blur augmentations
        A.MotionBlur(
            blur_limit=3,
            p=AUG_MOTION_BLUR_PROB
        ),
        
        # Contrast enhancement
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=AUG_CLAHE_PROB
        ),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
        min_area=0.001,
    ))


# ─────────────────────────────────────────────
# Augment One Sample
# ─────────────────────────────────────────────

def augment_sample(sample: Sample, transform: A.Compose, aug_idx: int) -> bool:
    """
    Generate one augmented copy of a sample.
    
    Args:
        sample: Sample to augment
        transform: Albumentations transform pipeline
        aug_idx: Augmentation index (0, 1, 2, ...)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image and labels
        image = load_image(sample.image_path)
        boxes, class_labels = load_yolo_labels(sample.label_path)
        
        # Skip if no boxes
        if len(boxes) == 0:
            log.warning(f"No boxes in {sample.stem}, skipping augmentation")
            return False
        
        # Apply augmentation
        augmented = transform(
            image=image,
            bboxes=boxes,
            class_labels=class_labels
        )
        
        # Get augmented data
        aug_image = augmented["image"]
        aug_boxes = np.array(augmented["bboxes"], dtype=np.float32)
        
        # Ensure class labels are integers (some Albumentations versions return floats)
        aug_class_labels = [int(x) for x in augmented["class_labels"]]
        
        # Skip if all boxes were filtered out
        if len(aug_boxes) == 0:
            log.debug(f"All boxes filtered for {sample.stem}_aug{aug_idx}, skipping")
            return False
        
        # Generate output filenames
        output_stem = f"{sample.stem}_aug{aug_idx}"
        
        aug_image_path = TRAIN_IMAGES_DIR / f"{output_stem}{TEST_SUFFIX}"
        aug_label_path = TRAIN_LABELS_DIR / f"{output_stem}{ANNOTATION_SUFFIX}"
        aug_template_path = TRAIN_TEMPLATES_DIR / f"{output_stem}{TEMP_SUFFIX}"
        
        # Write all files atomically
        atomic_write_files(
            image_path=aug_image_path,
            image=aug_image,
            label_path=aug_label_path,
            boxes=aug_boxes,
            class_labels=aug_class_labels,
            template_src=sample.template_path,
            template_dst=aug_template_path
        )
        
        return True
        
    except Exception as e:
        log.error(f"Failed to augment {sample.stem}_aug{aug_idx}: {e}")
        return False


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────

def augment_dataset() -> dict:
    """
    Augment the entire training dataset.
    
    Returns:
        Dict with augmentation statistics
    """
    # Remove old augmentations first
    remove_old_augmentations()
    
    log.info("Building augmentation pipeline...")
    transform = build_transform()
    
    # Build manifest
    samples = build_manifest()
    if not samples:
        log.error("No training samples found")
        return {"error": "No training samples found"}
    
    original_count = len(samples)
    augmented_count = 0
    successful_augmentations = 0
    failed_samples = []
    
    log.info(f"Generating augmentations ({AUG_COPIES_PER_IMAGE} copies per image)...")
    
    for idx, sample in enumerate(samples):
        success_count = 0
        for aug_idx in range(AUG_COPIES_PER_IMAGE):
            if augment_sample(sample, transform, aug_idx):
                success_count += 1
                augmented_count += 1
        
        if success_count > 0:
            successful_augmentations += 1
        else:
            failed_samples.append(sample.stem)
        
        # Log progress every 50 samples
        if (idx + 1) % 50 == 0:
            log.info(f"Progress: {idx + 1}/{original_count} samples")
    
    final_count = original_count + augmented_count
    
    # Build report
    report = {
        "original_images": original_count,
        "copies_per_image": AUG_COPIES_PER_IMAGE,
        "generated_images": augmented_count,
        "final_training_images": final_count,
        "samples_with_augmentations": successful_augmentations,
        "failed_samples": failed_samples[:10],
        "failed_count": len(failed_samples),
    }
    
    log.info(f"Original images: {original_count}")
    log.info(f"Augmented copies: {augmented_count}")
    log.info(f"Final images: {final_count}")
    if failed_samples:
        log.warning(f"Failed to augment {len(failed_samples)} samples")
    
    return report


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def write_report(report: dict, path: Path) -> None:
    """Write augmentation report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report written to {path}")


# ─────────────────────────────────────────────
# Terminal Output
# ─────────────────────────────────────────────

def print_summary(report: dict) -> None:
    """Print augmentation summary to console."""
    print("\n" + "="*50)
    print("DeepPCB Dataset Augmentation")
    print("="*50)
    
    if "error" in report:
        print(f"\n[ERROR] {report['error']}")
        return
    
    print(f"\nLoading training set...")
    print(f"{report['original_images']} samples")
    
    print(f"\nBuilding augmentation pipeline...")
    print(f"\nGenerating augmentations...")
    print(f"\nOriginal images : {report['original_images']}")
    print(f"Augmented copies: {report['generated_images']}")
    print(f"Final images    : {report['final_training_images']}")
    
    if report.get('failed_count', 0) > 0:
        print(f"Failed samples  : {report['failed_count']}")
        if report['failed_samples']:
            print(f"  First failures: {', '.join(report['failed_samples'][:5])}")
    
    print(f"\nSaving labels...")
    print(f"\nSaving templates...")
    print(f"\nReport written.")
    print("\n" + "="*50)
    print("Augmentation completed successfully.")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────

def main() -> None:
    """Execute the augmentation pipeline."""
    try:
        log.info("Starting DeepPCB dataset augmentation")
        
        # Run augmentation
        report = augment_dataset()
        
        if "error" in report:
            log.error(f"Augmentation failed: {report['error']}")
            print_summary(report)
            sys.exit(1)
        
        # Write report
        write_report(report, AUGMENTATION_REPORT_PATH)
        
        # Print summary
        print_summary(report)
        
        log.info("Augmentation completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        log.info("Augmentation interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error(f"Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()