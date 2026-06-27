import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# ─────────────────────────────────────────────
# Third Party
# ─────────────────────────────────────────────

from PIL import Image

# ─────────────────────────────────────────────
# src.utils imports
# ─────────────────────────────────────────────

from src.utils import paths
from src.utils import config
from src.utils import logger

# ─────────────────────────────────────────────
# Local imports from validate_dataset
# ─────────────────────────────────────────────

from src.data.validate_dataset import (
    Sample,
    build_manifest,
    read_annotation_lines,
    parse_annotation_line,
    get_image_size,
    load_flagged_stems,
)

# ─────────────────────────────────────────────
# Paths from utils.paths
# ─────────────────────────────────────────────

PCB_DATA_DIR = paths.PCB_DATA_DIR
LOGS_DIR = paths.LOGS_DIR
FLAGGED_PATH = paths.FLAGGED_PATH
OUTPUTS_DIR = paths.OUTPUTS_DIR

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABELS_DIR = OUTPUTS_DIR / "labels"
CONVERSION_REPORT_PATH = OUTPUTS_DIR / "reports" / "conversion_report.json"

# Ensure directories exist
LABELS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.setup_logging(log_file=LOGS_DIR / "convert.log")
log = logger.get_logger(__name__)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def convert_to_yolo_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_id: int,
    img_width: int,
    img_height: int
) -> Tuple[int, float, float, float, float]:
    """
    Convert DeepPCB annotation to YOLO format.
    
    DeepPCB format: x1 y1 x2 y2 class_id (1-indexed)
    YOLO format: class cx cy w h (0-indexed, normalized)
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates in pixels
        class_id: DeepPCB class ID (1-6)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Tuple of (yolo_class, cx, cy, width, height) normalized
    """
    # Convert class from 1-indexed to 0-indexed
    yolo_class = class_id - 1
    
    # Calculate center and dimensions
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize by image dimensions
    cx /= img_width
    cy /= img_height
    width /= img_width
    height /= img_height
    
    # Clamp to [0, 1] to avoid floating point edge cases
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return (yolo_class, cx, cy, width, height)


def write_yolo_annotation(output_path: Path, annotations: List[Tuple[int, float, float, float, float]]) -> None:
    """
    Write YOLO annotations to a file.
    
    Format: class cx cy w h (6 decimal places)
    One object per line.
    
    Args:
        output_path: Path to output .txt file
        annotations: List of (class, cx, cy, w, h) tuples
    """
    with output_path.open("w", encoding="utf-8") as f:
        for yolo_class, cx, cy, width, height in annotations:
            f.write(f"{yolo_class} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")


def convert_sample(sample: Sample, output_dir: Path) -> bool:
    """
    Convert a single sample from DeepPCB format to YOLO format.
    
    Args:
        sample: Sample object containing paths and metadata
        output_dir: Directory to write the YOLO label file
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Get image dimensions using get_image_size from validate_dataset
        img_width, img_height = get_image_size(sample.test_image)
        
        # Read annotation lines
        lines = read_annotation_lines(sample.annotation)
        
        if not lines:
            log.warning(f"Empty annotation file for {sample.stem}")
            return False
        
        # Convert each annotation
        yolo_annotations = []
        for line in lines:
            try:
                x1, y1, x2, y2, class_id = parse_annotation_line(line)
                
                # Validate class_id
                if class_id < 1 or class_id > 6:
                    log.warning(f"Invalid class_id {class_id} in {sample.stem}")
                    return False
                
                # Convert to YOLO format
                yolo_anno = convert_to_yolo_bbox(
                    x1, y1, x2, y2, class_id, img_width, img_height
                )
                yolo_annotations.append(yolo_anno)
                
            except ValueError as e:
                log.warning(f"Parse error in {sample.stem}: {e}")
                return False
        
        # Write output file
        output_path = output_dir / f"{sample.stem}.txt"
        write_yolo_annotation(output_path, yolo_annotations)
        
        return True
        
    except Exception as e:
        log.error(f"Failed to convert {sample.stem}: {e}")
        return False


def convert_dataset(
    output_dir: Path = LABELS_DIR,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Convert the entire DeepPCB dataset to YOLO format.
    
    Args:
        output_dir: Directory to write YOLO label files
        max_samples: Maximum number of samples to convert (for testing)
    
    Returns:
        Dictionary with conversion statistics
    """
    log.info("Starting DeepPCB annotation conversion")
    
    # Build manifest
    log.info("Building manifest...")
    manifest = build_manifest(PCB_DATA_DIR)
    total_samples = len(manifest)
    log.info(f"Found {total_samples} samples")
    
    # Limit for testing
    if max_samples is not None:
        manifest = manifest[:max_samples]
        log.info(f"Limited to {len(manifest)} samples for testing")
    
    # Load flagged samples
    log.info("Loading flagged samples...")
    flagged_stems = load_flagged_stems(FLAGGED_PATH)
    skipped_count = len(flagged_stems)
    log.info(f"Found {skipped_count} flagged samples to skip")
    
    # Convert samples
    log.info("Converting annotations...")
    converted_stems = []
    failed_stems = []
    skipped_stems = []
    
    for i, sample in enumerate(manifest, 1):
        # Check if flagged
        if sample.stem in flagged_stems:
            skipped_stems.append(sample.stem)
            log.debug(f"Skipping flagged sample {sample.stem}")
            continue
        
        # Convert
        if convert_sample(sample, output_dir):
            converted_stems.append(sample.stem)
        else:
            failed_stems.append(sample.stem)
            log.warning(f"Failed to convert {sample.stem}")
        
        # Progress logging
        if i % 100 == 0:
            log.info(f"Progress: {i}/{len(manifest)} samples processed")
    
    # Build results
    results = {
        "dataset": config.DATASET_NAME,
        "timestamp": datetime.now().isoformat(),
        "output_directory": str(output_dir),
        "total_samples": total_samples,
        "converted": len(converted_stems),
        "skipped": len(skipped_stems),
        "failed": len(failed_stems),
        "converted_stems": sorted(converted_stems),
        "skipped_stems": sorted(skipped_stems),
        "failed_stems": sorted(failed_stems),
    }
    
    return results


def write_conversion_report(results: Dict, path: Path) -> None:
    """Write conversion report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Conversion report written to {path}")


def print_summary(results: Dict) -> None:
    """Print conversion summary to console."""
    print("\n" + "="*50)
    print("DeepPCB Annotation Conversion")
    print("="*50)
    
    print(f"\nBuilding manifest...")
    print(f"{results['total_samples']} samples found")
    
    print(f"\nLoading flagged samples...")
    print(f"{results['skipped']} skipped")
    
    print(f"\nConverting...")
    print(f"  [OK] Converted : {results['converted']}")
    print(f"  [FAIL] Failed    : {results['failed']}")
    
    print(f"\nYOLO labels written to")
    print(f"  {results['output_directory']}")
    
    print(f"\nConversion report saved.")
    print("="*50 + "\n")


def main() -> None:
    """Execute the annotation conversion pipeline."""
    try:
        # Convert dataset
        results = convert_dataset(LABELS_DIR)
        
        # Write report
        write_conversion_report(results, CONVERSION_REPORT_PATH)
        
        # Print summary
        print_summary(results)
        
        # Exit with appropriate code
        if results['failed'] > 0:
            log.error(f"Conversion failed for {results['failed']} samples")
            sys.exit(1)
        elif results['skipped'] > 0 and results['failed'] == 0:
            log.warning(f"Conversion completed with {results['skipped']} skipped samples")
            sys.exit(0)
        else:
            log.info("Conversion completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        log.info("Conversion interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
