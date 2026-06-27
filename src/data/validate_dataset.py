# src/data/validate_dataset.py
"""
DeepPCB Dataset Validator

Validates the entire DeepPCB dataset:
- Checks all image files exist, are readable, and have correct dimensions
- Validates annotation file format and content
- Validates all bounding boxes against image dimensions
- Validates all class IDs against the known class map
- Generates a JSON report and a flagged samples list for downstream modules

Usage:
    python -m src.data.validate_dataset

Exits with code 0 if validation passes, 1 if failures exceed threshold.
"""

# ─────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────
# Third Party
# ─────────────────────────────────────────────

from PIL import Image
from PIL import UnidentifiedImageError

# ─────────────────────────────────────────────
# src.utils.paths
# ─────────────────────────────────────────────

from src.utils.paths import (
    PCB_DATA_DIR,
    LOGS_DIR,
    FLAGGED_PATH,
    VALIDATION_REPORT_PATH,
)

# ─────────────────────────────────────────────
# src.utils.config
# ─────────────────────────────────────────────

from src.utils.config import (
    DATASET_NAME,
    EXPECTED_PAIRS,
    EXPECTED_IMAGE_SIZE,
    MIN_VALID_PAIRS,
    TEST_SUFFIX,
    TEMP_SUFFIX,
    ANNOTATION_SUFFIX,
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    VALID_CLASS_IDS,
    FAILURE_RATE_THRESHOLD,
)

# ─────────────────────────────────────────────
# src.utils.logger
# ─────────────────────────────────────────────

from src.utils.logger import get_logger, setup_logging

# ─────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────

setup_logging(log_file=LOGS_DIR / "validate.log")
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def file_exists(path: Path) -> bool:
    """Check whether a file exists."""
    return path.exists() and path.is_file()


def get_image_size(path: Path) -> Tuple[int, int]:
    """Return image dimensions (width, height)."""
    with Image.open(path) as img:
        return img.size


def is_image_readable(path: Path) -> bool:
    """Check whether an image can be opened."""
    try:
        with Image.open(path) as img:
            img.verify()
        # reopen after verify()
        with Image.open(path):
            pass
        return True
    except (UnidentifiedImageError, OSError):
        return False


def read_annotation_lines(path: Path) -> List[str]:
    """Read annotation file, removing blank lines and whitespace."""
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def parse_annotation_line(line: str) -> Tuple[int, int, int, int, int]:
    """
    Parse one DeepPCB annotation.
    Format: x1 y1 x2 y2 class_id
    """
    tokens = line.split()
    if len(tokens) != 5:
        raise ValueError(f"Expected 5 values but got {len(tokens)} : {line}")
    try:
        x1, y1, x2, y2, class_id = map(int, tokens)
    except ValueError:
        raise ValueError(f"Annotation contains non-integer values : {line}")
    return x1, y1, x2, y2, class_id


def is_bbox_valid(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
    """Validate bounding box coordinates."""
    if x1 >= x2:
        return False
    if y1 >= y2:
        return False
    if x1 < 0 or y1 < 0:
        return False
    if x2 > img_w or y2 > img_h:
        return False
    return True


def get_bbox_invalid_reason(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> str:
    """Return human-readable reason for invalid bbox."""
    if x1 >= x2:
        return "x1 >= x2"
    if y1 >= y2:
        return "y1 >= y2"
    if x1 < 0 or y1 < 0:
        return "Negative coordinates"
    if x2 > img_w or y2 > img_h:
        return f"Coordinate exceeds image dimensions ({img_w}x{img_h})"
    return "Unknown validation error"


def load_flagged_stems(path: Path) -> Set[str]:
    """
    Read flagged_samples.txt and return a set of stems.
    Skips comment lines (starting with #).
    Returns empty set if file does not exist.
    """
    if not path.exists():
        return set()

    stems = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                stems.add(line)
    return stems


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Sample:
    """Represents one complete DeepPCB sample."""
    stem: str
    group: str
    test_image: Path
    template_image: Path
    annotation: Path  # Single annotation file


# ─────────────────────────────────────────────
# Validation functions
# ─────────────────────────────────────────────

def build_manifest(pcbdata_root: Path) -> List[Sample]:
    """
    Scan the PCBData directory and build a dataset manifest.
    
    DeepPCB structure:
        PCBData/
        ├── group00041/
        │   ├── 00041/           # Images
        │   │   ├── 00041001_test.jpg
        │   │   ├── 00041001_temp.jpg
        │   │   └── ...
        │   └── 00041_not/       # Annotations
        │       ├── 00041001.txt
        │       ├── 00041002.txt
        │       └── ...
        ├── group44000/
        │   ├── 44000/
        │   │   └── 44000077_test.jpg
        │   └── 44000_not/
        │       └── 44000077.txt
        └── ...
    """
    manifest = []
    missing_files = []
    
    # Step 1: Recursively find all .txt files
    all_txt_files = list(pcbdata_root.rglob(f"*{ANNOTATION_SUFFIX}"))
    logger.info(f"Found {len(all_txt_files)} total .txt files")
    
    # Step 2: Filter to only annotation files (parent folder ends with _not)
    # and ignore trainval.txt, test.txt
    annotation_files = [
        p for p in all_txt_files
        if p.parent.name.endswith("_not")
        and p.name not in ["trainval.txt", "test.txt"]
    ]
    logger.info(f"Found {len(annotation_files)} annotation files in _not directories")
    
    for anno_path in annotation_files:
        # Step 3: Derive stem from annotation filename
        # e.g., 44000077.txt -> 44000077
        stem = anno_path.stem
        
        # Step 4: Compute the image directory
        # anno_path.parent = .../group44000/44000_not
        # image_dir = .../group44000/44000
        anno_parent = anno_path.parent          # 44000_not
        group_dir = anno_parent.parent          # group44000
        image_dir_name = anno_parent.name.replace("_not", "")  # 44000
        image_dir = group_dir / image_dir_name  # group44000/44000
        
        # Step 5: Construct image paths
        test_path = image_dir / f"{stem}{TEST_SUFFIX}"
        temp_path = image_dir / f"{stem}{TEMP_SUFFIX}"
        
        # Step 6: Check if all three exist
        missing = []
        if not test_path.exists():
            missing.append("_test.jpg")
        if not temp_path.exists():
            missing.append("_temp.jpg")
        # annotation exists (we found it)
        
        if missing:
            missing_files.append({
                "stem": stem,
                "group": group_dir.name,
                "missing": missing
            })
            logger.warning(f"Missing files for {group_dir.name}/{stem}: {missing}")
            continue
        
        manifest.append(Sample(
            stem=stem,
            group=group_dir.name,
            test_image=test_path,
            template_image=temp_path,
            annotation=anno_path
        ))
    
    logger.info(f"Built manifest with {len(manifest)} complete samples")
    logger.info(f"Missing files for {len(missing_files)} samples")
    
    if len(manifest) < MIN_VALID_PAIRS:
        raise RuntimeError(
            f"Only {len(manifest)} valid samples found, "
            f"below minimum of {MIN_VALID_PAIRS}"
        )
    elif len(manifest) < EXPECTED_PAIRS:
        logger.warning(
            f"Found {len(manifest)} samples, "
            f"expected {EXPECTED_PAIRS}"
        )
    
    return manifest


def validate_images(samples: List[Sample]) -> Dict:
    """Validate all test/template images."""
    passed_stems = []
    failed_stems = set()
    unreadable_stems = []
    size_mismatch_stems = []

    for sample in samples:
        stem = sample.stem
        all_images_valid = True

        # Check test image
        if not is_image_readable(sample.test_image):
            unreadable_stems.append(f"{stem}_test")
            failed_stems.add(stem)
            all_images_valid = False
            logger.warning(f"Unreadable test image: {sample.test_image}")
        else:
            try:
                width, height = get_image_size(sample.test_image)
                if (width, height) != EXPECTED_IMAGE_SIZE:
                    size_mismatch_stems.append(f"{stem}_test")
                    logger.warning(
                        f"Test image {stem} size mismatch: "
                        f"got ({width}, {height}), expected {EXPECTED_IMAGE_SIZE}"
                    )
            except Exception as e:
                logger.error(f"Error reading test image {stem}: {e}")
                failed_stems.add(stem)
                all_images_valid = False

        # Check template image
        if not is_image_readable(sample.template_image):
            unreadable_stems.append(f"{stem}_temp")
            failed_stems.add(stem)
            all_images_valid = False
            logger.warning(f"Unreadable template image: {sample.template_image}")
        else:
            try:
                width, height = get_image_size(sample.template_image)
                if (width, height) != EXPECTED_IMAGE_SIZE:
                    size_mismatch_stems.append(f"{stem}_temp")
                    logger.warning(
                        f"Template image {stem} size mismatch: "
                        f"got ({width}, {height}), expected {EXPECTED_IMAGE_SIZE}"
                    )
            except Exception as e:
                logger.error(f"Error reading template image {stem}: {e}")
                failed_stems.add(stem)
                all_images_valid = False

        if all_images_valid:
            passed_stems.append(stem)

    return {
        "total_checked": len(samples),
        "passed": len(passed_stems),
        "failed": len(failed_stems),
        "unreadable": unreadable_stems,
        "size_mismatches": size_mismatch_stems,
        "failed_stems": list(failed_stems)
    }


def validate_annotations(samples: List[Sample]) -> Dict:
    """Validate annotation files."""
    passed_stems = []
    failed_stems = set()
    empty_files = []
    parse_errors = []

    for sample in samples:
        stem = sample.stem

        try:
            lines = read_annotation_lines(sample.annotation)
        except Exception as e:
            parse_errors.append({
                "stem": stem,
                "line": "N/A",
                "error": str(e)
            })
            failed_stems.add(stem)
            logger.error(f"Error reading annotation {stem}: {e}")
            continue

        if not lines:
            empty_files.append(stem)
            failed_stems.add(stem)
            logger.warning(f"Empty annotation file: {sample.annotation}")
            continue

        file_valid = True
        for line_num, line in enumerate(lines, 1):
            try:
                parse_annotation_line(line)
            except ValueError as e:
                parse_errors.append({
                    "stem": stem,
                    "line": line,
                    "error": str(e)
                })
                failed_stems.add(stem)
                file_valid = False
                logger.warning(f"Parse error in {stem} line {line_num}: {e}")

        if file_valid:
            passed_stems.append(stem)

    return {
        "total_checked": len(samples),
        "passed": len(passed_stems),
        "failed": len(failed_stems),
        "empty_files": empty_files,
        "parse_errors": parse_errors,
        "failed_stems": list(failed_stems)
    }


def validate_bboxes(samples: List[Sample], skip_stems: Set[str]) -> Dict:
    """
    Validate every bounding box.
    
    Args:
        samples: List of Sample objects
        skip_stems: Stems that already failed annotation validation
    
    Returns:
        Dict with bbox validation statistics
    """
    total_boxes = 0
    invalid_boxes = []
    failed_stems = set()

    sample_stems = {s.stem for s in samples}
    skipped_stems_count = len(skip_stems & sample_stems)

    for sample in samples:
        stem = sample.stem

        # Skip stems that already failed annotation validation
        if stem in skip_stems:
            continue

        # Get image size for bounds checking
        try:
            img_w, img_h = get_image_size(sample.test_image)
        except Exception as e:
            logger.error(f"Cannot get image size for {stem}: {e}")
            failed_stems.add(stem)
            continue

        try:
            lines = read_annotation_lines(sample.annotation)
        except Exception as e:
            logger.error(f"Cannot read annotation for {stem}: {e}")
            failed_stems.add(stem)
            continue

        for line in lines:
            total_boxes += 1

            try:
                x1, y1, x2, y2, class_id = parse_annotation_line(line)
            except ValueError as e:
                invalid_boxes.append({
                    "stem": stem,
                    "line": line,
                    "reason": f"Parse error: {e}"
                })
                failed_stems.add(stem)
                continue

            # Check validity
            if not is_bbox_valid(x1, y1, x2, y2, img_w, img_h):
                reason = get_bbox_invalid_reason(x1, y1, x2, y2, img_w, img_h)
                invalid_boxes.append({
                    "stem": stem,
                    "line": line,
                    "reason": reason
                })
                failed_stems.add(stem)

    return {
        "total_boxes": total_boxes,
        "skipped_stems_count": skipped_stems_count,
        "invalid_box_count": len(invalid_boxes),
        "invalid_boxes": invalid_boxes,
        "failed_stems": list(failed_stems)
    }


def validate_classes(samples: List[Sample], skip_stems: Set[str]) -> Dict:
    """
    Validate annotation class IDs.
    
    Args:
        samples: List of Sample objects
        skip_stems: Stems that already failed annotation validation
    
    Returns:
        Dict with class validation statistics
    """
    class_counter = Counter()
    total_lines = 0
    invalid_class_entries = []
    failed_stems = set()

    sample_stems = {s.stem for s in samples}
    skipped_stems_count = len(skip_stems & sample_stems)

    for sample in samples:
        stem = sample.stem

        # Skip stems that already failed annotation validation
        if stem in skip_stems:
            continue

        try:
            lines = read_annotation_lines(sample.annotation)
        except Exception as e:
            logger.error(f"Cannot read annotation for {stem}: {e}")
            failed_stems.add(stem)
            continue

        for line in lines:
            total_lines += 1

            try:
                _, _, _, _, class_id = parse_annotation_line(line)
            except ValueError as e:
                invalid_class_entries.append({
                    "stem": stem,
                    "line": line,
                    "class_id": "parse_error"
                })
                failed_stems.add(stem)
                continue

            if class_id not in VALID_CLASS_IDS:
                invalid_class_entries.append({
                    "stem": stem,
                    "line": line,
                    "class_id": class_id
                })
                failed_stems.add(stem)
                logger.warning(f"Invalid class ID {class_id} in {stem}")
            else:
                class_counter[class_id] += 1

    # Build named class counts
    named_counts = {
        CLASS_ID_TO_NAME.get(cid, f"unknown_{cid}"): count
        for cid, count in class_counter.items()
    }

    return {
        "total_annotations_checked": total_lines,
        "class_counts": named_counts,
        "skipped_stems_count": skipped_stems_count,
        "invalid_class_count": len(invalid_class_entries),
        "invalid_class_entries": invalid_class_entries,
        "failed_stems": list(failed_stems)
    }


# ─────────────────────────────────────────────
# Report functions
# ─────────────────────────────────────────────

def write_validation_report(results: Dict, path: Path) -> None:
    """Write validation report to JSON."""
    report = {
        "dataset": DATASET_NAME,
        "timestamp": datetime.now().isoformat(),
        "summary": results["summary"],
        "image_validation": results["images"],
        "annotation_validation": results["annotations"],
        "bbox_validation": results["bboxes"],
        "class_validation": results["classes"]
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Validation report written to {path}")


def write_flagged_samples(stems: Set[str], path: Path) -> None:
    """Write one failed sample stem per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_sorted = sorted(stems)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Generated by validate_dataset.py\n")
        f.write("# Stems listed here are excluded from all downstream modules\n")
        for stem in unique_sorted:
            f.write(f"{stem}\n")

    logger.info(f"Flagged {len(unique_sorted)} samples written to {path}")


# ─────────────────────────────────────────────
# Console output (ASCII only)
# ─────────────────────────────────────────────

def print_summary(summary: Dict, img_results: Dict, anno_results: Dict,
                  bbox_results: Dict, cls_results: Dict) -> None:
    """Print validation summary to console (ASCII only)."""
    print("\n" + "="*60)
    print("DEEPPCB DATASET VALIDATION SUMMARY")
    print("="*60)

    print("\n[STATS] Dataset Statistics:")
    print(f"  Total samples found: {summary['total_pairs_found']}")
    print(f"  [OK] Passed: {summary['total_passed']}")
    print(f"  [FAIL] Flagged: {summary['total_flagged']}")

    print("\n[IMAGES] Image Validation:")
    print(f"  Total samples checked: {img_results['total_checked']}")
    print(f"  [OK] Valid: {img_results['passed']}")
    print(f"  [FAIL] Failed: {img_results['failed']}")
    if img_results['unreadable']:
        print(f"  [WARN] Unreadable: {len(img_results['unreadable'])}")
    if img_results['size_mismatches']:
        print(f"  [WARN] Size mismatches: {len(img_results['size_mismatches'])}")

    print("\n[ANNOTATIONS] Annotation Validation:")
    print(f"  Total annotation files: {anno_results['total_checked']}")
    print(f"  [OK] Valid: {anno_results['passed']}")
    print(f"  [FAIL] Failed: {anno_results['failed']}")
    if anno_results['empty_files']:
        print(f"  [WARN] Empty files: {len(anno_results['empty_files'])}")
    if anno_results['parse_errors']:
        print(f"  [WARN] Parse errors: {len(anno_results['parse_errors'])}")

    print("\n[BBOX] Bounding Box Validation:")
    print(f"  Total boxes: {bbox_results['total_boxes']}")
    print(f"  Skipped stems: {bbox_results['skipped_stems_count']} (failed annotation validation)")
    print(f"  [OK] Valid boxes: {bbox_results['total_boxes'] - bbox_results['invalid_box_count']}")
    print(f"  [FAIL] Invalid boxes: {bbox_results['invalid_box_count']}")

    print("\n[CLASSES] Class Validation:")
    print(f"  Total annotations: {cls_results['total_annotations_checked']}")
    print(f"  Skipped stems: {cls_results['skipped_stems_count']} (failed annotation validation)")
    valid_classes = cls_results['total_annotations_checked'] - cls_results['invalid_class_count']
    print(f"  [OK] Valid classes: {valid_classes}")
    print(f"  [FAIL] Invalid classes: {cls_results['invalid_class_count']}")

    if cls_results['class_counts']:
        print("\n  Class Distribution (by ID order):")
        for class_name, count in sorted(
            cls_results['class_counts'].items(),
            key=lambda x: CLASS_NAME_TO_ID.get(x[0], 99)
        ):
            print(f"    {class_name}: {count}")

    print("\n" + "="*60)

    if summary['total_flagged'] > 0:
        print(f"\n[WARN] {summary['total_flagged']} samples flagged for manual review")
        print(f"       See {FLAGGED_PATH} for the list")
        failure_rate = summary['total_flagged'] / summary['total_pairs_found']
        if failure_rate > FAILURE_RATE_THRESHOLD:
            print(f"\n[ERROR] Too many failures ({failure_rate:.1%} > {FAILURE_RATE_THRESHOLD:.0%})")
        else:
            print(f"\n[OK] Validation passed with warnings (failures < {FAILURE_RATE_THRESHOLD:.0%})")
    else:
        print(f"\n[OK] All validations passed!")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────

def main() -> None:
    """Execute the complete dataset validation pipeline."""
    logger.info("Starting DeepPCB dataset validation")

    # Build manifest
    try:
        manifest = build_manifest(PCB_DATA_DIR)
    except RuntimeError as e:
        logger.error(f"Manifest build failed: {e}")
        sys.exit(1)

    logger.info(f"Built manifest with {len(manifest)} samples")

    # Run validations
    logger.info("Validating images...")
    img_results = validate_images(manifest)

    logger.info("Validating annotations...")
    anno_results = validate_annotations(manifest)

    # Get skip set for subsequent validations
    skip_stems = set(anno_results["failed_stems"])

    logger.info("Validating bounding boxes...")
    bbox_results = validate_bboxes(manifest, skip_stems)

    logger.info("Validating classes...")
    cls_results = validate_classes(manifest, skip_stems)

    # Collect all failed stems
    all_failed_stems = set()
    all_failed_stems.update(img_results["failed_stems"])
    all_failed_stems.update(anno_results["failed_stems"])
    all_failed_stems.update(bbox_results["failed_stems"])
    all_failed_stems.update(cls_results["failed_stems"])

    # Build summary
    summary = {
        "total_pairs_found": len(manifest),
        "total_flagged": len(all_failed_stems),
        "total_passed": len(manifest) - len(all_failed_stems)
    }

    # Compile all results
    results = {
        "summary": summary,
        "images": img_results,
        "annotations": anno_results,
        "bboxes": bbox_results,
        "classes": cls_results
    }

    # Write outputs
    write_validation_report(results, VALIDATION_REPORT_PATH)
    logger.info(f"Report written: {VALIDATION_REPORT_PATH}")
    write_flagged_samples(all_failed_stems, FLAGGED_PATH)

    # Verify round-trip of flagged stems file
    reloaded = load_flagged_stems(FLAGGED_PATH)
    assert len(reloaded) == len(all_failed_stems), (
        f"Flagged file write/read mismatch: "
        f"wrote {len(all_failed_stems)}, read back {len(reloaded)}"
    )
    logger.info("Flagged samples file verified (write/read round-trip passed)")

    # Print summary
    print_summary(summary, img_results, anno_results, bbox_results, cls_results)

    # Exit code
    failure_rate = len(all_failed_stems) / len(manifest) if manifest else 1.0
    if failure_rate > FAILURE_RATE_THRESHOLD:
        logger.error(f"Failure rate {failure_rate:.1%} exceeds threshold {FAILURE_RATE_THRESHOLD:.1%}")
        sys.exit(1)
    else:
        logger.info(f"Validation passed with failure rate {failure_rate:.1%}")
        sys.exit(0)


if __name__ == "__main__":
    main()