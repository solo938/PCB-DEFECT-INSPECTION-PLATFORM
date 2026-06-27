from pathlib import Path
import shutil
import random
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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
    Sample as ValidationSample,
    build_manifest as build_validation_manifest,
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
PROCESSED_DIR = paths.PROCESSED_DIR
LABELS_DIR = OUTPUTS_DIR / "labels"
SPLIT_MANIFEST_PATH = paths.SPLIT_MANIFEST_PATH

# ─────────────────────────────────────────────
# Constants from utils.config
# ─────────────────────────────────────────────

TRAIN_RATIO = config.TRAIN_RATIO
VAL_RATIO = config.VAL_RATIO
TEST_RATIO = config.TEST_RATIO
SPLIT_SEED = config.SPLIT_SEED
TEST_SUFFIX = config.TEST_SUFFIX
TEMP_SUFFIX = config.TEMP_SUFFIX
ANNOTATION_SUFFIX = config.ANNOTATION_SUFFIX

# ─────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.setup_logging(log_file=LOGS_DIR / "split.log")
log = logger.get_logger(__name__)


# ─────────────────────────────────────────────
# Part 2 — Dataclass
# ─────────────────────────────────────────────

@dataclass
class Sample:
    """Represents one complete DeepPCB sample for splitting."""
    stem: str
    group: str
    test_image: Path
    template_image: Path
    label: Path
    dominant_class: int


# ─────────────────────────────────────────────
# Part 3 — Helper Functions
# ─────────────────────────────────────────────

def find_triplet(stem: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Find test image, template image, and label for a given stem.
    
    Args:
        stem: Sample stem (e.g., '00041001')
    
    Returns:
        Tuple of (test_image, template_image, label_path)
        Missing files are returned as None
    """
    # Labels are in data/labels/
    label_path = LABELS_DIR / f"{stem}{ANNOTATION_SUFFIX}"
    
    # Images are still in PCBData
    # We need to find which group this stem belongs to
    # For now, we'll search for it in the manifest from validation
    # This will be handled differently in build_manifest
    
    # We'll find images in build_manifest using the validated manifest
    # So this function is simplified - it only finds the label
    if label_path.exists():
        return (None, None, label_path)
    else:
        return (None, None, None)


def get_dominant_class(label_path: Path) -> int:
    """
    Read a YOLO label file and return the most common class.
    
    Args:
        label_path: Path to YOLO label file
    
    Returns:
        Most common class ID (0-indexed)
    """
    try:
        lines = read_annotation_lines(label_path)
        if not lines:
            log.warning(f"Empty label file: {label_path}")
            return -1
        
        # Count classes (first token of each line)
        class_counts = Counter()
        for line in lines:
            parts = line.strip().split()
            if parts:
                try:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                except ValueError:
                    log.warning(f"Invalid class in {label_path}: {line}")
                    continue
        
        if not class_counts:
            return -1
        
        # Return most common class
        return class_counts.most_common(1)[0][0]
        
    except Exception as e:
        log.error(f"Error reading {label_path}: {e}")
        return -1


def build_manifest() -> List[Sample]:
    """
    Build a manifest of samples to split.
    
    Uses the validation manifest and filters out flagged samples.
    
    Returns:
        List of Sample objects
    """
    log.info("Building manifest...")
    
    # First, get validation manifest
    validation_samples = build_validation_manifest(PCB_DATA_DIR)
    log.info(f"Found {len(validation_samples)} samples from validation")
    
    # Load flagged samples
    flagged_stems = load_flagged_stems(FLAGGED_PATH)
    log.info(f"Found {len(flagged_stems)} flagged samples to skip")
    
    # Build split manifest
    samples = []
    skipped = 0
    
    for v_sample in validation_samples:
        # Skip flagged samples
        if v_sample.stem in flagged_stems:
            skipped += 1
            continue
        
        # Find label file
        label_path = LABELS_DIR / f"{v_sample.stem}{ANNOTATION_SUFFIX}"
        if not label_path.exists():
            log.warning(f"Label file missing for {v_sample.stem}: {label_path}")
            continue
        
        # Get dominant class
        dominant_class = get_dominant_class(label_path)
        if dominant_class == -1:
            log.warning(f"Could not determine dominant class for {v_sample.stem}")
            continue
        
        # Create sample
        samples.append(Sample(
            stem=v_sample.stem,
            group=v_sample.group,
            test_image=v_sample.test_image,
            template_image=v_sample.template_image,
            label=label_path,
            dominant_class=dominant_class
        ))
    
    log.info(f"Built manifest with {len(samples)} samples ({skipped} skipped)")
    return samples


# ─────────────────────────────────────────────
# Part 4 — Stratified Split
# ─────────────────────────────────────────────

def split_samples(samples: List[Sample]) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Perform stratified split based on dominant class.
    
    Args:
        samples: List of Sample objects
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    log.info("Creating stratified split...")
    
    # Group samples by dominant class
    class_groups: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        class_groups[sample.dominant_class].append(sample)
    
    # Initialize splits
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Create seeded randomizer for reproducibility
    rng = random.Random(SPLIT_SEED)
    
    # Split each class group
    for class_id, class_samples in sorted(class_groups.items()):
        # Shuffle with seed
        rng.shuffle(class_samples)
        
        n = len(class_samples)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val  # Remainder goes to test
        
        # Split
        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train:n_train + n_val])
        test_samples.extend(class_samples[n_train + n_val:])
        
        log.debug(f"Class {class_id}: {n_train} train, {n_val} val, {n_test} test")
    
    # Shuffle each split (different seeds to avoid correlation)
    train_rng = random.Random(SPLIT_SEED + 1)
    val_rng = random.Random(SPLIT_SEED + 2)
    test_rng = random.Random(SPLIT_SEED + 3)
    
    train_rng.shuffle(train_samples)
    val_rng.shuffle(val_samples)
    test_rng.shuffle(test_samples)
    
    log.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    return train_samples, val_samples, test_samples


# ─────────────────────────────────────────────
# Part 5 — Copy Files
# ─────────────────────────────────────────────

def copy_split(split_name: str, samples: List[Sample], processed_root: Path) -> Dict[str, int]:
    """
    Copy files for a split to the processed directory.
    
    Args:
        split_name: 'train', 'val', or 'test'
        samples: List of Sample objects
        processed_root: Root processed directory
    
    Returns:
        Dict with counts of copied files
    """
    # Create directories
    split_dir = processed_root / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    templates_dir = split_dir / "templates"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "images": 0,
        "labels": 0,
        "templates": 0
    }
    
    for sample in samples:
        # Copy test image
        dest_image = images_dir / f"{sample.stem}{TEST_SUFFIX}"
        shutil.copy2(sample.test_image, dest_image)
        stats["images"] += 1
        
        # Copy template image
        dest_template = templates_dir / f"{sample.stem}{TEMP_SUFFIX}"
        shutil.copy2(sample.template_image, dest_template)
        stats["templates"] += 1
        
        # Copy label
        dest_label = labels_dir / f"{sample.stem}{ANNOTATION_SUFFIX}"
        shutil.copy2(sample.label, dest_label)
        stats["labels"] += 1
    
    log.info(f"{split_name.capitalize()} copied: {stats['images']} images, "
             f"{stats['labels']} labels, {stats['templates']} templates")
    return stats


# ─────────────────────────────────────────────
# Part 6 — Split Manifest
# ─────────────────────────────────────────────

def write_manifest(train: List[Sample], val: List[Sample], test: List[Sample], path: Path) -> None:
    """
    Write split manifest to JSON file.
    
    Format: {"stem": "split_name", ...}
    """
    manifest = {}
    
    for sample in train:
        manifest[sample.stem] = "train"
    for sample in val:
        manifest[sample.stem] = "val"
    for sample in test:
        manifest[sample.stem] = "test"
    
    # Sort by stem for readability
    sorted_manifest = dict(sorted(manifest.items()))
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted_manifest, f, indent=2)
    
    log.info(f"Split manifest written to {path}")


# ─────────────────────────────────────────────
# Part 7 — Console Output
# ─────────────────────────────────────────────

def print_summary(samples: List[Sample], train: List[Sample], val: List[Sample], 
                  test: List[Sample], stats: Dict[str, Dict[str, int]]) -> None:
    """Print split summary to console."""
    print("\n" + "="*50)
    print("DeepPCB Dataset Split")
    print("="*50)
    
    print(f"\nLoading flagged samples...")
    flagged_stems = load_flagged_stems(FLAGGED_PATH)
    print(f"{len(flagged_stems)} skipped")
    
    print(f"\nBuilding manifest...")
    print(f"{len(samples)} samples")
    
    print(f"\nCreating stratified split...")
    print(f"\nTrain : {len(train)}")
    print(f"Val   : {len(val)}")
    print(f"Test  : {len(test)}")
    
    print(f"\nCopying files...")
    for split_name, split_stats in stats.items():
        print(f"  {split_name.capitalize()}: {split_stats['images']} images, "
              f"{split_stats['labels']} labels, {split_stats['templates']} templates")
    
    print(f"\nWriting split manifest...")
    print(f"\nDone.")
    print("\n" + "="*50)
    print("Dataset ready for augmentation.")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────

def main() -> None:
    """Execute the dataset split pipeline."""
    try:
        log.info("Starting DeepPCB dataset split")
        
        # Build manifest
        samples = build_manifest()
        if not samples:
            log.error("No samples found to split")
            sys.exit(1)
        
        # Create stratified split
        train, val, test = split_samples(samples)
        
        # Verify split sizes
        total = len(train) + len(val) + len(test)
        if total != len(samples):
            log.warning(f"Split size mismatch: {total} vs {len(samples)}")
        
        # Copy files
        stats = {}
        stats["train"] = copy_split("train", train, PROCESSED_DIR)
        stats["val"] = copy_split("val", val, PROCESSED_DIR)
        stats["test"] = copy_split("test", test, PROCESSED_DIR)
        
        # Write manifest
        write_manifest(train, val, test, SPLIT_MANIFEST_PATH)
        
        # Print summary
        print_summary(samples, train, val, test, stats)
        
        # Exit
        log.info("Split completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        log.info("Split interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error(f"Split failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
