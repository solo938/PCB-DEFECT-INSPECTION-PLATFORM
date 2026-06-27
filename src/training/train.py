# src/training/train.py
"""
YOLOv8 Training Entrypoint for DeepPCB Defect Detection.

Trains a YOLOv8 model on the processed DeepPCB dataset.
Supports CPU, CUDA, and Apple Silicon (MPS) devices.

Usage:
    python -m src.training.train

Exits with code 0 if training completes successfully, 1 if errors occur.
"""

# ─────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────

import sys
from pathlib import Path
import yaml
from datetime import datetime

# ─────────────────────────────────────────────
# Third Party
# ─────────────────────────────────────────────

from ultralytics import YOLO

# ─────────────────────────────────────────────
# src.utils imports
# ─────────────────────────────────────────────

from src.utils import paths
from src.utils import logger
from src.utils import config

# ─────────────────────────────────────────────
# Paths from utils.paths
# ─────────────────────────────────────────────

LOGS_DIR = paths.LOGS_DIR
OUTPUTS_DIR = paths.OUTPUTS_DIR
CONFIG_DIR = paths.CONFIG_DIR
DATASET_YAML_PATH = paths.DATASET_YAML_PATH

# ─────────────────────────────────────────────
# Constants from utils.config
# ─────────────────────────────────────────────

DEFAULT_MODEL = config.DEFAULT_MODEL
DEFAULT_EPOCHS = config.DEFAULT_EPOCHS
DEFAULT_IMG_SIZE = config.DEFAULT_IMG_SIZE
DEFAULT_BATCH = config.DEFAULT_BATCH
DEFAULT_WORKERS = config.DEFAULT_WORKERS
NUM_CLASSES = config.NUM_CLASSES

# ─────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.setup_logging(log_file=LOGS_DIR / "training.log")
log = logger.get_logger(__name__)


# ─────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────

def load_training_config(config_path: Path) -> dict:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to training.yaml
    
    Returns:
        Dictionary with training configuration
    """
    if not config_path.exists():
        log.warning(f"Training config not found at {config_path}, using defaults")
        return {
            "model": {"name": DEFAULT_MODEL},
            "training": {
                "epochs": DEFAULT_EPOCHS,
                "batch": DEFAULT_BATCH,
                "imgsz": DEFAULT_IMG_SIZE,
                "workers": DEFAULT_WORKERS,
            },
            "device": {"type": "mps"},
            "output": {"dir": str(OUTPUTS_DIR / "training"), "name": "pcb_defect_yolov8"},
        }
    
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_type: str) -> str:
    """
    Get the appropriate device string for YOLO.
    
    Args:
        device_type: 'cpu', 'cuda', or 'mps'
    
    Returns:
        Device string for YOLO
    """
    if device_type == "mps":
        # Check if MPS is available (Apple Silicon)
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            else:
                log.warning("MPS not available, falling back to CPU")
                return "cpu"
        except ImportError:
            log.warning("PyTorch not installed, falling back to CPU")
            return "cpu"
    elif device_type == "cuda":
        return "cuda"
    else:
        return "cpu"


def train() -> dict:
    """
    Train a YOLOv8 model on the DeepPCB dataset.
    
    Returns:
        Dictionary with training results and metadata
    """
    log.info("Starting YOLOv8 training on DeepPCB dataset")
    log.info(f"Dataset config: {DATASET_YAML_PATH}")
    
    # Load training configuration
    training_config_path = CONFIG_DIR / "training.yaml"
    train_cfg = load_training_config(training_config_path)
    
    # Extract configuration
    model_name = train_cfg.get("model", {}).get("name", DEFAULT_MODEL)
    epochs = train_cfg.get("training", {}).get("epochs", DEFAULT_EPOCHS)
    batch = train_cfg.get("training", {}).get("batch", DEFAULT_BATCH)
    imgsz = train_cfg.get("training", {}).get("imgsz", DEFAULT_IMG_SIZE)
    workers = train_cfg.get("training", {}).get("workers", DEFAULT_WORKERS)
    device_type = train_cfg.get("device", {}).get("type", "mps")
    output_dir = train_cfg.get("output", {}).get("dir", str(OUTPUTS_DIR / "training"))
    experiment_name = train_cfg.get("output", {}).get("name", "pcb_defect_yolov8")
    
    # Determine device
    device = get_device(device_type)
    log.info(f"Using device: {device}")
    
    # Check if dataset config exists
    if not DATASET_YAML_PATH.exists():
        log.error(f"Dataset config not found: {DATASET_YAML_PATH}")
        log.error("Please create configs/dataset.yaml first")
        return {"error": "Dataset config not found"}
    
    # Initialize model
    log.info(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Start training
    log.info(f"Training configuration:")
    log.info(f"  Epochs: {epochs}")
    log.info(f"  Batch: {batch}")
    log.info(f"  Image size: {imgsz}")
    log.info(f"  Workers: {workers}")
    log.info(f"  Output: {output_dir}/{experiment_name}")
    
    start_time = datetime.now()
    
    try:
        results = model.train(
            data=str(DATASET_YAML_PATH),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            workers=workers,
            device=device,
            project=output_dir,
            name=experiment_name,
            exist_ok=True,
            verbose=True,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        log.info(f"Training completed in {duration:.1f} minutes")
        log.info(f"Results saved to: {output_dir}/{experiment_name}")
        
        return {
            "success": True,
            "model_path": f"{output_dir}/{experiment_name}/weights/best.pt",
            "duration_minutes": duration,
            "epochs": epochs,
            "device": device,
        }
        
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        return {"error": "Training interrupted"}
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def print_summary(result: dict) -> None:
    """Print training summary to console."""
    print("\n" + "="*50)
    print("YOLOv8 Training Complete")
    print("="*50)
    
    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        return
    
    print(f"\n[SUCCESS] Training completed")
    print(f"  Model saved: {result.get('model_path', 'N/A')}")
    print(f"  Duration: {result.get('duration_minutes', 0):.1f} minutes")
    print(f"  Device: {result.get('device', 'N/A')}")
    print(f"  Epochs: {result.get('epochs', 'N/A')}")
    print("\n" + "="*50 + "\n")


def main() -> None:
    """Execute the training pipeline."""
    try:
        result = train()
        print_summary(result)
        
        if "error" in result:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()