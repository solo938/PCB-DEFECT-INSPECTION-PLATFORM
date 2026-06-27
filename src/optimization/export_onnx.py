# src/optimization/export_onnx.py
"""
Export YOLOv8 model to ONNX format for deployment.

Usage:
    python -m src.optimization.export_onnx \
        --weights runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt \
        --imgsz 480
"""

import argparse
from pathlib import Path
import time

from ultralytics import YOLO

from src.utils.paths import OUTPUTS_DIR
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def export_to_onnx(
    weights_path: str,
    img_size: int = 480,
    output_dir: str = None,
    simplify: bool = True,
    opset: int = 12,
) -> Path:
    """
    Export YOLOv8 model to ONNX format.
    
    Args:
        weights_path: Path to the trained model weights
        img_size: Image size for the model
        output_dir: Directory to save the ONNX file
        simplify: Whether to simplify the ONNX model
        opset: ONNX opset version
    
    Returns:
        Path to the exported ONNX file
    """
    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Determine output path
    weights_path = Path(weights_path)
    if output_dir is None:
        output_dir = weights_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{weights_path.stem}.onnx"
    
    logger.info(f"Exporting to ONNX: {onnx_path}")
    logger.info(f"Image size: {img_size}x{img_size}")
    logger.info(f"OP set: {opset}")
    
    start_time = time.time()
    
    try:
        # Export to ONNX
        model.export(
            format="onnx",
            imgsz=img_size,
            opset=opset,
            simplify=simplify,
            name=onnx_path.stem,
        )
        
        elapsed = time.time() - start_time
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ ONNX export completed in {elapsed:.2f}s")
        logger.info(f"   File: {onnx_path}")
        logger.info(f"   Size: {size_mb:.2f} MB")
        
        return onnx_path
        
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        raise


def export_to_onnx_int8(
    weights_path: str,
    img_size: int = 480,
    output_dir: str = None,
    opset: int = 12,
) -> Path:
    """
    Export to INT8 quantized ONNX for further optimization.
    
    Args:
        weights_path: Path to the trained model weights
        img_size: Image size for the model
        output_dir: Directory to save the ONNX file
        opset: ONNX opset version
    
    Returns:
        Path to the exported INT8 ONNX file
    """
    logger.info(f"Exporting INT8 quantized ONNX...")
    
    weights_path = Path(weights_path)
    if output_dir is None:
        output_dir = weights_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    int8_path = output_dir / f"{weights_path.stem}_int8.onnx"
    
    try:
        # First export to FP32 ONNX
        onnx_path = export_to_onnx(
            weights_path,
            img_size=img_size,
            output_dir=output_dir,
            simplify=True,
            opset=opset,
        )
        
        # Then apply INT8 quantization
        # Note: This is a placeholder - full INT8 quantization requires
        # calibration data and onnxruntime quantization tools
        # For now, we'll use the FP32 model and note that INT8 is not yet implemented
        
        logger.info(f"INT8 quantization not implemented yet.")
        logger.info(f"Using FP32 ONNX: {onnx_path}")
        logger.info(f"For INT8, use onnxruntime quantization tools with calibration data.")
        
        return onnx_path
        
    except Exception as e:
        logger.error(f"❌ INT8 export failed: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="Image size for the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as weights)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8 quantized ONNX (experimental)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX simplification",
    )
    args = parser.parse_args()
    
    # Setup logging
    from src.utils.logger import setup_logging
    setup_logging()
    
    if args.int8:
        export_to_onnx_int8(
            args.weights,
            img_size=args.imgsz,
            output_dir=args.output,
            opset=args.opset,
        )
    else:
        export_to_onnx(
            args.weights,
            img_size=args.imgsz,
            output_dir=args.output,
            simplify=not args.no_simplify,
            opset=args.opset,
        )


if __name__ == "__main__":
    main()