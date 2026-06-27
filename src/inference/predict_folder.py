# src/inference/predict_folder.py
"""
Folder inference CLI.
"""

import argparse
from pathlib import Path

from src.inference.predictor import PCBDetector
from src.inference.batch import run_batch_inference
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict defects in a folder of images")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Output directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="Image size",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving annotated images",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving JSON predictions",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Initialize detector
    detector = PCBDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        img_size=args.imgsz,
    )
    
    # Run batch inference
    summary = run_batch_inference(
        detector,
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        save_images=not args.no_save_images,
        save_json=not args.no_save_json,
    )
    
    print("\n" + "=" * 60)
    print("BATCH INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total images: {summary['total_images']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Avg detections per image: {summary['avg_detections']:.2f}")
    print(f"Avg inference time: {summary['avg_inference_ms']:.1f}ms")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()