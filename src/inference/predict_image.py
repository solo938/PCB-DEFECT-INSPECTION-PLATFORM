# src/inference/predict_image.py
"""
Single image inference CLI.
"""

import argparse
import json
from pathlib import Path

import cv2

from src.inference.predictor import PCBDetector
from src.inference.visualize import draw_detections, save_annotated_image
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict defects in a single image")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file",
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
        "--save",
        action="store_true",
        help="Save annotated image",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON export",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Initialize detector
    detector = PCBDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
    )
    
    # Run inference
    image_path = Path(args.image)
    result = detector.predict(image_path)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Image: {result['image_name']}")
    print(f"Detections: {result['num_detections']}")
    print(f"Inference time: {result['inference_time_ms']:.1f}ms")
    print("-" * 60)
    
    for det in result["detections"]:
        print(f"  {det['class_name']}: {det['confidence']:.3f} @ {det['bbox']}")
    
    print("=" * 60)
    
    # Save results
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image for annotation
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save annotated image
        save_annotated_image(image, result["detections"], output_dir / f"{image_path.stem}_annotated{image_path.suffix}")
        logger.info(f"Annotated image saved to {output_dir}")
        
        # Save JSON
        if not args.no_json:
            json_path = output_dir / f"{image_path.stem}.json"
            with json_path.open("w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Predictions saved to {json_path}")


if __name__ == "__main__":
    main()