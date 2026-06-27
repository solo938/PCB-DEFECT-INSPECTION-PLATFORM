# src/inference/predict_video.py
"""
Video inference for PCB defect detection.

Supports:
- Video files (.mp4, .avi, .mov)
- Frame skipping for performance
- Output video with annotations
- Progress tracking
- Detection logging

Usage:
    python -m src.inference.predict_video \
        --weights runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt \
        --source path/to/video.mp4 \
        --output outputs/video_output.mp4
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.inference.predictor import PCBDetector
from src.inference.visualize import draw_detections
from src.utils.paths import OUTPUTS_DIR
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class VideoInference:
    """
    Video inference for PCB defect detection.
    """
    
    def __init__(
        self,
        weights_path: str,
        source_path: Path,
        output_path: Optional[Path] = None,
        frame_skip: int = 1,
        conf_threshold: float = 0.45,
        device: str = "mps",
        img_size: int = 480,
        show_progress: bool = True,
    ):
        """
        Initialize video inference.
        
        Args:
            weights_path: Path to model weights
            source_path: Path to input video
            output_path: Path to output video
            frame_skip: Process every Nth frame
            conf_threshold: Confidence threshold
            device: Device for inference
            img_size: Image size for inference
            show_progress: Show progress bar
        """
        self.weights_path = weights_path
        self.source_path = Path(source_path)
        self.output_path = output_path
        self.frame_skip = frame_skip
        self.conf_threshold = conf_threshold
        self.device = device
        self.img_size = img_size
        self.show_progress = show_progress
        
        # Initialize detector
        self.detector = PCBDetector(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            device=device,
            img_size=img_size,
        )
        
        # Stats
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_detections": 0,
            "avg_inference_ms": 0,
            "fps": 0,
            "video_duration_s": 0,
            "processing_time_s": 0,
        }
        
        # Inference times for stats
        self.inference_times = []
        
        logger.info(f"Video inference initialized")
        logger.info(f"Source: {self.source_path}")
        logger.info(f"Frame skip: {frame_skip} (process every {frame_skip}th frame)")
    
    def run(self) -> None:
        """Process the video."""
        # Open video
        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.source_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.stats["total_frames"] = total_frames
        self.stats["video_duration_s"] = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {width}x{height}, {total_frames} frames, {fps:.1f}FPS")
        logger.info(f"Duration: {self.stats['video_duration_s']:.1f}s")
        
        # Setup video writer
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        else:
            writer = None
        
        # Process frames
        processed_count = 0
        frame_count = 0
        inference_times = []
        start_time = time.time()
        
        # Progress bar
        iterator = range(total_frames)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Processing video", unit="frames")
        
        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process every N-th frame
            if frame_count % self.frame_skip == 0:
                processed_count += 1
                
                # Run inference
                start = time.perf_counter()
                result = self.detector.predict(frame_rgb)
                end = time.perf_counter()
                
                inference_time = (end - start) * 1000
                inference_times.append(inference_time)
                
                self.stats["total_detections"] += result["num_detections"]
                
                # Draw detections
                annotated = draw_detections(frame_rgb, result["detections"])
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                
                display_frame = annotated_bgr
            else:
                display_frame = frame
            
            # Write frame
            if writer:
                writer.write(display_frame)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        # Update stats
        self.stats["processed_frames"] = processed_count
        self.stats["processing_time_s"] = time.time() - start_time
        
        if inference_times:
            self.stats["avg_inference_ms"] = np.mean(inference_times)
            self.stats["fps"] = 1000 / self.stats["avg_inference_ms"] if self.stats["avg_inference_ms"] > 0 else 0
        
        # Log results
        logger.info("Video processing complete!")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Processed frames: {processed_count}")
        logger.info(f"  Total detections: {self.stats['total_detections']}")
        logger.info(f"  Avg inference: {self.stats['avg_inference_ms']:.1f}ms")
        logger.info(f"  Processing time: {self.stats['processing_time_s']:.1f}s")
        
        if self.output_path:
            logger.info(f"  Output saved: {self.output_path}")
    
    def _save_stats(self) -> None:
        """Save inference statistics to JSON."""
        stats_path = self.source_path.parent / f"{self.source_path.stem}_stats.json"
        
        with stats_path.open("w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "source": str(self.source_path),
                "output": str(self.output_path) if self.output_path else None,
                "device": self.device,
                "conf_threshold": self.conf_threshold,
                "frame_skip": self.frame_skip,
                **self.stats,
            }, f, indent=2)
        
        logger.info(f"Stats saved: {stats_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Video inference for PCB defect detection")
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
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output video file (optional)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Frame skip (process every Nth frame)",
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
        help="Image size for inference",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Determine output path
    source_path = Path(args.source)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = source_path.parent / f"{source_path.stem}_annotated{source_path.suffix}"
    
    # Initialize and run
    inference = VideoInference(
        weights_path=args.weights,
        source_path=source_path,
        output_path=output_path,
        frame_skip=args.skip,
        conf_threshold=args.conf,
        device=args.device,
        img_size=args.imgsz,
        show_progress=not args.no_progress,
    )
    
    try:
        inference.run()
        inference._save_stats()
    except RuntimeError as e:
        logger.error(f"Video processing error: {e}")
        return


if __name__ == "__main__":
    main()