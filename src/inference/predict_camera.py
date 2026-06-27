# src/inference/predict_camera.py
"""
Real-time camera inference for PCB defect detection.

Supports:
- Live camera feed
- Frame skipping for performance
- Bounding box visualization
- Detection logging
- Keyboard controls (q to quit, s to save)

Usage:
    python -m src.inference.predict_camera \
        --weights runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt \
        --camera 0 \
        --skip 2
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from src.inference.predictor import PCBDetector
from src.inference.visualize import draw_detections
from src.utils.paths import OUTPUTS_DIR
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class CameraInference:
    """
    Real-time camera inference for PCB defect detection.
    """
    
    def __init__(
        self,
        weights_path: str,
        camera_id: int = 0,
        frame_skip: int = 2,
        conf_threshold: float = 0.45,
        device: str = "mps",
        img_size: int = 480,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize camera inference.
        
        Args:
            weights_path: Path to model weights
            camera_id: Camera device ID
            frame_skip: Process every Nth frame (1 = process all)
            conf_threshold: Confidence threshold
            device: Device for inference
            img_size: Image size for inference
            save_dir: Directory to save captured frames
        """
        self.weights_path = weights_path
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.conf_threshold = conf_threshold
        self.device = device
        self.img_size = img_size
        self.save_dir = save_dir
        
        # Initialize detector
        self.detector = PCBDetector(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            device=device,
            img_size=img_size,
        )
        
        # Frame counter
        self.frame_count = 0
        self.detection_count = 0
        
        # FPS tracking
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Stats
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_detections": 0,
            "avg_inference_ms": 0,
            "fps": 0,
        }
        
        logger.info(f"Camera inference initialized (camera {camera_id})")
        logger.info(f"Frame skip: {frame_skip} (process every {frame_skip}th frame)")
    
    def run(self) -> None:
        """Run the camera inference loop."""
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera: {width}x{height} @ {fps:.1f}FPS")
        
        # Create window
        window_name = "PCB Defect Detection - Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\n" + "=" * 60)
        print("CAMERA INFERENCE")
        print("=" * 60)
        print("Controls:")
        print("  'q' or 'ESC' - Quit")
        print("  's'           - Save current frame")
        print("  'r'           - Reset stats")
        print("  'f'           - Toggle fullscreen")
        print("=" * 60 + "\n")
        
        inference_times = []
        last_log_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, exiting...")
                    break
                
                self.frame_count += 1
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process every N-th frame
                if self.frame_count % self.frame_skip == 0:
                    self.processed_frames += 1
                    
                    # Run inference
                    start = time.perf_counter()
                    result = self.detector.predict(frame_rgb)
                    end = time.perf_counter()
                    
                    inference_time = (end - start) * 1000
                    inference_times.append(inference_time)
                    
                    self.total_detections += result["num_detections"]
                    
                    # Draw detections
                    annotated = draw_detections(frame_rgb, result["detections"])
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    
                    # Update stats
                    self.stats["total_frames"] = self.frame_count
                    self.stats["processed_frames"] = self.processed_frames
                    self.stats["total_detections"] = self.total_detections
                    
                    if inference_times:
                        self.stats["avg_inference_ms"] = np.mean(inference_times[-100:])
                    
                    # Update FPS
                    self.fps_counter += 1
                    if time.time() - self.fps_start_time >= 1.0:
                        self.stats["fps"] = self.fps_counter
                        self.fps_counter = 0
                        self.fps_start_time = time.time()
                    
                    # Add info overlay
                    self._add_overlay(annotated_bgr, result)
                    
                    display_frame = annotated_bgr
                
                else:
                    # Display frame without processing
                    display_frame = frame
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    self._save_frame(annotated_bgr if self.processed_frames > 0 else frame)
                elif key == ord('r'):  # Reset stats
                    self._reset_stats()
                elif key == ord('f'):  # Toggle fullscreen
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_FULLSCREEN)
                
                # Log every 60 seconds
                if time.time() - last_log_time >= 60:
                    logger.info(
                        f"Frames: {self.frame_count} | "
                        f"FPS: {self.stats['fps']:.1f} | "
                        f"Detections: {self.total_detections} | "
                        f"Avg: {self.stats['avg_inference_ms']:.1f}ms"
                    )
                    last_log_time = time.time()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save final stats
            self._save_stats()
            
            logger.info("Camera inference stopped")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Processed frames: {self.processed_frames}")
            logger.info(f"Total detections: {self.total_detections}")
            logger.info(f"Avg inference time: {self.stats['avg_inference_ms']:.1f}ms")
            logger.info(f"FPS: {self.stats['fps']:.1f}")
    
    def _add_overlay(self, image: np.ndarray, result: dict) -> None:
        """Add information overlay to the frame."""
        h, w = image.shape[:2]
        
        # Background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 90), (0, 0, 0), -1)
        image[10:100, 10:310] = overlay[10:100, 10:310] * 0.6 + 255 * 0.4
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        y = 35
        cv2.putText(image, f"FPS: {self.stats['fps']:.1f}", (20, y), font, font_scale, color, thickness)
        cv2.putText(image, f"Detections: {result['num_detections']}", (20, y + 20), font, font_scale, color, thickness)
        cv2.putText(image, f"Inference: {result['inference_time_ms']:.1f}ms", (20, y + 40), font, font_scale, color, thickness)
        cv2.putText(image, f"Frames: {self.frame_count}", (20, y + 60), font, font_scale, color, thickness)
    
    def _save_frame(self, frame: np.ndarray) -> None:
        """Save current frame to disk."""
        if self.save_dir is None:
            self.save_dir = OUTPUTS_DIR / "camera_captures"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"capture_{timestamp}.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), frame)
        logger.info(f"Frame saved: {save_path}")
    
    def _save_stats(self) -> None:
        """Save inference statistics to JSON."""
        if self.save_dir is None:
            self.save_dir = OUTPUTS_DIR / "camera_captures"
        
        stats_path = self.save_dir / "camera_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with stats_path.open("w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "device": self.device,
                "camera_id": self.camera_id,
                "frame_skip": self.frame_skip,
                "conf_threshold": self.conf_threshold,
                "total_frames": self.frame_count,
                "processed_frames": self.processed_frames,
                "total_detections": self.total_detections,
                "avg_inference_ms": self.stats.get("avg_inference_ms", 0),
                "fps": self.stats.get("fps", 0),
            }, f, indent=2)
        
        logger.info(f"Stats saved: {stats_path}")
    
    def _reset_stats(self) -> None:
        """Reset statistics counters."""
        self.frame_count = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_detections": 0,
            "avg_inference_ms": 0,
            "fps": 0,
        }
        logger.info("Stats reset")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Camera inference for PCB defect detection")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=2,
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
        "--save",
        type=str,
        default="outputs/camera_captures",
        help="Directory to save captured frames",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Initialize and run
    inference = CameraInference(
        weights_path=args.weights,
        camera_id=args.camera,
        frame_skip=args.skip,
        conf_threshold=args.conf,
        device=args.device,
        img_size=args.imgsz,
        save_dir=Path(args.save),
    )
    
    try:
        inference.run()
    except RuntimeError as e:
        logger.error(f"Camera error: {e}")
        return


if __name__ == "__main__":
    main()