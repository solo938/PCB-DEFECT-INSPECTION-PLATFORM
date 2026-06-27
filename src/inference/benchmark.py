# src/inference/benchmark.py
"""
Inference performance benchmarking for PCB defect detection.

Measures:
- Throughput (FPS)
- Latency (mean, p50, p95, p99)
- Memory usage
- CPU/GPU utilization

Usage:
    python -m src.inference.benchmark \
        --weights runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt \
        --source data/processed/test/images/ \
        --runs 100
"""

import argparse
import time
import json
import psutil
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.inference.predictor import PCBDetector
from src.utils.paths import OUTPUTS_DIR
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def get_memory_usage() -> Dict:
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "percent": process.memory_percent(),
    }


def benchmark_inference(
    weights_path: str,
    source_path: Path,
    n_runs: int = 100,
    conf_threshold: float = 0.45,
    device: str = "mps",
    img_size: int = 480,
    warmup_runs: int = 10,
) -> Dict:
    """
    Run inference benchmark on a set of images.
    
    Args:
        weights_path: Path to model weights
        source_path: Path to image or folder
        n_runs: Number of inference runs
        conf_threshold: Confidence threshold
        device: Device for inference
        img_size: Image size
        warmup_runs: Number of warmup runs
    
    Returns:
        Benchmark results dictionary
    """
    logger.info(f"Starting benchmark with {n_runs} runs...")
    
    # Initialize detector
    detector = PCBDetector(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        device=device,
        img_size=img_size,
    )
    
    # Load images
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        image_paths = image_paths[:n_runs]  # Limit to n_runs
    
    if not image_paths:
        raise ValueError(f"No images found in {source_path}")
    
    logger.info(f"Using {len(image_paths)} images for benchmarking")
    
    # Pre-load images
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    if not images:
        raise ValueError("Failed to load any images")
    
    # Warmup
    logger.info(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        detector.predict(images[0])
    
    # Benchmark
    logger.info(f"Running benchmark ({n_runs} runs)...")
    
    times = []
    detections_counts = []
    memory_samples = []
    
    for i in tqdm(range(n_runs), desc="Running inference"):
        img = images[i % len(images)]
        
        # Measure memory before
        mem_before = get_memory_usage()
        
        # Run inference with timing
        start = time.perf_counter()
        result = detector.predict(img)
        end = time.perf_counter()
        
        # Measure memory after
        mem_after = get_memory_usage()
        
        times.append((end - start) * 1000)  # Convert to ms
        detections_counts.append(result["num_detections"])
        memory_samples.append({
            "before": mem_before,
            "after": mem_after,
        })
    
    # Calculate statistics
    times_sorted = sorted(times)
    times_np = np.array(times)
    detections_np = np.array(detections_counts)
    
    stats = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000 / np.mean(times)),
        "total_runs": int(n_runs),
        "warmup_runs": int(warmup_runs),
        "image_size": int(img_size),
        "device": str(device),
        "model_type": str(detector.model_type),
    }
    
    # Memory stats
    memory_after = [m["after"] for m in memory_samples]
    stats["memory"] = {
        "rss_mean_mb": float(np.mean([m["rss_mb"] for m in memory_after])),
        "rss_max_mb": float(np.max([m["rss_mb"] for m in memory_after])),
        "percent_mean": float(np.mean([m["percent"] for m in memory_after])),
    }
    
    # Detection stats
    stats["detections"] = {
        "mean": float(np.mean(detections_counts)),
        "std": float(np.std(detections_counts)),
        "min": int(np.min(detections_counts)),
        "max": int(np.max(detections_counts)),
    }
    
    logger.info(f"Benchmark complete!")
    logger.info(f"  Mean: {stats['mean_ms']:.2f}ms ({stats['fps']:.1f} FPS)")
    logger.info(f"  P95: {stats['p95_ms']:.2f}ms")
    logger.info(f"  P99: {stats['p99_ms']:.2f}ms")
    
    return stats


def plot_benchmark_results(
    stats: Dict,
    output_path: Path,
) -> None:
    """
    Generate benchmark visualization plots.
    
    Args:
        stats: Benchmark results from benchmark_inference
        output_path: Path to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latency bar chart
    ax1 = axes[0]
    metrics = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    values = [stats.get(m, 0) for m in metrics]
    labels = ["Mean", "P50", "P95", "P99"]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    
    bars = ax1.bar(labels, values, color=colors)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(f"Inference Latency - {stats.get('device', 'Unknown')}")
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=10)
    
    # FPS display
    ax2 = axes[1]
    fps = stats.get("fps", 0)
    ax2.text(0.5, 0.6, f"{fps:.1f}", fontsize=48, ha="center", va="center",
             fontweight="bold", color="#2ecc71")
    ax2.text(0.5, 0.4, "FPS", fontsize=24, ha="center", va="center", color="#555")
    
    # Add metadata
    metadata = [
        f"Device: {stats.get('device', 'Unknown')}",
        f"Image size: {stats.get('image_size', 0)}x{stats.get('image_size', 0)}",
        f"Runs: {stats.get('total_runs', 0)}",
        f"Model: {stats.get('model_type', 'Unknown')}",
    ]
    for i, line in enumerate(metadata):
        ax2.text(0.5, 0.25 - i * 0.06, line, fontsize=10, ha="center", va="center", color="#555")
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Benchmark plot saved to {output_path}")


def save_benchmark_results(stats: Dict, output_path: Path) -> None:
    """Save benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    serializable_stats = convert_to_serializable(stats)
    
    with output_path.open("w") as f:
        json.dump(serializable_stats, f, indent=2)
    logger.info(f"Benchmark results saved to {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Inference benchmark")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/test/images/",
        help="Source image or folder",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of inference runs",
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
        "--output",
        type=str,
        default="outputs/benchmarks",
        help="Output directory",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Run benchmark
    stats = benchmark_inference(
        weights_path=args.weights,
        source_path=Path(args.source),
        n_runs=args.runs,
        conf_threshold=args.conf,
        device=args.device,
        img_size=args.imgsz,
    )
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_benchmark_results(stats, output_dir / "benchmark_results.json")
    
    # Generate plots
    plot_benchmark_results(stats, output_dir / "benchmark_plot.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device:          {stats['device']}")
    print(f"Model type:      {stats['model_type']}")
    print(f"Image size:      {stats['image_size']}x{stats['image_size']}")
    print(f"Runs:            {stats['total_runs']}")
    print("-" * 60)
    print(f"Mean latency:    {stats['mean_ms']:.2f}ms")
    print(f"P50 latency:     {stats['p50_ms']:.2f}ms")
    print(f"P95 latency:     {stats['p95_ms']:.2f}ms")
    print(f"P99 latency:     {stats['p99_ms']:.2f}ms")
    print(f"FPS:             {stats['fps']:.1f}")
    print("-" * 60)
    print(f"Memory (RSS):    {stats['memory']['rss_mean_mb']:.1f}MB")
    print(f"Detections avg:  {stats['detections']['mean']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()