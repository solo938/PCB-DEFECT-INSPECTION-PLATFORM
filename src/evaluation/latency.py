# src/evaluation/latency.py
"""
Latency benchmarking for PyTorch, ONNX, and INT8 models.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

import torch
import onnxruntime as ort

from ultralytics import YOLO

from src.utils.paths import OUTPUTS_DIR
from src.utils.logger import get_logger
from src.evaluation.utils import BENCHMARKS_DIR

logger = get_logger(__name__)


def benchmark_pytorch(
    weights_path: str,
    img_size: int = 480,
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "mps",
) -> Dict:
    """Benchmark PyTorch model."""
    logger.info("Benchmarking PyTorch FP32...")
    
    # Load model and move to device
    model = YOLO(weights_path)
    
    # Get the underlying PyTorch model and move to device
    if hasattr(model, 'model'):
        model.model.to(device)
        # Set to eval mode
        model.model.eval()
    
    # Create dummy input on the same device
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.model(dummy_input)
    
    # Synchronize device before timing
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.model(dummy_input)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        "format": "PyTorch FP32",
        "device": device,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "fps": 1000 / np.mean(times),
        "n_runs": n_runs,
    }


def benchmark_onnx(
    onnx_path: str,
    img_size: int = 480,
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict:
    """Benchmark ONNX model."""
    logger.info(f"Benchmarking ONNX FP32 from {onnx_path}...")
    
    if not Path(onnx_path).exists():
        logger.warning(f"ONNX file not found: {onnx_path}")
        return None
    
    # Create inference session with CPU (MPS not available for ONNX)
    session = ort.InferenceSession(
        onnx_path, 
        providers=["CPUExecutionProvider"]
    )
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "format": "ONNX FP32",
        "device": "CPU",
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "fps": 1000 / np.mean(times),
        "n_runs": n_runs,
    }


def benchmark_onnx_int8(
    int8_path: str,
    img_size: int = 480,
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Optional[Dict]:
    """Benchmark INT8 quantized ONNX model."""
    logger.info(f"Benchmarking ONNX INT8 from {int8_path}...")
    
    if not Path(int8_path).exists():
        logger.warning(f"INT8 ONNX file not found: {int8_path}")
        return None
    
    try:
        session = ort.InferenceSession(
            int8_path, 
            providers=["CPUExecutionProvider"]
        )
        
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(warmup_runs):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            "format": "ONNX INT8",
            "device": "CPU",
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "fps": 1000 / np.mean(times),
            "n_runs": n_runs,
        }
    except Exception as e:
        logger.warning(f"INT8 benchmark failed: {e}")
        return None


def get_model_size(path: str) -> float:
    """Get model file size in MB."""
    if Path(path).exists():
        return Path(path).stat().st_size / (1024 * 1024)
    return 0.0


def run_full_benchmark(
    weights_dir: str,
    img_size: int = 480,
    n_runs: int = 100,
    device: str = "mps",
) -> Dict:
    """Run all benchmarks and collect results."""
    weights_dir = Path(weights_dir)
    
    results = {}
    
    # PyTorch
    pytorch_path = weights_dir / "best.pt"
    if pytorch_path.exists():
        results["pytorch"] = benchmark_pytorch(
            str(pytorch_path), img_size, n_runs, device=device
        )
        results["pytorch"]["size_mb"] = get_model_size(str(pytorch_path))
        results["pytorch"]["file_path"] = str(pytorch_path)
    else:
        logger.warning(f"PyTorch model not found: {pytorch_path}")
    
    # ONNX FP32
    onnx_path = weights_dir / "best.onnx"
    if onnx_path.exists():
        results["onnx"] = benchmark_onnx(str(onnx_path), img_size, n_runs)
        if results["onnx"]:
            results["onnx"]["size_mb"] = get_model_size(str(onnx_path))
            results["onnx"]["file_path"] = str(onnx_path)
    else:
        logger.warning(f"ONNX model not found: {onnx_path}")
    
    # ONNX INT8
    int8_path = weights_dir / "best_int8.onnx"
    if int8_path.exists():
        results["int8"] = benchmark_onnx_int8(str(int8_path), img_size, n_runs)
        if results["int8"]:
            results["int8"]["size_mb"] = get_model_size(str(int8_path))
            results["int8"]["file_path"] = str(int8_path)
    
    return results


def print_benchmark_table(results: Dict) -> None:
    """Print benchmark results as a table."""
    print("\n" + "=" * 70)
    print("INFERENCE LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Image size: 480x480   Runs: 100")
    print("-" * 70)
    print(f"{'Format':<18} {'Device':<10} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'FPS':<10} {'Size (MB)':<10}")
    print("-" * 70)
    
    for key, data in results.items():
        if data:
            device = data.get('device', 'N/A')
            print(f"{data.get('format', key):<18} "
                  f"{device:<10} "
                  f"{data.get('mean_ms', 0):<12.2f} "
                  f"{data.get('p95_ms', 0):<12.2f} "
                  f"{data.get('p99_ms', 0):<12.2f} "
                  f"{data.get('fps', 0):<10.1f} "
                  f"{data.get('size_mb', 0):<10.2f}")
    
    print("=" * 70)
    
    # Speedup calculations
    if "pytorch" in results and results["pytorch"] and "onnx" in results and results["onnx"]:
        speedup = results["pytorch"]["mean_ms"] / results["onnx"]["mean_ms"]
        print(f"\nSpeedup ONNX vs PyTorch:    {speedup:.2f}x")
    
    if "pytorch" in results and results["pytorch"] and "int8" in results and results["int8"]:
        speedup_int8 = results["pytorch"]["mean_ms"] / results["int8"]["mean_ms"]
        print(f"Speedup INT8 vs PyTorch:    {speedup_int8:.2f}x")
        
        size_reduction = results["pytorch"]["size_mb"] / results["int8"]["size_mb"]
        print(f"Size reduction INT8:        {size_reduction:.2f}x")
    
    print("=" * 70 + "\n")


def save_benchmark_results(results: Dict, output_dir: Path) -> None:
    """Save benchmark results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(output_dir / "latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # CSV
    import csv
    with open(output_dir / "latency_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Format", "Device", "Mean_ms", "P95_ms", "P99_ms", "FPS", "Size_MB"])
        for key, data in results.items():
            if data:
                writer.writerow([
                    data.get("format", key),
                    data.get("device", "N/A"),
                    f"{data.get('mean_ms', 0):.2f}",
                    f"{data.get('p95_ms', 0):.2f}",
                    f"{data.get('p99_ms', 0):.2f}",
                    f"{data.get('fps', 0):.1f}",
                    f"{data.get('size_mb', 0):.2f}",
                ])
    
    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference latency benchmarking")
    parser.add_argument(
        "--weights_dir", 
        type=str,
        default="runs/detect/outputs/training/pcb_defect_yolov8/weights",
        help="Directory containing model weights"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=480,
        help="Image size for inference"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=100,
        help="Number of inference runs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device for PyTorch benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BENCHMARKS_DIR),
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    logger.info(f"Starting latency benchmark on device: {args.device}")
    
    results = run_full_benchmark(
        args.weights_dir,
        args.imgsz,
        args.runs,
        args.device,
    )
    
    print_benchmark_table(results)
    save_benchmark_results(results, Path(args.output))


if __name__ == "__main__":
    main()