# src/evaluation/report.py
"""
Generates evaluation report from all saved results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from src.evaluation.utils import EVAL_REPORT_DIR, BENCHMARKS_DIR, load_json
from src.utils.logger import get_logger
from src.utils.config import YOLO_CLASS_ID_TO_NAME

logger = get_logger(__name__)


def load_all_results() -> Dict:
    """Load all evaluation results."""
    results = {}
    
    # Metrics
    metrics_path = EVAL_REPORT_DIR / "metrics.json"
    if metrics_path.exists():
        results["metrics"] = load_json(metrics_path)
    
    # Failure summary
    failure_path = EVAL_REPORT_DIR / "failure_summary.json"
    if failure_path.exists():
        results["failure"] = load_json(failure_path)
    
    # Optimal thresholds
    thresholds_path = EVAL_REPORT_DIR / "optimal_thresholds.json"
    if thresholds_path.exists():
        results["thresholds"] = load_json(thresholds_path)
    
    # Map by box size
    map_size_path = EVAL_REPORT_DIR / "map_by_box_size.json"
    if map_size_path.exists():
        results["map_by_size"] = load_json(map_size_path)
    
    # Latency
    latency_path = BENCHMARKS_DIR / "latency_results.json"
    if latency_path.exists():
        results["latency"] = load_json(latency_path)
    
    return results


def generate_markdown_report(results: Dict, output_path: Path) -> None:
    """Generate evaluation report in Markdown."""
    lines = []
    
    # Header
    lines.append("# PCB Defect Detection — Evaluation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    if "metrics" in results:
        m = results["metrics"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| mAP@50 | {m.get('map50', 0):.4f} |")
        lines.append(f"| mAP@50-95 | {m.get('map', 0):.4f} |")
        lines.append(f"| Precision | {m.get('precision', 0):.4f} |")
        lines.append(f"| Recall | {m.get('recall', 0):.4f} |")
        lines.append(f"| F1 | {m.get('f1', 0):.4f} |")
    lines.append("")
    
    # Per-class metrics
    lines.append("## Per-Class Performance")
    lines.append("")
    if "metrics" in results and "per_class" in results["metrics"]:
        lines.append("| Class | AP@50 | AP@50-95 | Precision | Recall | F1 |")
        lines.append("|-------|-------|----------|-----------|--------|----|")
        for name, stats in results["metrics"]["per_class"].items():
            lines.append(
                f"| {name} | {stats.get('ap50', 0):.4f} | "
                f"{stats.get('ap', 0):.4f} | "
                f"{stats.get('precision', 0):.4f} | "
                f"{stats.get('recall', 0):.4f} | "
                f"{stats.get('f1', 0):.4f} |"
            )
    lines.append("")
    
    # Optimal thresholds
    if "thresholds" in results:
        lines.append("## Optimal Confidence Thresholds")
        lines.append("")
        lines.append("```yaml")
        for name, threshold in results["thresholds"].items():
            lines.append(f"  {name}: {threshold:.4f}")
        lines.append("```")
        lines.append("")
    
    # Latency benchmarks
    if "latency" in results:
        lines.append("## Inference Latency")
        lines.append("")
        lines.append("| Format | Mean (ms) | P95 (ms) | P99 (ms) | FPS | Size (MB) |")
        lines.append("|--------|-----------|----------|----------|-----|-----------|")
        for key, data in results["latency"].items():
            if data:
                lines.append(
                    f"| {data.get('format', key)} | "
                    f"{data.get('mean_ms', 0):.2f} | "
                    f"{data.get('p95_ms', 0):.2f} | "
                    f"{data.get('p99_ms', 0):.2f} | "
                    f"{data.get('fps', 0):.1f} | "
                    f"{data.get('size_mb', 0):.1f} |"
                )
        lines.append("")
    
    # Known limitations
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("1. **Small defect detection**: The model struggles with very small defects (< 10px)")
    lines.append("2. **Edge cases**: Defects near image borders can be missed due to limited context")
    lines.append("3. **Lighting variations**: Performance degrades under extreme lighting conditions")
    lines.append("4. **Overlapping defects**: The model occasionally misses defects when they overlap")
    lines.append("")
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report saved to {output_path}")


def main() -> None:
    results = load_all_results()
    generate_markdown_report(results, EVAL_REPORT_DIR / "evaluation_report.md")
    
    print(f"\n✅ Evaluation report generated: {EVAL_REPORT_DIR / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()