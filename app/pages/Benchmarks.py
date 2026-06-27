# app/pages/benchmarks.py
"""
Benchmarks page - Display model performance metrics, latency benchmarks,
and comparative analysis between PyTorch and ONNX formats.

Usage:
    This page is accessed from the sidebar navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Benchmarks | PCB Defect Inspector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# CSS Injection
# ─────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        .main {
            background-color: #F7F7F5;
        }
        
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #0A0A0A !important;
        }
        
        .mono {
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-left: 3px solid #1D3557;
            margin-bottom: 1rem;
        }
        
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 600;
            color: #0A0A0A;
        }
        
        .metric-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #457B9D;
        }
        
        .benchmark-table {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }
        
        .section-label {
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #457B9D;
            margin-bottom: 0.5rem;
        }
        
        hr {
            border: none;
            border-top: 1px solid #E0E0E0;
            margin: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_benchmark_results() -> dict:
    """Load benchmark results from JSON file."""
    benchmark_path = Path("outputs/benchmarks/benchmark_results.json")
    if benchmark_path.exists():
        with open(benchmark_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_latency_results() -> dict:
    """Load latency benchmark results."""
    latency_path = Path("outputs/benchmarks/latency_results.json")
    if latency_path.exists():
        with open(latency_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_evaluation_results() -> dict:
    """Load evaluation metrics."""
    eval_path = Path("outputs/eval_report/metrics.json")
    if eval_path.exists():
        with open(eval_path, "r") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────

def plot_latency_comparison(latency_results: dict):
    """Plot latency comparison between model formats."""
    if not latency_results:
        return None
    
    data = []
    formats = []
    mean_latencies = []
    p95_latencies = []
    
    for key, results in latency_results.items():
        if results:
            formats.append(results.get("format", key))
            mean_latencies.append(results.get("mean_ms", 0))
            p95_latencies.append(results.get("p95_ms", 0))
    
    if not formats:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(formats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_latencies, width, label='Mean', color='#1D3557')
    bars2 = ax.bar(x + width/2, p95_latencies, width, label='P95', color='#E63946')
    
    ax.set_xlabel('Model Format', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_fps_comparison(latency_results: dict):
    """Plot FPS comparison between model formats."""
    if not latency_results:
        return None
    
    formats = []
    fps_values = []
    
    for key, results in latency_results.items():
        if results:
            formats.append(results.get("format", key))
            fps_values.append(results.get("fps", 0))
    
    if not formats:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1D3557' if f != 'ONNX INT8' else '#E63946' for f in formats]
    bars = ax.bar(formats, fps_values, color=colors, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Model Format', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('Inference Throughput (FPS)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} FPS', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_size_comparison(latency_results: dict):
    """Plot model size comparison."""
    if not latency_results:
        return None
    
    formats = []
    sizes = []
    
    for key, results in latency_results.items():
        if results and results.get("size_mb", 0) > 0:
            formats.append(results.get("format", key))
            sizes.append(results.get("size_mb", 0))
    
    if not formats:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1D3557' if s == max(sizes) else '#457B9D' for s in sizes]
    bars = ax.bar(formats, sizes, color=colors, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Model Format', fontsize=12)
    ax.set_ylabel('Model Size (MB)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} MB', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_speedup_comparison(latency_results: dict):
    """Plot speedup comparison relative to PyTorch."""
    if not latency_results:
        return None
    
    # Find PyTorch baseline
    pytorch_mean = None
    for key, results in latency_results.items():
        if results and results.get("format") == "PyTorch FP32":
            pytorch_mean = results.get("mean_ms", 0)
            break
    
    if pytorch_mean is None or pytorch_mean == 0:
        return None
    
    formats = []
    speedups = []
    colors = []
    
    for key, results in latency_results.items():
        if results and results.get("mean_ms", 0) > 0:
            mean_ms = results.get("mean_ms", 0)
            speedup = pytorch_mean / mean_ms
            formats.append(results.get("format", key))
            speedups.append(speedup)
            colors.append('#2ECC71' if speedup >= 1.0 else '#E63946')
    
    if not formats:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(formats, speedups, color=colors, edgecolor='white', linewidth=1)
    
    ax.axhline(y=1.0, color='#0A0A0A', linestyle='--', linewidth=1, label='PyTorch Baseline')
    ax.set_xlabel('Model Format', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Speedup vs PyTorch FP32', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_per_class_performance(eval_results: dict):
    """Plot per-class performance metrics."""
    if not eval_results or "per_class" not in eval_results:
        return None
    
    per_class = eval_results["per_class"]
    classes = list(per_class.keys())
    
    if not classes:
        return None
    
    ap_values = [per_class[c].get("ap", 0) for c in classes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.array(ap_values))[::-1]
    bars = ax.barh(classes, ap_values, color=colors, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Per-Class Performance (AP@50-95)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.3f}', xy=(width + 0.01, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────

def main():
    """Main benchmarks page."""
    inject_css()
    
    st.title("📊 Performance Benchmarks")
    st.markdown(
        """
        <span style="color: #457B9D; font-family: 'Inter', sans-serif; font-size: 14px;">
            Comprehensive benchmarking of model performance across different formats and hardware.
        </span>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Load data
    latency_results = load_latency_results()
    eval_results = load_evaluation_results()
    
    # ─────────────────────────────────────────────
    # Key Metrics Row
    # ─────────────────────────────────────────────
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if eval_results:
            map50 = eval_results.get("map50", 0) * 100
            st.metric("mAP@50", f"{map50:.1f}%")
        else:
            st.metric("mAP@50", "N/A")
    
    with col2:
        if latency_results:
            # Get PyTorch FPS
            fps = 0
            for key, results in latency_results.items():
                if results and results.get("format") == "PyTorch FP32":
                    fps = results.get("fps", 0)
                    break
            if fps > 0:
                st.metric("PyTorch FPS", f"{fps:.1f}")
            else:
                st.metric("PyTorch FPS", "N/A")
        else:
            st.metric("PyTorch FPS", "N/A")
    
    with col3:
        if latency_results:
            # Get ONNX FPS
            fps = 0
            for key, results in latency_results.items():
                if results and results.get("format") == "ONNX FP32":
                    fps = results.get("fps", 0)
                    break
            if fps > 0:
                st.metric("ONNX FPS", f"{fps:.1f}")
            else:
                st.metric("ONNX FPS", "N/A")
        else:
            st.metric("ONNX FPS", "N/A")
    
    with col4:
        if eval_results:
            f1 = eval_results.get("f1", 0) * 100
            st.metric("F1 Score", f"{f1:.1f}%")
        else:
            st.metric("F1 Score", "N/A")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Latency & Throughput Charts
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">LATENCY & THROUGHPUT</span>', unsafe_allow_html=True)
    
    if latency_results:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_latency_comparison(latency_results)
            if fig:
                st.pyplot(fig)
            else:
                st.info("No latency data available")
        
        with col2:
            fig = plot_fps_comparison(latency_results)
            if fig:
                st.pyplot(fig)
            else:
                st.info("No FPS data available")
    else:
        st.warning("Latency benchmark results not found. Run `python -m src.evaluation.latency` first.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Model Size & Speedup
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">MODEL SIZE & OPTIMIZATION</span>', unsafe_allow_html=True)
    
    if latency_results:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_size_comparison(latency_results)
            if fig:
                st.pyplot(fig)
            else:
                st.info("No model size data available")
        
        with col2:
            fig = plot_speedup_comparison(latency_results)
            if fig:
                st.pyplot(fig)
            else:
                st.info("No speedup data available")
    else:
        st.warning("Model size data not available.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Per-Class Performance
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">PER-CLASS PERFORMANCE</span>', unsafe_allow_html=True)
    
    if eval_results:
        fig = plot_per_class_performance(eval_results)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No per-class performance data available")
    else:
        st.warning("Evaluation results not found. Run `python -m src.evaluation.metrics` first.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Benchmark Data Table
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">BENCHMARK SUMMARY</span>', unsafe_allow_html=True)
    
    if latency_results:
        # Build table
        table_data = []
        for key, results in latency_results.items():
            if results:
                table_data.append({
                    "Format": results.get("format", key),
                    "Device": results.get("device", "N/A"),
                    "Mean (ms)": f"{results.get('mean_ms', 0):.2f}",
                    "P95 (ms)": f"{results.get('p95_ms', 0):.2f}",
                    "P99 (ms)": f"{results.get('p99_ms', 0):.2f}",
                    "FPS": f"{results.get('fps', 0):.1f}",
                    "Size (MB)": f"{results.get('size_mb', 0):.1f}",
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No benchmark data available")
    else:
        st.warning("No latency benchmark results found.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Hardware Information
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">HARDWARE & CONFIGURATION</span>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Device</div>
                <div class="metric-value" style="font-size: 1.2rem;">Apple M4</div>
                <div style="font-size: 0.85rem; color: #457B9D;">MPS Acceleration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Memory</div>
                <div class="metric-value" style="font-size: 1.2rem;">16 GB</div>
                <div style="font-size: 0.85rem; color: #457B9D;">Unified Memory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value" style="font-size: 1.2rem;">YOLOv8n</div>
                <div style="font-size: 0.85rem; color: #457B9D;">3.0M Parameters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()