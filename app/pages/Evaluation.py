# app/pages/evaluation.py
"""
Evaluation page - Display model evaluation metrics, confusion matrix,
precision-recall curves, and failure analysis.

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
from PIL import Image

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Evaluation | PCB Defect Inspector",
    page_icon="📈",
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
def load_metrics() -> dict:
    """Load evaluation metrics."""
    path = Path("outputs/eval_report/metrics.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_failure_analysis() -> dict:
    """Load failure analysis results."""
    path = Path("outputs/eval_report/failure_summary.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────

def plot_confusion_matrix():
    """Display confusion matrix images if available."""
    cm_paths = [
        Path("outputs/eval_report/confusion_matrix_normalised.png"),
        Path("outputs/eval_report/confusion_matrix_raw.png"),
    ]
    
    for path in cm_paths:
        if path.exists():
            return Image.open(path)
    return None

def plot_pr_curves():
    """Display PR curve images if available."""
    pr_paths = [
        Path("outputs/eval_report/pr_curves_combined.png"),
        Path("outputs/eval_report/pr_curves_per_class.png"),
    ]
    
    for path in pr_paths:
        if path.exists():
            return Image.open(path)
    return None


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────

def main():
    """Main evaluation page."""
    inject_css()
    
    st.title("📈 Model Evaluation")
    st.markdown(
        """
        <span style="color: #457B9D; font-family: 'Inter', sans-serif; font-size: 14px;">
            Comprehensive model evaluation metrics, confusion matrix, and failure analysis.
        </span>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Load data
    metrics = load_metrics()
    
    # ─────────────────────────────────────────────
    # Key Metrics Row
    # ─────────────────────────────────────────────
    
    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            map50 = metrics.get("map50", 0) * 100
            st.metric("mAP@50", f"{map50:.1f}%")
        
        with col2:
            map = metrics.get("map", 0) * 100
            st.metric("mAP@50-95", f"{map:.1f}%")
        
        with col3:
            precision = metrics.get("precision", 0) * 100
            st.metric("Precision", f"{precision:.1f}%")
        
        with col4:
            recall = metrics.get("recall", 0) * 100
            st.metric("Recall", f"{recall:.1f}%")
        
        with col5:
            f1 = metrics.get("f1", 0) * 100
            st.metric("F1 Score", f"{f1:.1f}%")
    else:
        st.warning("No evaluation metrics found. Run `python -m src.evaluation.metrics` first.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Per-Class Metrics Table
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">PER-CLASS PERFORMANCE</span>', unsafe_allow_html=True)
    
    if metrics and "per_class" in metrics:
        per_class = metrics["per_class"]
        
        table_data = []
        for class_name, metrics_dict in per_class.items():
            table_data.append({
                "Class": class_name,
                "AP@50-95": f"{metrics_dict.get('ap', 0):.4f}",
                "AP@50": f"{metrics_dict.get('ap50', 0):.4f}",
                "Precision": f"{metrics_dict.get('precision', 0):.4f}",
                "Recall": f"{metrics_dict.get('recall', 0):.4f}",
                "F1": f"{metrics_dict.get('f1', 0):.4f}",
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Confusion Matrix
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">CONFUSION MATRIX</span>', unsafe_allow_html=True)
    
    cm_img = plot_confusion_matrix()
    if cm_img:
        st.image(cm_img, use_container_width=True)
    else:
        st.info("Confusion matrix images not found. Run `python -m src.evaluation.plots` first.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Precision-Recall Curves
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">PRECISION-RECALL CURVES</span>', unsafe_allow_html=True)
    
    pr_img = plot_pr_curves()
    if pr_img:
        st.image(pr_img, use_container_width=True)
    else:
        st.info("PR curve images not found. Run `python -m src.evaluation.precision_recall` first.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Failure Analysis
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">FAILURE ANALYSIS</span>', unsafe_allow_html=True)
    
    failure_data = load_failure_analysis()
    
    if failure_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_images = failure_data.get("total_images", 0)
            st.metric("Total Images", total_images)
        
        with col2:
            total_gt = failure_data.get("total_gt", 0)
            st.metric("Ground Truth Boxes", total_gt)
        
        with col3:
            total_fp = failure_data.get("total_fp", 0)
            st.metric("False Positives", total_fp)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_fn = failure_data.get("total_fn", 0)
            st.metric("False Negatives", total_fn)
        
        with col2:
            total_tp = failure_data.get("total_tp", 0)
            st.metric("True Positives", total_tp)
        
        with col3:
            if total_gt > 0:
                miss_rate = total_fn / total_gt * 100
                st.metric("Miss Rate", f"{miss_rate:.1f}%")
            else:
                st.metric("Miss Rate", "N/A")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Failure Visualization
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">TOP FAILURE CASES</span>', unsafe_allow_html=True)
    
    failure_images = [
        Path("outputs/eval_report/top10_missed.png"),
        Path("outputs/eval_report/top10_false_positives.png"),
    ]
    
    for img_path in failure_images:
        if img_path.exists():
            st.image(Image.open(img_path), use_container_width=True)
            st.caption(f"File: {img_path.name}")
    
    if not any(p.exists() for p in failure_images):
        st.info("Failure visualization images not found. Run `python -m src.evaluation.failure_analysis` first.")


if __name__ == "__main__":
    main()