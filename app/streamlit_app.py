# app/streamlit_app.py
"""
PCB Defect Inspection Platform - Streamlit Frontend

A production-quality UI for PCB defect detection with:
- Upload and inspect PCB images
- Visual detection overlay with numbered callouts
- Detection report table with severity indicators
- RAG-based defect explanations
- API status monitoring

Usage:
    streamlit run app/streamlit_app.py --server.port 8501
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import YOLO_CLASS_ID_TO_NAME


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_VERSION = "v1"
API_URL = f"{API_BASE_URL}/api/{API_VERSION}"

# Class mapping for display names
CLASS_DISPLAY_NAMES = {
    "open_circuit": "Open Circuit",
    "short": "Short Circuit",
    "mouse_bite": "Mouse Bite",
    "spur": "Spur",
    "spurious_copper": "Spurious Copper",
    "pin_hole": "Pin Hole",
    "missing_hole": "Missing Hole",
    "copper": "Copper",
}

# Class color mapping
CLASS_COLORS = {
    "open_circuit": "#E63946",
    "short": "#FF6B6B",
    "mouse_bite": "#F4A261",
    "spur": "#E9C46A",
    "spurious_copper": "#2A9D8F",
    "pin_hole": "#264653",
    "missing_hole": "#9B59B6",
    "copper": "#D4A574",
}

# Hardcoded RAG knowledge for fallback when API is unavailable
RAG_KNOWLEDGE = {
    "open_circuit": {
        "description": "A broken or interrupted conductive path on the PCB, preventing current flow.",
        "causes": [
            "Over-etching during manufacturing",
            "Mechanical damage during handling",
            "Thermal stress causing trace fracture"
        ],
        "inspection": [
            "Check for visual breaks in the copper trace",
            "Verify continuity with multimeter testing",
            "Review etching process parameters"
        ]
    },
    "short": {
        "description": "An unintended connection between two or more conductive traces, causing current leakage.",
        "causes": [
            "Solder bridge between adjacent pads",
            "Copper residue from incomplete etching",
            "Foreign conductive material contamination"
        ],
        "inspection": [
            "Visual inspection for solder bridges",
            "Check for copper splashes or debris",
            "Verify with resistance measurement"
        ]
    },
    "mouse_bite": {
        "description": "A small notch or indentation on the edge of a conductive trace, resembling a mouse bite.",
        "causes": [
            "Under-etching of copper",
            "Contamination on the photoresist",
            "Inconsistent exposure during photolithography"
        ],
        "inspection": [
            "Visual inspection under magnification",
            "Check for edge roughness on traces",
            "Verify against design specifications"
        ]
    },
    "spur": {
        "description": "An unwanted protrusion of copper from a trace, potentially causing shorts.",
        "causes": [
            "Over-etching or under-etching",
            "Photoresist defects",
            "Copper plating irregularities"
        ],
        "inspection": [
            "Inspect for copper protrusions",
            "Check for potential short paths",
            "Review etching uniformity"
        ]
    },
    "spurious_copper": {
        "description": "Unwanted copper deposits that should not be present on the PCB surface.",
        "causes": [
            "Copper plating bath contamination",
            "Incomplete resist removal",
            "Electroplating process issues"
        ],
        "inspection": [
            "Visual inspection for copper residues",
            "Check cleaning process effectiveness",
            "Review plating bath chemistry"
        ]
    },
    "pin_hole": {
        "description": "A small hole or void in the copper plating, potentially exposing the underlying substrate.",
        "causes": [
            "Air bubbles in the plating bath",
            "Insufficient plating time",
            "Surface contamination before plating"
        ],
        "inspection": [
            "Inspect for small holes or voids",
            "Check plating bath agitation",
            "Review pre-plating cleaning process"
        ]
    }
}


# ─────────────────────────────────────────────
# CSS Injection
# ─────────────────────────────────────────────

def inject_custom_css():
    """Inject custom CSS for precision instrument aesthetic."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap');
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container */
        .main {
            background-color: #F7F7F5;
            padding: 0 2rem;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #0A0A0A !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        
        /* Monospace data */
        .mono {
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Uploader styling */
        .stFileUploader {
            border: 2px dashed #1D3557 !important;
            border-radius: 0px !important;
            background: white !important;
            padding: 1rem !important;
        }
        .stFileUploader:hover {
            border-color: #457B9D !important;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 0px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            font-size: 14px !important;
            padding: 0.75rem 2rem !important;
            width: 100% !important;
            background-color: #1D3557 !important;
            color: white !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #0A0A0A !important;
            color: white !important;
        }
        
        /* Slider styling */
        .stSlider label {
            font-family: 'Inter', sans-serif !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            color: #457B9D !important;
        }
        
        /* Selectbox */
        .stSelectbox label {
            font-family: 'Inter', sans-serif !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            color: #457B9D !important;
        }
        
        /* Metrics */
        .stMetric label {
            font-family: 'Inter', sans-serif !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            color: #457B9D !important;
        }
        .stMetric .stMetricValue {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 20px !important;
            font-weight: 500 !important;
        }
        
        /* Dataframe */
        .stDataFrame {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            background-color: #1D3557 !important;
            color: white !important;
            border-left: 4px solid #E63946 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 0px !important;
        }
        .streamlit-expanderHeader:hover {
            background-color: #0A0A0A !important;
        }
        .streamlit-expanderContent {
            background-color: #FAFAF8 !important;
            padding: 1rem !important;
            border: 1px solid #E0E0E0 !important;
            border-top: none !important;
        }
        
        /* Divider */
        hr {
            border: none !important;
            border-top: 1px solid #E0E0E0 !important;
            margin: 1.5rem 0 !important;
        }
        
        /* Custom status indicator */
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background-color: #2ECC71;
        }
        .status-offline {
            background-color: #E63946;
        }
        .status-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 500;
        }
        
        /* Header */
        .header-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 28px;
            color: #0A0A0A;
            letter-spacing: -0.02em;
        }
        .header-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: #457B9D;
            font-weight: 400;
        }
        
        /* Empty state */
        .empty-state {
            border: 2px dashed #1D3557;
            border-radius: 0px;
            padding: 4rem 2rem;
            text-align: center;
            background: white;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .empty-state-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            color: #457B9D;
            font-weight: 400;
        }
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 1rem;
        }
        
        /* Section label */
        .section-label {
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #457B9D;
            margin-bottom: 0.5rem;
        }
        
        /* Badge */
        .badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            padding: 0.2rem 0.75rem;
            border-radius: 0px;
            display: inline-block;
        }
        .badge-red {
            background-color: #E63946;
            color: white;
        }
        .badge-blue {
            background-color: #1D3557;
            color: white;
        }
        
        /* Footer */
        .footer {
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            color: #457B9D;
            text-align: center;
            padding: 1.5rem 0 0 0;
            border-top: 1px solid #E0E0E0;
        }
        .footer a {
            color: #1D3557;
            text-decoration: none;
            font-weight: 500;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        
        /* Inference time */
        .inference-time {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #457B9D;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def get_display_class_name(class_name: str) -> str:
    """Get human-readable display name for a class."""
    if not class_name or class_name.startswith("Class "):
        return class_name
    
    if class_name in CLASS_DISPLAY_NAMES.values():
        return class_name
    
    class_key = class_name.lower().replace(" ", "_")
    if class_key in CLASS_DISPLAY_NAMES:
        return CLASS_DISPLAY_NAMES[class_key]
    
    return class_name.replace("_", " ").title()


def get_class_color(class_name: str) -> str:
    """Get color for a defect class."""
    class_key = class_name.lower().replace(" ", "_")
    return CLASS_COLORS.get(class_key, "#457B9D")


def get_rag_explanation(class_name: str) -> Dict:
    """Get RAG explanation for a defect class."""
    class_key = class_name.lower().replace(" ", "_")
    
    # Try API first
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/rag/explain/{class_key}",
            timeout=3,
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    if class_key in RAG_KNOWLEDGE:
        return RAG_KNOWLEDGE[class_key]
    
    # Try to find by display name
    for key, display in CLASS_DISPLAY_NAMES.items():
        if class_name.lower() in display.lower() or display.lower() in class_name.lower():
            if key in RAG_KNOWLEDGE:
                return RAG_KNOWLEDGE[key]
    
    display_name = get_display_class_name(class_name)
    return {
        "description": f"Defect class: {display_name}. This defect type is commonly found in PCB manufacturing.",
        "causes": [
            "Process variation during manufacturing",
            "Material quality issues",
            "Equipment calibration drift"
        ],
        "inspection": [
            "Visual inspection under appropriate magnification",
            "Process parameter review and optimization",
            "Quality check against IPC standards"
        ]
    }


# ─────────────────────────────────────────────
# API Functions
# ─────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_api_health(base_url: str = API_BASE_URL) -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False


@st.cache_data(ttl=60)
def get_api_metadata(base_url: str = API_BASE_URL) -> Dict:
    """Get API metadata."""
    try:
        response = requests.get(f"{base_url}/api/v1/metadata", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def call_predict_api(
    image_bytes: bytes,
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.3,
    base_url: str = API_BASE_URL,
) -> Dict:
    """Call the prediction API."""
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    params = {"conf": conf_threshold, "iou": iou_threshold}
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/predict",
            files=files,
            params=params,
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is it running on port 8000?"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Drawing Functions
# ─────────────────────────────────────────────

def get_bbox_coords(bbox) -> Tuple[int, int, int, int]:
    """Extract coordinates from bbox regardless of format."""
    if isinstance(bbox, dict):
        return int(bbox.get('x1', 0)), int(bbox.get('y1', 0)), int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
    elif isinstance(bbox, (list, tuple)):
        if len(bbox) >= 4:
            return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return 0, 0, 0, 0


def draw_detections(
    image_pil: Image.Image,
    detections: List[Dict],
    class_names: Dict[int, str],
) -> Image.Image:
    """Draw numbered bounding boxes on the image."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font_paths = [
            "/System/Library/Fonts/Monaco.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        font = None
        for fp in font_paths:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, 16)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for idx, det in enumerate(detections, 1):
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = get_bbox_coords(bbox)
        conf = det.get("confidence", 0)
        class_id = det.get("class_id", 0)
        class_name = class_names.get(class_id, f"Class {class_id}")
        display_name = get_display_class_name(class_name)
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        draw.rectangle([x1, y1, x2, y2], outline="#E63946", width=2)
        
        marker_size = 24
        marker_x1 = x1
        marker_y1 = y1 - marker_size
        marker_x2 = x1 + marker_size
        marker_y2 = y1
        if marker_y1 < 0:
            marker_y1 = y1
            marker_y2 = y1 + marker_size
        
        draw.rectangle([marker_x1, marker_y1, marker_x2, marker_y2], fill="#E63946")
        draw.text((x1 + 6, marker_y1 + 3), str(idx), fill="white", font=font)
        
        label = f"{display_name} {conf:.0%}"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        
        label_x = x1
        label_y = y2 + 4
        if label_y + label_height > img.height:
            label_y = y1 - label_height - 4
        
        draw.rectangle(
            [label_x - 2, label_y - 2, label_x + label_width + 2, label_y + label_height + 2],
            fill="white",
            outline="#0A0A0A",
            width=1,
        )
        draw.text((label_x, label_y), label, fill="#0A0A0A", font=font)
    
    return img


# ─────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────

def render_header(api_online: bool):
    """Render the header bar."""
    cols = st.columns([2, 3, 2])
    
    with cols[0]:
        st.markdown(
            """
            <div>
                <span class="header-title">PCB DEFECT INSPECTOR</span>
                <br>
                <span class="header-subtitle">YOLOv8 · DeepPCB · Industrial Inspection</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with cols[2]:
        status_color = "status-online" if api_online else "status-offline"
        status_text = "API ONLINE" if api_online else "API OFFLINE"
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 0.5rem;">
                <span class="status-dot {status_color}"></span>
                <span class="status-text">{status_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<hr>", unsafe_allow_html=True)


def render_controls(
    api_online: bool,
    metadata: Dict,
) -> Tuple[Optional[bytes], float, float, bool]:
    """Render the control panel."""
    st.markdown('<span class="section-label">INSPECTION CONTROLS</span>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload PCB Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Image", width=200)
    
    conf_threshold = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        format="%.2f",
    )
    
    iou_threshold = st.slider(
        "IOU THRESHOLD",
        min_value=0.1,
        max_value=0.7,
        value=0.3,
        step=0.05,
        format="%.2f",
    )
    
    run_button = st.button("RUN INSPECTION", use_container_width=True)
    
    if metadata:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size: 12px; color: #457B9D; font-family: 'JetBrains Mono', monospace;">
                Model: YOLOv8n<br>
                Device: {metadata.get('device', 'N/A')}<br>
                Image Size: {metadata.get('image_size', 'N/A')}px<br>
                Classes: {metadata.get('num_classes', 'N/A')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
    else:
        image_bytes = None
    
    return image_bytes, conf_threshold, iou_threshold, run_button


def render_results(
    image_bytes: Optional[bytes],
    detections: List[Dict],
    inference_time_ms: float,
    class_names: Dict[int, str],
):
    """Render the detection canvas."""
    if image_bytes is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">🔬</div>
                <div class="empty-state-text">UPLOAD A PCB IMAGE TO BEGIN INSPECTION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    
    img = Image.open(io.BytesIO(image_bytes))
    
    if detections:
        annotated_img = draw_detections(img, detections, class_names)
        st.image(annotated_img, use_container_width=True)
        st.markdown(
            f'<div class="inference-time">INFERENCE TIME: {inference_time_ms:.1f}ms</div>',
            unsafe_allow_html=True,
        )
    else:
        st.image(img, use_container_width=True)
        st.info("No defects detected in this image.")


def render_detection_report(detections: List[Dict], class_names: Dict[int, str]):
    """Render the detection report table."""
    if not detections:
        return
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span class="header-title" style="font-size: 18px;">DETECTION REPORT</span>
            <span class="badge badge-red">{len(detections)} DEFECTS FOUND</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    rows = []
    for idx, det in enumerate(detections, 1):
        class_id = det.get("class_id", 0)
        class_name = class_names.get(class_id, f"Class {class_id}")
        display_name = get_display_class_name(class_name)
        conf = det.get("confidence", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = get_bbox_coords(bbox)
        severity = severity_from_confidence(conf)
        
        rows.append({
            "#": idx,
            "Class": display_name,
            "Confidence": f"{conf:.1%}",
            "Location": f"{x1}, {y1} → {x2}, {y2}",
            "Severity": severity,
        })
    
    df = pd.DataFrame(rows)
    
    def color_severity(val):
        colors = {
            "HIGH": "background-color: #FFEBEE; color: #C62828; font-weight: 600;",
            "MEDIUM": "background-color: #FFF3E0; color: #E65100; font-weight: 500;",
            "LOW": "background-color: #FFF8E1; color: #F57F17;",
        }
        return colors.get(val, "")
    
    styled_df = df.style.map(color_severity, subset=["Severity"])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )


def render_rag_panel(detections: List[Dict], class_names: Dict[int, str]):
    """
    Render the RAG explanation panel using native Streamlit components.
    This avoids CSS conflicts with expander shadow DOM.
    """
    if not detections:
        return
    
    # Get unique classes detected
    unique_classes = {}
    for det in detections:
        class_id = det.get("class_id", 0)
        raw_name = class_names.get(class_id, f"Class {class_id}")
        display_name = get_display_class_name(raw_name)
        class_key = raw_name.lower().replace(" ", "_")
        
        if class_id not in unique_classes:
            unique_classes[class_id] = {
                "raw_name": raw_name,
                "display_name": display_name,
                "class_key": class_key,
                "color": CLASS_COLORS.get(class_key, "#457B9D"),
            }
    
    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 0.5rem;">
            <span class="header-title" style="font-size: 18px;">PROCESS ANALYSIS</span>
            <br>
            <span class="header-subtitle">Defect causes and recommended inspection actions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    for idx, (class_id, info) in enumerate(unique_classes.items(), 1):
        display_name = info["display_name"]
        class_key = info["class_key"]
        
        rag_data = get_rag_explanation(class_key)
        
        # ✅ FIX: Use native Streamlit components inside expander
        # This bypasses CSS conflicts with expander shadow DOM
        expander_label = f"[{idx}] {display_name}"
        
        with st.expander(expander_label, expanded=(idx == 1)):
            # Description — use st.write for clean rendering
            st.write("**What this defect looks like:**")
            st.write(rag_data.get('description', 'No description available.'))
            st.write("")  # spacing
            
            # Causes — use native st components
            st.write("**Probable causes:**")
            causes = rag_data.get("causes", ["Information not available"])
            for cause in causes:
                st.write(f"• {cause}")
            st.write("")  # spacing
            
            # Inspection — use native st components
            st.write("**Recommended inspection actions:**")
            inspection = rag_data.get("inspection", ["Information not available"])
            for action in inspection:
                st.write(f"• {action}")


def severity_from_confidence(conf: float) -> str:
    """Determine severity from confidence score."""
    if conf >= 0.85:
        return "HIGH"
    elif conf >= 0.65:
        return "MEDIUM"
    else:
        return "LOW"


def render_footer():
    """Render the footer."""
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="footer">
            PCB Defect Inspector · YOLOv8 · DeepPCB Dataset · Built by Sahariar Hasan
            <br>
            <a href="https://github.com/sahariarhasan/pcb-defect-inspection-platform" target="_blank">GitHub</a> ·
            <a href="https://linkedin.com/in/sahariarhasan" target="_blank">LinkedIn</a> ·
            <a href="/model_card" target="_blank">Model Card</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PCB Defect Inspector",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    inject_custom_css()
    
    # Session state initialization
    if "last_image_bytes" not in st.session_state:
        st.session_state.last_image_bytes = None
    if "last_detections" not in st.session_state:
        st.session_state.last_detections = []
    if "last_inference_time" not in st.session_state:
        st.session_state.last_inference_time = 0
    if "api_online" not in st.session_state:
        st.session_state.api_online = check_api_health()
    if "metadata" not in st.session_state:
        st.session_state.metadata = get_api_metadata()
    
    # ✅ FIX 1: Normalise class_names keys from string to int
    raw_class_names = st.session_state.metadata.get("class_names", {})
    st.session_state.metadata["class_names"] = {
        int(k): v for k, v in raw_class_names.items()
    }
    
    if st.session_state.api_online:
        st.session_state.api_online = check_api_health()
        if not st.session_state.api_online:
            st.warning("⚠️ API connection lost. Please check if the API is running.")
    
    render_header(st.session_state.api_online)
    
    col_left, col_right = st.columns([0.35, 0.65], gap="medium")
    
    with col_left:
        image_bytes, conf_threshold, iou_threshold, run_button = render_controls(
            st.session_state.api_online,
            st.session_state.metadata,
        )
        
        if run_button and image_bytes is not None:
            if not st.session_state.api_online:
                st.error("❌ API is offline. Please start the API server.")
            else:
                with st.spinner("🔬 Running inspection..."):
                    result = call_predict_api(
                        image_bytes,
                        conf_threshold,
                        iou_threshold,
                    )
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.session_state.last_image_bytes = image_bytes
                        st.session_state.last_detections = result.get("detections", [])
                        st.session_state.last_inference_time = result.get("inference_time_ms", 0)
                        st.success(f"✅ Inspection complete! {len(result.get('detections', []))} defects found.")
        
        elif run_button and image_bytes is None:
            st.warning("⚠️ Please upload an image first.")
    
    with col_right:
        render_results(
            st.session_state.last_image_bytes,
            st.session_state.last_detections,
            st.session_state.last_inference_time,
            st.session_state.metadata.get("class_names", {}),
        )
    
    st.markdown("---")
    
    render_detection_report(
        st.session_state.last_detections,
        st.session_state.metadata.get("class_names", {}),
    )
    
    render_rag_panel(
        st.session_state.last_detections,
        st.session_state.metadata.get("class_names", {}),
    )
    
    render_footer()


if __name__ == "__main__":
    main()