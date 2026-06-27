# app/pages/knowledge_assistant.py
"""
Knowledge Assistant page - RAG-based defect analysis and manufacturing guidance.

Provides:
- Defect class information
- Manufacturing process insights
- Recommended inspection actions
- Cross-reference with industry standards

Usage:
    This page is accessed from the sidebar navigation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import random

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Knowledge Assistant | PCB Defect Inspector",
    page_icon="🧠",
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
        
        .knowledge-card {
            background: white;
            padding: 1.5rem;
            border-left: 3px solid #1D3557;
            margin-bottom: 1rem;
        }
        
        .knowledge-card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #0A0A0A;
        }
        
        .knowledge-card-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
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
        
        .defect-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background-color: #E63946;
            color: white;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 0px;
        }
        
        .cause-item {
            padding: 0.5rem 0.75rem;
            background-color: #F0F0EE;
            margin-bottom: 0.5rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
        }
        
        .inspection-item {
            padding: 0.5rem 0.75rem;
            border-left: 2px solid #2ECC71;
            margin-bottom: 0.5rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "open_circuit": {
        "name": "Open Circuit",
        "description": "A broken or interrupted conductive path on the PCB, preventing current flow. This is one of the most critical defects in PCB manufacturing.",
        "severity": "Critical",
        "causes": [
            "Over-etching during manufacturing process",
            "Mechanical damage during handling or assembly",
            "Thermal stress causing trace fracture",
            "Insufficient copper thickness",
            "Contamination on the copper surface"
        ],
        "inspection": [
            "Visual inspection under magnification for breaks in copper traces",
            "Verify continuity with multimeter or flying probe testing",
            "Review etching process parameters and bath chemistry",
            "Check handling procedures for mechanical damage",
            "Perform thermal cycling test for stress-related fractures"
        ],
        "prevention": [
            "Control etching parameters within specified limits",
            "Implement proper handling procedures",
            "Use adequate copper thickness for design requirements",
            "Ensure clean substrate before copper deposition"
        ],
        "industry_standards": "IPC-A-600, IPC-6012, MIL-PRF-31032",
        "image_icon": "🔌"
    },
    "short": {
        "name": "Short Circuit",
        "description": "An unintended connection between two or more conductive traces, causing current leakage or electrical malfunction.",
        "severity": "Critical",
        "causes": [
            "Solder bridge between adjacent pads during assembly",
            "Copper residue from incomplete etching",
            "Foreign conductive material contamination",
            "Photoresist defects",
            "Copper plating irregularities"
        ],
        "inspection": [
            "Visual inspection for solder bridges and copper splashes",
            "Check for conductive debris on the PCB surface",
            "Verify with resistance measurement between adjacent traces",
            "Review soldering process parameters",
            "Inspect for copper residue under microscope"
        ],
        "prevention": [
            "Optimize solder paste stencil design",
            "Control reflow oven temperature profiles",
            "Ensure complete etching of copper traces",
            "Implement cleaning process for solder flux removal"
        ],
        "industry_standards": "IPC-A-600, IPC-6012, J-STD-001",
        "image_icon": "⚡"
    },
    "mouse_bite": {
        "name": "Mouse Bite",
        "description": "A small notch or indentation on the edge of a conductive trace, resembling a mouse bite. Can weaken the trace and cause reliability issues.",
        "severity": "Medium",
        "causes": [
            "Under-etching of copper traces",
            "Contamination on the photoresist",
            "Inconsistent exposure during photolithography",
            "Variations in developer concentration",
            "Substrate surface irregularities"
        ],
        "inspection": [
            "Visual inspection under magnification for edge roughness",
            "Check for missing copper on trace edges",
            "Verify against design specifications",
            "Inspect photoresist application quality",
            "Review etching uniformity across panel"
        ],
        "prevention": [
            "Optimize etching time and temperature",
            "Ensure proper photoresist adhesion",
            "Maintain consistent developer chemistry",
            "Regular calibration of exposure equipment"
        ],
        "industry_standards": "IPC-A-600, IPC-6012",
        "image_icon": "🐭"
    },
    "spur": {
        "name": "Spur",
        "description": "An unwanted protrusion of copper from a trace, potentially creating short paths or reducing trace spacing.",
        "severity": "Medium-High",
        "causes": [
            "Over-etching or under-etching of copper",
            "Photoresist defects or delamination",
            "Copper plating irregularities",
            "Exposure equipment misalignment",
            "Developer temperature variations"
        ],
        "inspection": [
            "Inspect for copper protrusions under magnification",
            "Check for potential short paths or reduced spacing",
            "Review etching process uniformity",
            "Inspect photoresist quality",
            "Measure trace spacing at affected locations"
        ],
        "prevention": [
            "Control etching process parameters",
            "Ensure photoresist quality and adhesion",
            "Regular maintenance of plating bath",
            "Calibrate exposure and development equipment"
        ],
        "industry_standards": "IPC-A-600, IPC-6012",
        "image_icon": "🔱"
    },
    "spurious_copper": {
        "name": "Spurious Copper",
        "description": "Unwanted copper deposits that should not be present on the PCB surface, potentially causing shorts or signal integrity issues.",
        "severity": "High",
        "causes": [
            "Copper plating bath contamination",
            "Incomplete resist removal during stripping",
            "Electroplating process issues",
            "Throwing power variation in plating bath",
            "Surface contamination before plating"
        ],
        "inspection": [
            "Visual inspection for copper residues",
            "Check cleaning process effectiveness",
            "Review plating bath chemistry and filtration",
            "Inspect for copper nodules or deposits",
            "Verify resist stripping process"
        ],
        "prevention": [
            "Maintain plating bath cleanliness",
            "Optimize electroplating parameters",
            "Ensure complete resist removal",
            "Implement effective pre-plating cleaning"
        ],
        "industry_standards": "IPC-A-600, IPC-6012",
        "image_icon": "🧪"
    },
    "pin_hole": {
        "name": "Pin Hole",
        "description": "A small hole or void in the copper plating, potentially exposing the underlying substrate to corrosion or contamination.",
        "severity": "Medium",
        "causes": [
            "Air bubbles in the plating bath",
            "Insufficient plating time or current",
            "Surface contamination before plating",
            "Low wetting agent concentration",
            "Agitation issues in the plating tank"
        ],
        "inspection": [
            "Inspect for small holes or voids in copper surface",
            "Check plating bath agitation system",
            "Review pre-plating cleaning process",
            "Verify plating time and current density",
            "Test for adhesion of plated copper"
        ],
        "prevention": [
            "Optimize plating bath agitation",
            "Maintain proper wetting agent concentration",
            "Ensure thorough cleaning before plating",
            "Control plating bath temperature",
            "Regular filtration of plating solution"
        ],
        "industry_standards": "IPC-A-600, IPC-6012, IPC-4552",
        "image_icon": "🕳️"
    }
}


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def get_defect_info(defect_name: str) -> dict:
    """Get defect information from knowledge base."""
    return KNOWLEDGE_BASE.get(defect_name, None)

def get_all_defect_names() -> list:
    """Get all defect class names."""
    return sorted(KNOWLEDGE_BASE.keys())


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────

def main():
    """Main knowledge assistant page."""
    inject_css()
    
    st.title("🧠 Knowledge Assistant")
    st.markdown(
        """
        <span style="color: #457B9D; font-family: 'Inter', sans-serif; font-size: 14px;">
            RAG-powered manufacturing knowledge for PCB defect analysis and process improvement.
        </span>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Search / Selection
    # ─────────────────────────────────────────────
    
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.markdown('<span class="section-label">DEFECT CLASS</span>', unsafe_allow_html=True)
        
        # Get all defect names with display names
        defect_options = {f"{v['image_icon']} {v['name']}": k for k, v in KNOWLEDGE_BASE.items()}
        
        selected_display = st.selectbox(
            "Select a defect type",
            options=list(defect_options.keys()),
            index=0,
            label_visibility="collapsed",
        )
        
        selected_defect = defect_options[selected_display]
        
        # Show quick stats
        if selected_defect in KNOWLEDGE_BASE:
            info = KNOWLEDGE_BASE[selected_defect]
            severity = info.get("severity", "Unknown")
            severity_color = {
                "Critical": "#E63946",
                "High": "#E67E22",
                "Medium-High": "#F1C40F",
                "Medium": "#2ECC71",
                "Low": "#3498DB",
            }.get(severity, "#457B9D")
            
            st.markdown(
                f"""
                <div style="background: white; padding: 1rem; margin-top: 1rem;">
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #457B9D; text-transform: uppercase; letter-spacing: 0.5px;">
                        Severity
                    </div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; color: {severity_color};">
                        {severity}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #457B9D; margin-top: 0.5rem;">
                        {info.get('industry_standards', 'N/A')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    with col2:
        st.markdown('<span class="section-label">DEFECT OVERVIEW</span>', unsafe_allow_html=True)
        
        if selected_defect in KNOWLEDGE_BASE:
            info = KNOWLEDGE_BASE[selected_defect]
            
            st.markdown(
                f"""
                <div class="knowledge-card">
                    <div class="knowledge-card-title">{info['image_icon']} {info['name']}</div>
                    <div class="knowledge-card-subtitle">{info['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Detailed Information
    # ─────────────────────────────────────────────
    
    if selected_defect in KNOWLEDGE_BASE:
        info = KNOWLEDGE_BASE[selected_defect]
        
        # Causes
        st.markdown('<span class="section-label">PROBABLE CAUSES</span>', unsafe_allow_html=True)
        
        for cause in info.get("causes", []):
            st.markdown(
                f"""
                <div class="cause-item">
                    <span style="font-weight: 500;">•</span> {cause}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Inspection Actions
        st.markdown('<span class="section-label">RECOMMENDED INSPECTION ACTIONS</span>', unsafe_allow_html=True)
        
        for inspection in info.get("inspection", []):
            st.markdown(
                f"""
                <div class="inspection-item">
                    <span style="font-weight: 500;">✓</span> {inspection}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Prevention
        st.markdown('<span class="section-label">PREVENTION STRATEGIES</span>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        prevention = info.get("prevention", [])
        
        for idx, strategy in enumerate(prevention):
            col = [col1, col2, col3][idx % 3]
            with col:
                st.markdown(
                    f"""
                    <div style="background: white; padding: 0.75rem; margin-bottom: 0.5rem; border-left: 3px solid #1D3557; min-height: 80px;">
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #0A0A0A;">
                            {strategy}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # All Defects Overview
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">ALL DEFECT CLASSES</span>', unsafe_allow_html=True)
    
    # Create a grid of all defects
    cols = st.columns(3)
    
    for idx, (key, info) in enumerate(KNOWLEDGE_BASE.items()):
        col = cols[idx % 3]
        with col:
            severity = info.get("severity", "Unknown")
            severity_color = {
                "Critical": "#E63946",
                "High": "#E67E22",
                "Medium-High": "#F1C40F",
                "Medium": "#2ECC71",
                "Low": "#3498DB",
            }.get(severity, "#457B9D")
            
            st.markdown(
                f"""
                <div style="background: white; padding: 0.75rem; margin-bottom: 0.5rem; border-left: 3px solid {severity_color};">
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 500; color: #0A0A0A;">
                        {info['image_icon']} {info['name']}
                    </div>
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #457B9D;">
                        {severity}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # Search / Filter
    # ─────────────────────────────────────────────
    
    st.markdown('<span class="section-label">SEARCH KNOWLEDGE BASE</span>', unsafe_allow_html=True)
    
    search_query = st.text_input("Search for specific defect information...", placeholder="e.g., thermal stress, photoresist, copper plating...")
    
    if search_query:
        st.markdown(
            f"""
            <div style="background: white; padding: 1rem; margin-top: 0.5rem;">
                <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #0A0A0A;">
                    <span style="font-weight: 500;">Search results for:</span> "{search_query}"
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #457B9D; margin-top: 0.5rem;">
                    🔍 Found in: 
                    {', '.join([f"{info['name']}" for key, info in KNOWLEDGE_BASE.items() if any(search_query.lower() in str(value).lower() for value in info.values())])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()