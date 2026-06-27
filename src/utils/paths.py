# src/utils/paths.py
"""
Centralized path definitions for the entire project.
All modules import paths from this file to maintain consistency.

No functions, no classes, no mkdir(), no logging.
Only Path constants.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Project Root
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/utils/paths.py -> src -> project_root


# ─────────────────────────────────────────────
# Raw Dataset Paths
# ─────────────────────────────────────────────

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PCB_DATA_DIR = RAW_DATA_DIR / "DeepPCB" / "PCBData"


# ─────────────────────────────────────────────
# Processed Dataset Paths
# ─────────────────────────────────────────────

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONVERTED_DIR = PROCESSED_DIR / "converted"
SPLIT_DIR = PROCESSED_DIR / "split"
AUGMENTED_DIR = PROCESSED_DIR / "augmented"


# ─────────────────────────────────────────────
# Output Paths
# ─────────────────────────────────────────────

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Logs
LOGS_DIR = OUTPUTS_DIR / "logs"

# Reports
REPORTS_DIR = OUTPUTS_DIR / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"
QA_REPORT_PATH = REPORTS_DIR / "qa_report.json"
DATASET_STATS_PATH = REPORTS_DIR / "dataset_stats.json"
EVAL_REPORT_DIR = REPORTS_DIR / "evaluation"

# Weights
WEIGHTS_DIR = OUTPUTS_DIR / "weights"

# Benchmarks
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"

# RAG Index
RAG_INDEX_DIR = OUTPUTS_DIR / "rag_index"

# Flags
FLAGGED_PATH = OUTPUTS_DIR / "flagged_samples.txt"


# ─────────────────────────────────────────────
# Configuration Paths
# ─────────────────────────────────────────────

CONFIG_DIR = PROJECT_ROOT / "configs"  # Note: configs (plural) in your structure
DATASET_YAML_PATH = CONFIG_DIR / "dataset.yaml"


# ─────────────────────────────────────────────
# Models Paths
# ─────────────────────────────────────────────

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"  # Already defined above, but keeping models dir separate


# ─────────────────────────────────────────────
# Knowledge Base Paths
# ─────────────────────────────────────────────

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
REFERENCE_IMAGES_DIR = KNOWLEDGE_BASE_DIR / "reference_images"
DEFECT_KNOWLEDGE_DIR = KNOWLEDGE_BASE_DIR / "defect_knowledge"


# ─────────────────────────────────────────────
# Split Manifest
# ─────────────────────────────────────────────

SPLIT_MANIFEST_PATH = OUTPUTS_DIR / "split_manifest.json"


# ─────────────────────────────────────────────
# App / Web Assets
# ─────────────────────────────────────────────

APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = PROJECT_ROOT / "assets"