"""
Configuration module for SIF Precursor Screening System.
Single source of truth for paths, thresholds, and hyper-parameters.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SRC_DIR = BASE_DIR / "src"

# File Paths
RAW_DATA_PATH = DATA_DIR / "industrial_safety.csv"
VECTORIZER_PATH = MODELS_DIR / "sif_vectorizer.pkl"
LR_MODEL_PATH = MODELS_DIR / "sif_lr_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "sif_rf_model.pkl"

METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.csv"
FLAGGED_TOP10_PATH = REPORTS_DIR / "flagged_top10.csv"
CV_CURVES_PATH = REPORTS_DIR / "cv_threshold_curve.csv"

# Classification & Evaluation Settings
RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 3
DEFAULT_OPERATING_THRESHOLD = 0.38  # High-recall target calibrated via CV
HIGH_RISK_LEVELS = ["IV", "V", "VI"]

# Ensure target directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
