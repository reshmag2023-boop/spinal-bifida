"""
Settings and Configuration for Explainable and Uncertainty-Aware Spina Bifida AI System.
"""

import os
from pathlib import Path
import yaml
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATASET_DIR = BASE_DIR / "dataset"
NORMAL_DIR = DATASET_DIR / "Normal"
SB_DIR = DATASET_DIR / "Spina_Bifida"
TEST_IMAGES_DIR = BASE_DIR / "test_images"
ARTIFACTS_DIR = BASE_DIR / "model_artifacts"
REPORTS_DIR = BASE_DIR / "reports_output"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_MODELS = ["resnet18", "custom_cnn", "efficientnet_b0", "vit_b_16"]
DEFAULT_MODEL = "resnet18"

CLASS_NAMES = ["Normal", "Spina_Bifida"]
CLASS_TO_IDX = {"Normal": 0, "Spina_Bifida": 1}

RISK_CONFIG_PATH = CONFIG_DIR / "risk_config.yaml"

def load_risk_config():
    """Load configurable risk weights and thresholds."""
    if not RISK_CONFIG_PATH.exists():
        return {
            "risk_categories": {
                "low": {"min": 0, "max": 30, "label": "LOW", "color": "#10B981"},
                "moderate": {"min": 31, "max": 60, "label": "MODERATE", "color": "#F59E0B"},
                "high": {"min": 61, "max": 100, "label": "HIGH", "color": "#EF4444"}
            },
            "weights": {
                "model_probability": 0.35,
                "cranial_indentation": 0.25,
                "ventricular_dilation": 0.15,
                "deep_abnormality": 0.15,
                "uncertainty_penalty": 0.10
            },
            "uncertainty": {
                "mc_samples": 20,
                "low_threshold": 0.02,
                "high_threshold": 0.06,
                "entropy_threshold": 0.60
            },
            "severity_thresholds": {
                "mild": 0.33,
                "moderate": 0.66,
                "severe": 1.00
            }
        }
    with open(RISK_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_risk_config(config_dict):
    """Save updated risk configuration."""
    with open(RISK_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)

DISCLAIMER_TEXT = (
    "RESEARCH PROTOTYPE ONLY: This system is an investigational artificial intelligence decision-support tool "
    "designed to evaluate fetal ultrasound scans for research and academic purposes. It is NOT a certified "
    "diagnostic medical device and must NEVER be used as the sole basis for clinical diagnosis or patient management. "
    "All computational risk scores, Grad-CAM attention heatmaps, and severity metrics require independent clinical "
    "validation and review by certified maternal-fetal medicine / radiology specialists."
)
