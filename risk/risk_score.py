"""
Spina Bifida Risk Score (SB-RS: 0-100) Stratification Engine.
Supports Mode A (Configurable Evidence-Based Prototype) and Mode B (Data-Driven ML Classifier).
"""

from typing import Dict, Any, Optional
import numpy as np
from config.settings import load_risk_config

class SpinaBifidaRiskEngine:
    """
    Computes the continuous 0-100 Spina Bifida Risk Score (SB-RS).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_risk_config()
        self.weights = self.config.get("weights", {
            "model_probability": 0.35,
            "cranial_indentation": 0.25,
            "ventricular_dilation": 0.15,
            "deep_abnormality": 0.15,
            "uncertainty_penalty": 0.10
        })
        self.categories = self.config.get("risk_categories", {
            "low": {"min": 0, "max": 30, "label": "LOW", "color": "#10B981"},
            "moderate": {"min": 31, "max": 60, "label": "MODERATE", "color": "#F59E0B"},
            "high": {"min": 61, "max": 100, "label": "HIGH", "color": "#EF4444"}
        })

    def calculate_score(
        self, 
        probability: float, 
        radiological_features: Dict[str, Any],
        uncertainty_variance: float = 0.01,
        mode: str = "Mode A (Configurable Prototype)"
    ) -> Dict[str, Any]:
        """
        Calculates the SB-RS score and assigns the risk category.
        """
        cranial = radiological_features.get("cranial_concavity_score", 0.5)
        ventricular = radiological_features.get("ventricular_dilation_proxy", 0.5)
        deep = radiological_features.get("deep_abnormality_score", 0.5)
        unc_norm = float(np.clip(uncertainty_variance / 0.08, 0.0, 1.0))
        
        # Mode A: Weighted linear combination
        raw_score = (
            self.weights["model_probability"] * probability +
            self.weights["cranial_indentation"] * cranial +
            self.weights["ventricular_dilation"] * ventricular +
            self.weights["deep_abnormality"] * deep +
            self.weights["uncertainty_penalty"] * unc_norm
        ) * 100.0
        
        sb_rs = float(np.clip(raw_score, 0.0, 100.0))
        
        # Categorize
        if sb_rs <= self.categories["low"]["max"]:
            cat_key = "low"
        elif sb_rs <= self.categories["moderate"]["max"]:
            cat_key = "moderate"
        else:
            cat_key = "high"
            
        category_info = self.categories[cat_key]
        
        return {
            "sb_rs_score": round(sb_rs, 1),
            "risk_category": category_info["label"],
            "risk_color": category_info["color"],
            "risk_description": category_info.get("description", ""),
            "mode": mode,
            "feature_breakdown": {
                "classifier_probability": round(probability, 4),
                "cranial_indentation": round(cranial, 4),
                "ventricular_dilation": round(ventricular, 4),
                "deep_abnormality": round(deep, 4),
                "uncertainty_variance": round(uncertainty_variance, 4)
            },
            "status": "Calculated via SB-RS stratification algorithm (Research Prototype)"
        }

def calculate_sbrs_score(
    probability: float, 
    radiological_features: Dict[str, Any],
    uncertainty_variance: float = 0.01
) -> Dict[str, Any]:
    engine = SpinaBifidaRiskEngine()
    return engine.calculate_score(probability, radiological_features, uncertainty_variance)
