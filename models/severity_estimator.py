"""
Second-Stage Spina Bifida Anatomical Severity Assessment Module.
Stratifies cases into Mild, Moderate, or Severe based on measurable
cranial morphological concavity, transventricular dilation proxy,
and deep feature variance.
"""

from typing import Dict, Any, Tuple
import numpy as np

class SeverityEstimator:
    """
    Evaluates morphological severity when Spina Bifida is detected.
    Grounds assessment in measurable image markers.
    """
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "mild": 0.35,
            "moderate": 0.65,
            "severe": 1.00
        }

    def estimate_severity(
        self, 
        is_spina_bifida: bool, 
        probability: float, 
        radiological_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates severity grade and quantitative composite index.
        """
        if not is_spina_bifida or probability < 0.50:
            return {
                "grade": "N/A (Normal Anatomy)",
                "severity_score": 0.0,
                "confidence": round((1.0 - probability) * 100, 1),
                "description": "Fetal cranial and ventricular parameters within normal physiological limits.",
                "color": "#10B981"
            }
            
        cranial_concavity = radiological_features.get("cranial_concavity_score", 0.5)
        ventricular = radiological_features.get("ventricular_dilation_proxy", 0.5)
        posterior = radiological_features.get("posterior_fossa_score", 0.5)
        
        # Weighted morphological severity index
        severity_score = float(
            0.45 * cranial_concavity + 0.35 * ventricular + 0.20 * posterior
        )
        # Weight by classifier confidence
        severity_score = float(np.clip(severity_score * (0.6 + 0.4 * probability), 0.0, 1.0))
        
        if severity_score <= self.thresholds["mild"]:
            grade = "Mild"
            description = "Subtle frontal bone flattening; minimal transventricular hypoechoic enlargement."
            color = "#3B82F6" # Blue
        elif severity_score <= self.thresholds["moderate"]:
            grade = "Moderate"
            description = "Definite bilateral frontal concavity (Lemon Sign); moderate transventricular widening."
            color = "#F59E0B" # Amber
        else:
            grade = "Severe"
            description = "Pronounced cranial scalloping with significant posterior fossa / ventricular disturbance."
            color = "#EF4444" # Red
            
        return {
            "grade": grade,
            "severity_score": round(severity_score, 4),
            "severity_percentage": round(severity_score * 100, 1),
            "description": description,
            "color": color,
            "evidence_metrics": {
                "cranial_indentation": round(cranial_concavity, 3),
                "ventricular_dilation": round(ventricular, 3),
                "posterior_fossa_disruption": round(posterior, 3)
            },
            "status": "Calculated from image morphological indicators (Research Prototype)"
        }
