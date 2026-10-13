"""
Risk Factor Attribution and Narrative Explainer Module.
Answers "WHY did the system assign this risk?" by breaking down
model probability, morphological concavity, transventricular proxies,
and uncertainty metrics into quantifiable contributing factors.
"""

from typing import Dict, Any, List
import numpy as np

class RiskFactorExplainer:
    """
    Decomposes the SB-RS risk score into factor contributions
    and generates natural language clinical reasoning summaries.
    """
    def __init__(self, risk_weights: Dict[str, float] = None):
        self.weights = risk_weights or {
            "model_probability": 0.35,
            "cranial_indentation": 0.25,
            "ventricular_dilation": 0.15,
            "deep_abnormality": 0.15,
            "uncertainty_penalty": 0.10
        }

    def compute_contributions(
        self, 
        probability: float, 
        radiological_features: Dict[str, Any],
        uncertainty_variance: float
    ) -> Dict[str, Any]:
        """
        Computes weighted point contributions to the 0-100 SB-RS score.
        """
        cranial = radiological_features.get("cranial_concavity_score", 0.5)
        ventricular = radiological_features.get("ventricular_dilation_proxy", 0.5)
        deep = radiological_features.get("deep_abnormality_score", 0.5)
        unc_norm = float(np.clip(uncertainty_variance / 0.08, 0.0, 1.0))
        
        # Raw component points out of 100
        p_contrib = probability * self.weights["model_probability"] * 100.0
        c_contrib = cranial * self.weights["cranial_indentation"] * 100.0
        v_contrib = ventricular * self.weights["ventricular_dilation"] * 100.0
        d_contrib = deep * self.weights["deep_abnormality"] * 100.0
        u_contrib = unc_norm * self.weights["uncertainty_penalty"] * 100.0
        
        total_points = p_contrib + c_contrib + v_contrib + d_contrib + u_contrib
        
        # Factor list for waterfall/bar chart
        factors = [
            {"factor": "Deep Classifier Signal", "contribution": round(p_contrib, 1), "value": f"{probability*100:.1f}%", "weight": f"{self.weights['model_probability']*100:.0f}%", "description": "ResNet/CNN softmax probability for Spina Bifida features"},
            {"factor": "Cranial Lemon Sign (Concavity)", "contribution": round(c_contrib, 1), "value": f"{cranial:.3f}", "weight": f"{self.weights['cranial_indentation']*100:.0f}%", "description": "Frontal bone scalloping & inward curvature index"},
            {"factor": "Transventricular Dilation Proxy", "contribution": round(v_contrib, 1), "value": f"{ventricular:.3f}", "weight": f"{self.weights['ventricular_dilation']*100:.0f}%", "description": "Hypoechoic lateral ventricle dilation ratio"},
            {"factor": "Deep Feature Embedding Abnormality", "contribution": round(d_contrib, 1), "value": f"{deep:.3f}", "weight": f"{self.weights['deep_abnormality']*100:.0f}%", "description": "Latent vector distance from normal baseline"},
            {"factor": "Epistemic Uncertainty Factor", "contribution": round(u_contrib, 1), "value": f"{uncertainty_variance:.4f}", "weight": f"{self.weights['uncertainty_penalty']*100:.0f}%", "description": "Monte Carlo Dropout dispersion penalty"}
        ]
        
        # Sort factors by contribution descending
        factors_sorted = sorted(factors, key=lambda x: x["contribution"], reverse=True)
        
        return {
            "total_computed_points": round(total_points, 1),
            "factors": factors_sorted,
            "top_driver": factors_sorted[0]["factor"],
            "secondary_driver": factors_sorted[1]["factor"]
        }

    def generate_narrative_explanation(
        self, 
        risk_category: str, 
        risk_score: float, 
        probability: float, 
        contributions: Dict[str, Any],
        focus_zone: str,
        uncertainty_level: str
    ) -> str:
        """
        Synthesizes a structured clinical research explanation.
        """
        top_driver = contributions.get("top_driver", "Deep Classifier Signal")
        sec_driver = contributions.get("secondary_driver", "Cranial Lemon Sign")
        
        if risk_category == "HIGH":
            narrative = (
                f"The system assigned a HIGH Risk Stratification score ({risk_score:.1f}/100) primarily driven by "
                f"pronounced {top_driver} and {sec_driver}. The deep learning model demonstrated a "
                f"{probability*100:.1f}% confidence in Spina Bifida-associated morphometry. "
                f"Visual attention concentrated predominantly on the {focus_zone}. "
                f"Uncertainty status is {uncertainty_level}. Immediate maternal-fetal medicine specialist ultrasound review is recommended."
            )
        elif risk_category == "MODERATE":
            narrative = (
                f"The system assigned a MODERATE Risk Stratification score ({risk_score:.1f}/100). "
                f"While the deep classifier signal was {probability*100:.1f}%, borderline morphological findings "
                f"({top_driver}) elevated the composite risk. Model attention mapped to the {focus_zone}. "
                f"Uncertainty status is {uncertainty_level}. A targeted follow-up anatomical ultrasound is advised to resolve equivocal features."
            )
        else:
            narrative = (
                f"The system assigned a LOW Risk Stratification score ({risk_score:.1f}/100). "
                f"Fetal cranial contour, ventricular area, and deep feature embeddings remain within normal limits "
                f"(Normal probability: {(1.0-probability)*100:.1f}%). "
                f"Uncertainty is {uncertainty_level}, confirming consistent standard cranial morphology."
            )
            
        return narrative
