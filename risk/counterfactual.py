"""
Counterfactual "What-If" Sensitivity Analysis Sandbox.
Enables researchers to perturb individual radiological indicators
and observe the simulated delta in SB-RS score and risk category.
"""

from typing import Dict, Any
from risk.risk_score import SpinaBifidaRiskEngine

class CounterfactualAnalyzer:
    """
    Computes simulated counterfactual risk responses to feature perturbations.
    """
    def __init__(self):
        self.engine = SpinaBifidaRiskEngine()

    def simulate_counterfactual(
        self,
        base_probability: float,
        base_radiological_features: Dict[str, Any],
        base_uncertainty: float,
        delta_cranial_concavity: float = 0.0,
        delta_ventricular_dilation: float = 0.0,
        delta_deep_abnormality: float = 0.0,
        delta_uncertainty: float = 0.0
    ) -> Dict[str, Any]:
        """
        Simulates what-if scenario by applying offsets to original indicators.
        """
        # Baseline score
        base_result = self.engine.calculate_score(
            base_probability, base_radiological_features, base_uncertainty
        )
        
        # Perturbed indicators
        cf_features = {
            "cranial_concavity_score": max(0.0, min(1.0, base_radiological_features.get("cranial_concavity_score", 0.5) + delta_cranial_concavity)),
            "ventricular_dilation_proxy": max(0.0, min(1.0, base_radiological_features.get("ventricular_dilation_proxy", 0.5) + delta_ventricular_dilation)),
            "deep_abnormality_score": max(0.0, min(1.0, base_radiological_features.get("deep_abnormality_score", 0.5) + delta_deep_abnormality)),
        }
        cf_uncertainty = max(0.001, min(0.12, base_uncertainty + delta_uncertainty))
        
        cf_result = self.engine.calculate_score(
            base_probability, cf_features, cf_uncertainty
        )
        
        delta_sbrs = cf_result["sb_rs_score"] - base_result["sb_rs_score"]
        
        return {
            "baseline_score": base_result["sb_rs_score"],
            "baseline_category": base_result["risk_category"],
            "counterfactual_score": cf_result["sb_rs_score"],
            "counterfactual_category": cf_result["risk_category"],
            "delta_sbrs": round(delta_sbrs, 1),
            "simulated_features": cf_features,
            "simulated_uncertainty": round(cf_uncertainty, 4),
            "scientific_note": "Counterfactual Sensitivity Model: For exploratory sensitivity research only; does not simulate physical biological intervention."
        }
