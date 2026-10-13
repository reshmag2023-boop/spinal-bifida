"""
Risk x Confidence 2D Decision Support Matrix.
Cross-analyzes composite SB-RS risk against epistemic uncertainty
to identify high-risk uncertain cases requiring mandatory specialist review.
"""

from typing import Dict, Any

class DecisionSupportMatrix:
    """
    Maps clinical case state into a 2D decision quadrant.
    """
    @staticmethod
    def map_decision_state(
        risk_category: str, 
        confidence_level: str, 
        sb_rs_score: float
    ) -> Dict[str, Any]:
        """
        Determines quadrant position and clinical recommended protocol.
        """
        risk_upper = risk_category.upper()
        conf_upper = confidence_level.upper()
        
        if risk_upper == "HIGH" and conf_upper == "HIGH":
            quadrant = "🔴 High Risk + High Confidence"
            action = "Immediate Specialist Referral: Pronounced Spina Bifida features confirmed with high model certainty."
            priority = "URGENT_CONFIRMED"
            color = "#EF4444"
            code = "H-H"
        elif risk_upper == "HIGH" and conf_upper in ["LOW", "MODERATE"]:
            quadrant = "🟠 High Risk + High Uncertainty"
            action = "Mandatory Senior Specialist Review: High risk score flagged but model uncertainty is elevated (possible atypical view or acoustic artifacts)."
            priority = "URGENT_UNCERTAIN"
            color = "#F97316"
            code = "H-L"
        elif risk_upper == "MODERATE" and conf_upper == "HIGH":
            quadrant = "🟡 Moderate Risk + High Confidence"
            action = "Targeted Surveillance: Borderline anatomical features confirmed; schedule targeted follow-up ultrasound in 2-3 weeks."
            priority = "MODERATE_CONFIRMED"
            color = "#EAB308"
            code = "M-H"
        elif risk_upper == "MODERATE" and conf_upper in ["LOW", "MODERATE"]:
            quadrant = "🟡 Moderate Risk + High Uncertainty"
            action = "Targeted Repeat Scan: Equivocal metrics compounded by high predictive variance; repeat imaging with optimized acoustic window."
            priority = "MODERATE_UNCERTAIN"
            color = "#F59E0B"
            code = "M-L"
        elif risk_upper == "LOW" and conf_upper == "HIGH":
            quadrant = "🟢 Low Risk + High Confidence"
            action = "Routine Prenatal Care: Standard fetal cranial biometry within physiological limits."
            priority = "ROUTINE_NORMAL"
            color = "#10B981"
            code = "L-H"
        else: # LOW + LOW
            quadrant = "🔵 Low Risk + High Uncertainty"
            action = "Technical Re-acquisition: Low risk computed but model confidence is degraded; verify image probe alignment."
            priority = "REACQUISITION_ADVISED"
            color = "#3B82F6"
            code = "L-L"
            
        return {
            "quadrant": quadrant,
            "code": code,
            "action_protocol": action,
            "priority": priority,
            "color": color,
            "matrix_coordinates": {
                "x_confidence": 1.0 if conf_upper == "HIGH" else (0.5 if conf_upper == "MODERATE" else 0.0),
                "y_risk": sb_rs_score
            }
        }
