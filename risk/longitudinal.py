"""
Longitudinal Gestational Scan Tracker and Risk Velocity Engine.
Tracks sequential ultrasound scans over gestational age (T1, T2, T3, ...)
and evaluates risk trend velocity (Increasing / Decreasing / Stable).
"""

from typing import List, Dict, Any
import numpy as np

class LongitudinalRiskTracker:
    """
    Manages longitudinal patient scan trajectories across gestational weeks.
    """
    @staticmethod
    def analyze_trajectory(scans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a sequence of timepoint scans.
        Each scan dict contains: {'timepoint': 'T1', 'gestational_week': 16, 'sb_rs_score': 38, ...}
        """
        if not scans:
            return {"status": "No longitudinal data", "trajectory": "Unknown"}
            
        # Sort by gestational week or timepoint index
        sorted_scans = sorted(scans, key=lambda s: s.get("gestational_week", 0))
        scores = [s["sb_rs_score"] for s in sorted_scans]
        weeks = [s.get("gestational_week", i+1) for i, s in enumerate(sorted_scans)]
        
        if len(scores) < 2:
            return {
                "status": "Single scan recorded",
                "trajectory": "Baseline Established",
                "risk_velocity": 0.0,
                "scans": sorted_scans
            }
            
        # Compute linear slope / velocity (delta risk per week)
        delta_score = scores[-1] - scores[0]
        delta_weeks = max(1, weeks[-1] - weeks[0])
        velocity = delta_score / delta_weeks
        
        if velocity > 2.0:
            trajectory = "Increasing Risk (Accelerating)"
            trend_color = "#EF4444"
            recommendation = "Rapid progression in morphological risk markers; schedule multidisciplinary fetal surgery consultation."
        elif velocity < -2.0:
            trajectory = "Decreasing Risk (Resolving)"
            trend_color = "#10B981"
            recommendation = "Anatomical markers normalizing; continue standard serial monitoring."
        else:
            trajectory = "Stable Risk Profile"
            trend_color = "#F59E0B"
            recommendation = "Metrics remain consistent across gestational intervals; maintain scheduled follow-ups."
            
        return {
            "trajectory": trajectory,
            "trend_color": trend_color,
            "risk_velocity_per_week": round(velocity, 2),
            "initial_score": scores[0],
            "latest_score": scores[-1],
            "score_delta": round(delta_score, 1),
            "recommendation": recommendation,
            "timepoints": [s.get("timepoint", f"T{i+1}") for i, s in enumerate(sorted_scans)],
            "weeks": weeks,
            "scores": scores,
            "scans": sorted_scans
        }
