"""
Reliability Calibration and Expected Calibration Error (ECE) Analysis.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class ReliabilityCalibrator:
    """
    Evaluates whether model confidence corresponds to observed empirical accuracy.
    """
    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Generates reliability diagram data across confidence bins.
        """
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
        
        confidences = []
        accuracies = []
        bin_counts = []
        ece = 0.0
        total_samples = len(y_true)
        
        for i in range(n_bins):
            b_low, b_high = bin_boundaries[i], bin_boundaries[i+1]
            if i == n_bins - 1:
                mask = (y_prob >= b_low) & (y_prob <= b_high)
            else:
                mask = (y_prob >= b_low) & (y_prob < b_high)
                
            count = int(np.sum(mask))
            bin_counts.append(count)
            
            if count > 0:
                acc = float(np.mean(y_true[mask]))
                conf = float(np.mean(y_prob[mask]))
                accuracies.append(round(acc, 4))
                confidences.append(round(conf, 4))
                ece += np.abs(conf - acc) * (count / (total_samples + 1e-6))
            else:
                accuracies.append(round(float(bin_centers[i]), 4))
                confidences.append(round(float(bin_centers[i]), 4))
                
        return {
            "bin_centers": [round(float(c), 3) for c in bin_centers],
            "accuracies": accuracies,
            "confidences": confidences,
            "bin_counts": bin_counts,
            "ece": round(float(ece), 4),
            "perfect_calibration": [round(float(c), 3) for c in bin_centers]
        }
