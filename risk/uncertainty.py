"""
Epistemic Uncertainty Quantification via Monte Carlo Dropout & Predictive Entropy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from config.settings import DEVICE, load_risk_config

class MonteCarloDropoutEstimator:
    """
    Executes stochastic forward passes with active dropout at test-time (MC Dropout)
    to estimate epistemic model uncertainty, predictive variance, and entropy.
    """
    def __init__(self, num_samples: int = 20):
        self.num_samples = num_samples
        config = load_risk_config()
        self.unc_config = config.get("uncertainty", {
            "low_threshold": 0.02,
            "high_threshold": 0.06,
            "entropy_threshold": 0.60
        })

    def estimate_uncertainty(
        self, 
        model: nn.Module, 
        input_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Runs T forward passes with active dropout and computes distribution stats.
        """
        model.eval()
        # Enable dropout layers
        if hasattr(model, "enable_mc_dropout"):
            model.enable_mc_dropout()
        else:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
                    
        input_tensor = input_tensor.to(DEVICE)
        sampled_probs = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1].item() # P(Spina Bifida)
                sampled_probs.append(probs)
                
        sampled_probs = np.array(sampled_probs)
        mean_prob = float(np.mean(sampled_probs))
        variance = float(np.var(sampled_probs))
        std_dev = float(np.std(sampled_probs))
        
        # 95% Confidence interval
        ci_lower = float(np.clip(np.percentile(sampled_probs, 2.5), 0.0, 1.0))
        ci_upper = float(np.clip(np.percentile(sampled_probs, 97.5), 0.0, 1.0))
        
        # Predictive Entropy: H(p) = - p*log2(p) - (1-p)*log2(1-p)
        eps = 1e-7
        p_safe = np.clip(mean_prob, eps, 1.0 - eps)
        entropy = float(- (p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe)))
        
        # Uncertainty Level Classification
        if variance >= self.unc_config["high_threshold"] or entropy >= self.unc_config["entropy_threshold"]:
            level = "HIGH"
            color = "#EF4444"
            badge = "⚠️ High Uncertainty — Specialist Review Required"
            confidence_level = "Low"
        elif variance >= self.unc_config["low_threshold"]:
            level = "MODERATE"
            color = "#F59E0B"
            badge = "⚖️ Moderate Uncertainty"
            confidence_level = "Moderate"
        else:
            level = "LOW"
            color = "#10B981"
            badge = "✅ High Confidence"
            confidence_level = "High"
            
        return {
            "mean_probability": round(mean_prob, 4),
            "variance": round(variance, 5),
            "std_dev": round(std_dev, 4),
            "entropy": round(entropy, 4),
            "confidence_interval_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "uncertainty_level": level,
            "confidence_level": confidence_level,
            "badge": badge,
            "color": color,
            "mc_samples": self.num_samples,
            "sample_distribution": [round(float(p), 4) for p in sampled_probs]
        }

def evaluate_uncertainty(model: nn.Module, input_tensor: torch.Tensor, samples: int = 20) -> Dict[str, Any]:
    estimator = MonteCarloDropoutEstimator(num_samples=samples)
    return estimator.estimate_uncertainty(model, input_tensor)
