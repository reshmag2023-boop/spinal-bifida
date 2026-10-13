"""
Model Evaluation Engine computing Accuracy, Sensitivity, Specificity, ROC-AUC, F1, and ECE.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)
from typing import Dict, Any, Tuple

from config.settings import DEVICE

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
    """
    Evaluates model on validation/test dataset and returns full medical AI metrics.
    """
    model = model.to(DEVICE)
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(DEVICE)
            t0 = time.time()
            outputs = model(inputs)
            t1 = time.time()
            
            inference_times.append((t1 - t0) * 1000 / inputs.size(0)) # ms per image
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Probability of Spina Bifida
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp + 1e-6)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    avg_latency = float(np.mean(inference_times))
    
    return {
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(recall), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "inference_time_ms": round(avg_latency, 2),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist()
    }
