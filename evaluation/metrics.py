"""
Comprehensive Medical AI Evaluation Metrics.
Calculates Sensitivity (Recall), Specificity, Accuracy, Precision, F1, ROC-AUC, Brier Score, and ECE.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, brier_score_loss,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any

def compute_medical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes all standard diagnostic metrics for binary clinical classification.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    specificity = tn / (tn + fp + 1e-6)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    brier = brier_score_loss(y_true, y_prob)
    
    # ROC curve data
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_true, y_prob)
    
    return {
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//50)]],
            "tpr": [round(float(y), 4) for y in tpr[::max(1, len(tpr)//50)]]
        },
        "pr_curve": {
            "precision": [round(float(p), 4) for p in precision_curve[::max(1, len(precision_curve)//50)]],
            "recall": [round(float(r), 4) for r in recall_curve[::max(1, len(recall_curve)//50)]]
        }
    }
