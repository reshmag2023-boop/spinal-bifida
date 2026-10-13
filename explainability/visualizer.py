"""
Visualizer Utilities for Multimodal Medical Explainability.
Generates overlays, bounding contours, and multi-view comparisons.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple

def create_gradcam_overlay(
    image_rgb: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Blends the Grad-CAM heatmap over the original ultrasound image.
    """
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(alpha * colored_heatmap_rgb + (1.0 - alpha) * image_rgb)
    return overlay

def detect_attention_focus_bbox(heatmap: np.ndarray, threshold: float = 0.60) -> Dict[str, Any]:
    """
    Finds the primary region of peak model attention and returns bounding box coordinates.
    """
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        center_x, center_y = x + w // 2, y + h // 2
        
        # Anatomical zone approximation
        zone = "Frontal Cranial Vault" if center_y < 100 else ("Transventricular Midline" if center_y < 160 else "Posterior Fossa")
        return {
            "has_focus": True,
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": [int(center_x), int(center_y)],
            "anatomical_zone": zone,
            "peak_intensity": round(float(np.max(heatmap)), 3)
        }
        
    return {
        "has_focus": False,
        "bbox": [0, 0, 0, 0],
        "center": [112, 112],
        "anatomical_zone": "Diffuse / Unfocused",
        "peak_intensity": round(float(np.max(heatmap)), 3)
    }

def generate_explainability_suite(
    image_rgb: np.ndarray, 
    image_clahe: np.ndarray,
    heatmap: np.ndarray
) -> Dict[str, Any]:
    """
    Creates complete explainability visualization payload.
    """
    overlay = create_gradcam_overlay(image_rgb, heatmap, alpha=0.45)
    clahe_overlay = create_gradcam_overlay(image_clahe, heatmap, alpha=0.45)
    focus_data = detect_attention_focus_bbox(heatmap)
    
    # Draw focus rectangle on overlay copy
    overlay_boxed = overlay.copy()
    if focus_data["has_focus"]:
        x, y, w, h = focus_data["bbox"]
        cv2.rectangle(overlay_boxed, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(overlay_boxed, f"AI Focus: {focus_data['anatomical_zone']}", (max(5, x), max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    
    return {
        "original_rgb": image_rgb,
        "clahe_rgb": image_clahe,
        "heatmap": heatmap,
        "overlay_rgb": overlay,
        "clahe_overlay_rgb": clahe_overlay,
        "overlay_boxed": overlay_boxed,
        "focus_data": focus_data,
        "scientific_disclaimer": "AI Attention Map: Indicates mathematical model activation; not a histopathologically validated lesion boundary."
    }
