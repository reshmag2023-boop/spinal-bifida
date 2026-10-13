"""
Radiological Feature Extraction Engine for Fetal Ultrasound Scans.
Extracts quantifiable image-derived morphological and texture indicators:
1. Cranial Frontal Bone Concavity / Indentation Index (Lemon Sign proxy)
2. Transventricular Hypoechoic Ratio (Ventriculomegaly proxy)
3. Posterior Fossa Acoustic Variance (Chiari II / Cisterna Magna effacement proxy)
4. Texture Heterogeneity & Gradient Dispersion
"""

import cv2
import numpy as np
from typing import Dict, Any, Union
import torch

class RadiologicalFeatureExtractor:
    """
    Extracts measurable radiological indicators from fetal ultrasound images.
    All metrics are computed mathematically from ultrasound pixel intensities
    and morphological contour gradients.
    """
    
    @staticmethod
    def extract_cranial_concavity(image_np: np.ndarray) -> float:
        """
        Measures frontal bone indentation / concavity ratio in upper cranial hemisphere.
        Higher values indicates inward scalloping (Lemon sign).
        Range: [0.0, 1.0]
        """
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()
            
        rows, cols = gray.shape
        # Focus on frontal cranial bone region (upper 45%)
        upper_region = gray[:int(rows * 0.45), :]
        
        # Adaptive thresholding to segment cranial vault outline
        blur = cv2.GaussianBlur(upper_region, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Compute horizontal edge curvature via Sobel
        sobel_x = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Inward pinch creates lateral inflection points (variance in lateral gradients)
        mid_col = cols // 2
        left_grad = gradient_mag[:, :mid_col]
        right_grad = gradient_mag[:, mid_col:]
        
        curvature_energy = (np.mean(left_grad) + np.mean(right_grad)) / 255.0
        # Normalize to 0.0 - 1.0
        concavity_score = float(np.clip(curvature_energy * 2.2, 0.0, 1.0))
        return concavity_score

    @staticmethod
    def extract_ventricular_dilation_proxy(image_np: np.ndarray) -> float:
        """
        Measures central transventricular hypoechoic (fluid-filled) area ratio.
        Ventriculomegaly in Spina Bifida manifests as dilated lateral ventricles.
        Range: [0.0, 1.0]
        """
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()
            
        rows, cols = gray.shape
        # Central transventricular brain slice region (middle 50% horizontally, middle 40% vertically)
        y_start, y_end = int(rows * 0.30), int(rows * 0.70)
        x_start, x_end = int(cols * 0.25), int(cols * 0.75)
        central_region = gray[y_start:y_end, x_start:x_end]
        
        # Ventricles appear as low-intensity hypoechoic (dark) fluid cavities
        low_intensity_mask = central_region < 50
        hypoechoic_ratio = np.sum(low_intensity_mask) / (central_region.size + 1e-6)
        
        # Scaled indicator
        ventricular_score = float(np.clip(hypoechoic_ratio * 2.5, 0.0, 1.0))
        return ventricular_score

    @staticmethod
    def extract_posterior_fossa_score(image_np: np.ndarray) -> float:
        """
        Measures posterior fossa acoustic texture disruption.
        In Chiari II, downward cerebellar displacement effaces the cisterna magna.
        Range: [0.0, 1.0]
        """
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()
            
        rows, cols = gray.shape
        # Posterior fossa region (lower 35%)
        posterior_region = gray[int(rows * 0.65):, :]
        
        # Texture variance using Laplacian operator
        laplacian_var = cv2.Laplacian(posterior_region, cv2.CV_64F).var()
        norm_laplacian = float(np.clip(laplacian_var / 800.0, 0.0, 1.0))
        return norm_laplacian

    @staticmethod
    def extract_image_sharpness_and_noise(image_np: np.ndarray) -> Dict[str, float]:
        """Evaluates overall image acquisition quality metrics."""
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()
            
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        snr_proxy = float(mean_intensity / (std_intensity + 1e-6))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        return {
            "mean_intensity": round(mean_intensity, 2),
            "contrast_std": round(std_intensity, 2),
            "snr_estimate": round(snr_proxy, 2),
            "laplacian_blur_index": round(blur_score, 2)
        }

    def extract_all_indicators(
        self, 
        image_np: np.ndarray, 
        deep_feature_norm: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive feature extraction returning all mathematical indicators.
        """
        concavity = self.extract_cranial_concavity(image_np)
        ventricular = self.extract_ventricular_dilation_proxy(image_np)
        posterior = self.extract_posterior_fossa_score(image_np)
        quality = self.extract_image_sharpness_and_noise(image_np)
        
        # Combined morphological abnormality composite
        morphological_composite = float(
            0.50 * concavity + 0.30 * ventricular + 0.20 * posterior
        )
        
        return {
            "cranial_concavity_score": round(concavity, 4),
            "ventricular_dilation_proxy": round(ventricular, 4),
            "posterior_fossa_score": round(posterior, 4),
            "deep_abnormality_score": round(float(np.clip(deep_feature_norm, 0.0, 1.0)), 4),
            "morphological_composite": round(morphological_composite, 4),
            "image_quality": quality,
            "annotation_status": "Derived image-radiological proxies (Research Prototype)"
        }
