"""
Unit and Integration Test Suite for Explainable and Uncertainty-Aware Spina Bifida AI System.
"""

import os
import pytest
import torch
import numpy as np
from pathlib import Path

from config.settings import (
    BASE_DIR, TEST_IMAGES_DIR, NORMAL_DIR, SB_DIR, DEVICE, IMG_SIZE, load_risk_config
)
from preprocessing.transforms import (
    load_and_preprocess_image, apply_clahe, get_train_transforms, get_val_transforms
)
from preprocessing.lemonizer import LemonSignGenerator
from preprocessing.feature_extractor import RadiologicalFeatureExtractor
from models.custom_cnn import SpinaBifidaCNN
from models.resnet_model import SpinaBifidaResNet
from models.efficientnet_model import SpinaBifidaEfficientNet
from models.vit_model import SpinaBifidaViT
from models.severity_estimator import SeverityEstimator
from risk.risk_score import SpinaBifidaRiskEngine
from risk.uncertainty import MonteCarloDropoutEstimator
from risk.decision_matrix import DecisionSupportMatrix
from risk.longitudinal import LongitudinalRiskTracker
from risk.counterfactual import CounterfactualAnalyzer
from explainability.gradcam import GradCAM, GradCAMPlusPlus
from explainability.visualizer import generate_explainability_suite
from explainability.risk_explainer import RiskFactorExplainer
from explainability.vit_attention import ViTAttentionExplainer
from reports.pdf_generator import generate_pdf_case_report

@pytest.fixture
def sample_image_path():
    test_files = list(TEST_IMAGES_DIR.glob("*.*"))
    if test_files:
        return str(test_files[0])
    normal_files = list(NORMAL_DIR.glob("*.*"))
    if normal_files:
        return str(normal_files[0])
    # Fallback dummy file
    dummy_path = str(TEST_IMAGES_DIR / "dummy_test.png")
    import cv2
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    cv2.imwrite(dummy_path, dummy_img)
    return dummy_path

def test_preprocessing_and_clahe(sample_image_path):
    tensor, raw_rgb, clahe_rgb = load_and_preprocess_image(sample_image_path)
    assert tensor.shape == (1, 3, 224, 224)
    assert raw_rgb.shape == (224, 224, 3)
    assert clahe_rgb.shape == (224, 224, 3)
    assert isinstance(tensor, torch.Tensor)

def test_lemonizer():
    generator = LemonSignGenerator(random_seed=42)
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 120
    deformed, pinch_val = generator.apply_lemon_deformation(dummy_img, pinch_strength=0.0025)
    assert deformed.shape == (224, 224, 3)
    assert pinch_val == 0.0025

def test_radiological_feature_extractor():
    extractor = RadiologicalFeatureExtractor()
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 100
    features = extractor.extract_all_indicators(dummy_img, deep_feature_norm=0.5)
    
    assert "cranial_concavity_score" in features
    assert "ventricular_dilation_proxy" in features
    assert "posterior_fossa_score" in features
    assert "morphological_composite" in features
    assert 0.0 <= features["cranial_concavity_score"] <= 1.0

def test_model_forward_passes():
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # CNN
    cnn = SpinaBifidaCNN()
    out_cnn = cnn(dummy_input)
    assert out_cnn.shape == (2, 2)
    
    # ResNet
    resnet = SpinaBifidaResNet(backbone="resnet18", pretrained=False)
    out_res = resnet(dummy_input)
    assert out_res.shape == (2, 2)
    feat_res = resnet.extract_features(dummy_input)
    assert feat_res.shape[0] == 2

def test_vit_model_and_attention():
    # Fast test of Vision Transformer forward pass, feature extraction, and attention
    vit = SpinaBifidaViT(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 1. Forward pass
    out_vit = vit(dummy_input)
    assert out_vit.shape == (2, 2)
    
    # 2. Latent feature extraction ([CLS] embedding)
    feats = vit.extract_features(dummy_input)
    assert feats.shape[0] == 2
    assert feats.shape[1] == vit.embed_dim
    
    # 3. MC Dropout activation
    vit.enable_mc_dropout()
    
    # 4. Attention map extraction (14x14 patches)
    attn_map = vit.get_attention_map(dummy_input)
    assert attn_map.shape == (2, 14, 14)
    
    # 5. ViT Attention Explainer module
    explainer = ViTAttentionExplainer(vit)
    single_tensor = dummy_input[:1]
    heatmap_cls = explainer.get_attention_heatmap(single_tensor, method="cls_attention")
    assert heatmap_cls.shape == (224, 224)
    assert 0.0 <= np.min(heatmap_cls) <= np.max(heatmap_cls) <= 1.0
    
    heatmap_rollout = explainer.get_attention_heatmap(single_tensor, method="attention_rollout")
    assert heatmap_rollout.shape == (224, 224)
    assert 0.0 <= np.min(heatmap_rollout) <= np.max(heatmap_rollout) <= 1.0

def test_severity_estimator():
    estimator = SeverityEstimator()
    dummy_feats = {
        "cranial_concavity_score": 0.85,
        "ventricular_dilation_proxy": 0.70,
        "posterior_fossa_score": 0.60
    }
    # Test SB positive
    sev_sb = estimator.estimate_severity(is_spina_bifida=True, probability=0.92, radiological_features=dummy_feats)
    assert sev_sb["grade"] in ["Mild", "Moderate", "Severe"]
    assert 0.0 <= sev_sb["severity_score"] <= 1.0
    
    # Test Normal
    sev_norm = estimator.estimate_severity(is_spina_bifida=False, probability=0.05, radiological_features=dummy_feats)
    assert "Normal" in sev_norm["grade"]

def test_risk_score_engine():
    engine = SpinaBifidaRiskEngine()
    dummy_feats = {
        "cranial_concavity_score": 0.75,
        "ventricular_dilation_proxy": 0.60,
        "deep_abnormality_score": 0.80
    }
    res = engine.calculate_score(probability=0.88, radiological_features=dummy_feats, uncertainty_variance=0.015)
    assert 0.0 <= res["sb_rs_score"] <= 100.0
    assert res["risk_category"] in ["LOW", "MODERATE", "HIGH"]

def test_mc_dropout_uncertainty():
    model = SpinaBifidaCNN()
    estimator = MonteCarloDropoutEstimator(num_samples=5)
    dummy_input = torch.randn(1, 3, 224, 224)
    unc = estimator.estimate_uncertainty(model, dummy_input)
    
    assert "mean_probability" in unc
    assert "variance" in unc
    assert "entropy" in unc
    assert unc["uncertainty_level"] in ["LOW", "MODERATE", "HIGH"]

def test_decision_matrix():
    matrix_res = DecisionSupportMatrix.map_decision_state(
        risk_category="HIGH", confidence_level="LOW", sb_rs_score=78.0
    )
    assert matrix_res["priority"] == "URGENT_UNCERTAIN"
    assert "Mandatory Senior Specialist Review" in matrix_res["action_protocol"]

def test_gradcam():
    model = SpinaBifidaResNet(backbone="resnet18", pretrained=False)
    target_layer = model.get_target_layer_for_gradcam()
    gradcam = GradCAM(model, target_layer)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(dummy_input, target_class=1)
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

def test_pdf_report_generation():
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 150
    detection_data = {"prediction": "Spina Bifida", "probability": 0.914}
    severity_data = {"grade": "Moderate", "severity_score": 0.58, "color": "#F59E0B"}
    risk_data = {"sb_rs_score": 76.5, "risk_category": "HIGH", "risk_color": "#EF4444"}
    uncertainty_data = {"uncertainty_level": "LOW", "variance": 0.012, "confidence_level": "High"}
    explanation_data = {
        "contributions": {
            "factors": [
                {"factor": "Deep Classifier Signal", "contribution": 32.0, "value": "91.4%", "weight": "35%"},
                {"factor": "Cranial Lemon Sign", "contribution": 18.5, "value": "0.74", "weight": "25%"}
            ]
        },
        "narrative": "High risk Spina Bifida stratification case.",
        "focus_data": {"anatomical_zone": "Frontal Cranial Vault"}
    }
    decision_data = {
        "quadrant": "🔴 High Risk + High Confidence",
        "action_protocol": "Immediate specialist referral advised."
    }
    
    pdf_path = generate_pdf_case_report(
        case_id="TEST-CASE-01",
        detection_data=detection_data,
        severity_data=severity_data,
        risk_data=risk_data,
        uncertainty_data=uncertainty_data,
        explanation_data=explanation_data,
        decision_data=decision_data,
        original_img_rgb=dummy_img,
        overlay_img_rgb=dummy_img,
        gestational_week=20,
        output_filename="test_case_report.pdf"
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000 # Valid PDF file generated
