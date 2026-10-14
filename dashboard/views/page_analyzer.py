"""
Live Clinical Risk Analyzer View (Page 1).
Full pipeline: Image Ingestion -> Preprocessing -> Classification -> Severity -> SB-RS -> Uncertainty -> Grad-CAM -> Decision Matrix -> What-If Sandbox -> PDF Report.
"""

import streamlit as st
import numpy as np
import cv2
import os
import glob
import torch
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

from config.settings import (
    DEVICE, TEST_IMAGES_DIR, NORMAL_DIR, SB_DIR, ARTIFACTS_DIR,
    CLASS_NAMES, load_risk_config
)
from preprocessing.transforms import load_and_preprocess_image, apply_clahe
from preprocessing.feature_extractor import RadiologicalFeatureExtractor
from models.resnet_model import SpinaBifidaResNet
from models.custom_cnn import SpinaBifidaCNN
from models.efficientnet_model import SpinaBifidaEfficientNet
from models.severity_estimator import SeverityEstimator
from risk.risk_score import SpinaBifidaRiskEngine
from risk.uncertainty import MonteCarloDropoutEstimator
from risk.decision_matrix import DecisionSupportMatrix
from risk.counterfactual import CounterfactualAnalyzer
from explainability.gradcam import GradCAM, GradCAMPlusPlus
from explainability.visualizer import generate_explainability_suite
from explainability.risk_explainer import RiskFactorExplainer
from reports.pdf_generator import generate_pdf_case_report
from dashboard.components import (
    render_disclaimer, render_sbrs_gauge, render_risk_factor_bar, render_decision_matrix_plot
)

@st.cache_resource
def load_cached_model(model_name: str = "resnet18"):
    """Loads and caches trained model weights."""
    if model_name == "custom_cnn":
        model = SpinaBifidaCNN()
        weights_path = ARTIFACTS_DIR / "spina_bifida_custom_cnn.pt"
    elif model_name == "efficientnet_b0":
        model = SpinaBifidaEfficientNet(pretrained=True)
        weights_path = ARTIFACTS_DIR / "spina_bifida_efficientnet_b0.pt"
    elif model_name in ["vit_transformer", "vit_b_16"]:
        from models.vit_model import SpinaBifidaViT
        model = SpinaBifidaViT(pretrained=True)
        weights_path = ARTIFACTS_DIR / "spina_bifida_vit_transformer.pt"
        if not weights_path.exists():
            weights_path = ARTIFACTS_DIR / "spina_bifida_vit.pt"
    else:
        model = SpinaBifidaResNet(backbone="resnet18", pretrained=True)
        weights_path = ARTIFACTS_DIR / "spina_bifida_resnet18.pt"
        
    if weights_path.exists():
        model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def render_page_analyzer():
    st.markdown('<div class="main-header">🔬 Fetal Ultrasound Risk Stratification Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable, Uncertainty-Aware Decision-Support System for Spina Bifida Evaluation</div>', unsafe_allow_html=True)
    
    render_disclaimer()
    
    # Sidebar options
    st.sidebar.markdown("### ⚙️ Diagnostic Settings")
    model_choice = st.sidebar.selectbox(
        "Active Deep Learning Architecture", 
        ["resnet18", "vit_transformer", "custom_cnn", "efficientnet_b0"], 
        index=0
    )
    
    is_vit = model_choice in ["vit_transformer", "vit_b_16"]
    if is_vit:
        explain_method = st.sidebar.selectbox("Explainability Method", ["ViT Self-Attention Map", "ViT Attention Rollout"], index=0)
    else:
        explain_method = st.sidebar.selectbox("Explainability Method", ["Grad-CAM", "Grad-CAM++"], index=0)
        
    mc_samples = st.sidebar.slider("Monte Carlo Dropout Samples (T)", min_value=5, max_value=40, value=20, step=5)
    gestational_week = st.sidebar.number_input("Gestational Age (Weeks)", min_value=12, max_value=38, value=20)
    
    model = load_cached_model(model_choice)
    feat_extractor = RadiologicalFeatureExtractor()
    sev_estimator = SeverityEstimator()
    risk_engine = SpinaBifidaRiskEngine()
    unc_estimator = MonteCarloDropoutEstimator(num_samples=mc_samples)
    explainer = RiskFactorExplainer()
    cf_analyzer = CounterfactualAnalyzer()
    
    # Target layer for Grad-CAM (if CNN)
    if not is_vit:
        if hasattr(model, "get_target_layer_for_gradcam"):
            target_layer = model.get_target_layer_for_gradcam()
        else:
            target_layer = model.conv4
        gradcam_engine = GradCAMPlusPlus(model, target_layer) if explain_method == "Grad-CAM++" else GradCAM(model, target_layer)
    else:
        gradcam_engine = None
    
    # 1. Ultrasound Input Selection
    st.markdown("#### 1. Input Ultrasound Selection")
    col_mode1, col_mode2 = st.columns([1, 1])
    
    with col_mode1:
        input_type = st.radio("Choose Input Mode:", ["Select Pre-loaded Benchmark Scans", "Upload Custom Ultrasound File (.JPG, .PNG)"])
        
    img_to_process = None
    case_label = "CASE-DEMO"
    
    if input_type == "Select Pre-loaded Benchmark Scans":
        test_files = glob.glob(str(TEST_IMAGES_DIR / "*.*"))
        # Add normal and spina bifida samples from dataset if available
        norm_samples = glob.glob(str(NORMAL_DIR / "*.*"))[:2]
        sb_samples = glob.glob(str(SB_DIR / "*.*"))[:2]
        sample_options = test_files + sb_samples + norm_samples
        
        if sample_options:
            selected_path = st.selectbox(
                "Select Fetal Ultrasound Scan:",
                sample_options,
                format_func=lambda x: f"{'🔴 Lemon Sign SB: ' if 'lemon' in x.lower() or 'Spina' in x else ('🟡 Banana Sign: ' if 'banana' in x.lower() else '🟢 Normal Scan: ')}{os.path.basename(x)}"
            )
            img_to_process = selected_path
            case_label = os.path.basename(selected_path).split(".")[0].upper()
    else:
        uploaded_file = st.file_uploader("Upload Ultrasound Scan", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            temp_path = os.path.join(ARTIFACTS_DIR, f"temp_upload_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            img_to_process = temp_path
            case_label = uploaded_file.name.split(".")[0].upper()
            
    if not img_to_process or not os.path.exists(img_to_process):
        st.info("👈 Please select or upload a fetal ultrasound image to run diagnostic stratification.")
        return
        
    # Run Complete Inference Pipeline
    with st.spinner("Analyzing ultrasound morphometry, generating Grad-CAM, and estimating uncertainty..."):
        tensor, img_resized, img_clahe = load_and_preprocess_image(img_to_process)
        tensor = tensor.to(DEVICE)
        
        # 1. Deterministic Model Prediction
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            sb_prob = float(probs[1])
            is_sb = pred_idx == 1
            pred_name = CLASS_NAMES[pred_idx]
            
        # 2. Epistemic Uncertainty via MC Dropout
        unc_data = unc_estimator.estimate_uncertainty(model, tensor)
        
        # 3. Radiological Proxy Feature Extraction
        # Deep feature energy
        with torch.no_grad():
            feat_vec = model.extract_features(tensor)
            deep_norm = float(torch.norm(feat_vec).item() / 50.0)
            
        rad_features = feat_extractor.extract_all_indicators(img_clahe, deep_feature_norm=deep_norm)
        
        # 4. Severity Assessment
        severity_data = sev_estimator.estimate_severity(is_sb, sb_prob, rad_features)
        
        # 5. SB-RS Risk Stratification (0-100)
        risk_data = risk_engine.calculate_score(sb_prob, rad_features, unc_data["variance"])
        
        # 6. Visual Explainability (ViT Attention vs CNN Grad-CAM)
        if is_vit or hasattr(model, "get_attention_map"):
            from explainability.vit_attention import ViTAttentionExplainer
            vit_explainer = ViTAttentionExplainer(model)
            rollout_flag = "attention_rollout" if "Rollout" in explain_method else "cls_attention"
            heatmap = vit_explainer.get_attention_heatmap(tensor, method=rollout_flag)
        else:
            heatmap = gradcam_engine.generate_heatmap(tensor, target_class=1 if is_sb else 0)
        vis_suite = generate_explainability_suite(img_resized, img_clahe, heatmap)
        
        # 7. Risk Factor Attribution Decomposition
        contributions = explainer.compute_contributions(sb_prob, rad_features, unc_data["variance"])
        narrative = explainer.generate_narrative_explanation(
            risk_data["risk_category"], risk_data["sb_rs_score"], sb_prob,
            contributions, vis_suite["focus_data"]["anatomical_zone"], unc_data["uncertainty_level"]
        )
        explanation_data = {
            "contributions": contributions,
            "narrative": narrative,
            "focus_data": vis_suite["focus_data"]
        }
        
        # 8. Risk x Confidence Decision Support Matrix
        decision_data = DecisionSupportMatrix.map_decision_state(
            risk_data["risk_category"], unc_data["confidence_level"], risk_data["sb_rs_score"]
        )
        
    # --- RENDER RESULTS ---
    st.markdown("---")
    st.markdown(f"### 📋 Case Stratification Results: `{case_label}`")
    
    # Metric Summary Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B;font-size:12px;font-weight:600;">AI DETECTION</span>
            <div style="font-size:20px;font-weight:700;color:{'#EF4444' if is_sb else '#10B981'};">{pred_name}</div>
            <div style="font-size:12px;color:#475569;">Confidence: <b>{sb_prob*100 if is_sb else (1-sb_prob)*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B;font-size:12px;font-weight:600;">SEVERITY GRADE</span>
            <div style="font-size:20px;font-weight:700;color:{severity_data['color']};">{severity_data['grade']}</div>
            <div style="font-size:12px;color:#475569;">Index: <b>{severity_data['severity_score']:.2f}</b> / 1.0</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B;font-size:12px;font-weight:600;">SB-RS RISK SCORE</span>
            <div style="font-size:20px;font-weight:700;color:{risk_data['risk_color']};">{risk_data['sb_rs_score']} / 100</div>
            <div style="font-size:12px;color:#475569;">Category: <b>{risk_data['risk_category']} RISK</b></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B;font-size:12px;font-weight:600;">EPISTEMIC UNCERTAINTY</span>
            <div style="font-size:20px;font-weight:700;color:{unc_data['color']};">{unc_data['uncertainty_level']}</div>
            <div style="font-size:12px;color:#475569;">Variance: <b>{unc_data['variance']:.4f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 2-Column Primary Analysis View
    c_left, c_right = st.columns([1.1, 1.0])
    
    with c_left:
        st.markdown(f"#### 🖼️ Visual Explainability & AI Attention ({explain_method})")
        v_tab1, v_tab2, v_tab3 = st.tabs(["AI Attention Overlay", "Side-by-Side Diagnostic", "CLAHE Enhanced"])
        
        with v_tab1:
            st.image(vis_suite["overlay_boxed"], caption=f"{explain_method} Overlay | Focus: {vis_suite['focus_data']['anatomical_zone']} (Peak: {vis_suite['focus_data']['peak_intensity']})", use_container_width=True)
        with v_tab2:
            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.image(vis_suite["original_rgb"], caption="Original Ultrasound Scan", use_container_width=True)
            with col_cam:
                st.image(vis_suite["heatmap"], caption=f"{explain_method} Spatial Map", use_container_width=True)
        with v_tab3:
            st.image(vis_suite["clahe_overlay_rgb"], caption=f"CLAHE Enhanced Ultrasound + {explain_method}", use_container_width=True)
            
        st.caption("ℹ️ *AI Attention Map indicates mathematical neural network gradient/attention activation; not a histopathologically confirmed lesion boundary.*")
        
    with c_right:
        st.markdown("#### 🎯 Risk Gauge & Decision Support Quadrant")
        st.plotly_chart(render_sbrs_gauge(risk_data["sb_rs_score"], risk_data["risk_category"], risk_data["risk_color"]), use_container_width=True)
        
        # Clinical Action Box
        st.markdown(f"""
        <div style="background-color:#F1F5F9;border-left:4px solid {decision_data['color']};padding:10px 14px;border-radius:4px;margin-bottom:10px;">
            <b style="color:{decision_data['color']};">{decision_data['quadrant']}</b><br/>
            <span style="font-size:12.5px;color:#334155;"><b>Action Protocol:</b> {decision_data['action_protocol']}</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Secondary Row: Risk Attribution & Decision Matrix
    st.markdown("---")
    r1, r2 = st.columns([1.2, 1.0])
    
    with r1:
        st.plotly_chart(render_risk_factor_bar(contributions["factors"]), use_container_width=True)
        st.markdown(f"**Clinical Research Narrative Summary:**")
        st.info(narrative)
        
    with r2:
        st.plotly_chart(render_decision_matrix_plot(decision_data["matrix_coordinates"]), use_container_width=True)
        st.markdown(f"""
        **Uncertainty Decomposition:**
        - **MC Dropout Predictive Mean:** `{unc_data['mean_probability']*100:.1f}%`
        - **95% Confidence Interval:** `[{unc_data['confidence_interval_95'][0]*100:.1f}%, {unc_data['confidence_interval_95'][1]*100:.1f}%]`
        - **Predictive Entropy:** `{unc_data['entropy']:.4f}`
        - **Epistemic Variance:** `{unc_data['variance']:.5f}`
        """)
        
    # Interactive Counterfactual Sandbox ("What-If" Analysis)
    st.markdown("---")
    with st.expander("🛠️ Interactive Counterfactual 'What-If' Risk Sandbox (Research Tool)", expanded=False):
        st.markdown("""
        *Simulate the mathematical sensitivity of the SB-RS risk engine to perturbations in individual radiological indicators.*
        *(For exploratory sensitivity research only; does not simulate physical biological intervention).*
        """)
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            delta_concavity = st.slider("Perturb Cranial Concavity (Lemon Sign)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
        with cf2:
            delta_vent = st.slider("Perturb Ventricular Dilation Proxy", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
        with cf3:
            delta_unc = st.slider("Perturb Uncertainty Variance", min_value=-0.05, max_value=0.05, value=0.0, step=0.01)
            
        cf_res = cf_analyzer.simulate_counterfactual(
            base_probability=sb_prob,
            base_radiological_features=rad_features,
            base_uncertainty=unc_data["variance"],
            delta_cranial_concavity=delta_concavity,
            delta_ventricular_dilation=delta_vent,
            delta_uncertainty=delta_unc
        )
        
        cfa, cfb, cfc = st.columns(3)
        cfa.metric("Baseline SB-RS Score", f"{cf_res['baseline_score']} / 100", cf_res['baseline_category'])
        cfb.metric("Simulated Counterfactual SB-RS", f"{cf_res['counterfactual_score']} / 100", f"{cf_res['delta_sbrs']:+0.1f} pts")
        cfc.metric("Simulated Category", cf_res['counterfactual_category'], f"Delta: {cf_res['delta_sbrs']:+0.1f}")
        
    # PDF & JSON Export Section
    st.markdown("---")
    st.markdown("#### 📄 Clinical Research Case Report Export")
    col_pdf, col_json = st.columns([1, 1])
    
    with col_pdf:
        if st.button("📑 Generate & Download Official PDF Case Report", type="primary"):
            with st.spinner("Compiling structured ReportLab PDF case document..."):
                pdf_path = generate_pdf_case_report(
                    case_id=case_label,
                    detection_data={"prediction": pred_name, "probability": sb_prob},
                    severity_data=severity_data,
                    risk_data=risk_data,
                    uncertainty_data=unc_data,
                    explanation_data=explanation_data,
                    decision_data=decision_data,
                    original_img_rgb=vis_suite["original_rgb"],
                    overlay_img_rgb=vis_suite["overlay_boxed"],
                    gestational_week=gestational_week
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                st.download_button(
                    label="⬇️ Click to Download PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
                st.success(f"✅ PDF Report generated: `{os.path.basename(pdf_path)}`")
                
    with col_json:
        export_payload = {
            "case_id": case_label,
            "gestational_week": gestational_week,
            "detection": {"class": pred_name, "probability": round(sb_prob, 4)},
            "severity": severity_data,
            "sb_rs_risk": risk_data,
            "uncertainty": unc_data,
            "decision_support": decision_data,
            "radiological_features": rad_features,
            "risk_contributions": contributions
        }
        st.download_button(
            label="💾 Export Machine-Readable JSON Record",
            data=str(export_payload),
            file_name=f"case_record_{case_label}.json",
            mime="application/json"
        )
