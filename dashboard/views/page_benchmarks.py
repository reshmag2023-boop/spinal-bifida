"""
Model Comparison & Calibration Analytics View (Page 3).
Compares Custom CNN, ResNet-18, EfficientNet-B0, and ViT with real computed metrics:
Accuracy, Sensitivity, Specificity, ROC-AUC, PR Curves, Confusion Matrices, and ECE Reliability Diagrams.
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config.settings import ARTIFACTS_DIR
from dashboard.components import render_disclaimer

def render_page_benchmarks():
    st.markdown('<div class="main-header">📈 Multi-Model Benchmark & Calibration Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Experimental Comparison of Deep Architectures on Fetal Ultrasound Detection</div>', unsafe_allow_html=True)
    
    render_disclaimer()
    
    bench_file = ARTIFACTS_DIR / "benchmarks.json"
    if not bench_file.exists():
        st.warning("⚠️ Benchmark metrics file `model_artifacts/benchmarks.json` not found. Please run the benchmark runner.")
        return
        
    with open(bench_file, "r") as f:
        benchmarks = json.load(f)
        
    # Table of Model Comparison
    table_rows = []
    for m_name, data in benchmarks.items():
        table_rows.append({
            "Architecture": m_name.upper(),
            "Accuracy": f"{data.get('accuracy', 0)*100:.2f}%",
            "Sensitivity (Recall)": f"{data.get('sensitivity', 0)*100:.2f}%",
            "Specificity": f"{data.get('specificity', 0)*100:.2f}%",
            "Precision": f"{data.get('precision', 0)*100:.2f}%",
            "F1-Score": f"{data.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{data.get('roc_auc', 0):.4f}",
            "ECE (Calibration)": f"{data.get('ece', 0):.4f}",
            "Latency (ms)": f"{data.get('inference_time_ms', 0):.1f} ms"
        })
        
    df_compare = pd.DataFrame(table_rows)
    st.markdown("### 🏆 Comprehensive Model Comparison Table")
    st.dataframe(df_compare, use_container_width=True)
    
    st.markdown("---")
    
    # Visual Performance Curves
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 📉 Comparative ROC Curves")
        fig_roc = go.Figure()
        colors_list = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
        
        for i, (m_name, data) in enumerate(benchmarks.items()):
            # Plot ROC curve
            sample_data = data.get("sample_predictions", {})
            y_true = np.array(sample_data.get("y_true", []))
            y_prob = np.array(sample_data.get("y_prob", []))
            
            if len(y_true) > 10:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f"{m_name.upper()} (AUC = {data.get('roc_auc', 0.5):.3f})",
                    line=dict(color=colors_list[i % len(colors_list)], width=2)
                ))
                
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance (AUC = 0.50)', line=dict(dash='dash', color='#94A3B8')))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            height=300,
            margin=dict(l=10, r=10, t=25, b=25),
            plot_bgcolor='white',
            legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with c2:
        st.markdown("#### 🎯 Reliability Calibration Curves (ECE)")
        fig_calib = go.Figure()
        
        for i, (m_name, data) in enumerate(benchmarks.items()):
            sample_data = data.get("sample_predictions", {})
            y_true = np.array(sample_data.get("y_true", []))
            y_prob = np.array(sample_data.get("y_prob", []))
            
            if len(y_true) > 10:
                from risk.calibration import ReliabilityCalibrator
                calib_res = ReliabilityCalibrator.compute_calibration_curve(y_true, y_prob, n_bins=8)
                fig_calib.add_trace(go.Scatter(
                    x=calib_res["confidences"], y=calib_res["accuracies"],
                    mode='lines+markers',
                    name=f"{m_name.upper()} (ECE = {data.get('ece', 0.05):.3f})",
                    line=dict(color=colors_list[i % len(colors_list)], width=2)
                ))
                
        fig_calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', line=dict(dash='dash', color='#94A3B8')))
        fig_calib.update_layout(
            xaxis_title="Mean Predicted Confidence",
            yaxis_title="Observed Empirical Accuracy",
            height=300,
            margin=dict(l=10, r=10, t=25, b=25),
            plot_bgcolor='white',
            legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig_calib, use_container_width=True)
        
    # Confusion Matrix Visualization
    st.markdown("---")
    st.markdown("### 🔲 Confusion Matrix Breakdown")
    cm_cols = st.columns(len(benchmarks))
    
    for idx, (m_name, data) in enumerate(benchmarks.items()):
        with cm_cols[idx]:
            cm = np.array(data.get("confusion_matrix", [[0, 0], [0, 0]]))
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=["Normal", "Spina Bifida"],
                y=["Normal", "Spina Bifida"],
                color_continuous_scale="Blues",
                title=f"<b>{m_name.upper()}</b>"
            )
            fig_cm.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
