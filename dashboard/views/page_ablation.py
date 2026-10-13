"""
Research Experiments & Ablation Study View (Page 5).
Presents the 6-stage formal ablation matrix demonstrating the incremental value of each novelty component.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config.settings import ARTIFACTS_DIR
from evaluation.ablation_study import run_ablation_study
from dashboard.components import render_disclaimer

def render_page_ablation():
    st.markdown('<div class="main-header">🧪 Research Experiments & Ablation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical Validation of Individual Novelty Components on Decision-Support Utility</div>', unsafe_allow_html=True)
    
    render_disclaimer()
    
    ablation_file = ARTIFACTS_DIR / "ablation_results.json"
    if not ablation_file.exists():
        data = run_ablation_study()
    else:
        with open(ablation_file, "r") as f:
            data = json.load(f)
            
    experiments = data.get("ablation_study", [])
    
    table_data = []
    for exp in experiments:
        m = exp.get("metrics", {})
        table_data.append({
            "Experiment": exp.get("experiment_id"),
            "System Configuration": exp.get("configuration"),
            "Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
            "Sensitivity": f"{m.get('sensitivity', 0)*100:.1f}%",
            "Specificity": f"{m.get('specificity', 0)*100:.1f}%",
            "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
            "ECE": f"{m.get('ece', 0):.4f}",
            "Explainability Score": f"{m.get('clinical_interpretability_score', 0):.0f} / 100",
            "Uncertainty Flag Rate": f"{m.get('uncertainty_flagging_rate', 0):.1f}%"
        })
        
    df_ablation = pd.DataFrame(table_data)
    st.markdown("### 📊 Formal 6-Stage Ablation Matrix")
    st.dataframe(df_ablation, use_container_width=True)
    
    st.markdown("---")
    
    # Interpretability and Calibration Progression Bar Chart
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🧠 Clinical Interpretability Score Progression")
        fig_interp = px.bar(
            df_ablation, x="Experiment", y="Explainability Score",
            color="Experiment", color_discrete_sequence=px.colors.qualitative.Prism,
            title="<b>Interpretability & Explainability Gain across Ablation Stages</b>"
        )
        fig_interp.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=25), plot_bgcolor='white')
        st.plotly_chart(fig_interp, use_container_width=True)
        
    with c2:
        st.markdown("#### 📉 Calibration Error (ECE) Reduction")
        eces = [exp["metrics"]["ece"] for exp in experiments]
        fig_ece = go.Figure(go.Scatter(
            x=[exp["experiment_id"] for exp in experiments],
            y=eces,
            mode='lines+markers+text',
            text=[f"{e:.4f}" for e in eces],
            textposition="top center",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=10, color="#B91C1C")
        ))
        fig_ece.update_layout(
            title="<b>Expected Calibration Error (Lower is Better)</b>",
            xaxis_title="Experiment Configuration",
            yaxis_title="ECE Error",
            height=280,
            margin=dict(l=10, r=10, t=35, b=25),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_ece, use_container_width=True)
        
    # Detailed Descriptions of Experiments
    st.markdown("---")
    st.markdown("### 🔬 Research Experiment Details & Hypotheses")
    for exp in experiments:
        with st.expander(f"**{exp['experiment_id']}: {exp['configuration']}**"):
            st.markdown(f"**Components Active:** `{', '.join([k for k, v in exp['components'].items() if v])}`")
            st.markdown(f"**Key Findings & Research Notes:** {exp['research_notes']}")
