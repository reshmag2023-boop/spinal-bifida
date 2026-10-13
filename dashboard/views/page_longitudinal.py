"""
Longitudinal Gestational Risk Progression View (Page 4).
Tracks multi-scan sequences across gestational timepoints (T1, T2, T3)
and calculates risk velocity (delta SB-RS / delta t).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from risk.longitudinal import LongitudinalRiskTracker
from dashboard.components import render_disclaimer

def render_page_longitudinal():
    st.markdown('<div class="main-header">📅 Dynamic Longitudinal Risk Progression</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Serial Gestational Ultrasound Tracking and Risk Velocity Trajectory Analysis</div>', unsafe_allow_html=True)
    
    render_disclaimer()
    
    # Pre-configured research case histories
    case_options = {
        "CASE-L01: Progressive Morphological Risk (16w -> 28w)": [
            {"timepoint": "T1 (16w)", "gestational_week": 16, "sb_rs_score": 38.0, "category": "MODERATE", "cranial_indentation": 0.28, "uncertainty": 0.045},
            {"timepoint": "T2 (20w)", "gestational_week": 20, "sb_rs_score": 52.5, "category": "MODERATE", "cranial_indentation": 0.44, "uncertainty": 0.032},
            {"timepoint": "T3 (24w)", "gestational_week": 24, "sb_rs_score": 68.0, "category": "HIGH", "cranial_indentation": 0.65, "uncertainty": 0.018},
            {"timepoint": "T4 (28w)", "gestational_week": 28, "sb_rs_score": 79.5, "category": "HIGH", "cranial_indentation": 0.82, "uncertainty": 0.012}
        ],
        "CASE-L02: Stable Normal Morphometry (18w -> 26w)": [
            {"timepoint": "T1 (18w)", "gestational_week": 18, "sb_rs_score": 14.0, "category": "LOW", "cranial_indentation": 0.12, "uncertainty": 0.015},
            {"timepoint": "T2 (22w)", "gestational_week": 22, "sb_rs_score": 16.5, "category": "LOW", "cranial_indentation": 0.14, "uncertainty": 0.014},
            {"timepoint": "T3 (26w)", "gestational_week": 26, "sb_rs_score": 15.0, "category": "LOW", "cranial_indentation": 0.11, "uncertainty": 0.011}
        ],
        "CASE-L03: Borderline Equivocal Trajectory (17w -> 25w)": [
            {"timepoint": "T1 (17w)", "gestational_week": 17, "sb_rs_score": 45.0, "category": "MODERATE", "cranial_indentation": 0.35, "uncertainty": 0.058},
            {"timepoint": "T2 (21w)", "gestational_week": 21, "sb_rs_score": 42.0, "category": "MODERATE", "cranial_indentation": 0.32, "uncertainty": 0.048},
            {"timepoint": "T3 (25w)", "gestational_week": 25, "sb_rs_score": 44.5, "category": "MODERATE", "cranial_indentation": 0.34, "uncertainty": 0.040}
        ]
    }
    
    selected_case = st.selectbox("Select Longitudinal Patient Case History:", list(case_options.keys()))
    scans = case_options[selected_case]
    
    tracker = LongitudinalRiskTracker()
    trajectory_data = tracker.analyze_trajectory(scans)
    
    # Trajectory Metric Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trajectory Classification", trajectory_data["trajectory"])
    c2.metric("Risk Velocity (ΔSB-RS / week)", f"{trajectory_data['risk_velocity_per_week']:+0.2f} pts/wk")
    c3.metric("Initial Score (T1)", f"{trajectory_data['initial_score']:.1f}")
    c4.metric("Latest Score (Latest)", f"{trajectory_data['latest_score']:.1f}", f"{trajectory_data['score_delta']:+0.1f} pts")
    
    st.markdown("---")
    
    # Longitudinal Progression Plot
    fig_long = go.Figure()
    
    # Risk zone bands
    fig_long.add_hrect(y0=60, y1=100, fillcolor="#FEE2E2", opacity=0.4, line_width=0, annotation_text="HIGH RISK ZONE (61-100)", annotation_position="top left")
    fig_long.add_hrect(y0=30, y1=60, fillcolor="#FEF3C7", opacity=0.4, line_width=0, annotation_text="MODERATE RISK ZONE (31-60)", annotation_position="top left")
    fig_long.add_hrect(y0=0, y1=30, fillcolor="#ECFDF5", opacity=0.4, line_width=0, annotation_text="LOW RISK ZONE (0-30)", annotation_position="top left")
    
    fig_long.add_trace(go.Scatter(
        x=trajectory_data["weeks"],
        y=trajectory_data["scores"],
        mode='lines+markers+text',
        name='SB-RS Progression',
        line=dict(color=trajectory_data["trend_color"], width=3),
        marker=dict(size=12, color=trajectory_data["trend_color"]),
        text=[f"{s:.1f} ({c['category']})" for s, c in zip(trajectory_data["scores"], scans)],
        textposition="top center"
    ))
    
    fig_long.update_layout(
        title="<b>Longitudinal SB-RS Risk Trajectory Across Gestational Weeks</b>",
        xaxis=dict(title="Gestational Age (Weeks)", tickvals=trajectory_data["weeks"]),
        yaxis=dict(title="SB-RS Risk Score (0–100)", range=[0, 105]),
        height=320,
        margin=dict(l=10, r=20, t=35, b=25),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig_long, use_container_width=True)
    
    # Table of Scans
    st.markdown("### 📋 Serial Ultrasound Scan Records")
    df_scans = pd.DataFrame(scans)
    st.dataframe(df_scans, use_container_width=True)
    st.info(f"💡 **Clinical Trajectory Recommendation:** {trajectory_data['recommendation']}")
