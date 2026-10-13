"""
Reusable UI Components, Plots, and Medical Dashboard Widgets.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def apply_custom_css():
    """Injects modern, clean, clinical research CSS styles."""
    st.markdown("""
    <style>
    /* Global styles */
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 18px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 12.5px;
        color: #92400E;
        margin-bottom: 16px;
    }
    .risk-badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
    }
    .risk-badge-moderate {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
    }
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

def render_disclaimer():
    st.markdown("""
    <div class="disclaimer-banner">
        <b>🔬 RESEARCH DECISION-SUPPORT PROTOTYPE:</b> This software is an investigational research prototype for academic ultrasound analysis. 
        It is <b>NOT</b> a certified diagnostic medical device. All outputs, risk scores, and Grad-CAM attention maps require independent review by qualified specialists.
    </div>
    """, unsafe_allow_html=True)

def render_sbrs_gauge(score: float, category: str, color: str):
    """Renders circular gauge for the 0-100 SB-RS score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>SB-RS Score</b><br><span style='font-size:14px;color:{color};font-weight:bold;'>{category} RISK</span>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 30], 'color': '#ECFDF5'},
                {'range': [30, 60], 'color': '#FFFBEB'},
                {'range': [60, 100], 'color': '#FEF2F2'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        font={'family': 'Helvetica, Arial, sans-serif'}
    )
    return fig

def render_risk_factor_bar(factors: List[Dict[str, Any]]):
    """Renders horizontal bar chart of risk factor contributions."""
    df = pd.DataFrame(factors)
    # Sort for bottom-up display
    df = df.sort_values(by="contribution", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["contribution"],
        y=df["factor"],
        orientation='h',
        marker=dict(
            color=df["contribution"],
            colorscale='Tealgrn',
            line=dict(color='#0F766E', width=1)
        ),
        text=[f"+{c:.1f} pts" for c in df["contribution"]],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"<b>{row['factor']}</b><br>Observed: {row['value']}<br>Weight: {row['weight']}<br>Impact: +{row['contribution']:.1f} points" for _, row in df.iterrows()]
    ))
    fig.update_layout(
        title="<b>Risk Factor Contribution Attribution (Why this Risk?)</b>",
        xaxis_title="Points Contributed to SB-RS (0–100)",
        yaxis_title="",
        height=240,
        margin=dict(l=10, r=20, t=35, b=25),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', range=[0, max(40, max(df['contribution']) + 5)])
    )
    return fig

def render_decision_matrix_plot(current_point: Dict[str, Any]):
    """Renders 2D Risk x Confidence Decision Support Quadrant."""
    fig = go.Figure()
    
    # Background zones
    # High Risk Zone
    fig.add_shape(type="rect", x0=0, y0=60, x1=0.5, y1=100, fillcolor="#FEE2E2", opacity=0.4, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=60, x1=1.0, y1=100, fillcolor="#FEF2F2", opacity=0.7, layer="below", line_width=0)
    # Mod Risk Zone
    fig.add_shape(type="rect", x0=0, y0=30, x1=1.0, y1=60, fillcolor="#FEF3C7", opacity=0.4, layer="below", line_width=0)
    # Low Risk Zone
    fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=30, fillcolor="#E0E7FF", opacity=0.4, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=0, x1=1.0, y1=30, fillcolor="#ECFDF5", opacity=0.6, layer="below", line_width=0)
    
    # Current case point
    curr_x = current_point.get("x_confidence", 0.8)
    curr_y = current_point.get("y_risk", 75.0)
    curr_color = current_point.get("color", "#EF4444")
    curr_label = current_point.get("quadrant", "Current Case")
    
    fig.add_trace(go.Scatter(
        x=[curr_x],
        y=[curr_y],
        mode='markers+text',
        marker=dict(size=16, color=curr_color, symbol='diamond', line=dict(color='black', width=2)),
        text=["<b>CURRENT CASE</b>"],
        textposition="top center",
        name="Evaluated Scan"
    ))
    
    # Add quadrant labels
    fig.add_annotation(x=0.85, y=85, text="<b>High Risk<br>High Confidence</b>", showarrow=False, font=dict(color="#B91C1C", size=10))
    fig.add_annotation(x=0.20, y=85, text="<b>High Risk<br>High Uncertainty ⚠️</b>", showarrow=False, font=dict(color="#C2410C", size=10))
    fig.add_annotation(x=0.50, y=45, text="<b>Moderate Risk (Equivocal)</b>", showarrow=False, font=dict(color="#B45309", size=10))
    fig.add_annotation(x=0.85, y=15, text="<b>Low Risk<br>High Confidence ✅</b>", showarrow=False, font=dict(color="#15803D", size=10))
    fig.add_annotation(x=0.20, y=15, text="<b>Low Risk<br>Uncertain 🔵</b>", showarrow=False, font=dict(color="#1D4ED8", size=10))
    
    fig.update_layout(
        title="<b>Risk × Confidence Decision Support Matrix</b>",
        xaxis=dict(title="Model Certainty (1 - Epistemic Uncertainty)", range=[-0.05, 1.05], tickvals=[0.1, 0.5, 0.9], ticktext=["Uncertain", "Moderate", "Confident"]),
        yaxis=dict(title="SB-RS Risk Score (0-100)", range=[-2, 105]),
        height=260,
        margin=dict(l=10, r=20, t=35, b=25),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig
