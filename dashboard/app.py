"""
Main Streamlit Application Entrypoint for Explainable & Uncertainty-Aware Spina Bifida AI System.
"""

import streamlit as st

# Must be very first Streamlit command
st.set_page_config(
    page_title="Spina Bifida AI Risk Stratification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.components import apply_custom_css
from dashboard.views.page_analyzer import render_page_analyzer
from dashboard.views.page_cohort import render_page_cohort
from dashboard.views.page_benchmarks import render_page_benchmarks
from dashboard.views.page_longitudinal import render_page_longitudinal
from dashboard.views.page_ablation import render_page_ablation

def main():
    apply_custom_css()
    
    st.sidebar.image("https://img.icons8.com/fluency/96/ultrasound.png", width=64)
    st.sidebar.markdown("## 🧠 SPINA BIFIDA AI")
    st.sidebar.markdown("**Decision-Support Platform**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "🔬 Live Clinical Risk Analyzer",
            "📊 Dataset & Cohort Analytics",
            "📈 Multi-Model Benchmarks & Calibration",
            "📅 Longitudinal Risk Tracker",
            "🧪 Research Experiments & Ablation Study"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:11px;color:#94A3B8;'>"
        "<b>Project Title:</b><br/>"
        "Explainable and Uncertainty-Aware AI-Based Spina Bifida Detection and Dynamic Risk Stratification from Fetal Ultrasound Images<br/><br/>"
        "<b>Status:</b> Research Prototype<br/>"
        "<b>Environment:</b> PyTorch 2.x / Torchvision"
        "</div>",
        unsafe_allow_html=True
    )
    
    if page == "🔬 Live Clinical Risk Analyzer":
        render_page_analyzer()
    elif page == "📊 Dataset & Cohort Analytics":
        render_page_cohort()
    elif page == "📈 Multi-Model Benchmarks & Calibration":
        render_page_benchmarks()
    elif page == "📅 Longitudinal Risk Tracker":
        render_page_longitudinal()
    elif page == "🧪 Research Experiments & Ablation Study":
        render_page_ablation()

if __name__ == "__main__":
    main()
