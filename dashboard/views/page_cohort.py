"""
Dataset & Cohort Analytics View (Page 2).
Examines dataset distributions (1,999 HC18 Normal vs 1,200 Lemon Sign SB),
image dimensions, quality metrics, and interactive Lemonizer simulator.
"""

import streamlit as st
import glob
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import NORMAL_DIR, SB_DIR, TEST_IMAGES_DIR
from preprocessing.lemonizer import LemonSignGenerator
from dashboard.components import render_disclaimer

def render_page_cohort():
    st.markdown('<div class="main-header">📊 Dataset & Cohort Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fetal Head Ultrasound Cohort Characteristics and Morphological Deformation Inspector</div>', unsafe_allow_html=True)
    
    render_disclaimer()
    
    normal_files = glob.glob(str(NORMAL_DIR / "*.*"))
    sb_files = glob.glob(str(SB_DIR / "*.*"))
    test_files = glob.glob(str(TEST_IMAGES_DIR / "*.*"))
    
    total_samples = len(normal_files) + len(sb_files)
    
    # Cohort Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans in Cohort", f"{total_samples:,}")
    c2.metric("Normal HC18 Scans", f"{len(normal_files):,}", f"{len(normal_files)/max(1, total_samples)*100:.1f}%")
    c3.metric("Spina Bifida Scans", f"{len(sb_files):,}", f"{len(sb_files)/max(1, total_samples)*100:.1f}%")
    c4.metric("Image Resolution", "224 × 224 × 3", "Rescaled")
    
    st.markdown("---")
    
    # Charts Row
    ch1, ch2 = st.columns(2)
    with ch1:
        dist_df = pd.DataFrame({
            "Class": ["Normal Fetal Anatomy", "Spina Bifida (Cranial Lemon Sign)"],
            "Count": [len(normal_files), len(sb_files)],
            "Color": ["#10B981", "#EF4444"]
        })
        fig_pie = px.pie(
            dist_df, names="Class", values="Count", title="<b>Cohort Class Distribution</b>",
            color="Class", color_discrete_map={"Normal Fetal Anatomy": "#10B981", "Spina Bifida (Cranial Lemon Sign)": "#EF4444"},
            hole=0.4
        )
        fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with ch2:
        split_df = pd.DataFrame({
            "Split": ["Training Set (80%)", "Validation / Test Set (20%)"],
            "Scans": [int(total_samples * 0.8), int(total_samples * 0.2)]
        })
        fig_bar = px.bar(
            split_df, x="Split", y="Scans", title="<b>Stratified Dataset Partitioning</b>",
            color="Split", color_discrete_sequence=["#3B82F6", "#8B5CF6"]
        )
        fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10), plot_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # Interactive Lemonizer Inspector
    st.markdown("---")
    st.markdown("### 🍋 Cranial Lemon Sign Morphological Deformation Simulator")
    st.markdown("""
    *The Cranial Lemon Sign is an inward bilateral scalloping of the frontal bones caused by downward traction from open neural tube defects (Chiari II malformation).*
    *Test how the physics-based mathematical warp engine simulates this hallmark sign on any baseline normal scan.*
    """)
    
    lemonizer = LemonSignGenerator()
    
    if normal_files:
        selected_normal = st.selectbox(
            "Select Baseline Normal Scan to Deform:",
            normal_files[:10],
            format_func=lambda x: os.path.basename(x)
        )
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            pinch_val = st.slider("Pinch Strength Factor (α)", min_value=0.0005, max_value=0.0050, value=0.0025, step=0.0005, format="%.4f")
        with col_p2:
            add_noise = st.checkbox("Add Ultrasound Speckle Noise", value=True)
        with col_p3:
            add_blur = st.checkbox("Add Acoustic Motion Blur", value=True)
            
        img_norm = cv2.imread(selected_normal)
        if img_norm is not None:
            img_norm_rgb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
            img_norm_resized = cv2.resize(img_norm_rgb, (224, 224))
            
            deformed_bgr, _ = lemonizer.apply_lemon_deformation(
                cv2.cvtColor(img_norm_resized, cv2.COLOR_RGB2BGR),
                pinch_strength=pinch_val,
                add_noise=add_noise,
                add_blur=add_blur
            )
            deformed_rgb = cv2.cvtColor(deformed_bgr, cv2.COLOR_BGR2RGB)
            
            diff_map = np.abs(deformed_rgb.astype(int) - img_norm_resized.astype(int)).astype(np.uint8)
            
            v1, v2, v3 = st.columns(3)
            with v1:
                st.image(img_norm_resized, caption="1. Baseline Normal Ultrasound", use_container_width=True)
            with v2:
                st.image(deformed_rgb, caption=f"2. Simulated Lemon Sign (α={pinch_val})", use_container_width=True)
            with v3:
                st.image(diff_map, caption="3. Morphological Deformation Delta Map", use_container_width=True)
