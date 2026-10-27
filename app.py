"""CardioFusion-AI: Cardiovascular Disease Prediction Dashboard.

Subtle, modern clinical prediction interface for patient records.
Generates disease classification (Coronary Artery Disease vs Healthy),
risk probability, key biomarker drivers, and recommended clinical actions.

RESEARCH PROTOTYPE — NOT AN ACCREDITED MEDICAL DEVICE.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.clinical_lab_fusion import (
    ClinicalLabMultimodalSystem,
    CLINICAL_FEATURES,
    LAB_FEATURES,
    ALL_FEATURES,
    FEATURE_METADATA,
)
from src.models.batch_predictor import (
    load_patients_from_folder,
    predict_patient_cohort,
    normalize_patient_df,
)

logging.basicConfig(level=logging.WARNING)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CardioFusion-AI | Cardiovascular Disease Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SUBTLE, CONTEMPORARY CLINICAL STYLING
# - Modern Minimalist Dark Theme
# - Zero tacky emojis or decorative clutter
# - MANDATORY: ALL SELECTBOX/RADIO/DROPDOWN OPTIONS IN SOLID BLACK (#000000)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-dark: #0B0F19;
    --bg-card: #111827;
    --border-color: #1F2937;
    --primary: #4F46E5;
    --primary-light: #6366F1;
    --text-main: #F3F4F6;
    --text-muted: #9CA3AF;
    --danger: #EF4444;
    --warning: #F59E0B;
    --success: #10B981;
}

.stApp {
    background: #0B0F19 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

body, .stMarkdown, p, span, label, div {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main);
}

/* ============================================================
   OPTIONS STYLING: MANDATORY BLACK COLOR WORDS ON LIGHT PILLS
   ============================================================ */

/* Selectbox dropdowns and containers */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] * {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Popover dropdown menus */
div[data-baseweb="popover"], div[data-baseweb="popover"] * {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}
ul[data-baseweb="menu"], ul[data-baseweb="menu"] * {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Selectable dropdown options */
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 700 !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid #F3F4F6 !important;
}
li[role="option"]:hover {
    background-color: #F3F4F6 !important;
    color: #000000 !important;
}
li[role="option"] * {
    color: #000000 !important;
    font-weight: 700 !important;
}
div[role="listbox"] * {
    color: #000000 !important;
    background-color: #FFFFFF !important;
}

/* Radio buttons options */
div[role="radiogroup"] {
    gap: 6px !important;
}
div[role="radiogroup"] label {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    cursor: pointer !important;
    margin: 2px 4px 2px 0px !important;
}
div[role="radiogroup"] label:hover {
    border-color: #4F46E5 !important;
    background: #F9FAFB !important;
}
div[role="radiogroup"] label * {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}

/* Text Inputs and File Uploader */
.stTextInput input {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    border: 1px solid #D1D5DB !important;
}

[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1px dashed #374151 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
}

/* Cards & containers */
.cf-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 14px;
}

.cf-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #F9FAFB;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.cf-subtitle {
    font-size: 0.92rem;
    color: #9CA3AF;
    margin-bottom: 22px;
}

/* Subtle badges */
.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

/* Buttons */
.stButton > button {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 8px 18px !important;
    transition: background 0.15s ease-in-out !important;
}
.stButton > button:hover {
    background: #4338CA !important;
}

/* Download Button (Image 1 fix: Black text on white button) */
div.stDownloadButton > button,
[data-testid="stDownloadButton"] button {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}
div.stDownloadButton > button,
div.stDownloadButton > button *,
[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] button * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}
div.stDownloadButton > button:hover {
    background-color: #F3F4F6 !important;
    border-color: #4F46E5 !important;
}
div.stDownloadButton > button:hover * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Dataframe & Element Toolbar (Icons clearly outlined, NOT solid squares) */
[data-testid="stElementToolbar"],
div[data-testid="stElementToolbar"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
    padding: 2px 4px !important;
}
[data-testid="stElementToolbar"] button,
div[data-testid="stElementToolbar"] button,
[data-testid="stElementToolbarButton"] button {
    background-color: transparent !important;
    color: #111827 !important;
    border: none !important;
    padding: 4px 6px !important;
    border-radius: 4px !important;
}
[data-testid="stElementToolbar"] button:hover,
div[data-testid="stElementToolbar"] button:hover,
[data-testid="stElementToolbarButton"] button:hover {
    background-color: #EEF2FF !important;
}

/* Ensure SVG icons render with crisp black lines, NOT filled black blocks */
[data-testid="stElementToolbar"] svg,
[data-testid="stElementToolbarButton"] svg,
[data-testid="stElementToolbar"] button svg,
[data-testid="stElementToolbarButton"] button svg {
    color: #111827 !important;
}

/* Search input inside dataframe toolbar */
[data-testid="stElementToolbar"] input {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}
[data-testid="stElementToolbar"] span,
[data-testid="stElementToolbar"] p {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
}

/* Tooltips & Context Popovers (Makes "Fullscreen", "Download as CSV", "Search" 100% visible) */
div[data-baseweb="tooltip"],
div[data-baseweb="tooltip"] > div,
div[role="tooltip"],
div[role="tooltip"] > div,
[data-testid="stTooltipContent"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18) !important;
}

div[data-baseweb="tooltip"] *,
div[role="tooltip"] *,
[data-testid="stTooltipContent"] *,
div[data-baseweb="tooltip"] p,
div[data-baseweb="tooltip"] span,
div[role="tooltip"] p,
div[role="tooltip"] span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL SYSTEM
# ============================================================
@st.cache_resource
def load_system() -> Optional[ClinicalLabMultimodalSystem]:
    model_path = Path("models/clinical_lab_multimodal_system.pkl")
    if model_path.exists():
        return ClinicalLabMultimodalSystem.load(str(model_path))
    return None

system = load_system()

if system is None:
    st.error("Model file not found at `models/clinical_lab_multimodal_system.pkl`. Please run `python scripts/train_multimodal_clinical_lab.py`.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="cf-title">CardioFusion-AI | Cardiovascular Disease Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cf-subtitle">Calibrated multimodal risk estimation and biomarker attribution from patient clinical and laboratory records.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# PATIENT DATA INGESTION
# ============================================================
st.markdown(
    """
    <div style="font-size:1.1rem; font-weight:700; color:#F3F4F6; margin-bottom:4px;">
        Patient Records Ingestion
    </div>
    <div style="font-size:0.88rem; color:#9CA3AF; margin-bottom:14px;">
        Upload patient files (CSV, XLSX, JSON) or load a reference cohort to generate disease predictions and biomarker evaluations.
    </div>
    """,
    unsafe_allow_html=True,
)

c_up1, c_up2 = st.columns([3, 1])

with c_up1:
    uploaded_files = st.file_uploader(
        "Upload Patient Records File:",
        type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with c_up2:
    load_demo_btn = st.button("Load Sample Patient Records", use_container_width=True)

df_loaded: Optional[pd.DataFrame] = None
source_description = ""

# Handle Sample Patient Records button
if load_demo_btn:
    demo_path = Path("data/sample_patients/patient_cohort_batch.csv")
    if demo_path.exists():
        df_loaded = pd.read_csv(demo_path)
        st.session_state["active_patient_df"] = df_loaded
        st.session_state["active_source_desc"] = "Sample Patient Cohort (15 records)"
    else:
        st.warning("Sample records file not found at data/sample_patients/patient_cohort_batch.csv")

# Handle uploaded files
if uploaded_files:
    file_dfs = []
    file_names = []
    for up in uploaded_files:
        try:
            if up.name.endswith(".csv"):
                d = pd.read_csv(up)
            elif up.name.endswith(".xlsx") or up.name.endswith(".xls"):
                d = pd.read_excel(up)
            elif up.name.endswith(".json"):
                d = pd.read_json(up)
            d["source_file"] = up.name
            file_dfs.append(d)
            file_names.append(up.name)
        except Exception as e:
            st.error(f"Failed to read file {up.name}: {e}")

    if file_dfs:
        df_loaded = pd.concat(file_dfs, ignore_index=True)
        st.session_state["active_patient_df"] = df_loaded
        st.session_state["active_source_desc"] = f"Uploaded file(s): {', '.join(file_names)}"

# Fallback to cached session data or initial load
if df_loaded is None and "active_patient_df" in st.session_state:
    df_loaded = st.session_state["active_patient_df"]
    source_description = st.session_state.get("active_source_desc", "")
elif df_loaded is None:
    # Auto-load demo batch initially so user immediately sees real predictions
    demo_path = Path("data/sample_patients/patient_cohort_batch.csv")
    if demo_path.exists():
        df_loaded = pd.read_csv(demo_path)
        st.session_state["active_patient_df"] = df_loaded
        st.session_state["active_source_desc"] = "Sample Patient Cohort (15 records)"
        source_description = st.session_state["active_source_desc"]

if "active_source_desc" in st.session_state and not source_description:
    source_description = st.session_state["active_source_desc"]

# ============================================================
# COHORT PREDICTIONS DISPLAY
# ============================================================
if df_loaded is not None and not df_loaded.empty:
    st.markdown(
        f'<div style="font-size:0.85rem; color:#9CA3AF; margin-top:8px; margin-bottom:16px;">Active data: <strong>{source_description}</strong> ({len(df_loaded)} patient records)</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Computing multimodal risk predictions..."):
        summary_df, patient_details = predict_patient_cohort(df_loaded, system)

    # Overview Metrics
    total_p = len(summary_df)
    cad_cases = int((summary_df["Risk_Level"].isin(["High Risk", "Critical Risk"])).sum())
    mod_cases = int((summary_df["Risk_Level"] == "Moderate Risk").sum())
    healthy_cases = int((summary_df["Risk_Level"] == "Low Risk").sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Ingested Patients", f"{total_p}")
    with m2:
        st.metric("CAD Cases (High / Critical)", f"{cad_cases}", f"{cad_cases/total_p*100:.1f}% positive", delta_color="inverse")
    with m3:
        st.metric("Moderate Risk (Borderline)", f"{mod_cases}", "Outpatient evaluation")
    with m4:
        st.metric("Healthy / Low Risk", f"{healthy_cases}", "Negative screening")

    st.markdown("---")

    # Risk Filter
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        risk_filter = st.selectbox(
            "Filter Patients by Risk Category:",
            ["All Patients", "High & Critical Risk Only", "Moderate Risk Only", "Low Risk Only"],
            index=0,
        )

    df_display = summary_df.copy()
    if risk_filter == "High & Critical Risk Only":
        df_display = df_display[df_display["Risk_Level"].isin(["High Risk", "Critical Risk"])]
    elif risk_filter == "Moderate Risk Only":
        df_display = df_display[df_display["Risk_Level"] == "Moderate Risk"]
    elif risk_filter == "Low Risk Only":
        df_display = df_display[df_display["Risk_Level"] == "Low Risk"]

    # Formatted cohort table
    st.markdown(
        f'<div style="font-size:1.05rem; font-weight:700; color:#F3F4F6; margin-bottom:8px;">Patient Predictions Cohort ({len(df_display)} patients displayed)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        df_display[[
            "Patient_ID", "Age", "Sex", "Predicted_Disease", "Risk_Score",
            "Risk_Level", "Top_Risk_Drivers", "Recommended_Action"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # Download CSV button
    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Prediction Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="cardiofusion_patient_predictions.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ============================================================
    # INDIVIDUAL PATIENT DRILL-DOWN INSPECTION
    # ============================================================
    st.markdown(
        """
        <div style="font-size:1.1rem; font-weight:700; color:#F3F4F6; margin-bottom:4px;">
            Individual Patient Biomarker Analysis
        </div>
        <div style="font-size:0.88rem; color:#9CA3AF; margin-bottom:12px;">
            Select a patient identifier to view the individual disease breakdown and underlying biomarker contributions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_pid = st.selectbox(
        "Select Patient ID:",
        summary_df["Patient_ID"].tolist(),
        index=0,
    )

    p_idx = summary_df["Patient_ID"].tolist().index(selected_pid)
    p_res = patient_details[p_idx]
    p_row = summary_df.iloc[p_idx]

    c_det1, c_det2, c_det3 = st.columns([1.1, 1.4, 1.5])
    with c_det1:
        st.markdown(
            f"""
            <div class="cf-card" style="text-align:center;">
                <div style="font-size:0.8rem; color:#9CA3AF; text-transform:uppercase; font-weight:600;">Disease Risk Score</div>
                <div style="font-size:2.8rem; font-weight:800; color:{p_res['risk_color']}; margin:4px 0;">{p_res['risk_percent']}</div>
                <div class="risk-badge" style="background:{p_res['risk_color']}22; color:{p_res['risk_color']}; border:1px solid {p_res['risk_color']};">
                    {p_res['risk_level']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_det2:
        st.markdown(
            f"""
            <div class="cf-card">
                <div style="font-size:0.8rem; color:#9CA3AF; text-transform:uppercase; font-weight:600;">Predicted Disease Status</div>
                <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF; margin:6px 0 4px 0;">{p_row['Predicted_Disease']}</div>
                <div style="font-size:0.82rem; color:#9CA3AF;">Patient Identifier: <strong>{selected_pid}</strong> (Age {p_row['Age']}, {p_row['Sex']})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_det3:
        st.markdown(
            f"""
            <div class="cf-card">
                <div style="font-size:0.8rem; color:#9CA3AF; text-transform:uppercase; font-weight:600;">Recommended Clinical Action</div>
                <div style="font-size:0.9rem; color:#E5E7EB; margin-top:6px; line-height:1.5;">{p_row['Recommended_Action']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly horizontal bar of factors for this patient
    shap_df_p = p_res.get("shap_attributions", pd.DataFrame())
    if not shap_df_p.empty:
        p_top = shap_df_p.head(8).copy()
        # Clean labels of any special dash characters
        p_top["label"] = p_top["label"].astype(str).str.replace("–", "-")
        p_top["Impact Direction"] = p_top["shap_value"].apply(lambda v: "Increases Risk" if v > 0 else "Lowers Risk (Protective)")

        fig_p = px.bar(
            p_top,
            x="shap_value",
            y="label",
            color="Impact Direction",
            orientation="h",
            color_discrete_map={"Increases Risk": "#EF4444", "Lowers Risk (Protective)": "#10B981"},
            labels={"shap_value": "Predictive Weight", "label": "Clinical / Laboratory Biomarker"},
            title=f"Key Biomarkers Influencing Risk for Patient {selected_pid}",
            template="plotly_dark",
        )
        fig_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.7)",
            yaxis={"autorange": "reversed"},
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_p, use_container_width=True)

# ============================================================
# MODEL BENCHMARK & ACCURACY
# ============================================================
st.markdown("---")
with st.expander("View Model Accuracy & Benchmark Validation"):
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test Accuracy", "86.9%", "Independent Test Set")
    with m_col2:
        st.metric("AUROC Score", "0.936", "[0.870, 0.984] 95% CI")
    with m_col3:
        st.metric("Sensitivity (Recall)", "96.4%", "True Positive Rate")
    with m_col4:
        st.metric("Brier Calibration", "0.096", "Well-calibrated probabilities")

    st.markdown(
        """
        <div style="font-size:0.82rem; color:#9CA3AF; margin-top:10px; line-height:1.5;">
            Evaluated on held-out test splits using calibrated multimodal gradient boosted ensembles (XGBoost).
            Class probabilities are calibrated using isotonic regression on validation folds to ensure reliability.
        </div>
        """,
        unsafe_allow_html=True,
    )
