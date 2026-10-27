"""Batch and folder-level patient cardiovascular disease predictor.

Ingests patient records from files or folders (CSV, Excel, JSON),
runs calibrated multimodal predictions, and generates human-readable
disease classifications, risk stratifications, and top biomarker drivers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.models.clinical_lab_fusion import (
    ClinicalLabMultimodalSystem,
    ALL_FEATURES,
    CLINICAL_FEATURES,
    LAB_FEATURES,
    FEATURE_METADATA,
)

logger = logging.getLogger(__name__)

# Common column name aliases for flexible ingestion
COLUMN_ALIASES: Dict[str, str] = {
    # Clinical
    "age": "age",
    "patient_age": "age",
    "sex": "sex",
    "gender": "sex",
    "biological_sex": "sex",
    "cp": "cp",
    "chest_pain": "cp",
    "chestpain": "cp",
    "chest_pain_type": "cp",
    "trestbps": "trestbps",
    "blood_pressure": "trestbps",
    "resting_bp": "trestbps",
    "bp": "trestbps",
    "exang": "exang",
    "angina": "exang",
    "exercise_angina": "exang",
    # Lab
    "chol": "chol",
    "cholesterol": "chol",
    "serum_chol": "chol",
    "fbs": "fbs",
    "blood_sugar": "fbs",
    "fasting_blood_sugar": "fbs",
    "restecg": "restecg",
    "ecg": "restecg",
    "resting_ecg": "restecg",
    "thalach": "thalach",
    "max_hr": "thalach",
    "heart_rate": "thalach",
    "max_heart_rate": "thalach",
    "oldpeak": "oldpeak",
    "st_depression": "oldpeak",
    "slope": "slope",
    "st_slope": "slope",
    "ca": "ca",
    "vessels": "ca",
    "fluoroscopy_vessels": "ca",
    "major_vessels": "ca",
    "thal": "thal",
    "thallium": "thal",
    "thallium_defect": "thal",
}

# Clinical defaults if missing
DEFAULT_MEDIANS: Dict[str, float] = {
    "age": 55.0,
    "sex": 1.0,
    "cp": 3.0,
    "trestbps": 130.0,
    "exang": 0.0,
    "chol": 240.0,
    "fbs": 0.0,
    "restecg": 0.0,
    "thalach": 150.0,
    "oldpeak": 1.0,
    "slope": 2.0,
    "ca": 0.0,
    "thal": 3.0,
}


def normalize_patient_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Standardize column names and map common aliases.

    Args:
        df: Input DataFrame containing patient records.

    Returns:
        Clean DataFrame aligned with ALL_FEATURES, and list of patient IDs.
    """
    df_clean = df.copy()
    col_map = {}
    for col in df_clean.columns:
        clean_key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if clean_key in COLUMN_ALIASES:
            col_map[col] = COLUMN_ALIASES[clean_key]

    df_clean = df_clean.rename(columns=col_map)

    # Extract or generate patient IDs
    id_cols = [c for c in df.columns if str(c).strip().lower() in ("id", "patient_id", "patientid", "patient", "name")]
    if id_cols:
        patient_ids = df[id_cols[0]].astype(str).tolist()
    else:
        patient_ids = [f"PT-{i+1:03d}" for i in range(len(df))]

    # Fill any missing required feature columns with standard cohort defaults
    for feat in ALL_FEATURES:
        if feat not in df_clean.columns:
            logger.warning(f"Feature '{feat}' missing in input data. Imputing with default {DEFAULT_MEDIANS[feat]}")
            df_clean[feat] = DEFAULT_MEDIANS[feat]
        else:
            # Coerce numeric and fill NaNs
            df_clean[feat] = pd.to_numeric(df_clean[feat], errors="coerce").fillna(DEFAULT_MEDIANS[feat])

    return df_clean, patient_ids


def predict_patient_cohort(
    df: pd.DataFrame,
    system: ClinicalLabMultimodalSystem,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run multimodal cardiovascular prediction for a cohort of patients.

    Args:
        df: Raw DataFrame containing one or more patient records.
        system: Loaded ClinicalLabMultimodalSystem instance.

    Returns:
        summary_df: Formatted table with diagnosis, risk, drivers, and actions.
        patient_details: List of per-patient detailed dictionaries including SHAP.
    """
    df_clean, patient_ids = normalize_patient_df(df)

    rows = []
    patient_details = []

    for i in range(len(df_clean)):
        row = df_clean.iloc[i]
        pid = patient_ids[i]

        patient_dict = {f: float(row[f]) for f in ALL_FEATURES}

        # Predict using system
        res = system.predict_patient(patient_dict, clinical_available=True, lab_available=True)
        prob = float(res.get("risk_probability", 0.0))
        pred = int(res.get("prediction", 0))

        # Disease outcome text
        if prob < 0.35:
            disease_outcome = "Healthy / Low Risk"
            risk_tier = "Low Risk"
            action = "Routine annual cardiovascular prevention"
        elif prob < 0.65:
            disease_outcome = "Possible Early-Stage CAD"
            risk_tier = "Moderate Risk"
            action = "Outpatient cardiology consult & lifestyle intervention"
        elif prob < 0.85:
            disease_outcome = "Coronary Artery Disease (CAD)"
            risk_tier = "High Risk"
            action = "Cardiology evaluation & diagnostic coronary angiography"
        else:
            disease_outcome = "Severe Coronary Artery Disease (CAD)"
            risk_tier = "Critical Risk"
            action = "Urgent cardiology referral & coronary intervention assessment"

        # Extract top drivers
        shap_df = res.get("shap_attributions", pd.DataFrame())
        if not shap_df.empty:
            risk_drivers = shap_df[shap_df["shap_value"] > 0].head(3)
            protective = shap_df[shap_df["shap_value"] < 0].head(3)

            driver_str = ", ".join([f"{r['label']} (+{r['shap_value']:.2f})" for _, r in risk_drivers.iterrows()])
            protective_str = ", ".join([f"{r['label']} ({r['shap_value']:.2f})" for _, r in protective.iterrows()])
        else:
            driver_str = "N/A"
            protective_str = "N/A"

        sex_label = "Male" if row.get("sex", 1) == 1 else "Female"

        summary_row = {
            "Patient_ID": pid,
            "Age": int(row.get("age", 55)),
            "Sex": sex_label,
            "Predicted_Disease": disease_outcome,
            "Risk_Score": f"{prob * 100:.1f}%",
            "Risk_Probability": round(prob, 4),
            "Risk_Level": risk_tier,
            "Top_Risk_Drivers": driver_str if driver_str else "None detected",
            "Top_Protective_Factors": protective_str if protective_str else "None detected",
            "Recommended_Action": action,
        }
        rows.append(summary_row)

        res["patient_id"] = pid
        res["inputs"] = patient_dict
        res["disease_outcome"] = disease_outcome
        res["recommended_action"] = action
        patient_details.append(res)

    summary_df = pd.DataFrame(rows)
    return summary_df, patient_details


def load_patients_from_folder(
    folder_path: Union[str, Path]
) -> Tuple[pd.DataFrame, List[str]]:
    """Scan a folder and combine all CSV, XLSX, and JSON patient files.

    Args:
        folder_path: Path to directory containing patient files.

    Returns:
        Combined DataFrame of patients, and list of loaded filenames.
    """
    path = Path(folder_path)
    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    dfs = []
    loaded_files = []

    # Look for CSVs
    for p in path.glob("*.csv"):
        try:
            d = pd.read_csv(p)
            d["source_file"] = p.name
            dfs.append(d)
            loaded_files.append(p.name)
        except Exception as e:
            logger.warning(f"Failed to read CSV {p.name}: {e}")

    # Look for XLSX
    for p in path.glob("*.xlsx"):
        try:
            d = pd.read_excel(p)
            d["source_file"] = p.name
            dfs.append(d)
            loaded_files.append(p.name)
        except Exception as e:
            logger.warning(f"Failed to read Excel {p.name}: {e}")

    # Look for JSON
    for p in path.glob("*.json"):
        try:
            d = pd.read_json(p)
            d["source_file"] = p.name
            dfs.append(d)
            loaded_files.append(p.name)
        except Exception as e:
            logger.warning(f"Failed to read JSON {p.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid CSV, XLSX, or JSON patient files found in {folder_path}")

    combined = pd.concat(dfs, ignore_index=True)
    return combined, loaded_files
