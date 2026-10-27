"""CLI utility to predict cardiovascular disease from a folder or file of patient records.

Usage:
    python scripts/predict_from_folder.py --folder data/sample_patients
    python scripts/predict_from_folder.py --file data/sample_patients/patient_cohort_batch.csv --output reports/my_predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.models.clinical_lab_fusion import ClinicalLabMultimodalSystem
from src.models.batch_predictor import load_patients_from_folder, predict_patient_cohort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict_from_folder")


def main():
    parser = argparse.ArgumentParser(description="Predict cardiovascular disease for patient records from folder or file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=str, help="Path to directory containing patient CSV/XLSX/JSON files.")
    group.add_argument("--file", type=str, help="Path to single patient CSV/XLSX/JSON file.")
    parser.add_argument("--model-path", type=str, default="models/clinical_lab_multimodal_system.pkl", help="Path to trained model pickle.")
    parser.add_argument("--output", type=str, default="reports/patient_predictions_report.csv", help="Output CSV path for prediction report.")
    args = parser.parse_args()

    # 1. Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Train it first with python scripts/train_multimodal_clinical_lab.py")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    system = ClinicalLabMultimodalSystem.load(str(model_path))

    # 2. Ingest patients
    if args.folder:
        folder_path = Path(args.folder)
        logger.info(f"Scanning folder: {folder_path}...")
        df_patients, files = load_patients_from_folder(folder_path)
        logger.info(f"Successfully loaded {len(df_patients)} patient records from {len(files)} files: {files}")
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        logger.info(f"Loading patient file: {file_path}...")
        if file_path.suffix.lower() == ".csv":
            df_patients = pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xlsx", ".xls"):
            df_patients = pd.read_excel(file_path)
        elif file_path.suffix.lower() == ".json":
            df_patients = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # 3. Run predictions
    logger.info(f"Running multimodal risk prediction on {len(df_patients)} patients...")
    summary_df, details = predict_patient_cohort(df_patients, system)

    # 4. Save and summarize
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Prediction report saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print("CARDIOVASCULAR DISEASE PREDICTION REPORT (PATIENT COHORT)")
    print("=" * 95)
    display_cols = ["Patient_ID", "Age", "Sex", "Predicted_Disease", "Risk_Score", "Risk_Level", "Recommended_Action"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 95)

    high_risk_count = (summary_df["Risk_Level"].isin(["High Risk", "Critical Risk"])).sum()
    low_risk_count = (summary_df["Risk_Level"] == "Low Risk").sum()
    print(f"\nCohort Overview: {len(summary_df)} Total Patients | {high_risk_count} High/Critical Risk | {low_risk_count} Low Risk\n")


if __name__ == "__main__":
    main()
