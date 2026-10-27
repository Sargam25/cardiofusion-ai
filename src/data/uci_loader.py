"""UCI Heart Disease dataset loader.

Loads the Cleveland Heart Disease dataset (Janosi et al., 1989) from the
UCI Machine Learning Repository using the `ucimlrepo` package.

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

This dataset provides clinical and lab-type features for 303 patients.
It does NOT contain raw ECG waveforms — only derived ECG metrics.

Features used:
  Clinical: age, sex, cp (chest pain type), trestbps (resting BP),
            exang (exercise-induced angina)
  Lab-type: chol (serum cholesterol), fbs (fasting blood sugar > 120),
            thal (thallium stress test), ca (fluoroscopy major vessels),
            oldpeak (ST depression), restecg (resting ECG category),
            thalach (max HR achieved), slope (ST slope)
  Target: num > 0 → 1 (CAD present), 0 (absent)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Feature categories for explainability labeling
CLINICAL_COLS = ["age", "sex", "cp", "trestbps", "exang"]
LAB_COLS = ["chol", "fbs", "thal", "ca", "oldpeak", "restecg", "thalach", "slope"]
ALL_FEATURE_COLS = CLINICAL_COLS + LAB_COLS
TARGET_COL = "target"

# Human-readable descriptions for the dashboard
FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)",
    "restecg": "Resting ECG Results (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)",
    "thalach": "Maximum Heart Rate Achieved (bpm)",
    "exang": "Exercise-Induced Angina (1=Yes, 0=No)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment (1=Up, 2=Flat, 3=Down)",
    "ca": "Number of Major Vessels by Fluoroscopy (0-3)",
    "thal": "Thallium Stress Test (3=Normal, 6=Fixed defect, 7=Reversible defect)",
}


class UCIHeartLoader:
    """Loads and preprocesses the UCI Heart Disease (Cleveland) dataset.

    Data is fetched automatically via `ucimlrepo` on first call.
    Results are cached to avoid repeated downloads.
    """

    def __init__(self, cache_dir: str = "data/uci"):
        self.cache_dir = Path(cache_dir)
        self._raw_df: Optional[pd.DataFrame] = None

    def load_raw(self, use_cache: bool = True) -> pd.DataFrame:
        """Load raw UCI Heart Disease data.

        Args:
            use_cache: If True, load from cache if available.

        Returns:
            DataFrame with original feature names and binary target.
        """
        if self._raw_df is not None:
            return self._raw_df

        cache_path = self.cache_dir / "heart_disease_cleveland.csv"
        if use_cache and cache_path.exists():
            logger.info(f"Loading UCI Heart Disease from cache: {cache_path}")
            df = pd.read_csv(cache_path)
            self._raw_df = df
            return df

        logger.info("Fetching UCI Heart Disease dataset via ucimlrepo...")
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        df = X.copy()
        df[TARGET_COL] = (y.values.ravel() > 0).astype(int)
        df.columns = [c.strip().lower() for c in df.columns]

        # Ensure directory exists and cache
        self.download_and_save(str(self.cache_dir), df=df)
        self._raw_df = df
        return df

    def download_and_save(self, save_dir: str, df: Optional[pd.DataFrame] = None) -> Path:
        """Download and save the dataset to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if df is None:
            df = self.load_raw(use_cache=False)
        path = save_dir / "heart_disease_cleveland.csv"
        df.to_csv(path, index=False)
        logger.info(f"UCI Heart Disease saved to {path}")
        return path

    def get_feature_matrix(
        self,
        return_feature_types: bool = False,
    ) -> Any:
        """Return feature matrix X and binary target y."""
        df = self.load_raw()
        available_features = [c for c in ALL_FEATURE_COLS if c in df.columns]
        X = df[available_features].copy()
        y = df[TARGET_COL].copy()

        if return_feature_types:
            feature_types = {}
            for f in available_features:
                if f in CLINICAL_COLS:
                    feature_types[f] = "clinical"
                elif f in LAB_COLS:
                    feature_types[f] = "lab"
                else:
                    feature_types[f] = "unknown"
            return X, y, feature_types

        return X, y

    def get_train_val_test_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Patient-level split (UCI Cleveland has 303 independent patient records)."""
        X, y = self.get_feature_matrix()
        strat = y if stratify else None

        # First split: separate test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )

        # Second split: separate val from remaining
        val_frac = val_size / (1.0 - test_size)
        strat_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac,
            random_state=random_state, stratify=strat_temp
        )

        logger.info(
            f"UCI split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_data_summary(self) -> Dict[str, Any]:
        """Return summary statistics for the UCI dataset."""
        df = self.load_raw()
        available_features = [c for c in ALL_FEATURE_COLS if c in df.columns]
        X = df[available_features]
        y = df[TARGET_COL]
        return {
            "n_records": len(df),
            "n_features": len(available_features),
            "clinical_features": [f for f in CLINICAL_COLS if f in df.columns],
            "lab_features": [f for f in LAB_COLS if f in df.columns],
            "class_distribution": y.value_counts().to_dict(),
            "positive_rate": float(y.mean()),
            "missing_values": X.isnull().sum().to_dict(),
            "missing_pct": (X.isnull().sum() / len(X) * 100).to_dict(),
        }
