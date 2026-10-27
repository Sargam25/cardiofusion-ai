"""Clinical and Laboratory Multimodal Fusion System.

Provides:
- Clinical-Only model (Demographics, Symptoms, Vitals: age, sex, cp, trestbps, exang)
- Laboratory-Only model (Biomarkers & Diagnostics: chol, fbs, thalach, oldpeak, slope, ca, thal, restecg)
- Early Fusion model (All modalities combined)
- Missing-Modality-Aware Fusion model with modality dropout training and explicit masking
- Isotonic probability calibration and SHAP feature attribution
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from src.evaluation.metrics import compute_metrics, select_threshold_on_val

logger = logging.getLogger(__name__)

CLINICAL_FEATURES = ["age", "sex", "cp", "trestbps", "exang"]
LAB_FEATURES = ["chol", "fbs", "thal", "ca", "oldpeak", "restecg", "thalach", "slope"]
ALL_FEATURES = CLINICAL_FEATURES + LAB_FEATURES

FEATURE_METADATA = {
    # Clinical Modality
    "age": {"modality": "Clinical", "label": "Age", "unit": "years", "type": "continuous", "normal_range": "18–100"},
    "sex": {"modality": "Clinical", "label": "Biological Sex", "unit": "1=Male, 0=Female", "type": "binary", "normal_range": "0 or 1"},
    "cp": {"modality": "Clinical", "label": "Chest Pain Type", "unit": "1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic", "type": "categorical", "normal_range": "1–4"},
    "trestbps": {"modality": "Clinical", "label": "Resting Blood Pressure", "unit": "mm Hg", "type": "continuous", "normal_range": "90–120 mm Hg"},
    "exang": {"modality": "Clinical", "label": "Exercise-Induced Angina", "unit": "1=Yes, 0=No", "type": "binary", "normal_range": "0"},
    # Laboratory Modality
    "chol": {"modality": "Laboratory", "label": "Serum Cholesterol", "unit": "mg/dl", "type": "continuous", "normal_range": "< 200 mg/dl"},
    "fbs": {"modality": "Laboratory", "label": "Fasting Blood Sugar > 120", "unit": "1=True, 0=False", "type": "binary", "normal_range": "0 (<=120 mg/dl)"},
    "thalach": {"modality": "Laboratory", "label": "Max Heart Rate Achieved", "unit": "bpm", "type": "continuous", "normal_range": "100–200 bpm"},
    "oldpeak": {"modality": "Laboratory", "label": "ST Depression (Exercise vs Rest)", "unit": "mm", "type": "continuous", "normal_range": "< 1.0 mm"},
    "slope": {"modality": "Laboratory", "label": "Peak Exercise ST Slope", "unit": "1=Upsloping, 2=Flat, 3=Downsloping", "type": "categorical", "normal_range": "1"},
    "ca": {"modality": "Laboratory", "label": "Fluoroscopy Major Vessels (0–3)", "unit": "count", "type": "integer", "normal_range": "0"},
    "thal": {"modality": "Laboratory", "label": "Thallium Scintigraphy", "unit": "3=Normal, 6=Fixed, 7=Reversible defect", "type": "categorical", "normal_range": "3"},
    "restecg": {"modality": "Laboratory", "label": "Resting ECG Category", "unit": "0=Normal, 1=ST-T abnormality, 2=LV hypertrophy", "type": "categorical", "normal_range": "0"},
}


class ClinicalLabMultimodalSystem:
    """Manages individual modality models and missing-modality-aware fusion."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.is_fitted = False

    def train_all_configurations(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, Any]]:
        """Train Clinical-only, Lab-only, Early Fusion, and Missing-Modality-Aware models."""
        y_tr = y_train.values.astype(int)
        y_v = y_val.values.astype(int)
        y_te = y_test.values.astype(int)

        scale_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

        # -----------------------------------------------------------------
        # 1. Clinical-Only Model
        # -----------------------------------------------------------------
        clin_cols = [c for c in CLINICAL_FEATURES if c in X_train.columns]
        X_tr_clin = X_train[clin_cols].values
        X_v_clin = X_val[clin_cols].values
        X_te_clin = X_test[clin_cols].values

        m_clin = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            random_state=self.random_state, early_stopping_rounds=20,
            eval_metric="logloss", verbosity=0
        )
        m_clin.fit(X_tr_clin, y_tr, eval_set=[(X_v_clin, y_v)], verbose=False)
        cal_clin = CalibratedClassifierCV(m_clin, method="isotonic", cv="prefit")
        cal_clin.fit(X_v_clin, y_v)

        val_probs_clin = cal_clin.predict_proba(X_v_clin)[:, 1]
        thresh_clin = select_threshold_on_val(y_v, val_probs_clin, criterion="youden")
        test_probs_clin = cal_clin.predict_proba(X_te_clin)[:, 1]
        metrics_clin = compute_metrics(y_te, test_probs_clin, threshold=thresh_clin, n_bootstrap=500, random_state=self.random_state)

        self.models["clinical_only"] = m_clin
        self.calibrators["clinical_only"] = cal_clin
        self.thresholds["clinical_only"] = thresh_clin
        self.metrics["clinical_only"] = metrics_clin

        # -----------------------------------------------------------------
        # 2. Laboratory-Only Model
        # -----------------------------------------------------------------
        lab_cols = [c for c in LAB_FEATURES if c in X_train.columns]
        X_tr_lab = X_train[lab_cols].values
        X_v_lab = X_val[lab_cols].values
        X_te_lab = X_test[lab_cols].values

        m_lab = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            random_state=self.random_state, early_stopping_rounds=20,
            eval_metric="logloss", verbosity=0
        )
        m_lab.fit(X_tr_lab, y_tr, eval_set=[(X_v_lab, y_v)], verbose=False)
        cal_lab = CalibratedClassifierCV(m_lab, method="isotonic", cv="prefit")
        cal_lab.fit(X_v_lab, y_v)

        val_probs_lab = cal_lab.predict_proba(X_v_lab)[:, 1]
        thresh_lab = select_threshold_on_val(y_v, val_probs_lab, criterion="youden")
        test_probs_lab = cal_lab.predict_proba(X_te_lab)[:, 1]
        metrics_lab = compute_metrics(y_te, test_probs_lab, threshold=thresh_lab, n_bootstrap=500, random_state=self.random_state)

        self.models["lab_only"] = m_lab
        self.calibrators["lab_only"] = cal_lab
        self.thresholds["lab_only"] = thresh_lab
        self.metrics["lab_only"] = metrics_lab

        # -----------------------------------------------------------------
        # 3. Early Fusion Model (Clinical + Lab)
        # -----------------------------------------------------------------
        all_cols = clin_cols + lab_cols
        X_tr_full = X_train[all_cols].values
        X_v_full = X_val[all_cols].values
        X_te_full = X_test[all_cols].values

        m_fusion = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            random_state=self.random_state, early_stopping_rounds=25,
            eval_metric="logloss", verbosity=0
        )
        m_fusion.fit(X_tr_full, y_tr, eval_set=[(X_v_full, y_v)], verbose=False)
        cal_fusion = CalibratedClassifierCV(m_fusion, method="isotonic", cv="prefit")
        cal_fusion.fit(X_v_full, y_v)

        val_probs_fusion = cal_fusion.predict_proba(X_v_full)[:, 1]
        thresh_fusion = select_threshold_on_val(y_v, val_probs_fusion, criterion="youden")
        test_probs_fusion = cal_fusion.predict_proba(X_te_full)[:, 1]
        metrics_fusion = compute_metrics(y_te, test_probs_fusion, threshold=thresh_fusion, n_bootstrap=500, random_state=self.random_state)

        self.models["early_fusion"] = m_fusion
        self.calibrators["early_fusion"] = cal_fusion
        self.thresholds["early_fusion"] = thresh_fusion
        self.metrics["early_fusion"] = metrics_fusion

        # -----------------------------------------------------------------
        # 4. Missing-Modality-Aware Fusion Model
        # Augment training data with modality dropout (25% lab missing, 25% clin missing)
        # -----------------------------------------------------------------
        X_tr_mma, y_tr_mma = self._create_modality_dropout_data(
            X_train, y_tr, clin_cols, lab_cols, seed=self.random_state
        )
        X_v_mma, y_v_mma = self._create_modality_dropout_data(
            X_val, y_v, clin_cols, lab_cols, seed=self.random_state + 1
        )

        m_mma = xgb.XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, scale_pos_weight=scale_pos,
            random_state=self.random_state, early_stopping_rounds=30,
            eval_metric="logloss", verbosity=0
        )
        m_mma.fit(X_tr_mma, y_tr_mma, eval_set=[(X_v_mma, y_v_mma)], verbose=False)
        cal_mma = CalibratedClassifierCV(m_mma, method="isotonic", cv="prefit")
        cal_mma.fit(X_v_mma, y_v_mma)

        val_probs_mma = cal_mma.predict_proba(X_v_mma)[:, 1]
        thresh_mma = select_threshold_on_val(y_v_mma, val_probs_mma, criterion="youden")

        # Evaluate on Complete Test Set
        X_te_mma_full = self._format_input_with_masks(X_test, clin_cols, lab_cols, clin_avail=True, lab_avail=True)
        test_probs_mma_full = cal_mma.predict_proba(X_te_mma_full)[:, 1]
        metrics_mma_full = compute_metrics(y_te, test_probs_mma_full, threshold=thresh_mma, n_bootstrap=500, random_state=self.random_state)

        # Evaluate on Lab-Missing Test Set
        X_te_mma_nolab = self._format_input_with_masks(X_test, clin_cols, lab_cols, clin_avail=True, lab_avail=False)
        test_probs_mma_nolab = cal_mma.predict_proba(X_te_mma_nolab)[:, 1]
        metrics_mma_nolab = compute_metrics(y_te, test_probs_mma_nolab, threshold=thresh_mma, n_bootstrap=500, random_state=self.random_state)

        # Evaluate on Clinical-Missing Test Set
        X_te_mma_noclin = self._format_input_with_masks(X_test, clin_cols, lab_cols, clin_avail=False, lab_avail=True)
        test_probs_mma_noclin = cal_mma.predict_proba(X_te_mma_noclin)[:, 1]
        metrics_mma_noclin = compute_metrics(y_te, test_probs_mma_noclin, threshold=thresh_mma, n_bootstrap=500, random_state=self.random_state)

        self.models["missing_modality_aware"] = m_mma
        self.calibrators["missing_modality_aware"] = cal_mma
        self.thresholds["missing_modality_aware"] = thresh_mma
        self.metrics["missing_modality_aware_full"] = metrics_mma_full
        self.metrics["missing_modality_aware_nolab"] = metrics_mma_nolab
        self.metrics["missing_modality_aware_noclin"] = metrics_mma_noclin

        self.is_fitted = True
        return self.metrics

    def predict_patient(
        self,
        patient_data: Dict[str, Any],
        clinical_available: bool = True,
        lab_available: bool = True,
    ) -> Dict[str, Any]:
        """Predict cardiovascular risk with missing-modality handling and SHAP attribution."""
        if not self.is_fitted:
            raise RuntimeError("Multimodal system has not been trained. Run train_all_configurations() first.")

        clin_cols = [c for c in CLINICAL_FEATURES if c in patient_data]
        lab_cols = [c for c in LAB_FEATURES if c in patient_data]

        # Case 1: Both modalities available -> use Early Fusion or MMA Full
        if clinical_available and lab_available:
            features = clin_cols + lab_cols
            x_arr = np.array([[patient_data[c] for c in features]], dtype=np.float32)
            cal = self.calibrators["early_fusion"]
            model = self.models["early_fusion"]
            thresh = self.thresholds["early_fusion"]
            mode = "Full Modalities (Clinical + Laboratory)"

        # Case 2: Only Clinical available -> use Clinical-Only Model
        elif clinical_available and not lab_available:
            features = clin_cols
            x_arr = np.array([[patient_data[c] for c in features]], dtype=np.float32)
            cal = self.calibrators["clinical_only"]
            model = self.models["clinical_only"]
            thresh = self.thresholds["clinical_only"]
            mode = "Clinical Only (Laboratory Missing / Pending)"

        # Case 3: Only Laboratory available -> use Lab-Only Model
        elif not clinical_available and lab_available:
            features = lab_cols
            x_arr = np.array([[patient_data[c] for c in features]], dtype=np.float32)
            cal = self.calibrators["lab_only"]
            model = self.models["lab_only"]
            thresh = self.thresholds["lab_only"]
            mode = "Laboratory Only (Clinical Exam Missing)"

        else:
            return {
                "error": "At least one modality (Clinical or Laboratory) must be available.",
                "risk_probability": 0.0,
                "risk_level": "Unknown",
                "prediction": 0,
            }

        # Predict calibrated probability
        prob = float(cal.predict_proba(x_arr)[0, 1])
        prediction = int(prob >= thresh)

        # Risk categorization
        if prob < 0.20:
            risk_level = "Low Risk"
            risk_color = "#10B981"  # green
        elif prob < 0.50:
            risk_level = "Moderate Risk"
            risk_color = "#F59E0B"  # yellow/orange
        elif prob < 0.80:
            risk_level = "High Risk"
            risk_color = "#EF4444"  # red
        else:
            risk_level = "Critical Risk"
            risk_color = "#DC2626"  # deep red

        # Calculate Native SHAP feature contributions
        dmat = xgb.DMatrix(x_arr, feature_names=features)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)[0]
        feat_contribs = contribs[:-1]
        base_val = float(contribs[-1])

        shap_df = pd.DataFrame({
            "feature": features,
            "value": [patient_data[f] for f in features],
            "shap_value": feat_contribs,
            "abs_impact": np.abs(feat_contribs),
            "modality": [FEATURE_METADATA.get(f, {}).get("modality", "Other") for f in features],
            "label": [FEATURE_METADATA.get(f, {}).get("label", f) for f in features],
            "unit": [FEATURE_METADATA.get(f, {}).get("unit", "") for f in features],
        }).sort_values("abs_impact", ascending=False)

        return {
            "mode": mode,
            "risk_probability": prob,
            "risk_percent": f"{prob * 100:.1f}%",
            "threshold": thresh,
            "prediction": prediction,
            "prediction_label": "CAD Indicated (Positive)" if prediction == 1 else "No Significant CAD (Negative)",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "clinical_available": clinical_available,
            "lab_available": lab_available,
            "shap_attributions": shap_df,
            "base_value": base_val,
        }

    def _create_modality_dropout_data(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        clin_cols: List[str],
        lab_cols: List[str],
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data augmented with modality dropout and mask flags."""
        rng = np.random.RandomState(seed)
        n = len(df)

        # We will create 3 copies:
        # 1. Full data (both mask flags = 1.0)
        # 2. Lab dropped (lab values = median, mask_lab = 0.0)
        # 3. Clinical dropped (clin values = median, mask_clin = 0.0)
        medians = df[clin_cols + lab_cols].median()

        # Copy 1: Full
        X_full = df[clin_cols + lab_cols].values
        m_clin_full = np.ones((n, 1), dtype=np.float32)
        m_lab_full = np.ones((n, 1), dtype=np.float32)
        X1 = np.hstack([X_full, m_clin_full, m_lab_full])
        y1 = y.copy()

        # Copy 2: Lab dropped
        X_nolab = df[clin_cols].values
        X_lab_dummy = np.tile(medians[lab_cols].values, (n, 1))
        m_clin_nolab = np.ones((n, 1), dtype=np.float32)
        m_lab_nolab = np.zeros((n, 1), dtype=np.float32)
        X2 = np.hstack([X_nolab, X_lab_dummy, m_clin_nolab, m_lab_nolab])
        y2 = y.copy()

        # Copy 3: Clinical dropped
        X_clin_dummy = np.tile(medians[clin_cols].values, (n, 1))
        X_lab = df[lab_cols].values
        m_clin_noclin = np.zeros((n, 1), dtype=np.float32)
        m_lab_noclin = np.ones((n, 1), dtype=np.float32)
        X3 = np.hstack([X_clin_dummy, X_lab, m_clin_noclin, m_lab_noclin])
        y3 = y.copy()

        X_aug = np.vstack([X1, X2, X3])
        y_aug = np.concatenate([y1, y2, y3])

        # Shuffle
        idx = rng.permutation(len(X_aug))
        return X_aug[idx].astype(np.float32), y_aug[idx].astype(int)

    def _format_input_with_masks(
        self,
        df: pd.DataFrame,
        clin_cols: List[str],
        lab_cols: List[str],
        clin_avail: bool = True,
        lab_avail: bool = True,
    ) -> np.ndarray:
        n = len(df)
        medians = df[clin_cols + lab_cols].median()

        X_clin = df[clin_cols].values if clin_avail else np.tile(medians[clin_cols].values, (n, 1))
        X_lab = df[lab_cols].values if lab_avail else np.tile(medians[lab_cols].values, (n, 1))
        m_clin = np.full((n, 1), 1.0 if clin_avail else 0.0, dtype=np.float32)
        m_lab = np.full((n, 1), 1.0 if lab_avail else 0.0, dtype=np.float32)

        return np.hstack([X_clin, X_lab, m_clin, m_lab]).astype(np.float32)

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"ClinicalLabMultimodalSystem saved to {path}")

    @classmethod
    def load(cls, filepath: str) -> "ClinicalLabMultimodalSystem":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"ClinicalLabMultimodalSystem loaded from {filepath}")
        return obj
