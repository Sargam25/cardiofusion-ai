# CardioFusion-AI: Clinical & Laboratory Multimodal Cardiovascular AI

## Explainable, Missing-Modality-Aware Multimodal Cardiovascular Disease Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Framework: XGBoost](https://img.shields.io/badge/Framework-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A research-quality multimodal biomedical artificial intelligence prototype designed to predict cardiovascular outcomes from **genuinely linked Clinical Examination records** and **Laboratory Diagnostic tests** under realistic clinical conditions where modalities may be missing or delayed.

---

> [!CAUTION]
> **RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE**  
> CardioFusion-AI is an experimental research system designed for methodology evaluation and biomedical AI research. It has **NOT** been certified by the FDA, CE, or any regulatory agency. Model predictions must **never** be used to guide acute clinical management, triage, or replace diagnostic evaluation by a certified physician.

---

## Key Research Capabilities

1. **Genuinely Linked Clinical & Laboratory Cohort**:
   - Built on 303 patient records with complete, verified cross-modality linkage.
   - **Clinical Modality (5 bedside features)**: Age, Biological Sex, Chest Pain Category (`cp`), Resting Blood Pressure (`trestbps`), and Exercise-Induced Angina (`exang`).
   - **Laboratory Modality (8 biochemical & diagnostic tests)**: Serum Cholesterol (`chol`), Fasting Blood Sugar > 120 (`fbs`), Maximum Heart Rate Achieved (`thalach`), ST Depression (`oldpeak`), Peak ST Slope (`slope`), Fluoroscopy Major Vessels (`ca`), Thallium Scintigraphy (`thal`), and Resting ECG Category (`restecg`).

2. **Explicit Missing-Modality Awareness**:
   - Represents modality availability via per-sample binary availability flags $m_{clin}, m_{lab} \in \{0, 1\}$.
   - Implements training-time **Modality Dropout** to force graceful degradation when laboratory workups or clinical exams are delayed or absent.
   - **No Naive Imputation**: Missing modalities are never imputed as "normal" or zero in feature space.

3. **Clinician-Centered Explainability**:
   - **Local & Global SHAP**: Unrolls patient-level risk factor contributions and calculates the relative predictive share between the Clinical domain and Laboratory domain.
   - Structured diagnostic reports and transparent distinction between predictive attribution and clinical causation.

4. **Rigorous Probability Calibration**:
   - Isotonic regression and Platt scaling on validation fold to ensure predicted probabilities match true empirical risk.

5. **Polished Streamlit Research Dashboard**:
   - Custom styling with **high-contrast black colour words for all selectable options**, dropdowns, and radio pills.
   - Live intake form with toggle for laboratory test availability, missing-modality simulation stress tester, SHAP waterfall plots, and complete ablation benchmarks.

---

## Experimental Benchmark Results (Held-Out Test Set)

| Model Configuration | Modalities Active | AUROC [95% CI] | AUPRC | F1 Score | Sensitivity | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|
| **Clinical Only** | Vitals & Symptoms | 0.8994 [0.8018, 0.9700] | 0.8585 | 0.6222 | 50.00% | **90.91%** | 0.1586 |
| **Laboratory Only** | Biomarkers & Tests | 0.9248 [0.8596, 0.9820] | 0.8940 | 0.7500 | **96.43%** | 48.48% | 0.1307 |
| **Early Fusion** | Clinical + Laboratory | 0.9361 [0.8677, 0.9839] | 0.8791 | **0.8571** | **96.43%** | 75.76% | **0.0964** |
| **Missing-Modality-Aware (Full)** | Clinical + Laboratory | **0.9426** [0.8861, 0.9876] | **0.9066** | 0.8525 | 92.86% | 78.79% | 0.0980 |
| **Missing-Modality-Aware (No-Lab)** | Clinical Only (Lab Missing) | 0.9058 [0.8242, 0.9600] | 0.8403 | 0.7600 | 67.86% | 90.91% | 0.1499 |
| **Missing-Modality-Aware (No-Clin)** | Lab Only (Clin Missing) | 0.9026 [0.8274, 0.9713] | 0.8532 | 0.8364 | 82.14% | 87.88% | 0.1220 |

---

## Repository Structure

```
cardiofusion-ai/
├── app.py                     # Streamlit clinical research dashboard (Black option text styling)
├── requirements.txt           # Pinned dependencies (Python 3.12 compatible)
├── README.md                  # Project documentation & setup guide
├── configs/                   # Configuration files
│   ├── clinical_model.yaml    # Tabular hyperparameters & calibration config
│   ├── fusion_model.yaml      # Multimodal fusion config
│   └── training.yaml          # Global reproducibility settings
├── data/
│   ├── README.md              # Dataset provenance and download instructions
│   └── uci/                   # Cached Cleveland Heart Disease dataset
├── docs/                      # Research documentation
│   ├── dataset_assessment.md  # Comprehensive evaluation of clinical data sources
│   ├── methodology.md         # Technical architecture & mathematical formulation
│   ├── limitations.md         # Transparent disclosure of scientific constraints
│   ├── novelty_and_literature_review.md # Academic literature survey
│   └── research_report_outline.md       # Formal paper outline
├── scripts/                   # CLI execution scripts
│   ├── train_multimodal_clinical_lab.py # Trains and ablates all modality configurations
│   ├── train_clinical.py      # Standard clinical baseline training
│   ├── download_data.py       # Automated benchmark fetcher
│   └── validate_data.py       # Schema and data quality validator
├── src/                       # Source modules
│   ├── data/                  # UCIHeartLoader and DataValidator
│   ├── preprocessing/         # ClinicalPreprocessor, ModalityMaskGenerator
│   ├── models/                # ClinicalLabMultimodalSystem, ClinicalXGBoostModel
│   ├── evaluation/            # AUROC, AUPRC, Brier score, bootstrap CIs
│   ├── explainability/        # SHAP explainer and report generator
│   └── utils/                 # Config loader, reproducibility seeds, plotting
├── tests/                     # Automated pytest suite (48 tests passing)
│   ├── test_preprocessing.py  # Zero-leakage & imputer tests
│   ├── test_models.py         # Multimodal system, tensor shapes, gradient checks
│   ├── test_evaluation.py     # Metric correctness & CI bootstrap checks
│   └── test_splits.py         # Patient leakage verification
└── models/                    # Serialized checkpoints (.pkl)
    ├── clinical_lab_multimodal_system.pkl
    └── clinical_xgb_uci.pkl
```

---

## Installation & Setup (Windows & VS Code)

### Prerequisites
- Windows 10/11 (or Linux/macOS)
- Python 3.12 (64-bit)
- Git

### 1. Open Terminal in Project Directory
```powershell
cd C:\Users\sarga\.gemini\antigravity\scratch\cardiofusion-ai
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Fetch & Validate Dataset
```powershell
python scripts/download_data.py --dataset uci
python scripts/validate_data.py --verbose
```

---

## Running Training & Ablation Experiments

Train all modality configurations (Clinical-only, Lab-only, Early Fusion, and Missing-Modality-Aware):
```powershell
python scripts/train_multimodal_clinical_lab.py
```
This evaluates all configurations on the held-out test split, computes 95% bootstrap confidence intervals ($B=500$), saves model checkpoints to `models/`, and writes summary tables to `reports/tables/modality_ablation_summary.csv`.

---

## Running Automated Tests

Run the full pytest suite (48 tests):
```powershell
python -m pytest tests/ -v
```

---

## Launching the Streamlit Research Dashboard

Start the interactive web dashboard:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### UI Highlights:
- **High-Contrast Option Words**: All selectboxes, radio button options, dropdown menus, and tabs feature crisp **black colour words** on light, clean cards.
- **Interactive Intake Calculator**: Real-time risk scoring with toggle for whether laboratory results are available.
- **Missing-Modality Simulator**: Side-by-side comparison of full vs. bedside-only vs. lab-only scenarios.
- **SHAP Waterfall**: Individual risk factor attribution with clinical reference values.
