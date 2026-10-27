"""Train and evaluate the Clinical & Laboratory Multimodal System.

Trains:
1. Clinical-Only Model (Demographics, Symptoms, Vitals)
2. Laboratory-Only Model (Biochemical & Diagnostic Tests)
3. Early Fusion Model (Clinical + Lab concatenated)
4. Missing-Modality-Aware Fusion Model (Evaluated under Full, No-Lab, and No-Clinical conditions)

Usage:
    python scripts/train_multimodal_clinical_lab.py
"""
import json
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.data.uci_loader import UCIHeartLoader
from src.models.clinical_lab_fusion import ClinicalLabMultimodalSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_multimodal_clinical_lab")


def main():
    logger.info("==================================================================")
    logger.info("TRAINING CLINICAL & LABORATORY MULTIMODAL CARDIOVASCULAR AI")
    logger.info("==================================================================")

    # 1. Load data
    loader = UCIHeartLoader(cache_dir="data/uci")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_train_val_test_split(
        test_size=0.2, val_size=0.1, random_state=42, stratify=True
    )

    logger.info(f"Dataset loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"Target distribution in Train: {dict(y_train.value_counts())}")

    # 2. Train multimodal system
    system = ClinicalLabMultimodalSystem(random_state=42)
    logger.info("Training Clinical-Only, Lab-Only, Early-Fusion, and Missing-Modality-Aware models...")
    all_metrics = system.train_all_configurations(X_train, y_train, X_val, y_val, X_test, y_test)

    # 3. Print Results Summary
    logger.info("\n" + "=" * 75)
    logger.info("TEST SET EVALUATION RESULTS ACROSS MODALITY CONFIGURATIONS")
    logger.info("=" * 75)

    summary_rows = []
    config_names = [
        ("Clinical Only (Vitals/Symptoms)", "clinical_only"),
        ("Laboratory Only (Biomarkers/Tests)", "lab_only"),
        ("Early Fusion (Clinical + Lab)", "early_fusion"),
        ("Missing-Modality: Full Modalities", "missing_modality_aware_full"),
        ("Missing-Modality: Lab Missing", "missing_modality_aware_nolab"),
        ("Missing-Modality: Clinical Missing", "missing_modality_aware_noclin"),
    ]

    for display_name, key in config_names:
        m = all_metrics.get(key, {})
        auroc = m.get("auroc", 0.0)
        auprc = m.get("auprc", 0.0)
        f1 = m.get("f1", 0.0)
        sens = m.get("sensitivity", 0.0)
        spec = m.get("specificity", 0.0)
        brier = m.get("brier_score", 0.0)
        ci_lower = m.get("bootstrap_ci", {}).get("auroc", {}).get("lower", 0.0)
        ci_upper = m.get("bootstrap_ci", {}).get("auroc", {}).get("upper", 0.0)

        logger.info(
            f"{display_name:<38} | AUROC: {auroc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] | "
            f"F1: {f1:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | Brier: {brier:.4f}"
        )

        summary_rows.append({
            "Configuration": display_name,
            "AUROC": round(auroc, 4),
            "AUROC_95CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "AUPRC": round(auprc, 4),
            "F1_Score": round(f1, 4),
            "Sensitivity": round(sens, 4),
            "Specificity": round(spec, 4),
            "Brier_Score": round(brier, 4),
        })

    # 4. Save model and results
    output_model_path = Path("models/clinical_lab_multimodal_system.pkl")
    system.save(str(output_model_path))

    reports_dir = Path("reports/tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(reports_dir / "modality_ablation_summary.csv", index=False)

    with open(reports_dir / "modality_ablation_summary.json", "w", encoding="utf-8") as f:
        # Convert non-serializable objects
        clean_metrics = {}
        for k, v in all_metrics.items():
            clean_metrics[k] = {
                mk: mv.tolist() if hasattr(mv, "tolist") else mv
                for mk, mv in v.items()
                if not isinstance(mv, np.ndarray)
            }
        json.dump(clean_metrics, f, indent=2)

    logger.info(f"Model saved to: {output_model_path}")
    logger.info(f"Ablation summary saved to: {reports_dir / 'modality_ablation_summary.csv'}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
