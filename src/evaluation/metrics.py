"""Unified evaluation metrics for classification tasks.

Provides AUROC, AUPRC, F1, sensitivity, specificity, confusion matrix,
Brier score, calibration curves, and bootstrap confidence intervals.

Never optimize hyperparameters on test set.
All threshold selection is documented and performed on validation data.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def select_threshold_on_val(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    criterion: str = "youden",
) -> float:
    """Select optimal decision threshold on validation set.

    Args:
        y_val: Ground truth binary labels on validation set.
        y_prob_val: Predicted probabilities for positive class.
        criterion: 'youden' (maximize sensitivity + specificity - 1) or 'f1'.

    Returns:
        Selected threshold as float in [0.05, 0.95].
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_score = -float("inf")

    y_val = np.asarray(y_val).ravel()
    y_prob_val = np.asarray(y_prob_val).ravel()

    for t in thresholds:
        y_pred = (y_prob_val >= t).astype(int)
        if criterion == "youden":
            cm = confusion_matrix(y_val, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sens + spec - 1.0
            else:
                score = 0.0
        elif criterion == "f1":
            score = float(f1_score(y_val, y_pred, zero_division=0))
        else:
            score = float(accuracy_score(y_val, y_pred))

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    logger.debug(f"Selected threshold {best_thresh:.3f} using criterion '{criterion}' (score: {best_score:.4f})")
    return float(best_thresh)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute binary classification metrics with optional bootstrap confidence intervals.

    Args:
        y_true: Ground truth binary labels (N,).
        y_prob: Predicted probabilities for positive class (N,).
        threshold: Classification threshold.
        n_bootstrap: Number of bootstrap iterations for 95% CIs.
        random_state: Seed for bootstrap sampling.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float(np.mean(y_true))

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    accuracy = float(accuracy_score(y_true, y_pred))

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
        "positive_prevalence": float(np.mean(y_true)),
    }

    if n_bootstrap > 0 and len(y_true) > 10:
        rng = np.random.default_rng(random_state)
        boot_auroc = []
        boot_auprc = []
        boot_f1 = []
        boot_brier = []

        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            y_b = y_true[idx]
            p_b = y_prob[idx]
            if len(np.unique(y_b)) < 2:
                continue

            try:
                boot_auroc.append(float(roc_auc_score(y_b, p_b)))
            except ValueError:
                pass
            try:
                boot_auprc.append(float(average_precision_score(y_b, p_b)))
            except ValueError:
                pass
            y_pred_b = (p_b >= threshold).astype(int)
            boot_f1.append(float(f1_score(y_b, y_pred_b, zero_division=0)))
            boot_brier.append(float(brier_score_loss(y_b, p_b)))

        bootstrap_ci = {}
        for name, vals in [("auroc", boot_auroc), ("auprc", boot_auprc), ("f1", boot_f1), ("brier", boot_brier)]:
            if vals:
                low = float(np.percentile(vals, 2.5))
                high = float(np.percentile(vals, 97.5))
                bootstrap_ci[name] = {"lower": low, "upper": high}
            else:
                bootstrap_ci[name] = {"lower": 0.0, "upper": 1.0}

        metrics["bootstrap_ci"] = bootstrap_ci

    return metrics
