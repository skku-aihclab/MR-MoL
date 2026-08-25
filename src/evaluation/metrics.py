"""Evaluation metrics for molecular tasks."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, cohen_kappa_score, balanced_accuracy_score,
)


def compute_classification_metrics(
    y_true: List[int],
    y_prob: List[float],
    y_pred: List[int] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics (Accuracy, F1, MCC, ROC-AUC).

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        y_pred: Predicted binary labels (optional, computed from y_prob if None)

    Returns:
        Dictionary with Accuracy, F1, MCC, ROC-AUC
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Compute predictions from probabilities if not provided
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = np.array(y_pred)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # F1 score (binary)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0.0)

    # Matthews Correlation Coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
        else:
            roc_auc = 0.5  # Undefined, return random baseline
    except:
        roc_auc = 0.5

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_acc,
        "f1": float(f1),
        "mcc": float(mcc),
        "cohen_kappa": kappa,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "valid_count": len(y_true),
        "total_count": len(y_true),
    }


def compute_multilabel_classification_metrics(
    y_true: List[int],
    y_prob: List[float],
    sub_task_ids: List[str],
    y_pred: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute macro-averaged classification metrics for multi-label datasets
    (Tox21, SIDER, ClinTox) where (molecule, sub_task) pairs are flattened.

    ROC-AUC and other metrics are computed per sub-task, then macro-averaged.
    Sub-tasks with only one ground-truth class are skipped (MoleculeNet convention).

    Args:
        y_true: Ground truth binary labels per sample
        y_prob: Predicted positive-class probabilities per sample
        sub_task_ids: Sub-task identifier per sample (e.g. "NR-AR", "CT_TOX")
        y_pred: Predicted binary labels (optional, derived from y_prob if None)

    Returns:
        Macro-averaged metrics plus a `per_task` dict of per-sub-task metrics.
    """
    if not (len(y_true) == len(y_prob) == len(sub_task_ids)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}, "
            f"sub_task_ids={len(sub_task_ids)}"
        )

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    sub_task_arr = np.array(sub_task_ids)

    if y_pred is None:
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)
    else:
        y_pred_arr = np.array(y_pred)

    per_task: Dict[str, Dict[str, float]] = {}
    auc_list, acc_list, f1_list, mcc_list = [], [], [], []
    bacc_list, kappa_list, sens_list, spec_list = [], [], [], []
    skipped = []

    for sub_task in sorted(set(sub_task_ids)):
        mask = sub_task_arr == sub_task
        yt = y_true_arr[mask]
        yp = y_prob_arr[mask]
        ypred = y_pred_arr[mask]

        task_acc = accuracy_score(yt, ypred)
        task_f1 = f1_score(yt, ypred, average='binary', zero_division=0.0)
        try:
            task_mcc = matthews_corrcoef(yt, ypred)
        except Exception:
            task_mcc = 0.0

        try:
            tn, fp, fn, tp = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0
        task_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        task_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        task_bacc = float(balanced_accuracy_score(yt, ypred))
        try:
            task_kappa = float(cohen_kappa_score(yt, ypred))
        except Exception:
            task_kappa = 0.0

        if len(np.unique(yt)) > 1:
            task_auc = roc_auc_score(yt, yp)
            auc_list.append(task_auc)
        else:
            task_auc = None
            skipped.append(sub_task)

        per_task[sub_task] = {
            "roc_auc": float(task_auc) if task_auc is not None else None,
            "accuracy": float(task_acc),
            "balanced_accuracy": task_bacc,
            "f1": float(task_f1),
            "mcc": float(task_mcc),
            "cohen_kappa": task_kappa,
            "sensitivity": task_sens,
            "specificity": task_spec,
            "count": int(mask.sum()),
            "positive_count": int(yt.sum()),
        }
        acc_list.append(task_acc)
        f1_list.append(task_f1)
        mcc_list.append(task_mcc)
        bacc_list.append(task_bacc)
        kappa_list.append(task_kappa)
        sens_list.append(task_sens)
        spec_list.append(task_spec)

    if not auc_list:
        raise RuntimeError(
            "no sub-task had both classes present; cannot compute ROC-AUC"
        )

    return {
        "accuracy": float(np.mean(acc_list)),
        "balanced_accuracy": float(np.mean(bacc_list)),
        "f1": float(np.mean(f1_list)),
        "mcc": float(np.mean(mcc_list)),
        "cohen_kappa": float(np.mean(kappa_list)),
        "sensitivity": float(np.mean(sens_list)),
        "specificity": float(np.mean(spec_list)),
        "roc_auc": float(np.mean(auc_list)),
        "num_sub_tasks": len(per_task),
        "num_sub_tasks_scored": len(auc_list),
        "skipped_sub_tasks": skipped,
        "per_task": per_task,
        "valid_count": len(y_true),
        "total_count": len(y_true),
    }


def compute_regression_metrics(
    y_true: List[float],
    y_pred: List[float],
) -> Dict[str, float]:
    """
    Compute regression metrics (RMSE, MAE, R²).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE, MAE, R²
    """
    # Filter out None values
    total_count = len(y_true)
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]

    if not valid_pairs:
        return {
            "rmse": float('inf'),
            "mae": float('inf'),
            "r2": 0.0,
            "valid_rate": 0.0,
            "valid_count": 0,
            "total_count": total_count,
        }

    valid_true, valid_pred = zip(*valid_pairs)
    y_true_arr = np.array(valid_true)
    y_pred_arr = np.array(valid_pred)

    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mae = mean_absolute_error(y_true_arr, y_pred_arr)

    try:
        r2 = r2_score(y_true_arr, y_pred_arr)
    except:
        r2 = 0.0

    error = y_pred_arr - y_true_arr
    mean_error = np.mean(error)
    std_error = np.std(error)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "valid_rate": len(valid_pairs) / total_count,
        "valid_count": len(valid_pairs),
        "total_count": total_count,
    }


