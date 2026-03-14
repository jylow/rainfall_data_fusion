
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
)


# ================================================================
# ===  CORE METRIC FUNCTIONS
# ================================================================

def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error between predictions and targets."""
    valid = (~np.isnan(preds)) & (~np.isnan(targets))
    return mean_absolute_error(targets[valid], preds[valid])


def compute_binary_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    pos_label: int = 1,
    zero_division: int = 0,
) -> dict:
    """
    Threshold continuous predictions/targets into binary classes
    (rain >= threshold → 1, else → 0) and compute precision, recall, F1.

    Parameters
    ----------
    preds : np.ndarray      – continuous model predictions
    targets : np.ndarray    – continuous ground-truth values
    threshold : float        – rainfall threshold (mm) for positive class
    pos_label : int          – which class is "positive" (default 1 = rain)
    zero_division : int      – value returned when a metric is undefined

    Returns
    -------
    dict with keys: precision, recall, f1, confusion_matrix, threshold,
                    support_pos, support_neg
    """
    valid = (~np.isnan(preds)) & (~np.isnan(targets))
    preds_v = preds[valid]
    targets_v = targets[valid]

    pred_labels = (preds_v >= threshold).astype(int)
    true_labels = (targets_v >= threshold).astype(int)

    precision = precision_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    recall = recall_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    f1 = f1_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
        "support_pos": int(true_labels.sum()),
        "support_neg": int((1 - true_labels).sum()),
    }


# ================================================================
# ===  PER-STATION WRAPPER
# ================================================================

def compute_per_station_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    station_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute MAE and binary classification metrics for each station.

    Parameters
    ----------
    preds : np.ndarray        – flat array of predictions  [N,]
    targets : np.ndarray      – flat array of targets      [N,]
    station_ids : np.ndarray  – flat array of station IDs  [N,]
    threshold : float          – rainfall threshold (mm)

    Returns
    -------
    dict mapping station_id → { mae, precision, recall, f1, ... }
    """
    unique_ids = np.unique(station_ids)
    results = {}

    for sid in unique_ids:
        mask = station_ids == sid
        p = preds[mask]
        t = targets[mask]

        if len(p) < 2:
            continue

        mae = compute_mae(p, t)
        cls = compute_binary_classification_metrics(p, t, threshold=threshold)

        results[int(sid)] = {
            "mae": mae,
            **cls,
        }

    return results


def print_metrics_summary(global_metrics: dict, per_station: dict = None):
    """Pretty-print global and (optionally) per-station metrics."""
    print("\n" + "=" * 60)
    print("  GLOBAL METRICS")
    print("=" * 60)
    print(f"  MAE            : {global_metrics['mae']:.4f}")
    print(f"  Threshold      : {global_metrics['threshold']}")
    print(f"  Precision      : {global_metrics['precision']:.4f}")
    print(f"  Recall         : {global_metrics['recall']:.4f}")
    print(f"  F1 Score       : {global_metrics['f1']:.4f}")
    print(f"  Support (pos)  : {global_metrics['support_pos']}")
    print(f"  Support (neg)  : {global_metrics['support_neg']}")
    cm = global_metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    if per_station:
        print("\n" + "=" * 60)
        print("  PER-STATION METRICS")
        print("=" * 60)
        header = f"  {'Station':>8s} | {'MAE':>8s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'Pos':>5s} | {'Neg':>5s}"
        print(header)
        print("  " + "-" * len(header))
        for sid in sorted(per_station.keys()):
            m = per_station[sid]
            print(
                f"  {sid:>8d} | {m['mae']:8.4f} | {m['precision']:6.4f} | "
                f"{m['recall']:6.4f} | {m['f1']:6.4f} | {m['support_pos']:5d} | {m['support_neg']:5d}"
            )
    print("=" * 60)
