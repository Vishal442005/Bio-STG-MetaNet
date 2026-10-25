"""
Publication-Quality Neurodiagnostic Evaluation Metrics.
Computes Accuracy, Sensitivity, Specificity, Precision, F1-Score,
Cohen's Kappa, ROC-AUC, and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)


def compute_all_metrics(y_true, y_pred, y_probs=None):
    """
    Computes a comprehensive dictionary of biomedical classification metrics.
    y_true: 1D array of ground truth {0, 1}
    y_pred: 1D array of binary predictions {0, 1}
    y_probs: 1D array of positive class probabilities (for ROC-AUC)
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity (Recall)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    
    auc = 0.5
    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs)
        except Exception:
            auc = 0.5
            
    # Diagnostic Odds Ratio: (TP * TN) / (FP * FN)
    dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
    
    return {
        "accuracy": acc * 100.0,
        "sensitivity": sens * 100.0,
        "specificity": spec * 100.0,
        "precision": prec * 100.0,
        "f1_score": f1 * 100.0,
        "kappa": kappa,
        "roc_auc": auc * 100.0,
        "dor": dor,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
