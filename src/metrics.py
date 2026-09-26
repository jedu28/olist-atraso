"""Métricas de classificação num ponto de operação (threshold) específico."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    """Métricas de ranking (independem do corte) + métricas do ponto de operação.

    ROC-AUC e PR-AUC avaliam o score em todos os cortes de uma vez; precisão,
    recall, F1 e a matriz de confusão dependem do `threshold`.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "flagged_rate": float(pred.mean()),
        "base_rate": float(y_true.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Resumo legível para log/terminal."""
    return (
        f"threshold={metrics['threshold']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
        f"precisão={metrics['precision']:.1%} | recall={metrics['recall']:.1%} | F1={metrics['f1']:.3f} | "
        f"sinalizados={metrics['flagged_rate']:.1%} (TP={metrics['tp']:,} FP={metrics['fp']:,} "
        f"FN={metrics['fn']:,} TN={metrics['tn']:,})"
    )
