"""Avalia o pipeline treinado no split temporal de teste.

Separado do train.py porque em produção você quer poder reavaliar um
modelo já salvo sem re-treinar.
"""
import argparse
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from src.features import build_dataset
from src.train import CAT_FEATURES, NUM_FEATURES, temporal_split

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "pipeline.joblib"


def evaluate(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "recall": recall_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    print(classification_report(y_test, preds))
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"{key}: {value:.4f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Nenhum modelo salvo em {args.model_path}. Rode `python -m src.train` primeiro "
            "(ele precisa ser ajustado para salvar o pipeline com joblib.dump)."
        )

    pipeline = joblib.load(args.model_path)
    df = build_dataset()
    _, test_df = temporal_split(df)
    X_test, y_test = test_df[NUM_FEATURES + CAT_FEATURES], test_df["atrasado"]

    evaluate(pipeline, X_test, y_test, threshold=args.threshold)


if __name__ == "__main__":
    main()
