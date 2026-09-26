"""Avalia o artefato salvo no split temporal de teste.

Separado do train.py porque em produção você quer poder reavaliar um
modelo já salvo sem re-treinar. Usa o threshold salvo no artefato; passe
`--threshold` só para simular outro ponto de operação.

    python -m src.evaluate [--model-path PATH] [--threshold 0.2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report

from src.artifact import ModelArtifact
from src.config import MODEL_PATH
from src.features import build_dataset
from src.metrics import classification_metrics, format_metrics
from src.modeling import split_xy, temporal_split


def evaluate(artifact: ModelArtifact, threshold: float | None = None) -> dict[str, float]:
    threshold = artifact.threshold if threshold is None else threshold
    _, test_df = temporal_split(build_dataset())
    X_test, y_test = split_xy(test_df)

    proba = artifact.predict_proba(X_test)
    print(classification_report(y_test, proba >= threshold, target_names=["no prazo", "atrasado"]))
    metrics = classification_metrics(y_test, proba, threshold)
    print(format_metrics(metrics))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=None, help="sobrescreve o threshold salvo")
    args = parser.parse_args()

    evaluate(ModelArtifact.load(args.model_path), threshold=args.threshold)


if __name__ == "__main__":
    main()
