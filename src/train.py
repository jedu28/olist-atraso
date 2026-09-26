"""Treina o modelo final e salva o artefato (pipeline + threshold).

Passos:
1. dataset com features point-in-time e split temporal (80/20);
2. XGBoost com os hiperparâmetros do Optuna (`config.XGB_TUNED_PARAMS`);
3. threshold escolhido no OOF temporal do treino (`MIN_RECALL` de recall);
4. avaliação única no teste e gravação do artefato.

    python -m src.train [--compare-baselines] [--min-recall 0.5]
"""
from __future__ import annotations

import argparse
import logging

from src.artifact import ModelArtifact
from src.config import MIN_RECALL, MODEL_PATH, XGB_TUNED_PARAMS
from src.features import build_dataset
from src.metrics import classification_metrics, format_metrics
from src.modeling import build_tuned_xgb_pipeline, compare_baselines, split_xy, temporal_split
from src.thresholds import choose_threshold, temporal_oof_proba

logger = logging.getLogger(__name__)


def train(min_recall: float = MIN_RECALL, compare: bool = False) -> ModelArtifact:
    df = build_dataset()
    train_df, test_df = temporal_split(df)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    logger.info("Treino: %s pedidos (atraso %.1f%%) | Teste: %s pedidos (atraso %.1f%%)",
                f"{len(train_df):,}", 100 * y_train.mean(), f"{len(test_df):,}", 100 * y_test.mean())

    if compare:
        logger.info("Baselines no mesmo split:\n%s", compare_baselines(X_train, y_train, X_test, y_test).round(4))

    pipeline = build_tuned_xgb_pipeline()

    logger.info("Escolhendo o threshold no OOF temporal do treino (recall mínimo %.0f%%)...", 100 * min_recall)
    oof_proba, oof_y = temporal_oof_proba(pipeline, X_train, y_train)
    threshold = choose_threshold(oof_y, oof_proba, min_recall=min_recall)
    oof_metrics = classification_metrics(oof_y, oof_proba, threshold)
    logger.info("OOF temporal (prometido): %s", format_metrics(oof_metrics))

    pipeline.fit(X_train, y_train)
    test_metrics = classification_metrics(y_test, pipeline.predict_proba(X_test)[:, 1], threshold)
    logger.info("Teste (entregue):         %s", format_metrics(test_metrics))

    artifact = ModelArtifact(
        pipeline=pipeline,
        threshold=threshold,
        metadata={
            "model": "XGBoost",
            "params": XGB_TUNED_PARAMS,
            "min_recall": min_recall,
            "train_period": [str(train_df["order_purchase_timestamp"].min()), str(train_df["order_purchase_timestamp"].max())],
            "test_period": [str(test_df["order_purchase_timestamp"].min()), str(test_df["order_purchase_timestamp"].max())],
            "oof_metrics": oof_metrics,
            "test_metrics": test_metrics,
        },
    )
    path = artifact.save(MODEL_PATH)
    logger.info("Artefato salvo em %s (threshold %.4f)", path, threshold)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL, help="recall mínimo no OOF temporal")
    parser.add_argument("--compare-baselines", action="store_true", help="treina e compara os baselines antes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    train(min_recall=args.min_recall, compare=args.compare_baselines)


if __name__ == "__main__":
    main()
