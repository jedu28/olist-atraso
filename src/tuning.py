"""Busca de hiperparâmetros do XGBoost com Optuna.

O resultado fica fixado em `src.config.XGB_TUNED_PARAMS`; este módulo existe
para a busca ser reproduzível (`make tune`) e não depender do notebook.
"""
from __future__ import annotations

import optuna
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit, cross_val_score

from src.config import N_TEMPORAL_SPLITS, RANDOM_STATE
from src.modeling import build_xgb_pipeline


def xgb_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
    }


def tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    cv: BaseCrossValidator | None = None,
) -> optuna.Study:
    """Maximiza o ROC-AUC médio na CV. O sampler tem semente fixa (busca reproduzível).

    A CV padrão é temporal (`TimeSeriesSplit`, `X` ordenado por data): cada fold
    valida só no futuro do treino, como em produção. Com CV aleatória a busca
    otimiza uma métrica inflada (~0.80 contra ~0.60 na temporal) e escolhe
    hiperparâmetros que generalizam pior para meses novos.
    """
    cv = cv or TimeSeriesSplit(n_splits=N_TEMPORAL_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_xgb_pipeline(xgb_search_space(trial))
        return cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    from src.features import build_dataset
    from src.modeling import split_xy, temporal_split

    train_df, _ = temporal_split(build_dataset())
    study = tune_xgb(*split_xy(train_df))
    print(f"Melhor ROC-AUC (CV): {study.best_value:.4f}")
    print("Copie para src/config.py -> XGB_TUNED_PARAMS:")
    print("XGB_TUNED_PARAMS = {")
    for key, value in study.best_params.items():
        print(f"    {key!r}: {value!r},")
    print("}")


if __name__ == "__main__":
    main()
