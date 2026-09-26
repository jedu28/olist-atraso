"""Construção dos pipelines de modelo e do split temporal.

Split temporal porque o modelo em produção sempre vai prever o futuro a
partir do passado — validar com split aleatório superestimaria a
performance real.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    CAT_FEATURES,
    FEATURES,
    NUM_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_FRACTION,
    TIMESTAMP_COLUMN,
    XGB_BASE_PARAMS,
    XGB_DEFAULT_PARAMS,
    XGB_TUNED_PARAMS,
)


# --- Split --------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino = pedidos mais antigos; teste = a fração mais recente."""
    df = df.sort_values(TIMESTAMP_COLUMN)
    cutoff = int(len(df) * (1 - test_fraction))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]


# --- Pipelines ----------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)])


def build_pipeline(model: BaseEstimator | None = None) -> Pipeline:
    """Pré-processamento + estimador. Sem estimador, usa a regressão logística baseline."""
    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_xgb_pipeline(params: dict | None = None) -> Pipeline:
    """Pipeline XGBoost. `params` sobrescreve os hiperparâmetros padrão (V1)."""
    return build_pipeline(XGBClassifier(**{**XGB_DEFAULT_PARAMS, **(params or {}), **XGB_BASE_PARAMS}))


def build_tuned_xgb_pipeline() -> Pipeline:
    """Pipeline XGBoost com os hiperparâmetros encontrados pelo Optuna (V2+)."""
    return build_xgb_pipeline(XGB_TUNED_PARAMS)


# --- Baselines ----------------------------------------------------------------
def candidate_models() -> dict[str, BaseEstimator]:
    return {
        "Regressão Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }


def compare_baselines(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Treina cada modelo candidato no mesmo split e devolve ROC-AUC, PR-AUC e recall (corte 0.5)."""
    rows = {}
    for name, model in candidate_models().items():
        proba = build_pipeline(model).fit(X_train, y_train).predict_proba(X_test)[:, 1]
        rows[name] = {
            "ROC-AUC": roc_auc_score(y_test, proba),
            "PR-AUC": average_precision_score(y_test, proba),
            "Recall (corte 0.5)": recall_score(y_test, proba >= 0.5),
        }
    return pd.DataFrame(rows).T
