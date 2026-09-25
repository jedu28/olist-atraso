"""Baseline: regressão logística com split temporal (não aleatório).

Split temporal porque o modelo em produção sempre vai prever o futuro a
partir do passado — validar com split aleatório superestimaria a
performance real.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import build_dataset

NUM_FEATURES = [
    "preco_total",
    "frete_total",
    "n_itens",
    "product_weight_g",
    "estados_diferentes",
    "dia_semana_compra",
    "mes_compra",
    "seller_pedidos_anteriores",
    "seller_taxa_atraso_historica",
]
CAT_FEATURES = ["product_category_name", "customer_state", "seller_state"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_FEATURES),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("order_purchase_timestamp")
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def main():
    df = build_dataset()
    train_df, test_df = temporal_split(df)

    X_train, y_train = train_df[NUM_FEATURES + CAT_FEATURES], train_df["atrasado"]
    X_test, y_test = test_df[NUM_FEATURES + CAT_FEATURES], test_df["atrasado"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"PR-AUC: {average_precision_score(y_test, proba):.4f}")
    print(f"Recall: {recall_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    return pipeline


if __name__ == "__main__":
    main()
