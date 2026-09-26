"""Escolha do threshold (corte de probabilidade) que transforma score em decisão.

O corte é escolhido sempre em probabilidades out-of-fold (OOF) do treino —
nunca no teste, que é o conjunto de avaliação final — e com OOF **temporal**:
cada bloco do treino é previsto por um modelo que só viu o passado dele,
exatamente a situação de produção. O OOF de uma CV aleatória é otimista e
leva a um corte alto demais (ver notebooks/02_modelagem.ipynb, seção 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.config import MIN_RECALL, N_TEMPORAL_SPLITS


def choose_threshold(y_true, proba, min_recall: float = MIN_RECALL) -> float:
    """Maior corte de probabilidade que ainda garante `min_recall`.

    O corte de 0.5 é arbitrário para uma classe rara (~8% de atraso): o modelo
    quase nunca passa de 0.5 de proba, mesmo com ranking informativo. Aqui
    escolhemos pela curva precision-recall.

    Entre os cortes que atingem o recall mínimo, pegamos o mais alto (o de maior
    precisão). Se nenhum atinge, cai no menor corte disponível.
    """
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    validos = recall[:-1] >= min_recall
    return float(thresholds[validos][-1] if validos.any() else thresholds[0])


def temporal_oof_proba(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = N_TEMPORAL_SPLITS
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades out-of-fold respeitando o tempo.

    `X` precisa estar ordenado por data. O primeiro bloco da TimeSeriesSplit não
    tem passado para treinar, então fica sem previsão e é descartado: o retorno
    é `(proba, y)` só das linhas que receberam previsão.
    """
    proba = pd.Series(np.nan, index=X.index)
    for idx_treino, idx_valid in TimeSeriesSplit(n_splits=n_splits).split(X):
        modelo = clone(pipeline).fit(X.iloc[idx_treino], y.iloc[idx_treino])
        proba.iloc[idx_valid] = modelo.predict_proba(X.iloc[idx_valid])[:, 1]
    tem_previsao = proba.notna().to_numpy()
    return proba.to_numpy()[tem_previsao], y.to_numpy()[tem_previsao]
