import numpy as np
from sklearn.metrics import recall_score

from src.config import MIN_RECALL
from src.thresholds import choose_threshold


def test_threshold_garante_recall_minimo():
    # Classe rara (10% de positivos) com probabilidades baixas: nenhum caso
    # passa de 0.5, então o corte padrão zeraria o recall.
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.linspace(0.01, 0.30, 90), np.linspace(0.20, 0.45, 10)])

    threshold = choose_threshold(y, proba, min_recall=MIN_RECALL)

    assert recall_score(y, (proba >= 0.5).astype(int)) == 0.0
    assert recall_score(y, (proba >= threshold).astype(int)) >= MIN_RECALL


def test_threshold_escolhe_o_corte_mais_alto_que_atende_o_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45])

    threshold = choose_threshold(y, proba, min_recall=0.5)

    # Qualquer corte acima do escolhido derruba o recall abaixo do mínimo.
    assert recall_score(y, (proba >= threshold).astype(int)) >= 0.5
    maiores = proba[proba > threshold]
    for corte in maiores:
        assert recall_score(y, (proba >= corte).astype(int)) < 0.5


def test_threshold_cai_no_menor_corte_quando_recall_inatingivel():
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.7, 0.1])

    # min_recall=1.0 só seria atendido incluindo o positivo de proba 0.1, que é
    # o menor valor da curva; com 1.1 nenhum corte serve e o fallback entra.
    threshold = choose_threshold(y, proba, min_recall=1.1)

    assert threshold == 0.1


def test_oof_temporal_so_preve_com_modelos_do_passado():
    import pandas as pd
    from sklearn.dummy import DummyClassifier

    from src.thresholds import temporal_oof_proba

    # A taxa de positivos sobe com o tempo. Um modelo que só vê o passado prevê
    # a taxa média do passado, sempre abaixo da taxa do bloco que está prevendo.
    n = 600
    X = pd.DataFrame({"t": range(n)})
    y = pd.Series((np.arange(n) % 10 < np.arange(n) // 60).astype(int))

    proba, y_oof = temporal_oof_proba(DummyClassifier(strategy="prior"), X, y, n_splits=5)

    # o primeiro bloco (n // 6 linhas) não tem passado e é descartado
    assert len(proba) == len(y_oof) == n - n // 6
    np.testing.assert_array_equal(y_oof, y.to_numpy()[n // 6:])
    assert proba.max() < y.mean()
