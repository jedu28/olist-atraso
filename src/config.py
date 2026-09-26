"""Configuração central do projeto: caminhos, constantes e contrato de features.

Tudo que mais de um módulo precisa saber mora aqui, para existir uma única
fonte de verdade (antes a lista de features e o caminho do modelo estavam
duplicados entre treino, avaliação e notebooks).
"""
from pathlib import Path

# --- Caminhos -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
MODEL_PATH = PROCESSED_DIR / "model.joblib"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"

# --- Reprodutibilidade e split ------------------------------------------------
RANDOM_STATE = 42
TEST_FRACTION = 0.2  # os últimos 20% dos pedidos no tempo viram teste
N_TEMPORAL_SPLITS = 5  # folds da TimeSeriesSplit usada para escolher o threshold

# --- Regra de negócio do threshold --------------------------------------------
# Entre os cortes que pegam pelo menos esta fração dos atrasos, usa-se o de
# maior precisão. Perder um atraso é o erro caro; alarme falso é o barato.
MIN_RECALL = 0.5

# --- Contrato de dados --------------------------------------------------------
ID_COLUMN = "order_id"
TIMESTAMP_COLUMN = "order_purchase_timestamp"
TARGET = "atrasado"

NUM_FEATURES = [
    # pedido
    "preco_total",
    "frete_total",
    "n_itens",
    "product_weight_g",
    "payment_installments",
    "payment_value_total",
    "prazo_prometido",
    # geografia
    "estados_diferentes",
    "distancia_km",
    # vendedor (só desfechos conhecidos na data da compra)
    "seller_pedidos_anteriores",
    "seller_taxa_atraso_historica",
    # clima do marketplace
    "clima_atraso_7d",
    "clima_atraso_30d",
    "clima_choque",
    "pressao_demanda",
    # calendário
    "dia_semana_compra",
    "mes_compra",
    "mes_sin",
    "mes_cos",
    "dias_para_natal",
    "semana_black_friday",
]
CAT_FEATURES = ["product_category_name", "customer_state", "seller_state"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# --- Hiperparâmetros ----------------------------------------------------------
XGB_BASE_PARAMS = {
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# V1: ponto de partida, sem busca.
XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
}

# V2+: resultado da busca do Optuna (`src.tuning.tune_xgb`: 30 trials, ROC-AUC
# médio numa TimeSeriesSplit de 5 folds, semente fixa), reproduzida em
# notebooks/02_modelagem.ipynb. Fixados aqui para o treino ser rápido e
# determinístico; rode `make tune` para refazer a busca.
XGB_TUNED_PARAMS = {
    "n_estimators": 276,
    "max_depth": 4,
    "learning_rate": 0.046281728250508816,
    "subsample": 0.6308643418930251,
    "colsample_bytree": 0.8549072272510246,
    "min_child_weight": 6,
    "reg_lambda": 0.0030507643098015153,
    "scale_pos_weight": 5.4397767527678,
}
