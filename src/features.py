"""Junta as tabelas da Olist e monta o dataset de features + alvo.

Regra de ouro: só entram features que existiriam no momento da compra.
Nada sobre a entrega em si (datas de entrega, transportadora, etc.) pode
vazar para dentro do X — e agregados históricos só podem usar desfechos que
já eram conhecidos naquele instante.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, FEATURES, ID_COLUMN, RAW_DIR, TARGET, TIMESTAMP_COLUMN
from src.data import load_raw

DELIVERED_COLUMN = "order_delivered_customer_date"
ESTIMATED_COLUMN = "order_estimated_delivery_date"


# --- Alvo ---------------------------------------------------------------------
def build_target(orders: pd.DataFrame) -> pd.DataFrame:
    """1 = pedido entregue depois da data estimada, 0 = no prazo.

    Descarta pedidos sem entrega registrada (cancelados, ainda em trânsito
    na data de corte etc.) porque o alvo não existe para eles ainda.
    """
    df = orders.copy()
    for col in (TIMESTAMP_COLUMN, ESTIMATED_COLUMN, DELIVERED_COLUMN):
        df[col] = pd.to_datetime(df[col])

    df = df.dropna(subset=[DELIVERED_COLUMN])
    df[TARGET] = (df[DELIVERED_COLUMN] > df[ESTIMATED_COLUMN]).astype(int)
    return df


# --- Agregações por pedido ----------------------------------------------------
def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pedido. Vendedor e produto são os do primeiro item."""
    return (
        items.groupby(ID_COLUMN)
        .agg(
            preco_total=("price", "sum"),
            frete_total=("freight_value", "sum"),
            n_itens=("order_item_id", "count"),
            seller_id=("seller_id", "first"),
            product_id=("product_id", "first"),
        )
        .reset_index()
    )


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    return (
        payments.groupby(ID_COLUMN)
        .agg(
            payment_installments=("payment_installments", "max"),
            payment_value_total=("payment_value", "sum"),
        )
        .reset_index()
    )


# --- Geografia ----------------------------------------------------------------
def build_zip_coords(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Uma lat/lng por prefixo de CEP (média, já que o raw tem várias por prefixo)."""
    return (
        geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )


def haversine_km(lat1, lng1, lat2, lng2):
    """Distância em km sobre a superfície da Terra (aceita arrays)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def add_geo_features(df: pd.DataFrame, zip_coords: pd.DataFrame) -> pd.DataFrame:
    """Distância cliente–vendedor e se o pedido cruza fronteira estadual."""
    for party in ("customer", "seller"):
        coords = zip_coords.rename(
            columns={
                "zip_code_prefix": f"{party}_zip_code_prefix",
                "geolocation_lat": f"{party}_lat",
                "geolocation_lng": f"{party}_lng",
            }
        )
        df = df.merge(coords, on=f"{party}_zip_code_prefix", how="left")

    df["distancia_km"] = haversine_km(df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"])
    df["estados_diferentes"] = (df["customer_state"] != df["seller_state"]).astype(int)
    return df


# --- Histórico do vendedor ----------------------------------------------------
def add_seller_history(df: pd.DataFrame) -> pd.DataFrame:
    """Volume e taxa de atraso do vendedor com os desfechos conhecidos na data da compra.

    Um pedido anterior só entra na conta depois de **entregue**: é na entrega
    que se descobre se ele atrasou. Contar todos os pedidos *comprados* antes
    (a versão anterior desta função) vazava o desfecho de pedidos que ainda
    estavam em trânsito — 7,6% dos pares de pedidos do mesmo vendedor.
    """
    df = df.sort_values(TIMESTAMP_COLUMN).copy()

    desfechos = df[["seller_id", DELIVERED_COLUMN, TARGET]].dropna().sort_values(DELIVERED_COLUMN)
    por_vendedor = desfechos.groupby("seller_id")[TARGET]
    desfechos["entregas"] = por_vendedor.cumcount() + 1
    desfechos["atrasos"] = por_vendedor.cumsum()

    # Para cada compra, o último desfecho do mesmo vendedor estritamente antes dela.
    historico = pd.merge_asof(
        df[[TIMESTAMP_COLUMN, "seller_id"]],
        desfechos[[DELIVERED_COLUMN, "seller_id", "entregas", "atrasos"]],
        left_on=TIMESTAMP_COLUMN,
        right_on=DELIVERED_COLUMN,
        by="seller_id",
        allow_exact_matches=False,
    )
    entregas = historico["entregas"].fillna(0).to_numpy()
    atrasos = historico["atrasos"].fillna(0).to_numpy()

    df["seller_pedidos_anteriores"] = entregas.astype(int)
    df["seller_taxa_atraso_historica"] = np.divide(
        atrasos, entregas, out=np.zeros_like(atrasos, dtype=float), where=entregas > 0
    )
    return df


# --- Clima do marketplace -----------------------------------------------------
def _window_stats(event_times: np.ndarray, values: np.ndarray, query_times: np.ndarray, window: pd.Timedelta):
    """(média, contagem) de `values` cujos eventos caem em [query - window, query).

    `event_times` precisa estar ordenado.
    """
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.searchsorted(event_times, query_times, side="left")
    lo = np.searchsorted(event_times, query_times - window, side="left")

    count = (hi - lo).astype(float)
    total = cumsum[hi] - cumsum[lo]
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return mean, count


def add_marketplace_climate(df: pd.DataFrame) -> pd.DataFrame:
    """Sinais agregados do marketplace que revelam choques externos (greves,
    enchentes, picos de demanda) e o ciclo sazonal da operação.

    Como no histórico do vendedor, um pedido passado só entra na conta depois
    de entregue. Agregar por data de compra faria a feature enxergar desfechos
    que ainda não existiam no momento da previsão.
    """
    df = df.sort_values(TIMESTAMP_COLUMN).copy()
    compras = df[TIMESTAMP_COLUMN].to_numpy()

    desfechos = df[[DELIVERED_COLUMN, TARGET]].dropna().sort_values(DELIVERED_COLUMN)
    conhecido_em = desfechos[DELIVERED_COLUMN].to_numpy()
    atrasos = desfechos[TARGET].to_numpy(dtype=float)

    taxa_7d, _ = _window_stats(conhecido_em, atrasos, compras, pd.Timedelta("7D"))
    taxa_30d, _ = _window_stats(conhecido_em, atrasos, compras, pd.Timedelta("30D"))

    df["clima_atraso_7d"] = taxa_7d
    df["clima_atraso_30d"] = taxa_30d
    # Acima de 1 = a operação está piorando agora em relação ao próprio mês.
    # É essa razão que denuncia um choque externo enquanto ele acontece, antes
    # de ele diluir na média de 30 dias.
    df["clima_choque"] = np.divide(taxa_7d, taxa_30d, out=np.full_like(taxa_7d, np.nan), where=taxa_30d > 0)

    # Volume de compras não depende de desfecho, então pode ser contado na hora:
    # é o que capta surto de demanda (Black Friday) antes de virar atraso.
    zeros = np.zeros(len(compras))
    _, vol_7d = _window_stats(compras, zeros, compras, pd.Timedelta("7D"))
    _, vol_90d = _window_stats(compras, zeros, compras, pd.Timedelta("90D"))
    base_7d = vol_90d / (90 / 7)
    df["pressao_demanda"] = np.divide(vol_7d, base_7d, out=np.full_like(vol_7d, np.nan), where=base_7d > 0)

    return df


# --- Calendário ---------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sazonalidade do calendário em forma que o modelo consegue usar.

    Mês como inteiro coloca dezembro e janeiro nos extremos opostos da escala;
    seno/cosseno devolvem a circularidade do ano. A distância até o Natal e a
    janela da Black Friday marcam os picos que esticam os prazos de entrega.
    """
    df = df.copy()
    ts = df[TIMESTAMP_COLUMN]

    mes = ts.dt.month
    df["dia_semana_compra"] = ts.dt.dayofweek
    df["mes_compra"] = mes
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12)

    natal = pd.to_datetime(dict(year=ts.dt.year, month=12, day=25))
    dias_para_natal = (natal - ts).dt.days
    df["dias_para_natal"] = dias_para_natal.where(dias_para_natal >= 0, dias_para_natal + 365)

    df["semana_black_friday"] = ((mes == 11) & ts.dt.day.between(20, 30)).astype(int)
    return df


def add_promised_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """Prazo, em dias, que a Olist prometeu ao cliente na hora da compra.

    Não é vazamento: a data estimada é mostrada no checkout; o que não se sabe
    é a data real de entrega, que é a outra metade do alvo. Prazo folgado é
    mais fácil de cumprir, e a Olist alarga a estimativa em época de pico —
    então essa feature também carrega a leitura que a operação faz da própria carga.
    """
    df = df.copy()
    df["prazo_prometido"] = (df[ESTIMATED_COLUMN] - df[TIMESTAMP_COLUMN]).dt.days
    return df


# --- Montagem -----------------------------------------------------------------
def build_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Dataset final: uma linha por pedido entregue, ordenado pela data da compra."""
    raw = load_raw(raw_dir)

    df = (
        build_target(raw["orders"])
        .merge(aggregate_items(raw["items"]), on=ID_COLUMN, how="inner")
        .merge(raw["products"][["product_id", "product_category_name", "product_weight_g"]], on="product_id", how="left")
        .merge(raw["customers"][["customer_id", "customer_state", "customer_zip_code_prefix"]], on="customer_id", how="left")
        .merge(raw["sellers"][["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id", how="left")
        .merge(aggregate_payments(raw["payments"]), on=ID_COLUMN, how="left")
    )

    df = add_geo_features(df, build_zip_coords(raw["geolocation"]))
    df = add_promised_lead_time(df)
    df = add_seller_history(df)
    df = add_marketplace_climate(df)
    df = add_calendar_features(df)

    return df[[ID_COLUMN, TIMESTAMP_COLUMN, *FEATURES, TARGET]].sort_values(TIMESTAMP_COLUMN)


def main() -> None:
    dataset = build_dataset()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"Dataset salvo em {DATASET_PATH} — {len(dataset):,} pedidos, taxa de atraso {dataset[TARGET].mean():.1%}")


if __name__ == "__main__":
    main()
