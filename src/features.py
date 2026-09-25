"""Junta as tabelas da Olist e monta o dataset de features + alvo.

Regra de ouro: só entram features que existiriam no momento da compra.
Nada sobre a entrega em si (datas de entrega, transportadora, etc.) pode
vazar para dentro do X.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    files = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
    }
    return {name: pd.read_csv(raw_dir / fname) for name, fname in files.items()}


def build_target(orders: pd.DataFrame) -> pd.DataFrame:
    """1 = pedido entregue depois da data estimada, 0 = no prazo.

    Descarta pedidos sem entrega registrada (cancelados, ainda em trânsito
    na data de corte etc.) porque o alvo não existe para eles ainda.
    """
    df = orders.copy()
    for col in ("order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"):
        df[col] = pd.to_datetime(df[col])

    df = df.dropna(subset=["order_delivered_customer_date"])
    df["atrasado"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)
    return df


def add_seller_history(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de atraso histórica do vendedor, olhando só para pedidos
    anteriores à compra atual (expanding window ordenada no tempo).

    Sem isso, usar a taxa de atraso do vendedor "geral" vazaria informação
    do futuro (inclusive do próprio pedido) para o passado.
    """
    df = df.sort_values("order_purchase_timestamp").copy()
    grouped = df.groupby("seller_id")["atrasado"]
    df["seller_pedidos_anteriores"] = grouped.cumcount()
    cum_atrasos = grouped.cumsum() - df["atrasado"]
    df["seller_taxa_atraso_historica"] = (cum_atrasos / df["seller_pedidos_anteriores"]).fillna(0.0)
    return df


def build_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    raw = load_raw(raw_dir)

    orders = build_target(raw["orders"])

    items = (
        raw["items"]
        .groupby("order_id")
        .agg(
            preco_total=("price", "sum"),
            frete_total=("freight_value", "sum"),
            n_itens=("order_item_id", "count"),
            seller_id=("seller_id", "first"),
            product_id=("product_id", "first"),
        )
        .reset_index()
    )

    products = raw["products"][["product_id", "product_category_name", "product_weight_g"]]
    customers = raw["customers"][["customer_id", "customer_state"]]
    sellers = raw["sellers"][["seller_id", "seller_state"]]

    df = (
        orders.merge(items, on="order_id", how="inner")
        .merge(products, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left")
    )

    df["estados_diferentes"] = (df["customer_state"] != df["seller_state"]).astype(int)
    df["dia_semana_compra"] = df["order_purchase_timestamp"].dt.dayofweek
    df["mes_compra"] = df["order_purchase_timestamp"].dt.month

    df = add_seller_history(df)

    features = [
        "preco_total",
        "frete_total",
        "n_itens",
        "product_category_name",
        "product_weight_g",
        "customer_state",
        "seller_state",
        "estados_diferentes",
        "dia_semana_compra",
        "mes_compra",
        "seller_pedidos_anteriores",
        "seller_taxa_atraso_historica",
    ]
    cols = ["order_id", "order_purchase_timestamp", *features, "atrasado"]
    return df[cols]


if __name__ == "__main__":
    dataset = build_dataset()
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    print(f"Dataset salvo em {out_path} — {len(dataset)} linhas, taxa de atraso: {dataset['atrasado'].mean():.3f}")
