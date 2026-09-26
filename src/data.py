"""Leitura dos CSVs brutos do dataset público da Olist (Kaggle)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Carrega as tabelas usadas pelo projeto, indexadas por um nome curto."""
    missing = [fname for fname in RAW_FILES.values() if not (raw_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"CSVs da Olist ausentes em {raw_dir}: {', '.join(missing)}. "
            "Baixe em https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce."
        )
    return {name: pd.read_csv(raw_dir / fname) for name, fname in RAW_FILES.items()}
