import pandas as pd

from src.features import add_seller_history, build_target


def test_build_target_marks_late_orders():
    orders = pd.DataFrame(
        {
            "order_id": ["1", "2"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-01"],
            "order_estimated_delivery_date": ["2018-01-10", "2018-01-10"],
            "order_delivered_customer_date": ["2018-01-15", "2018-01-05"],
        }
    )
    result = build_target(orders)
    assert result.set_index("order_id")["atrasado"].to_dict() == {"1": 1, "2": 0}


def test_build_target_drops_undelivered_orders():
    orders = pd.DataFrame(
        {
            "order_id": ["1"],
            "order_purchase_timestamp": ["2018-01-01"],
            "order_estimated_delivery_date": ["2018-01-10"],
            "order_delivered_customer_date": [None],
        }
    )
    result = build_target(orders)
    assert len(result) == 0


def test_seller_history_only_uses_past_orders():
    df = pd.DataFrame(
        {
            "seller_id": ["a", "a", "a"],
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03"]),
            "atrasado": [1, 1, 0],
        }
    )
    result = add_seller_history(df)
    # no primeiro pedido do vendedor não há histórico
    assert result.iloc[0]["seller_pedidos_anteriores"] == 0
    assert result.iloc[0]["seller_taxa_atraso_historica"] == 0.0
    # no terceiro pedido, o histórico é feito só dos 2 anteriores (ambos atrasados)
    assert result.iloc[2]["seller_pedidos_anteriores"] == 2
    assert result.iloc[2]["seller_taxa_atraso_historica"] == 1.0
