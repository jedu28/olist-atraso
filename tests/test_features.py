import numpy as np
import pandas as pd

from src.features import add_calendar_features, add_seller_history, build_target, haversine_km


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


def _seller_orders(purchases, deliveries, late, seller="a"):
    return pd.DataFrame(
        {
            "seller_id": [seller] * len(purchases),
            "order_purchase_timestamp": pd.to_datetime(purchases),
            "order_delivered_customer_date": pd.to_datetime(deliveries),
            "atrasado": late,
        }
    )


def test_seller_history_only_uses_past_orders():
    df = _seller_orders(
        purchases=["2018-01-01", "2018-01-10", "2018-01-20"],
        deliveries=["2018-01-05", "2018-01-15", "2018-01-25"],
        late=[1, 1, 0],
    )
    result = add_seller_history(df)
    # no primeiro pedido do vendedor não há histórico
    assert result.iloc[0]["seller_pedidos_anteriores"] == 0
    assert result.iloc[0]["seller_taxa_atraso_historica"] == 0.0
    # no terceiro pedido, o histórico é feito só dos 2 anteriores (ambos atrasados)
    assert result.iloc[2]["seller_pedidos_anteriores"] == 2
    assert result.iloc[2]["seller_taxa_atraso_historica"] == 1.0


def test_seller_history_ignores_orders_not_yet_delivered():
    # O pedido de 01/01 atrasou, mas só foi entregue em 20/01: na compra de
    # 10/01 esse desfecho ainda não existia e não pode entrar no histórico.
    df = _seller_orders(
        purchases=["2018-01-01", "2018-01-10", "2018-01-25"],
        deliveries=["2018-01-20", "2018-01-12", "2018-01-30"],
        late=[1, 0, 0],
    )
    result = add_seller_history(df)
    assert result.iloc[1]["seller_pedidos_anteriores"] == 0
    # Em 25/01 os dois primeiros já foram entregues (1 atraso em 2).
    assert result.iloc[2]["seller_pedidos_anteriores"] == 2
    assert result.iloc[2]["seller_taxa_atraso_historica"] == 0.5


def test_seller_history_is_per_seller():
    df = pd.concat(
        [
            _seller_orders(["2018-01-01"], ["2018-01-02"], [1], seller="a"),
            _seller_orders(["2018-01-10"], ["2018-01-12"], [0], seller="b"),
        ]
    )
    result = add_seller_history(df).set_index("seller_id")
    assert result.loc["b", "seller_pedidos_anteriores"] == 0


def test_calendar_features_wrap_around_new_year():
    df = pd.DataFrame({"order_purchase_timestamp": pd.to_datetime(["2017-12-24", "2017-12-26", "2017-11-24"])})
    result = add_calendar_features(df)
    assert result["dias_para_natal"].tolist() == [1, 364, 31]
    assert result["semana_black_friday"].tolist() == [0, 0, 1]
    # dezembro e janeiro ficam vizinhos na representação circular
    dez, jan = np.array([result.iloc[0]["mes_sin"], result.iloc[0]["mes_cos"]]), np.array([0.5, np.sqrt(3) / 2])
    assert np.linalg.norm(dez - jan) < 0.6


def test_haversine_known_distance():
    # São Paulo -> Rio de Janeiro ≈ 360 km
    assert 340 < haversine_km(-23.55, -46.63, -22.91, -43.17) < 380
