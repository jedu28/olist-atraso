import numpy as np
import pandas as pd

from src.artifact import ModelArtifact
from src.config import CAT_FEATURES, FEATURES, NUM_FEATURES
from src.metrics import classification_metrics
from src.modeling import build_pipeline, temporal_split


def _toy_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in NUM_FEATURES})
    for col in CAT_FEATURES:
        df[col] = rng.choice(["SP", "RJ", "MG"], size=n)
    df["order_purchase_timestamp"] = pd.date_range("2018-01-01", periods=n, freq="h")[::-1]
    df["atrasado"] = (df["preco_total"] + rng.normal(scale=0.5, size=n) > 1).astype(int)
    return df


def test_temporal_split_keeps_test_in_the_future():
    train, test = temporal_split(_toy_dataset(), test_fraction=0.25)
    assert len(test) == 50
    assert train["order_purchase_timestamp"].max() <= test["order_purchase_timestamp"].min()


def test_classification_metrics_confusion_matrix_adds_up():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, proba, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["flagged_rate"] == 0.5
    assert m["recall"] == 0.5


def test_artifact_roundtrip_keeps_threshold(tmp_path):
    df = _toy_dataset()
    pipeline = build_pipeline().fit(df[FEATURES], df["atrasado"])
    artifact = ModelArtifact(pipeline=pipeline, threshold=0.3)

    loaded = ModelArtifact.load(artifact.save(tmp_path / "model.joblib"))

    assert loaded.threshold == 0.3
    np.testing.assert_array_equal(loaded.predict(df), (pipeline.predict_proba(df[FEATURES])[:, 1] >= 0.3).astype(int))
