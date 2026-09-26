"""Artefato do modelo: pipeline + threshold + metadados, salvos juntos.

O threshold faz parte do modelo tanto quanto os pesos: o mesmo pipeline com
corte 0.5 ou 0.15 é outro classificador. Salvar os dois separados (antes o
corte era passado à mão na linha de comando) abre espaço para servir o
pipeline certo com o corte errado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURES, MODEL_PATH


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    threshold: float
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 = sinalizar como risco de atraso, usando o threshold salvo."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("saved_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> ModelArtifact:
        if not path.exists():
            raise FileNotFoundError(f"Nenhum modelo salvo em {path}. Rode `make train` primeiro.")
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError(f"{path} não contém um ModelArtifact (encontrado: {type(artifact).__name__}).")
        return artifact
