"""Modelo custom (pyfunc) que empaqueta preprocesamiento + estimador (R5).

Garantiza que batch y online usen el mismo artefacto: reciben las features de un cliente
(una fila por cliente y corte) y devuelven la probabilidad de fuga en una columna estable.
"""
from __future__ import annotations

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd

OUTPUT_COLUMN = "probabilidad_fuga"


class ChurnModel(mlflow.pyfunc.PythonModel):
    """Envuelve un Pipeline de scikit-learn guardado como artefacto `sklearn_model`."""

    def load_context(self, context):
        self.pipeline = mlflow.sklearn.load_model(context.artifacts["sklearn_model"])
        preprocess = self.pipeline.named_steps["preprocess"]
        columns = {name: list(cols) for name, _, cols in preprocess.transformers_}
        self.numeric = columns.get("num", [])
        self.categorical = columns.get("cat", [])

    def _validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        expected = self.numeric + self.categorical
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        out = frame[expected].copy()
        for col in self.numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        return out

    def predict(self, context, model_input, params=None):
        frame = self._validate(pd.DataFrame(model_input))
        proba = self.pipeline.predict_proba(frame)[:, 1]
        return pd.DataFrame({OUTPUT_COLUMN: np.asarray(proba, dtype="float64")})
