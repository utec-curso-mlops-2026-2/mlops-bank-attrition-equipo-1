"""Pipelines de scikit-learn: el preprocesamiento viaja con el modelo (R5)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def build_pipeline(
    name: str, params: dict, numeric: list[str], categorical: list[str], seed: int
) -> Pipeline:
    """Imputación + escalado/codificación + estimador. Se ajusta solo con train y se reutiliza igual."""
    if name not in ESTIMATORS:
        raise KeyError(f"Modelo desconocido: {name}. Opciones: {sorted(ESTIMATORS)}")
    # add_indicator conserva la señal de los nulos (p. ej. ANTIGUEDAD nula => 76 % de fuga)
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if name == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(numeric_steps), numeric)]
    if categorical:
        cat_steps = [
            ("impute", SimpleImputer(strategy="constant", fill_value="NULO")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    estimator = ESTIMATORS[name](**{**params, "random_state": seed})
    return Pipeline([("preprocess", preprocess), ("model", estimator)])
