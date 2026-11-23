# Contrato del modelo (para batch y API: R7, R8)

- **Registry**: `bank-attrition-churn`; cargar siempre por alias: `models:/bank-attrition-churn@champion`.
- **Tipo**: pyfunc custom; el preprocesamiento (imputación, escalado, one-hot) viaja dentro del modelo.
- **Entrada**: DataFrame con **una fila por cliente y corte** y las columnas de la signature (`python -m bank_attrition.registry contract`).
  Las features de requerimientos las calcula `bank_attrition.features.build_features(clientes, requerimientos, spec, cfg)`
  con `spec = contract["metadata"]["feature_spec"]` (se guarda con el modelo). Batch y API deben usar esa misma función.
  `ID_CORRELATIVO` y `CODMES` son metadatos: no entran al modelo.
- **Salida**: DataFrame con la columna `probabilidad_fuga` (float en [0, 1]).
- **Decil/segmento**: `bank_attrition.metrics.assign_decile(score)` (1 = menor riesgo, 10 = mayor).
- **Versión**: `mlflow.MlflowClient().get_model_version_by_alias(name, "champion").version` para reportarla en cada salida.
- **Errores**: columnas faltantes o con tipo inválido hacen fallar `predict` con un mensaje de esquema; la API debe validar antes y responder 4xx.

```python
import mlflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.pyfunc.load_model("models:/bank-attrition-churn@champion")
scores = model.predict(features_df)["probabilidad_fuga"]
```
El entorno del modelo (`requirements.txt`/`conda.yaml`) se guarda junto al artefacto; instalar `mlflow==3.18.0` en batch/API.
