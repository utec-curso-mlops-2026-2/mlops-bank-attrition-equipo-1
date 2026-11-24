# Ejemplos de request y response (contrato de inferencia online, R8)

Los ejemplos se generaron con la **signature real** del modelo `bank-attrition-churn` (champion v1): 58 features,
57 numéricas y 1 categórica. Los valores son **sintéticos**, no pertenecen a ningún cliente: cada variable numérica
toma la mediana de entrenamiento y la categórica, la moda. Las respuestas son las que devolvió el contenedor real
(`python -m bank_attrition.container serve`) con el champion v1.

| Archivo | Qué muestra |
|---|---|
| `request.json` | Solicitud válida: `ID_CORRELATIVO` y `CODMES` identifican al cliente y no entran al modelo. `features` trae las 58 columnas de la signature |
| `response.json` | 200: `probabilidad_fuga`, `modelo_nombre`, `modelo_version`, `modelo_alias`, `fecha_prediccion` (UTC) y `request_id` |
| `request_invalido.json` | Le falta `EDAD` y trae `ANTIGUEDAD` con texto |
| `response_invalido.json` | 422 `entrada_invalida`, con **todos** los problemas en `detalle` |

## Reglas de validación (`src/bank_attrition/serve.py`)
- `Content-Type: application/json`. Para Batch Transform se usa `application/jsonlines`, con una línea por cliente y
  las features al primer nivel. Cualquier otro tipo responde **415**.
- Una solicitud lleva **un cliente**, o hasta 100 con `{"instances": [ {...}, {...} ]}`; en ese caso la respuesta
  es `{"predicciones": [...]}`.
- Deben estar **todas** las features de la signature. Un valor `null` está permitido: lo imputa el pipeline del
  modelo. Las numéricas aceptan números o texto numérico (`"30"`), pero no texto libre ni booleanos.
- Un JSON mal formado o una feature faltante o no numérica responde **422** con el detalle. Un error interno
  responde **500** con el `request_id`, sin mostrar trazas.
- La lista de columnas se obtiene con `python -m bank_attrition.registry contract`. Las features de requerimientos
  (`N_REQUERIMIENTOS_VENTANA`, `REQ_*`) se calculan con `bank_attrition.features.build_features` y el `feature_spec`
  del modelo, igual que en el batch.

## Probarlo en local
Hacen falta el perfil SSO de AWS y el servidor MLflow encendido. El contenedor lee la contraseña de MLflow desde SSM.
```bash
export MLFLOW_TRACKING_URI=http://50.16.69.79:5000 MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_AUTH_SSM_PARAMETER=/equipo1/mlflow/admin-password PORT=8080
python -m bank_attrition.container serve
curl -s -X POST localhost:8080/invocations -H "Content-Type: application/json" --data-binary @examples/request.json
```
