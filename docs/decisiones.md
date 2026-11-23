# Decisiones técnicas (R1–R5)

| Decisión | Motivo | Alternativas / límites |
|---|---|---|
| Ventana de 6 meses de requerimientos hasta el corte | Coincide con la cobertura real de los datos (mes -5 a 0) y evita mirar el futuro | Ventanas más cortas (aportan poco: los requerimientos casi no mejoran el modelo) |
| Corregir CODMES 201208 → 202408 | Error de codificación evidenciado en el EDA | No corregir (perdería todos los requerimientos del mes 0); **pendiente de confirmación** |
| Conservar requerimientos repetidos | Compatibles con reclamos reiterados; efecto en AUC ±0,001 | Eliminarlos (`drop_duplicate_requirements: true`) |
| Indicador de nulos + categoría NULO | ANTIGUEDAD nula = 76 % de fuga | Imputar sin indicador (pierde señal); excluir la variable (AUC 0,849 → 0,823) |
| Imputación y escalado dentro del Pipeline de sklearn | Se ajusta solo con train y viaja con el modelo | Imputar fuera del modelo (riesgo de inconsistencia batch/online) |
| 3 modelos: regresión logística, random forest, gradient boosting | Baseline interpretable + dos no lineales | XGBoost/LightGBM (más dependencias) |
| Selección por AUC medio en CV sobre train | Test queda como evaluación independiente | Seleccionar por KS o captura top 20 % |
| Gate: AUC≥0,80, KS≥0,45, captura top 20 %≥0,55 | Criterio explícito antes de promover; calibrado con el EDA (modelo base ≈ 0,85 / 0,54 / 0,63) con margen | Subir el umbral si el equipo mejora el modelo |
| Modelo pyfunc custom con salida `probabilidad_fuga` | Salida estable para batch y API | `mlflow.sklearn` estándar (devuelve etiquetas) |
| Serialización cloudpickle del Pipeline | skops rechaza tipos de numpy en esta versión | Fijar `skops_trusted_types` |
| MLflow en EC2 + S3 (`--serve-artifacts`) | Servidor compartido sin credenciales de S3 en clientes | Databricks Free (un solo usuario), servicio administrado de la UTEC |
| Feature Store | Opcional; no iniciado | SageMaker Feature Store o tabla de features en S3 versionada (`feature_release` en el manifiesto) |
| OOT sin métricas de desempeño | No tiene target; solo distribución del score (insumo para drift, R9) | — |
