# Decisiones técnicas

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

## R6–R9 (CI/CD, inferencia y monitoreo)
| Decisión | Motivo | Alternativas / límites |
|---|---|---|
| GitHub Actions entra a AWS por **OIDC** (rol `equipo1-github-actions-role`) | Sin claves de AWS en el repo; credenciales de 1 h | Claves IAM en Secrets (descartado). La confianza del rol **no se puede editar** con nuestro permiso: si cambia, se recrea el rol |
| `train.yml` manual (`workflow_dispatch`) | Entrenar es una decisión, no un efecto de cada push; el gate y los alias deciden la promoción | Disparo por cambio de datos o schedule (bonus de reentrenamiento) |
| **Batch con SageMaker Batch Transform** dentro de un SageMaker Pipeline de 3 pasos (preparar → Transform → publicar), con EventBridge Scheduler y CloudWatch | Es el patrón visto en clase (S11). El Transform ve mini-lotes de una sola entrada: el cruce clientes + requerimientos va antes; los deciles relativos, el PSI y el run de MLflow necesitan el mes completo y van después | Lambda + EventBridge o un único Processing Job (más simples, pero se alejan de lo visto en clase) |
| **Una sola imagen** Docker (ECR) para Processing, Transform y endpoint | El mismo código y las mismas features en batch y online | Una imagen por componente |
| El modelo **no** va dentro de la imagen: el contenedor lo carga del Registry de MLflow por alias al arrancar | Lo pide R5 y permite cambiar de champion sin reconstruir la imagen | Empaquetar `model.tar.gz` (como en los labs), que perdería la carga por alias |
| Contraseña de MLflow leída de **SSM** por el contenedor (`MLFLOW_AUTH_SSM_PARAMETER`) | No viaja en la imagen, en el pipeline ni en el Model. SageMaker además enmascara las variables cuyo nombre contiene `PASSWORD` | Pasarla como variable de entorno (no funciona en SageMaker y la expone) |
| PSI con la población de **entrenamiento** como referencia, transformada y scoreada con la misma versión del modelo | La referencia es lo que el modelo "conoce". Semáforo del curso: 0,10 / 0,25 | Referencia móvil (mes anterior) cuando haya varios meses de aplicación |
| Feature Store + Athena (S10): **bonus, no implementado** | R3 lo marca como opcional. El paso 1 del pipeline está aislado para reemplazarlo por una consulta Athena | Implementarlo si sobra tiempo |
