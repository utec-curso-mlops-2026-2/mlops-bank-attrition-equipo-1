# R7 · Inferencia batch en AWS con SageMaker Batch Transform (+ R9 data drift)

Cada mes, el área de retención recibe en S3 la lista de clientes **ordenada por probabilidad de fuga**, con decil y
segmento de riesgo. El proceso es un **SageMaker Pipeline** de tres pasos que se lanza solo, con
**EventBridge Scheduler**. Es el mismo patrón que la sesión 11 del curso: imagen en ECR, Batch Transform, Scheduler
y CloudWatch.

```text
EventBridge Scheduler  equipo1-batch-mensual   cron(0 6 1 * ? *) America/Lima  (día 1, 06:00)
        │  sagemaker:StartPipelineExecution (rol equipo1-scheduler-batch-role)
        ▼
SageMaker Pipeline  equipo1-bank-attrition-batch   (parámetros: Release, ModelAlias, tipos de instancia)
  1. PrepararFeatures     Processing  ml.t3.medium   container prepare
       MLflow Registry: alias champion → versión N + feature_spec
       raw/<yyyymm>/ (S3) → features del mes → batch/<yyyymm>/<ejecución>/input/registros.jsonl + manifest.json
  2. ScorearTransform     Batch Transform  ml.m5.large   container serve (SplitType=Line, MultiRecord)
       carga models:/bank-attrition-churn@champion → output/registros.jsonl.out
  3. PublicarScoresDrift  Processing  ml.t3.medium   container publish
       valida salida vs. manifiesto → deciles y segmentos sobre el mes completo
       → scores/<yyyymm>/scores_<yyyymm>_<yyyymmdd>_v<N>.csv
       → monitoring/<yyyymm>/drift_<yyyymm>_<yyyymmdd>_v<N>.{csv,json}   (PSI, R9)
       → run de MLflow  batch-<yyyymm>-<fecha>  (tag stage=batch_inference)
Logs: CloudWatch /aws/sagemaker/ProcessingJobs y /aws/sagemaker/TransformJobs
```

## Por qué tres pasos y no solo Batch Transform
Batch Transform parte la entrada en mini-lotes y cada llamada al contenedor solo ve una parte del mes. El trabajo
necesita tres cosas que no caben en eso:

| Necesidad | Paso |
|---|---|
| Juntar **dos** archivos crudos (clientes + requerimientos, ventana de 6 meses). El Transform recibe una sola entrada | 1 · Preparar (equivale al snapshot de Athena de la S11) |
| Decil **relativo** y segmento. Dependen del mes completo | 3 · Publicar |
| PSI de variables y score vs. entrenamiento (R9), y run de MLflow | 3 · Publicar |

Los tres pasos usan **la misma imagen** y el mismo código que el pipeline de entrenamiento: `build_features` con el
`feature_spec` guardado en el modelo, que no se recalcula.

## Trazabilidad y controles
- El modelo **siempre** se carga del Registry de MLflow por alias. El paso 1 fija la versión en `manifest.json`. El
  paso 3 **falla** si la salida del Transform trae otra versión (por ejemplo, si el alias cambió a mitad del lote),
  si faltan o sobran clientes, o si hay probabilidades fuera de [0, 1].
- Cada ejecución escribe en su propio prefijo `batch/<yyyymm>/<id de ejecución>/`, así que los reintentos no pisan
  lotes anteriores. El bucket tiene versionado.
- **Sin secretos** en la imagen ni en el pipeline: el contenedor lee la contraseña de MLflow de SSM
  (`/equipo1/mlflow/admin-password`) con su rol.
- El CSV de scores trae `fecha_score`, `modelo_nombre`, `modelo_version` y `modelo_alias`. El run de MLflow guarda el
  manifiesto, el resumen por decil, el reporte de drift y la distribución del score.

## Data drift (R9)
PSI por variable (bins por cuantiles de entrenamiento y un bin propio para nulos) y del score. Semáforo:
**< 0,10 estable · 0,10–0,25 alerta · > 0,25 crítico** (`config.yaml → monitoring`). La referencia es la población de
entrenamiento, transformada y scoreada con la misma versión del modelo. El reporte se guarda en S3 (CSV + JSON con
resumen) y en MLflow (métricas `psi_*`, tabla `monitoring/drift_report.json`, tag `drift_score`).

## Recursos AWS
| Recurso | Nombre | Definición |
|---|---|---|
| Repositorio ECR (tags inmutables, escaneo al subir, conserva 10 imágenes) | `equipo1-bank-attrition` | `cd.yml` construye `:<commit>` |
| Rol de ejecución SageMaker | `equipo1-sagemaker-execution-role` | [`infra/sagemaker/`](../infra/sagemaker/) |
| Rol de Scheduler | `equipo1-scheduler-batch-role` | [`infra/sagemaker/`](../infra/sagemaker/) |
| SageMaker Model (uno por commit) | `equipo1-bank-attrition-<commit>` | `aws_batch.ensure_model` |
| SageMaker Pipeline | `equipo1-bank-attrition-batch` | `aws_batch.pipeline_definition` |
| Schedule mensual | `equipo1-batch-mensual` | `aws_batch.upsert_schedule` |
| Permisos de despliegue del rol OIDC | política `equipo1-github-actions-deploy` | [`infra/github-actions/iam-policy-deploy.json`](../infra/github-actions/iam-policy-deploy.json) |

**Costo**: se paga solo mientras corren los jobs, unos 10–15 minutos por ejecución en `ml.t3.medium` y
`ml.m5.large`, menos de USD 0,10. No queda nada encendido entre ejecuciones.

## Operación
Despliegue: automático con `cd.yml` al integrar a `master`, o a mano desde Actions → *Despliegue* → *Run workflow*.
Desde una terminal con el perfil SSO del equipo:
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::654654589924:role/equipo1-sagemaker-execution-role
export SCHEDULER_ROLE_ARN=arn:aws:iam::654654589924:role/equipo1-scheduler-batch-role
export MLOPS_DATA_BUCKET=equipo-1-mlops-bank-attrition-654654589924 MLFLOW_TRACKING_URI=http://50.16.69.79:5000
python -m bank_attrition.aws_batch run --wait                 # ejecución manual
python -m bank_attrition.aws_batch schedule-once --minutes 5  # ejecución automática de evidencia (one-time)
python -m bank_attrition.aws_batch status                     # estado de la última ejecución
```
En la consola: **SageMaker AI → Pipelines → equipo1-bank-attrition-batch** (grafo y ejecuciones),
**Batch transform jobs / Processing jobs**, **EventBridge → Scheduler → Schedules** y **CloudWatch → Log groups**.

## Evidencias
Ver [evidencias.md](evidencias.md).
