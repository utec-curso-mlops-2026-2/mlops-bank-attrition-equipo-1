# Evidencias

Enlaces verificables por requerimiento. Las fechas están en UTC. El servidor MLflow es `http://50.16.69.79:5000`
(usuario y contraseña en SSM).

## R6 · CI/CD
| Evidencia | Enlace / identificador |
|---|---|
| Entrenamiento automatizado (`train.yml`): tests → train (OIDC, S3, MLflow remoto, Registry) → smoke-score | [run 45338045572](https://github.com/utec-curso-mlops-2026-2/mlops-bank-attrition-equipo-1/actions/runs/45338045572), que registró la v2 como `challenger` |
| CD (`cd.yml`): tests → imagen en ECR → Model + Pipeline + schedule → ejecución completa del pipeline | [run 45407955766](https://github.com/utec-curso-mlops-2026-2/mlops-bank-attrition-equipo-1/actions/runs/45407955766) (24/11/2026, lanzado a mano desde `feature/batch-transform`) |

## R7 · Inferencia batch en AWS (SageMaker Batch Transform)
**Ejecución manual** (la prueba posterior del CD), `i7b7vqvjti2u`, del 24/11/2026: **Succeeded**.

| Paso | Job | Resultado |
|---|---|---|
| PrepararFeatures (Processing) | `pipelines-i7b7vqvjti2u-PrepararFeatures-wafJlYnxVa` | 14.000 clientes → `batch/202408/i7b7vqvjti2u/input/registros.jsonl` y `manifest.json` (champion **v1**) |
| ScorearTransform (Batch Transform) | `pipelines-i7b7vqvjti2u-ScorearTransform-RltlXfD9Oe` | `batch/202408/i7b7vqvjti2u/output/registros.jsonl.out` |
| PublicarScoresDrift (Processing) | `pipelines-i7b7vqvjti2u-PublicarScoresDrift-nc7J2pKof1` | `scores/202408/scores_202408_20261124_v1.csv` y `monitoring/202408/drift_202408_20261124_v1.{csv,json}` |

- **MLflow**: run `batch-202408-20261124` (`78b3b8f4555c4293ab0e492a07dda422`, tag `stage=batch_inference`).
  - 14.000 clientes; score medio 0,157, p90 0,445.
  - Segmentos Alto/Medio/Bajo: 20 % / 30 % / 50 %.
- **Logs**: CloudWatch `/aws/sagemaker/ProcessingJobs` y `/aws/sagemaker/TransformJobs`, con streams `pipelines-i7b7vqvjti2u-*`.

**Ejecución automática** con EventBridge Scheduler: `489clq7fmntx`, del 24/11/2026, **Succeeded**.
- La disparó el schedule de una sola vez `equipo1-batch-mensual-evidencia-202611241413`, a las 14:13 UTC.
- Duró de 14:13:08 a 14:21:38 UTC, unos 8,5 minutos.
- SageMaker registra como creador a `assumed-role/equipo1-scheduler-batch-role`: **la inició el Scheduler, no una
  persona ni el CD**.
- Los tres pasos terminaron en `Succeeded`: `pipelines-489clq7fmntx-PrepararFeatures-zgGFUzxSeG`,
  `pipelines-489clq7fmntx-ScorearTransform-alyFRLkM6O` y `pipelines-489clq7fmntx-PublicarScoresDrift-kOANrylXVt`.
- Escribió el mismo CSV del día (`scores_202408_20261124_v1.csv`). El bucket tiene versionado, así que las dos
  ejecuciones quedan como versiones distintas del objeto. Los intermedios de cada una viven en su propio prefijo
  `batch/202408/<ejecución>/`.
- La programación recurrente es el schedule `equipo1-batch-mensual`: `cron(0 6 1 * ? *)`, America/Lima, estado
  `ENABLED`.

## R9 · Data drift (paso del batch)
Reporte `monitoring/202408/drift_202408_20261124_v1.json`:
- PSI del score: **0,0007, estable**.
- 58 variables: todas estables. La de mayor PSI es `NRO_ENTID_SSFF_MENOS2`, con 0,0008.
- Lectura: el dataset de aplicación (OOT) y el de entrenamiento tienen el **mismo corte**, 202408 (ver la corrección
  de `CODMES` en [contrato_datos.md](contrato_datos.md)), y salen de la misma población. Por eso se espera un PSI
  cercano a 0. El semáforo y el reporte quedan listos para detectar drift cuando llegue un mes nuevo en
  `raw/<yyyymm>/`.
