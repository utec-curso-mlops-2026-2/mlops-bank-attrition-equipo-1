# Trabajo Integrador MLOps · Fuga de clientes de UTEC Bank

Solución MLOps de punta a punta para predecir la probabilidad de fuga (attrition) a 6 meses: datos versionados en S3,
pipeline de entrenamiento modular, trazabilidad en MLflow, Model Registry con alias, CI/CD con GitHub Actions e
inferencia batch y online en AWS. Curso MLOps UTEC Posgrado 2026-2.

## Equipo y roles
| Integrante | Rol |
|---|---|
| Fernando | MLOps Engineer – Platform & Integration (R1–R5: repo, pipeline, datos, MLflow, Registry) |
| Renzo | Data & Automation |
| Fabrizio | Modeling & Deployment |

## Estado por requerimiento
| Req. | Estado |
|---|---|
| R1 Git/GitHub | Plantilla + gitflow: ver [docs/gitflow.md](docs/gitflow.md) |
| R2 Pipeline | Implementado y probado con los datos reales: ver [docs/eda_calidad_datos.md](docs/eda_calidad_datos.md) (AUC test 0,85). Scoreo del OOT con deciles y segmentos en S3 (`bank_attrition.score`) |
| R3 Datos | S3 `raw/<yyyymm>/` + manifiesto con SHA-256: [docs/datos_s3.md](docs/datos_s3.md). Feature Store: pendiente |
| R4 MLflow | Servidor remoto: [docs/mlflow_server.md](docs/mlflow_server.md); trazabilidad implementada |
| R5 Registry | Modelo pyfunc con preprocesamiento, signature, alias `champion`/`challenger` y gate de validación |
| R6–R10 | Pendiente (ver contrato en [docs/contrato_modelo.md](docs/contrato_modelo.md)) |

## Estructura
```
config/config.yaml        parámetros del pipeline (rutas, semillas, hiperparámetros, umbrales)
src/bank_attrition/       config, data, features, models, metrics, tracking, train, registry, pyfunc_model, score
tests/                    pruebas unitarias y e2e sintético (pytest)
docs/                     contratos, decisiones, runbooks y evidencias
examples/                 request/response de ejemplo
.github/workflows/        ci.yml (activo), train.yml y cd.yml (R6, pendientes)
```

## Requisitos y configuración
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
export MLFLOW_TRACKING_URI=http://<servidor-mlflow>:5000   # PowerShell: $env:MLFLOW_TRACKING_URI="..."
export MLFLOW_EXPERIMENT_NAME=equipo-1-fuga
export MLOPS_DATA_BUCKET=<bucket>                          # solo si data.source: s3
export AWS_PROFILE=<perfil-sso-o-local>                    # nunca claves en el repo
```
Sin `MLFLOW_TRACKING_URI` el pipeline usa SQLite local (solo desarrollo; R4 exige servidor remoto compartido).

## Cómo ejecutar
```bash
export MLOPS_DATA_SOURCE=s3                                       # leer raw/<yyyymm>/ desde S3 (por defecto: local)
python -m bank_attrition.train --config config/config.yaml        # entrena 3 modelos, elige, registra y valida
python -m bank_attrition.registry promote                         # challenger -> champion si cumple el gate y mejora
python -m bank_attrition.registry contract                        # signature/metadata del champion
python -m bank_attrition.score --config config/config.yaml        # scorea el OOT con el champion y sube el CSV a S3
python -m bank_attrition.data manifest                            # actualiza docs/manifiesto_datos.json
pytest && ruff check .
```

### Salida del scoreo (OOT / batch)
`s3://<bucket>/scores/<yyyymm>/scores_<yyyymm>_<yyyymmdd>_v<version>.csv` (y una copia local en `outputs/scores/`),
**ordenado de mayor a menor probabilidad**, una fila por cliente:

| Columna | Descripción |
|---|---|
| `ID_CORRELATIVO`, `CODMES` | cliente y mes de corte |
| `probabilidad_fuga` | probabilidad de fuga a 6 meses (0–1) |
| `decil_riesgo` | decil relativo dentro del mes: 10 = 10 % de mayor riesgo, 1 = menor |
| `segmento_riesgo` | `Alto` (deciles 9–10), `Medio` (6–8), `Bajo` (1–5); cortes en `scoring.segments` |
| `fecha_score`, `modelo_nombre`, `modelo_version`, `modelo_alias` | trazabilidad del scoreo |

Opciones: `--alias challenger` para scorear con otro alias y `--no-upload` para no subir a S3. Cada ejecución registra un
run `score-<yyyymm>-<fecha>` (tag `stage=scoring`) con la distribución del score y el resumen por decil.

## Flujo del pipeline
1. `data`: descarga/lee `raw/<yyyymm>/` y calcula SHA-256 por archivo.
2. `features`: una fila por (cliente, CODMES); requerimientos agregados en ventana de N meses **sin mirar el futuro**.
3. Split train/test estratificado con semilla; limpieza e imputación **dentro del pipeline**, ajustada solo con train.
4. Entrenamiento de 3 modelos (regresión logística, random forest, gradient boosting) con CV; elige por `cv_auc_mean`.
5. Métricas: AUC, Gini, KS, lift y % de fugas capturadas en el top 10 %/20 %. El OOT (sin target) solo registra la distribución del score.
6. MLflow: run padre + un run hijo por modelo; params, métricas, ROC, matriz de confusión, importancias, tags (commit, versión de datos, autor).
7. Registry: versión con signature + input example + entorno; si cumple el gate (`registry.min_*`) recibe alias.
8. Scoreo (`score`): carga el modelo **por alias**, construye las features del OOT con el `feature_spec` guardado en el
   modelo (no se recalcula), asigna decil y segmento, y escribe el CSV priorizado en S3 + un run de MLflow.

## Consumo del modelo
`mlflow.pyfunc.load_model("models:/bank-attrition-churn@champion")` → `predict(df)` devuelve `probabilidad_fuga`.
Nunca cargar desde archivo local. Detalle en [docs/contrato_modelo.md](docs/contrato_modelo.md).

## Uso de IA
Se usó un asistente de IA para estructurar el código y la documentación; todo el código fue revisado y puede ser explicado por el equipo.
