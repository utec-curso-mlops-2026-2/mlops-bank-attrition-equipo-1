# R6 · CI/CD con GitHub Actions

| Workflow | Disparador | Qué hace | Estado |
|---|---|---|---|
| `ci.yml` | PR y push a `develop`/`master` | Instalación, `ruff` y `pytest`. Es un **check obligatorio** del ruleset: si falla, el PR no se puede integrar | Activo |
| `train.yml` | Manual (`workflow_dispatch`) | Pruebas → entrenamiento contra MLflow remoto con datos de S3 → registro con gate y alias → prueba de humo de scoreo | Activo |
| `cd.yml` | Merge a `master` | Construir y desplegar el batch (R7) y la API (R8) en AWS | Pendiente (fase 4) |

## Acceso a AWS sin claves: OIDC
GitHub Actions asume el rol **`equipo1-github-actions-role`** con credenciales temporales (1 h). En el repo no hay
claves de AWS. La definición del rol está en [`infra/github-actions/`](../infra/github-actions/).

- **Confianza**: solo este repositorio, desde las ramas `develop`, `master` y `feature/*`, y desde el environment
  `production`.
  - El repo usa el formato de *subject* **inmutable** de GitHub (`repo:<org>@<id>/<repo>@<id>:ref:...`, ver
    `gh api repos/<org>/<repo>/actions/oidc/customization/sub`). La política acepta ese formato y también el clásico.
  - Una política escrita solo con `repo:<org>/<repo>:...` **no funciona** en este repo.
  - Nuestro permiso SSO puede crear roles, pero no editar su política de confianza después (`UpdateAssumeRolePolicy`
    solo aplica a `mlops-s13-*`). Para cambiarla hay que recrear el rol.
- **Permisos** (mínimos): leer `raw/*` del bucket de datos. Los permisos de despliegue se agregan con el CD.

## Variables y secretos (Settings → Secrets and variables → Actions)
| Nombre | Tipo | Valor / origen |
|---|---|---|
| `AWS_ROLE_ARN` | variable | `arn:aws:iam::654654589924:role/equipo1-github-actions-role` |
| `AWS_REGION` | variable | `us-east-1` |
| `MLOPS_DATA_BUCKET` | variable | `equipo-1-mlops-bank-attrition-654654589924` |
| `MLFLOW_TRACKING_URI` | variable | `http://50.16.69.79:5000` |
| `MLFLOW_EXPERIMENT_NAME` | variable | `equipo-1-fuga` |
| `MLFLOW_TRACKING_USERNAME` | variable | `admin` |
| `MLFLOW_TRACKING_PASSWORD` | **secret** | copia de SSM `/equipo1/mlflow/admin-password` |

Los valores sensibles solo se pasan a los jobs que los usan (`train`, `smoke-score`). El job de pruebas no los recibe.

## `train.yml`
```text
tests ──► train ──(outputs: run_id, model_version, validation, alias, test_auc)──► smoke-score
            └─ artefacto train-result-<run>.json          └─ artefacto score-summary-<run>.json
```
1. **tests**: `ruff` y `pytest`.
2. **train**:
   - Entra a AWS por OIDC.
   - Verifica que el servidor MLflow responda. Si la EC2 está apagada, falla con un mensaje claro.
   - Entrena con `MLOPS_DATA_SOURCE=s3`, registra en el Registry (gate de calidad y alias `champion`/`challenger`) y
     publica los **outputs** y un **artefacto** JSON. También deja un resumen en la página del run.
3. **smoke-score**:
   - Solo corre si el gate no falló.
   - Usa los outputs de `train`.
   - Carga el modelo **por alias** (input `score_alias`) y scorea el OOT sin subir nada a S3.
   - Valida que haya filas y 10 deciles, y guarda el resumen por decil. Los artefactos **no contienen datos de clientes**.

**Cómo lanzarlo**: Actions → *Entrenamiento (R6)* → *Run workflow*, eligiendo la rama y los inputs `register` y
`score_alias`. También desde la terminal:
```bash
gh workflow run train.yml --ref develop -f register=true -f score_alias=champion
gh run watch            # seguir la ejecución
```

## Problemas encontrados
| Síntoma | Causa | Solución |
|---|---|---|
| `401 You are not authenticated` en el job `train` | El secret se había guardado desde PowerShell con un salto de línea al final | Volver a cargarlo sin salto de línea: `aws ssm get-parameter ... --output text \| tr -d '\r\n' \| gh secret set MLFLOW_TRACKING_PASSWORD` |
| `ParameterNotFound` al leer `/equipo1/...` desde **Git Bash** en Windows | Git Bash convierte los argumentos que empiezan con `/` en rutas de Windows | `export MSYS_NO_PATHCONV=1` antes del comando, o usar PowerShell |
