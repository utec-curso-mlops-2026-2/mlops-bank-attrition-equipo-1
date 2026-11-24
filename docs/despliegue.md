# Despliegue y operación

Guía para operar lo que ya está en AWS y continuar lo pendiente. El detalle de cada componente está en
[mlflow_server.md](mlflow_server.md), [datos_s3.md](datos_s3.md), [cicd.md](cicd.md) y
[batch_sagemaker.md](batch_sagemaker.md). Las decisiones de diseño están en [decisiones.md](decisiones.md).

## 1. Recursos en AWS
Cuenta compartida del curso, región `us-east-1`. Todos los recursos del equipo empiezan con `equipo1` o
`equipo-1`. **Ninguno guarda claves en el repo.**

| Recurso | Nombre | Para qué |
|---|---|---|
| Bucket S3 (versionado) | `equipo-1-mlops-bank-attrition-654654589924` | `raw/<yyyymm>/` datos crudos · `mlflow-artifacts/` · `batch/` intermedios · `scores/` salida batch · `monitoring/` drift |
| EC2 + Elastic IP | `equipo1-mlflow` (`i-0d95a741eb899609f`), `http://50.16.69.79:5000` | Servidor MLflow con basic-auth (usuario `admin`) |
| Parámetros SSM | `/equipo1/mlflow/admin-password`, `/equipo1/mlflow/flask-secret-key` | Credenciales de MLflow. **Leerlas de SSM, nunca copiarlas al chat ni al repo** |
| Rol OIDC de GitHub | `equipo1-github-actions-role` | `train.yml` y `cd.yml`. Políticas `equipo1-train` y `equipo1-github-actions-deploy` ([`infra/github-actions/`](../infra/github-actions/)) |
| ECR | `equipo1-bank-attrition` | Imagen de inferencia, una por commit (tags inmutables) |
| Roles de servicio | `equipo1-sagemaker-execution-role`, `equipo1-scheduler-batch-role` | Jobs de SageMaker y disparo del pipeline ([`infra/sagemaker/`](../infra/sagemaker/)) |
| SageMaker Pipeline | `equipo1-bank-attrition-batch` | Batch R7: preparar → Batch Transform → publicar |
| EventBridge Scheduler | `equipo1-batch-mensual` | Día 1 de cada mes, 06:00 (America/Lima) |
| Model Registry (MLflow) | `bank-attrition-churn` | Alias `champion` (v1) y `challenger` (v2) |

## 2. Acceso
1. `aws configure sso` con la URL y la región SSO del portal del curso, rol **PowerUserAccess**, región de recursos
   `us-east-1`. Después, `aws sso login --profile <perfil>`.
2. `python -m venv .venv`, activar el entorno y `pip install -e ".[dev]"`.
3. Variables mínimas para trabajar contra AWS y MLflow:
   ```bash
   export AWS_PROFILE=<perfil> AWS_DEFAULT_REGION=us-east-1
   export MLFLOW_TRACKING_URI=http://50.16.69.79:5000 MLFLOW_EXPERIMENT_NAME=equipo-1-fuga
   export MLFLOW_TRACKING_USERNAME=admin MLFLOW_AUTH_SSM_PARAMETER=/equipo1/mlflow/admin-password
   export MLOPS_DATA_SOURCE=s3 MLOPS_DATA_BUCKET=equipo-1-mlops-bank-attrition-654654589924
   ```
   Con `MLFLOW_AUTH_SSM_PARAMETER`, el código lee la contraseña de SSM con tus credenciales SSO. Para la interfaz web
   de MLflow, consúltala con `aws ssm get-parameter --name /equipo1/mlflow/admin-password --with-decryption`.
4. **Sin `MLFLOW_EXPERIMENT_NAME`, MLflow crea un experimento `bank-attrition` en el servidor compartido.**
   Defínela siempre.

## 3. Operación
| Tarea | Cómo |
|---|---|
| Entrenar y registrar | Actions → *Entrenamiento (R6)* → *Run workflow* (o `gh workflow run train.yml --ref develop`). En local: `python -m bank_attrition.train` |
| Promover challenger → champion | `python -m bank_attrition.registry promote` (respeta el gate) |
| Desplegar el batch | Automático al integrar a `master` (`cd.yml`). A mano: Actions → *Despliegue* → *Run workflow* |
| Ejecutar el batch ya | `python -m bank_attrition.aws_batch run --wait` (requiere además `SAGEMAKER_ROLE_ARN`, ver [batch_sagemaker.md](batch_sagemaker.md)) |
| Estado de la última ejecución | `python -m bank_attrition.aws_batch status`, o en la consola: SageMaker AI → Pipelines |
| Logs | CloudWatch → `/aws/sagemaker/ProcessingJobs`, `/aws/sagemaker/TransformJobs` (streams `pipelines-<ejecución>-*`) |
| Resultados | `s3://…/scores/<yyyymm>/`, `s3://…/monitoring/<yyyymm>/` y runs `batch-*` en MLflow (tag `stage=batch_inference`) |
| Nuevo mes de datos | Subir los archivos a `raw/<yyyymm>/`, actualizar `docs/manifiesto_datos.json` (`python -m bank_attrition.data manifest`) y ejecutar el pipeline con el parámetro `Release=<yyyymm>` |

## 4. Costos y encendido
| Recurso | Costo aproximado | Qué hacer |
|---|---|---|
| EC2 de MLflow (t3.small, 20 GB) | ~USD 15/mes encendida | Se puede apagar: `aws ec2 stop-instances --instance-ids i-0d95a741eb899609f`. **Prenderla antes de entrenar, ejecutar el batch o la demo** (`start-instances`). El servicio arranca solo |
| Elastic IP | ~USD 3,6/mes | Mantenerla: la URL de MLflow depende de ella |
| Pipeline batch | < USD 0,10 por ejecución | No deja nada encendido |
| Endpoint online (R8) | Depende del tipo | Si es Serverless, solo cobra por invocación. Uno en tiempo real (`ml.t2.medium`) cobra por hora: **apagarlo después de la demo** |

## 5. Problemas conocidos
| Síntoma | Causa | Solución |
|---|---|---|
| `Unsupported manifest media type ... oci` al crear el Model | SageMaker no acepta imágenes con manifiesto OCI | Construir con `docker buildx build --provenance=false --sbom=false --output type=image,...,oci-mediatypes=false` (ya está en `cd.yml`) |
| El contenedor no responde `/ping` y en los logs aparece `HIDDEN_DUE_TO_SECURITY_REASONS` | SageMaker enmascara las variables del Model cuyo nombre contiene `PASSWORD`, `SECRET`, `TOKEN`... | No usar esas palabras en los nombres de variables (hay una prueba que lo verifica) |
| `AccessDenied` al cambiar la confianza de un rol | Nuestro permiso no incluye `UpdateAssumeRolePolicy` | Recrear el rol con la confianza completa desde el inicio |
| `ParameterNotFound` o rutas raras con `/equipo1/...` en **Git Bash** | Git Bash convierte los argumentos y variables que empiezan con `/` | `export MSYS_NO_PATHCONV=1`, o usar PowerShell |
| `401` de MLflow en GitHub Actions | Secret guardado con un salto de línea al final | Volver a cargarlo sin salto de línea (ver [cicd.md](cicd.md)) |

## 6. Pendiente y cómo continuar
1. **R8 · Endpoint online.** La imagen (`serve`) ya resuelve el contrato: JSON → probabilidad y versión, con
   validación 422 (ver [`examples/`](../examples/)). Falta desplegarla:
   - SageMaker Model con la misma imagen y el mismo entorno (`aws_batch.ensure_model` sirve de base).
   - EndpointConfig, de preferencia **Serverless** (`ServerlessConfig`, 3–6 GB de memoria: el modelo se carga
     desde MLflow al arrancar), y Endpoint.
   - Agregarlo a `cd.yml` con una prueba posterior que envíe `examples/request.json` con
     `aws sagemaker-runtime invoke-endpoint --content-type application/json --body fileb://examples/request.json out.json`.
   - Agregar al rol de despliegue los permisos de endpoint (`CreateEndpointConfig`, `CreateEndpoint`,
     `UpdateEndpoint`, `DescribeEndpoint`, `InvokeEndpoint`) sobre `equipo1-bank-attrition-*`.
   - Documentar en el README cómo invocarlo.
2. **R9 · Plan de model/concept drift** para cuando lleguen las etiquetas reales (6 meses después): métricas (AUC, KS,
   captura top 20 % por cohorte), frecuencia y criterio de reentrenamiento. Va en el informe. El PSI ya corre en cada
   batch.
3. **R10 · Diagrama de arquitectura** en [arquitectura.md](arquitectura.md), con los componentes de la sección 1 y
   los disparadores: PR → CI, `workflow_dispatch` → entrenamiento, merge a `master` → CD, schedule → batch.
4. **Informe PDF** (máximo 10 páginas) en `docs/`, y el **PR `develop → master` antes del 26/11**: se califica el
   último commit en `master`.
