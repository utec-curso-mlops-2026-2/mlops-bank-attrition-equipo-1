# R4 · Servidor MLflow remoto compartido (AWS)

**Estado: en operación desde el 24/11/2026.** URL: `http://50.16.69.79:5000` (requiere usuario y contraseña).

## Arquitectura

| Componente | Recurso | Detalle |
|---|---|---|
| Cómputo | EC2 `i-0d95a741eb899609f` (`equipo1-mlflow`), t3.small, Amazon Linux 2023, us-east-1a | 20 GB gp3 + 2 GB de swap; MLflow **3.18.0** (igual que `requirements.txt`) con `systemd` |
| IP fija | Elastic IP `50.16.69.79` | La URL no cambia aunque la EC2 se apague y se vuelva a prender |
| Backend store | SQLite en el disco de la EC2 (`/opt/mlflow/data/mlflow.db`) | Experimentos, runs y Model Registry |
| Artifact store | `s3://equipo-1-mlops-bank-attrition-654654589924/mlflow-artifacts/` | El servidor hace de proxy (`--serve-artifacts`): **los clientes no necesitan permisos de S3** |
| Permisos | Rol `equipo1-mlflow-ec2-role` + perfil de instancia `equipo1-mlflow-ec2-profile` | Lee y escribe solo en `mlflow-artifacts/`, lee los parámetros `/equipo1/mlflow/*` y usa SSM (Session Manager, sin SSH) |
| Red | Security Group `equipo1-mlflow-sg` | Solo el puerto 5000 abierto (sin SSH) |
| Autenticación | MLflow `basic-auth` | Usuario `admin`; la contraseña y la clave CSRF están en SSM Parameter Store |

Archivos para reproducir el servidor: [`infra/mlflow/`](../infra/mlflow/) (`user-data.sh`, `mlflow-start.sh`,
`mlflow.service`, políticas IAM).

### Por qué el puerto está abierto a internet y protegido con usuario y contraseña
Los runners de GitHub Actions usan IPs que cambian, así que un Security Group limitado a las IP del equipo dejaría fuera
al workflow de entrenamiento (R6). Por eso el puerto 5000 está abierto y MLflow exige usuario y contraseña.
**Limitación conocida**: la conexión es HTTP sin TLS. Para producción habría que agregar HTTPS (un ALB con ACM o nginx
con un certificado) o restringir la entrada al rango de IP del runner en cada ejecución.

## Cómo conectarse (equipo)

Requisito: AWS CLI v2 con el perfil SSO de la cuenta del curso (ver README).

```powershell
$env:AWS_PROFILE = "mlops-labs"
$env:MLFLOW_TRACKING_URI = "http://50.16.69.79:5000"
$env:MLFLOW_EXPERIMENT_NAME = "equipo-1-fuga"
$env:MLFLOW_TRACKING_USERNAME = aws ssm get-parameter --name /equipo1/mlflow/admin-username --query Parameter.Value --output text
$env:MLFLOW_TRACKING_PASSWORD = aws ssm get-parameter --name /equipo1/mlflow/admin-password --with-decryption --query Parameter.Value --output text
```
```bash
export AWS_PROFILE=mlops-labs MLFLOW_TRACKING_URI=http://50.16.69.79:5000 MLFLOW_EXPERIMENT_NAME=equipo-1-fuga
export MLFLOW_TRACKING_USERNAME=$(aws ssm get-parameter --name /equipo1/mlflow/admin-username --query Parameter.Value --output text)
export MLFLOW_TRACKING_PASSWORD=$(aws ssm get-parameter --name /equipo1/mlflow/admin-password --with-decryption --query Parameter.Value --output text)
```
Para abrir la UI en el navegador, entrar a `http://50.16.69.79:5000` con el mismo usuario y contraseña.

**GitHub Actions** (ya configurado en el repo): secret `MLFLOW_TRACKING_PASSWORD`, y variables
`MLFLOW_TRACKING_USERNAME` y `MLFLOW_EXPERIMENT_NAME`.

Verificación rápida: `python -c "import mlflow; print(mlflow.search_experiments())"`.

## Primer entrenamiento registrado en el servidor (24/11/2026)
- Experimento `equipo-1-fuga`: run padre `train-202408` (`b55266f220594311b36c4d78a426c4aa`) y 3 runs hijos.
- Datos leídos desde `s3://equipo-1-mlops-bank-attrition-654654589924/raw/202408/` (tags `data_version`, `data_uri`,
  `git_commit` y `author`).
- Mejor modelo: `hist_gradient_boosting`, con AUC test 0,8487, KS 0,5429 y captura del 63 % en el top 20 %.
- `bank-attrition-churn` **v1**: pasó la validación, con alias `champion` y `challenger`. Se puede cargar por alias desde
  una máquina cliente.

## Operación
- **Ver los logs y el estado** (sin SSH, con SSM):
  `aws ssm start-session --target i-0d95a741eb899609f` (requiere el *Session Manager plugin*) y luego
  `sudo journalctl -u mlflow -f`. También se puede usar `aws ssm send-command` con `AWS-RunShellScript`.
- **Reiniciar**: `sudo systemctl restart mlflow`. El servicio arranca solo al encender la EC2.
- **Apagar o encender** para ahorrar:
  `aws ec2 stop-instances --instance-ids i-0d95a741eb899609f` (o `start-instances`). La IP se mantiene.
- **Respaldo**: los artefactos ya están en S3. Para el registro, copiar `/opt/mlflow/data/mlflow.db` a S3 antes de
  terminar la instancia.
- **Costo aproximado** (us-east-1): EC2 t3.small ≈ USD 0,021/h (≈ USD 15/mes si está siempre encendida), Elastic IP
  ≈ USD 0,005/h, EBS 20 GB ≈ USD 1,6/mes y S3 unos céntimos.

## Problemas encontrados al montarlo
| Síntoma | Causa | Solución |
|---|---|---|
| `ImportError ... Flask-WTF` | `--app-name basic-auth` necesita el extra de autenticación | Instalar `mlflow[auth]==3.18.0` |
| `A static secret key needs to be set for CSRF protection` | MLflow 3.18 exige una clave fija con basic-auth | `MLFLOW_FLASK_SERVER_SECRET_KEY` leída de SSM |
| El servicio muere por OOM y la instancia se bloquea | 2 GB de RAM no alcanzan para 2 workers | Agregar 2 GB de swap y usar `--workers 1` |
| Peticiones con la IP pública rechazadas | Protección contra DNS rebinding de MLflow 3.x | `--allowed-hosts` con la Elastic IP |
