#!/bin/bash
# Arranque del servidor MLflow del equipo 1 (lo ejecuta mlflow.service).
# Las credenciales se leen de SSM Parameter Store en cada arranque; no viven en el repo.
set -euo pipefail
export AWS_DEFAULT_REGION=us-east-1
PUBLIC_IP=50.16.69.79   # Elastic IP asociada a la instancia
PW=$(aws ssm get-parameter --name /equipo1/mlflow/admin-password --with-decryption --query Parameter.Value --output text)
umask 077
cat > /opt/mlflow/data/basic_auth.ini <<INI
[mlflow]
default_permission = READ
database_uri = sqlite:////opt/mlflow/data/basic_auth.db
admin_username = admin
admin_password = ${PW}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
INI
export MLFLOW_AUTH_CONFIG_PATH=/opt/mlflow/data/basic_auth.ini
export MLFLOW_FLASK_SERVER_SECRET_KEY=$(aws ssm get-parameter --name /equipo1/mlflow/flask-secret-key --with-decryption --query Parameter.Value --output text)
exec /opt/mlflow/venv/bin/mlflow server \
  --app-name basic-auth \
  --host 0.0.0.0 --port 5000 --workers 1 \
  --backend-store-uri sqlite:////opt/mlflow/data/mlflow.db \
  --artifacts-destination s3://equipo-1-mlops-bank-attrition-654654589924/mlflow-artifacts \
  --serve-artifacts \
  --allowed-hosts "${PUBLIC_IP}:5000,${PUBLIC_IP},localhost:5000,localhost,127.0.0.1:5000"
