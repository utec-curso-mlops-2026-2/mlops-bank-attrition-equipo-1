#!/bin/bash
# Bootstrap de la EC2 del servidor MLflow (Amazon Linux 2023). Solo instala dependencias;
# la configuración (credenciales desde SSM) la aplica mlflow-start.sh en cada arranque.
set -euxo pipefail
dnf install -y python3.11 python3.11-pip sqlite
useradd -r -m -d /opt/mlflow mlflow || true
sudo -u mlflow python3.11 -m venv /opt/mlflow/venv
sudo -u mlflow /opt/mlflow/venv/bin/pip install --upgrade pip
# Misma versión de MLflow que requirements.txt; [auth] habilita usuario/contraseña.
sudo -u mlflow /opt/mlflow/venv/bin/pip install "mlflow[auth]==3.18.0" boto3
sudo -u mlflow mkdir -p /opt/mlflow/data
# t3.small (2 GB) se queda sin memoria con MLflow 3.18: swap de 2 GB.
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo "/swapfile none swap sw 0 0" >> /etc/fstab
touch /var/log/mlflow-bootstrap.done
