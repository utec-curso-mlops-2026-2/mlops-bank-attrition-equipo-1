# R4 · Servidor MLflow remoto compartido (AWS)

Arquitectura mínima: **EC2** (t3.small) con MLflow + **SQLite en EBS** (o RDS PostgreSQL) como backend store y
**S3** como artifact store, accedido por el servidor (`--serve-artifacts`): los clientes no necesitan credenciales de S3.

```bash
# En la EC2 (Amazon Linux/Ubuntu) con un rol de instancia que permita s3:Get/Put/List en el bucket de artefactos
python3 -m venv ~/mlflow-env && source ~/mlflow-env/bin/activate
pip install mlflow==3.18.0 boto3          # MISMA versión que requirements.txt
mkdir -p ~/mlflow && cd ~/mlflow
nohup mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////home/ec2-user/mlflow/mlflow.db \
  --artifacts-destination s3://<bucket-mlflow>/artifacts --serve-artifacts \
  > mlflow.log 2>&1 &
```
- **Seguridad**: Security Group solo con el puerto 5000 desde las IP del equipo (o túnel SSH
  `ssh -L 5000:localhost:5000 ec2-user@<ip>`). MLflow no trae autenticación por defecto; no exponerlo a `0.0.0.0/0`.
- **Persistencia**: para sobrevivir reinicios usar `systemd` o `tmux`; hacer respaldo de `mlflow.db`.
- **Costos**: apagar la EC2 cuando no se use; configurar AWS Budgets. Los artefactos en S3 cuestan céntimos.
- **Clientes**: `export MLFLOW_TRACKING_URI=http://<ip-o-dns>:5000` y `MLFLOW_EXPERIMENT_NAME=equipo-1-fuga`.
- **Verificar**: `python -c "import mlflow; mlflow.set_tracking_uri('http://<ip>:5000'); print(mlflow.search_experiments())"`.

Alternativa si la UTEC provee un servidor administrado: usar su URL en `MLFLOW_TRACKING_URI` (el código no cambia).
Evidencia para el informe: captura de la UI con el experimento, runs hijos y el modelo en *Models*.
