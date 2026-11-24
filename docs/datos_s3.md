# R3 · Datos en S3

Estructura (una carpeta por fecha de corte; no sobrescribir, crear otra carpeta si cambian los datos):
```
s3://<bucket>/raw/<yyyymm>/train_clientes_sample.csv
                           train_requerimientos_sample.csv
                           oot_clientes_sample.csv
                           oot_requerimientos_sample.csv
                           diccionario_datos.xlsx
```
Subida (con perfil SSO/local; activar *Versioning* en el bucket para obtener `VersionId`):
```bash
aws s3api put-bucket-versioning --bucket <bucket> --versioning-configuration Status=Enabled
aws s3 cp ./kaggle/ s3://<bucket>/raw/202408/ --recursive
```
Ejecución: `MLOPS_DATA_SOURCE=s3` (por defecto `local`), `data.release: "202408"` en `config/config.yaml` y
`MLOPS_DATA_BUCKET=<bucket>`.

Salidas del scoreo (lista priorizada para retención): `s3://<bucket>/scores/<yyyymm>/scores_<yyyymm>_<yyyymmdd>_v<version>.csv`
(ver README, sección "Salida del scoreo").
Registro: `python -m bank_attrition.data manifest` completa `docs/manifiesto_datos.json` (SHA-256, filas, URI, VersionId).
Cada run de MLflow guarda `data_version`, `data_uri` y `data_manifest.json`.

**Pendiente**: confirmar el `yyyymm` real de corte (`CODMES`) al revisar los CSV. Feature Store (opcional): ver `docs/decisiones.md`.
