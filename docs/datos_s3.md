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

## Versión registrada: `raw/202408/` (24/11/2026)
Bucket `equipo-1-mlops-bank-attrition-654654589924` (us-east-1) con **versionado activo**. El detalle completo está en
[`manifiesto_datos.json`](manifiesto_datos.json):

| Archivo | Filas | Tamaño | VersionId | SHA-256 (inicio) |
|---|---|---|---|---|
| `train_clientes_sample.csv` | 56 000 | 8,9 MB | `EDfwcipExQJDsQ8ypw8bsgxWJoKkwKYB` | `9f4c909d…` |
| `train_requerimientos_sample.csv` | 41 313 | 2,5 MB | `zr.2G2XJH7BUdrl6lPqdd14RsCP0ZXGq` | `4d986f2a…` |
| `oot_clientes_sample.csv` | 14 000 | 2,2 MB | `uSnn7lRnuEH30rEf0XXofQs3rUJrKmPI` | `471d9c4a…` |
| `oot_requerimientos_sample.csv` | 10 104 | 0,6 MB | `Bh6LJQoAvSKsRPbCByIHcprT5mnRfLBc` | `4569a487…` |
| `diccionario_datos.xlsx` | — | 11 KB | `ysBESjyMvq0Jk6GpBeSZgyXhE_zacSU_` | `342f511d…` |

**Cómo se generó**: `python -m bank_attrition.data manifest` con `data.source: s3`. Por cada objeto:
1. Consulta el `VersionId` vigente.
2. **Descarga esa versión exacta** y calcula el SHA-256 y las filas.

No usa la copia local de `data/raw/`, que podría estar desactualizada, ni el ETag, que no es un hash del contenido.

**Verificación (24/11)**: hay 1 versión por objeto, el `VersionId` registrado es el vigente, y el SHA-256 coincide con la
copia local usada para entrenar.

**Para reproducir exactamente estos datos**:
`aws s3api get-object --bucket <bucket> --key raw/202408/<archivo> --version-id <VersionId> <archivo>`.

**Política temporal** (`temporal_policy` en el manifiesto): `CODMES` 201208 → 202408 se corrige al leer los datos, sin
modificar el archivo crudo. Está documentado en `docs/decisiones.md` y queda **pendiente de confirmación docente**.

**Si llegan datos nuevos**: subirlos a otra carpeta `raw/<yyyymm>/` (no sobrescribir), cambiar `data.release` y volver a
generar el manifiesto.

Feature Store (opcional): ver `docs/decisiones.md`.
