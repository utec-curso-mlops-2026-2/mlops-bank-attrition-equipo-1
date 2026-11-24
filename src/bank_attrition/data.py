"""Acceso y versionamiento de datos crudos (R3).

Los datos viven en S3 con la estructura s3://<bucket>/<prefix>/<yyyymm>/<archivo>.csv.
El pipeline los descarga a data/raw/<yyyymm>/ (ignorado por Git) y registra en MLflow
la versión, la ruta y el SHA-256 de cada archivo.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def release_dir(cfg: dict) -> Path:
    data = cfg["data"]
    return Path(data["local_dir"]) / str(data["release"])


def data_uri(cfg: dict) -> str:
    """Ruta lógica de la versión de datos usada (S3 si corresponde, local si no)."""
    data = cfg["data"]
    if data["source"] == "s3":
        return f"s3://{data['s3_bucket']}/{data['s3_prefix']}/{data['release']}/"
    return str(release_dir(cfg))


def fetch_raw(cfg: dict) -> Path:
    """Garantiza que los archivos de la versión existan en local; si source=s3 los descarga."""
    data = cfg["data"]
    target = release_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)
    if data["source"] == "s3":
        import boto3  # import diferido: solo se necesita al leer de S3

        if not data.get("s3_bucket"):
            raise ValueError("data.s3_bucket vacío: define MLOPS_DATA_BUCKET")
        client = boto3.client("s3")
        for name in data["files"].values():
            destination = target / name
            if not destination.exists():
                key = f"{data['s3_prefix']}/{data['release']}/{name}"
                client.download_file(data["s3_bucket"], key, str(destination))
    missing = [n for n in data["files"].values() if not (target / n).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos en {target}: {missing}")
    return target


def load_tables(cfg: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (clientes, requerimientos) de 'train' u 'oot'."""
    folder = fetch_raw(cfg)
    files = cfg["data"]["files"]
    clientes = pd.read_csv(folder / files[f"{split}_clientes"])
    requerimientos = pd.read_csv(folder / files[f"{split}_requerimientos"])
    return apply_codmes_corrections(clientes, requerimientos, cfg)


def apply_codmes_corrections(
    clientes: pd.DataFrame, requerimientos: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrige CODMES mal codificados (p. ej. 201208 -> 202408) en clientes y requerimientos."""
    if cfg["data"].get("drop_duplicate_requirements"):
        requerimientos = requerimientos.drop_duplicates()
    fixes = {int(k): int(v) for k, v in (cfg["data"].get("codmes_corrections") or {}).items()}
    if not fixes:
        return clientes, requerimientos
    schema = cfg["schema"]
    clientes = clientes.copy()
    requerimientos = requerimientos.copy()
    clientes[schema["period_col"]] = clientes[schema["period_col"]].replace(fixes)
    requerimientos[schema["req_period_col"]] = requerimientos[schema["req_period_col"]].replace(fixes)
    return clientes, requerimientos


def data_manifest(cfg: dict) -> dict:
    """Manifiesto de la versión de datos: ruta, archivos, filas y SHA-256."""
    folder = fetch_raw(cfg)
    files = {}
    for logical, name in cfg["data"]["files"].items():
        path = folder / name
        files[logical] = {"filename": name, "rows": row_count(path), "sha256": sha256_file(path)}
    return {"release": str(cfg["data"]["release"]), "uri": data_uri(cfg), "files": files}


def row_count(path: str | Path) -> int | None:
    """Filas de datos de un CSV (sin cabecera); None para otros formatos (p. ej. el diccionario .xlsx)."""
    if Path(path).suffix.lower() != ".csv":
        return None
    with open(path, "rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def s3_key(cfg: dict, filename: str) -> str:
    data = cfg["data"]
    return f"{data['s3_prefix']}/{data['release']}/{filename}"


def temporal_policy(cfg: dict) -> dict:
    """Correcciones temporales aplicadas a los datos crudos (hoy: CODMES mal codificado)."""
    fixes = {int(k): int(v) for k, v in (cfg["data"].get("codmes_corrections") or {}).items()}
    return {
        "status": "aplicada" if fixes else "sin cambios",
        "decision_reference": "docs/decisiones.md; docs/eda_calidad_datos.md (hallazgo 1)",
        "authorized_changes": [
            {
                "column": cfg["schema"]["period_col"], "from": old, "to": new,
                "files": ["train_clientes", "train_requerimientos", "oot_clientes", "oot_requerimientos"],
                "applied_in": "bank_attrition.data.apply_codmes_corrections (al leer; el archivo crudo no se modifica)",
                "confirmation": "pendiente de confirmación docente",
            }
            for old, new in fixes.items()
        ],
    }


def _manifest_files(cfg: dict) -> dict[str, str]:
    """Archivos que registra el manifiesto: los del pipeline + los de referencia (diccionario)."""
    data = cfg["data"]
    return {**data["files"], **(data.get("reference_files") or {})}


def _entries_from_s3(cfg: dict) -> dict[str, dict]:
    """Registra la versión vigente de cada objeto en S3: descarga ESE VersionId y calcula SHA-256 y filas.

    No usa la copia local de data/raw/ (podría estar desactualizada) ni el ETag (no es un hash del contenido).
    """
    import tempfile

    import boto3  # import diferido: solo se necesita con S3

    bucket = cfg["data"].get("s3_bucket")
    if not bucket:
        raise ValueError("data.s3_bucket vacío: define MLOPS_DATA_BUCKET")
    client = boto3.client("s3")
    entries = {}
    with tempfile.TemporaryDirectory() as tmp:
        for logical, name in _manifest_files(cfg).items():
            key = s3_key(cfg, name)
            head = client.head_object(Bucket=bucket, Key=key)
            version_id = head.get("VersionId")
            local = Path(tmp) / name
            extra = {"VersionId": version_id} if version_id and version_id != "null" else None
            client.download_file(bucket, key, str(local), ExtraArgs=extra)
            modified = head.get("LastModified")
            entries[logical] = {
                "filename": name, "status": "registrado", "source_uri": f"s3://{bucket}/{key}",
                "version_id": version_id, "sha256": sha256_file(local), "row_count": row_count(local),
                "size_bytes": head.get("ContentLength"),
                "last_modified_utc": modified.isoformat() if modified is not None else None,
            }
    return entries


def _entries_from_local(cfg: dict) -> dict[str, dict]:
    folder = release_dir(cfg)
    entries = {}
    for logical, name in _manifest_files(cfg).items():
        path = folder / name
        if not path.exists():
            if logical in cfg["data"]["files"]:
                raise FileNotFoundError(f"Falta {path}: el manifiesto no puede registrar la versión")
            entries[logical] = {"filename": name, "status": "no encontrado", "source_uri": None,
                                "version_id": None, "sha256": None}
            continue
        entries[logical] = {
            "filename": name, "status": "registrado", "source_uri": path.as_posix(),
            "version_id": None, "sha256": sha256_file(path), "row_count": row_count(path),
            "size_bytes": path.stat().st_size,
        }
    return entries


def write_manifest(cfg: dict, path: str | Path = "docs/manifiesto_datos.json") -> dict:
    """Rellena docs/manifiesto_datos.json: por archivo URI, VersionId (S3), SHA-256, filas y tamaño."""
    import json
    from datetime import datetime, timezone

    base = Path(path)
    doc = json.loads(base.read_text(encoding="utf-8")) if base.exists() else {"files": []}
    entries = _entries_from_s3(cfg) if cfg["data"]["source"] == "s3" else _entries_from_local(cfg)
    by_name = {f["logical_name"]: f for f in doc.get("files", [])}
    for logical, info in entries.items():
        by_name.setdefault(logical, {"logical_name": logical}).update(info)
    files = list(by_name.values())
    complete = all(f.get("status") == "registrado" for f in files)
    doc.update(
        status="registrado" if complete else "parcial",
        dataset_release=str(cfg["data"]["release"]),
        source_system=cfg["data"]["source"],
        recorded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        temporal_policy=temporal_policy(cfg),
        files=files,
    )
    base.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="Utilidades de datos (R3)")
    parser.add_argument("action", choices=["manifest"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.action == "manifest":
        write_manifest(cfg)
        print("docs/manifiesto_datos.json actualizado")


if __name__ == "__main__":
    main()
