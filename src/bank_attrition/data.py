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
        with open(path, "rb") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
        files[logical] = {"filename": name, "rows": rows, "sha256": sha256_file(path)}
    return {"release": str(cfg["data"]["release"]), "uri": data_uri(cfg), "files": files}


def write_manifest(cfg: dict, path: str | Path = "docs/manifiesto_datos.json") -> dict:
    """Rellena docs/manifiesto_datos.json con SHA-256, filas, URI y (si hay S3) VersionId."""
    import json
    from datetime import datetime, timezone

    manifest = data_manifest(cfg)
    base = Path(path)
    doc = json.loads(base.read_text(encoding="utf-8")) if base.exists() else {"files": []}
    client = None
    if cfg["data"]["source"] == "s3":
        import boto3

        client = boto3.client("s3")
    by_name = {f["logical_name"]: f for f in doc.get("files", [])}
    for logical, info in manifest["files"].items():
        entry = by_name.setdefault(logical, {"logical_name": logical})
        entry.update(filename=info["filename"], sha256=info["sha256"], row_count=info["rows"],
                     status="registrado", source_uri=manifest["uri"] + info["filename"])
        if client is not None:
            head = client.head_object(
                Bucket=cfg["data"]["s3_bucket"],
                Key=f"{cfg['data']['s3_prefix']}/{cfg['data']['release']}/{info['filename']}",
            )
            entry["version_id"] = head.get("VersionId")
    doc.update(
        status="registrado", dataset_release=manifest["release"],
        recorded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        files=list(by_name.values()),
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
