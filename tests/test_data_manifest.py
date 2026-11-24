"""Manifiesto de datos (R3): SHA-256, filas, URI, VersionId de S3 y política temporal."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from bank_attrition import data as data_mod
from bank_attrition.data import row_count, sha256_file, write_manifest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "manifiesto_datos.json"


@pytest.fixture()
def manifest_path(tmp_path):
    target = tmp_path / "manifiesto_datos.json"
    shutil.copy(TEMPLATE, target)
    return target


def _by_name(doc):
    return {f["logical_name"]: f for f in doc["files"]}


def test_row_count_only_for_csv(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    xlsx = tmp_path / "b.xlsx"
    xlsx.write_bytes(b"PK\x03\x04binario")
    assert row_count(csv) == 2
    assert row_count(xlsx) is None


def test_local_manifest_registers_pipeline_files(cfg, manifest_path):
    doc = write_manifest(cfg, manifest_path)
    files = _by_name(doc)
    folder = Path(cfg["data"]["local_dir"]) / str(cfg["data"]["release"])
    for logical, name in cfg["data"]["files"].items():
        entry, path = files[logical], folder / name
        assert entry["status"] == "registrado"
        assert entry["sha256"] == sha256_file(path)
        assert entry["row_count"] == len(pd.read_csv(path))
        assert entry["source_uri"].endswith(f"/{cfg['data']['release']}/{name}")  # ruta bien formada
    # sin el diccionario en local el manifiesto no puede declararse completo
    assert files["diccionario"]["status"] == "no encontrado"
    assert doc["status"] == "parcial"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == doc


def test_local_manifest_fails_if_a_pipeline_file_is_missing(cfg, manifest_path):
    folder = Path(cfg["data"]["local_dir"]) / str(cfg["data"]["release"])
    (folder / cfg["data"]["files"]["oot_clientes"]).unlink()
    with pytest.raises(FileNotFoundError, match="oot_clientes"):
        write_manifest(cfg, manifest_path)


def test_temporal_policy_documents_codmes_correction(cfg, manifest_path):
    policy = write_manifest(cfg, manifest_path)["temporal_policy"]
    assert policy["status"] == "aplicada"
    change = policy["authorized_changes"][0]
    assert (change["column"], change["from"], change["to"]) == ("CODMES", 201208, 202408)


class FakeS3:
    """Cliente S3 mínimo: sirve los archivos de una carpeta local como objetos versionados."""

    def __init__(self, folder: Path):
        self.folder, self.downloads = folder, []

    def head_object(self, Bucket, Key):
        path = self.folder / Path(Key).name
        return {"VersionId": f"v-{path.name}", "ContentLength": path.stat().st_size,
                "LastModified": datetime(2026, 11, 23, 22, 35, tzinfo=timezone.utc)}

    def download_file(self, bucket, key, filename, ExtraArgs=None):
        self.downloads.append((bucket, key, (ExtraArgs or {}).get("VersionId")))
        shutil.copy(self.folder / Path(key).name, filename)


def test_s3_manifest_hashes_the_exact_object_version(cfg, manifest_path, monkeypatch, tmp_path):
    local = data_mod.release_dir(cfg)
    bucket_dir = tmp_path / "s3-objetos"
    shutil.copytree(local, bucket_dir)
    (bucket_dir / "diccionario_datos.xlsx").write_bytes(b"PK\x03\x04diccionario")
    # la copia local de data/raw/ queda desactualizada: el manifiesto NO debe usarla
    (local / cfg["data"]["files"]["oot_clientes"]).write_text("copia,vieja\n1,2\n", encoding="utf-8")
    fake = FakeS3(bucket_dir)
    monkeypatch.setattr("boto3.client", lambda service: fake)
    cfg["data"].update(source="s3", s3_bucket="bucket-prueba")
    doc = write_manifest(cfg, manifest_path)

    files = _by_name(doc)
    assert doc["status"] == "registrado" and doc["source_system"] == "s3"
    for logical, name in {**cfg["data"]["files"], "diccionario": "diccionario_datos.xlsx"}.items():
        entry = files[logical]
        assert entry["source_uri"] == f"s3://bucket-prueba/raw/{cfg['data']['release']}/{name}"
        assert entry["version_id"] == f"v-{name}"
        assert entry["sha256"] == sha256_file(bucket_dir / name)
        assert entry["last_modified_utc"].startswith("2026-11-23T22:35")
    assert files["diccionario"]["row_count"] is None
    # cada archivo se descargó pidiendo explícitamente su VersionId
    assert all(version == f"v-{Path(key).name}" for _, key, version in fake.downloads)
    assert len(fake.downloads) == 5
