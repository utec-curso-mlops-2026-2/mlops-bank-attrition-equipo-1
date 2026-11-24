"""Contrato HTTP del contenedor de inferencia (Batch Transform y endpoint online): validación y errores claros."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bank_attrition.serve import create_app

META = {"name": "bank-attrition-churn", "alias": "champion", "version": "7",
        "numeric": ["EDAD", "N_REQUERIMIENTOS_VENTANA"], "categorical": ["SEXO"]}


class StubModel:
    """Probabilidad determinística: permite comprobar el cableado sin MLflow."""

    def predict(self, frame):
        assert list(frame.columns) == ["EDAD", "N_REQUERIMIENTOS_VENTANA", "SEXO"]
        assert str(frame["EDAD"].dtype) == "float64"
        return pd.DataFrame({"probabilidad_fuga": frame["EDAD"].fillna(50) / 100})


@pytest.fixture()
def client():
    return TestClient(create_app(StubModel(), META))


def online(features, **meta):
    return {"ID_CORRELATIVO": "C1", "CODMES": 202408, **meta, "features": features}


def test_ping(client):
    assert client.get("/ping").status_code == 200


def test_online_prediction_returns_probability_and_version(client):
    resp = client.post("/invocations", json=online({"EDAD": 40, "N_REQUERIMIENTOS_VENTANA": 2, "SEXO": "F"}))
    assert resp.status_code == 200
    body = resp.json()
    assert body["probabilidad_fuga"] == pytest.approx(0.40)
    assert body["modelo_version"] == "7" and body["modelo_alias"] == "champion"
    assert body["ID_CORRELATIVO"] == "C1" and body["request_id"]


def test_online_nulls_are_allowed_and_numeric_strings_are_coerced(client):
    resp = client.post("/invocations", json=online({"EDAD": "30", "N_REQUERIMIENTOS_VENTANA": None, "SEXO": None}))
    assert resp.status_code == 200 and resp.json()["probabilidad_fuga"] == pytest.approx(0.30)


def test_online_batch_of_instances(client):
    payload = {"instances": [online({"EDAD": 10, "N_REQUERIMIENTOS_VENTANA": 0, "SEXO": "M"}),
                             online({"EDAD": 90, "N_REQUERIMIENTOS_VENTANA": 1, "SEXO": "F"}, ID_CORRELATIVO="C2")]}
    preds = client.post("/invocations", json=payload).json()["predicciones"]
    assert [p["ID_CORRELATIVO"] for p in preds] == ["C1", "C2"]


@pytest.mark.parametrize(("payload", "fragment"), [
    (online({"EDAD": 40, "SEXO": "F"}), "faltan features requeridas: N_REQUERIMIENTOS_VENTANA"),
    (online({"EDAD": "cuarenta", "N_REQUERIMIENTOS_VENTANA": 1, "SEXO": "F"}), "deben ser numéricas: EDAD"),
    (online({"EDAD": True, "N_REQUERIMIENTOS_VENTANA": 1, "SEXO": "F"}), "deben ser numéricas: EDAD"),
    ({"ID_CORRELATIVO": "C1", "EDAD": 40}, "se espera"),
    (online({"EDAD": 1, "N_REQUERIMIENTOS_VENTANA": 1, "SEXO": "F"}, ID_CORRELATIVO=""), "falta ID_CORRELATIVO"),
])
def test_online_validation_errors_are_422_with_detail(client, payload, fragment):
    resp = client.post("/invocations", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"] == "entrada_invalida"
    assert any(fragment in d for d in resp.json()["detalle"])


def test_malformed_json_and_unsupported_type(client):
    bad = client.post("/invocations", content=b"{no es json", headers={"Content-Type": "application/json"})
    assert bad.status_code == 422 and "JSON mal formado" in bad.json()["detalle"][0]
    csv = client.post("/invocations", content=b"a,b", headers={"Content-Type": "text/csv"})
    assert csv.status_code == 415


def test_jsonlines_batch_keeps_ids_and_order(client):
    lines = [{"ID_CORRELATIVO": f"C{i}", "CODMES": 202408, "EDAD": i, "N_REQUERIMIENTOS_VENTANA": 0, "SEXO": "F"}
             for i in range(5)]
    body = "\n".join(json.dumps(x) for x in lines).encode()
    resp = client.post("/invocations", content=body, headers={"Content-Type": "application/jsonlines"})
    assert resp.status_code == 200
    out = [json.loads(x) for x in resp.text.splitlines()]
    assert [o["ID_CORRELATIVO"] for o in out] == [f"C{i}" for i in range(5)]
    assert all(o["modelo_version"] == "7" for o in out)


def test_internal_errors_do_not_leak_traces():
    class Broken:
        def predict(self, frame):
            raise RuntimeError("secreto interno")

    resp = TestClient(create_app(Broken(), META)).post(
        "/invocations", json=online({"EDAD": 1, "N_REQUERIMIENTOS_VENTANA": 1, "SEXO": "F"}))
    assert resp.status_code == 500 and "secreto" not in resp.text


def test_examples_match_the_server_contract():
    """examples/ documenta el contrato real: el request válido pasa y el inválido devuelve el detalle publicado."""
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "examples"
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    features = request["features"]
    meta = {**META, "numeric": [k for k, v in features.items() if not isinstance(v, str)],
            "categorical": [k for k, v in features.items() if isinstance(v, str)]}

    class Echo:
        def predict(self, frame):
            return pd.DataFrame({"probabilidad_fuga": [0.2] * len(frame)})

    client = TestClient(create_app(Echo(), meta))
    ok = client.post("/invocations", json=request)
    assert ok.status_code == 200
    response = json.loads((folder / "response.json").read_text(encoding="utf-8"))
    assert set(ok.json()) == set(response)

    bad = client.post("/invocations", json=json.loads((folder / "request_invalido.json").read_text(encoding="utf-8")))
    expected = json.loads((folder / "response_invalido.json").read_text(encoding="utf-8"))
    assert bad.status_code == 422 and bad.json() == expected
