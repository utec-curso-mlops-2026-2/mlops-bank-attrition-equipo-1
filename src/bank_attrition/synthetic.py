"""Generador de datos sintéticos con el mismo formato que los CSV reales del curso.

Solo para pruebas y demostraciones locales; NO sustituye a los datos reales.
`miscoded=True` reproduce el defecto real del train: el mes de referencia viene como 201208 en vez de 202408.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUT = 202408


def make_synthetic(
    n_clients: int = 1500, seed: int = 0, with_target: bool = True, miscoded: bool = False
):
    rng = np.random.default_rng(seed)
    ingreso = rng.integers(1, 10, n_clients)
    clientes = pd.DataFrame({
        "ID_CORRELATIVO": np.arange(n_clients) + seed * 1_000_000,
        "CODMES": CUT,
        "FLG_BANCARIZADO": rng.integers(0, 2, n_clients),
        "RANG_INGRESO": [f"Rang_ingreso_{i:02d}" for i in ingreso],
        "FLAG_LIMA_PROVINCIA": rng.choice(["Lima", "Provincia"], n_clients, p=[0.6, 0.4]),
        "EDAD": rng.integers(18, 80, n_clients).astype(float),
        "ANTIGUEDAD": rng.integers(0, 12, n_clients).astype(float),
        "RANG_SDO_PASIVO_MENOS0": rng.choice(["Cero", "Rango_SDO_01", "Rango_SDO_02", "Rango_SDO_03"], n_clients),
        "NRO_ACCES_CANAL1_MENOS0": rng.poisson(1.0, n_clients),
    })
    for col in ["RANG_INGRESO", "EDAD", "ANTIGUEDAD"]:
        clientes.loc[rng.random(n_clients) < 0.05, col] = np.nan
    n_req = rng.poisson(1.0, n_clients)
    rows = []
    for idx, count in enumerate(n_req):
        for _ in range(count):
            rows.append((
                clientes.at[idx, "ID_CORRELATIVO"], int(rng.choice([202403, 202404, 202405, 202406, 202407, 202408])),
                str(rng.choice(["Reclamo", "Solicitud"], p=[0.7, 0.3])),
                str(rng.choice(["PROCEDE TOTAL", "NO PROCEDE", "PROCEDE PARCIAL"], p=[0.5, 0.4, 0.1])),
            ))
    req = pd.DataFrame(rows, columns=["ID_CORRELATIVO", "CODMES", "TIPO_REQUERIMIENTO2", "DICTAMEN"])
    if with_target:
        activo = clientes["NRO_ACCES_CANAL1_MENOS0"].to_numpy()
        signal = (
            -0.25 * ingreso - 0.15 * clientes["ANTIGUEDAD"].fillna(-5).to_numpy()
            + (clientes["RANG_SDO_PASIVO_MENOS0"] == "Rango_SDO_01").to_numpy() * 0.8
            - 0.3 * activo + clientes["ANTIGUEDAD"].isna().to_numpy() * 2.5
        )
        prob = 1 / (1 + np.exp(-(signal - 0.3 + rng.normal(0, 0.5, n_clients))))
        clientes["ATTRITION"] = (rng.random(n_clients) < prob).astype(int)
    if miscoded:
        clientes["CODMES"] = 201208
        req["CODMES"] = req["CODMES"].replace({CUT: 201208})
    return clientes, req
