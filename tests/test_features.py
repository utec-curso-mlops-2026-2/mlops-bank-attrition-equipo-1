import pandas as pd
import pytest

from bank_attrition.features import (
    build_features,
    fit_feature_spec,
    month_index,
    prepare_input,
    split_columns,
    to_ordinal,
)


def _cfg(base_cfg):
    cfg = dict(base_cfg)
    cfg["features"] = {**base_cfg["features"], "window_months": 3, "min_category_count": 1}
    cfg["schema"] = {**base_cfg["schema"], "req_categorical_cols": ["TIPO"], "ordinal_cols": []}
    return cfg


def test_month_index_handles_year_boundary():
    idx = month_index(pd.Series([202401, 202312, 202311]))
    assert idx.iloc[0] - idx.iloc[1] == 1
    assert idx.iloc[0] - idx.iloc[2] == 2


def test_window_excludes_future_and_old_requirements(base_cfg):
    cfg = _cfg(base_cfg)
    clientes = pd.DataFrame({"ID_CORRELATIVO": ["a"], "CODMES": [202403], "EDAD": [40.0]})
    req = pd.DataFrame({
        "ID_CORRELATIVO": ["a"] * 5,
        "CODMES": [202312, 202401, 202402, 202403, 202404],  # 202312 fuera de ventana; 202404 futuro
        "TIPO": ["X"] * 5,
    })
    spec = fit_feature_spec(clientes, req, cfg)
    table = build_features(clientes, req, spec, cfg)
    assert table.loc[0, "N_REQUERIMIENTOS_VENTANA"] == 3


def test_one_row_per_client_and_zero_for_no_requirements(base_cfg):
    cfg = _cfg(base_cfg)
    clientes = pd.DataFrame({"ID_CORRELATIVO": ["a", "b"], "CODMES": [202403, 202403]})
    req = pd.DataFrame({"ID_CORRELATIVO": ["a", "a"], "CODMES": [202403, 202402], "TIPO": ["X", "Y"]})
    spec = fit_feature_spec(clientes, req, cfg)
    table = build_features(clientes, req, spec, cfg)
    assert len(table) == 2
    assert table.set_index("ID_CORRELATIVO").loc["b", "N_REQUERIMIENTOS_VENTANA"] == 0
    assert table.set_index("ID_CORRELATIVO").loc["a", "REQ_TIPO_NUNIQUE"] == 2


def test_duplicate_client_period_is_rejected(base_cfg):
    cfg = _cfg(base_cfg)
    clientes = pd.DataFrame({"ID_CORRELATIVO": ["a", "a"], "CODMES": [202403, 202403]})
    req = pd.DataFrame({"ID_CORRELATIVO": ["a"], "CODMES": [202403], "TIPO": ["X"]})
    spec = fit_feature_spec(clientes, req, cfg)
    with pytest.raises(ValueError):
        build_features(clientes, req, spec, cfg)


def test_split_columns_and_prepare_input_types(base_cfg):
    cfg = _cfg(base_cfg)
    table = pd.DataFrame({
        "ID_CORRELATIVO": ["a", "b"], "CODMES": [1, 1], "ATTRITION": [0, 1],
        "EDAD": [30, None], "SEGMENTO": ["A", None],
    })
    numeric, categorical = split_columns(table, cfg)
    assert numeric == ["EDAD"] and categorical == ["SEGMENTO"]
    prepared = prepare_input(table, numeric, categorical)
    assert str(prepared["EDAD"].dtype) == "float64"
    assert prepared["SEGMENTO"].tolist() == ["A", None]


def test_to_ordinal_converts_ranges_and_cero():
    series = pd.Series(["Rango_SDO_03", "Cero", None, "Rang_ingreso_10", "Rango_04"])
    result = to_ordinal(series)
    assert result.iloc[0] == 3 and result.iloc[1] == 0 and pd.isna(result.iloc[2])
    assert result.iloc[3] == 10 and result.iloc[4] == 4


def test_codmes_correction_restores_window_features(base_cfg):
    """El 201208 del train se corrige a 202408; sin la corrección los requerimientos se perderían."""
    from bank_attrition.data import apply_codmes_corrections

    clientes = pd.DataFrame({"ID_CORRELATIVO": [1], "CODMES": [201208], "EDAD": [30.0]})
    req = pd.DataFrame({
        "ID_CORRELATIVO": [1, 1], "CODMES": [201208, 202407], "TIPO_REQUERIMIENTO2": ["Reclamo"] * 2,
        "DICTAMEN": ["NO PROCEDE"] * 2,
    })
    cfg = dict(base_cfg)
    cfg["features"] = {**base_cfg["features"], "window_months": 6, "min_category_count": 1}
    fixed_c, fixed_r = apply_codmes_corrections(clientes, req, cfg)
    assert fixed_c["CODMES"].iloc[0] == 202408
    spec = fit_feature_spec(fixed_c, fixed_r, cfg)
    assert build_features(fixed_c, fixed_r, spec, cfg).loc[0, "N_REQUERIMIENTOS_VENTANA"] == 2
    spec_bad = fit_feature_spec(clientes, req, cfg)  # sin corrección: solo cuenta el req de 201208
    assert build_features(clientes, req, spec_bad, cfg).loc[0, "N_REQUERIMIENTOS_VENTANA"] == 1


def test_duplicate_requirements_are_kept_by_default_and_droppable(base_cfg):
    from bank_attrition.data import apply_codmes_corrections

    clientes = pd.DataFrame({"ID_CORRELATIVO": [1], "CODMES": [202408]})
    req = pd.DataFrame({
        "ID_CORRELATIVO": [1, 1], "CODMES": [202407, 202407], "TIPO_REQUERIMIENTO2": ["Reclamo"] * 2,
    })
    keep_cfg = {**base_cfg, "data": {**base_cfg["data"], "drop_duplicate_requirements": False}}
    drop_cfg = {**base_cfg, "data": {**base_cfg["data"], "drop_duplicate_requirements": True}}
    assert len(apply_codmes_corrections(clientes, req, keep_cfg)[1]) == 2
    assert len(apply_codmes_corrections(clientes, req, drop_cfg)[1]) == 1
