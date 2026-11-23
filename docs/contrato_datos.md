# Contrato de datos

- **Fuentes**: ver `docs/manifiesto_datos.json` (URI en S3, SHA-256, filas, VersionId). Procedencia: Kaggle *Bank Attrition Detection*.
- **Claves**: `ID_CORRELATIVO` (cliente) y `CODMES` (yyyymm, corte). Una fila por cliente y corte; el pipeline falla si hay duplicados.
- **Corrección de CODMES**: en el train, `CODMES=201208` (clientes y 7 976 requerimientos) se interpreta como `202408`. Está en `config.yaml > data.codmes_corrections` y se audita en MLflow (`config_used.json`). Pendiente de confirmación docente (ver `docs/eda_calidad_datos.md`).
- **Target**: `ATTRITION` (1 = fuga en los próximos 6 meses). Existe solo en train; el OOT no tiene etiqueta.
- **Formato**: clientes en ancho (`_MENOS0` a `_MENOS5`); requerimientos transaccionales (tipo, dictamen, producto, submotivo).
- **Requerimientos**: se agregan por cliente en la ventana de 6 meses que termina en el `CODMES` del cliente
  (se excluyen meses posteriores al corte): total, y conteo por tipo (Reclamo/Solicitud) y por dictamen. PRODUCTO_SERVICIO_2 y SUBMOTIVO_2 no se usan (22 y 81 niveles dispersos; aportan casi nada).
  Cliente sin requerimientos: conteos = 0, medias/máximos = NaN (los imputa el pipeline).
- **Calidad/preparación**: rangos → número ordinal; imputación por mediana con indicador de nulo (numéricas) y categoría `NULO` (categóricas), ajustada con train; `handle_unknown=ignore` en one-hot.
- **Limitaciones**: la etiqueta se conoce 6 meses después; el OOT solo permite monitorear drift, no desempeño; los CSV son *samples*.
- **Pendiente**: confirmar con docentes la corrección de CODMES, el origen de los nulos de ANTIGUEDAD y los duplicados de requerimientos.
