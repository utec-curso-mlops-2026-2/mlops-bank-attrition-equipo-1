# EDA y calidad de datos (base del informe, secciones 3 y 4)

Fecha del análisis: 23/11/2026. Fuente: los 4 CSV + `diccionario_datos.xlsx` de Kaggle (*Bank Attrition Detection*).

## Volúmenes
| Archivo | Filas | Notas |
|---|---|---|
| train_clientes | 56 000 | 1 fila por cliente, sin duplicados de ID; ATTRITION = 15,5 % |
| train_requerimientos | 41 313 | 50 % de los clientes tiene ≥1 requerimiento; 1 790 filas exactamente repetidas |
| oot_clientes | 14 000 | Sin ATTRITION; **0 IDs en común con train** |
| oot_requerimientos | 10 104 | meses 202403–202408; 395 filas repetidas |

## Hallazgos y decisión tomada
| # | Hallazgo | Evidencia | Decisión |
|---|---|---|---|
| 1 | **`CODMES` del train mal codificado**: todos los clientes traen 201208 y 7 976 requerimientos también | El resto de requerimientos va de 202403 a 202407; el OOT va de 202403 a 202408; el diccionario fija 202408 como mes 0; el volumen de 201208 (7 976) es coherente con un mes más | Corregir 201208 → 202408 (`data.codmes_corrections`), registrar el cambio en MLflow y **confirmar con los docentes** |
| 2 | La base de clientes ya viene **en formato ancho** (`_MENOS0` … `_MENOS5`); los requerimientos son transaccionales | Diccionario, columna CORTE | Features de clientes directas; requerimientos agregados en ventana de 6 meses |
| 3 | **Nulos informativos**: ANTIGUEDAD nula → 76 % de fuga (vs 15,5 %); EDAD nula 25 %; RANG_INGRESO nulo 24 % | tasas de fuga por nulo | Conservar el indicador de nulo (`add_indicator`) y categoría `NULO`; **riesgo**: si el nulo aparece después de la fuga sería fuga de información (confirmar origen). Proporción de nulos similar en OOT (2,6 % vs 2,5 %) |
| 3b | **Por qué ANTIGUEDAD es nula** (análisis): los nulos NO son aleatorios: con las demás variables se predice el nulo con AUC 0,83. Esos clientes están desvinculados: nómina 9 % (vs 39 %), saldo pasivo en cero 50 % (vs 18 %), accesos canal 3 de 1,1 (vs 5,7) y saldo activo casi nulo. | comparación de perfiles | Es un nulo **informativo de baja vinculación** (no un error aleatorio). La proporción es igual en OOT (2,6 % vs 2,5 %) y está disponible al puntuar, así que se usa; el mecanismo exacto sigue siendo una hipótesis. Cualquier nulo eleva la fuga (12 % sin nulos → 29 % con uno). |
| 4 | Valores atípicos de ANTIGUEDAD (12–20): 11 clientes, todos con fuga | ver tabla de ANTIGUEDAD | Se mantienen (pocos); no inflan el modelo |
| 4b | **Requerimientos repetidos** (4,3 % train, 3,9 % OOT): 1 790 filas idénticas, estables entre meses (3–5 %). Se esperarían ≈376 por azar entre requerimientos del mismo cliente y mes; hay unas 4,8 veces más; además 3 146 requerimientos repiten cliente + mes + submotivo | conteos | Más compatible con **reclamos reiterados del mismo cliente** que con un error de carga. Sin id de requerimiento no se puede probar. Se conservan (`drop_duplicate_requirements: false`); eliminarlos cambia el AUC en ±0,001 (3 semillas), por lo que la decisión no afecta al modelo |
| 5 | Rangos ordenados (`Rang_ingreso_0X`, `Rango_SDO_XX`, `Cero`) | valores | Convertidos a número ordinal |
| 6 | Sin requerimientos → fuga 21,8 %; con ≥1 → ~9 % (la inactividad pesa más que el reclamo) | tasas por nº de requerimientos | Se conserva como feature (`N_REQUERIMIENTOS_VENTANA`) |
| 7 | **Sin deriva train → OOT**: PSI máximo por variable 0,0008; PSI del score 0,0007 | misma distribución | El reporte de drift mostrará "estable"; para el B3 hará falta **simular** deriva |
| 8 | Hay 1 solo corte temporal en el train | CODMES único | Split aleatorio estratificado; **no se puede validar estabilidad temporal** (limitación a declarar) |

## Señal predictiva (holdout 30 %, gradient boosting)
- AUC univariada máxima 0,65 (ANTIGUEDAD): no hay variable "demasiado buena" que sugiera fuga evidente.
- Importancia (permutación): saldo pasivo (≈0,15), ANTIGUEDAD (0,06), ingreso (0,04), nº de productos (0,02).
- Ablación: sin ANTIGUEDAD AUC 0,823; sin ANTIGUEDAD ni saldo pasivo 0,772 (el modelo sigue siendo útil sin la variable con nulos sospechosos).
- Los requerimientos aportan casi nada una vez conocidas las variables de cliente (AUC 0,8547 → 0,8538).

## Resultado del pipeline con datos reales (semilla 42, split 70/30)
| Modelo | AUC CV (train) | AUC test | KS test | % fugas en top 20 % |
|---|---|---|---|---|
| Regresión logística | 0,801 | 0,800 | 0,450 | 55,5 % |
| Random forest | 0,827 | 0,827 | 0,508 | 58,8 % |
| **Gradient boosting (elegido)** | **0,851** | **0,849** | **0,543** | **63,0 %** |

Lift top 10 % = 4,2 (el 10 % de clientes con mayor score fuga 4,2 veces más que el promedio). Scores OOT: media 0,157, p90 0,44.
Gate del Registry (AUC ≥ 0,80, KS ≥ 0,45, captura top 20 % ≥ 0,55): **aprobado** → alias `champion`.

## Preguntas para los docentes
1. ¿Confirman que 201208 es un error y debe leerse 202408?
2. ¿Por qué ANTIGUEDAD es nula en clientes con 76 % de fuga? ¿Existe al momento de puntuar?
3. (Opcional) ¿Las filas repetidas de requerimientos son duplicados de carga o reclamos reiterados? No afecta al modelo.
4. ¿Cuál es el horizonte exacto del target (6 meses desde 202408)?
