# Proyecto 1 — Competencia de Modelación

**CC3092 Deep Learning y sistemas inteligentes**
**Esteban Cárcamo**

Predicción del precio de venta de viviendas (Ames Housing) mediante un Multi-Layer Perceptron.
Métrica: RMSE en dólares.

**Resultado final:** RMSE de **28,375 ± 892 USD** por validación cruzada sobre `train_dev`, y
**24,197 USD** sobre el test interno de 176 viviendas nunca vistas. El mejor baseline clásico
(GradientBoosting) obtuvo 31,877 USD bajo el mismo protocolo.

---

## 2.1 Análisis exploratorio de datos

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

### Dimensiones y tipos de variables

1168 observaciones × 81 columnas: 79 features, un identificador y el target. No hay filas
duplicadas ni identificadores repetidos. La clasificación por rol estadístico —que no coincide
con el `dtype` que infiere pandas— es:

| Tipo | N | Ejemplos |
|---|---|---|
| Continua | 19 | `GrLivArea`, `LotArea`, `TotalBsmtSF` |
| Discreta (conteo) | 9 | `FullBath`, `BedroomAbvGr`, `GarageCars` |
| Ordinal numérica | 2 | `OverallQual`, `OverallCond` |
| Ordinal categórica | 21 | `ExterQual`, `KitchenQual`, `BsmtExposure` |
| Nominal | 23 | `Neighborhood` (25 niveles), `MSSubClass` |
| Temporal | 5 | `YearBuilt`, `YrSold`, `MoSold` |

Dos reclasificaciones deliberadas respecto de lo que infiere pandas:

- **`MSSubClass` pasa de numérica a nominal.** Sus valores (20, 60, 190) son códigos de clase de
  vivienda, no magnitudes. Tratarlo como número impone un orden y unas distancias inexistentes.
- **21 columnas de texto se identifican como ordinales.** `ExterQual` toma valores
  `Po < Fa < TA < Gd < Ex`. Codificarlas con one-hot destruiría ese orden; el mapeo entero lo
  conserva y ahorra 67 columnas dummy (88 categorías → 21 columnas).

### Variable objetivo

`SalePrice`, en USD. Rango 34,900 – 745,000.

| Estadístico | Valor |
|---|---|
| Media | 181,442 |
| Mediana | 165,000 |
| Desviación estándar | 77,264 |
| P95 | 327,300 |
| Asimetría (skew) | 1.74 |
| Curtosis | 5.48 |
| Shapiro-Wilk | W = 0.881, p = 5.5·10⁻²⁹ |
| Asimetría tras `log1p` | 0.12 |

La distribución tiene una cola derecha marcada: la media supera a la mediana en 16,441 USD y el
máximo está a 7.3 desviaciones estándar. La transformación `log1p` reduce la asimetría a 0.12 y
alinea el Q-Q plot.

**Figura:** `reports/figures/01_target_distribucion.png`

### Estadísticas descriptivas

Las features numéricas abarcan cinco órdenes de magnitud, de 10⁰ (`BsmtHalfBath`) a 10⁵
(`LotArea`). Las más asimétricas son `MiscVal` (skew 22.0), `PoolArea` (14.4), `LotArea` (12.0) y
`3SsnPorch` (9.8).

Nueve columnas son cuasi-constantes (más del 97 % en un solo valor): `Utilities` es 99.9 %
`AllPub`, `Street` es 99.7 % `Pave`, `Condition2` es 99.1 % `Norm`.

### Valores nulos

**Hallazgo de parsing.** `pandas.read_csv` incluye los literales `"NA"` y `"None"` en su lista de
nulos por defecto. En Ames, `NA` significa *"la vivienda no tiene esa característica"* y `None` es
un valor legítimo de `MasVnrType`. La lectura ingenua convierte 677 categorías válidas en `NaN`:

| | Lectura por defecto | Lectura correcta |
|---|---|---|
| `MasVnrType` = `None` | 0 | 677 |
| `MasVnrType` = `NaN` | 683 | 6 |

Con la lectura ingenua, `MasVnrType` aparenta un 58 % de nulos y sería candidata a eliminarse;
en realidad tiene 6 faltantes genuinos, que además coinciden exactamente con los 6 nulos de
`MasVnrArea`. Todo el proyecto lee con `keep_default_na=False, na_values=[""]`.

**Inconsistencia de formato.** Tres columnas (`MSZoning`, `Exterior1st`, `Exterior2nd`) contienen
comillas simples literales en los valores con espacios (`'Wd Sdng'`). Sin limpiar, el mismo nivel
queda representado de dos formas y genera columnas dummy espurias.

**Clasificación de los 19 nulos restantes.** La distinción entre `NA` estructural y faltante
genuino se verificó cruzando cada bloque contra su variable de área:

| Verificación | Resultado | n |
|---|---|---|
| `TotalBsmtSF == 0` cuando `BsmtQual` es nulo | True | 28 |
| `GarageArea == 0` cuando `GarageType` es nulo | True | 64 |
| `PoolArea == 0` cuando `PoolQC` es nulo | True | 1162 |

Los tres al 100 %. El único faltante genuino relevante es `LotFrontage` (217 nulos, 18.6 %). Un
t-test de `SalePrice` entre las casas con y sin el dato da p = 0.856: la ausencia no se asocia al
precio, compatible con un mecanismo MAR/MCAR. La mediana de `LotFrontage` varía entre 21.0 y 91.5
pies según el barrio, lo que justifica imputar por mediana de grupo y no global.

**Figura:** `reports/figures/02_valores_nulos.png`

### Valores atípicos e inconsistencias

Se identifican dos observaciones influyentes:

| Id | GrLivArea | OverallQual | SaleCondition | SalePrice |
|---|---|---|---|---|
| 524 | 4,676 | 10 | Partial | 184,750 |
| 1299 | 5,642 | 10 | Partial | 160,000 |

Son ventas parciales de obra sin terminar: el precio registrado no corresponde a la vivienda
terminada. Rompen la relación área–precio.

Se ejecutaron 12 chequeos de integridad lógica; dos encontraron violaciones, ambas menores y
plausibles tras inspección:

- **5 garajes anteriores a la vivienda** (desfases de 1 a 10 años en casas antiguas o
  remodeladas). Plausible si el garaje proviene de una estructura previa del lote.
- **4 viviendas sin dormitorios sobre rasante** (`MSSubClass` 20/40/80/120), con 3 a 5
  habitaciones y precios normales o altos. Son plantas atípicas, no errores.

Ninguna justifica descartar filas. No se recortan outliers por regla del IQR: son viviendas
legítimas y el dataset de prueba contendrá casos igualmente extremos.

**Figuras:** `03_outliers_bivariados.png`, `04_outliers_boxplots.png`

### Correlaciones y relación con el target

Ranking de features numéricas (Pearson con `SalePrice`):

| Feature | r |
|---|---|
| `OverallQual` | 0.786 |
| `GrLivArea` | 0.696 |
| `GarageCars` | 0.652 |
| `TotalBsmtSF` | 0.630 |

Para las categóricas se usa la razón de correlación η², que mide la fracción de varianza de
`SalePrice` explicada por la partición:

| Feature | η² |
|---|---|
| `Neighborhood` | 0.53 |
| `KitchenQual` | 0.46 |
| `BsmtQual` | 0.46 |
| `ExterQual` | 0.45 |

**Multicolinealidad.** Seis pares con \|r\| ≥ 0.65, todos entre features que miden lo mismo:
`GarageArea`/`GarageCars` (0.88), `1stFlrSF`/`TotalBsmtSF` (0.83), `GrLivArea`/`TotRmsAbvGrd`
(0.82), `GarageYrBlt`/`YearBuilt` (0.82).

**No linealidad.** Los ajustes LOWESS de `OverallQual` y `TotalSF` contra el precio se curvan
hacia arriba: pasar de calidad 9 a 10 aporta mucho más que pasar de 4 a 5. Esto justifica el MLP
frente a un modelo lineal, y se confirma empíricamente en la sección 2.3 (Ridge queda en 47,042
contra 28,375 del MLP).

**Figuras:** `05_matriz_correlacion.png`, `06_ranking_features.png`, `07_features_vs_target.png`,
`08_categoricas_vs_target.png`, `09_neighborhood.png`

### Decisiones de preprocesamiento derivadas del EDA

Cada decisión se clasifica como **firme** (corrige un defecto objetivo del dato) o **hipótesis**
(requiere validación empírica en la sección 2.3).

| # | Decisión | Justificación | Estado |
|---|---|---|---|
| 0 | Leer con `keep_default_na=False` | Recupera 677 categorías legítimas | Firme |
| 1 | Eliminar `Id` | Identificador sin señal | Firme |
| 2 | Quitar comillas literales | 3 columnas afectadas | Firme |
| 3 | `MSSubClass` como nominal | Es un código, no una magnitud | Firme |
| 4 | Mapeo ordinal para 21 columnas | Conserva el orden, ahorra 67 columnas | Firme |
| 5 | `NA` estructural → `"None"` / 0 | Verificado al 100 % contra las áreas | Firme |
| 6 | `LotFrontage` por mediana de barrio | p = 0.856; rango 21–91.5 ft entre barrios | Firme |
| 7 | `Electrical`/`MasVnrType`/`MasVnrArea` por moda | 1, 6 y 6 nulos | Firme |
| 8 | Estandarizar las numéricas | Escalas de 10⁰ a 10⁵ | Firme |
| 9 | Ajustar el escalador solo con train | Evita fuga de información | Firme |
| 10 | One-hot con `handle_unknown="ignore"` | Protege contra categorías nuevas | Firme |
| 11 | No recortar outliers por IQR | Son viviendas legítimas | Firme |
| 12 | Agrupar categorías con n < 5 | 42 niveles raros en 17 columnas | Hipótesis |
| 13 | Descartar las 9 cuasi-constantes | Varianza casi nula | Hipótesis |
| 14 | `log1p` a numéricas con skew > 0.75 | `MiscVal` skew 22.0 | Hipótesis |
| 15 | Entrenar sobre `log1p(SalePrice)` | Skew 1.74 → 0.12 | Hipótesis |
| 16 | Añadir `TotalSF`, `TotalBaths`, `HouseAge` | Condensan columnas colineales | Hipótesis |
| 17 | Evaluar quitar `Id` 524 y 1299 | Ventas `Partial` que rompen la relación | Hipótesis |

### Restricción dominante

Tras codificar, la dimensionalidad ronda las **232 features para 1168 observaciones**: una razón
de aproximadamente 5:1. En ese régimen el riesgo principal no es la falta de capacidad del modelo
sino el **sobreajuste**. Esta predicción del EDA se confirmó: la ablación que más degrada el
resultado en la sección 2.3 es justamente quitar toda la regularización (+6,809 USD).

---

## 2.2 Metodología de desarrollo

Notebook: [`notebooks/02_split.ipynb`](notebooks/02_split.ipynb) · Código: [`src/`](src/)

### Arquitectura considerada

MLP implementado en PyTorch, configurable en profundidad, ancho, activación y regularización:

```
Entrada (n features)
  → [Linear(h) → BatchNorm1d(h) → Activación → Dropout(p)] × k capas
  → Linear(1)
```

Se evaluaron siete arquitecturas: `(64,)`, `(128,)`, `(128,64)`, `(256,128)`, `(512,256)`,
`(256,128,64)` y `(128,64,32)`, todas con la misma regularización activa. Los resultados están en
la sección 2.3.

### Estrategia de división de datos

La estrategia no se asumió: se midió cuál es capaz de distinguir una mejora real del ruido.

**Ruido de un holdout simple.** La misma configuración entrenada con 12 particiones 80/20
distintas produce:

| | Valor |
|---|---|
| RMSE medio | 24,687 |
| Desviación estándar | 4,353 |
| Rango | 17,829 – 30,351 |
| Amplitud | 12,522 USD |

Un esquema que varía 12,522 USD por azar no puede validar mejoras del orden de unos pocos miles.

**Corrección del estimador.** El RMSE de validación cruzada se calcula agrupando las predicciones
out-of-fold en un solo vector, no promediando el RMSE de cada fold. Promediar sesga la comparación
por la concavidad de la raíz cuadrada: repartir los errores uniformemente entre folds eleva el
promedio aunque el error total sea idéntico. El sesgo medido fue de −2,572 USD para k=3 y −375
para k=10, lo que hacía incomparables distintos valores de k y penalizaba artificialmente a la
estratificación.

**Origen de la varianza.** El diagnóstico de errores out-of-fold revela que el RMSE está dominado
por muy pocas observaciones:

| Observaciones | % del MSE total |
|---|---|
| 5 peores | 28.5 % |
| 10 peores | 39.2 % |
| 25 peores | 50.8 % |

El error mediano es de 10,585 USD mientras el RMSE es 25,949: 2.5 veces mayor. Entre las peores
predicciones aparecen las dos anomalías que el EDA había marcado (`Id` 524 y 1299), y cuatro de
las seis peores son ventas `Partial`.

**Consecuencia sobre la partición.** Un primer intento estratificado por decil dejó las 8
viviendas más caras del dataset del lado de `train_dev`, produciendo un test interno no
representativo. Se detectó porque el error mediano era casi idéntico en ambas particiones (10,263
vs 10,585) mientras el RMSE difería en 9,000 USD. La partición definitiva usa 20 cuantiles y la
semilla se elige por representatividad de la distribución de precios —criterio que no involucra
ningún modelo:

| | dev | test | global |
|---|---|---|---|
| Media | 181,505 | 181,093 | 181,442 |
| P95 | 327,300 | 326,218 | 327,300 |
| Top-8 más caras | 7 | 1 | 8 |

**Esquema adoptado:**

```
train.csv (1168)
├── train_dev (85 %, 992)    → KFold(5) × 3 semillas, RMSE out-of-fold agrupado
│                              decide TODAS las iteraciones
└── test interno (15 %, 176) → estimación final, se mide UNA sola vez
```

El holdout inicial se estratifica; la validación cruzada interna no (mostró menor variación entre
semillas: 516 vs 2,090 USD sobre el dataset completo). La partición se congela por `Id` en
`data/processed/split.json`.

**Umbral de significancia.** El ruido se mide sobre `train_dev`, que es donde se deciden las
iteraciones:

| | Valor |
|---|---|
| RMSE con 6 semillas | 29,045 |
| Desviación entre semillas | 2,662 |
| Rango | 26,728 – 34,869 |
| Error estándar con 3 semillas | ±1,537 |
| Error estándar con 8 semillas | ±941 |

`train_dev` es 5.2× más ruidoso que el dataset completo, porque concentra las viviendas extremas.
**Una iteración cuenta como mejora real solo si baja el RMSE en más de 3,074 USD** (dos errores
estándar).

**Figuras:** `11_split_estratificacion.png`, `12_curva_aprendizaje.png`, `13_concentracion_error.png`

### Prevención de fuga de información

El pipeline de preprocesamiento se reajusta **dentro de cada fold**, usando exclusivamente su
porción de entrenamiento. Ajustarlo antes de partir filtraría al conjunto de validación las
medianas de imputación y los parámetros de normalización.

En el A/B de outliers, las observaciones se excluyen **solo del entrenamiento**; la validación se
mantiene íntegra. Excluirlas también de validación bajaría el RMSE sin que el modelo mejore.

### Función de pérdida, optimizador e hiperparámetros

| Componente | Valor final | Alternativas evaluadas |
|---|---|---|
| Pérdida | MSE | Huber |
| Optimizador | AdamW | — |
| Tasa de aprendizaje | 1e-3 | 3e-4, 3e-3 |
| Weight decay | 1e-2 | 0, 1e-5, 1e-4, 1e-3 |
| Batch size | 64 | 32, 128 |
| Scheduler | Cosine annealing | ReduceLROnPlateau, ninguno |
| Activación | ReLU | GELU, SiLU |
| Épocas máximas | 600 | 250 |
| Paciencia (early stopping) | 80 | 35 |

**Transformación del target.** Se entrena sobre `log1p(SalePrice)` adicionalmente estandarizado
(media 0, desviación 1), y se revierte con `expm1` antes de calcular cualquier métrica. La
estandarización del target no es cosmética: sin ella la red arranca prediciendo 0 mientras el
objetivo en escala logarítmica vale ~12, y necesita miles de pasos solo para desplazar el sesgo.
Corregirlo llevó el RMSE de 75,000 a 26,342 con la misma arquitectura.

### Técnicas de regularización

Dropout, batch normalization, weight decay y early stopping sobre el RMSE de validación. La
contribución individual de cada una se cuantifica por ablación en la sección 2.3.

---

## 2.3 Resultados de iteraciones

Notebook: [`notebooks/04_iteraciones.ipynb`](notebooks/04_iteraciones.ipynb) ·
Registro completo: [`reports/experiments.csv`](reports/experiments.csv)

Protocolo: `KFold(5)` × 3 semillas sobre `train_dev`, RMSE out-of-fold agrupado. Umbral de mejora
real: 3,074 USD.

### Baselines de referencia

Medidos con el mismo protocolo ([`notebooks/03_baselines.ipynb`](notebooks/03_baselines.ipynb)):

| Modelo | RMSE | σ entre semillas |
|---|---|---|
| GradientBoosting (500) | 31,877 | 2,456 |
| RandomForest (400) | 33,367 | 1,433 |
| Ridge (α=10) | 47,042 | 10,707 |
| ElasticNet | 47,258 | 13,071 |
| Media (Dummy) | 77,786 | 44 |
| Mediana (Dummy) | 79,514 | 50 |

**Figura:** `14_baselines.png`

### Punto de partida y ablaciones

| Id | Configuración | Val | Train | Gap |
|---|---|---|---|---|
| it00 | MLP mínimo sin regularizar | 34,564 | 22,738 | 11,825 |
| it01 | **Referencia:** bn + dropout 0.2 + wd 1e-4 | 33,624 | 15,316 | 18,308 |

Ablaciones sobre la referencia (Δ positivo = quitar el componente empeora):

| Id | Ablación | Δ RMSE | Δ gap | Veredicto |
|---|---|---|---|---|
| it02 | sin batch norm | +3,606 | +3,284 | **necesario** |
| it03 | sin dropout | +1,625 | +481 | dentro del ruido |
| it04 | sin weight decay | −55 | −82 | indiferente |
| it05 | sin `log1p` en el target | −399 | +1,459 | dentro del ruido |
| it06 | sin ninguna regularización | +6,809 | +6,117 | **necesario** |
| it07 | preprocesamiento mínimo | +3,330 | +324 | **necesario** |

Batch normalization y el preprocesamiento derivado del EDA superan el umbral individualmente.
Quitar toda la regularización a la vez cuesta 6,809 USD, más que la suma de las ablaciones
individuales, lo que indica que los componentes actúan de forma complementaria.

`it05` ilustra el valor del umbral: quitar la transformación logarítmica *parece* mejorar en 399
USD, pero está muy por debajo del ruido. Sin el umbral se habría concluido erróneamente que la
transformación perjudica.

### Arquitectura

| Id | Arquitectura | Pesos | Val | Gap |
|---|---|---|---|---|
| it10 | (128, 64) | 34,113 | **32,096** | 14,068 |
| it12 | (256, 128, 64) | 93,505 | 32,194 | 15,004 |
| it08 | (64,) | 13,057 | 32,349 | 15,153 |
| it09 | (128,) | 26,113 | 32,351 | 18,702 |
| it13 | (128, 64, 32) | 36,225 | 32,378 | 13,329 |
| it11 | (512, 256) | 235,777 | 32,846 | 17,301 |
| it01 | (256, 128) | 85,377 | 33,624 | 18,308 |

`(512, 256)` tiene siete veces más parámetros que `(128, 64)` y obtiene un resultado peor, con un
gap 23 % mayor. Con 992 observaciones de entrenamiento, la capacidad adicional se destina a
memorizar. **Todas las arquitecturas caen dentro de un rango de 1,528 USD**, por debajo del
umbral: la arquitectura no es el factor determinante.

### Intensidad de la regularización

| Id | Configuración | Val | Gap |
|---|---|---|---|
| it14 | dropout 0.1 | 32,192 | 18,647 |
| it15 | dropout 0.35 | **31,827** | 10,493 |
| it16 | dropout 0.5 | 34,577 | 9,214 |
| it17 | dropout 0.65 | 35,288 | **8,526** |
| it18 | weight decay 1e-5 | 31,905 | 10,518 |
| it19 | weight decay 1e-3 | 31,975 | 9,779 |
| it20 | weight decay 1e-2 | **31,766** | 10,218 |

`it17` tiene el gap más bajo de todo el barrido y el peor RMSE. Sobre-regularizar no elimina el
sobreajuste: eleva el error de entrenamiento hasta igualar el de validación. El gap es un
diagnóstico, no un objetivo.

**Figura:** `15_regularizacion.png`

### Preprocesamiento

| Id | Variante | Δ vs. completo | Veredicto |
|---|---|---|---|
| it21 | sin features derivadas | +2,286 | dentro del ruido |
| it22 | sin `log1p` a sesgadas | +748 | dentro del ruido |
| it23 | sin agrupar categorías raras | −571 | dentro del ruido |
| it24 | conservando cuasi-constantes | +4,184 | **hace falta descartarlas** |
| it25 | agrupación de raras más agresiva | +12 | dentro del ruido |

De las hipótesis del EDA, solo el descarte de columnas cuasi-constantes supera el umbral.

### A/B de valores atípicos

| Id | Exclusión (solo del entrenamiento) | n | Δ | Veredicto |
|---|---|---|---|---|
| it26 | puntos influyentes (`Id` 524, 1299) | 1 | +307 | dentro del ruido |
| it27 | `Partial` de >3000 ft² | 11 | +182 | dentro del ruido |
| it28 | ambos grupos | 11 | +182 | dentro del ruido |
| it29 | el 0.5 % más caro | 5 | +361 | dentro del ruido |

**Ninguna forma de excluir outliers mejoró el resultado.** Era el experimento con mayor impacto
esperado según el diagnóstico de la sección 2.2, y el resultado fue negativo. Se conservan todas
las observaciones.

### Optimización

| Id | Cambio | Val | Gap |
|---|---|---|---|
| it30 | pérdida Huber | 32,471 | **4,596** |
| it31 | batch size 32 | 34,578 | 14,958 |
| it32 | batch size 128 | 31,690 | 9,610 |
| it33 | lr 3e-4 | 34,653 | 10,628 |
| it34 | lr 3e-3 | 31,714 | 11,351 |
| it35 | scheduler cosine | **31,082** | 10,492 |
| it36 | activación GELU | 31,393 | 10,872 |
| it37 | activación SiLU | 31,593 | 12,220 |
| it38 | mejor config, entrenamiento largo | **30,282** | 13,309 |

La pérdida Huber produce el gap más bajo de las 39 iteraciones y un RMSE peor: al penalizar menos
los errores grandes, el modelo deja de esforzarse en las viviendas caras, que son justamente las
que dominan la métrica.

### Confirmación de los finalistas

Con 3 semillas la incertidumbre es de ±1,537 USD, del mismo orden que las diferencias entre las
mejores iteraciones. Los 5 finalistas se re-evaluaron con 8 semillas (±941):

| Id | Descripción | 3 semillas | 8 semillas | Error estándar |
|---|---|---|---|---|
| it38 | mejor config, entrenamiento largo | 30,282 | **28,375** | ±892 |
| it37 | activación SiLU | 31,593 | 29,170 | ±1,001 |
| it35 | scheduler cosine | 31,082 | 29,373 | ±908 |
| it36 | activación GELU | 31,393 | 29,666 | ±862 |
| it23 | sin agrupar categorías raras | 31,195 | 30,608 | ±870 |

Los cinco valores bajaron al medir con más semillas. La ventaja de `it38` sobre `it37` es de 795
USD, **por debajo del error estándar**: están estadísticamente empatados. Se selecciona `it38` por
ser el mejor punto estimado, no porque la diferencia sea significativa.

**Configuración ganadora:** `hidden=(128, 64)`, dropout 0.35, batch norm, weight decay 1e-2,
scheduler cosine, activación ReLU, batch size 64, lr 1e-3, target en `log1p` estandarizado, con
features derivadas y descarte de cuasi-constantes.

### Curvas de entrenamiento

**Figuras:** `16_curvas_entrenamiento.png`, `17_iteraciones_gap.png`

Las curvas de `it00`, `it01` e `it38` muestran el patrón esperado: ambas curvas descienden al
inicio, la de entrenamiento continúa bajando mientras la de validación se aplana. La separación
vertical entre curvas es el sobreajuste. Early stopping seleccionó la época 143 en el modelo final
de 600 máximas.

### Problemas encontrados durante el desarrollo

| Problema | Síntoma | Causa | Solución |
|---|---|---|---|
| La red no aprendía | RMSE 75,000 ≈ predecir la media | Target en escala log valía ~12, la red arrancaba en 0 | Estandarizar el target |
| Estimador de CV sesgado | 3-fold daba 36,317 y 10-fold 25,672 | Promediar RMSE por fold; la raíz es cóncava | Agrupar predicciones out-of-fold |
| Test interno no representativo | Mismo modelo: 25,949 global vs 34,820 en dev | Las 8 casas más caras del mismo lado | 20 cuantiles y semilla por representatividad |
| Umbral incorrecto por 5× | — | Ruido medido sobre 1168 filas, no sobre `train_dev` | Medir donde se decide; añadir confirmación con 8 semillas |
| Barrido sesgado | Redes grandes daban 49,000 | Arquitectura barrida sin regularización activa | Ablaciones sobre referencia regularizada |
| Pipelines incompatibles | `shapes cannot be multiplied (120x171 and 180x512)` | Validación transformada con pipeline distinto al de entrenamiento | Usar el mismo pipeline en ambos |

---

## 2.4 Discusión de resultados

### Qué cambios tuvieron mayor impacto

Ordenados por magnitud del efecto medido:

1. **Estandarizar el target (−48,658 USD).** No es un hiperparámetro sino la corrección de un
   defecto de implementación, pero es con diferencia el cambio de mayor impacto del proyecto. Sin
   él la red no converge en ningún número razonable de épocas.
2. **Regularización en conjunto (−6,809 USD).** La ablación completa confirma la predicción del
   EDA: con una razón de 5:1 entre features y observaciones, el sobreajuste domina.
3. **Batch normalization (−3,606 USD).** El único componente de regularización que supera el
   umbral por sí solo.
4. **Preprocesamiento derivado del EDA (−3,330 USD).** El trabajo de la sección 2.1 se traduce en
   una mejora medible.
5. **Entrenamiento prolongado (−800 USD respecto de `it35`).** Elevar el máximo de épocas de 250 a
   600 con paciencia 80 permitió que el scheduler cosine completara su ciclo.

Lo que **no** tuvo impacto medible: la arquitectura (todas dentro de 1,528 USD), la exclusión de
outliers (las cuatro variantes dentro del ruido), la función de activación, y la mayoría de las
hipótesis de feature engineering del EDA.

Este reparto es en sí un resultado. La intuición sugiere que en deep learning la arquitectura es
la palanca principal; en un problema tabular con menos de mil observaciones, la evidencia dice que
lo determinante es el control de la capacidad efectiva y la calidad del preprocesamiento.

### Por qué falló el A/B de outliers

El diagnóstico de la sección 2.2 mostró que 24 observaciones explican la mitad del MSE, lo que
sugería que eliminarlas del entrenamiento produciría la mayor ganancia del proyecto. No ocurrió:
las cuatro variantes quedaron dentro del ruido.

La interpretación es que esas viviendas no están *contaminando* el aprendizaje —el modelo no
aprende peor por su culpa—, sino que son **intrínsecamente impredecibles con las features
disponibles**. Una vivienda de 5,642 ft² con calidad 10 vendida a 160,000 USD no tiene ninguna
señal en el dataset que anticipe ese precio; la explicación (`SaleCondition = Partial`, obra sin
terminar) está registrada pero no es suficiente para estimar cuánto descuento implica.

Excluirlas del entrenamiento no mejora la predicción de las *otras* viviendas extremas, que es lo
que haría falta. Y como el dataset de la competencia contendrá casos análogos, entrenar sin ellos
solo empeoraría la capacidad de manejarlos.

### Análisis de errores del modelo final

Sobre las 176 viviendas del test interno:

| Métrica | Valor |
|---|---|
| RMSE | 24,197 |
| MAE | 16,155 |
| Error mediano | 10,342 |
| MAPE | 9.0 % |
| R² | 0.894 |

Desglose por quintil de precio:

| Quintil | n | Precio medio | RMSE | Error relativo | Sesgo |
|---|---|---|---|---|---|
| Q1 | 36 | 102,641 | 18,387 | 14.1 % | **+4,904** |
| Q2 | 35 | 136,844 | 7,655 | 4.7 % | −1,613 |
| Q3 | 35 | 164,789 | 16,285 | 7.1 % | −5,909 |
| Q4 | 35 | 202,273 | 23,192 | 8.8 % | −4,983 |
| Q5 | 35 | 301,161 | 41,652 | 10.3 % | **−13,377** |

**Patrón de regresión a la media.** El sesgo pasa de +4,904 en el quintil más barato a −13,377 en
el más caro: el modelo **sobreestima las viviendas baratas y subestima las caras**, comprimiendo
sus predicciones hacia el centro de la distribución.

Tiene dos causas concurrentes. Primera, la densidad de datos: la mayoría de las observaciones está
en el rango medio, y minimizar el error cuadrático total incentiva a acertar donde hay más masa.
Segunda, la escasez de ejemplos en los extremos: hay pocas viviendas por encima de 400,000 USD, y
las features disponibles no distinguen bien entre una de 400,000 y una de 600,000.

El error relativo es peor en Q1 (14.1 %) que en Q5 (10.3 %), pero el error absoluto es mucho mayor
en Q5. Dado que la métrica de la competencia es RMSE en dólares, **el desempeño en el quintil
superior es el que más pesa en la nota**, pese a representar solo el 20 % de las observaciones.

**Figura:** `18_analisis_errores.png`

### Discrepancia entre la CV y el test interno

La validación cruzada estima 28,375 USD y el test interno da 24,197: una diferencia de 4,177 USD a
favor del test.

La explicación es el tamaño y la composición del test. Con 176 observaciones, y sabiendo que unas
pocas viviendas extremas determinan el RMSE, la estimación es inherentemente volátil. El test
contiene solo 1 de las 8 viviendas más caras del dataset (proporcional, pero con altísima varianza
en un n tan chico).

**La estimación por validación cruzada es la referencia más confiable** para anticipar el
desempeño en la competencia: usa las 992 observaciones de `train_dev` y promedia 8 semillas, con
un error estándar de ±892 USD frente a la única medición del test.

### Limitaciones del enfoque y del dataset

**Del dataset.** Con 1168 observaciones y 232 features tras codificar, la razón 5:1 limita
estructuralmente lo que cualquier modelo de alta capacidad puede lograr. La cobertura temporal
(2006–2010) incluye la crisis inmobiliaria estadounidense, y el año 2010 solo llega a mitad de
año. Todas las viviendas pertenecen a una sola ciudad, de modo que el modelo no generaliza fuera
de Ames.

**De la métrica.** El RMSE está dominado por un 2 % de las observaciones. Esto implica que el
resultado de la competencia dependerá considerablemente de qué viviendas extremas contenga el
conjunto de prueba —una fuente de varianza que ningún participante puede controlar ni optimizar.

**Del modelo.** Un MLP sobre datos tabulares compite en desventaja frente a los métodos de
ensamble de árboles, que manejan interacciones y no linealidades sin necesitar codificación
explícita ni escalado. Que el MLP haya superado a GradientBoosting aquí (28,375 vs 31,877) se debe
en buena medida al trabajo de preprocesamiento, no a una ventaja intrínseca de la arquitectura.

**Del procedimiento.** La búsqueda por coordenadas explora una fracción del espacio de
hiperparámetros y no captura interacciones entre ellos. Una búsqueda bayesiana habría sido más
exhaustiva, pero con un ruido de ±1,537 USD por configuración, la capacidad de discriminar entre
candidatos cercanos habría seguido siendo el cuello de botella.

### Compromiso entre complejidad y generalización

Los resultados permiten cuantificar este compromiso de forma directa. La arquitectura `(512, 256)`
tiene 235,777 parámetros —237 por cada observación de entrenamiento— y obtiene 32,846 USD con un
gap de 17,301. La arquitectura `(128, 64)` tiene 34,113 parámetros y obtiene 32,096 con un gap de
14,068. **Multiplicar la capacidad por siete empeora tanto el error como la generalización.**

El barrido de dropout muestra la otra cara. Al aumentarlo de 0.35 a 0.65 el gap cae de 10,493 a
8,526 —el modelo memoriza menos— pero el RMSE sube de 31,827 a 35,288. La reducción del gap no
proviene de mejorar la validación sino de degradar el entrenamiento.

La conclusión operativa es que el gap sirve para **diagnosticar** en qué régimen está el modelo,
pero optimizarlo directamente lleva a soluciones sub-regularizadas o sobre-regularizadas. El
criterio de selección debe ser siempre el error de validación, usando el gap para entender *por
qué* una configuración funciona.

---

## 2.5 Conclusiones

### Desempeño final e interpretación

El modelo final obtiene **28,375 ± 892 USD de RMSE** por validación cruzada y **24,197 USD** sobre
el test interno. Frente al mejor baseline clásico (GradientBoosting, 31,877 USD) la mejora es de
3,502 USD y sobrevive a dos errores estándar, por lo que es estadísticamente significativa.

En términos del problema: sobre una vivienda de precio mediano (165,000 USD), el error típico
—medido por la mediana del error absoluto— es de **10,342 USD, un 6.3 %**. El MAPE global es del
9.0 % y el modelo explica el 89.4 % de la varianza de los precios.

La brecha entre el error mediano (10,342) y el RMSE (24,197) es en sí el resultado más
característico del problema: en la vivienda típica el modelo funciona bien, y la métrica está
determinada por un puñado de casos atípicos.

### Principales aprendizajes técnicos

**Presentar bien el objetivo importa más que la arquitectura.** Estandarizar el target mejoró el
RMSE en 48,658 USD; el mejor cambio de arquitectura aportó 1,528. Un modelo que no converge no es
un problema de capacidad.

**El estimador debe validarse antes que el modelo.** Dos de los seis errores del proyecto fueron
del procedimiento de medición, no del modelo: promediar RMSE por fold y medir el ruido sobre la
partición equivocada. Ambos habrían producido conclusiones erróneas que el modelo no podía
corregir.

**Sin cuantificar el ruido no hay conclusiones válidas.** Con un umbral de 3,074 USD, 31 de las 39
iteraciones resultaron indistinguibles del azar. Con el umbral incorrecto de 516 USD que usé
inicialmente, habría reportado como mejoras un conjunto de diferencias que eran ruido de
partición.

**El gap diagnostica, no optimiza.** Las configuraciones con menor gap (`it17` con dropout 0.65,
`it30` con Huber) están entre las de peor RMSE.

### Aprendizajes metodológicos

**Verificar antes que asumir.** El `NA` estructural se confirmó cruzando contra las áreas
correspondientes; la imputación de `LotFrontage` se justificó con un t-test; la representatividad
de la partición se comprobó contra la distribución de precios. Ninguna de estas verificaciones era
obligatoria, y todas cambiaron una decisión.

**Los resultados que sorprenden suelen ser bugs.** Cinco de los seis problemas se detectaron
porque dos números que debían coincidir no coincidían. Aceptar un resultado inesperado como
"así son los datos" habría dejado los seis sin detectar.

**Los resultados negativos también son resultados.** El A/B de outliers, que era el experimento de
mayor impacto esperado, no produjo mejora. Documentarlo evita repetirlo y aporta una comprensión
del problema —esas viviendas son impredecibles, no ruidosas— que un resultado positivo no habría
dado.

### Trabajo futuro

Con más tiempo o recursos, en orden de retorno esperado:

1. **Ensamblar el MLP con GradientBoosting.** Los dos modelos cometen errores distintos (el MLP
   captura relaciones suaves, los árboles capturan umbrales), así que promediarlos suele superar a
   ambos. Es la vía de mejora más probable y no se exploró por restricción de alcance del
   enunciado, que pide implementar un MLP.
2. **Embeddings para las categóricas de alta cardinalidad.** `Neighborhood` es la feature
   categórica más predictiva (η² = 0.53) y actualmente consume 25 columnas one-hot. Un embedding
   aprendido de dimensión 4–8 reduciría la dimensionalidad y permitiría que barrios similares
   compartan información.
3. **Más semillas por configuración.** Con ±892 USD de error estándar, las cinco mejores
   iteraciones están empatadas. Aumentar a 20 semillas reduciría la incertidumbre a ±595 y
   permitiría discriminar entre ellas.
4. **Modelado explícito de `SaleCondition`.** Cuatro de las seis peores predicciones son ventas
   `Partial`. Un modelo separado, o una feature de interacción entre `SaleCondition` y el tamaño,
   podría capturar el descuento sistemático de la obra sin terminar.
5. **Búsqueda bayesiana de hiperparámetros** sobre el espacio conjunto, en lugar de por
   coordenadas, para capturar interacciones entre profundidad, dropout y weight decay.

### Expectativa para la competencia

La estimación realista es **~28,000 USD de RMSE**, no los 24,197 del test interno. La varianza
esperada es alta y depende de la composición del conjunto de prueba: si contiene varias viviendas
por encima de 500,000 USD, el RMSE de todos los participantes subirá; si contiene pocas, bajará.
Esta sensibilidad es una propiedad de la métrica combinada con la distribución del dataset, y no
es optimizable desde el modelo.

---

## 2.6 Repositorio

**https://github.com/ecarcamo/CC3092-DEEP-LEARNING** — directorio [`pry1/`](.)

### Contenido

| Ruta | Descripción |
|---|---|
| `notebooks/01_eda.ipynb` | Análisis exploratorio (§2.1) |
| `notebooks/02_split.ipynb` | Estrategia de división (§2.2) |
| `notebooks/03_baselines.ipynb` | Modelos de referencia |
| `notebooks/04_iteraciones.ipynb` | 39 iteraciones documentadas (§2.3) |
| `notebooks/05_final.ipynb` | Modelo final y análisis de errores |
| `src/` | Pipeline, modelo y validación cruzada compartidos |
| `predict.py` | Generación de predicciones |
| `models/` | Pipeline, pesos y metadatos serializados |
| `reports/experiments.csv` | Bitácora completa de experimentos |
| `reports/figures/` | 18 figuras |

### Reproducir

```bash
cd pry1
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd notebooks
for nb in 01_eda 02_split 03_baselines 04_iteraciones 05_final; do
  ../venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

Los notebooks se ejecutan en orden: cada uno consume artefactos del anterior. La corrida completa
toma aproximadamente 25 minutos con GPU.

### Generar predicciones

```bash
python predict.py --input <dataset_de_prueba.csv>
```

Escribe `submissions/predicciones.csv`. Si el CSV de entrada incluye `SalePrice`, reporta además
el RMSE. El script verifica que el pipeline y el modelo correspondan entre sí antes de predecir.

---

## Anexo — Validación con el archivo de muestra y reentrenamiento final (16 de agosto de 2026)

*Sección añadida el día previo a la competencia, después de que el catedrático entregara un
archivo de muestra con el formato del dataset de prueba. No modifica ni reemplaza las secciones
2.1–2.6 anteriores, que documentan el trabajo tal como se entregó originalmente.*

### Qué se recibió

`data/raw/pipeline_test.csv` (5 viviendas, mismas 79 columnas que `train.csv` sin `SalePrice`) y
`data/raw/expected_output.csv`, con la instrucción explícita del catedrático: *"El formato del
output debe ser idéntico al formato en el archivo `expected_output.csv`. La columna `ID` es la
misma columna `ID` del archivo de muestra."*

**Aclaración importante:** los valores de `Prediction` en `expected_output.csv` son `1, 2, 3, 4, 5`
— un marcador de posición para mostrar la forma del archivo, no precios reales. Por lo tanto este
archivo sirve para validar **formato de salida**, no para calcular un RMSE de precisión. No se
usó, ni se podía usar, como conjunto de validación.

### Hallazgo: el formato de salida no coincidía

`predict.py` escribía la columna de salida como `Id,SalePrice`. El formato exigido es
`Id,Prediction`. Es un defecto real, no una hipótesis: de no corregirse, el script habría
producido un archivo con el nombre de columna equivocado el día de la competencia.

**Arreglo:** una constante `OUTPUT_COL = "Prediction"` en `predict.py`, que separa el nombre de la
columna de *salida* (`Prediction`, exigido por el formato de la competencia) del nombre de la
columna del *target* al leer datos de entrenamiento (`SalePrice`, que sigue siendo `TARGET` en
`src/data.py` porque así se llama en `train.csv`). Verificado contra `expected_output.csv`:
mismas columnas, mismos `Id` y en el mismo orden.

### Decisión: reentrenar el modelo final con el 100 % de `train.csv`

El modelo en `models/` se había entrenado únicamente con `train_dev` (992 de 1168 filas, 85 %),
porque el 15 % restante —el test interno de 176 filas— se reservó para medir el RMSE una sola vez
sin fuga de información (§2.2). Esa medición ya se hizo y ya está documentada
(RMSE test interno = 24,197 USD, §2.4). No queda ninguna decisión de modelado pendiente que ese
15 % pudiera contaminar, así que usarlo ahora para entrenar el modelo que se despliega no es fuga
de información hacia una decisión — no hay decisión que tomar.

Se reentrenó `models/final_pipeline.joblib` y `models/final_model.pt` con las 1168 filas
completas, en el notebook nuevo [`06_reentrenamiento_final.ipynb`](notebooks/06_reentrenamiento_final.ipynb).
**La configuración ganadora (`it38`) se mantuvo exactamente igual** — mismo `hidden=(128, 64)`,
dropout 0.35, batch norm, weight decay 1e-2, scheduler cosine, target en `log1p` estandarizado. No
se realizó ninguna búsqueda de hiperparámetros nueva ni se tocó ninguna decisión de la sección
2.3: la única variable que cambia es la cantidad de datos que ve el modelo desplegado.

**Por qué esto no aumenta el riesgo de sobreajuste.** El sobreajuste, documentado extensamente en
la sección 2.4, viene de tener poca capacidad reguladora frente a muchas features (~232 features
para 992 observaciones). Añadir un 17.7 % más de datos (992 → 1168) con exactamente la misma
regularización relaja esa razón, no la empeora. El riesgo que sí existe —y que se reconoce como
límite abajo— es no volver a medir el RMSE de forma independiente después de este cambio.

### Qué cambió técnicamente

| | Modelo anterior | Modelo reentrenado |
|---|---|---|
| Filas de entrenamiento | 992 (`train_dev`) | 1168 (100 % de `train.csv`) |
| Features tras preprocesar | 218 | 219 |
| Épocas de entrenamiento | 144 | 70 |
| Arquitectura / regularización | `(128,64)`, dropout 0.35, bn, wd 1e-2 | **idéntica** |

El número de features cambió de 218 a 219 porque el agrupamiento de categorías raras del pipeline
(`OneHotEncoder`) se reajusta con más datos y una categoría que antes quedaba agrupada ahora tiene
representación propia — es un efecto esperado de ajustar el pipeline sobre un conjunto distinto,
no un cambio de diseño. Las épocas de early stopping (70 vs. 144) provienen de un split de
early-stopping distinto (12 % de 1168 vs. 12 % de 992, semillas iguales) y no son comparables
directamente entre sí; ambas son válidas para su propio conjunto.

### Qué NO cambió y qué NO se recalculó

**La estimación de referencia para la competencia sigue siendo la de la sección 2.5: 28,375 ±
892 USD (RMSE por validación cruzada sobre `train_dev`).** No se recalculó un nuevo RMSE de test
para el modelo reentrenado porque, al usar el 100 % de los datos para entrenar, ya no queda ningún
subconjunto sin fuga de información con el cual medirlo honestamente. El notebook 06 corre un
chequeo de humo (predicciones sobre los datos ya vistos en entrenamiento) únicamente para descartar
errores gruesos —NaNs, escalas absurdas, signos invertidos—, y lo etiqueta explícitamente como *no
es una estimación de generalización* para evitar que se confunda con una métrica válida.

### Verificación de extremo a extremo

El notebook 06 corre `predict.py` sobre `pipeline_test.csv` con el modelo reentrenado y compara la
salida contra `expected_output.csv`: mismas columnas (`Id`, `Prediction`), mismos `Id` en el mismo
orden, 5 filas predichas y 5 esperadas. El flujo completo —lectura del CSV, transformación,
predicción, escritura— funciona sin pasos manuales, tal como debe correr el lunes.

### Limitación reconocida de este cambio

A diferencia del resto del proyecto, donde cada decisión se validó empíricamente contra un umbral
de ruido medido (§2.2–2.3), esta decisión específica —entrenar con más datos manteniendo fija la
configuración— se apoya en un argumento teórico (más datos + misma regularización ⇒ generalización
igual o mejor) y no en una medición nueva, porque medirla habría requerido volver a reservar datos
y perder la ventaja que se buscaba. Es la decisión más razonable disponible el día previo a la
competencia, pero se documenta como tal: un supuesto, no un resultado medido.

### Artefactos afectados

`models/final_pipeline.joblib`, `models/final_model.pt` y `models/metadata.json` se sobrescribieron
con la versión reentrenada. La versión anterior (992 filas) queda recuperable en el historial de
git de este repositorio si hiciera falta revertir.
