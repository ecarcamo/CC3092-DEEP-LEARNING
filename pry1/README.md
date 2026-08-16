# Proyecto 1 — Competencia de Modelación

**CC3092 Deep Learning y sistemas inteligentes**

Predicción del precio de venta de viviendas (dataset **Ames Housing**) mediante un
**Multi-Layer Perceptron**. La métrica de la competencia es **RMSE** sobre un dataset de prueba
held-out que se entrega el día de la presentación.

| | |
|---|---|
| Modalidad | Individual |
| Presentación | 17 de agosto de 2026 |
| Entrega del trabajo escrito | 21 de agosto de 2026 |
| Métrica objetivo | RMSE |
| Modelo | Multi-Layer Perceptron (PyTorch) |

---

## El día de la competencia

```bash
python predict.py --input <dataset_nuevo.csv>
```

Eso es todo. El script carga el pipeline y los pesos de `models/`, aplica **exactamente** las
mismas transformaciones del entrenamiento y escribe `submissions/predicciones.csv`. Si el CSV
trae la columna `SalePrice`, además imprime el RMSE.

Verifica que el pipeline y el modelo correspondan entre sí (compara el número de features) y
falla con un mensaje claro si no. Acepta `--output` para elegir otra ruta.

---

## Estructura

```
pry1/
├── data/
│   ├── raw/train.csv               # dataset entregado (1168 × 81)
│   └── processed/
│       ├── column_types.json       # contrato de tipos derivado del EDA
│       ├── split.json              # partición congelada por Id + umbral de ruido
│       └── mejor_config.json       # configuración ganadora
├── notebooks/
│   ├── 01_eda.ipynb                # §2.1 Análisis exploratorio
│   ├── 02_split.ipynb              # §2.2 Estrategia de división
│   ├── 03_baselines.ipynb          # referencias clásicas
│   ├── 04_iteraciones.ipynb        # §2.3 Bitácora de experimentos
│   └── 05_final.ipynb              # modelo final y análisis de errores
├── src/
│   ├── data.py                     # carga y particiones
│   ├── preprocessing.py            # pipeline sklearn serializable
│   ├── model.py                    # MLP + entrenamiento con early stopping
│   └── experiments.py              # validación cruzada y bitácora
├── models/                         # pipeline + pesos + metadata
├── reports/
│   ├── figures/                    # 18 figuras para el trabajo escrito
│   ├── experiments.csv             # todas las iteraciones
│   └── baselines.csv
├── predict.py                      # script de la competencia
└── requirements.txt
```

---

## Reproducir

Requiere **Python 3.11**.

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

Los notebooks se ejecutan **en orden**: cada uno consume artefactos del anterior. La corrida
completa toma ~25 min con GPU.

> **VS Code:** seleccionar `pry1/venv/bin/python` como intérprete del kernel.

---

## Resultados

| | RMSE (USD) |
|---|---|
| Predecir la media | 77,786 |
| Ridge | 47,042 |
| RandomForest (400) | 33,367 |
| GradientBoosting (500) | 31,877 |
| **MLP final** | **28,375 ± 892** |

RMSE sobre el test interno (176 viviendas nunca vistas): **24,197 USD** — MAPE 9.0 %, R² 0.894.

**Configuración ganadora:** `hidden=(128, 64)`, dropout 0.35, batch norm, weight decay 1e-2,
scheduler cosine, target en `log1p` revertido con `expm1`, features derivadas activadas.

---

## Método

**Partición.** `train.csv` se divide en `train_dev` (85 %, 992) y test interno (15 %, 176). Todas
las iteraciones se deciden sobre `train_dev` con `KFold(5)` y el test interno se toca una única
vez, en el notebook 05.

**Estimador.** El RMSE de validación se calcula **agrupando las predicciones out-of-fold** en un
solo vector, no promediando el RMSE de cada fold. Promediarlos sesga la comparación por la
concavidad de la raíz: repartir los errores uniformemente entre folds sube el promedio aunque
el error total sea idéntico.

**Ruido.** La misma configuración repetida con 6 semillas sobre `train_dev` varía entre 26,728 y
34,869 USD (σ = 2,662). Por eso ninguna iteración se declara mejora sin superar 2 errores
estándar, y los 5 finalistas se re-evalúan con 8 semillas antes de elegir al ganador.

---

## Hallazgos

**Del EDA** ([01_eda.ipynb](notebooks/01_eda.ipynb)):

- `pandas` interpreta los literales `"NA"` y `"None"` como nulos, pero en Ames son categorías
  válidas. La lectura ingenua destruye 677 categorías de `MasVnrType`, que aparenta 58 % de nulos
  cuando en realidad tiene 6. Todo el proyecto lee con `keep_default_na=False, na_values=[""]`.
- `SalePrice` tiene skew 1.74; `log1p` lo lleva a 0.12.
- 19 columnas con nulos, en su mayoría `NA` estructurales verificados contra sus áreas.
  El único faltante genuino relevante es `LotFrontage` (18.6 %).
- ~232 features tras codificar frente a 1168 observaciones (≈5:1): el riesgo dominante es el
  **sobreajuste**.

**Del modelado** ([02_split.ipynb](notebooks/02_split.ipynb), [04_iteraciones.ipynb](notebooks/04_iteraciones.ipynb)):

- **El RMSE está dominado por un puñado de observaciones.** 24 viviendas (2.1 %) explican la
  mitad del MSE. Las peores son las anomalías `Id` 524 y 1299 que el EDA marcó, más ventas
  `Partial`. Esto obliga a estratificar la partición por la cola alta de precios y a promediar
  varias semillas antes de declarar cualquier mejora.
- **La regularización es lo que más mueve la aguja.** Quitar batch norm cuesta +3,600 USD; quitar
  toda la regularización, +6,800. Las ablaciones están en la §A del notebook 04.
- **Más capacidad no ayuda.** `hidden=(512, 256)` es peor que `(128, 64)` pese a tener 10× más
  pesos: con 992 observaciones la red grande memoriza.
- **El ensemble no aportó** (+63 USD, dentro del ruido), así que el modelo final es único —
  más simple de cargar y reproducir.

**Limitación conocida.** El test interno tiene 176 filas y su RMSE (24,197) es sensiblemente
mejor que la estimación por CV (28,375). La diferencia es esperable: contiene solo 1 de las 8
viviendas más caras del dataset. La estimación por validación cruzada es la referencia más
confiable para anticipar el desempeño el día de la competencia.

---

## Estado

| Sección del enunciado | Entregable | Estado |
|---|---|---|
| 2.1 Análisis exploratorio | `01_eda.ipynb` | ✅ |
| 2.2 Metodología de desarrollo | `02_split.ipynb`, `src/` | ✅ |
| 2.3 Resultados de iteraciones | `04_iteraciones.ipynb`, `reports/experiments.csv` | ✅ |
| 2.4 Discusión de resultados | `05_final.ipynb` (insumos) + trabajo escrito | ⬜ |
| 2.5 Conclusiones | Trabajo escrito | ⬜ |
| 2.6 Enlace al repositorio | Este README | ✅ |
