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
| Modelo | Multi-Layer Perceptron (MLP) |

---

## Estructura del repositorio

```
pry1/
├── data/
│   ├── raw/train.csv           # dataset entregado (1168 × 81)
│   └── processed/
│       └── column_types.json   # contrato de preprocesamiento derivado del EDA
├── notebooks/
│   └── 01_eda.ipynb            # §2.1 Análisis exploratorio de datos
├── reports/figures/            # figuras exportadas para el trabajo escrito
├── src/                        # código reutilizable (pipeline, modelo, entrenamiento)
├── models/                     # pesos y pipeline serializados
├── docs/                       # enunciado del proyecto
├── requirements.txt
└── venv/                       # entorno virtual (no versionado)
```

---

## Reproducir los resultados

Requiere **Python 3.11**.

```bash
cd pry1
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ejecutar el EDA:

```bash
jupyter lab notebooks/01_eda.ipynb
```

o sin abrir la interfaz:

```bash
cd notebooks
../venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
```

El notebook regenera todas las figuras en `reports/figures/` y el contrato de preprocesamiento
en `data/processed/column_types.json`.

> **VS Code:** seleccionar el intérprete `pry1/venv/bin/python` como kernel del notebook
> (`Ctrl+Shift+P` → *Python: Select Interpreter*).

---

## Estado del proyecto

| Sección del enunciado | Entregable | Estado |
|---|---|---|
| 2.1 Análisis exploratorio de datos | `notebooks/01_eda.ipynb` | ✅ Completo |
| 2.2 Metodología de desarrollo | `notebooks/02_preprocesamiento.ipynb`, `03_modelo.ipynb` | ⬜ Pendiente |
| 2.3 Resultados de iteraciones | Bitácora de experimentos | ⬜ Pendiente |
| 2.4 Discusión de resultados | Trabajo escrito | ⬜ Pendiente |
| 2.5 Conclusiones | Trabajo escrito | ⬜ Pendiente |
| 2.6 Enlace al repositorio | Este README | ⬜ Pendiente |

---

## Hallazgos del EDA

El análisis completo está en [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb). Lo esencial:

- **1168 viviendas × 79 features.** 35 numéricas, 21 ordinales y 23 nominales
  (`Neighborhood` con 25 niveles). Sin duplicados.
- **Trampa de parsing.** `pandas` interpreta los literales `"NA"` y `"None"` como nulos, pero en
  Ames son categorías válidas ("no tiene la característica"). La lectura ingenua destruye 677
  categorías legítimas de `MasVnrType`. Todo el proyecto lee con
  `keep_default_na=False, na_values=[""]`.
- **Target asimétrico.** `SalePrice` tiene skew 1.74; `log1p` lo lleva a 0.12. Entrenar en escala
  logarítmica es una hipótesis a validar, porque la métrica se mide en la escala original.
- **Nulos.** 19 columnas, en su mayoría `NA` estructurales verificados contra sus áreas
  correspondientes. El único faltante genuino relevante es `LotFrontage` (18.6 %).
- **Señal.** `OverallQual` (r = 0.786) y `GrLivArea` (r = 0.696) dominan; `Neighborhood` es la
  categórica más informativa (η² = 0.53). Varias relaciones son no lineales, lo que respalda el MLP.
- **Restricción dominante.** ~232 features tras codificar frente a 1168 observaciones (≈5:1):
  el riesgo principal es el **sobreajuste**, no la falta de capacidad.

Las 22 decisiones de preprocesamiento derivadas del EDA están tabuladas en la sección 11 del
notebook, cada una marcada como **firme** (corrige un defecto objetivo) o **hipótesis**
(requiere validación A/B en la sección 2.3).
