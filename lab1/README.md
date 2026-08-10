# CC3092 — Laboratorio 1: Entrenamiento de Redes Neuronales

MLP de regresión en PyTorch sobre el dataset California Housing (`sklearn.datasets.fetch_california_housing`) para predecir `MedHouseVal`.

## Contenido

| Archivo | Descripción |
|---|---|
| `lab1.ipynb` | Notebook del laboratorio (datos, investigación, entrenamiento, resultados) |
| `requirements.txt` | Dependencias del entorno |

El análisis detallado y las conclusiones están en el reporte PDF del curso (no sustituido por este README).

## Requisitos

- Python 3.11+
- Dependencias en `requirements.txt` (PyTorch, pandas, numpy, matplotlib, scikit-learn, Jupyter)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook lab1.ipynb
```

Selecciona el kernel del `venv` al abrir el notebook.

## Qué hace el notebook

1. Carga y exploración del dataset; split train/val/test (60/20/20) y `StandardScaler`.
2. Documentación de capas, pérdidas y optimizadores de PyTorch para un MLP.
3. Entrenamiento de 12 configuraciones (arquitectura, activación, optimizador/lr, regularización, batch/epochs).
4. Curvas de pérdida, selección del mejor modelo por MSE de validación y evaluación única en test.
5. Tabla resumen de resultados.

## Resultado principal

Mejor configuración según MSE de validación: MLP `[128, 64]`, ReLU, Adam (`lr=0.001`), batch 32, 40 epochs.

Métricas en test (evaluación única): MSE ≈ 0.2809, MAE ≈ 0.3532, RMSE ≈ 0.5300.
