# CC3092 — Laboratorio 2: Redes Neuronales Convolucionales

Clasificación de dígitos manuscritos (MNIST, 10 clases) en PyTorch, comparando un **MLP** que recibe la
imagen aplanada contra una **CNN** que la recibe como tensor 2D.

## Contenido

| Archivo | Descripción |
|---|---|
| `lab2.ipynb` | Notebook del laboratorio (datos, investigación de capas, 14 iteraciones, evaluación final) |
| `requirements.txt` | Dependencias del entorno |
| `Laboratorio_2_CNN.md` | Enunciado del laboratorio |

## Requisitos

- Python 3.11+
- Dependencias en `requirements.txt` (PyTorch + torchvision con CUDA, pandas, numpy, matplotlib, scikit-learn, Jupyter)
- GPU opcional: el notebook detecta CUDA y cae a CPU si no hay.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook lab2.ipynb
```

Selecciona el kernel del `venv` al abrir el notebook. MNIST se descarga automáticamente a `data/`
(ignorado por git).

## Qué hace el notebook

1. Exploración de MNIST: conteo de clases y balance, dimensiones y rango de píxeles, ejemplos
   etiquetados; split 48 000 / 12 000 / 10 000 (train / validación / test) y normalización con los
   estadísticos del train.
2. Investigación de las capas de `torch.nn` para CNN (`Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d`,
   `Flatten`, `CrossEntropyLoss`), más los conceptos de tensor, campo receptivo y coste en parámetros
   frente a un MLP.
3. Modelos configurables (`MLP` y `CNN`) y búsqueda sistemática de hiperparámetros: **14 iteraciones**
   (7 por arquitectura) cambiando uno o dos factores a la vez, registrando pérdida por epoch, métricas de
   validación, parámetros entrenables y tiempo de entrenamiento.
4. Curvas de pérdida train/val de las 14 iteraciones, selección de la mejor configuración por F1 macro de
   validación y evaluación única en test con matrices de confusión.
5. Comparación de arquitecturas (parámetros vs. accuracy, tiempos de entrenamiento e inferencia) y
   discusión de resultados.

## Resultados principales

Mejores configuraciones según F1 macro de validación:

- **MLP:** `[256, 128]` + dropout 0.3, Adam `lr=1e-3`, batch 128, 10 epochs — 235 146 parámetros.
- **CNN:** `[32, 64]` conv 3x3 + BatchNorm2d + average pooling, Adam `lr=1e-3`, batch 128, 10 epochs —
  421 834 parámetros.

Evaluación única sobre las 10 000 imágenes de test:

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Errores |
|---|---|---|---|---|---|
| MLP | 0.9788 | 0.9788 | 0.9786 | 0.9787 | 212 |
| CNN | **0.9932** | **0.9932** | **0.9931** | **0.9932** | **68** |

La CNN reduce la tasa de error de 2.12% a 0.68% (−67.9%). La ventaja no viene del número de parámetros:
la CNN más pequeña probada (206 922 parámetros, menos que el MLP final) también supera a todos los MLP,
y solo el 4.5% de los parámetros de la CNN está en las capas convolucionales.

El análisis detallado y las conclusiones están en el reporte PDF del curso.
