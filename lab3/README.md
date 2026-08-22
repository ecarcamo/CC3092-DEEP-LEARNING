# Laboratorio 3 — Redes Neuronales Recurrentes y LSTM

Clasificación binaria de sentimiento sobre **IMDB Reviews** (HuggingFace `stanfordnlp/imdb`)
comparando tres arquitecturas: **MLP** (baseline con embeddings promediados), **RNN simple**
(`nn.RNN`, many-to-one) y **LSTM** (`nn.LSTM`, many-to-one).

## Estructura

```
lab3/
├── src/lab3/            # código reutilizable (importado por el notebook)
│   ├── data.py          # tokenización, vocabulario, padding/truncamiento, DataLoaders
│   ├── models.py        # MLPBaseline, SimpleRNNClassifier, LSTMClassifier
│   ├── engine.py        # loop de entrenamiento, métricas, early stopping
│   ├── experiments.py   # definición de las 15 iteraciones + experimento de longitud
│   └── plots.py         # curvas de pérdida, matrices de confusión y figuras del informe
├── scripts/             # ejecución por lotes de los experimentos (guardan JSON en results/)
├── notebooks/           # notebook completo y comentado (entregable)
├── results/             # métricas por iteración (JSON) y figuras (PNG)
└── informe/             # informe PDF (máx. 4 páginas)
```

## Reproducir

```bash
# venv compartido del curso, en la raíz del repo
../venv/bin/pip install -r requirements.txt
../venv/bin/python scripts/run_experiments.py      # 15 iteraciones (5 por arquitectura)
../venv/bin/python scripts/run_length_experiment.py  # sección 4.1
../venv/bin/python scripts/run_final_test.py         # evaluación única sobre test
```

## Decisiones principales

| Decisión | Elección | Motivo |
|---|---|---|
| Tokenización | word-level con regex, minúsculas, limpieza de `<br />` | simple, transparente y suficiente para sentiment analysis; evita dependencias externas |
| Vocabulario | top 20 000 palabras por frecuencia (`min_freq=2`), construido solo con train | deja apenas 2.75 % de tokens OOV en train y 3.49 % en test; guardar los 79 124 tipos costaría ~7.6 M de parámetros extra |
| OOV | token `<unk>` (id 1), `<pad>` (id 0) | las palabras raras se mapean a `<unk>`; `<pad>` se ignora vía `padding_idx` y `pack_padded_sequence` |
| Longitud máxima | 300 tokens en la rejilla principal; 50 / 200 / 500 en el experimento 4.1 | la mediana es 176 tokens; 300 deja completas el 77 % de las reseñas y conserva el 81 % de los tokens |
| Splits | 20 000 train / 5 000 val (del split oficial de train) y 25 000 test | test se usa **una sola vez**, al final |

## Resultados

| Arquitectura | Mejor configuración | Params | Test accuracy | Test F1 | Tiempo de entrenamiento |
|---|---|---|---|---|---|
| MLP (baseline) | `mlp-05-lr-alto` | 2 626 049 | **0.8684** | 0.8684 | 3.7 s |
| RNN simple | `rnn-03-lr-bajo` | 2 593 153 | 0.8040 | 0.8039 | 55.1 s |
| LSTM | `lstm-04-hidden-256` | 2 955 521 | 0.8656 | 0.8655 | 54.9 s |

Experimento de longitud de secuencia (accuracy de validación): la brecha LSTM − RNN crece con la
longitud (+5.1 puntos con 50 tokens, +6.1 con 200, +6.6 con 500), que es la firma del vanishing
gradient. El análisis completo está en `informe/informe_lab3.pdf` y en el notebook.
