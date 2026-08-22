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
│   └── experiments.py   # definición de las 15 iteraciones + experimento de longitud
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
| Vocabulario | top 20 000 palabras por frecuencia (`min_freq=2`) | cubre >95 % de los tokens; recorta la cola larga que solo añade parámetros |
| OOV | token `<unk>` (id 1), `<pad>` (id 0) | las palabras raras se mapean a `<unk>`; `<pad>` se ignora vía `padding_idx` y `pack_padded_sequence` |
| Longitud máxima | 300 tokens (200 en el ablation corto, 500 en el largo) | la mediana de las reseñas es ~174 tokens; 300 cubre ~75 % completas sin disparar el costo |
| Splits | 20 000 train / 5 000 val (del split oficial de train) y 25 000 test | test se usa **una sola vez**, al final |

