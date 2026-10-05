# Laboratorio 7 — NLP end-to-end y Embeddings

**CC3092 · Deep Learning y Sistemas Inteligentes** — Esteban Cárcamo (23016). Entrega: lunes 5 de octubre de 2026.

Pipeline completo texto → tokens → IDs → vectores → modelo → salida: preprocesamiento de **WikiText-103**, un
**skip-gram con negative sampling (SGNS) escrito desde cero en PyTorch**, comparación con **Word2Vec de gensim** y
**GloVe 100d**, verificación de la aritmética vectorial y clasificación de **AG News** con esos embeddings.

## Estructura

```
lab7/
├── src/lab7/
│   ├── data.py          normalización, tokenizador, vocabulario, subsampling y conteo de pares
│   ├── corpus.py        subconjunto de 25 M tokens y fracciones anidadas (25 %, 50 %)
│   ├── sgns.py          modelo SGNS (dos nn.Embedding), pares en GPU por epoch y entrenamiento
│   ├── evaluation.py    analogías 3CosAdd/3CosMul en GPU, analogia(), WordSim/SimLex, paralelismo
│   ├── models.py        carga de SGNS, gensim y GloVe; vocabulario compartido
│   ├── classify.py      AG News: TF-IDF + LR y EmbeddingBag(mean) + MLP
│   └── plots.py         figuras
├── scripts/             01 exploración · 02 SGNS · 03 gensim · 04 aritmética · 05 clasificación · 06 tablas
├── notebooks/lab7_nlp_embeddings.ipynb   entregable, ejecutado
├── results/             JSON de cada experimento, figures/ y vectors/best_sgns.kv (mejor modelo)
└── informe/             informe_lab7.html → informe_lab7.pdf (5 páginas)
```

## Reproducir

```bash
# venv compartido del curso, en la raíz del repo
../venv/bin/pip install -r requirements.txt
../venv/bin/python scripts/01_explore_corpus.py          # sección 2 (~1 min)
../venv/bin/python scripts/02_train_sgns.py I01 I02 I03 I04 I05 I06 I07 I08 I09 I10 I11   # ~35 min en RTX 4050
../venv/bin/python scripts/02_train_sgns.py I12 --config '{"window":10,"negatives":15,"lr":0.005,"epochs":8,"description":"ventana 10 + 15 neg + lr 5e-3, 8 epochs"}' --select
../venv/bin/python scripts/03_train_gensim.py --params '{"dim":100,"window":5,"negatives":5,"sample":1e-4,"min_count":5,"epochs":5}' --runs GB25 GB50 GB100 GFULL
../venv/bin/python scripts/03_train_gensim.py --params '{"dim":100,"window":10,"negatives":15,"sample":1e-4,"min_count":5,"epochs":8}' --runs G100
../venv/bin/python scripts/04_vector_arithmetic.py
../venv/bin/python scripts/05_classification.py           # ~15 min
../venv/bin/python scripts/06_tables_figures.py
```

Cargar los vectores del mejor modelo:

```python
from gensim.models import KeyedVectors
kv = KeyedVectors.load("results/vectors/best_sgns.kv")
kv.most_similar(positive=["king", "woman"], negative=["man"])
```

## Decisiones principales

| Decisión | Elección | Motivo |
|---|---|---|
| Corpus | primeros 8 802 artículos de WikiText-103 train = 25,0 M tokens | el mínimo pedido es 20 M; las fracciones de 25 % y 50 % son anidadas |
| Normalización | minúsculas, sin puntuación, `@,@`/`@.@` unen números, contracciones estilo Treebank | coincidir con el vocabulario de GloVe 6B |
| Vocabulario | `min_count` = 5 → 90 566 palabras, 1,18 % de tokens desconocidos | 41 805 con `min_count` 20 a cambio de 3 % de desconocidos |
| Subsampling | fórmula de word2vec con t = 10⁻⁴ | descarta el 96 % de «the»; 240 M → 141 M pares por epoch |
| Selección | analogías + WordSim-353; SimLex-999 solo al final | lo pide el enunciado |
| Mejor modelo | mejor SGNS de d = 100 (I12) | comparación directa con GloVe-100 y mismo clasificador |

## Resultados

| | SGNS propio (I12) | Word2Vec gensim | GloVe 6B |
|---|---|---|---|
| Tokens · vocabulario · dimensión | 24,7 M · 90 566 · 100 | 25,0 M · 90 566 · 100 | 6 000 M · 400 000 · 100 |
| Analogías 3CosAdd (sem / sint / total) | 47,2 / 46,4 / **46,7 %** | 45,3 / 43,5 / 44,0 % | 61,2 / 68,3 / **66,2 %** |
| WordSim-353 · SimLex-999 (ρ) | **0,665** · 0,279 | 0,660 · 0,290 | 0,533 · 0,298 |
| Paralelismo (coseno medio) | 0,204 | 0,198 | 0,406 |
| OOV en AG News · F1 de test | 3,0 % · 91,72 | 3,0 % · 91,62 | 0,7 % · 91,73 |

- La distancia con GloVe se explica sobre todo por el corpus: gensim pasa de 17,4 % (6 M tokens) a 51,7 % (86 M).
- La aritmética es aproximada: sin excluir la consulta, el vecino más cercano de b − a + c es c en 14 de 18 analogías.
- AG News: TF-IDF + LR gana con todos los datos (92,1 %); con el 1 % los embeddings preentrenados ganan por 5 puntos a
  TF-IDF y por 14 a los aleatorios.

El análisis completo está en `informe/informe_lab7.pdf` y en el notebook.
