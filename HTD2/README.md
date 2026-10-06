# Hoja de trabajo 2 — Transformers y mecanismos de atención

**CC3092 · Deep Learning y Sistemas Inteligentes** — Esteban Cárcamo (23016). Entrega: lunes 5 de octubre de 2026.

Atención implementada desde cero y verificada contra PyTorch, efecto del escalamiento por √d_k, costo computacional frente a
la longitud y análisis de la atención de **BERT** y **GPT-2** (patrones por cabeza, correferencia de «it», *attention sinks*,
entropía por capa y ablación de cabezas).

## Estructura

```
HTD2/
├── notebooks/htd2_attention.ipynb   entregable, ejecutado
├── results/
│   ├── figures/                     gráficas del notebook y del informe
│   ├── results.json                 cifras usadas en el informe
│   └── bertviz_tired.html, bertviz_wide.html   vistas interactivas de bertviz (opcional)
└── informe/                         informe_htd2.html → informe_htd2.pdf (5 páginas)
```

## Reproducir

```bash
# venv compartido del curso, en la raíz del repo
../venv/bin/pip install -r requirements.txt
cd notebooks
../../venv/bin/python -m nbconvert --to notebook --execute --inplace htd2_attention.ipynb   # ~1 min en RTX 4050
```

El PDF se genera desde el HTML con Chromium:

```bash
cd informe
chromium --headless --no-pdf-header-footer --print-to-pdf=informe_htd2.pdf "file://$PWD/informe_htd2.html"
```

## Resultados principales

| Experimento | Resultado |
|---|---|
| Atención propia vs `F.scaled_dot_product_attention` / `nn.MultiheadAttention` | error máximo 4.8e-7 / 1.2e-7 |
| Var(q·k) con d_k = 1024 | 1 034 sin escalar, 1.01 escalada; softmax sin escalar: 0.95 en una llave |
| Atención ingenua vs SDPA (fp16, 8 cabezas) | ingenua O(n²) en memoria, sin memoria con n = 16 384; SDPA 33 MB con n = 32 768 |
| «it» → animal / street en BERT | 22 de 144 cabezas cambian; la mejor es L7-H11 (0.28 → animal con *tired*, 0.59 → street con *wide*) |
| GPT-2 | triangular inferior exacta; el primer token recibe 70–80 % de la atención en las capas 6–11 (*sink*) |
| Entropía normalizada | baja de la capa 1 a la 8 (BERT 0.77 → 0.42, GPT-2 0.69 → 0.34) |
| Ablación de cabezas en GPT-2 | 71 de 144 cambian la pérdida < 0.01; apagar el 25 % menos importante: perplejidad 23.9 → 30.9 |
