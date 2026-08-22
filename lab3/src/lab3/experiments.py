"""Definición de las 15 iteraciones (5 por arquitectura) y del experimento de la sección 4.1.

Criterio de diseño de la rejilla: **cambiar una cosa a la vez** respecto a una configuración
base por arquitectura. Una búsqueda aleatoria daría mejores números, pero no permitiría
responder la pregunta de la sección 6 ("¿qué hiperparámetro tuvo el mayor impacto?"), que
exige poder atribuir cada salto de métrica a un cambio concreto.
"""

from __future__ import annotations

from .engine import ExperimentConfig

# --------------------------------------------------------------------------------------
# MLP (baseline): promedio de embeddings. No usa el orden de las palabras.
# --------------------------------------------------------------------------------------
MLP_CONFIGS = [
    ExperimentConfig(
        name="mlp-01-base",
        kind="mlp",
        model_kwargs={"embed_dim": 64, "hidden_dims": (64,), "dropout": 0.0},
        lr=1e-3, epochs=10,
        note="base: embeddings pequeños, una capa oculta, sin regularización",
    ),
    ExperimentConfig(
        name="mlp-02-capacidad",
        kind="mlp",
        model_kwargs={"embed_dim": 128, "hidden_dims": (256, 128), "dropout": 0.0},
        lr=1e-3, epochs=10,
        note="+capacidad: embed 64->128 y cabeza (256,128); aísla el efecto del tamaño",
    ),
    ExperimentConfig(
        name="mlp-03-dropout",
        kind="mlp",
        model_kwargs={"embed_dim": 128, "hidden_dims": (256, 128), "dropout": 0.5},
        lr=1e-3, epochs=12,
        note="misma capacidad que 02 + dropout 0.5: ¿controla el sobreajuste?",
    ),
    ExperimentConfig(
        name="mlp-04-weight-decay",
        kind="mlp",
        model_kwargs={"embed_dim": 128, "hidden_dims": (256, 128), "dropout": 0.5},
        lr=1e-3, weight_decay=1e-4, epochs=12,
        note="03 + weight decay 1e-4: segunda vía de regularización sobre los embeddings",
    ),
    ExperimentConfig(
        name="mlp-05-lr-alto",
        kind="mlp",
        model_kwargs={"embed_dim": 128, "hidden_dims": (256, 128), "dropout": 0.5},
        lr=1e-2, epochs=10,
        note="03 con lr 1e-3 -> 1e-2: sonda del impacto negativo del learning rate",
    ),
]

# --------------------------------------------------------------------------------------
# RNN simple (nn.RNN, tanh, many-to-one sobre el último estado oculto)
# --------------------------------------------------------------------------------------
RNN_CONFIGS = [
    ExperimentConfig(
        name="rnn-01-base",
        kind="rnn",
        model_kwargs={"embed_dim": 128, "hidden_dim": 128, "dropout": 0.0},
        lr=1e-3, epochs=20, patience=4,
        note="base: 1 capa, sin dropout ni clipping",
    ),
    ExperimentConfig(
        name="rnn-02-clip",
        kind="rnn",
        model_kwargs={"embed_dim": 128, "hidden_dim": 128, "dropout": 0.0},
        lr=1e-3, clip_grad=1.0, epochs=20, patience=4,
        note="base + gradient clipping 1.0: el BPTT de una RNN simple explota con facilidad",
    ),
    ExperimentConfig(
        name="rnn-03-lr-bajo",
        kind="rnn",
        model_kwargs={"embed_dim": 128, "hidden_dim": 128, "dropout": 0.0},
        lr=3e-4, clip_grad=1.0, epochs=20, patience=4,
        note="02 con lr 1e-3 -> 3e-4: pasos más pequeños para un optimizador inestable",
    ),
    ExperimentConfig(
        name="rnn-04-hidden-256",
        kind="rnn",
        model_kwargs={"embed_dim": 128, "hidden_dim": 256, "dropout": 0.3},
        lr=3e-4, clip_grad=1.0, epochs=20, patience=4,
        note="03 + hidden 128->256 y dropout 0.3: más memoria de estado con regularización",
    ),
    ExperimentConfig(
        name="rnn-05-bidireccional",
        kind="rnn",
        model_kwargs={
            "embed_dim": 128, "hidden_dim": 128, "dropout": 0.3, "bidirectional": True
        },
        lr=3e-4, clip_grad=1.0, epochs=20, patience=4,
        note="03 + bidireccional: el final de la reseña deja de estar a 300 pasos del gradiente",
    ),
]

# --------------------------------------------------------------------------------------
# LSTM (nn.LSTM, misma configuración many-to-one que la RNN)
# --------------------------------------------------------------------------------------
LSTM_CONFIGS = [
    ExperimentConfig(
        name="lstm-01-base",
        kind="lstm",
        model_kwargs={"embed_dim": 128, "hidden_dim": 128, "dropout": 0.0},
        lr=1e-3, epochs=20, patience=4,
        note="base: espejo exacto de rnn-01 para aislar el efecto de las compuertas",
    ),
    ExperimentConfig(
        name="lstm-02-clip",
        kind="lstm",
        model_kwargs={"embed_dim": 128, "hidden_dim": 128, "dropout": 0.0},
        lr=1e-3, clip_grad=1.0, epochs=20, patience=4,
        note="base + gradient clipping 1.0 (mismo cambio que rnn-02, para comparar)",
    ),
    ExperimentConfig(
        name="lstm-03-dropout",
        kind="lstm",
        model_kwargs={"embed_dim": 128, "hidden_dim": 128, "dropout": 0.5},
        lr=1e-3, clip_grad=1.0, epochs=20, patience=4,
        note="02 + dropout 0.5 en embeddings y salida",
    ),
    ExperimentConfig(
        name="lstm-04-hidden-256",
        kind="lstm",
        model_kwargs={"embed_dim": 128, "hidden_dim": 256, "dropout": 0.5},
        lr=1e-3, clip_grad=1.0, epochs=20, patience=4,
        note="03 + hidden 128->256: cuadruplica los parámetros de las compuertas",
    ),
    ExperimentConfig(
        name="lstm-05-bidireccional",
        kind="lstm",
        model_kwargs={
            "embed_dim": 128, "hidden_dim": 256, "dropout": 0.5, "bidirectional": True
        },
        lr=1e-3, clip_grad=1.0, weight_decay=1e-5, epochs=20, patience=4,
        note="04 + bidireccional + weight decay 1e-5: configuración más grande de la rejilla",
    ),
]

ALL_CONFIGS = MLP_CONFIGS + RNN_CONFIGS + LSTM_CONFIGS

# Longitudes del experimento de la sección 4.1: contexto muy corto vs. reseña casi completa.
LENGTH_EXPERIMENT_LENS = [50, 200, 500]
