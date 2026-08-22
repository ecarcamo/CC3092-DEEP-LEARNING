"""Las tres arquitecturas del laboratorio: MLP (baseline), RNN simple y LSTM.

Las tres comparten el mismo front-end (`nn.Embedding` con `padding_idx=0`) y la misma
cabeza (una capa lineal a **un solo logit**, entrenada con `BCEWithLogitsLoss`). Lo único
que cambia es cómo se resume la secuencia en un vector de tamaño fijo:

| Modelo | Resumen de la secuencia |
|---|---|
| `MLPBaseline` | promedio de los embeddings de los tokens reales (bolsa de palabras "densa": pierde el orden) |
| `SimpleRNNClassifier` | último estado oculto `h_T` de la `nn.RNN` (many-to-one) |
| `LSTMClassifier` | último estado oculto `h_T` de la `nn.LSTM` (many-to-one) |
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import PAD_IDX


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPBaseline(nn.Module):
    """Baseline sin noción de orden: promedia embeddings y clasifica con un MLP.

    El promedio se hace con una máscara construida a partir de las longitudes reales; si
    dividiéramos entre la longitud del batch con padding incluido, las reseñas cortas
    quedarían sistemáticamente atenuadas.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128,),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                                  # (B, T, E)
        mask = (x != PAD_IDX).unsqueeze(-1).float()              # (B, T, 1)
        # El denominador sale de la propia máscara (y no de `lengths`) porque `lengths`
        # vive en CPU: las recurrentes lo necesitan así para pack_padded_sequence.
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)                     # (B,)


class _RecurrentClassifier(nn.Module):
    """Base común de RNN y LSTM: embedding -> recurrente empaquetada -> h_T -> lineal."""

    rnn: nn.RNN | nn.LSTM

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # Dropout sobre los embeddings: nn.RNN/nn.LSTM solo aplican su `dropout` interno
        # ENTRE capas apiladas, así que con num_layers=1 ese parámetro no hace nada.
        self.embed_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def _last_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        """`h_n` viene como (num_layers * num_directions, B, H); tomamos la última capa."""
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(x))
        # pack_padded_sequence evita que la recurrente procese los <pad>: sin esto, el
        # "último estado oculto" de una reseña corta sería el resultado de digerir decenas
        # de pasos de padding y la señal real quedaría diluida.
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden  # LSTM devuelve (h_n, c_n)
        return self.fc(self.out_dropout(self._last_hidden(h_n))).squeeze(-1)


class SimpleRNNClassifier(_RecurrentClassifier):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
    ):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )


class LSTMClassifier(_RecurrentClassifier):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )


MODEL_REGISTRY = {
    "mlp": MLPBaseline,
    "rnn": SimpleRNNClassifier,
    "lstm": LSTMClassifier,
}


def build_model(kind: str, vocab_size: int, **kwargs) -> nn.Module:
    return MODEL_REGISTRY[kind](vocab_size=vocab_size, **kwargs)
