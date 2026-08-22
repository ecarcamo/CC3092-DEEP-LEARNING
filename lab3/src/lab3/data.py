"""Carga, tokenización y preparación de IMDB Reviews.

Decisiones de diseño (se justifican en el notebook y el informe):

* **Tokenización word-level con regex.** No usamos un tokenizador subword (BPE/WordPiece)
  porque el objetivo del laboratorio es comparar arquitecturas recurrentes, no tokenizadores;
  un vocabulario de palabras mantiene la capa `nn.Embedding` interpretable y el pipeline
  autocontenido (sin dependencias externas). Conservamos `!` y `?` como tokens propios porque
  cargan señal de sentimiento.
* **`<pad>` = 0 y `<unk>` = 1.** `<pad>` se pasa como `padding_idx` a `nn.Embedding` (su
  embedding queda fijo en cero y no recibe gradiente) y además se descuenta con
  `pack_padded_sequence` en las recurrentes. Todo token fuera de vocabulario cae en `<unk>`.
* **Truncamiento por la cabeza.** Nos quedamos con los primeros `max_len` tokens. Es la
  convención habitual y permite comparar de forma limpia el experimento de la sección 4.1
  (mismo prefijo, distinta cantidad de contexto).
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
_BR_RE = re.compile(r"<br\s*/?>")
_TOKEN_RE = re.compile(r"[a-z0-9']+|[!?]")


# --------------------------------------------------------------------------------------
# Tokenización
# --------------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Minúsculas, elimina los saltos de línea HTML de IMDB y extrae palabras / '!' / '?'."""
    text = _BR_RE.sub(" ", text.lower())
    return _TOKEN_RE.findall(text)


def build_vocab(
    token_lists: list[list[str]], max_size: int = 20_000, min_freq: int = 2
) -> dict[str, int]:
    """Vocabulario por frecuencia descendente, construido **solo con el split de train**.

    Construirlo con validación o test filtraría información del conjunto de evaluación
    hacia el modelo (data leakage).
    """
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab.get(tok, UNK_IDX) for tok in tokens]


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
@dataclass
class RawSplits:
    """Textos crudos + etiquetas de cada partición."""

    train_texts: list[str]
    train_labels: list[int]
    val_texts: list[str]
    val_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


def load_raw_splits(val_size: int = 5_000, seed: int = 42) -> RawSplits:
    """IMDB oficial: 25k train + 25k test.

    El split oficial viene ordenado por clase (primero todos los negativos), así que
    barajamos con semilla fija antes de recortar validación, y estratificamos para que
    train y val queden balanceados 50/50 igual que el dataset original.
    """
    from datasets import load_dataset

    imdb = load_dataset("stanfordnlp/imdb")  # el alias corto "imdb" ya no resuelve en HF
    train_texts = list(imdb["train"]["text"])
    train_labels = np.asarray(imdb["train"]["label"])

    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(train_labels == cls)
        rng.shuffle(cls_idx)
        val_idx.extend(cls_idx[: val_size // 2].tolist())
    val_mask = np.zeros(len(train_labels), dtype=bool)
    val_mask[val_idx] = True

    return RawSplits(
        train_texts=[t for t, m in zip(train_texts, val_mask) if not m],
        train_labels=train_labels[~val_mask].tolist(),
        val_texts=[t for t, m in zip(train_texts, val_mask) if m],
        val_labels=train_labels[val_mask].tolist(),
        test_texts=list(imdb["test"]["text"]),
        test_labels=list(imdb["test"]["label"]),
    )


@dataclass
class EncodedCorpus:
    """Secuencias ya tokenizadas y mapeadas a ids (sin truncar todavía)."""

    train_ids: list[list[int]]
    train_labels: list[int]
    val_ids: list[list[int]]
    val_labels: list[int]
    test_ids: list[list[int]]
    test_labels: list[int]
    vocab: dict[str, int]
    train_tokens: list[list[str]]  # se conservan para la exploración del notebook

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def prepare_corpus(
    max_vocab: int = 20_000, min_freq: int = 2, seed: int = 42, use_cache: bool = True
) -> EncodedCorpus:
    """Tokeniza el corpus completo una sola vez y lo cachea en disco.

    El truncamiento a `max_len` NO se hace aquí sino en el `Dataset`, para poder reutilizar
    el mismo corpus codificado en el experimento de longitudes de la sección 4.1.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"corpus_v{max_vocab}_f{min_freq}_s{seed}.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    raw = load_raw_splits(seed=seed)
    train_tokens = [tokenize(t) for t in raw.train_texts]
    val_tokens = [tokenize(t) for t in raw.val_texts]
    test_tokens = [tokenize(t) for t in raw.test_texts]

    vocab = build_vocab(train_tokens, max_size=max_vocab, min_freq=min_freq)
    corpus = EncodedCorpus(
        train_ids=[encode(t, vocab) for t in train_tokens],
        train_labels=raw.train_labels,
        val_ids=[encode(t, vocab) for t in val_tokens],
        val_labels=raw.val_labels,
        test_ids=[encode(t, vocab) for t in test_tokens],
        test_labels=raw.test_labels,
        vocab=vocab,
        train_tokens=train_tokens,
    )
    with cache_path.open("wb") as fh:
        pickle.dump(corpus, fh)
    return corpus


# --------------------------------------------------------------------------------------
# Dataset / DataLoader
# --------------------------------------------------------------------------------------
class IMDBSequenceDataset(Dataset):
    """Devuelve `(ids truncados, longitud real, etiqueta)`."""

    def __init__(self, sequences: list[list[int]], labels: list[int], max_len: int):
        self.sequences = [s[:max_len] if s else [UNK_IDX] for s in sequences]
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), len(seq), self.labels[idx]


def collate_batch(batch):
    """Padding dinámico: cada batch se rellena a la longitud de su reseña más larga.

    Por eso hace falta padding: las reseñas tienen longitud variable y un tensor
    `(batch, tiempo)` exige un rectángulo. Guardamos las longitudes reales para que las
    recurrentes puedan usar `pack_padded_sequence` y el MLP promedie solo tokens reales.
    """
    seqs, lengths, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return (
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def make_dataloaders(
    corpus: EncodedCorpus,
    max_len: int = 300,
    batch_size: int = 64,
    include_test: bool = False,
    seed: int = 42,
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            IMDBSequenceDataset(corpus.train_ids, corpus.train_labels, max_len),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            generator=generator,
            drop_last=False,
        ),
        "val": DataLoader(
            IMDBSequenceDataset(corpus.val_ids, corpus.val_labels, max_len),
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collate_batch,
        ),
    }
    if include_test:
        loaders["test"] = DataLoader(
            IMDBSequenceDataset(corpus.test_ids, corpus.test_labels, max_len),
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collate_batch,
        )
    return loaders
