"""Loop de entrenamiento, métricas y registro de iteraciones.

Todo experimento se describe con un `ExperimentConfig` y se ejecuta con `run_experiment`,
que devuelve un diccionario serializable a JSON con: hiperparámetros, curva de pérdida
train/val por epoch, métricas de validación por epoch, número de parámetros entrenables y
tiempo de entrenamiento. Eso es exactamente lo que pide la sección 4 del enunciado.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

from .data import EncodedCorpus, make_dataloaders
from .models import build_model, count_parameters


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    """Una iteración = una combinación de hiperparámetros sobre una arquitectura."""

    name: str
    kind: str                                   # "mlp" | "rnn" | "lstm"
    model_kwargs: dict = field(default_factory=dict)
    lr: float = 1e-3
    weight_decay: float = 0.0
    clip_grad: float | None = None              # None = sin gradient clipping
    batch_size: int = 64
    max_len: int = 300
    epochs: int = 8
    patience: int = 3                           # early stopping sobre val_loss
    seed: int = 42
    note: str = ""                              # qué se cambió respecto a la iteración previa


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy + precision/recall/F1 macro (las clases están balanceadas 50/50)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    probs, targets = [], []
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "loss": total_loss / n,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        **compute_metrics(y_true, y_pred),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, clip_grad) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x, lengths), y)
        loss.backward()
        if clip_grad is not None:
            # Recorta la norma global de los gradientes: es la defensa estándar contra el
            # exploding gradient del BPTT en secuencias largas (crítico en la RNN simple).
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


def run_experiment(
    config: ExperimentConfig,
    corpus: EncodedCorpus,
    device: torch.device | None = None,
    verbose: bool = True,
    return_model: bool = False,
) -> dict:
    device = device or get_device()
    set_seed(config.seed)

    loaders = make_dataloaders(
        corpus, max_len=config.max_len, batch_size=config.batch_size, seed=config.seed
    )
    model = build_model(config.kind, corpus.vocab_size, **config.model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history: list[dict] = []
    best_val_loss, best_state, best_epoch, epochs_no_improve = float("inf"), None, 0, 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, config.clip_grad
        )
        val = evaluate(model, loaders["val"], criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                **{k: val[k] for k in ("accuracy", "precision", "recall", "f1")},
            }
        )
        if verbose:
            print(
                f"  [{config.name}] epoch {epoch:>2} | train {train_loss:.4f} | "
                f"val {val['loss']:.4f} | acc {val['accuracy']:.4f} | f1 {val['f1']:.4f}"
            )

        # Selección de modelo por val_loss (no por accuracy: es más estable y detecta antes
        # el sobreajuste, que en estos modelos aparece como val_loss subiendo con accuracy
        # todavía plana).
        if val["loss"] < best_val_loss - 1e-4:
            best_val_loss, best_epoch, epochs_no_improve = val["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                if verbose:
                    print(f"  [{config.name}] early stopping en epoch {epoch}")
                break

    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    best = history[best_epoch - 1]
    result = {
        "name": config.name,
        "kind": config.kind,
        "config": asdict(config),
        "n_params": count_parameters(model),
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_time_s": train_time,
        "val_metrics": {k: best[k] for k in ("accuracy", "precision", "recall", "f1")},
        "val_loss": best["val_loss"],
    }
    if return_model:
        result["model"] = model
    return result


def save_results(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "model"} for r in results]
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))


def load_results(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
