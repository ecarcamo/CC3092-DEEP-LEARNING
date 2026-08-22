"""Evaluación final: reentrena la mejor configuración de cada arquitectura y la evalúa
**una única vez** sobre el conjunto de test (25 000 reseñas nunca vistas).

Guarda:
* `results/test_results.json` — métricas y matriz de confusión por modelo.
* `results/test_predictions.npz` — etiqueta real, predicción, probabilidad y longitud en
  tokens de cada reseña de test, para el análisis de errores del notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab3.data import make_dataloaders, prepare_corpus                     # noqa: E402
from lab3.engine import confusion, evaluate, get_device, run_experiment    # noqa: E402
from run_length_experiment import best_config_per_kind                     # noqa: E402


def main() -> None:
    device = get_device()
    corpus = prepare_corpus()
    best = best_config_per_kind(ROOT / "results" / "iterations.json")

    summary, preds = [], {}
    for kind in ("mlp", "rnn", "lstm"):
        config = best[kind]
        print(f"\n=== {kind.upper()} — mejor configuración: {config.name} ===")
        # Se reentrena con la misma semilla: reproduce la iteración ganadora y devuelve el
        # modelo con los pesos del mejor epoch (los que seleccionó el early stopping).
        run = run_experiment(config, corpus, device=device, return_model=True)
        model: nn.Module = run["model"]

        test_loader = make_dataloaders(
            corpus, max_len=config.max_len, batch_size=config.batch_size, include_test=True
        )["test"]
        test = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
        cm = confusion(test["y_true"], test["y_pred"])

        summary.append({
            "kind": kind,
            "name": config.name,
            "n_params": run["n_params"],
            "train_time_s": run["train_time_s"],
            "best_epoch": run["best_epoch"],
            "max_len": config.max_len,
            "val_metrics": run["val_metrics"],
            "test_loss": test["loss"],
            "test_metrics": {k: test[k] for k in ("accuracy", "precision", "recall", "f1")},
            "confusion_matrix": cm.tolist(),
        })
        preds[f"{kind}_pred"] = test["y_pred"]
        preds[f"{kind}_prob"] = test["y_prob"]
        preds["y_true"] = test["y_true"]

        m = summary[-1]["test_metrics"]
        print(f"  TEST acc {m['accuracy']:.4f} | precision {m['precision']:.4f} "
              f"| recall {m['recall']:.4f} | f1 {m['f1']:.4f}")
        print(f"  matriz de confusión (filas = real):\n{cm}")
        torch.save(model.state_dict(), ROOT / "results" / f"best_{kind}.pt")

    preds["test_lengths"] = np.array([len(s) for s in corpus.test_ids])
    np.savez_compressed(ROOT / "results" / "test_predictions.npz", **preds)
    (ROOT / "results" / "test_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
