"""Sección 4.1: mejores configuraciones de RNN y LSTM entrenadas con distintos max_len.

Se reutiliza la *misma* configuración ganadora de cada arquitectura y solo se cambia el
truncamiento, de modo que cualquier diferencia sea atribuible a la longitud de la secuencia.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab3.data import prepare_corpus                                       # noqa: E402
from lab3.engine import ExperimentConfig, get_device, run_experiment       # noqa: E402
from lab3.experiments import ALL_CONFIGS, LENGTH_EXPERIMENT_LENS           # noqa: E402


def best_config_per_kind(results_path: Path) -> dict[str, ExperimentConfig]:
    """Mejor iteración de cada arquitectura según val_loss (criterio de selección)."""
    results = json.loads(results_path.read_text())
    by_name = {c.name: c for c in ALL_CONFIGS}
    best: dict[str, dict] = {}
    for r in results:
        if r["kind"] not in best or r["val_loss"] < best[r["kind"]]["val_loss"]:
            best[r["kind"]] = r
    return {kind: by_name[r["name"]] for kind, r in best.items()}


def main() -> None:
    device = get_device()
    corpus = prepare_corpus()
    best = best_config_per_kind(ROOT / "results" / "iterations.json")

    results = []
    for kind in ("rnn", "lstm"):
        base = best[kind]
        print(f"\nMejor {kind.upper()}: {base.name}")
        for max_len in LENGTH_EXPERIMENT_LENS:
            config = replace(base, name=f"{base.name}@len{max_len}", max_len=max_len)
            print(f"  -> max_len = {max_len}")
            r = run_experiment(config, corpus, device=device)
            r["max_len"] = max_len
            results.append(r)

    (ROOT / "results" / "length_experiment.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    print("\nResumen sección 4.1:")
    for r in results:
        m = r["val_metrics"]
        print(f"  {r['kind']:<5} max_len {r['max_len']:>3} | val_loss {r['val_loss']:.4f} "
              f"| acc {m['accuracy']:.4f} | f1 {m['f1']:.4f} | {r['train_time_s']:.1f}s")


if __name__ == "__main__":
    main()
