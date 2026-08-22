"""Ejecuta las 15 iteraciones de la sección 4 y guarda los resultados en results/iterations.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab3.data import prepare_corpus                                    # noqa: E402
from lab3.engine import get_device, run_experiment, save_results        # noqa: E402
from lab3.experiments import ALL_CONFIGS                                # noqa: E402


def main() -> None:
    device = get_device()
    print(f"Dispositivo: {device}")
    corpus = prepare_corpus()
    print(f"Vocabulario: {corpus.vocab_size:,} | train {len(corpus.train_ids):,} "
          f"| val {len(corpus.val_ids):,} | test {len(corpus.test_ids):,}")

    results = []
    for i, config in enumerate(ALL_CONFIGS, 1):
        print(f"\n[{i}/{len(ALL_CONFIGS)}] {config.name} — {config.note}")
        results.append(run_experiment(config, corpus, device=device))
        save_results(results, ROOT / "results" / "iterations.json")  # guardado incremental

    print("\nResumen (mejor epoch de cada iteración):")
    for r in results:
        m = r["val_metrics"]
        print(f"  {r['name']:<22} params {r['n_params']:>9,} | val_loss {r['val_loss']:.4f} "
              f"| acc {m['accuracy']:.4f} | f1 {m['f1']:.4f} | {r['train_time_s']:.1f}s")


if __name__ == "__main__":
    main()
