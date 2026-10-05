import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.corpus import load_subset
from lab7.data import CACHE_DIR, RESULTS_DIR, build_vocab, count_words, encode
from lab7.evaluation import load_analogies, to_keyed_vectors
from lab7.models import BEST_PATH, load_sgns, selection_score
from lab7.sgns import SGNSConfig, train_sgns

BASE = SGNSConfig()
ITERATIONS = {
    "I01": replace(BASE, name="I01", description="base"),
    "I02": replace(BASE, name="I02", dim=50, description="dim 50"),
    "I03": replace(BASE, name="I03", dim=300, description="dim 300"),
    "I04": replace(BASE, name="I04", window=2, description="ventana 2"),
    "I05": replace(BASE, name="I05", window=10, description="ventana 10"),
    "I06": replace(BASE, name="I06", negatives=15, description="15 negativos"),
    "I07": replace(BASE, name="I07", sample=1e-3, description="subsampling 1e-3"),
    "I08": replace(BASE, name="I08", min_count=20, description="min_count 20"),
    "I09": replace(BASE, name="I09", lr=5e-3, description="lr 5e-3"),
    "I10": replace(BASE, name="I10", corpus_fraction=0.25, description="corpus 25 %"),
    "I11": replace(BASE, name="I11", corpus_fraction=0.5, description="corpus 50 %"),
}
COMPARISON_DIM = 100
RESULTS_PATH = RESULTS_DIR / "sgns_iterations.json"
RUNS_DIR = CACHE_DIR / "runs"


def load_results():
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}


def run(config, sections, force=False):
    results = load_results()
    if config.name in results and not force:
        print(f"skip {config.name}")
        return
    lines = load_subset(config.corpus_fraction)
    vocab = build_vocab(count_words(lines), config.min_count)
    ids, line_ids = encode(lines, vocab)
    matrix, result = train_sgns(config, vocab, ids, line_ids, sections)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RUNS_DIR / f"{config.name}.npy", matrix.astype(np.float32))
    (RUNS_DIR / f"{config.name}.vocab.json").write_text(json.dumps({"words": vocab.words, "counts": vocab.counts.tolist()}))
    results = load_results()
    results[config.name] = result
    RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False))


def select_best():
    results = load_results()
    scores = {k: selection_score(v["final"]) for k, v in results.items()}
    candidates = [k for k, v in results.items() if v["config"]["dim"] == COMPARISON_DIM]
    best = max(candidates, key=scores.get)
    BEST_PATH.write_text(json.dumps({"name": best, "scores": scores}, indent=1))
    words, matrix = load_sgns(best)
    vectors_dir = RESULTS_DIR / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    to_keyed_vectors(words, matrix).save(str(vectors_dir / "best_sgns.kv"))
    print("best", best, round(scores[best], 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=[])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=str, help="JSON con overrides para una iteración nueva")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    sections = load_analogies()
    for name in args.names:
        config = ITERATIONS.get(name, replace(BASE, name=name))
        if args.config:
            config = replace(config, **json.loads(args.config))
        if args.epochs:
            config = replace(config, epochs=args.epochs)
        run(config, sections, force=args.force)
    if args.select:
        select_best()


if __name__ == "__main__":
    main()
