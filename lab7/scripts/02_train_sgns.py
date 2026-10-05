import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.corpus import load_subset
from lab7.data import CACHE_DIR, RESULTS_DIR, build_vocab, count_words, encode
from lab7.evaluation import load_analogies
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(ITERATIONS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=str, help="JSON con overrides para una iteración nueva")
    args = parser.parse_args()
    sections = load_analogies()
    for name in args.names:
        config = ITERATIONS.get(name, replace(BASE, name=name))
        if args.config:
            config = replace(config, **json.loads(args.config))
        if args.epochs:
            config = replace(config, epochs=args.epochs)
        run(config, sections, force=args.force)


if __name__ == "__main__":
    main()
