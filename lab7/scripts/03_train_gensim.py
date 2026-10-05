import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.corpus import FULL_PATH, write_subset_file
from lab7.data import CACHE_DIR, RESULTS_DIR
from lab7.evaluation import load_analogies
from lab7.sgns import evaluate_model

RESULTS_PATH = RESULTS_DIR / "gensim_runs.json"
RUNS_DIR = CACHE_DIR / "runs"
WORKERS = 16


class EpochLogger(CallbackAny2Vec):
    def __init__(self, name, sections):
        self.name = name
        self.sections = sections
        self.history = []
        self.previous_loss = 0.0
        self.epoch_start = None
        self.epoch = 0

    def on_epoch_begin(self, model):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, model):
        elapsed = time.perf_counter() - self.epoch_start
        self.epoch += 1
        cumulative = model.get_latest_training_loss()
        loss = cumulative - self.previous_loss
        self.previous_loss = cumulative
        kv = model.wv
        metrics = evaluate_model(kv.index_to_key, kv.vectors, self.sections)
        record = {"epoch": self.epoch, "loss": float(loss), "epoch_time_s": elapsed, **metrics}
        self.history.append(record)
        print(f"[{self.name}] epoch {self.epoch} loss={loss:.0f} analogy={record['analogy_total']:.3f} "
              f"(sem {record['analogy_semantic']:.3f} / syn {record['analogy_syntactic']:.3f}) "
              f"ws={record['wordsim_spearman']:.3f} t={elapsed:.1f}s", flush=True)


def train(name, corpus_file, params, sections):
    logger = EpochLogger(name, sections)
    start = time.perf_counter()
    model = Word2Vec(
        corpus_file=str(corpus_file),
        sg=1,
        hs=0,
        vector_size=params["dim"],
        window=params["window"],
        negative=params["negatives"],
        sample=params["sample"],
        min_count=params["min_count"],
        epochs=params["epochs"],
        alpha=0.025,
        min_alpha=0.0001,
        workers=WORKERS,
        seed=42,
        compute_loss=True,
        callbacks=[logger],
    )
    total = time.perf_counter() - start
    result = {
        "params": params,
        "corpus_tokens": int(model.corpus_total_words),
        "vocab_size": len(model.wv),
        "history": logger.history,
        "train_time_s": float(sum(h["epoch_time_s"] for h in logger.history)),
        "total_time_s": total,
        "hardware": f"CPU, {WORKERS} hilos",
        "final": {k: v for k, v in logger.history[-1].items() if k != "neighbors"},
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(RUNS_DIR / f"{name}.kv"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="JSON con dim, window, negatives, sample, min_count, epochs")
    parser.add_argument("--runs", nargs="*", default=["G100"])
    args = parser.parse_args()
    params = json.loads(args.params)
    sections = load_analogies()
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    fractions = {"G100": 1.0, "GB100": 1.0, "GB25": 0.25, "GB50": 0.5}
    for name in args.runs:
        if name in results:
            print("skip", name)
            continue
        if name == "GFULL":
            corpus_file = FULL_PATH
        else:
            corpus_file = CACHE_DIR / f"wiki_{int(fractions[name] * 100)}.txt"
            if not corpus_file.exists():
                write_subset_file(fractions[name], corpus_file)
        results[name] = train(name, corpus_file, params, sections)
        results[name]["corpus_fraction"] = fractions.get(name)
        RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
