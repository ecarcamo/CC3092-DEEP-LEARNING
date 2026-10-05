import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.data import FIGURES_DIR, RESULTS_DIR
from lab7.models import best_sgns_name
from lab7.plots import category_bars, corpus_size_plot, iteration_curves, loss_vs_quality

GLOVE_INFO = {
    "corpus_tokens": 6_000_000_000,
    "vocab_size": 400_000,
    "dim": 100,
    "train_time": "~85 min para construir X (1 hilo) + 50 iteraciones; 14 min/iteración con 300 d en 32 núcleos",
    "hardware": "Intel Xeon E5-2658 dual a 2.1 GHz, 32 núcleos (Pennington et al., 2014)",
}


def load(name):
    return json.loads((RESULTS_DIR / name).read_text())


def main():
    sgns = load("sgns_iterations.json")
    gensim = load("gensim_runs.json")
    arithmetic = load("vector_arithmetic.json")
    classification = load("classification.json")
    best = best_sgns_name()

    iteration_curves(sgns, ["I01", "I02", "I03"], FIGURES_DIR / "02_curves_dim.png")
    iteration_curves(sgns, ["I01", "I04", "I05", "I06"], FIGURES_DIR / "03_curves_window_negatives.png")
    iteration_curves(sgns, ["I10", "I11", "I01"], FIGURES_DIR / "04_curves_corpus.png")
    loss_vs_quality(sgns, FIGURES_DIR / "05_loss_vs_quality.png")
    category_bars(arithmetic["models"], FIGURES_DIR / "08_analogy_categories.png")

    sgns_points = [(sgns[k]["corpus_tokens"], sgns[k]["final"]["analogy_total"]) for k in ("I10", "I11", "I01")]
    gensim_points = [(g["corpus_tokens"], g["final"]["analogy_total"]) for g in gensim.values()]
    glove_acc = arithmetic["models"]["GloVe"]["analogy_add"]["total"]["accuracy"]
    corpus_size_plot(sgns_points, gensim_points, glove_acc, FIGURES_DIR / "12_analogies_vs_corpus.png")

    best_label = classification["best_per_init"]
    table = {}
    for name in ("SGNS", "gensim", "GloVe"):
        m = arithmetic["models"][name]
        if name == "SGNS":
            r = sgns[best]
            info = {"corpus_tokens": r["corpus_tokens"], "vocab_size": r["vocab_size"], "dim": r["config"]["dim"],
                    "train_time_s": r["train_time_s"], "hardware": r["gpu"], "peak_gpu_mb": r["peak_gpu_mb"]}
        elif name == "gensim":
            r = gensim["G100"]
            info = {"corpus_tokens": r["corpus_tokens"], "vocab_size": r["vocab_size"], "dim": r["params"]["dim"],
                    "train_time_s": r["train_time_s"], "hardware": r["hardware"]}
        else:
            info = dict(GLOVE_INFO)
        table[name] = {
            **info,
            **{f"add_{k}": m["analogy_add"][k]["accuracy"] for k in ("semantic", "syntactic", "total")},
            **{f"mul_{k}": m["analogy_mul"][k]["accuracy"] for k in ("semantic", "syntactic", "total")},
            "coverage": m["analogy_add"]["total"]["coverage"],
            "wordsim": m["wordsim"]["spearman"],
            "simlex": m["simlex"]["spearman"],
            "parallelism": m["parallelism_mean"],
            "oov_ag_news": classification["oov_token_rate"][name],
            "test_f1": classification["test"][best_label[name]]["f1"],
            "test_variant": best_label[name],
        }
    (RESULTS_DIR / "comparison.json").write_text(json.dumps({"best_sgns": best, "table": table}, indent=1, ensure_ascii=False))
    for k, v in table.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
