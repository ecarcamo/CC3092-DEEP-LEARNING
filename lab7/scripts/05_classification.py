import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.classify import (
    LABELS,
    BagClassifier,
    build_vocab,
    confusion,
    encode,
    load_ag_news,
    metrics,
    oov_rate,
    predict,
    pretrained_matrix,
    subsample,
    tfidf_baseline,
    train_classifier,
)
from lab7.data import FIGURES_DIR, RESULTS_DIR
from lab7.models import load_three
from lab7.plots import classification_curves, confusion_figure, fraction_plot

FRACTIONS = [0.01, 0.05, 0.1, 0.25, 1.0]
DIM = 100


def make_model(kind, vocab_words, embeddings, freeze):
    if kind == "random":
        return BagClassifier(len(vocab_words), DIM)
    words, matrix = embeddings[kind]
    weights, _ = pretrained_matrix(vocab_words, words, matrix)
    return BagClassifier(len(vocab_words), weights.shape[1], weights=weights, freeze=freeze)


def variants():
    yield "random", "aleatorio", False
    for kind in ("SGNS", "gensim", "GloVe"):
        yield kind, f"{kind} congelado", True
        yield kind, f"{kind} fine-tuning", False


def main():
    device = torch.device("cuda")
    data = load_ag_news()
    embeddings = load_three()
    vocab_words, index = build_vocab(data["train"][0])
    all_texts = data["train"][0] + data["val"][0] + data["test"][0]
    oov = {name: oov_rate(all_texts, set(words)) for name, (words, _) in embeddings.items()}
    coverage = {name: pretrained_matrix(vocab_words, *embeddings[name])[1] for name in embeddings}
    print("oov", oov, "vocab coverage", coverage, flush=True)

    ids = {split: encode(data[split][0], index) for split in ("train", "val", "test")}
    train = (ids["train"], data["train"][1])
    val = (ids["val"], data["val"][1])
    test = (ids["test"], data["test"][1])

    runs = {}
    models = {}
    vectorizer, clf, tfidf = tfidf_baseline(data["train"], data["val"])
    runs["TF-IDF + LR"] = tfidf
    print("tfidf", tfidf["val"], flush=True)
    for kind, label, freeze in variants():
        model = make_model(kind, vocab_words, embeddings, freeze)
        runs[label] = train_classifier(model, train, val, device)
        runs[label]["init"] = kind
        runs[label]["freeze"] = freeze
        models[label] = model
        print(label, runs[label]["val"], runs[label]["params_trainable"], round(runs[label]["train_time_s"], 1), flush=True)

    best_per_init = {"TF-IDF": "TF-IDF + LR"}
    for kind in ("random", "SGNS", "gensim", "GloVe"):
        candidates = [k for k, r in runs.items() if r.get("init") == kind]
        best_per_init[kind] = max(candidates, key=lambda k: runs[k]["val"]["f1"])

    test_results, matrices, errors = {}, {}, {}
    for kind, label in best_per_init.items():
        if kind == "TF-IDF":
            preds = clf.predict(vectorizer.transform(data["test"][0]))
        else:
            preds, _ = predict(models[label], *test, device)
        test_results[label] = metrics(test[1], preds)
        matrices[label] = confusion(test[1], preds)
        wrong = np.where(preds != test[1])[0][:40]
        errors[label] = [{"text": data["raw_test"][i][:220], "true": LABELS[test[1][i]], "pred": LABELS[preds[i]]} for i in wrong]
        print("TEST", label, test_results[label], flush=True)

    confusion_figure(matrices, ["World", "Sports", "Business", "Sci/Tech"], FIGURES_DIR / "11_confusion.png")
    classification_curves({k: v for k, v in runs.items() if "history" in v}, FIGURES_DIR / "10_classification_curves.png")

    fraction_runs = {}
    for fraction in FRACTIONS:
        seeds = [0, 1, 2] if fraction <= 0.1 else [0]
        for seed in seeds:
            sub = subsample(data["train"][0], data["train"][1], fraction, seed)
            key = f"{fraction}"
            vec, f_clf, _ = tfidf_baseline(sub, data["val"], seed)
            preds = f_clf.predict(vec.transform(data["test"][0]))
            fraction_runs.setdefault("TF-IDF + LR", {}).setdefault(key, []).append(metrics(test[1], preds)["f1"])
            sub_ids = (encode(sub[0], index), sub[1])
            max_epochs = 10 if fraction >= 0.25 else 60
            patience = 2 if fraction >= 0.25 else 4
            for kind, label, freeze in variants():
                if kind == "gensim":
                    continue
                model = make_model(kind, vocab_words, embeddings, freeze)
                train_classifier(model, sub_ids, val, device, max_epochs=max_epochs, patience=patience, seed=seed)
                preds, _ = predict(model, *test, device)
                fraction_runs.setdefault(label, {}).setdefault(key, []).append(metrics(test[1], preds)["f1"])
            print("fraction", fraction, seed, {k: round(v[key][-1], 4) for k, v in fraction_runs.items()}, flush=True)

    fraction_plot({k: {float(f): v for f, v in pts.items()} for k, pts in fraction_runs.items()},
                  FIGURES_DIR / "13_f1_vs_fraction.png")

    out = {
        "split_sizes": {k: len(data[k][1]) for k in ("train", "val", "test")},
        "classifier_vocab": len(vocab_words),
        "oov_token_rate": oov,
        "classifier_vocab_coverage": coverage,
        "runs": runs,
        "best_per_init": best_per_init,
        "test": test_results,
        "confusion": matrices,
        "errors": errors,
        "fractions": fraction_runs,
    }
    (RESULTS_DIR / "classification.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
