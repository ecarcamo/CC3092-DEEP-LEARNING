import json

import numpy as np

from .data import CACHE_DIR, RESULTS_DIR

RUNS_DIR = CACHE_DIR / "runs"
BEST_PATH = RESULTS_DIR / "best_sgns.json"


def selection_score(final):
    return (final["analogy_total"] + final["wordsim_spearman"]) / 2


def load_sgns(name):
    vocab = json.loads((RUNS_DIR / f"{name}.vocab.json").read_text())
    return vocab["words"], np.load(RUNS_DIR / f"{name}.npy")


def best_sgns_name():
    return json.loads(BEST_PATH.read_text())["name"]


def load_gensim(name="G100"):
    from gensim.models import KeyedVectors

    kv = KeyedVectors.load(str(RUNS_DIR / f"{name}.kv"))
    return list(kv.index_to_key), kv.vectors


def load_glove():
    import gensim.downloader as api

    kv = api.load("glove-wiki-gigaword-100")
    return list(kv.index_to_key), kv.vectors


def load_three():
    name = best_sgns_name()
    return {"SGNS": load_sgns(name), "gensim": load_gensim(), "GloVe": load_glove()}


def shared_vocabulary(models, reference_words, size=30000):
    sets = [set(words) for words, _ in models.values()]
    shared = []
    for w in reference_words:
        if all(w in s for s in sets):
            shared.append(w)
            if len(shared) == size:
                break
    return shared


def restrict(words, matrix, subset):
    index = {w: i for i, w in enumerate(words)}
    return list(subset), matrix[[index[w] for w in subset]]
