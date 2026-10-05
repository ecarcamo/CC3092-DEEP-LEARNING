from itertools import combinations

import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

ANALOGIES_PATH = datapath("questions-words.txt")
WORDSIM_PATH = datapath("wordsim353.tsv")
SIMLEX_PATH = datapath("simlex999.txt")
SEMANTIC_PREFIX = "gram"
CONTROL_WORDS = ["king", "france", "computer", "good", "january", "run"]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_analogies(path=ANALOGIES_PATH):
    sections = {}
    current = None
    with open(path) as f:
        for line in f:
            if line.startswith(":"):
                current = line[1:].strip()
                sections[current] = []
            else:
                sections[current].append(line.lower().split())
    return sections


def is_semantic(category):
    return not category.startswith(SEMANTIC_PREFIX)


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _score(target_abc, unit, method):
    a, b, c = target_abc
    if method == "add":
        query = unit[b] - unit[a] + unit[c]
        query = query / query.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return query @ unit.T
    sim_a = (unit[a] @ unit.T + 1) / 2
    sim_b = (unit[b] @ unit.T + 1) / 2
    sim_c = (unit[c] @ unit.T + 1) / 2
    return sim_b * sim_c / (sim_a + 1e-3)


def analogy_accuracy(words, matrix, sections=None, method="add", batch_size=2048):
    sections = sections or load_analogies()
    dev = device()
    unit = torch.tensor(normalize_rows(matrix), device=dev)
    index = {w: i for i, w in enumerate(words)}
    per_category = {}
    for category, questions in sections.items():
        valid = [q for q in questions if all(w in index for w in q)]
        correct = 0
        if valid:
            ids = torch.tensor([[index[w] for w in q] for q in valid], device=dev)
            for start in range(0, len(ids), batch_size):
                chunk = ids[start:start + batch_size]
                a, b, c, d = chunk.T
                scores = _score((a, b, c), unit, method)
                rows = torch.arange(len(chunk), device=dev)
                for col in (a, b, c):
                    scores[rows, col] = -float("inf")
                correct += int((scores.argmax(dim=1) == d).sum())
        per_category[category] = {"correct": correct, "evaluated": len(valid), "total": len(questions)}
    return summarize_analogies(per_category)


def summarize_analogies(per_category):
    def agg(filter_fn):
        cats = [v for k, v in per_category.items() if filter_fn(k)]
        correct = sum(c["correct"] for c in cats)
        evaluated = sum(c["evaluated"] for c in cats)
        total = sum(c["total"] for c in cats)
        return {
            "accuracy": correct / evaluated if evaluated else 0.0,
            "correct": correct,
            "evaluated": evaluated,
            "total": total,
            "coverage": evaluated / total if total else 0.0,
        }

    return {
        "semantic": agg(is_semantic),
        "syntactic": agg(lambda k: not is_semantic(k)),
        "total": agg(lambda k: True),
        "per_category": {
            k: {**v, "accuracy": v["correct"] / v["evaluated"] if v["evaluated"] else 0.0}
            for k, v in per_category.items()
        },
    }


def to_keyed_vectors(words, matrix):
    kv = KeyedVectors(vector_size=matrix.shape[1])
    kv.add_vectors(list(words), np.asarray(matrix, dtype=np.float32))
    return kv


def word_pairs_spearman(kv, path):
    _, spearman, oov = kv.evaluate_word_pairs(path, restrict_vocab=len(kv.index_to_key), case_insensitive=True)
    return {"spearman": float(spearman.statistic), "pvalue": float(spearman.pvalue), "oov_ratio": float(oov)}


def neighbors(words, matrix, queries=CONTROL_WORDS, k=5):
    unit = normalize_rows(matrix)
    index = {w: i for i, w in enumerate(words)}
    out = {}
    for q in queries:
        if q not in index:
            out[q] = []
            continue
        sims = unit @ unit[index[q]]
        sims[index[q]] = -np.inf
        top = np.argsort(-sims)[:k]
        out[q] = [(words[i], round(float(sims[i]), 3)) for i in top]
    return out


def analogia(a, b, c, k, words, matrix, exclude=True, unit=None):
    unit = normalize_rows(matrix) if unit is None else unit
    index = {w: i for i, w in enumerate(words)}
    query = unit[index[b]] - unit[index[a]] + unit[index[c]]
    query = query / np.linalg.norm(query)
    sims = unit @ query
    if exclude:
        for w in (a, b, c):
            sims[index[w]] = -np.inf
    top = np.argsort(-sims)[:k]
    return [(words[i], float(sims[i])) for i in top], sims


def answer_rank(sims, target_index):
    return int((sims > sims[target_index]).sum()) + 1


def category_pairs(sections):
    pairs = {}
    for category, questions in sections.items():
        seen = []
        for a, b, c, d in questions:
            for pair in ((a, b), (c, d)):
                if pair not in seen:
                    seen.append(pair)
        pairs[category] = seen
    return pairs


def parallelism(words, matrix, pairs_by_category, allowed=None, max_pairs=200):
    unit = normalize_rows(matrix)
    index = {w: i for i, w in enumerate(words)}
    allowed = allowed if allowed is not None else index
    out = {}
    for category, pairs in pairs_by_category.items():
        valid = [(a, b) for a, b in pairs if a in index and b in index and a in allowed and b in allowed]
        valid = valid[:max_pairs]
        if len(valid) < 2:
            out[category] = None
            continue
        diffs = np.stack([unit[index[b]] - unit[index[a]] for a, b in valid])
        diffs = normalize_rows(diffs)
        sims = diffs @ diffs.T
        upper = sims[np.triu_indices(len(valid), k=1)]
        out[category] = {"mean_cosine": float(upper.mean()), "pairs": len(valid)}
    return out


def evaluate_embeddings(words, matrix, sections=None, restrict=30000, methods=("add",)):
    words_r = list(words[:restrict])
    matrix_r = matrix[:restrict]
    result = {f"analogy_{m}": analogy_accuracy(words_r, matrix_r, sections, method=m) for m in methods}
    kv = to_keyed_vectors(words, matrix)
    result["wordsim"] = word_pairs_spearman(kv, WORDSIM_PATH)
    return result
