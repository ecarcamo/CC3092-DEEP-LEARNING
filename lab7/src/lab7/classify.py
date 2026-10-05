import copy
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .data import parallel_tokenize

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def load_ag_news(seed=42):
    from datasets import load_dataset

    news = load_dataset("fancyzhx/ag_news")
    train_text = [t.split() for t in parallel_tokenize(news["train"]["text"])]
    test_text = [t.split() for t in parallel_tokenize(news["test"]["text"])]
    train_y = np.array(news["train"]["label"])
    test_y = np.array(news["test"]["label"])
    idx_train, idx_val = train_test_split(
        np.arange(len(train_y)), test_size=0.1, stratify=train_y, random_state=seed
    )
    return {
        "train": ([train_text[i] for i in idx_train], train_y[idx_train]),
        "val": ([train_text[i] for i in idx_val], train_y[idx_val]),
        "test": (test_text, test_y),
        "raw_test": news["test"]["text"],
    }


def subsample(texts, labels, fraction, seed):
    if fraction >= 1.0:
        return texts, labels
    idx, _ = train_test_split(np.arange(len(labels)), train_size=fraction, stratify=labels, random_state=seed)
    return [texts[i] for i in idx], labels[idx]


def build_vocab(texts, min_freq=2):
    counter = Counter(w for t in texts for w in t)
    words = ["<unk>"] + sorted((w for w, c in counter.items() if c >= min_freq), key=lambda w: -counter[w])
    return words, {w: i for i, w in enumerate(words)}


def oov_rate(texts, vocabulary):
    total = 0
    missing = 0
    for t in texts:
        total += len(t)
        missing += sum(1 for w in t if w not in vocabulary)
    return missing / total


def encode(texts, index):
    ids = [np.array([index.get(w, 0) for w in t], dtype=np.int64) for t in texts]
    return ids


def batches(ids, labels, batch_size, shuffle, rng):
    order = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
    for start in range(0, len(order), batch_size):
        sel = order[start:start + batch_size]
        seqs = [ids[i] for i in sel]
        lengths = np.array([len(s) for s in seqs])
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        flat = np.concatenate(seqs) if lengths.sum() else np.zeros(0, dtype=np.int64)
        yield torch.from_numpy(flat), torch.from_numpy(offsets), torch.from_numpy(labels[sel])


class BagClassifier(nn.Module):
    def __init__(self, vocab_size, dim, hidden=128, classes=4, dropout=0.3, weights=None, freeze=False):
        super().__init__()
        if weights is not None:
            self.embedding = nn.EmbeddingBag.from_pretrained(torch.tensor(weights), freeze=freeze, mode="mean")
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, dim, mode="mean")
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, flat, offsets):
        return self.mlp(self.embedding(flat, offsets))


def pretrained_matrix(vocab_words, emb_words, emb_matrix, seed=0):
    rng = np.random.default_rng(seed)
    index = {w: i for i, w in enumerate(emb_words)}
    std = float(emb_matrix.std())
    matrix = rng.normal(0, std, (len(vocab_words), emb_matrix.shape[1])).astype(np.float32)
    found = 0
    for i, w in enumerate(vocab_words):
        j = index.get(w)
        if j is not None:
            matrix[i] = emb_matrix[j]
            found += 1
    return matrix, found / len(vocab_words)


def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p), "recall": float(r), "f1": float(f1)}


def predict(model, ids, labels, device, batch_size=1024):
    model.eval()
    preds, loss_sum = [], 0.0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for flat, offsets, y in batches(ids, labels, batch_size, False, rng):
            logits = model(flat.to(device), offsets.to(device))
            loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), loss_sum / len(labels)


def train_classifier(model, train, val, device, lr=1e-3, batch_size=256, max_epochs=10, patience=2, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    best_f1, best_state, wait = -1.0, None, 0
    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for flat, offsets, y in batches(*train, batch_size, True, rng):
            logits = model(flat.to(device), offsets.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        preds, val_loss = predict(model, *val, device)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / n, "val_loss": val_loss, **metrics(val[1], preds)}
        history.append(record)
        if record["f1"] > best_f1:
            best_f1, best_state, wait = record["f1"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best = max(history, key=lambda h: h["f1"])
    return {"history": history, "best_epoch": best["epoch"], "val": {k: best[k] for k in ("accuracy", "precision", "recall", "f1")},
            "params_total": total, "params_trainable": trainable, "train_time_s": elapsed}


def tfidf_baseline(train, val, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(analyzer=lambda x: x + [a + " " + b for a, b in zip(x, x[1:])], min_df=2, sublinear_tf=True)
    start = time.perf_counter()
    x_train = vectorizer.fit_transform(train[0])
    clf = LogisticRegression(max_iter=1000, C=10.0, random_state=seed)
    clf.fit(x_train, train[1])
    elapsed = time.perf_counter() - start
    preds = clf.predict(vectorizer.transform(val[0]))
    params = clf.coef_.size + clf.intercept_.size
    return vectorizer, clf, {"val": metrics(val[1], preds), "params_total": int(params), "params_trainable": int(params),
                             "train_time_s": elapsed, "features": x_train.shape[1]}


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()
