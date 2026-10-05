import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = LAB_DIR / ".cache"
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

HEADING_RE = re.compile(r"^\s*= [^=].* =\s*$")
ANY_HEADING_RE = re.compile(r"^\s*(= )+.*( =)+\s*$")
TOKEN_RE = re.compile(
    r"[a-z]+(?=n't\b)|n't\b|'[a-z]+\b|[a-z](?:\.[a-z])+\.?|[a-z]+|\d+(?:[.,]\d+)*"
)
WIKITEXT_MARKERS = (
    (" @-@ ", " "),
    (" @,@ ", ","),
    (" @.@ ", "."),
)


def is_article_title(line):
    return bool(HEADING_RE.match(line))


def is_heading(line):
    return bool(ANY_HEADING_RE.match(line))


def normalize(text):
    text = text.lower()
    for marker, replacement in WIKITEXT_MARKERS:
        text = text.replace(marker, replacement)
    text = text.replace("&#39;", "'").replace("#39;", "'").replace("\\", " ")
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+'\s*t\b", "'t", text)
    text = re.sub(r"n\s+'t\b", "n't", text)
    return text


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


def nltk_tokenize(text):
    from nltk.tokenize import TreebankWordTokenizer

    return TreebankWordTokenizer().tokenize(text.lower())


def load_wikitext():
    from datasets import load_dataset

    return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")


def split_articles(lines):
    articles = []
    current = []
    for line in lines:
        if is_article_title(line):
            if current:
                articles.append(current)
            current = []
        elif line.strip() and not is_heading(line):
            current.append(line)
    if current:
        articles.append(current)
    return articles


def corpus_counts(lines):
    raw_tokens = 0
    non_empty = 0
    titles = 0
    for line in lines:
        if line.strip():
            non_empty += 1
            raw_tokens += len(line.split())
            if is_article_title(line):
                titles += 1
    return {"articles": titles, "lines": len(lines), "non_empty_lines": non_empty, "raw_tokens": raw_tokens}


def tokenize_lines(lines):
    return [tokenize(line) for line in lines]


def _tokenize_chunk(lines):
    return [" ".join(tokenize(line)) for line in lines]


def parallel_tokenize(lines, workers=16, chunk=20000):
    from multiprocessing import Pool

    chunks = [lines[i:i + chunk] for i in range(0, len(lines), chunk)]
    with Pool(workers) as pool:
        out = pool.map(_tokenize_chunk, chunks)
    return [line for part in out for line in part]


def build_subset(articles_tokenized, target_tokens):
    lines = []
    article_starts = []
    total = 0
    for article in articles_tokenized:
        article_starts.append(len(lines))
        for line in article:
            if line:
                lines.append(line)
                total += len(line)
        if total >= target_tokens:
            break
    return lines, article_starts, total


def fraction_of_subset(lines, article_starts, fraction):
    total = sum(len(line) for line in lines)
    target = fraction * total
    acc = 0
    for start, end in zip(article_starts, article_starts[1:] + [len(lines)]):
        acc += sum(len(line) for line in lines[start:end])
        if acc >= target:
            return lines[:end]
    return lines


def write_lines(lines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(" ".join(line) + "\n")


def read_lines(path):
    with open(path) as f:
        return [line.split() for line in f]


@dataclass
class Vocab:
    words: list
    counts: np.ndarray

    @property
    def index(self):
        if not hasattr(self, "_index"):
            self._index = {w: i for i, w in enumerate(self.words)}
        return self._index

    def __len__(self):
        return len(self.words)


def count_words(lines):
    counter = Counter()
    for line in lines:
        counter.update(line)
    return counter


def build_vocab(counter, min_count):
    items = sorted(((w, c) for w, c in counter.items() if c >= min_count), key=lambda x: (-x[1], x[0]))
    return Vocab([w for w, _ in items], np.array([c for _, c in items], dtype=np.int64))


def encode(lines, vocab):
    index = vocab.index
    ids = []
    line_ids = []
    for n, line in enumerate(lines):
        encoded = [index[w] for w in line if w in index]
        ids.extend(encoded)
        line_ids.extend([n] * len(encoded))
    return np.array(ids, dtype=np.int32), np.array(line_ids, dtype=np.int32)


def keep_probability(counts, sample):
    total = counts.sum()
    threshold = sample * total
    prob = (np.sqrt(counts / threshold) + 1) * (threshold / counts)
    return np.minimum(prob, 1.0)


def coverage_of_top(counts_sorted, ks):
    total = counts_sorted.sum()
    cum = np.cumsum(counts_sorted)
    return {int(k): float(cum[min(k, len(cum)) - 1] / total) for k in ks}


def zipf_slope(counts_sorted, max_rank=10000):
    ranks = np.arange(1, min(max_rank, len(counts_sorted)) + 1)
    slope, intercept = np.polyfit(np.log(ranks), np.log(counts_sorted[: len(ranks)]), 1)
    return float(slope), float(intercept)


def line_lengths(line_ids):
    return np.bincount(line_ids)


def count_pairs(lengths, window, dynamic=False):
    lengths = np.asarray(lengths, dtype=np.int64)
    total = 0.0
    for offset in range(1, window + 1):
        weight = (window - offset + 1) / window if dynamic else 1.0
        total += 2 * weight * np.clip(lengths - offset, 0, None).sum()
    return float(total)
