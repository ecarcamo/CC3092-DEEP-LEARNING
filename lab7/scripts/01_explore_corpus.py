import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.data import (
    CACHE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    build_vocab,
    corpus_counts,
    count_pairs,
    count_words,
    coverage_of_top,
    encode,
    is_article_title,
    is_heading,
    keep_probability,
    line_lengths,
    load_wikitext,
    nltk_tokenize,
    parallel_tokenize,
    tokenize,
    zipf_slope,
)

TARGET_TOKENS = 25_000_000
MIN_COUNTS = [1, 5, 10, 20]
SAMPLE = 1e-4
WINDOW = 5
SENTENCES = [
    "Don't you think it's too late?",
    "The U.S. economy grew 3.5% in 2004, according to Dr. Smith.",
    "She's a well-known state-of-the-art researcher.",
    "I can't believe they won't come; we'd better leave.",
    "The children's toys cost $1,200.50 at Mr. Brown's shop.",
    "E-mail me at john@example.com or visit www.example.com.",
    "He said: \"Rock 'n' roll is here to stay!\"",
    "The 1990s were a decade of change (e.g., the Internet).",
    "Mother-in-law's recipes are self-explanatory, aren't they?",
    "New York-based firms hired 20,000 workers in Q3.",
    "It is 5 p.m. and the U.K. team's coach isn't here.",
    "Prices fell 2.5 percent to 1,234.56 points on Wall St.",
]


def explore_splits(dataset):
    return {split: corpus_counts(dataset[split]["text"]) for split in dataset}


def tokenize_train(lines):
    tokenized = parallel_tokenize(lines)
    articles = []
    current = None
    for raw, tokens in zip(lines, tokenized):
        if is_article_title(raw):
            current = []
            articles.append(current)
        elif current is not None and tokens and not is_heading(raw):
            current.append(tokens.split())
    return articles


def build_subset(articles, target):
    lines, starts, total = [], [], 0
    for article in articles:
        starts.append(len(lines))
        lines.extend(article)
        total += sum(len(line) for line in article)
        if total >= target:
            break
    return lines, starts, total


def write_corpus(lines, path):
    with open(path, "w") as f:
        for line in lines:
            f.write(" ".join(line) + "\n")


def tokenizer_comparison():
    rows = []
    for sentence in SENTENCES:
        ours = tokenize(sentence)
        theirs = nltk_tokenize(sentence)
        rows.append({"sentence": sentence, "ours": ours, "nltk": theirs, "equal": ours == theirs,
                     "only_ours": sorted(set(ours) - set(theirs)), "only_nltk": sorted(set(theirs) - set(ours))})
    return rows


def subsampling_report(vocab, ids, line_ids):
    keep = keep_probability(vocab.counts, SAMPLE)
    rng = np.random.default_rng(42)
    mask = rng.random(len(ids)) < keep[ids]
    report = {}
    for word in ["the", "of", "and", "in", "king", "computer"]:
        idx = vocab.index[word]
        occurrences = int((ids == idx).sum())
        kept = int(mask[ids == idx].sum())
        report[word] = {
            "frequency": float(vocab.counts[idx] / vocab.counts.sum()),
            "expected_discard": float(1 - keep[idx]),
            "measured_discard": float(1 - kept / occurrences),
        }
    lengths_before = line_lengths(line_ids)
    lengths_after = np.bincount(line_ids[mask], minlength=len(lengths_before))
    pairs = {
        "window": WINDOW,
        "tokens_before": int(len(ids)),
        "tokens_after": int(mask.sum()),
        "fixed_before": count_pairs(lengths_before, WINDOW),
        "fixed_after": count_pairs(lengths_after, WINDOW),
        "dynamic_before": count_pairs(lengths_before, WINDOW, dynamic=True),
        "dynamic_after": count_pairs(lengths_after, WINDOW, dynamic=True),
    }
    return report, pairs


def zipf_figure(counts, slope, intercept, path):
    import matplotlib.pyplot as plt

    ranks = np.arange(1, len(counts) + 1)
    sel = np.unique(np.logspace(0, np.log10(len(counts)), 400).astype(int)) - 1
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.loglog(ranks[sel], counts[sel], ".", ms=3, label="WikiText-103 (25 M tokens)")
    ax.loglog(ranks[sel], np.exp(intercept) * ranks[sel] ** slope, "--", lw=1.2,
              label=f"ajuste: pendiente = {slope:.2f}")
    ax.set_xlabel("Rango de la palabra")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Ley de Zipf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return ranks[sel].tolist(), counts[sel].tolist()


def main():
    t0 = time.time()
    CACHE_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_wikitext()
    splits = explore_splits(dataset)
    print("splits", splits, flush=True)

    articles = tokenize_train(dataset["train"]["text"])
    full_lines = [line for article in articles for line in article]
    full_tokens = sum(len(line) for line in full_lines)
    write_corpus(full_lines, CACHE_DIR / "wiki_full.txt")
    print("full train normalized tokens", full_tokens, flush=True)

    lines, starts, total = build_subset(articles, TARGET_TOKENS)
    write_corpus(lines, CACHE_DIR / "wiki_25m.txt")
    (CACHE_DIR / "wiki_25m_article_starts.json").write_text(json.dumps(starts))
    print("subset", len(starts), "articles", len(lines), "lines", total, "tokens", flush=True)

    counter = count_words(lines)
    full_vocab = build_vocab(counter, 1)
    counts = full_vocab.counts
    slope, intercept = zipf_slope(counts)
    zipf_points = zipf_figure(counts, slope, intercept, FIGURES_DIR / "01_zipf.png")

    min_count_table = []
    for mc in MIN_COUNTS:
        size = int((counts >= mc).sum())
        unk = float(counts[counts < mc].sum() / counts.sum())
        min_count_table.append({"min_count": mc, "vocab_size": size, "unk_rate": unk})

    vocab = build_vocab(counter, 5)
    ids, line_ids = encode(lines, vocab)
    subsampling, pairs = subsampling_report(vocab, ids, line_ids)

    stats = {
        "splits": splits,
        "train_articles_parsed": len(articles),
        "train_normalized_tokens": full_tokens,
        "train_lines_kept": len(full_lines),
        "subset": {"articles": len(starts), "lines": len(lines), "tokens": total,
                   "types": len(full_vocab), "fraction_of_train": total / full_tokens},
        "zipf": {"slope": slope, "intercept": intercept, "points": zipf_points,
                 "top20": list(zip(full_vocab.words[:20], counts[:20].tolist()))},
        "coverage": coverage_of_top(counts, [10, 1000, 30000]),
        "min_count": min_count_table,
        "subsampling": {"sample": SAMPLE, "words": subsampling},
        "pairs": pairs,
        "tokenizer_comparison": tokenizer_comparison(),
        "elapsed_s": time.time() - t0,
    }
    (RESULTS_DIR / "corpus_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("zipf", "tokenizer_comparison")}, indent=1))


if __name__ == "__main__":
    main()
