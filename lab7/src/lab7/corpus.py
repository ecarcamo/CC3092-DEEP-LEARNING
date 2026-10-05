import json

from .data import CACHE_DIR, read_lines

SUBSET_PATH = CACHE_DIR / "wiki_25m.txt"
STARTS_PATH = CACHE_DIR / "wiki_25m_article_starts.json"
FULL_PATH = CACHE_DIR / "wiki_full.txt"


def load_subset(fraction=1.0):
    lines = read_lines(SUBSET_PATH)
    if fraction >= 1.0:
        return lines
    starts = json.loads(STARTS_PATH.read_text())
    target = fraction * sum(len(line) for line in lines)
    acc = 0
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        acc += sum(len(line) for line in lines[start:end])
        if acc >= target:
            return lines[:end]
    return lines


def write_subset_file(fraction, path):
    lines = load_subset(fraction)
    with open(path, "w") as f:
        for line in lines:
            f.write(" ".join(line) + "\n")
    return sum(len(line) for line in lines)
