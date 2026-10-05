import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7.data import FIGURES_DIR, RESULTS_DIR
from lab7.evaluation import (
    SIMLEX_PATH,
    WORDSIM_PATH,
    analogia,
    analogy_accuracy,
    answer_rank,
    category_pairs,
    load_analogies,
    normalize_rows,
    parallelism,
    to_keyed_vectors,
    word_pairs_spearman,
)
from lab7.models import load_three, restrict, shared_vocabulary
from lab7.plots import tsne_figure

ANALOGIES = [
    ("género", "man", "woman", "king", "queen"),
    ("género", "brother", "sister", "father", "mother"),
    ("género", "actor", "actress", "prince", "princess"),
    ("país–capital", "france", "paris", "italy", "rome"),
    ("país–capital", "japan", "tokyo", "germany", "berlin"),
    ("país–capital", "spain", "madrid", "russia", "moscow"),
    ("país–gentilicio", "france", "french", "germany", "german"),
    ("país–gentilicio", "china", "chinese", "japan", "japanese"),
    ("país–gentilicio", "italy", "italian", "mexico", "mexican"),
    ("comparativo/superlativo", "good", "better", "bad", "worse"),
    ("comparativo/superlativo", "big", "bigger", "small", "smaller"),
    ("comparativo/superlativo", "fast", "fastest", "strong", "strongest"),
    ("tiempo verbal", "walk", "walked", "run", "ran"),
    ("tiempo verbal", "go", "went", "see", "saw"),
    ("tiempo verbal", "eat", "ate", "take", "took"),
    ("plural", "car", "cars", "child", "children"),
    ("plural", "dog", "dogs", "city", "cities"),
    ("plural", "man", "men", "woman", "women"),
]

GROUPS = {
    "países": "france germany italy spain russia china japan india brazil mexico canada australia egypt turkey greece poland sweden norway denmark finland ireland portugal belgium netherlands austria switzerland hungary romania ukraine iran iraq israel pakistan afghanistan indonesia vietnam thailand korea argentina chile peru colombia venezuela cuba nigeria kenya ethiopia morocco algeria libya sudan syria lebanon jordan serbia croatia bulgaria albania scotland wales england",
    "números": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion dozen first second third fourth fifth sixth seventh eighth ninth tenth half quarter double triple single several few many numerous zero hundreds thousands millions twice thrice once",
    "animales": "dog cat horse cow pig sheep goat chicken duck goose rabbit mouse rat wolf fox bear lion tiger leopard elephant giraffe zebra monkey ape gorilla deer elk moose camel donkey mule snake lizard frog toad turtle crocodile shark whale dolphin seal otter beaver squirrel bat owl eagle hawk falcon parrot pigeon crow swan penguin salmon trout bee ant spider butterfly",
    "verbos": "walk run jump swim climb throw catch push pull carry lift drop eat drink sleep sing dance write read speak listen watch build destroy create break open close buy sell pay borrow lend give take send receive bring teach learn study think believe know understand remember forget explain describe discuss argue decide choose",
    "tiempo": "january february march april may june july august september october november december monday tuesday wednesday thursday friday saturday sunday morning afternoon evening night midnight noon week month year decade century spring summer autumn winter yesterday today tomorrow weekend dawn dusk season annual weekly daily hourly",
    "colores": "red blue green yellow black white orange purple pink brown grey gray violet scarlet crimson maroon turquoise beige ivory golden silver bronze cyan magenta navy olive teal indigo emerald amber lavender",
    "profesiones": "doctor nurse teacher lawyer engineer farmer soldier pilot sailor priest bishop judge architect scientist professor student actor singer writer poet painter sculptor composer musician journalist editor photographer chef baker butcher carpenter plumber mechanic electrician merchant banker accountant economist politician diplomat ambassador senator governor mayor president minister officer sergeant captain detective surgeon dentist pharmacist librarian",
    "deportes": "football soccer baseball basketball hockey tennis golf cricket rugby boxing wrestling swimming cycling skiing skating volleyball handball badminton fencing rowing sailing surfing athletics marathon sprint polo karate judo archery gymnastics",
}


def evaluate_individual(name, words, matrix):
    unit = normalize_rows(matrix)
    index = {w: i for i, w in enumerate(words)}
    kv = to_keyed_vectors(words, matrix)
    rows = []
    for kind, a, b, c, expected in ANALOGIES:
        if not all(w in index for w in (a, b, c, expected)):
            rows.append({"type": kind, "query": [a, b, c], "expected": expected, "missing": True})
            continue
        top, sims = analogia(a, b, c, 5, words, matrix, exclude=True, unit=unit)
        top_all, _ = analogia(a, b, c, 5, words, matrix, exclude=False, unit=unit)
        gensim_top = [w for w, _ in kv.most_similar(positive=[b, c], negative=[a], topn=5)]
        rows.append({
            "type": kind,
            "query": [a, b, c],
            "expected": expected,
            "top5": [(w, round(s, 4)) for w, s in top],
            "correct": top[0][0] == expected,
            "rank": answer_rank(sims, index[expected]),
            "cosine_expected": round(float(sims[index[expected]]), 4),
            "top5_no_exclusion": [(w, round(s, 4)) for w, s in top_all],
            "first_no_exclusion": top_all[0][0],
            "first_is_query": top_all[0][0] in (a, b, c),
            "matches_gensim": [w for w, _ in top] == gensim_top,
        })
    return rows


def main():
    sections = load_analogies()
    models = load_three()
    reference = models["SGNS"][0]
    shared = shared_vocabulary(models, reference)
    shared_set = set(shared)
    pairs = category_pairs(sections)
    total_questions = sum(len(q) for q in sections.values())

    out = {"shared_vocab_size": len(shared), "total_questions": total_questions, "models": {}}
    for name, (words, matrix) in models.items():
        print(name, len(words), matrix.shape, flush=True)
        sw, sm = restrict(words, matrix, shared)
        kv = to_keyed_vectors(words, matrix)
        par = parallelism(words, matrix, pairs, allowed=shared_set)
        valid_par = [v["mean_cosine"] for v in par.values() if v]
        individual = evaluate_individual(name, words, matrix)
        out["models"][name] = {
            "vocab_size": len(words),
            "dim": int(matrix.shape[1]),
            "analogy_add": analogy_accuracy(sw, sm, sections, method="add"),
            "analogy_mul": analogy_accuracy(sw, sm, sections, method="mul"),
            "wordsim": word_pairs_spearman(kv, WORDSIM_PATH),
            "simlex": word_pairs_spearman(kv, SIMLEX_PATH),
            "parallelism": par,
            "parallelism_mean": float(np.mean(valid_par)),
            "individual": individual,
            "individual_accuracy": float(np.mean([r.get("correct", False) for r in individual])),
            "matches_gensim_all": all(r.get("matches_gensim", True) for r in individual),
        }
        m = out["models"][name]
        print(f"  add={m['analogy_add']['total']['accuracy']:.3f} mul={m['analogy_mul']['total']['accuracy']:.3f} "
              f"cov={m['analogy_add']['total']['coverage']:.3f} ws={m['wordsim']['spearman']:.3f} "
              f"simlex={m['simlex']['spearman']:.3f} par={m['parallelism_mean']:.3f} "
              f"indiv={m['individual_accuracy']:.2f} gensim_match={m['matches_gensim_all']}", flush=True)

    words, matrix = models["SGNS"]
    out["tsne"] = tsne_figure(words, matrix, GROUPS, FIGURES_DIR / "09_tsne_sgns.png")
    print("tsne", out["tsne"]["silhouette_original"], out["tsne"]["silhouette_2d"], out["tsne"]["n_words"])
    (RESULTS_DIR / "vector_arithmetic.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
