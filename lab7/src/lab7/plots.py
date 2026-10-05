import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

COLORS = ["#2a6fdb", "#e4572e", "#29a36a", "#f2a541", "#7b4fd6", "#17a2b8", "#d63a8c", "#6c757d", "#8c564b", "#111111"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tsne_figure(words, matrix, groups, path, seed=42):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    index = {w: i for i, w in enumerate(words)}
    selected, labels, seen = [], [], set()
    for g, (group, text) in enumerate(groups.items()):
        for w in text.split():
            if w in index and w not in seen:
                seen.add(w)
                selected.append(w)
                labels.append(g)
    vectors = matrix[[index[w] for w in selected]]
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    coords = TSNE(n_components=2, perplexity=30, init="pca", metric="cosine", random_state=seed).fit_transform(unit)
    labels = np.array(labels)
    sil_orig = float(silhouette_score(unit, labels, metric="cosine"))
    sil_2d = float(silhouette_score(coords, labels))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    names = list(groups)
    for g, name in enumerate(names):
        pts = coords[labels == g]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=COLORS[g], label=name, alpha=0.8)
    rng = np.random.default_rng(seed)
    for g in range(len(names)):
        members = np.where(labels == g)[0]
        for i in rng.choice(members, size=min(6, len(members)), replace=False):
            ax.annotate(selected[i], coords[i], fontsize=6.5, color=COLORS[g])
    ax.set_title("t-SNE de ~500 palabras (mejor SGNS)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    _save(fig, path)
    return {"words": selected, "labels": labels.tolist(), "groups": names, "coords": coords.round(3).tolist(),
            "silhouette_original": sil_orig, "silhouette_2d": sil_2d, "n_words": len(selected)}


def iteration_curves(results, names, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, name in enumerate(names):
        hist = results[name]["history"]
        epochs = [h["epoch"] for h in hist]
        label = f"{name} ({results[name]['config']['description']})"
        steps = results[name]["step_losses"]
        axes[0].plot([s["step"] for s in steps], [s["loss"] for s in steps], color=COLORS[k], lw=0.8, label=label)
        axes[1].plot(epochs, [h["analogy_total"] for h in hist], "o-", color=COLORS[k], label=label)
        axes[2].plot(epochs, [h["wordsim_spearman"] for h in hist], "o-", color=COLORS[k], label=label)
    axes[0].set(title="Pérdida de entrenamiento", xlabel="paso", ylabel="pérdida SGNS")
    axes[1].set(title="Accuracy de analogías (3CosAdd)", xlabel="epoch", ylabel="accuracy")
    axes[2].set(title="Spearman WordSim-353", xlabel="epoch", ylabel="ρ")
    axes[1].legend(fontsize=6.5)
    _save(fig, path)


def loss_vs_quality(results, path):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for k, (name, r) in enumerate(results.items()):
        f = r["history"][-1]
        ax.scatter(f["loss"], f["analogy_total"], color=COLORS[k % len(COLORS)])
        ax.annotate(f"{name} {r['config']['description']}", (f["loss"], f["analogy_total"]), fontsize=6.5)
    ax.set(xlabel="pérdida final (último epoch)", ylabel="accuracy de analogías", title="¿Menor pérdida = mejores embeddings?")
    _save(fig, path)


def corpus_size_plot(sgns_points, gensim_points, glove_acc, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for points, label, color in ((sgns_points, "SGNS (PyTorch)", COLORS[0]), (gensim_points, "Word2Vec gensim", COLORS[1])):
        xs, ys = zip(*sorted(points))
        ax.plot(xs, ys, "o-", color=color, label=label)
    ax.axhline(glove_acc, color=COLORS[2], ls="--", label="GloVe 6B (referencia)")
    ax.set_xscale("log")
    ticks = sorted(x for x, _ in list(sgns_points) + list(gensim_points))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([ticks[0], ticks[2], ticks[4], ticks[-1]], [f"{t / 1e6:.0f} M" for t in (ticks[0], ticks[2], ticks[4], ticks[-1])])
    ax.set(xlabel="tokens del corpus de entrenamiento", ylabel="accuracy total de analogías (3CosAdd)",
           title="Analogías vs tamaño del corpus")
    ax.legend(fontsize=8)
    _save(fig, path)


def fraction_plot(curves, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k, (label, points) in enumerate(curves.items()):
        xs = sorted(points)
        means = [np.mean(points[x]) for x in xs]
        stds = [np.std(points[x]) for x in xs]
        ax.errorbar(xs, means, yerr=stds, fmt="o-", ms=4, capsize=2, color=COLORS[k], label=label)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(xs, [f"{100 * x:g} %" for x in xs])
    ax.set(xlabel="fracción de train de AG News", ylabel="F1 macro en test", title="F1 de test vs datos de entrenamiento")
    ax.legend(fontsize=7)
    _save(fig, path)


def classification_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for k, (name, r) in enumerate(runs.items()):
        hist = r["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], "-", color=COLORS[k], label=f"{name} train")
        axes[0].plot(epochs, [h["val_loss"] for h in hist], "--", color=COLORS[k], label=f"{name} val")
        axes[1].plot(epochs, [h["f1"] for h in hist], "o-", color=COLORS[k], ms=3, label=name)
    axes[0].set(title="Pérdida (— train, -- validación)", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="F1 macro en validación", xlabel="epoch", ylabel="F1")
    axes[1].legend(fontsize=7)
    _save(fig, path)


def confusion_figure(matrices, labels, path):
    fig, axes = plt.subplots(1, len(matrices), figsize=(2.6 * len(matrices), 2.8))
    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        cm = np.array(cm)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(name, fontsize=9)
    _save(fig, path)


def category_bars(models, path):
    cats = list(next(iter(models.values()))["analogy_add"]["per_category"])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    width = 0.8 / len(models)
    for k, (name, m) in enumerate(models.items()):
        accs = [m["analogy_add"]["per_category"][c]["accuracy"] for c in cats]
        ax.bar(np.arange(len(cats)) + k * width, accs, width, color=COLORS[k], label=name)
    ax.set_xticks(np.arange(len(cats)) + width, [c.replace("gram", "g") for c in cats], rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="accuracy (3CosAdd)", title="Analogías por categoría (vocabulario compartido de 30 000)")
    ax.legend(fontsize=8)
    _save(fig, path)
