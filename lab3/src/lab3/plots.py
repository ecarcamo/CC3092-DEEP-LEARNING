"""Gráficas del laboratorio: distribución de longitudes, curvas de pérdida y matrices de confusión."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KIND_TITLES = {"mlp": "MLP (baseline)", "rnn": "RNN simple", "lstm": "LSTM"}
FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def _save(fig, filename: str | None):
    if filename:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / filename, dpi=140, bbox_inches="tight")


def plot_length_distribution(lengths: np.ndarray, max_len: int = 300, filename: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(lengths, bins=80, color="#4C72B0", edgecolor="white")
    axes[0].axvline(max_len, color="#C44E52", ls="--", label=f"max_len = {max_len}")
    axes[0].axvline(np.median(lengths), color="#55A868", ls="--",
                    label=f"mediana = {np.median(lengths):.0f}")
    axes[0].set(xlabel="tokens por reseña", ylabel="frecuencia",
                title="Distribución de longitudes (train)")
    axes[0].legend()

    axes[1].boxplot(lengths, vert=False, widths=0.6)
    axes[1].set(xlabel="tokens por reseña", title="Boxplot (escala log)", xscale="log")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_loss_curves(results: list[dict], kind: str, filename: str | None = None):
    """Una fila de subplots con la curva train/val de cada iteración de una arquitectura."""
    subset = [r for r in results if r["kind"] == kind]
    fig, axes = plt.subplots(1, len(subset), figsize=(4 * len(subset), 3.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, subset):
        epochs = [h["epoch"] for h in r["history"]]
        ax.plot(epochs, [h["train_loss"] for h in r["history"]], marker="o", ms=3, label="train")
        ax.plot(epochs, [h["val_loss"] for h in r["history"]], marker="s", ms=3, label="val")
        ax.axvline(r["best_epoch"], color="grey", ls=":", lw=1)
        ax.set(xlabel="epoch",
               title=f"{r['name']}\nval_acc {r['val_metrics']['accuracy']:.3f}")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    fig.suptitle(f"Curvas de pérdida — {KIND_TITLES[kind]}", y=1.06, fontsize=13)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_confusion_matrices(entries: list[dict], filename: str | None = None):
    """`entries`: [{'title': str, 'cm': 2x2}] — una matriz por modelo final."""
    fig, axes = plt.subplots(1, len(entries), figsize=(4 * len(entries), 3.6))
    axes = np.atleast_1d(axes)
    labels = ["negativa", "positiva"]

    for ax, e in zip(axes, entries):
        cm = np.asarray(e["cm"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
               xlabel="predicción", ylabel="real", title=e["title"])
    fig.suptitle("Matrices de confusión sobre test", y=1.04, fontsize=13)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_length_experiment(length_results: list[dict], filename: str | None = None):
    """Accuracy de validación de RNN vs LSTM para cada longitud máxima de secuencia."""
    lens = sorted({r["max_len"] for r in length_results})
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35

    for offset, kind, color in [(-width / 2, "rnn", "#C44E52"), (width / 2, "lstm", "#4C72B0")]:
        accs = [next(r["val_metrics"]["accuracy"] for r in length_results
                     if r["kind"] == kind and r["max_len"] == L) for L in lens]
        ax.bar(np.arange(len(lens)) + offset, accs, width, label=KIND_TITLES[kind], color=color)
        for x, a in zip(np.arange(len(lens)) + offset, accs):
            ax.text(x, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)

    ax.set(xticks=range(len(lens)), xticklabels=[f"max_len = {L}" for L in lens],
           ylabel="accuracy (validación)", ylim=(0.5, 1.0),
           title="Sección 4.1 — efecto de la longitud de secuencia")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)
    return fig
