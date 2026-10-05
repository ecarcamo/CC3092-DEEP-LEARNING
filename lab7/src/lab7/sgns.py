import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import keep_probability
from .evaluation import evaluate_embeddings, neighbors


@dataclass
class SGNSConfig:
    name: str = "I01"
    dim: int = 100
    window: int = 5
    negatives: int = 5
    sample: float = 1e-4
    min_count: int = 5
    lr: float = 2e-3
    epochs: int = 5
    batch_size: int = 8192
    corpus_fraction: float = 1.0
    seed: int = 42
    chunk_tokens: int = 1_000_000
    log_every: int = 200
    description: str = ""
    extra: dict = field(default_factory=dict)


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, dim, sparse=True)
        self.out_embed = nn.Embedding(vocab_size, dim, sparse=True)
        nn.init.uniform_(self.in_embed.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, centers, contexts, negatives):
        v = self.in_embed(centers)
        u_pos = self.out_embed(contexts)
        u_neg = self.out_embed(negatives)
        pos = (v * u_pos).sum(dim=1)
        neg = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(dim=1)).mean()

    def embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()


def noise_table(counts, size=20_000_000, power=0.75, device="cuda"):
    probs = counts.astype(np.float64) ** power
    probs /= probs.sum()
    table = np.repeat(np.arange(len(counts), dtype=np.int64), np.round(probs * size).astype(np.int64))
    return torch.tensor(table, device=device)


def epoch_pairs(ids, line_ids, keep_prob, window, chunk_tokens, generator):
    keep = torch.rand(len(ids), device=ids.device, generator=generator) < keep_prob[ids]
    kept_ids = ids[keep]
    kept_lines = line_ids[keep]
    n = len(kept_ids)
    starts = torch.randperm((n + chunk_tokens - 1) // chunk_tokens, generator=generator, device=ids.device).cpu() * chunk_tokens
    for start in starts.tolist():
        end = min(start + chunk_tokens, n)
        pos = torch.arange(start, end, device=ids.device)
        effective = torch.randint(1, window + 1, (end - start,), device=ids.device, generator=generator)
        centers, contexts = [], []
        for offset in range(1, window + 1):
            for sign in (1, -1):
                other = pos + sign * offset
                valid = (other >= 0) & (other < n) & (effective >= offset)
                other_c = other.clamp(0, n - 1)
                valid &= kept_lines[other_c] == kept_lines[pos]
                centers.append(kept_ids[pos[valid]])
                contexts.append(kept_ids[other_c[valid]])
        centers = torch.cat(centers)
        contexts = torch.cat(contexts)
        perm = torch.randperm(len(centers), device=ids.device, generator=generator)
        yield centers[perm], contexts[perm], len(starts), n


def evaluate_model(words, matrix, sections):
    metrics = evaluate_embeddings(words, matrix, sections)
    analogy = metrics["analogy_add"]
    return {
        "analogy_semantic": analogy["semantic"]["accuracy"],
        "analogy_syntactic": analogy["syntactic"]["accuracy"],
        "analogy_total": analogy["total"]["accuracy"],
        "analogy_coverage": analogy["total"]["coverage"],
        "wordsim_spearman": metrics["wordsim"]["spearman"],
        "neighbors": neighbors(words, matrix),
    }


def train_sgns(config, vocab, ids_np, line_ids_np, sections, verbose=True):
    dev = torch.device("cuda")
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=dev)
    generator.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()

    ids = torch.tensor(ids_np, dtype=torch.long, device=dev)
    line_ids = torch.tensor(line_ids_np, dtype=torch.long, device=dev)
    keep_prob = torch.tensor(keep_probability(vocab.counts, config.sample), dtype=torch.float32, device=dev)
    table = noise_table(vocab.counts, device=dev)

    model = SkipGramNS(len(vocab), config.dim).to(dev)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=config.lr)

    history = []
    step_losses = []
    global_step = 0
    total_start = time.perf_counter()
    for epoch in range(config.epochs):
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        pairs_seen, chunk_index, kept_tokens = 0, 0, 0
        for centers, contexts, n_chunks, kept_tokens in epoch_pairs(
            ids, line_ids, keep_prob, config.window, config.chunk_tokens, generator
        ):
            progress = (epoch + chunk_index / n_chunks) / config.epochs
            for group in optimizer.param_groups:
                group["lr"] = config.lr * max(1e-4, 1 - progress)
            for start in range(0, len(centers), config.batch_size):
                c = centers[start:start + config.batch_size]
                o = contexts[start:start + config.batch_size]
                neg_idx = torch.randint(0, len(table), (len(c), config.negatives), device=dev, generator=generator)
                loss = model(c, o, table[neg_idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if global_step % config.log_every == 0:
                    step_losses.append({"epoch": epoch + 1, "step": global_step, "loss": float(loss.item())})
                global_step += 1
                pairs_seen += len(c)
            chunk_index += 1
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        epoch_losses = [s["loss"] for s in step_losses if s["epoch"] == epoch + 1]
        metrics = evaluate_model(vocab.words, model.embeddings(), sections)
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(epoch_losses)),
            "pairs": pairs_seen,
            "kept_tokens": int(kept_tokens),
            "epoch_time_s": epoch_time,
            **metrics,
        }
        history.append(record)
        if verbose:
            print(
                f"[{config.name}] epoch {epoch + 1}/{config.epochs} loss={record['loss']:.4f} "
                f"analogy={record['analogy_total']:.3f} (sem {record['analogy_semantic']:.3f} / "
                f"syn {record['analogy_syntactic']:.3f}) ws={record['wordsim_spearman']:.3f} "
                f"pairs={pairs_seen / 1e6:.1f}M t={epoch_time:.1f}s",
                flush=True,
            )
    total_time = time.perf_counter() - total_start
    result = {
        "config": asdict(config),
        "corpus_tokens": int(len(ids_np)),
        "vocab_size": len(vocab),
        "history": history,
        "step_losses": step_losses,
        "train_time_s": float(sum(h["epoch_time_s"] for h in history)),
        "total_time_s": total_time,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
        "gpu": torch.cuda.get_device_name(0),
        "final": {k: v for k, v in history[-1].items() if k != "neighbors"},
    }
    return model.embeddings(), result
