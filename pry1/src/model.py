"""MLP en PyTorch con early stopping.

El RMSE se reporta SIEMPRE en la escala original (USD), aunque el entrenamiento
ocurra sobre log1p(SalePrice): la métrica de la competencia se mide en USD, así que
comparar iteraciones en escala logarítmica daría un ranking distinto al real.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    batchnorm: bool = True
    activation: str = "relu"
    # --- optimización
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 400
    patience: int = 40           # early stopping sobre el RMSE de validación
    scheduler: str = "plateau"   # "plateau" | "cosine" | "none"
    loss: str = "mse"            # "mse" | "huber"
    # --- datos
    log_target: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


_ACTIVACIONES = {
    "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
    "leakyrelu": nn.LeakyReLU, "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(self, in_dim: int, cfg: ModelConfig):
        super().__init__()
        act = _ACTIVACIONES[cfg.activation]
        capas: list[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden:
            capas.append(nn.Linear(prev, h))
            if cfg.batchnorm:
                capas.append(nn.BatchNorm1d(h))
            capas.append(act())
            if cfg.dropout > 0:
                capas.append(nn.Dropout(cfg.dropout))
            prev = h
        capas.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*capas)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@dataclass
class Historia:
    """Curvas de entrenamiento y métricas de la corrida (para §2.3 del enunciado)."""
    train_rmse: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_rmse: float = float("inf")
    best_train_rmse: float = float("inf")

    @property
    def gap(self) -> float:
        """Brecha de generalización en USD. Crece cuando el modelo memoriza."""
        return self.best_val_rmse - self.best_train_rmse

    @property
    def gap_ratio(self) -> float:
        return self.gap / self.best_train_rmse if self.best_train_rmse else float("nan")


def _predict(model, X, device) -> np.ndarray:
    """Salida cruda de la red, en el espacio estandarizado del target."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def _a_usd(z: np.ndarray, model: "MLP", log_target: bool) -> np.ndarray:
    """Revierte la estandarización del target y, si aplica, la transformación log."""
    v = z * model.y_sigma_ + model.y_mu_
    return np.expm1(v) if log_target else v


def fit_mlp(X_train, y_train, X_val, y_val, cfg: ModelConfig,
            device: str = "cpu") -> tuple[MLP, Historia]:
    """Entrena con early stopping y devuelve el modelo en su mejor época."""
    set_seed(cfg.seed)
    dev = torch.device(device)

    # Transformación del target. Además de log1p opcional, se estandariza: sin esto la red
    # arranca prediciendo 0 y necesita miles de pasos solo para llevar el bias hasta ~12.
    fwd = (lambda v: np.log1p(v)) if cfg.log_target else (lambda v: v)

    ytr_t = np.asarray(fwd(y_train), dtype=np.float32)
    y_mu, y_sigma = float(ytr_t.mean()), float(ytr_t.std()) or 1.0
    ytr_z = (ytr_t - y_mu) / y_sigma

    Xtr_t = torch.as_tensor(np.asarray(X_train, dtype=np.float32), device=dev)
    ytr_tt = torch.as_tensor(ytr_z, device=dev)

    model = MLP(Xtr_t.shape[1], cfg).to(dev)
    # Parámetros de la estandarización del target: viajan con el modelo al guardarlo
    model.y_mu_, model.y_sigma_ = y_mu, y_sigma
    inv = lambda z: _a_usd(z, model, cfg.log_target)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.HuberLoss(delta=1.0) if cfg.loss == "huber" else nn.MSELoss()

    if cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    elif cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    else:
        sched = None

    hist = Historia()
    n = Xtr_t.shape[0]
    mejor_estado, sin_mejora = None, 0
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if idx.numel() < 2:      # BatchNorm necesita al menos 2 muestras
                continue
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_tt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()

        tr = _rmse(y_train, inv(_predict(model, X_train, dev)))
        va = _rmse(y_val, inv(_predict(model, X_val, dev)))
        hist.train_loss.append(total / n)
        hist.train_rmse.append(tr)
        hist.val_rmse.append(va)

        if sched is not None:
            sched.step(va) if cfg.scheduler == "plateau" else sched.step()

        if va < hist.best_val_rmse:
            hist.best_val_rmse, hist.best_train_rmse, hist.best_epoch = va, tr, epoch
            mejor_estado = {k: v.detach().clone() for k, v in model.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= cfg.patience:
                break

    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)
    return model, hist


def predict_usd(model: MLP, X, cfg: ModelConfig, device: str = "cpu") -> np.ndarray:
    """Predicción en USD, revirtiendo estandarización y transformación log del target."""
    z = _predict(model, np.asarray(X, dtype=np.float32), torch.device(device))
    return _a_usd(z, model, cfg.log_target)
