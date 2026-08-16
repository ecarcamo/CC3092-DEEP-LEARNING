"""Validación cruzada y bitácora de experimentos.

El preprocesamiento se ajusta DENTRO de cada fold, usando solo su partición de
entrenamiento. Ajustarlo antes de partir filtraría al set de validación las medianas
de imputación y los parámetros de normalización, inflando el resultado.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .data import ROOT, price_bins, rmse
from .model import Historia, ModelConfig, fit_mlp, predict_usd
from .preprocessing import PreprocessConfig, build_preprocessor

REGISTRO = ROOT / "reports" / "experiments.csv"


@dataclass
class ResultadoCV:
    id: str
    descripcion: str
    train_rmse: float          # media entre folds
    val_rmse: float
    val_std: float             # dispersión entre folds: mide cuán confiable es la estimación
    gap: float                 # val - train, en USD
    gap_ratio: float
    epocas_media: float
    segundos: float
    n_features: int
    pre_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    fold_val_rmse: list[float] = field(default_factory=list)
    historias: list[Historia] = field(default_factory=list)

    def fila(self) -> dict:
        """Fila plana para la tabla de iteraciones (§2.3 del enunciado)."""
        return {
            "id": self.id,
            "descripcion": self.descripcion,
            "train_rmse": round(self.train_rmse, 1),
            "val_rmse": round(self.val_rmse, 1),
            "val_std": round(self.val_std, 1),
            "gap": round(self.gap, 1),
            "gap_ratio": round(self.gap_ratio, 3),
            "epocas": round(self.epocas_media, 1),
            "seg": round(self.segundos, 1),
            "n_features": self.n_features,
            "pre_config": json.dumps(self.pre_config),
            "model_config": json.dumps(self.model_config),
        }


def cv_evaluate(X, y, pre_cfg: PreprocessConfig, model_cfg: ModelConfig,
                n_splits: int = 5, seed: int = 42, stratify: bool = True,
                id: str = "", descripcion: str = "", device: str = "cpu",
                guardar_historias: bool = True) -> ResultadoCV:
    """K-fold sobre train_dev. Devuelve métricas agregadas y la dispersión entre folds."""
    t0 = time.perf_counter()
    if stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(X, price_bins(y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(X)

    tr_rmses, va_rmses, epocas, historias = [], [], [], []
    n_features = 0

    for k, (i_tr, i_va) in enumerate(folds):
        X_tr, X_va = X.iloc[i_tr], X.iloc[i_va]
        y_tr, y_va = y.iloc[i_tr].to_numpy(), y.iloc[i_va].to_numpy()

        pre = build_preprocessor(pre_cfg)            # nuevo pipeline por fold
        A = pre.fit_transform(X_tr, y_tr)            # ajustado solo con el train del fold
        B = pre.transform(X_va)
        n_features = A.shape[1]

        cfg_fold = ModelConfig(**{**model_cfg.to_dict(), "seed": model_cfg.seed + k})
        _, hist = fit_mlp(A, y_tr, B, y_va, cfg_fold, device=device)

        tr_rmses.append(hist.best_train_rmse)
        va_rmses.append(hist.best_val_rmse)
        epocas.append(hist.best_epoch)
        if guardar_historias:
            historias.append(hist)

    tr_m, va_m = float(np.mean(tr_rmses)), float(np.mean(va_rmses))
    return ResultadoCV(
        id=id, descripcion=descripcion,
        train_rmse=tr_m, val_rmse=va_m, val_std=float(np.std(va_rmses)),
        gap=va_m - tr_m, gap_ratio=(va_m - tr_m) / tr_m if tr_m else float("nan"),
        epocas_media=float(np.mean(epocas)), segundos=time.perf_counter() - t0,
        n_features=n_features,
        pre_config=pre_cfg.to_dict(), model_config=model_cfg.to_dict(),
        fold_val_rmse=[float(v) for v in va_rmses], historias=historias,
    )


class Bitacora:
    """Registro incremental de iteraciones, persistido en reports/experiments.csv."""

    def __init__(self, path: Path = REGISTRO):
        self.path = path
        self.filas: list[dict] = []
        if path.exists():
            self.filas = pd.read_csv(path).to_dict("records")

    def añadir(self, r: ResultadoCV) -> ResultadoCV:
        self.filas = [f for f in self.filas if f["id"] != r.id]   # reemplaza si se repite
        self.filas.append(r.fila())
        self.guardar()
        return r

    def guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.filas).to_csv(self.path, index=False)

    def tabla(self, cols: list[str] | None = None) -> pd.DataFrame:
        cols = cols or ["id", "descripcion", "train_rmse", "val_rmse", "val_std",
                        "gap", "gap_ratio", "epocas", "n_features"]
        df = pd.DataFrame(self.filas)
        return df[cols].sort_values("val_rmse").reset_index(drop=True) if len(df) else df
