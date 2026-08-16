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
    val_rmse: float            # RMSE agrupado sobre las predicciones out-of-fold
    val_std: float             # dispersión entre folds: diagnóstico de estabilidad
    gap: float                 # val - train, en USD
    gap_ratio: float
    epocas_media: float
    segundos: float
    n_features: int
    pre_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    fold_val_rmse: list[float] = field(default_factory=list)
    historias: list[Historia] = field(default_factory=list)
    oof_pred: np.ndarray | None = None   # predicción out-of-fold de cada observación

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
                guardar_historias: bool = True,
                excluir_de_train: np.ndarray | None = None) -> ResultadoCV:
    """K-fold sobre train_dev.

    El RMSE de validación se calcula AGRUPANDO las predicciones out-of-fold en un solo
    vector, no promediando el RMSE de cada fold. Promediar los RMSE introduce un sesgo por
    la concavidad de la raíz: repartir los errores uniformemente entre folds sube el
    promedio aunque el error total sea el mismo, lo que penalizaba artificialmente a la
    estratificación y hacía que k=3 y k=10 no fueran comparables entre sí.

    `excluir_de_train` es una máscara booleana de filas que se quitan SOLO de la porción de
    entrenamiento de cada fold. La validación se mantiene íntegra: sacar observaciones
    difíciles del set de validación bajaría el RMSE sin que el modelo haya mejorado, y el
    dataset de prueba de la competencia sí las va a contener.
    """
    t0 = time.perf_counter()
    if stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(X, price_bins(y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(X)

    tr_rmses, va_rmses, epocas, historias = [], [], [], []
    oof = np.full(len(y), np.nan)
    n_features = 0

    for k, (i_tr, i_va) in enumerate(folds):
        if excluir_de_train is not None:
            i_tr = i_tr[~np.asarray(excluir_de_train)[i_tr]]   # solo del train, no del val
        X_tr, X_va = X.iloc[i_tr], X.iloc[i_va]
        y_tr, y_va = y.iloc[i_tr].to_numpy(), y.iloc[i_va].to_numpy()

        pre = build_preprocessor(pre_cfg)            # nuevo pipeline por fold
        A = pre.fit_transform(X_tr, y_tr)            # ajustado solo con el train del fold
        B = pre.transform(X_va)
        n_features = A.shape[1]

        cfg_fold = ModelConfig(**{**model_cfg.to_dict(), "seed": model_cfg.seed + k})
        modelo, hist = fit_mlp(A, y_tr, B, y_va, cfg_fold, device=device)

        oof[i_va] = predict_usd(modelo, B, cfg_fold, device=device)
        tr_rmses.append(hist.best_train_rmse)
        va_rmses.append(hist.best_val_rmse)
        epocas.append(hist.best_epoch)
        if guardar_historias:
            historias.append(hist)

    tr_m = float(np.mean(tr_rmses))
    va_m = rmse(y.to_numpy(), oof)                   # agrupado, no promediado
    return ResultadoCV(
        id=id, descripcion=descripcion,
        train_rmse=tr_m, val_rmse=va_m, val_std=float(np.std(va_rmses)),
        gap=va_m - tr_m, gap_ratio=(va_m - tr_m) / tr_m if tr_m else float("nan"),
        epocas_media=float(np.mean(epocas)), segundos=time.perf_counter() - t0,
        n_features=n_features,
        pre_config=pre_cfg.to_dict(), model_config=model_cfg.to_dict(),
        fold_val_rmse=[float(v) for v in va_rmses], historias=historias, oof_pred=oof,
    )


def cv_multiseed(X, y, pre_cfg: PreprocessConfig, model_cfg: ModelConfig,
                 semillas: tuple[int, ...] = (42, 43, 44), n_splits: int = 5,
                 stratify: bool = False, id: str = "", descripcion: str = "",
                 device: str = "cpu",
                 excluir_de_train: np.ndarray | None = None) -> ResultadoCV:
    """Repite la CV con varias semillas y promedia.

    Una sola semilla no basta: la varianza del estimador está dominada por un puñado de
    observaciones anómalas, y el resultado cambia según en qué fold caigan. Promediar
    varias semillas es lo que permite distinguir una mejora real del ruido de partición.
    """
    corridas = [
        cv_evaluate(X, y, pre_cfg, model_cfg, n_splits=n_splits, seed=s, stratify=stratify,
                    id=f"{id}_s{s}", descripcion=descripcion, device=device,
                    guardar_historias=(s == semillas[0]), excluir_de_train=excluir_de_train)
        for s in semillas
    ]
    val = float(np.mean([c.val_rmse for c in corridas]))
    tr = float(np.mean([c.train_rmse for c in corridas]))
    return ResultadoCV(
        id=id, descripcion=descripcion,
        train_rmse=tr, val_rmse=val,
        val_std=float(np.std([c.val_rmse for c in corridas])),   # dispersión entre semillas
        gap=val - tr, gap_ratio=(val - tr) / tr if tr else float("nan"),
        epocas_media=float(np.mean([c.epocas_media for c in corridas])),
        segundos=float(np.sum([c.segundos for c in corridas])),
        n_features=corridas[0].n_features,
        pre_config=pre_cfg.to_dict(), model_config=model_cfg.to_dict(),
        fold_val_rmse=[v for c in corridas for v in c.fold_val_rmse],
        historias=corridas[0].historias, oof_pred=corridas[0].oof_pred,
    )


class Bitacora:
    """Registro incremental de iteraciones, persistido en reports/experiments.csv."""

    def __init__(self, path: Path = REGISTRO, reset: bool = False):
        """`reset=True` descarta el registro previo.

        El notebook de iteraciones es la única fuente de la bitácora, así que al reejecutarlo
        debe partir de cero: arrastrar filas de corridas anteriores mezcla resultados obtenidos
        con particiones o código distintos.
        """
        self.path = path
        self.filas: list[dict] = []
        if path.exists() and not reset:
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
