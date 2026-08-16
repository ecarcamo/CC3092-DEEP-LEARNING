"""Carga del dataset y estrategias de división.

La lectura del CSV usa keep_default_na=False porque pandas interpreta los
literales "NA" y "None" como nulos, y en Ames son categorías válidas
("la vivienda no tiene esa característica"). Ver §1.1 del EDA.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CONTRATO = ROOT / "data" / "processed" / "column_types.json"

TARGET = "SalePrice"
ID_COL = "Id"

# Columnas donde el vacío significa "no posee la característica" (verificado en §5 del EDA)
NA_ESTRUCTURAL_CAT = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
]
NA_ESTRUCTURAL_NUM = ["GarageYrBlt"]


def load_contrato() -> dict:
    """Contrato de tipos de columna producido por el EDA."""
    return json.loads(CONTRATO.read_text(encoding="utf-8"))


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lee el CSV preservando 'NA'/'None' como categorías y limpia las comillas literales."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for c in df.select_dtypes(include="str").columns:
        df[c] = df[c].str.strip().str.strip("'").str.strip()
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separa features y target. Devuelve y=None si el CSV no trae SalePrice."""
    y = df[TARGET].astype(float) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    return X, y


def price_bins(y: pd.Series, n_bins: int = 20) -> np.ndarray:
    """Estratos por cuantil de precio, para que la cola derecha se reparta uniformemente.

    Se usan 20 bins y no 10: con 10, el decil superior agrupa ~117 viviendas y las pocas
    verdaderamente extremas (>500 k) pueden caer todas del mismo lado por azar. Como el RMSE
    está dominado por esas observaciones, eso hace que la partición no sea representativa.
    """
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").to_numpy()


def holdout_split(X, y, test_size=0.15, seed=42, stratify=True, n_bins: int = 20):
    """Partición train_dev / test interno. El test interno se toca una sola vez."""
    strat = price_bins(y, n_bins) if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
