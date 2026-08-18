"""Pipeline de preprocesamiento serializable.

Todo el preprocesamiento vive en un objeto sklearn que se ajusta SOLO con datos de
entrenamiento y se guarda con joblib. El día de la competencia se carga y se aplica
tal cual al dataset nuevo: mismas transformaciones, mismos parámetros de normalización
(recomendación 2 del enunciado).

Las decisiones implementadas provienen de la §11 del EDA. Las marcadas como
"hipótesis" quedan detrás de flags de `PreprocessConfig` para poder hacer A/B.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NA_ESTRUCTURAL_CAT, NA_ESTRUCTURAL_NUM, load_contrato

_CONTRATO = load_contrato()
ORDINAL_MAPS: dict[str, list[str]] = _CONTRATO["ordinal_maps"]
NOMINAL_COLS: list[str] = _CONTRATO["columnas"]["nominales"]
CUASI_CONSTANTES: list[str] = _CONTRATO["columnas_cuasi_constantes"]


@dataclass
class PreprocessConfig:
    """Flags de las decisiones que el EDA dejó como hipótesis a validar."""
    derived: bool = True            # §11.15 — TotalSF, HouseAge, TotalBaths...
    log_skewed: bool = True         # §11.12 — log1p a numéricas con |skew| > umbral
    skew_threshold: float = 0.75
    drop_quasi_constant: bool = True  # §11.9
    rare_min_count: int = 5         # §11.8 — agrupa categorías raras (0 = desactivado)
    scale: bool = True              # §11.10 — estandarizar
    neighborhood_target_encoding: bool = False   # investigación post-competencia (17/ago)

    def to_dict(self) -> dict:
        return asdict(self)


class NeighborhoodTargetEncoder(TransformerMixin, BaseEstimator):
    """Reemplaza `Neighborhood` por la media suavizada de precio del barrio, ajustada
    SOLO con el train de cada fold/partición.

    `Neighborhood` es la categórica más predictiva del dataset (η²=0.53, ver INFORME.md)
    y consume ~25 columnas one-hot. En un dataset de 217 features para 1168 filas, esas
    25 columnas dispersas agravan la razón features/observaciones que el propio EDA
    identificó como el riesgo dominante. Codificarla como una sola columna continua
    reduce esa dimensionalidad sin perder la señal.

    La media se suaviza hacia la media global con `smoothing` para que los barrios con
    pocas observaciones en el fold no memoricen su propio precio (equivalente a un prior
    bayesiano con `smoothing` pseudo-observaciones en la media global).
    """

    def __init__(self, col: str = "Neighborhood", smoothing: float = 10.0):
        self.col = col
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("NeighborhoodTargetEncoder necesita y en fit()")
        y = np.asarray(y, dtype=float)
        self.media_global_ = float(y.mean())
        tmp = pd.DataFrame({self.col: X[self.col].to_numpy(), "y": y})
        stats = tmp.groupby(self.col)["y"].agg(["mean", "count"])
        self.mapa_ = (
            (stats["count"] * stats["mean"] + self.smoothing * self.media_global_)
            / (stats["count"] + self.smoothing)
        ).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Se sobrescribe con floats manteniendo el nombre de columna: el selector de
        # numéricas la recoge automáticamente por dtype, sin tocar el resto del pipeline.
        X[self.col] = X[self.col].map(self.mapa_).fillna(self.media_global_).astype(float)
        return X


class AmesCleaner(BaseEstimator, TransformerMixin):
    """Limpieza con reglas de dominio: NA estructural, ordinales y features derivadas.

    Es un transformer con estado: aprende del set de entrenamiento las medianas de
    LotFrontage por barrio y la mediana global de respaldo.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.frontage_por_barrio_ = (
            X.groupby("Neighborhood")["LotFrontage"].median()
            if {"Neighborhood", "LotFrontage"} <= set(X.columns) else pd.Series(dtype=float)
        )
        self.frontage_global_ = float(X["LotFrontage"].median()) if "LotFrontage" in X else 0.0
        self.columnas_salida_ = self.transform(X).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cfg = self.config

        # 1. NA estructural: el vacío significa "no posee la característica"
        for c in NA_ESTRUCTURAL_CAT:
            if c in X.columns:
                X[c] = X[c].astype("object").fillna("NA")
        for c in NA_ESTRUCTURAL_NUM:
            if c in X.columns:
                X[c] = X[c].fillna(0)
        if "MasVnrType" in X.columns:
            X["MasVnrType"] = X["MasVnrType"].astype("object").fillna("None")

        # 2. LotFrontage: mediana por barrio aprendida en fit (§11.6)
        if "LotFrontage" in X.columns:
            faltan = X["LotFrontage"].isna()
            if faltan.any():
                por_barrio = X.loc[faltan, "Neighborhood"].map(self.frontage_por_barrio_)
                X.loc[faltan, "LotFrontage"] = por_barrio.fillna(self.frontage_global_)

        # 3. Ordinales → entero respetando el orden semántico (§11.4)
        for col, niveles in ORDINAL_MAPS.items():
            if col in X.columns:
                mapa = {nivel: i for i, nivel in enumerate(niveles)}
                X[col] = X[col].astype("object").map(mapa).astype(float)

        # 4. MSSubClass es un código de clase, no una magnitud (§11.3)
        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)

        # 5. Features derivadas (§11.15)
        if cfg.derived:
            X = self._agregar_derivadas(X)

        # 6. Columnas cuasi-constantes (§11.9)
        if cfg.drop_quasi_constant:
            X = X.drop(columns=[c for c in CUASI_CONSTANTES if c in X.columns])

        return X

    @staticmethod
    def _agregar_derivadas(X: pd.DataFrame) -> pd.DataFrame:
        def col(name):
            return X[name] if name in X.columns else 0

        X["TotalSF"] = col("TotalBsmtSF") + col("1stFlrSF") + col("2ndFlrSF")
        X["TotalBaths"] = (col("FullBath") + 0.5 * col("HalfBath")
                           + col("BsmtFullBath") + 0.5 * col("BsmtHalfBath"))
        X["TotalPorchSF"] = (col("OpenPorchSF") + col("EnclosedPorch") + col("3SsnPorch")
                             + col("ScreenPorch") + col("WoodDeckSF"))
        X["HouseAge"] = (col("YrSold") - col("YearBuilt")).clip(lower=0)
        X["RemodAge"] = (col("YrSold") - col("YearRemodAdd")).clip(lower=0)
        X["IsRemodeled"] = (col("YearRemodAdd") != col("YearBuilt")).astype(int)
        X["HasGarage"] = (col("GarageArea") > 0).astype(int)
        X["HasBsmt"] = (col("TotalBsmtSF") > 0).astype(int)
        X["HasFireplace"] = (col("Fireplaces") > 0).astype(int)
        X["Has2ndFlr"] = (col("2ndFlrSF") > 0).astype(int)
        X["OverallGrade"] = col("OverallQual") * col("OverallCond")
        return X


class Log1pSesgadas(TransformerMixin, BaseEstimator):
    """Aplica log1p a las columnas con asimetría alta, aprendidas en fit.

    Solo actúa sobre columnas no negativas: log1p no está definido por debajo de -1.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore"):
            sesgo = pd.DataFrame(X).skew().to_numpy()
        no_negativas = np.nanmin(X, axis=0) >= 0
        self.mask_ = (np.abs(sesgo) > self.threshold) & no_negativas
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.mask_] = np.log1p(X[:, self.mask_])
        return X


def _selector_numericas(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include="number").columns.tolist()


def _selector_nominales(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(exclude="number").columns.tolist()


def build_preprocessor(config: PreprocessConfig | None = None) -> Pipeline:
    """Pipeline completo: limpieza → imputación → transformación → codificación → escalado."""
    cfg = config or PreprocessConfig()

    pasos_num = [("imputer", SimpleImputer(strategy="median"))]
    if cfg.log_skewed:
        pasos_num.append(("log1p", Log1pSesgadas(threshold=cfg.skew_threshold)))
    if cfg.scale:
        pasos_num.append(("scaler", StandardScaler()))

    # handle_unknown="ignore" protege contra categorías nuevas en el dataset de prueba
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist" if cfg.rare_min_count else "ignore",
        min_frequency=cfg.rare_min_count or None,
        sparse_output=False,
    )
    pasos_cat = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]

    pasos_pipeline = [("cleaner", AmesCleaner(cfg))]
    if cfg.neighborhood_target_encoding:
        pasos_pipeline.append(("neigh_te", NeighborhoodTargetEncoder()))

    return Pipeline([
        *pasos_pipeline,
        ("encoder", ColumnTransformer([
            ("num", Pipeline(pasos_num), _selector_numericas),
            ("cat", Pipeline(pasos_cat), _selector_nominales),
        ], remainder="drop", verbose_feature_names_out=False)),
    ])
