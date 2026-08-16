#!/usr/bin/env python
"""Predicción sobre un dataset nuevo — script de la competencia.

Carga el pipeline de preprocesamiento y el modelo entrenados, aplica exactamente las
mismas transformaciones que en entrenamiento y escribe las predicciones. Si el CSV de
entrada trae la columna SalePrice, además reporta el RMSE.

Uso el día de la presentación:

    python predict.py --input data/raw/test.csv
    python predict.py --input data/raw/test.csv --output submissions/preds.csv

No requiere ningún paso manual: todo lo que necesita está en models/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import ID_COL, TARGET, load_raw, rmse
from src.model import MLP, ModelConfig

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def cargar_artefactos(dir_modelos: Path = MODELS):
    """Carga pipeline, pesos y metadatos guardados por el notebook 05.

    Devuelve una lista de modelos: si el notebook 05 decidió usar un ensemble, las
    predicciones se promedian entre todos.
    """
    meta = json.loads((dir_modelos / "metadata.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(dir_modelos / "final_pipeline.joblib")

    cfg = ModelConfig(**{**meta["model_config"], "hidden": tuple(meta["model_config"]["hidden"])})
    estado = torch.load(dir_modelos / "final_model.pt", map_location="cpu", weights_only=True)
    pesos = estado.get("ensemble") or [estado]

    modelos = []
    for p in pesos:
        m = MLP(meta["n_features"], cfg)
        m.load_state_dict(p["state_dict"])
        m.y_mu_, m.y_sigma_ = p["y_mu"], p["y_sigma"]
        m.eval()
        modelos.append(m)
    return pipeline, modelos, cfg, meta


def predecir(csv_entrada: Path, dir_modelos: Path = MODELS) -> pd.DataFrame:
    pipeline, modelos, cfg, meta = cargar_artefactos(dir_modelos)

    df = load_raw(csv_entrada)          # misma lectura que en entrenamiento (keep_default_na=False)
    ids = df[ID_COL] if ID_COL in df.columns else pd.Series(range(len(df)), name=ID_COL)
    y_real = df[TARGET].astype(float) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])

    A = pipeline.transform(X)           # transform, NUNCA fit: los parámetros vienen del train
    if A.shape[1] != meta["n_features"]:
        raise ValueError(
            f"El preprocesamiento produjo {A.shape[1]} features y el modelo espera "
            f"{meta['n_features']}. El pipeline y el modelo no corresponden entre sí."
        )

    tensor = torch.as_tensor(np.asarray(A, dtype=np.float32))
    predicciones = []
    for m in modelos:
        with torch.no_grad():
            z = m(tensor).numpy()
        p = z * m.y_sigma_ + m.y_mu_
        predicciones.append(np.expm1(p) if cfg.log_target else p)
    pred = np.clip(np.mean(predicciones, axis=0), 0, None)  # un precio negativo nunca es válido

    salida = pd.DataFrame({ID_COL: ids.to_numpy(), TARGET: pred})
    if y_real is not None:
        salida.attrs["rmse"] = rmse(y_real, pred)
        salida.attrs["y_real"] = y_real.to_numpy()
    return salida


def main() -> int:
    p = argparse.ArgumentParser(description="Predice SalePrice sobre un dataset nuevo.")
    p.add_argument("--input", "-i", required=True, type=Path, help="CSV de entrada")
    p.add_argument("--output", "-o", type=Path,
                   default=ROOT / "submissions" / "predicciones.csv")
    p.add_argument("--models", type=Path, default=MODELS, help="directorio con los artefactos")
    args = p.parse_args()

    if not args.input.exists():
        print(f"ERROR: no existe {args.input}", file=sys.stderr)
        return 1
    if not (args.models / "metadata.json").exists():
        print(f"ERROR: faltan artefactos en {args.models}. Ejecutá el notebook 05 primero.",
              file=sys.stderr)
        return 1

    salida = predecir(args.input, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.output, index=False)

    print(f"Predicciones : {len(salida)} filas → {args.output}")
    print(f"Modelos      : {json.loads((args.models / 'metadata.json').read_text())['n_modelos']} (promediados)")
    print(f"Rango        : {salida[TARGET].min():,.0f} – {salida[TARGET].max():,.0f} USD")
    print(f"Media        : {salida[TARGET].mean():,.0f} USD")
    if "rmse" in salida.attrs:
        print(f"\nRMSE         : {salida.attrs['rmse']:,.2f} USD")
    else:
        print("\n(el CSV no trae SalePrice, así que no se puede calcular el RMSE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
