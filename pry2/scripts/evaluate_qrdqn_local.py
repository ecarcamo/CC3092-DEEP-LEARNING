import csv
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import QRDQN

from resume_qrdqn_local import (
    RAIZ,
    crear_entorno_base,
)


SEMILLAS = [100, 101, 102, 103, 104]
MAX_STEPS = 30_000

MODELOS = {
    "QRDQN_best_7250k": (
        RAIZ
        / "models"
        / "QRDQN_01"
        / "resume_local"
        / "best_model.zip"
    ),
    "QRDQN_final_8000k": (
        RAIZ
        / "models"
        / "QRDQN_01"
        / "resume_local"
        / "modelo_final_8M.zip"
    ),
}

CARPETA_SALIDA = (
    RAIZ
    / "results"
    / "QRDQN_01"
    / "resume_local"
)

CSV_SALIDA = (
    CARPETA_SALIDA
    / "evaluacion_formal_episodios.csv"
)

JSON_SALIDA = (
    CARPETA_SALIDA
    / "evaluacion_formal_resumen.json"
)


def evaluar_modelo(nombre, ruta):
    print(f"\nEvaluando {nombre}...")

    modelo = QRDQN.load(
        ruta,
        device="cuda",
    )

    env = crear_entorno_base()
    episodios = []

    for numero, seed in enumerate(
        SEMILLAS,
        start=1,
    ):
        observacion, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        recompensa_total = 0.0
        pasos = 0
        terminado = False
        truncado = False

        while (
            not terminado
            and not truncado
            and pasos < MAX_STEPS
        ):
            accion, _ = modelo.predict(
                observacion,
                deterministic=True,
            )

            (
                observacion,
                recompensa,
                terminado,
                truncado,
                _,
            ) = env.step(int(accion))

            recompensa_total += float(recompensa)
            pasos += 1

        episodio = {
            "modelo": nombre,
            "episodio": numero,
            "seed": seed,
            "pasos": pasos,
            "recompensa_total": recompensa_total,
        }
        episodios.append(episodio)

        print(
            f"{nombre} | episodio {numero} | "
            f"seed={seed} | "
            f"recompensa={recompensa_total:.1f} | "
            f"pasos={pasos}"
        )

    env.close()

    recompensas = np.array([
        episodio["recompensa_total"]
        for episodio in episodios
    ])

    resumen = {
        "modelo": nombre,
        "episodios": len(episodios),
        "promedio": float(recompensas.mean()),
        "maximo": float(recompensas.max()),
        "minimo": float(recompensas.min()),
        "desviacion": float(recompensas.std()),
    }

    return episodios, resumen


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    episodios_totales = []
    resumenes = []

    for nombre, ruta in MODELOS.items():
        if not ruta.exists():
            raise FileNotFoundError(ruta)

        episodios, resumen = evaluar_modelo(
            nombre,
            ruta,
        )

        episodios_totales.extend(episodios)
        resumenes.append(resumen)

    with open(
        CSV_SALIDA,
        "w",
        newline="",
        encoding="utf-8",
    ) as archivo:
        campos = [
            "modelo",
            "episodio",
            "seed",
            "pasos",
            "recompensa_total",
        ]

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos,
        )
        escritor.writeheader()
        escritor.writerows(episodios_totales)

    with open(
        JSON_SALIDA,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            resumenes,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print("\nComparación formal:")

    for resumen in resumenes:
        print(
            f"{resumen['modelo']}: "
            f"promedio={resumen['promedio']:.1f}, "
            f"máximo={resumen['maximo']:.1f}, "
            f"mínimo={resumen['minimo']:.1f}, "
            f"desviación={resumen['desviacion']:.1f}"
        )

    print("\nArchivos guardados:")
    print(CSV_SALIDA)
    print(JSON_SALIDA)


if __name__ == "__main__":
    main()