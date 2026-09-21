#!/usr/bin/env python3
"""Comparación formal de los campeones del Proyecto 2.

Evalúa todos los modelos con exactamente el mismo ALE/SpaceInvaders-v5,
preprocesamiento, seeds y recompensas reales. No recorta recompensas y no usa
EpisodicLifeEnv durante la evaluación.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from sb3_contrib import QRDQN

from train_rainbow_local import Config, RainbowNetwork


gym.register_envs(ale_py)


def crear_entorno(seed: int):
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        frameskip=1,
        repeat_action_probability=0.25,
        full_action_space=False,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.action_space.seed(seed)
    return env


def cargar_rainbow(ruta: Path, device: torch.device):
    checkpoint = torch.load(ruta, map_location=device, weights_only=False)
    config = Config(**checkpoint["config"])
    modelo = RainbowNetwork(actions=6, config=config).to(device)
    modelo.load_state_dict(checkpoint["online"])
    modelo.eval()
    return modelo, int(checkpoint["global_step"])


def accion_rainbow(modelo, observacion, device):
    tensor = torch.as_tensor(
        np.asarray(observacion), device=device
    ).unsqueeze(0)
    with torch.no_grad():
        return int(modelo(tensor).argmax(dim=1).item())


def accion_qrdqn(modelo, observacion, _device):
    accion, _ = modelo.predict(observacion, deterministic=True)
    return int(np.asarray(accion).item())


def evaluar_modelo(
    nombre,
    tipo,
    ruta,
    episodios,
    seed_inicial,
    device,
):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {nombre}: {ruta}")

    if tipo == "rainbow":
        modelo, pasos_modelo = cargar_rainbow(ruta, device)
        elegir_accion = accion_rainbow
    else:
        modelo = QRDQN.load(ruta, device=device)
        pasos_modelo = int(modelo.num_timesteps)
        elegir_accion = accion_qrdqn

    env = crear_entorno(seed_inicial)
    resultados = []
    print(f"\nEvaluando {nombre} ({pasos_modelo:,} pasos)...", flush=True)
    for indice in range(episodios):
        seed = seed_inicial + indice
        observacion, _ = env.reset(seed=seed)
        recompensa_total = 0.0
        pasos = 0
        terminado = False
        while not terminado and pasos < 30_000:
            accion = elegir_accion(modelo, observacion, device)
            observacion, recompensa, terminated, truncated, _ = env.step(
                accion
            )
            recompensa_total += float(recompensa)
            pasos += 1
            terminado = bool(terminated or truncated)

        resultado = {
            "modelo": nombre,
            "tipo": tipo,
            "timesteps_modelo": pasos_modelo,
            "episodio": indice + 1,
            "seed": seed,
            "pasos": pasos,
            "recompensa": recompensa_total,
        }
        resultados.append(resultado)
        print(
            f"{nombre} | episodio {indice + 1:02d}/{episodios} | "
            f"seed={seed} | recompensa={recompensa_total:.1f} | pasos={pasos}",
            flush=True,
        )

    env.close()
    recompensas = np.asarray(
        [resultado["recompensa"] for resultado in resultados], dtype=float
    )
    desviacion = float(recompensas.std())
    resumen = {
        "modelo": nombre,
        "tipo": tipo,
        "ruta": str(ruta),
        "timesteps_modelo": pasos_modelo,
        "episodios": episodios,
        "seed_inicial": seed_inicial,
        "seed_final": seed_inicial + episodios - 1,
        "recompensa_acumulada": float(recompensas.sum()),
        "promedio": float(recompensas.mean()),
        "mediana": float(np.median(recompensas)),
        "maximo": float(recompensas.max()),
        "minimo": float(recompensas.min()),
        "desviacion": desviacion,
        "intervalo_95": float(1.96 * desviacion / np.sqrt(episodios)),
    }
    return resultados, resumen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    if args.episodes < 5:
        raise ValueError("La evaluación debe usar al menos 5 episodios.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    raiz = Path(__file__).resolve().parents[1]
    salida = raiz / "results" / f"comparacion_finalistas_{args.episodes}ep"
    salida.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    candidatos = [
        (
            "Rainbow_campeon_10750k",
            "rainbow",
            raiz
            / "models"
            / "competencia"
            / "SpaceInvaders_Rainbow_campeon.pt",
        ),
    ]

    todos_episodios = []
    resumenes = []
    for nombre, tipo, ruta in candidatos:
        episodios, resumen = evaluar_modelo(
            nombre,
            tipo,
            ruta,
            args.episodes,
            args.seed,
            device,
        )
        todos_episodios.extend(episodios)
        resumenes.append(resumen)

    ruta_episodios = salida / "evaluacion_episodios.csv"
    with open(ruta_episodios, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=todos_episodios[0].keys())
        escritor.writeheader()
        escritor.writerows(todos_episodios)

    resumenes.sort(key=lambda item: item["promedio"], reverse=True)
    ruta_resumen = salida / "evaluacion_resumen.json"
    with open(ruta_resumen, "w", encoding="utf-8") as archivo:
        json.dump(resumenes, archivo, indent=2)

    print("\nCOMPARACIÓN FINAL", flush=True)
    for posicion, resumen in enumerate(resumenes, start=1):
        print(
            f"{posicion}. {resumen['modelo']} | "
            f"promedio={resumen['promedio']:.1f} ± {resumen['intervalo_95']:.1f} | "
            f"acumulada={resumen['recompensa_acumulada']:.1f} | "
            f"máximo={resumen['maximo']:.1f} | mínimo={resumen['minimo']:.1f}",
            flush=True,
        )
    print("\nArchivos:", flush=True)
    print(ruta_episodios, flush=True)
    print(ruta_resumen, flush=True)


if __name__ == "__main__":
    main()
