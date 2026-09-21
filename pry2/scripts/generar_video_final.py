#!/usr/bin/env python3
"""Genera el video final del agente Rainbow para el Proyecto 2."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordVideo,
)

from train_rainbow_local import Config, RainbowNetwork


gym.register_envs(ale_py)

RAIZ = Path(__file__).resolve().parents[1]

RUTA_MODELO = (
    RAIZ
    / "models"
    / "competencia"
    / "SpaceInvaders_Rainbow_FINAL.pt"
)

RUTA_EVALUACION = (
    RAIZ
    / "results"
    / "evaluacion_competencia_final_5ep"
    / "evaluacion_episodios.csv"
)

CARPETA_VIDEOS = RAIZ / "videos"
VIDEO_FINAL = CARPETA_VIDEOS / "SpaceInvaders_Rainbow_FINAL.mp4"


def seleccionar_mejor_seed():
    if not RUTA_EVALUACION.exists():
        raise FileNotFoundError(
            f"No existe la evaluación final: {RUTA_EVALUACION}"
        )

    with open(RUTA_EVALUACION, encoding="utf-8") as archivo:
        resultados = list(csv.DictReader(archivo))

    if not resultados:
        raise RuntimeError("La evaluación final no contiene episodios.")

    mejor = max(
        resultados,
        key=lambda fila: float(fila["recompensa"]),
    )

    return int(mejor["seed"]), float(mejor["recompensa"])


def crear_entorno(carpeta_video: Path, seed: int):
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        frameskip=1,
        repeat_action_probability=0.25,
        full_action_space=False,
        render_mode="rgb_array",
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

    env = FrameStackObservation(
        env,
        stack_size=4,
    )

    env = RecordVideo(
        env,
        video_folder=str(carpeta_video),
        episode_trigger=lambda episodio: episodio == 0,
        name_prefix="SpaceInvaders_Rainbow_FINAL",
        fps=15,
        disable_logger=True,
    )

    env.action_space.seed(seed)
    return env


def cargar_modelo(device: torch.device):
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No existe el modelo final: {RUTA_MODELO}"
        )

    checkpoint = torch.load(
        RUTA_MODELO,
        map_location=device,
        weights_only=False,
    )

    config = Config(**checkpoint["config"])

    modelo = RainbowNetwork(
        actions=6,
        config=config,
    ).to(device)

    modelo.load_state_dict(checkpoint["online"])
    modelo.eval()

    return modelo, int(checkpoint["global_step"])


def seleccionar_accion(modelo, observacion, device):
    tensor = torch.as_tensor(
        np.asarray(observacion),
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        return int(modelo(tensor).argmax(dim=1).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla del episodio. Si se omite, usa la mejor de la evaluación.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mejor_seed, recompensa_esperada = seleccionar_mejor_seed()

    seed = (
        args.seed
        if args.seed is not None
        else mejor_seed
    )

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_captura = (
        CARPETA_VIDEOS
        / "capturas_finales"
        / marca
    )
    carpeta_captura.mkdir(parents=True, exist_ok=True)
    CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)

    modelo, pasos_modelo = cargar_modelo(device)
    env = crear_entorno(carpeta_captura, seed)

    observacion, _ = env.reset(seed=seed)
    recompensa_total = 0.0
    pasos = 0
    terminado = False

    while not terminado and pasos < 30_000:
        accion = seleccionar_accion(
            modelo,
            observacion,
            device,
        )

        observacion, recompensa, terminated, truncated, _ = env.step(
            accion
        )

        recompensa_total += float(recompensa)
        pasos += 1
        terminado = bool(terminated or truncated)

    env.close()

    videos_generados = sorted(
        carpeta_captura.glob("*.mp4"),
        key=lambda ruta: ruta.stat().st_mtime,
    )

    if not videos_generados:
        raise RuntimeError("Gymnasium no generó el archivo MP4.")

    video_generado = videos_generados[-1]
    shutil.copy2(video_generado, VIDEO_FINAL)

    registro = {
        "modelo": str(RUTA_MODELO),
        "timesteps_modelo": pasos_modelo,
        "device": str(device),
        "seed": seed,
        "recompensa_total": recompensa_total,
        "recompensa_esperada": (
            recompensa_esperada
            if seed == mejor_seed
            else None
        ),
        "pasos_entorno": pasos,
        "video": str(VIDEO_FINAL),
        "captura_original": str(video_generado),
    }

    ruta_registro = (
        CARPETA_VIDEOS
        / "SpaceInvaders_Rainbow_FINAL.json"
    )

    with open(
        ruta_registro,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(registro, archivo, indent=2)

    print("\nVIDEO FINAL GENERADO")
    print("Modelo:", RUTA_MODELO)
    print("Pasos del modelo:", pasos_modelo)
    print("Dispositivo:", device)
    print("Seed:", seed)
    print("Recompensa:", recompensa_total)
    print("Pasos del episodio:", pasos)
    print("Video:", VIDEO_FINAL)
    print("Registro:", ruta_registro)


if __name__ == "__main__":
    main()