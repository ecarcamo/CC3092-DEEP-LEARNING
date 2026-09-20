import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch

from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor


gym.register_envs(ale_py)

RAIZ = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    RAIZ
    / "checkpoints"
    / "QRDQN_01"
    / "full_8M"
    / "qrdqn_space_invaders_2750000_steps.zip"
)

CARPETA_CHECKPOINTS = (
    RAIZ / "checkpoints" / "QRDQN_01" / "resume_local"
)
CARPETA_MODELOS = (
    RAIZ / "models" / "QRDQN_01" / "resume_local"
)
CARPETA_RESULTADOS = (
    RAIZ / "results" / "QRDQN_01" / "resume_local"
)
CARPETA_TENSORBOARD = (
    RAIZ / "logs" / "tensorboard"
)
RUTA_MONITOR = (
    RAIZ / "logs" / "QRDQN_01_resume_local"
)

TOTAL_TIMESTEPS = 8_000_000
WARMUP_REPLAY = 100_000
FRECUENCIA = 250_000
SEED = 42

for carpeta in [
    CARPETA_CHECKPOINTS,
    CARPETA_MODELOS,
    CARPETA_RESULTADOS,
    CARPETA_TENSORBOARD,
    RAIZ / "logs",
]:
    carpeta.mkdir(parents=True, exist_ok=True)


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


def crear_entorno_base():
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

    env = FrameStackObservation(
        env,
        stack_size=4,
    )

    return env


def crear_entorno_entrenamiento():
    env = crear_entorno_base()
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    # Monitor conserva la recompensa real.
    env = Monitor(
        env,
        filename=str(RUTA_MONITOR),
    )

    # El agente recibe recompensa recortada.
    env = ClipRewardEnv(env)

    return env


def crear_entorno_evaluacion():
    env = crear_entorno_base()
    env = Monitor(env)

    env.reset(seed=1042)
    env.action_space.seed(1042)

    return env


class RegistroProgresoCallback(BaseCallback):
    def __init__(self, ruta, primer_registro):
        super().__init__()
        self.ruta = Path(ruta)
        self.proximo = primer_registro
        self.inicio = None

    def _on_training_start(self):
        self.inicio = time.time()

    def _on_step(self):
        if self.num_timesteps >= self.proximo:
            datos = {
                "estado": "en_progreso",
                "timesteps": int(self.num_timesteps),
                "duracion_local_minutos": (
                    time.time() - self.inicio
                ) / 60,
                "actualizado_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            with open(
                self.ruta,
                "w",
                encoding="utf-8",
            ) as archivo:
                json.dump(datos, archivo, indent=4)

            self.proximo += FRECUENCIA

        return True


def guardar_registro(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Solo valida el checkpoint y la GPU.",
    )
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    entorno_entrenamiento = (
        crear_entorno_entrenamiento()
    )

    modelo = QRDQN.load(
        CHECKPOINT,
        env=entorno_entrenamiento,
        device="cuda",
    )

    pasos_iniciales = int(modelo.num_timesteps)
    pasos_restantes = (
        TOTAL_TIMESTEPS - pasos_iniciales
    )

    observacion, _ = entorno_entrenamiento.reset(
        seed=SEED
    )
    accion, _ = modelo.predict(
        observacion,
        deterministic=True,
    )

    print("\nCheckpoint recuperado")
    print("Ruta:", CHECKPOINT)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Pasos recuperados:", pasos_iniciales)
    print("Pasos restantes:", pasos_restantes)
    print("Observación:", observacion.shape)
    print("Acción de prueba:", int(accion))
    print(
        "Cuantiles:",
        modelo.policy.n_quantiles,
    )

    if args.check_only:
        print("\nValidación completada; no se entrenó.")
        entorno_entrenamiento.close()
        return

    # El replay buffer no estaba incluido en el checkpoint.
    # Se recolectan 100k experiencias antes de volver a
    # actualizar la red.
    modelo.learning_starts = (
        pasos_iniciales + WARMUP_REPLAY
    )

    # Durante los 100k pasos de reconstrucción del replay
    # buffer se aumenta temporalmente la exploración.
    progreso_fin_warmup = 1.0 - (
        (pasos_iniciales + WARMUP_REPLAY)
        / TOTAL_TIMESTEPS
    )


    def exploracion_reanudacion(
        progreso_restante,
    ):
        if progreso_restante > progreso_fin_warmup:
            return 0.10

        return 0.01


    modelo.exploration_schedule = (
        exploracion_reanudacion
    )
    modelo.exploration_rate = 0.10

    modelo.tensorboard_log = str(
        CARPETA_TENSORBOARD
    )

    print(
        "Reconstrucción del replay:",
        WARMUP_REPLAY,
        "pasos con exploración 0.10",
    )
    print(
        "Después continuará con exploración 0.01"
    )

    entorno_evaluacion = (
        crear_entorno_evaluacion()
    )

    callback_checkpoint = CheckpointCallback(
        save_freq=FRECUENCIA,
        save_path=str(CARPETA_CHECKPOINTS),
        name_prefix="qrdqn_resume",
        save_replay_buffer=False,
    )

    callback_evaluacion = EvalCallback(
        entorno_evaluacion,
        best_model_save_path=str(
            CARPETA_MODELOS
        ),
        log_path=str(CARPETA_RESULTADOS),
        eval_freq=FRECUENCIA,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    primer_registro = (
        (pasos_iniciales // FRECUENCIA) + 1
    ) * FRECUENCIA

    ruta_registro = (
        CARPETA_RESULTADOS
        / "registro_entrenamiento.json"
    )

    callback_registro = RegistroProgresoCallback(
        ruta=ruta_registro,
        primer_registro=primer_registro,
    )

    callbacks = CallbackList([
        callback_registro,
        callback_checkpoint,
        callback_evaluacion,
    ])

    inicio = time.time()
    inicio_utc = datetime.now(
        timezone.utc
    ).isoformat()

    guardar_registro(
        ruta_registro,
        {
            "estado": "iniciando_reanudacion",
            "checkpoint": str(CHECKPOINT),
            "timesteps_iniciales": pasos_iniciales,
            "timesteps_objetivo": TOTAL_TIMESTEPS,
            "warmup_replay": WARMUP_REPLAY,
            "inicio_utc": inicio_utc,
        },
    )

    try:
        modelo.learn(
            total_timesteps=pasos_restantes,
            reset_num_timesteps=False,
            callback=callbacks,
            tb_log_name="QRDQN_01_resume_local",
            progress_bar=True,
        )

    except BaseException:
        ruta_emergencia = (
            CARPETA_MODELOS
            / f"modelo_emergencia_{modelo.num_timesteps}"
        )
        modelo.save(str(ruta_emergencia))

        guardar_registro(
            ruta_registro,
            {
                "estado": "interrumpido",
                "timesteps": int(
                    modelo.num_timesteps
                ),
                "modelo_emergencia": str(
                    ruta_emergencia.with_suffix(
                        ".zip"
                    )
                ),
            },
        )
        raise

    duracion_minutos = (
        time.time() - inicio
    ) / 60

    ruta_final = (
        CARPETA_MODELOS / "modelo_final_8M"
    )
    modelo.save(str(ruta_final))

    guardar_registro(
        ruta_registro,
        {
            "estado": "completado",
            "timesteps": int(
                modelo.num_timesteps
            ),
            "inicio_utc": inicio_utc,
            "fin_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "duracion_minutos": duracion_minutos,
            "modelo_final": str(
                ruta_final.with_suffix(".zip")
            ),
        },
    )

    print("\nEntrenamiento local terminado")
    print("Pasos:", modelo.num_timesteps)
    print(
        "Duración:",
        round(duracion_minutos / 60, 2),
        "horas",
    )
    print("Modelo:", ruta_final)

    entorno_entrenamiento.close()
    entorno_evaluacion.close()


if __name__ == "__main__":
    main()