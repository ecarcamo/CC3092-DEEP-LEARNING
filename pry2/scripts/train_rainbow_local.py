#!/usr/bin/env python3
"""Entrenamiento Rainbow DQN para ALE/SpaceInvaders-v5.

Adaptación para el Proyecto 2 basada en:
- Hessel et al. (2018), Rainbow: Combining Improvements in Deep RL.
- La implementación de referencia rainbow_atari.py de CleanRL.

La evaluación usa el entorno canónico del proyecto y recompensas reales.
Durante el aprendizaje se recortan las recompensas y se usa EpisodicLifeEnv,
dos prácticas estándar de entrenamiento Atari que no alteran la evaluación.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter


gym.register_envs(ale_py)


@dataclass
class Config:
    env_id: str = "ALE/SpaceInvaders-v5"
    seed: int = 42
    max_steps: int = 30_000_000
    max_hours: float = 12.0
    learning_rate: float = 6.25e-5
    buffer_size: int = 100_000
    learning_starts: int = 80_000
    batch_size: int = 32
    gamma: float = 0.99
    train_frequency: int = 4
    target_update_frequency: int = 8_000
    n_step: int = 3
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_epsilon: float = 1e-6
    noisy_sigma: float = 0.5
    checkpoint_frequency: int = 250_000
    evaluation_frequency: int = 250_000
    evaluation_episodes: int = 5
    repeat_action_probability: float = 0.25
    base_frameskip: int = 1
    full_action_space: bool = False
    noop_max: int = 30
    frame_skip: int = 4
    screen_size: int = 84
    grayscale_obs: bool = True
    scale_obs: bool = False
    terminal_on_life_loss: bool = False
    stack_size: int = 4
    training_episodic_life: bool = True
    training_reward_clipping: bool = True
    evaluation_raw_reward: bool = True
    experiment: str = "Rainbow_01_full_12h"


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


def crear_entorno_base(config: Config, seed: int, monitor_path: str | None):
    env = gym.make(
        config.env_id,
        frameskip=config.base_frameskip,
        repeat_action_probability=config.repeat_action_probability,
        full_action_space=config.full_action_space,
    )
    env = AtariPreprocessing(
        env,
        noop_max=config.noop_max,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=config.grayscale_obs,
        scale_obs=config.scale_obs,
    )
    env = Monitor(env, filename=monitor_path)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def crear_entorno_entrenamiento(config: Config, monitor_path: str):
    env = crear_entorno_base(config, config.seed, monitor_path)
    # Monitor queda dentro de EpisodicLife para registrar partidas completas.
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = FrameStackObservation(env, stack_size=config.stack_size)
    return env


def crear_entorno_evaluacion(config: Config):
    env = crear_entorno_base(config, 10_000 + config.seed, None)
    env = FrameStackObservation(env, stack_size=config.stack_size)
    return env


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma / math.sqrt(in_features))
        self.bias_sigma.data.fill_(sigma / math.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, device: torch.device):
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scaled_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, value):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(value, weight, bias)


class RainbowNetwork(nn.Module):
    def __init__(self, actions: int, config: Config):
        super().__init__()
        self.actions = actions
        self.n_atoms = config.n_atoms
        self.register_buffer(
            "support",
            torch.linspace(config.v_min, config.v_max, config.n_atoms),
        )
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_hidden = NoisyLinear(3136, 512, config.noisy_sigma)
        self.value_output = NoisyLinear(512, config.n_atoms, config.noisy_sigma)
        self.adv_hidden = NoisyLinear(3136, 512, config.noisy_sigma)
        self.adv_output = NoisyLinear(
            512, actions * config.n_atoms, config.noisy_sigma
        )

    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_output.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_output.reset_noise()

    def distribution(self, observations):
        observations = observations.float() / 255.0
        features = self.features(observations)
        value = F.relu(self.value_hidden(features))
        value = self.value_output(value).view(-1, 1, self.n_atoms)
        advantage = F.relu(self.adv_hidden(features))
        advantage = self.adv_output(advantage).view(
            -1, self.actions, self.n_atoms
        )
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1).clamp(min=1e-5)

    def forward(self, observations):
        distribution = self.distribution(observations)
        return (distribution * self.support).sum(dim=-1)


class SegmentTrees:
    def __init__(self, capacity: int):
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.capacity = capacity
        self.tree_capacity = tree_capacity
        self.sum_tree = np.zeros(2 * tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * tree_capacity, np.inf, dtype=np.float64)

    def update(self, index: int, value: float):
        position = index + self.tree_capacity
        self.sum_tree[position] = value
        self.min_tree[position] = value
        position //= 2
        while position:
            self.sum_tree[position] = (
                self.sum_tree[2 * position] + self.sum_tree[2 * position + 1]
            )
            self.min_tree[position] = min(
                self.min_tree[2 * position], self.min_tree[2 * position + 1]
            )
            position //= 2

    def sample(self, mass: float):
        position = 1
        while position < self.tree_capacity:
            left = 2 * position
            if mass <= self.sum_tree[left]:
                position = left
            else:
                mass -= self.sum_tree[left]
                position = left + 1
        return position - self.tree_capacity


class PrioritizedNStepReplay:
    def __init__(self, observation_shape, config: Config):
        capacity = config.buffer_size
        self.capacity = capacity
        self.alpha = config.per_alpha
        self.epsilon = config.per_epsilon
        self.gamma = config.gamma
        self.n_step = config.n_step
        self.observations = np.empty(
            (capacity, *observation_shape), dtype=np.uint8
        )
        self.next_observations = np.empty(
            (capacity, *observation_shape), dtype=np.uint8
        )
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.trees = SegmentTrees(capacity)
        self.n_step_buffer = deque()
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def _store_first(self):
        reward = 0.0
        next_observation = self.n_step_buffer[0][3]
        done = False
        steps = 0
        for _, _, item_reward, item_next, item_done in list(
            self.n_step_buffer
        )[: self.n_step]:
            reward += (self.gamma**steps) * item_reward
            next_observation = item_next
            done = bool(item_done)
            steps += 1
            if done:
                break

        observation, action = self.n_step_buffer[0][:2]
        index = self.position
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.discounts[index] = self.gamma**steps
        self.trees.update(index, self.max_priority**self.alpha)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.n_step_buffer.popleft()

    def add(self, observation, action, reward, next_observation, done):
        self.n_step_buffer.append(
            (
                np.asarray(observation, dtype=np.uint8).copy(),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.uint8).copy(),
                bool(done),
            )
        )
        if len(self.n_step_buffer) >= self.n_step:
            self._store_first()
        if done:
            while self.n_step_buffer:
                self._store_first()

    def sample(self, batch_size: int, beta: float, device: torch.device):
        total = self.trees.sum_tree[1]
        segment = total / batch_size
        indices = []
        for batch_index in range(batch_size):
            low = segment * batch_index
            high = segment * (batch_index + 1)
            index = self.trees.sample(random.uniform(low, high))
            indices.append(index)
        indices = np.asarray(indices, dtype=np.int64)
        priorities = self.trees.sum_tree[
            indices + self.trees.tree_capacity
        ]
        probabilities = priorities / total
        minimum_probability = self.trees.min_tree[1] / total
        maximum_weight = (self.size * minimum_probability) ** (-beta)
        weights = (self.size * probabilities) ** (-beta) / maximum_weight

        return {
            "observations": torch.as_tensor(
                self.observations[indices], device=device
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            "actions": torch.as_tensor(
                self.actions[indices], device=device
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices], device=device
            ),
            "dones": torch.as_tensor(
                self.dones[indices], device=device, dtype=torch.float32
            ),
            "discounts": torch.as_tensor(
                self.discounts[indices], device=device
            ),
            "weights": torch.as_tensor(
                weights, device=device, dtype=torch.float32
            ),
            "indices": indices,
        }

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            priority = max(float(priority), self.epsilon)
            self.max_priority = max(self.max_priority, priority)
            self.trees.update(int(index), priority**self.alpha)


def actualizar_modelo(
    online: RainbowNetwork,
    target: RainbowNetwork,
    optimizer,
    replay: PrioritizedNStepReplay,
    beta: float,
    config: Config,
    device: torch.device,
):
    batch = replay.sample(config.batch_size, beta, device)
    online.reset_noise()
    target.reset_noise()

    with torch.no_grad():
        next_actions = online(batch["next_observations"]).argmax(dim=1)
        next_distribution = target.distribution(batch["next_observations"])
        next_distribution = next_distribution[
            torch.arange(config.batch_size, device=device), next_actions
        ]
        support = target.support.unsqueeze(0)
        projected_support = batch["rewards"].unsqueeze(1) + (
            1.0 - batch["dones"].unsqueeze(1)
        ) * batch["discounts"].unsqueeze(1) * support
        projected_support.clamp_(config.v_min, config.v_max)
        delta = (config.v_max - config.v_min) / (config.n_atoms - 1)
        atom_positions = (projected_support - config.v_min) / delta
        lower = atom_positions.floor().long()
        upper = atom_positions.ceil().long()
        lower_weight = upper.float() - atom_positions
        upper_weight = atom_positions - lower.float()
        equal = upper == lower
        lower_weight[equal] = 1.0
        upper_weight[equal] = 0.0
        projection = torch.zeros_like(next_distribution)
        offset = (
            torch.arange(config.batch_size, device=device) * config.n_atoms
        ).unsqueeze(1)
        projection.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_distribution * lower_weight).view(-1),
        )
        projection.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_distribution * upper_weight).view(-1),
        )

    current_distribution = online.distribution(batch["observations"])
    current_distribution = current_distribution[
        torch.arange(config.batch_size, device=device), batch["actions"]
    ]
    losses = -(projection * current_distribution.log()).sum(dim=1)
    loss = (batch["weights"] * losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(
        batch["indices"], losses.detach().cpu().numpy() + config.per_epsilon
    )
    return float(loss.item())


def evaluar(modelo, env, config: Config, device: torch.device):
    resultados = []
    modelo.eval()
    for episodio in range(config.evaluation_episodes):
        observacion, _ = env.reset(seed=100 + episodio)
        recompensa_total = 0.0
        pasos = 0
        terminado = False
        while not terminado and pasos < 30_000:
            tensor = torch.as_tensor(
                np.asarray(observacion), device=device
            ).unsqueeze(0)
            with torch.no_grad():
                accion = int(modelo(tensor).argmax(dim=1).item())
            observacion, recompensa, terminated, truncated, _ = env.step(
                accion
            )
            recompensa_total += float(recompensa)
            pasos += 1
            terminado = terminated or truncated
        resultados.append(
            {
                "episodio": episodio + 1,
                "seed": 100 + episodio,
                "recompensa": recompensa_total,
                "pasos": pasos,
            }
        )
    modelo.train()
    recompensas = np.asarray(
        [resultado["recompensa"] for resultado in resultados], dtype=float
    )
    resumen = {
        "promedio": float(recompensas.mean()),
        "maximo": float(recompensas.max()),
        "minimo": float(recompensas.min()),
        "desviacion": float(recompensas.std()),
    }
    return resultados, resumen


def guardar_checkpoint(ruta, online, target, optimizer, paso, config, mejor):
    torch.save(
        {
            "global_step": int(paso),
            "online": online.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mean_reward": float(mejor),
            "config": asdict(config),
        },
        ruta,
    )


def guardar_evaluaciones(ruta_json, ruta_csv, historial):
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(historial, archivo, indent=2)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        campos = [
            "timesteps",
            "promedio",
            "maximo",
            "minimo",
            "desviacion",
            "horas_transcurridas",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for evaluacion in historial:
            escritor.writerow(
                {campo: evaluacion[campo] for campo in campos}
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="Rainbow_01_full_12h")
    parser.add_argument("--max-steps", type=int, default=30_000_000)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        experiment=args.experiment,
        max_steps=args.max_steps,
        max_hours=args.hours,
        seed=args.seed,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; no se iniciará el entrenamiento.")

    raiz = Path(__file__).resolve().parents[1]
    carpeta_checkpoints = raiz / "checkpoints" / config.experiment
    carpeta_modelos = raiz / "models" / config.experiment
    carpeta_resultados = raiz / "results" / config.experiment
    carpeta_tensorboard = raiz / "logs" / "tensorboard" / config.experiment
    for carpeta in (
        carpeta_checkpoints,
        carpeta_modelos,
        carpeta_resultados,
        carpeta_tensorboard,
        raiz / "logs",
    ):
        carpeta.mkdir(parents=True, exist_ok=True)

    with open(
        carpeta_resultados / "configuracion_experimento.json",
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(asdict(config), archivo, indent=2)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    env = crear_entorno_entrenamiento(
        config, str(raiz / "logs" / f"{config.experiment}_monitor")
    )
    eval_env = crear_entorno_evaluacion(config)
    observacion, _ = env.reset(seed=config.seed)
    if tuple(observacion.shape) != (4, 84, 84):
        raise RuntimeError(f"Observación inesperada: {observacion.shape}")

    online = RainbowNetwork(env.action_space.n, config).to(device)
    target = RainbowNetwork(env.action_space.n, config).to(device)
    target.load_state_dict(online.state_dict())
    target.train()
    optimizer = torch.optim.Adam(
        online.parameters(), lr=config.learning_rate, eps=1.5e-4
    )
    replay = PrioritizedNStepReplay(observacion.shape, config)
    writer = SummaryWriter(str(carpeta_tensorboard))

    detener = False

    def solicitar_detencion(signum, frame):
        nonlocal detener
        detener = True
        print(f"\nSeñal {signum} recibida; guardando antes de salir...", flush=True)

    signal.signal(signal.SIGTERM, solicitar_detencion)
    signal.signal(signal.SIGINT, solicitar_detencion)

    historial = []
    mejor_promedio = -float("inf")
    inicio = time.time()
    limite_segundos = config.max_hours * 3600
    proximo_checkpoint = config.checkpoint_frequency
    proxima_evaluacion = config.evaluation_frequency
    proximo_log = 25_000
    paso = 0
    episodios = 0
    recompensa_episodio_recortada = 0.0
    perdida_reciente = float("nan")

    print("\nRainbow preparado", flush=True)
    print("Experimento:", config.experiment, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Observación:", observacion.shape, flush=True)
    print("Acciones:", env.action_space, flush=True)
    print("Máximo:", config.max_steps, "pasos o", config.max_hours, "horas", flush=True)
    print("Replay:", config.buffer_size, "transiciones", flush=True)

    estado_final = "completado"
    try:
        while paso < config.max_steps and not detener:
            transcurrido = time.time() - inicio
            if transcurrido >= limite_segundos:
                print("\nLímite de tiempo alcanzado.", flush=True)
                break

            online.train()
            online.reset_noise()
            tensor = torch.as_tensor(
                np.asarray(observacion), device=device
            ).unsqueeze(0)
            with torch.no_grad():
                accion = int(online(tensor).argmax(dim=1).item())

            siguiente, recompensa, terminated, truncated, info = env.step(accion)
            terminado = bool(terminated or truncated)
            replay.add(observacion, accion, recompensa, siguiente, terminado)
            recompensa_episodio_recortada += float(recompensa)
            observacion = siguiente
            paso += 1

            if terminado:
                episodios += 1
                writer.add_scalar(
                    "train/recompensa_recortada_episodio",
                    recompensa_episodio_recortada,
                    paso,
                )
                observacion, _ = env.reset()
                recompensa_episodio_recortada = 0.0

            if (
                paso >= config.learning_starts
                and paso % config.train_frequency == 0
                and replay.size >= config.batch_size
            ):
                fraccion_tiempo = min(1.0, transcurrido / limite_segundos)
                beta = config.per_beta_start + (
                    1.0 - config.per_beta_start
                ) * fraccion_tiempo
                perdida_reciente = actualizar_modelo(
                    online, target, optimizer, replay, beta, config, device
                )
                writer.add_scalar("train/loss", perdida_reciente, paso)
                writer.add_scalar("train/per_beta", beta, paso)

            if paso % config.target_update_frequency == 0:
                target.load_state_dict(online.state_dict())

            if paso >= proximo_log:
                transcurrido = time.time() - inicio
                velocidad = paso / max(transcurrido, 1e-9)
                restante_tiempo = max(0.0, limite_segundos - transcurrido)
                restante_pasos = max(0, config.max_steps - paso)
                eta = min(restante_tiempo, restante_pasos / max(velocidad, 1e-9))
                print(
                    f"pasos={paso:,} | episodios={episodios:,} | "
                    f"SPS={velocidad:.1f} | loss={perdida_reciente:.4f} | "
                    f"ETA={eta / 3600:.2f} h",
                    flush=True,
                )
                writer.add_scalar("time/SPS", velocidad, paso)
                proximo_log += 25_000

            if paso >= proximo_checkpoint:
                ruta = carpeta_checkpoints / f"rainbow_{paso}_steps.pt"
                guardar_checkpoint(
                    ruta, online, target, optimizer, paso, config, mejor_promedio
                )
                print("Checkpoint:", ruta, flush=True)
                proximo_checkpoint += config.checkpoint_frequency

            if paso >= proxima_evaluacion:
                detalles, resumen = evaluar(online, eval_env, config, device)
                evaluacion = {
                    "timesteps": paso,
                    **resumen,
                    "horas_transcurridas": (time.time() - inicio) / 3600,
                    "episodios": detalles,
                }
                historial.append(evaluacion)
                guardar_evaluaciones(
                    carpeta_resultados / "evaluaciones.json",
                    carpeta_resultados / "evaluaciones.csv",
                    historial,
                )
                writer.add_scalar("eval/recompensa_promedio", resumen["promedio"], paso)
                writer.add_scalar("eval/recompensa_maxima", resumen["maximo"], paso)
                print(
                    f"EVAL pasos={paso:,} | promedio={resumen['promedio']:.1f} | "
                    f"máximo={resumen['maximo']:.1f} | mínimo={resumen['minimo']:.1f}",
                    flush=True,
                )
                if resumen["promedio"] > mejor_promedio:
                    mejor_promedio = resumen["promedio"]
                    guardar_checkpoint(
                        carpeta_modelos / "best_model.pt",
                        online,
                        target,
                        optimizer,
                        paso,
                        config,
                        mejor_promedio,
                    )
                    print("Nuevo mejor modelo.", flush=True)
                proxima_evaluacion += config.evaluation_frequency

    except BaseException:
        estado_final = "interrumpido"
        guardar_checkpoint(
            carpeta_modelos / f"modelo_emergencia_{paso}.pt",
            online,
            target,
            optimizer,
            paso,
            config,
            mejor_promedio,
        )
        raise
    finally:
        guardar_checkpoint(
            carpeta_modelos / "modelo_final.pt",
            online,
            target,
            optimizer,
            paso,
            config,
            mejor_promedio,
        )
        resumen_final = {
            "estado": (
                "detenido_con_seguridad" if detener else estado_final
            ),
            "timesteps": paso,
            "duracion_horas": (time.time() - inicio) / 3600,
            "mejor_promedio": mejor_promedio,
            "fin_utc": datetime.now(timezone.utc).isoformat(),
            "modelo_final": str(carpeta_modelos / "modelo_final.pt"),
            "mejor_modelo": str(carpeta_modelos / "best_model.pt"),
        }
        with open(
            carpeta_resultados / "registro_entrenamiento.json",
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(resumen_final, archivo, indent=2)
        writer.close()
        env.close()
        eval_env.close()

    print("\nEntrenamiento Rainbow terminado", flush=True)
    print("Pasos:", paso, flush=True)
    print("Horas:", round((time.time() - inicio) / 3600, 2), flush=True)
    print("Mejor promedio:", mejor_promedio, flush=True)
    print("Modelo final:", carpeta_modelos / "modelo_final.pt", flush=True)


if __name__ == "__main__":
    main()
