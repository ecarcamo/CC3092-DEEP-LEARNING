"""
Módulo de funciones reutilizables para interactuar con el Arcade Learning Environment.

Laboratorio 5 - CC3092 Deep Learning y Sistemas Inteligentes.

Este módulo concentra la infraestructura pedida en la sección 3 del enunciado: crear
entornos (con grabación de video opcional), ejecutar agentes que no aprenden y producir
los videos de las partidas. No entrena nada: es la base sobre la que se montarán los
agentes entrenados de laboratorios posteriores.

ESTADO: esqueleto. Las firmas y los contratos están fijados; las implementaciones se
escriben en el siguiente paso del laboratorio.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym

# ale-py registra los entornos ALE/* en Gymnasium. Sin esta llamada,
# gym.make("ALE/SpaceInvaders-v5") falla con NameNotFound.
import ale_py

gym.register_envs(ale_py)


# Firma que debe cumplir cualquier agente de este módulo: recibe la observación
# actual y el entorno, y devuelve una acción válida del espacio de acción.
FuncionAgente = Callable[[Any, gym.Env], Any]


def crear_entorno(
    nombre_entorno: str,
    video_folder: str | None = None,
    name_prefix: str = "video",
    episode_trigger: Callable[[int], bool] | None = None,
    render_mode: str = "rgb_array",
    **kwargs: Any,
) -> gym.Env:
    """Crea un entorno de Gymnasium, opcionalmente con grabación de video.

    Es genérica: funciona igual para ALE/SpaceInvaders-v5 que para CartPole-v1 o
    cualquier otro entorno registrado.

    Args:
        nombre_entorno: id del entorno, p. ej. "ALE/SpaceInvaders-v5".
        video_folder: si se indica, el entorno se envuelve en
            ``gymnasium.wrappers.RecordVideo`` y los .mp4 se escriben en esa carpeta.
            Si es None, no se graba nada.
        name_prefix: prefijo de los archivos de video generados.
        episode_trigger: función ``episodio -> bool`` que decide qué episodios se
            graban. Por defecto se graban todos.
        render_mode: debe ser "rgb_array" para poder grabar; con None el video sale vacío.
        **kwargs: argumentos extra para ``gym.make`` (obs_type, frameskip,
            full_action_space, repeat_action_probability, ...).

    Returns:
        El entorno listo para usar. Si se grabó video, hay que llamar a ``env.close()``
        para que el archivo se escriba a disco.
    """
    raise NotImplementedError  # TODO: paso 3 del laboratorio


def agente_aleatorio(observation: Any, env: gym.Env) -> Any:
    """Agente de referencia (*baseline*): ignora la observación y actúa al azar.

    Args:
        observation: observación actual (no se usa; está en la firma para que todos
            los agentes del módulo sean intercambiables).
        env: entorno del que se muestrea la acción.

    Returns:
        Una acción muestreada de ``env.action_space.sample()``.
    """
    raise NotImplementedError  # TODO: paso 3 del laboratorio


def agente_regla_simple(observation: Any, env: gym.Env) -> Any:
    """Agente heurístico no entrenado para Space Invaders.

    Política fija escrita a mano, sin aprendizaje, para comparar contra el agente
    aleatorio. Aparece en la tabla de entregables del enunciado.

    Args:
        observation: observación actual del entorno.
        env: entorno (para consultar el espacio de acción).

    Returns:
        Una acción válida del espacio de acción.
    """
    raise NotImplementedError  # TODO: paso 3 del laboratorio


def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: FuncionAgente,
    max_steps: int = 10000,
) -> tuple[int, float]:
    """Ejecuta un episodio completo con la función de agente indicada.

    Corre hasta que ``terminated`` o ``truncated`` sean verdaderos, o hasta alcanzar
    ``max_steps``.

    Args:
        env: entorno ya creado (posiblemente con grabación activada).
        funcion_agente: función ``(observation, env) -> action``.
        max_steps: tope de pasos del episodio.

    Returns:
        ``(pasos, recompensa_total)``: número de pasos ejecutados y retorno acumulado.
    """
    raise NotImplementedError  # TODO: paso 3 del laboratorio


def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: FuncionAgente,
    video_folder: str,
    name_prefix: str,
    n_episodios: int = 1,
    max_steps: int = 10000,
    **kwargs: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Función de alto nivel: crea el entorno con grabación, juega y guarda los videos.

    Combina las funciones anteriores: crea el entorno con RecordVideo, ejecuta
    ``n_episodios`` episodios completos, cierra el entorno (imprescindible para que los
    .mp4 se escriban a disco) y devuelve las rutas de los videos junto con las métricas.

    Args:
        nombre_entorno: id del entorno.
        funcion_agente: agente a usar (p. ej. ``agente_aleatorio``).
        video_folder: carpeta destino de los videos.
        name_prefix: prefijo de los archivos generados.
        n_episodios: cuántos episodios jugar y grabar.
        max_steps: tope de pasos por episodio.
        **kwargs: argumentos extra para ``crear_entorno``.

    Returns:
        ``(rutas_videos, metricas)`` donde ``metricas`` es una lista de diccionarios con
        los pasos y la recompensa total de cada episodio.
    """
    raise NotImplementedError  # TODO: paso 3 del laboratorio
