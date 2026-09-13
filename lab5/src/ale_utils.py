"""
Módulo de funciones reutilizables para interactuar con el Arcade Learning Environment.

Laboratorio 5 - CC3092 Deep Learning y Sistemas Inteligentes.

Este módulo concentra la infraestructura pedida en la sección 3 del enunciado: crear
entornos (con grabación de video opcional), ejecutar agentes que no aprenden y producir
los videos de las partidas. No entrena nada: es la base sobre la que se montarán los
agentes entrenados de laboratorios posteriores.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

# ale-py registra los entornos ALE/* en Gymnasium. Sin esta llamada,
# gym.make("ALE/SpaceInvaders-v5") falla con NameNotFound.
import ale_py

gym.register_envs(ale_py)


# Firma que debe cumplir cualquier agente de este módulo: recibe la observación
# actual y el entorno, y devuelve una acción válida del espacio de acción.
FuncionAgente = Callable[[Any, gym.Env], Any]


# --------------------------------------------------------------------------------------
# Creación de entornos
# --------------------------------------------------------------------------------------

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
    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        if episode_trigger is None:
            # Por defecto se graban todos los episodios.
            def episode_trigger(episodio: int) -> bool:  # noqa: F811
                return True

        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=episode_trigger,
            disable_logger=True,
        )

    return env


# --------------------------------------------------------------------------------------
# Agentes que no aprenden
# --------------------------------------------------------------------------------------

def agente_aleatorio(observation: Any, env: gym.Env) -> Any:
    """Agente de referencia (*baseline*): ignora la observación y actúa al azar.

    Args:
        observation: observación actual (no se usa; está en la firma para que todos
            los agentes del módulo sean intercambiables).
        env: entorno del que se muestrea la acción.

    Returns:
        Una acción muestreada de ``env.action_space.sample()``.
    """
    return env.action_space.sample()


# Colores RGB de los sprites de Space Invaders, medidos sobre la observación real.
_COLOR_JUGADOR = (50, 132, 50)      # cañón láser, en la parte baja de la pantalla
_COLOR_ALIEN = (134, 134, 29)       # formación de invasores
_COLOR_BUNKER = (181, 83, 40)       # búnkeres destructibles

# El color del cañón aparece también en el marcador (y~10) y en el indicador de vidas
# (y~196-200), así que cada elemento se busca solo en su banda de filas.
_BANDA_JUGADOR = slice(185, 195)
_BANDA_ALIENS = slice(20, 150)
_BANDA_BUNKERES = slice(150, 185)

# Media anchura aproximada de un invasor: por debajo de esta desalineación el disparo
# ya impacta, así que no merece la pena seguir moviéndose.
_TOLERANCIA_PX = 4

# El Atari 2600 dibuja varios sprites en cuadros alternos, así que el cañón desaparece
# de la observación en aproximadamente uno de cada dos pasos. Se recuerda su última
# posición conocida para no quedarse ciego en esos cuadros.
_ULTIMA_X_JUGADOR: dict[str, float | None] = {"x": None}


def _columnas_con_color(imagen: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Índices de las columnas que contienen al menos un píxel del color dado."""
    mascara = np.all(imagen == np.array(color, dtype=imagen.dtype), axis=-1)
    return np.flatnonzero(mascara.any(axis=0))


def agente_regla_simple(observation: Any, env: gym.Env) -> Any:
    """Agente heurístico no entrenado para Space Invaders.

    Política fija, sin aprendizaje, con tres reglas:

    1. Localiza el cañón y la formación de invasores por su color en la imagen.
    2. Descarta los invasores cuya columna está tapada por un búnker: disparar ahí
       destruye la propia cobertura en lugar de sumar puntos.
    3. Se desplaza hacia el invasor alcanzable más cercano y dispara mientras se mueve
       (RIGHTFIRE / LEFTFIRE), aprovechando que el joystick del 2600 permite mover y
       disparar en la misma acción.

    No esquiva los proyectiles enemigos, y no por simplificar: con el ``frameskip=4``
    por defecto de la versión v5, los proyectiles se dibujan en cuadros alternos y
    **no aparecen nunca** en la observación que recibe el agente. Esquivarlos exigiría
    el *max-pooling* de los dos últimos cuadros que aplica ``AtariPreprocessing``.

    Solo funciona con la observación RGB por defecto (210x160x3). Con cualquier otra
    (RAM, escala de grises, imagen preprocesada) recae en una acción aleatoria.

    Args:
        observation: observación actual del entorno.
        env: entorno (para consultar el espacio y los nombres de las acciones).

    Returns:
        Una acción válida del espacio de acción.
    """
    acciones = {
        nombre: indice
        for indice, nombre in enumerate(env.unwrapped.get_action_meanings())
    }
    # Si el entorno no ofrece las acciones combinadas, se usan las simples.
    disparar = acciones.get("FIRE", 0)
    derecha = acciones.get("RIGHTFIRE", acciones.get("RIGHT", disparar))
    izquierda = acciones.get("LEFTFIRE", acciones.get("LEFT", disparar))

    obs = np.asarray(observation)
    if obs.ndim != 3 or obs.shape[-1] != 3:
        return agente_aleatorio(observation, env)

    columnas_jugador = _columnas_con_color(obs[_BANDA_JUGADOR], _COLOR_JUGADOR)
    if len(columnas_jugador):
        x_jugador = float(columnas_jugador.mean())
        _ULTIMA_X_JUGADOR["x"] = x_jugador
    else:
        # Cuadro en el que el cañón no se dibujó: se usa su última posición conocida.
        x_jugador = _ULTIMA_X_JUGADOR["x"]
        if x_jugador is None:
            return disparar

    columnas_alien = _columnas_con_color(obs[_BANDA_ALIENS], _COLOR_ALIEN)
    if not len(columnas_alien):
        return disparar

    # Las columnas tapadas por un búnker no son objetivos válidos.
    columnas_bunker = set(
        _columnas_con_color(obs[_BANDA_BUNKERES], _COLOR_BUNKER).tolist()
    )
    columnas_libres = np.array(
        [c for c in columnas_alien if c not in columnas_bunker], dtype=int
    )
    if len(columnas_libres):
        columnas_alien = columnas_libres

    x_objetivo = float(
        columnas_alien[np.argmin(np.abs(columnas_alien - x_jugador))]
    )

    if abs(x_objetivo - x_jugador) <= _TOLERANCIA_PX:
        return disparar
    return derecha if x_objetivo > x_jugador else izquierda


# --------------------------------------------------------------------------------------
# Ejecución de episodios
# --------------------------------------------------------------------------------------

def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: FuncionAgente,
    max_steps: int = 10000,
    seed: int | None = None,
) -> tuple[int, float]:
    """Ejecuta un episodio completo con la función de agente indicada.

    Corre hasta que ``terminated`` o ``truncated`` sean verdaderos, o hasta alcanzar
    ``max_steps``.

    Args:
        env: entorno ya creado (posiblemente con grabación activada).
        funcion_agente: función ``(observation, env) -> action``.
        max_steps: tope de pasos del episodio.
        seed: semilla opcional para ``env.reset``, para poder reproducir la partida.

    Returns:
        ``(pasos, recompensa_total)``: número de pasos ejecutados y retorno acumulado.
    """
    observation, _ = env.reset(seed=seed)

    pasos = 0
    recompensa_total = 0.0
    terminated = truncated = False

    while not (terminated or truncated) and pasos < max_steps:
        accion = funcion_agente(observation, env)
        observation, recompensa, terminated, truncated, _ = env.step(accion)
        recompensa_total += float(recompensa)
        pasos += 1

    return pasos, recompensa_total


# --------------------------------------------------------------------------------------
# Generación de videos
# --------------------------------------------------------------------------------------

def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: FuncionAgente,
    video_folder: str,
    name_prefix: str,
    n_episodios: int = 1,
    max_steps: int = 10000,
    seed: int | None = None,
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
        seed: semilla del primer episodio; los siguientes usan seed+1, seed+2, ...
        **kwargs: argumentos extra para ``crear_entorno``.

    Returns:
        ``(rutas_videos, metricas)`` donde ``metricas`` es una lista de diccionarios con
        los pasos y la recompensa total de cada episodio.
    """
    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        name_prefix=name_prefix,
        **kwargs,
    )

    metricas: list[dict[str, Any]] = []
    try:
        for episodio in range(n_episodios):
            semilla = None if seed is None else seed + episodio
            pasos, recompensa_total = ejecutar_episodio(
                env, funcion_agente, max_steps=max_steps, seed=semilla
            )
            metricas.append(
                {
                    "episodio": episodio,
                    "entorno": nombre_entorno,
                    "agente": getattr(funcion_agente, "__name__", str(funcion_agente)),
                    "semilla": semilla,
                    "pasos": pasos,
                    "recompensa_total": recompensa_total,
                }
            )
    finally:
        # Sin este close() el .mp4 se queda a medias y no llega a escribirse a disco.
        env.close()

    # RecordVideo nombra los archivos "{name_prefix}-episode-{n}.mp4". Derivar la ruta
    # del índice de episodio (en vez de listar la carpeta) mantiene el emparejamiento
    # correcto cuando episode_trigger graba solo algunos episodios, y evita recoger
    # videos de ejecuciones anteriores que sigan en la carpeta.
    rutas_videos: list[str] = []
    for metrica in metricas:
        ruta = os.path.join(
            video_folder, f"{name_prefix}-episode-{metrica['episodio']}.mp4"
        )
        if os.path.exists(ruta):
            metrica["video"] = ruta
            rutas_videos.append(ruta)
        else:
            metrica["video"] = None

    return rutas_videos, metricas
