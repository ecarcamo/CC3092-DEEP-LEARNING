# Laboratorio 5 — Agentes en el Arcade Learning Environment (ALE): Space Invaders

**CC3092 · Deep Learning y Sistemas Inteligentes** — Entrega: lunes 14 de septiembre de 2026.

Infraestructura para que un agente que **no aprende** (aleatorio o de regla simple)
interactúe con un entorno de Atari 2600 a través de ALE, y para **grabar en video** sus
partidas. En este laboratorio no se entrena ningún agente: el objetivo es dejar
funcionando el código base que usarán los laboratorios y proyectos posteriores.

## Estructura

```
lab5/
├── docs/                          # enunciado del laboratorio (PDF)
├── informe/
│   ├── investigacion.md           # investigación completa y extendida [FUENTE]
│   ├── informe_lab5.html          # maquetación del informe
│   └── informe_lab5.pdf           # entregable: informe de 3 páginas
├── notebooks/
│   └── lab5_ale_space_invaders.ipynb   # entregable: módulo en uso + métricas
├── src/
│   └── ale_utils.py               # entregable: módulo de funciones reutilizables
├── videos/                        # entregable: .mp4 de los episodios grabados
├── results/
│   ├── summary.json               # métricas de todas las ejecuciones
│   └── figures/                   # gráficas generadas por el notebook
├── requirements.txt
└── .gitignore
```

## Entorno

Entorno virtual propio del laboratorio, en `lab5/.venv` (Python 3.11):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

Versiones instaladas y verificadas: `gymnasium 1.3.0`, `ale-py 0.12.1`, `moviepy 2.2.1`,
`opencv-python 5.0.0`, `numpy 2.4.6`.

Dos detalles que suelen romper este montaje:

- **Los ROMs ya vienen incluidos** desde `ale-py 0.9`. No hace falta `AutoROM`.
- Hay que **registrar los entornos** explícitamente, o `gym.make("ALE/SpaceInvaders-v5")`
  falla con `NameNotFound`:

  ```python
  import gymnasium as gym
  import ale_py
  gym.register_envs(ale_py)
  ```

- Para grabar video, el entorno debe crearse con `render_mode="rgb_array"` y hay que
  llamar a `env.close()` al terminar, o el `.mp4` no se escribe a disco.

## Verificación rápida

```bash
.venv/bin/python -c "
import gymnasium as gym, ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
print(env.observation_space, env.action_space)
print(env.unwrapped.get_action_meanings())
env.close()"
```

Salida esperada:

```
Box(0, 255, (210, 160, 3), uint8) Discrete(6)
['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
```

## Reproducir

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab5_ale_space_invaders.ipynb
```

Tarda unos 20 segundos y regenera los videos, las figuras y `results/summary.json`.
Los resultados son **reproducibles**: dos ejecuciones seguidas dan los mismos números.

Para regenerar el PDF del informe:

```bash
cd informe
chromium --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=informe_lab5.pdf informe_lab5.html
```

## El módulo

`src/ale_utils.py` expone cinco funciones:

| Función | Qué hace |
|---|---|
| `crear_entorno` | Crea el entorno y lo envuelve en `RecordVideo` si se le pasa `video_folder`. Genérica: reenvía `**kwargs` a `gym.make`, así que sirve para cualquier entorno registrado. |
| `agente_aleatorio` | Línea base: `env.action_space.sample()`. |
| `agente_regla_simple` | Política fija sin aprendizaje: localiza cañón e invasores por color, descarta las columnas tapadas por un búnker y dispara mientras se mueve hacia el objetivo. |
| `ejecutar_episodio` | Corre un episodio hasta `terminated`/`truncated` o `max_steps` y devuelve `(pasos, retorno)`. |
| `generar_video_agente` | Combina las anteriores, cierra el entorno y devuelve rutas de video más métricas. |

## Resultados

Diez episodios por agente en `ALE/SpaceInvaders-v5`, con las mismas semillas:

| Agente | Retorno medio | Desv. típica | Pasos medios |
|---|---|---|---|
| `agente_aleatorio` | 123.5 | 73.3 | 461.0 |
| `agente_regla_simple` | **246.0** | **34.8** | 522.1 |
| *(control)* quedarse quieto disparando | 285.0 | — | — |

El agente de regla simple duplica el retorno del aleatorio y gana en las diez semillas. Pero
la política degenerada de no moverse y disparar siempre puntúa aún más alto: buena parte del
mérito no está en apuntar, sino en no dejar de disparar. Moverse es caro porque, con
`frameskip=4`, **los proyectiles enemigos no aparecen en la observación** (medido en el
notebook: visibles en 346 de 1 200 cuadros con `frameskip=1`, en 0 con `frameskip=4`), así
que el agente se mueve a ciegas. Es la demostración práctica de por qué `AtariPreprocessing`
hace *max-pooling* de los dos últimos cuadros.

### Figuras

- `results/figures/observacion_preprocesada.png` — imagen cruda frente a los 4 cuadros
  apilados de 84×84 que recibiría la red.
- `results/figures/comparacion_agentes.png` — retorno por semilla y distribución de ambos agentes.

## Entregables

| Entregable | Ruta |
|---|---|
| PDF (3 páginas) | `informe/informe_lab5.pdf` |
| Notebook | `notebooks/lab5_ale_space_invaders.ipynb` |
| Módulo de funciones | `src/ale_utils.py` |
| Video del agente aleatorio | `videos/aleatorio_spaceinvaders-episode-0.mp4` |
| Video del agente de regla simple | `videos/regla_simple_spaceinvaders-episode-0.mp4` |
