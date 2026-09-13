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
│   ├── investigacion.md           # investigación completa (secciones 1 y 2) [FUENTE]
│   └── informe_lab5.pdf           # entregable: informe de máximo 3 páginas
├── notebooks/
│   └── lab5_ale_space_invaders.ipynb   # entregable: módulo en uso + métricas
├── src/
│   └── ale_utils.py               # entregable: módulo de funciones reutilizables
├── videos/                        # entregable: .mp4 de los episodios grabados
├── results/figures/               # gráficas generadas
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

## Entregables

| Entregable | Ruta | Estado |
|---|---|---|
| PDF (máx. 3 páginas) | `informe/informe_lab5.pdf` | pendiente |
| Investigación (fuente) | `informe/investigacion.md` | **listo** |
| Módulo de funciones | `src/ale_utils.py` | esqueleto |
| Notebook | `notebooks/lab5_ale_space_invaders.ipynb` | pendiente |
| Video del agente aleatorio | `videos/` | pendiente |
