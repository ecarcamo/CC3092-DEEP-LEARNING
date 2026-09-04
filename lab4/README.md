# Laboratorio 4 — Fundamentos de Aprendizaje por Refuerzo y Gymnasium

Investigación de los fundamentos del aprendizaje por refuerzo (RL) y de la librería
**Gymnasium**, más un módulo exploratorio que ejecuta un **agente aleatorio** en dos
entornos (CartPole-v1 y FrozenLake-v1) y una **política simple no aprendida** para
CartPole-v1, comparando su desempeño.

## Estructura

```
lab4/
├── notebooks/
│   └── lab4_rl_gymnasium.ipynb   # entregable: investigación + módulo exploratorio + gráficas
├── results/
│   └── figures/                  # figuras generadas por el notebook (PNG)
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducir

```bash
# venv compartido del curso, en la raíz del repo
../venv/bin/pip install -r requirements.txt
../venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/lab4_rl_gymnasium.ipynb
```

O bien abrir el notebook en Jupyter y ejecutar todas las celdas de arriba a abajo.

## Contenido del notebook

1. **Fundamentos del aprendizaje por refuerzo** — RL vs. aprendizaje supervisado/no
   supervisado, componentes (agente, entorno, estado, acción, recompensa, política),
   MDP, funciones de valor V(s) y Q(s,a), ecuación de Bellman, exploración vs.
   explotación, taxonomías (episódico/continuo, on/off-policy, model-based/model-free)
   y Q-Learning.
2. **La librería Gymnasium** — qué es y su relación con OpenAI Gym, la API de un `Env`
   (`reset`, `step`, `render`, `close`), los espacios (`Discrete`, `Box`,
   `MultiDiscrete`), un catálogo de entornos y los *wrappers*.
3. **Módulo de prueba** — agente aleatorio en CartPole-v1 y FrozenLake-v1, gráficas de
   recompensa por episodio y una política simple para CartPole-v1 comparada contra el
   agente aleatorio.

## Resultados del módulo de prueba

Agente aleatorio y política simple evaluados sobre 20 episodios (gymnasium 1.3.0,
semillas fijas). Retorno = recompensa total acumulada por episodio.

| Entorno | Agente | Retorno medio | Desv. estándar | Nota |
|---|---|---|---|---|
| CartPole-v1 | aleatorio | 19.50 | 9.59 | recompensa densa (+1 por paso) |
| CartPole-v1 | **política simple** | **194.80** | 34.36 | ~10× mejor que el aleatorio, sin aprender |
| FrozenLake-v1 | aleatorio | 0.05 | 0.22 | 1/20 éxitos (≈2.4 % en 500 episodios) |

Figuras en `results/figures/` y métricas completas en `results/summary.json`:

- `cartpole_random_rewards.png` — recompensa por episodio del agente aleatorio en CartPole-v1.
- `frozenlake_random_rewards.png` — recompensa por episodio del agente aleatorio en FrozenLake-v1.
- `cartpole_policy_comparison.png` — política simple vs. agente aleatorio en CartPole-v1.

El análisis y la discusión completos se desarrollan en el informe escrito (PDF).
